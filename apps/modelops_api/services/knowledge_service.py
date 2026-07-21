"""KnowledgeService — 阶段 4–5 只读 Neo4j 知识图谱访问层。

职责：
- resolve_alert: 给定指标代码和严重度，返回对应的告警类型
- get_entity: 按 entity_code 查询实体
- query_relations: 按源实体和关系类型查询出边

Neo4j 不可用时回退到内置默认映射，保证监控链路不中断。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog
from neo4j import AsyncDriver as Neo4jAsyncDriver

from packages.models.common.enums import QueryProfileCode, Severity
from packages.models.knowledge.kg_entity import KgEntity, KgRelation
from packages.models.knowledge.query_profile import QueryProfile

logger = structlog.get_logger(__name__)

# ── Query Profile：生产监控用最小权重/置信度 ──

_MONITORING_PROFILE = QueryProfile(
    profile_code=QueryProfileCode.PRODUCTION_MONITORING,
    min_effective_weight=0.3,
    min_evidence_case_count=0,
    min_confidence_lower_bound=0.0,
)

# ── 内置默认 Metric→Alert 映射（Neo4j 不可用时的降级后备）──

_DEFAULT_METRIC_ALERT_MAP: dict[str, dict] = {
    "FEATURE_PSI": {
        "alert_code": "HIGH_FEATURE_PSI",
        "severity": Severity.HIGH,
        "description": "特征分布漂移超过阈值",
    },
    "SCORE_PSI": {
        "alert_code": "HIGH_SCORE_PSI",
        "severity": Severity.HIGH,
        "description": "分数分布漂移超过阈值",
    },
    "AUC": {
        "alert_code": "AUC_DROP",
        "severity": Severity.WARNING,
        "description": "AUC 低于基准",
    },
    "KS": {
        "alert_code": "KS_DROP",
        "severity": Severity.WARNING,
        "description": "KS 统计量低于基准",
    },
    "MISSING_RATE": {
        "alert_code": "MISSING_RATE_SPIKE",
        "severity": Severity.WARNING,
        "description": "缺失率异常上升",
    },
    "SCHEMA_CONSISTENCY": {
        "alert_code": "SCHEMA_CHANGE",
        "severity": Severity.HIGH,
        "description": "输入模式与训练时不一致",
    },
    "SAMPLE_SIZE": {
        "alert_code": "SAMPLE_SIZE_LOW",
        "severity": Severity.INFO,
        "description": "监控样本量不足以可靠评估",
    },
}


@dataclass
class AlertTypeResult:
    """resolve_alert() 返回的告警类型结果。"""

    alert_code: str
    metric_code: str
    severity: Severity
    effective_weight: float = 1.0
    description: str = ""
    from_neo4j: bool = True


class KnowledgeService:
    """只读知识图谱访问服务。

    构造函数接受 Neo4j 异步驱动，所有方法为 async。
    Neo4j 不可用时自动降级到内置默认映射。
    """

    def __init__(self, driver: Neo4jAsyncDriver):
        self.driver = driver

    # ── Metric → Alert 映射 ──

    async def resolve_alert(
        self, metric_code: str, severity: Severity | None = None
    ) -> AlertTypeResult | None:
        """给定违反阈值的指标代码，返回对应的告警类型。

        优先从 Neo4j 查询，失败时回退到内置默认映射。
        如果 metric_code 在 Neo4j 和默认映射中均不存在，返回 None。

        Cypher:
            MATCH (m:Metric {entity_code: $metric_code})
                  -[r:BREACHES_THRESHOLD]->(a:AlertType)
            OPTIONAL MATCH (a)-[:HAS_SEVERITY]->(s:Severity)
            WHERE r.effective_weight >= $min_weight
            RETURN a.entity_code, a.name, s.entity_code, r.effective_weight
        """
        try:
            async with self.driver.session(
                database="neo4j", default_access_mode="READ"
            ) as session:
                result = await session.run(
                    """
                    MATCH (m:Metric {entity_code: $metric_code})
                          -[r:BREACHES_THRESHOLD]->(a:AlertType)
                    WHERE r.effective_weight >= $min_weight AND r.enabled = true
                    OPTIONAL MATCH (a)-[:HAS_SEVERITY]->(s:Severity)
                    RETURN a.entity_code AS alert_code,
                           a.name AS alert_name,
                           s.entity_code AS severity_code,
                           r.effective_weight AS weight
                    LIMIT 1
                    """,
                    metric_code=metric_code,
                    min_weight=_MONITORING_PROFILE.min_effective_weight,
                )
                record = await result.single()
                if record:
                    sev = (
                        Severity(record["severity_code"])
                        if record["severity_code"]
                        else (severity or Severity.WARNING)
                    )
                    return AlertTypeResult(
                        alert_code=record["alert_code"],
                        metric_code=metric_code,
                        severity=sev,
                        effective_weight=record["weight"],
                        description=record.get("alert_name", ""),
                        from_neo4j=True,
                    )
        except Exception:
            logger.warning(
                "neo4j_resolve_alert_failed_falling_back",
                metric_code=metric_code,
                exc_info=True,
            )

        # 降级：使用内置默认映射
        default = _DEFAULT_METRIC_ALERT_MAP.get(metric_code)
        if default:
            return AlertTypeResult(
                alert_code=default["alert_code"],
                metric_code=metric_code,
                severity=default["severity"],
                effective_weight=1.0,
                description=default["description"],
                from_neo4j=False,
            )
        return None

    # ── 实体查询 ──

    async def get_entity(self, entity_code: str) -> KgEntity | None:
        """按 entity_code 查询单个知识实体。

        Cypher:
            MATCH (n {entity_code: $entity_code})
            WHERE n.enabled = true
            RETURN n
        """
        try:
            async with self.driver.session(
                database="neo4j", default_access_mode="READ"
            ) as session:
                result = await session.run(
                    """
                    MATCH (n {entity_code: $entity_code})
                    WHERE n.enabled = true
                    RETURN n.entity_code AS entity_code,
                           n.entity_type AS entity_type,
                           n.name AS name,
                           n.namespace AS namespace,
                           n.is_core AS is_core,
                           n.enabled AS enabled,
                           n.schema_version AS schema_version,
                           n.attributes_json AS attributes_json
                    LIMIT 1
                    """,
                    entity_code=entity_code,
                )
                record = await result.single()
                if record:
                    return KgEntity(
                        entity_code=record["entity_code"],
                        entity_type=record["entity_type"],
                        name=record["name"],
                        namespace=record.get("namespace", "CORE"),
                        is_core=record.get("is_core", False),
                        enabled=record.get("enabled", True),
                        schema_version=record.get("schema_version"),
                        attributes_json=record.get("attributes_json"),
                    )
        except Exception:
            logger.warning(
                "neo4j_get_entity_failed",
                entity_code=entity_code,
                exc_info=True,
            )
        return None

    # ── 关系查询 ──

    async def query_relations(
        self, source_entity_code: str, relation_type: str | None = None
    ) -> list[KgRelation]:
        """查询从指定实体出发的关系。

        Cypher:
            MATCH (s {entity_code: $source_code})-[r]->(t)
            WHERE r.enabled = true
              AND ($rel_type IS NULL OR r.relation_type = $rel_type)
              AND r.effective_weight >= $min_weight
            RETURN r, t.entity_code
        """
        relations: list[KgRelation] = []
        try:
            async with self.driver.session(
                database="neo4j", default_access_mode="READ"
            ) as session:
                result = await session.run(
                    """
                    MATCH (s {entity_code: $source_code})-[r]->(t)
                    WHERE r.enabled = true
                      AND ($rel_type IS NULL OR r.relation_type = $rel_type)
                      AND r.effective_weight >= $min_weight
                    RETURN r.relation_key AS relation_key,
                           r.source_entity_code AS source_entity_code,
                           r.relation_type AS relation_type,
                           r.target_entity_code AS target_entity_code,
                           r.initial_prior_weight AS initial_prior_weight,
                           r.effective_weight AS effective_weight,
                           r.confidence_lower_bound AS confidence_lower_bound,
                           r.confidence_upper_bound AS confidence_upper_bound,
                           r.evidence_case_count AS evidence_case_count,
                           r.weight_version AS weight_version,
                           r.enabled AS enabled
                    """,
                    source_code=source_entity_code,
                    rel_type=relation_type,
                    min_weight=_MONITORING_PROFILE.min_effective_weight,
                )
                async for record in result:
                    relations.append(
                        KgRelation(
                            relation_key=record["relation_key"],
                            source_entity_code=record["source_entity_code"],
                            relation_type=record["relation_type"],
                            target_entity_code=record["target_entity_code"],
                            initial_prior_weight=record["initial_prior_weight"],
                            effective_weight=record["effective_weight"],
                            confidence_lower_bound=record.get("confidence_lower_bound", 0.0),
                            confidence_upper_bound=record.get("confidence_upper_bound", 0.0),
                            evidence_case_count=record.get("evidence_case_count", 0),
                            weight_version=record["weight_version"],
                            enabled=record.get("enabled", True),
                        )
                    )
        except Exception:
            logger.warning(
                "neo4j_query_relations_failed",
                source_code=source_entity_code,
                relation_type=relation_type,
                exc_info=True,
            )
        return relations
