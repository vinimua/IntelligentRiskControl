"""KnowledgeService — 阶段 4–5 只读 Neo4j 知识图谱访问层。

职责：
- resolve_alert: 给定指标代码和严重度，返回对应的告警类型
- get_entity: 按 entity_code 查询实体
- query_relations: 按源实体和关系类型查询出边

Neo4j 不可用时回退到内置默认映射，保证监控链路不中断。
"""

from __future__ import annotations

from dataclasses import dataclass
import json

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

_SUPPORTED_TRAINING_ALGORITHMS = {
    "lightgbm",
    "logistic_regression",
    "random_forest",
}

# ── 资格门禁告警本地定义（Neo4j 不可用时 fail-safe，不允许 fail-open）──
_DEFAULT_GATE_ALERTS: dict[str, str] = {
    "SAMPLE_SIZE_LOW": "DATA_ELIGIBILITY_BLOCK",
}

# A7 实施定稿 §2.1：诊断层历史码 → A7 策略层正式码
_A7_ROOT_CAUSE_MAPPING: dict[str, str] = {
    "feature_drift": "FEATURE_DRIFT",
    "population_shift": "SEGMENT_DRIFT",
}


def _coerce_json_dict(value) -> dict:
    """Neo4j 不支持 Map 属性，hyperparameters/sample_weight_policy 以 JSON
    字符串存储；读取时转换回 dict。"""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
    return value or {}

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
    "PR_AUC": {
        "alert_code": "PR_AUC_DROP",
        "severity": Severity.WARNING,
        "description": "PR-AUC 低于基准",
    },
    "BAD_RECALL": {
        "alert_code": "BAD_RECALL_DROP",
        "severity": Severity.WARNING,
        "description": "坏样本召回率低于基准",
    },
    "BRIER": {
        "alert_code": "CALIBRATION_DEGRADE",
        "severity": Severity.WARNING,
        "description": "Brier 校准误差增大",
    },
    "ECE": {
        "alert_code": "CALIBRATION_DEGRADE",
        "severity": Severity.WARNING,
        "description": "期望校准误差增大",
    },
    "BAD_RATE": {
        "alert_code": "BAD_RATE_SHIFT",
        "severity": Severity.WARNING,
        "description": "坏样本率偏离基准",
    },
    "OUTLIER_RATE": {
        "alert_code": "OUTLIER_RATE_SPIKE",
        "severity": Severity.WARNING,
        "description": "异常值率异常上升",
    },
    "PREDICTION_MEAN": {
        "alert_code": "PREDICTION_MEAN_SHIFT",
        "severity": Severity.WARNING,
        "description": "预测均值发生偏移",
    },
}


@dataclass
class AlertResult:
    """resolve_alert() 返回的告警结果。"""

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
    ) -> AlertResult | None:
        """给定违反阈值的指标代码，返回对应的告警类型。

        优先从 Neo4j 查询，失败时回退到内置默认映射。
        如果 metric_code 在 Neo4j 和默认映射中均不存在，返回 None。

        Cypher:
            MATCH (m:Metric {entity_code: $metric_code})
                  -[r:TRIGGERS]->(a:Alert)
            WHERE r.effective_weight >= $min_weight
            RETURN a.entity_code, a.name, r.effective_weight
        """
        try:
            async with self.driver.session(
                database="neo4j", default_access_mode="READ"
            ) as session:
                result = await session.run(
                    """
                    MATCH (m:Metric {entity_code: $metric_code})
                          -[r:TRIGGERS]->(a:Alert)
                    WHERE r.effective_weight >= $min_weight AND r.enabled = true
                    RETURN a.entity_code AS alert_code,
                           a.name AS alert_name,
                           r.effective_weight AS weight
                    LIMIT 1
                    """,
                    metric_code=metric_code,
                    min_weight=_MONITORING_PROFILE.min_effective_weight,
                )
                record = await result.single()
                if record:
                    return AlertResult(
                        alert_code=record["alert_code"],
                        metric_code=metric_code,
                        severity=severity or Severity.WARNING,
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
            return AlertResult(
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

    async def query_candidate_root_causes(
        self, alert_code: str
    ) -> list["CandidateRootCause"]:
        """查询告警对应的候选根因（Alert ─INDICATES→ RootCause ─BELONGS_TO→ Dimension）。

        任务二诊断入口：给定告警代码，返回所有可能的根因及权重快照。
        """
        import uuid as _uuid

        from packages.models.diagnosis.diagnosis_context import CandidateRootCause

        candidates: list[CandidateRootCause] = []
        try:
            async with self.driver.session(
                database="neo4j", default_access_mode="READ"
            ) as session:
                result = await session.run(
                    """
                    MATCH (a:Alert {entity_code: $alert_code})
                          -[r:INDICATES]->(rc:RootCause)
                    WHERE r.enabled = true
                      AND coalesce(a.root_cause_recall_enabled, true) = true
                      AND coalesce(a.summary_only, false) = false
                    OPTIONAL MATCH (rc)-[:BELONGS_TO]->(d:Dimension)
                    RETURN a.entity_code AS alert_code,
                           r.relation_key AS relation_key,
                           rc.entity_code AS root_cause_code,
                           d.entity_code AS dimension_code,
                           r.effective_weight AS effective_weight,
                           r.evidence_case_count AS evidence_case_count,
                           r.confidence_lower_bound AS confidence_lower_bound,
                           r.weight_version AS weight_version,
                           coalesce(r.supporting_only, false) AS supporting_only,
                           coalesce(r.required_context, []) AS required_context,
                           coalesce(r.causal_distance, '') AS causal_distance
                    ORDER BY r.effective_weight DESC
                    """,
                    alert_code=alert_code,
                )
                async for record in result:
                    candidates.append(
                        CandidateRootCause(
                            diagnosis_candidate_id=str(_uuid.uuid4()),
                            alert_code=record["alert_code"],
                            relation_key=record["relation_key"],
                            root_cause_code=record["root_cause_code"],
                            dimension_code=record["dimension_code"] or "UNKNOWN",
                            effective_weight_snapshot=record["effective_weight"] or 0.0,
                            evidence_case_count_snapshot=record["evidence_case_count"] or 0,
                            confidence_lower_bound_snapshot=record["confidence_lower_bound"] or 0.0,
                            supporting_only=record["supporting_only"],
                            required_context=list(record["required_context"] or []),
                            causal_distance=record["causal_distance"] or "",
                        )
                    )
        except Exception:
            logger.warning(
                "neo4j_query_candidate_root_causes_failed",
                alert_code=alert_code,
                exc_info=True,
            )
        return candidates

    async def query_gate_blocking_alerts(self, alert_codes: list[str]) -> list[dict]:
        """查询告警中标记 gate_only 的资格门禁告警。

        gate_only=true 的告警（如 SAMPLE_SIZE_LOW）不参与根因召回，
        而是阻断性能类根因诊断。返回 [{alert_code, gate_semantics}]。

        Neo4j 不可用时按本地门禁定义 fail-safe（资格门禁属于安全规则，
        不能因 KG 不可用而 fail-open）。
        """
        if not alert_codes:
            return []
        try:
            async with self.driver.session(
                database="neo4j", default_access_mode="READ"
            ) as session:
                result = await session.run(
                    """
                    MATCH (a:Alert)
                    WHERE a.entity_code IN $codes AND a.gate_only = true
                    RETURN a.entity_code AS alert_code,
                           coalesce(a.gate_semantics, 'DATA_ELIGIBILITY_BLOCK') AS gate_semantics
                    """,
                    codes=alert_codes,
                )
                return [
                    {"alert_code": record["alert_code"],
                     "gate_semantics": record["gate_semantics"]}
                    async for record in result
                ]
        except Exception:
            logger.warning(
                "neo4j_query_gate_alerts_failed — 使用本地门禁定义 fail-safe",
                exc_info=True,
            )
            return [
                {"alert_code": code, "gate_semantics": semantics}
                for code in alert_codes
                if (semantics := _DEFAULT_GATE_ALERTS.get(code))
            ]

    async def query_iteration_context(
        self, root_cause_code: str, diagnosis_run_id: str = "",
        severity: float | None = None,
        algorithm: str | None = None,
        decay_level: int | None = None,
        business_round: int | None = None,
        available_context_codes: list[str] | None = None,
    ):
        """P3 KG: 查询 RootCause → Strategy 候选 + 反向校验 Strategy → RootCause。

        RootCause─RECOMMENDS→Strategy（正向推荐）
        Strategy─MITIGATES→RootCause（反向缓解, 用于校验策略是否真的能解决这个根因）

        边上结构化过滤条件（对应 A7 applicability_condition）:
        - min_severity: 低于此严重度不推荐该策略
        - applicable_algorithms: 不适用于此算法的策略被过滤
        - min_decay_level: 持续性等级 NONE=0 / SHORT_TERM_7D=1 / SUSTAINED_30D=2 /
          SEVERE=3，低于该等级的衰退不推荐（任务一：SHORT_TERM_7D 仅继续观察）
        - min_business_round: 第二轮策略（如 feature_selection_retrain）在第一轮
          不会被召回
        - strategy_tier: "full" / "light" / "minimal" — 路由读取
        """
        import uuid as _uuid
        from packages.models.iteration.iteration_context import (
            IterationContext, StrategyCandidate,
        )

        candidates: list[StrategyCandidate] = []
        context_pack_id = str(_uuid.uuid4())
        retrieval_degraded = False
        context_weight_version = ""

        # A7 §2.1: 诊断层历史码映射到策略层正式码（feature_drift→FEATURE_DRIFT）
        kg_root_cause_code = _A7_ROOT_CAUSE_MAPPING.get(
            str(root_cause_code), root_cause_code
        )

        try:
            async with self.driver.session(
                database="neo4j", default_access_mode="READ"
            ) as session:
                # 正向：RootCause → Strategy（按 severity + algorithm 过滤）
                result = await session.run(
                    """
                    MATCH (rc:RootCause {entity_code: $root_cause_code})
                          -[rec:RECOMMENDS]->(s:Strategy)
                    WHERE rec.enabled = true
                      AND (s.strategy_scope IS NULL OR s.strategy_scope = 'A7_TRAINING')
                      AND ($severity IS NULL
                           OR rec.min_severity IS NULL
                           OR $severity >= rec.min_severity)
                      AND ($severity IS NULL
                           OR rec.max_severity IS NULL
                           OR $severity <= rec.max_severity)
                      AND ($algorithm IS NULL
                           OR rec.applicable_algorithms IS NULL
                           OR $algorithm IN rec.applicable_algorithms)
                      AND ($decay_level IS NULL
                           OR coalesce(rec.min_decay_level, 0) <= $decay_level)
                      AND ($business_round IS NULL
                           OR coalesce(rec.min_business_round, 1) <= $business_round)
                      AND ($available_contexts IS NULL
                           OR rec.required_context IS NULL
                           OR ALL(x IN rec.required_context
                                  WHERE x IN $available_contexts))
                    OPTIONAL MATCH (s)-[mit:MITIGATES]->(rc)
                    WHERE mit.enabled = true
                    RETURN rc.entity_code AS root_cause_code,
                           rec.relation_key AS recommends_relation_key,
                           rec.effective_weight AS effective_weight,
                           rec.evidence_case_count AS evidence_case_count,
                           s.entity_code AS strategy_code,
                           s.training_cost_level AS training_cost_level,
                           s.risk_level AS risk_level,
                           s.executor_code AS executor_code,
                           s.algorithm AS algorithm,
                           s.feature_schema_version AS feature_schema_version,
                           s.preprocessing_version AS preprocessing_version,
                           s.label_versions AS label_versions,
                           s.allowed_training_window_ids AS allowed_training_window_ids,
                           s.validation_window_ids AS validation_window_ids,
                           s.hyperparameters AS hyperparameters,
                           s.sample_weight_policy AS sample_weight_policy,
                           coalesce(s.primary_training_mode, 'FULL_RETRAIN') AS primary_training_mode,
                           coalesce(rec.weight_version, '') AS relation_weight_version,
                           mit.relation_key AS mitigates_relation_key,
                           mit.evidence_case_count AS mitigates_case_count,
                           mit.confidence_lower_bound AS mitigates_confidence,
                           mit.effective_weight AS mitigates_weight,
                           coalesce(
                               rec.historical_effectiveness,
                               s.historical_effectiveness
                           ) AS historical_effectiveness,
                           coalesce(
                               rec.historical_effectiveness,
                               s.historical_effectiveness,
                               rec.effective_weight
                           ) AS strategy_rank_score,
                           CASE
                               WHEN rec.historical_effectiveness IS NOT NULL
                                 OR s.historical_effectiveness IS NOT NULL
                               THEN 'CALIBRATED_HISTORY'
                               ELSE 'INITIAL_PRIOR'
                           END AS rank_score_source,
                           coalesce(rec.support_case_count, rec.evidence_case_count, 0) AS support_case_count,
                           coalesce(rec.total_case_count, rec.evidence_case_count, 0) AS total_case_count,
                           coalesce(rec.natural_case_count, rec.evidence_case_count, 0) AS natural_case_count,
                           coalesce(rec.strategy_tier, 'full') AS strategy_tier,
                           coalesce(rec.required_context, []) AS required_context
                    ORDER BY rec.effective_weight DESC,
                             coalesce(rec.min_severity, 0.0) ASC
                    """,
                    root_cause_code=kg_root_cause_code,
                    severity=severity,
                    algorithm=algorithm,
                    decay_level=decay_level,
                    business_round=business_round,
                    available_contexts=available_context_codes,
                )
                relation_weight_versions: list[str] = []
                async for record in result:
                    sc = record["strategy_code"]
                    case_count = record["support_case_count"] or 0
                    if record["relation_weight_version"]:
                        relation_weight_versions.append(record["relation_weight_version"])

                    # 历史有效率与先验分离：
                    # - historical_effectiveness：真实历史有效率，无历史案例时为 None
                    # - strategy_rank_score：排序分。有真实历史 = 历史有效率；
                    #   无历史 = 初始先验权重（专家先验，不伪装成历史有效率）
                    eff_weight = record["effective_weight"] or 0.5
                    mitigates_weight = record["mitigates_weight"] or 0.0
                    historical_effectiveness = record["historical_effectiveness"]
                    strategy_rank_score = record["strategy_rank_score"]
                    if strategy_rank_score is None:
                        strategy_rank_score = eff_weight * 0.7 + mitigates_weight * 0.3
                    rank_score_source = record["rank_score_source"] or "INITIAL_PRIOR"
                    algorithm = record["algorithm"]
                    if algorithm:
                        algorithm = str(algorithm).lower()
                    if algorithm not in _SUPPORTED_TRAINING_ALGORITHMS:
                        algorithm = None

                    candidates.append(
                        StrategyCandidate(
                            strategy_code=sc,
                            recommends_relation_key=record["recommends_relation_key"],
                            mitigates_relation_key=record["mitigates_relation_key"] or "",
                            relation_effective_weight_snapshot=eff_weight,
                            historical_effectiveness=historical_effectiveness,
                            strategy_rank_score=strategy_rank_score,
                            rank_score_source=rank_score_source,
                            support_case_count=case_count,
                            total_case_count=record["total_case_count"] or case_count,
                            natural_case_count=record["natural_case_count"] or 0,
                            confidence_lower_bound=record["mitigates_confidence"] or eff_weight * 0.5,
                            required_data_codes=[],
                            allowed_training_window_ids=record["allowed_training_window_ids"] or [],
                            validation_window_ids=record["validation_window_ids"] or [],
                            algorithm=algorithm,
                            feature_schema_version=record["feature_schema_version"],
                            preprocessing_version=record["preprocessing_version"],
                            label_versions=record["label_versions"] or [],
                            # Neo4j 不支持 Map 属性，图谱中以 JSON 字符串存储
                            hyperparameters=_coerce_json_dict(record["hyperparameters"]),
                            sample_weight_policy=_coerce_json_dict(record["sample_weight_policy"]),
                            training_cost_level=record["training_cost_level"] or "MEDIUM",
                            risk_level=record["risk_level"] or "LOW",
                            executor_code=record["executor_code"] or "MODEL_RETRAIN",
                            strategy_tier=record["strategy_tier"] or "full",
                            required_context=list(record["required_context"] or []),
                            primary_training_mode=(
                                record["primary_training_mode"] or "FULL_RETRAIN"
                            ),
                        )
                    )

                # weight_version 来自实际候选关系的权重版本，不做固定值
                context_weight_version = (
                    relation_weight_versions[0] if relation_weight_versions else ""
                )

                logger.info(
                    "query_iteration_context_done",
                    root_cause_code=root_cause_code,
                    candidate_count=len(candidates),
                )

        except Exception:
            logger.warning(
                "neo4j_query_iteration_context_failed",
                root_cause_code=root_cause_code,
                exc_info=True,
            )
            retrieval_degraded = True
            # Neo4j 不可用时降级为空，后续 YAML 规则兜底

        return IterationContext(
            context_pack_id=context_pack_id,
            diagnosis_run_id=diagnosis_run_id,
            root_cause_code=root_cause_code,
            weight_version=context_weight_version,
            strategy_candidates=candidates,
            rules=None,
            retrieved_references=[],
            retrieval_degraded=retrieval_degraded,
        )

    # ── P0: 部署 KG 查询 ─────────────────────────────────────────────

    async def query_deployment_context(
        self,
        alert_codes: list[str],
        stage: str,
        model_id: str = "",
        min_weight: float = 0.3,
        alert_payloads: list[dict] | None = None,
    ):
        """查询 DeploymentAlert → DeploymentRisk → DeploymentStrategy。

        对每个告警查询 KG：
        1. DeploymentAlert -[INDICATES]-> DeploymentRisk
        2. DeploymentRisk -[RECOMMENDS]-> DeploymentStrategy
        3. DeploymentStrategy -[MITIGATES]-> DeploymentRisk（反向校验）
        """
        import uuid as _uuid
        from packages.models.deployment.deployment_context import (
            DeploymentContext, DeploymentRisk, DeploymentStrategyCandidate,
        )

        context_pack_id = str(_uuid.uuid4())
        retrieval_degraded = False
        risks: list[DeploymentRisk] = []
        alert_payloads = alert_payloads or []
        alert_by_code = {
            str(payload.get("alert_code")): payload
            for payload in alert_payloads
            if payload.get("alert_code")
        }

        if not alert_codes:
            return DeploymentContext(
                context_pack_id=context_pack_id,
                model_id=model_id,
                stage=stage,
                retrieval_degraded=False,
            )

        try:
            async with self.driver.session(
                database="neo4j", default_access_mode="READ"
            ) as session:
                result = await session.run(
                    """
                    UNWIND $alert_codes AS alert_code
                    MATCH (a:DeploymentAlert {entity_code: alert_code})
                          -[ind:INDICATES]->(risk:DeploymentRisk)
                    WHERE ind.enabled = true
                      AND ind.effective_weight >= $min_weight
                    OPTIONAL MATCH (risk)-[rec:RECOMMENDS]->(strategy:DeploymentStrategy)
                    WHERE rec.enabled = true
                      AND (
                        strategy.allowed_stages IS NULL
                        OR size(strategy.allowed_stages) = 0
                        OR $stage IN strategy.allowed_stages
                      )
                    OPTIONAL MATCH (stage_node:DeploymentStage {entity_code: $stage})
                          -[allows:ALLOWS]->(strategy)
                    WHERE allows IS NULL OR allows.enabled = true
                    OPTIONAL MATCH (policy:DeploymentPolicy)-[con:CONSTRAINS]->(strategy)
                    WHERE con IS NULL OR con.enabled = true
                    OPTIONAL MATCH (strategy)-[mit:MITIGATES]->(risk)
                    WHERE mit.enabled = true
                    RETURN
                      alert_code,
                      risk.entity_code AS risk_code,
                      risk.name AS risk_name,
                      ind.relation_key AS risk_relation_key,
                      ind.effective_weight AS risk_weight,
                      ind.confidence_lower_bound AS risk_confidence,
                      strategy.entity_code AS strategy_code,
                      strategy.action_type AS action_type,
                      strategy.parameters AS strategy_parameters,
                      rec.relation_key AS strategy_relation_key,
                      rec.effective_weight AS strategy_weight,
                      rec.confidence_lower_bound AS strategy_confidence,
                      rec.evidence_case_count AS support_case_count,
                      rec.natural_case_count AS natural_case_count,
                      mit.relation_key AS mitigates_relation_key,
                      mit.effective_weight AS mitigates_weight,
                      strategy.allowed_stages AS allowed_stages,
                      collect(DISTINCT policy.entity_code) AS policy_refs
                    ORDER BY ind.effective_weight DESC, rec.effective_weight DESC
                    """,
                    alert_codes=alert_codes,
                    min_weight=min_weight,
                    stage=stage,
                )

                # Group by risk_code
                risk_map: dict[str, DeploymentRisk] = {}
                async for record in result:
                    rk = record["risk_code"] or "unknown_risk"
                    if rk not in risk_map:
                        alert_code = str(record["alert_code"])
                        payload = alert_by_code.get(alert_code, {})
                        risk_map[rk] = DeploymentRisk(
                            risk_code=rk,
                            risk_name=record["risk_name"],
                            relation_key=record["risk_relation_key"] or "",
                            effective_weight_snapshot=record["risk_weight"] or 0.0,
                            confidence_lower_bound_snapshot=record["risk_confidence"] or 0.0,
                            severity="HIGH" if (record["risk_weight"] or 0) > 0.7 else "MEDIUM",
                            alert_codes=[alert_code],
                            evidence_detail={
                                "alerts": [payload] if payload else [],
                                "stage": stage,
                            },
                        )
                    else:
                        alert_code = str(record["alert_code"])
                        if alert_code not in risk_map[rk].alert_codes:
                            risk_map[rk].alert_codes.append(alert_code)
                        payload = alert_by_code.get(alert_code)
                        if payload:
                            risk_map[rk].evidence_detail.setdefault("alerts", []).append(payload)

                    sc = record["strategy_code"]
                    if sc:
                        strategy_parameters = record["strategy_parameters"] or {}
                        if isinstance(strategy_parameters, str):
                            try:
                                strategy_parameters = json.loads(strategy_parameters)
                            except json.JSONDecodeError:
                                strategy_parameters = {"raw": strategy_parameters}
                        risk_map[rk].strategy_candidates.append(
                            DeploymentStrategyCandidate(
                                strategy_code=str(sc),
                                relation_key=record["strategy_relation_key"] or "",
                                effective_weight_snapshot=record["strategy_weight"] or 0.0,
                                confidence_lower_bound_snapshot=record["strategy_confidence"] or 0.0,
                                support_case_count=record["support_case_count"] or 0,
                                natural_case_count=record["natural_case_count"] or 0,
                                action_type=record["action_type"] or "",
                                parameters=strategy_parameters,
                                allowed_stages=record["allowed_stages"] or [],
                                policy_refs=record["policy_refs"] or [],
                                mitigates_relation_key=record["mitigates_relation_key"],
                            )
                        )

                risks = list(risk_map.values())

                logger.info(
                    "deployment_kg_query_done",
                    alert_count=len(alert_codes),
                    risk_count=len(risks),
                    strategy_count=sum(len(r.strategy_candidates) for r in risks),
                )

        except Exception:
            logger.warning(
                "neo4j_deployment_context_failed",
                alert_codes=alert_codes,
                exc_info=True,
            )
            retrieval_degraded = True

        # Collect gatekeeper rule refs from strategies
        rule_refs: list[str] = []
        for risk in risks:
            for sc in risk.strategy_candidates:
                ref = f"KG:{sc.strategy_code}"
                if ref not in rule_refs:
                    rule_refs.append(ref)

        return DeploymentContext(
            context_pack_id=context_pack_id,
            model_id=model_id,
            stage=stage,
            deployment_alerts=alert_payloads,
            deployment_risks=risks,
            gatekeeper_rule_refs=rule_refs,
            retrieval_degraded=retrieval_degraded,
            degradation_reason="Neo4j unavailable" if retrieval_degraded else None,
        )
