"""Seed 知识图谱：阶段 4 监控所需的初始实体与关系。

幂等 — 使用 MERGE，可安全重复运行。
仅覆盖 Metric → AlertType → Severity 映射。

运行方式：
    python -m apps.modelops_api.scripts.seed_knowledge_graph

阶段 5 将扩展 RootCause、Validator 等诊断实体。
阶段 8 将添加文档抽取和权重校准。
"""

from __future__ import annotations

import asyncio
import sys

from neo4j import AsyncGraphDatabase

from apps.modelops_api.config import settings

# ── 实体定义 ──

METRICS = [
    {"entity_code": "FEATURE_PSI", "entity_type": "Metric", "name": "特征PSI"},
    {"entity_code": "SCORE_PSI", "entity_type": "Metric", "name": "分数PSI"},
    {"entity_code": "AUC", "entity_type": "Metric", "name": "AUC"},
    {"entity_code": "KS", "entity_type": "Metric", "name": "KS"},
    {"entity_code": "MISSING_RATE", "entity_type": "Metric", "name": "缺失率"},
    {"entity_code": "SCHEMA_CONSISTENCY", "entity_type": "Metric", "name": "模式一致性"},
    {"entity_code": "SAMPLE_SIZE", "entity_type": "Metric", "name": "样本量"},
]

ALERT_TYPES = [
    {"entity_code": "HIGH_FEATURE_PSI", "entity_type": "AlertType", "name": "特征PSI漂移"},
    {"entity_code": "HIGH_SCORE_PSI", "entity_type": "AlertType", "name": "分数PSI漂移"},
    {"entity_code": "AUC_DROP", "entity_type": "AlertType", "name": "AUC下降"},
    {"entity_code": "KS_DROP", "entity_type": "AlertType", "name": "KS下降"},
    {"entity_code": "MISSING_RATE_SPIKE", "entity_type": "AlertType", "name": "缺失率异常"},
    {"entity_code": "SCHEMA_CHANGE", "entity_type": "AlertType", "name": "模式变化"},
    {"entity_code": "SAMPLE_SIZE_LOW", "entity_type": "AlertType", "name": "样本量不足"},
]

SEVERITIES = [
    {"entity_code": "INFO", "entity_type": "Severity", "name": "信息"},
    {"entity_code": "WARNING", "entity_type": "Severity", "name": "警告"},
    {"entity_code": "HIGH", "entity_type": "Severity", "name": "高"},
    {"entity_code": "CRITICAL", "entity_type": "Severity", "name": "严重"},
]

# ── 关系定义 ──
# (Metric)-[BREACHES_THRESHOLD]->(AlertType)
BREACHES_RELATIONS = [
    ("FEATURE_PSI", "HIGH_FEATURE_PSI"),
    ("SCORE_PSI", "HIGH_SCORE_PSI"),
    ("AUC", "AUC_DROP"),
    ("KS", "KS_DROP"),
    ("MISSING_RATE", "MISSING_RATE_SPIKE"),
    ("SCHEMA_CONSISTENCY", "SCHEMA_CHANGE"),
    ("SAMPLE_SIZE", "SAMPLE_SIZE_LOW"),
]

# (AlertType)-[HAS_SEVERITY]->(Severity)
SEVERITY_RELATIONS = [
    ("HIGH_FEATURE_PSI", "HIGH"),
    ("HIGH_SCORE_PSI", "HIGH"),
    ("AUC_DROP", "WARNING"),
    ("KS_DROP", "WARNING"),
    ("MISSING_RATE_SPIKE", "WARNING"),
    ("SCHEMA_CHANGE", "HIGH"),
    ("SAMPLE_SIZE_LOW", "INFO"),
]

WEIGHT_VERSION = "seed_v1"


async def seed() -> int:
    """将监控实体和关系写入 Neo4j。返回写入的节点数。"""
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    entity_count = 0
    relation_count = 0

    async with driver.session(database="neo4j") as session:
        # ① 创建实体（Metric + AlertType + Severity）
        all_entities = METRICS + ALERT_TYPES + SEVERITIES
        for ent in all_entities:
            entity_type = ent["entity_type"]
            await session.run(
                "MERGE (n:" + entity_type + " {entity_code: $entity_code}) "
                "SET n.name = $name, "
                "    n.namespace = 'MONITORING', "
                "    n.is_core = true, "
                "    n.enabled = true",
                entity_code=ent["entity_code"],
                name=ent["name"],
            )
            entity_count += 1

        # ② 创建 BREACHES_THRESHOLD 关系
        for source, target in BREACHES_RELATIONS:
            relation_key = f"{source}|BREACHES_THRESHOLD|{target}"
            await session.run(
                """
                MATCH (s {entity_code: $source})
                MATCH (t {entity_code: $target})
                MERGE (s)-[r:BREACHES_THRESHOLD]->(t)
                SET r.relation_key = $key,
                    r.initial_prior_weight = 1.0,
                    r.prior_strength = 1.0,
                    r.effective_weight = 1.0,
                    r.confidence_lower_bound = 0.0,
                    r.confidence_upper_bound = 0.0,
                    r.evidence_case_count = 0,
                    r.natural_case_count = 0,
                    r.scenario_case_count = 0,
                    r.support_count = 0,
                    r.against_count = 0,
                    r.neutral_count = 0,
                    r.support_strength = 0.0,
                    r.against_strength = 0.0,
                    r.weight_version = $weight_version,
                    r.enabled = true
                """,
                source=source,
                target=target,
                key=relation_key,
                weight_version=WEIGHT_VERSION,
            )
            relation_count += 1

        # ③ 创建 HAS_SEVERITY 关系
        for source, target in SEVERITY_RELATIONS:
            relation_key = f"{source}|HAS_SEVERITY|{target}"
            await session.run(
                """
                MATCH (s {entity_code: $source})
                MATCH (t {entity_code: $target})
                MERGE (s)-[r:HAS_SEVERITY]->(t)
                SET r.relation_key = $key,
                    r.initial_prior_weight = 1.0,
                    r.prior_strength = 1.0,
                    r.effective_weight = 1.0,
                    r.confidence_lower_bound = 0.0,
                    r.confidence_upper_bound = 0.0,
                    r.evidence_case_count = 0,
                    r.natural_case_count = 0,
                    r.scenario_case_count = 0,
                    r.support_count = 0,
                    r.against_count = 0,
                    r.neutral_count = 0,
                    r.support_strength = 0.0,
                    r.against_strength = 0.0,
                    r.weight_version = $weight_version,
                    r.enabled = true
                """,
                source=source,
                target=target,
                key=relation_key,
                weight_version=WEIGHT_VERSION,
            )
            relation_count += 1

    await driver.close()

    print(
        f"KG 种子完成：{entity_count} 个实体 "
        f"（{len(METRICS)} Metric + {len(ALERT_TYPES)} AlertType + {len(SEVERITIES)} Severity），"
        f"{relation_count} 条关系 "
        f"（{len(BREACHES_RELATIONS)} BREACHES_THRESHOLD + {len(SEVERITY_RELATIONS)} HAS_SEVERITY）"
    )
    return entity_count


if __name__ == "__main__":
    asyncio.run(seed())
    sys.exit(0)
