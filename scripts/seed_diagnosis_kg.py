"""Seed minimal diagnosis KG nodes and Alert -> RootCause edges.

This script follows the formal KG path:
Alert -[:INDICATES]-> RootCause.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from neo4j import AsyncGraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from apps.modelops_api.config import settings
from apps.modelops_api.services.knowledge_observation_mapper import (
    SCENARIO_TO_ROOT_CAUSE,
)

ALERTS = {
    "AUC_DROP": "AUC显著下降",
    "KS_DROP": "KS显著下降",
    "PR_AUC_DROP": "PR-AUC显著下降",
    "BAD_RECALL_DROP": "坏样本召回下降",
    "CALIBRATION_DEGRADE": "概率校准恶化",
    "HIGH_FEATURE_PSI": "特征PSI漂移",
    "HIGH_SCORE_PSI": "分数PSI漂移",
    "MISSING_RATE_SPIKE": "缺失率异常上升",
    "SCHEMA_MISMATCH": "Schema不一致",
    "SAMPLE_SIZE_LOW": "样本量不足",
    "BAD_RATE_SHIFT": "坏样本率变化",
    "PERFORMANCE_DECAY": "性能持续衰退",
}

ROOT_CAUSES = {
    "business_policy_change": ("业务政策变化", "BUSINESS"),
    "data_pipeline_issue": ("数据管道或预处理问题", "DATA"),
    "data_quality_issue": ("数据质量问题", "DATA"),
    "feature_drift": ("特征分布漂移", "FEATURE"),
    "feature_failure": ("特征失效", "FEATURE"),
    "label_distribution_shift": ("标签分布或违约模式变化", "DATA"),
    "population_shift": ("客群结构迁移", "BUSINESS"),
}

DIMENSIONS = {
    "BUSINESS": "业务维度",
    "DATA": "数据维度",
    "FEATURE": "特征维度",
    "MODEL": "模型维度",
}


def _alert_for_scenario(scenario_name: str) -> list[str]:
    if scenario_name == "missing_rate_anomaly":
        return ["MISSING_RATE_SPIKE", "AUC_DROP", "KS_DROP"]
    if scenario_name in {"covariate_drift", "numeric_scaling_anomaly", "preprocessing_version_mismatch"}:
        return ["HIGH_FEATURE_PSI", "AUC_DROP", "KS_DROP"]
    if scenario_name in {"concept_drift", "bad_rate_shift", "fraud_pattern_shift"}:
        return ["AUC_DROP", "KS_DROP", "BAD_RATE_SHIFT"]
    if scenario_name in {"customer_mix_shift", "policy_selection_shift"}:
        return ["HIGH_FEATURE_PSI", "BAD_RATE_SHIFT", "AUC_DROP", "KS_DROP"]
    if scenario_name in {"feature_staleness", "key_feature_failure", "multi_root_cause"}:
        return ["HIGH_FEATURE_PSI", "MISSING_RATE_SPIKE", "AUC_DROP", "KS_DROP"]
    return ["AUC_DROP", "KS_DROP"]


async def seed() -> None:
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    async with driver.session(database="neo4j") as session:
        for code, name in ALERTS.items():
            await session.run(
                """
                MERGE (a:Alert {entity_code: $code})
                SET a.name = $name,
                    a.entity_type = 'Alert',
                    a.namespace = 'DIAGNOSIS',
                    a.enabled = true
                """,
                code=code,
                name=name,
            )

        for code, name in DIMENSIONS.items():
            await session.run(
                """
                MERGE (d:Dimension {entity_code: $code})
                SET d.name = $name,
                    d.entity_type = 'Dimension',
                    d.namespace = 'DIAGNOSIS',
                    d.enabled = true
                """,
                code=code,
                name=name,
            )

        for code, (name, dimension) in ROOT_CAUSES.items():
            await session.run(
                """
                MATCH (d:Dimension {entity_code: $dimension})
                MERGE (r:RootCause {entity_code: $code})
                SET r.name = $name,
                    r.entity_type = 'RootCause',
                    r.namespace = 'DIAGNOSIS',
                    r.enabled = true
                MERGE (r)-[rel:BELONGS_TO]->(d)
                SET rel.relation_key = $belongs_key,
                    rel.relation_type = 'BELONGS_TO',
                    rel.enabled = true
                """,
                code=code,
                name=name,
                dimension=dimension,
                belongs_key=f"{code}|BELONGS_TO|{dimension}",
            )

        relation_count = 0
        for scenario_name, root_cause in SCENARIO_TO_ROOT_CAUSE.items():
            for alert_code in _alert_for_scenario(scenario_name):
                relation_key = f"{alert_code}|INDICATES|{root_cause}"
                await session.run(
                    """
                    MATCH (a:Alert {entity_code: $alert_code})
                    MATCH (r:RootCause {entity_code: $root_cause})
                    MERGE (a)-[rel:INDICATES]->(r)
                    SET rel.relation_key = $relation_key,
                        rel.source_entity_code = $alert_code,
                        rel.relation_type = 'INDICATES',
                        rel.target_entity_code = $root_cause,
                        rel.initial_prior_weight = coalesce(rel.initial_prior_weight, 0.10),
                        rel.prior_strength = coalesce(rel.prior_strength, 1.0),
                        rel.effective_weight = coalesce(rel.effective_weight, 0.10),
                        rel.confidence_lower_bound = coalesce(rel.confidence_lower_bound, 0.0),
                        rel.confidence_upper_bound = coalesce(rel.confidence_upper_bound, 0.0),
                        rel.evidence_case_count = coalesce(rel.evidence_case_count, 0),
                        rel.natural_case_count = coalesce(rel.natural_case_count, 0),
                        rel.scenario_case_count = coalesce(rel.scenario_case_count, 0),
                        rel.support_count = coalesce(rel.support_count, 0),
                        rel.against_count = coalesce(rel.against_count, 0),
                        rel.neutral_count = coalesce(rel.neutral_count, 0),
                        rel.support_strength = coalesce(rel.support_strength, 0.0),
                        rel.against_strength = coalesce(rel.against_strength, 0.0),
                        rel.weight_version = coalesce(rel.weight_version, 'SCENARIO_INIT_V0'),
                        rel.enabled = true
                    """,
                    alert_code=alert_code,
                    root_cause=root_cause,
                    relation_key=relation_key,
                )
                relation_count += 1

    await driver.close()
    print(
        "Diagnosis KG seed completed: "
        f"{len(ALERTS)} Alert, {len(ROOT_CAUSES)} RootCause, "
        f"{len(DIMENSIONS)} Dimension, {relation_count} INDICATES relations."
    )


if __name__ == "__main__":
    asyncio.run(seed())
