"""Seed minimal diagnosis KG nodes and Alert -> RootCause edges.

This script follows the formal KG path:
Alert -[:INDICATES]-> RootCause.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from neo4j import AsyncGraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from apps.modelops_api.config import settings
from apps.modelops_api.services.knowledge_observation_mapper import (
    SCENARIO_TO_ROOT_CAUSE,
)

# ── Metric 节点（2026-08-11 定稿：5 个新指标）──
METRICS = {
    # FEATURE_PSI 是已有指标，创建 Metric 节点供 DERIVED_FROM 引用
    "FEATURE_PSI": {
        "name": "特征PSI漂移", "category": "distribution", "metric_role": "PRIMARY_SIGNAL",
        "is_core": True, "direction": "DEVIATION_BAD",
        "trigger_enabled": True, "root_cause_enabled": True, "policy_trigger_enabled": False,
    },
    "PREDICTION_MEAN": {
        "name": "平均预测概率", "category": "distribution", "metric_role": "PRIMARY_SIGNAL",
        "is_core": False, "direction": "TWO_SIDED_DEVIATION",
        "baseline_type": "W0_SAME_MODEL_VERSION", "trigger_enabled": True,
    },
    "MAX_FEATURE_PSI_7D": {
        "name": "7D窗口最大特征PSI", "category": "distribution", "metric_role": "DERIVED_CONTEXT",
        "canonical_metric_code": "FEATURE_PSI", "window_days": 7,
        "trigger_enabled": False, "root_cause_enabled": False, "policy_trigger_enabled": False,
    },
    "MAX_FEATURE_PSI_30D": {
        "name": "30D窗口最大特征PSI", "category": "distribution", "metric_role": "DERIVED_CONTEXT",
        "canonical_metric_code": "FEATURE_PSI", "window_days": 30,
        "trigger_enabled": False, "root_cause_enabled": False, "policy_trigger_enabled": False,
    },
    "DATA_QUALITY_SCORE": {
        "name": "数据质量综合分", "category": "data_quality", "metric_role": "DERIVED_SUMMARY",
        "is_core": False, "is_composite": True, "summary_only": True,
        "trigger_enabled": False, "root_cause_enabled": False, "policy_trigger_enabled": False,
    },
    "OUTLIER_RATE": {
        "name": "离群率异常增量", "category": "data_quality", "metric_role": "PRIMARY_SIGNAL",
        "is_core": True, "direction": "HIGHER_IS_WORSE",
        "value_semantics": "MAX_POSITIVE_DELTA_VS_W0", "baseline_type": "W0_FROZEN",
        "trigger_enabled": True,
    },
}

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
    # 2026-08-11 新增
    "PREDICTION_MEAN_SHIFT": "平均预测概率偏移",
    "OUTLIER_RATE_SPIKE": "离群率突增",
}

ROOT_CAUSES = {
    "business_policy_change": ("业务政策变化", "BUSINESS"),
    "data_pipeline_issue": ("数据管道或预处理问题", "DATA"),
    "data_quality_issue": ("数据质量问题", "DATA"),
    "feature_drift": ("特征分布漂移", "FEATURE"),
    "feature_failure": ("特征失效", "FEATURE"),
    "population_shift": ("客群结构迁移", "BUSINESS"),
    # 2026-08-11: label_distribution_shift 已拆分并删除
    "PRIOR_PROBABILITY_SHIFT": ("标签先验概率漂移", "DATA"),
    "CONCEPT_DRIFT": ("概念漂移", "MODEL"),
    "FRAUD_PATTERN_SHIFT": ("欺诈模式变化", "MODEL"),
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


async def _merge_indicates(
    session,
    alert_code: str,
    root_cause: str,
    weight: float,
    supporting_only: bool = False,
    required_context: list[str] | None = None,
) -> None:
    """创建或更新 INDICATES 关系 — 诊断边唯一创建入口。

    所有治理字段（案例数/置信度/权重版本等）在此统一初始化，
    禁止绕过本函数直接 MERGE INDICATES。causal_distance 不在本函数
    设置，统一由 seed 尾部的治理收窄段落维护（全量覆盖 + DIRECT 白名单）。
    """
    await session.run(
        """
        MATCH (a:Alert {entity_code: $alert_code})
        MATCH (r:RootCause {entity_code: $root_cause})
        MERGE (a)-[rel:INDICATES]->(r)
        SET rel.relation_key = $relation_key,
            rel.source_entity_code = $alert_code,
            rel.relation_type = 'INDICATES',
            rel.target_entity_code = $root_cause,
            rel.initial_prior_weight = coalesce(rel.initial_prior_weight, $weight),
            rel.prior_strength = coalesce(rel.prior_strength, 1.0),
            rel.effective_weight =
              CASE
                WHEN rel.last_calibrated_at IS NULL THEN $weight
                ELSE rel.effective_weight
              END,
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
            rel.supporting_only = coalesce(rel.supporting_only, $supporting_only),
            rel.required_context = coalesce(rel.required_context, $required_context),
            rel.candidate_only = true,
            rel.direct_confirmation = false,
            rel.direct_policy_enabled = false,
            rel.weight_status =
              CASE
                WHEN rel.last_calibrated_at IS NULL THEN 'PENDING_EMPIRICAL_CALIBRATION'
                ELSE rel.weight_status
              END,
            rel.enabled = true
        """,
        alert_code=alert_code, root_cause=root_cause,
        relation_key=f"{alert_code}|INDICATES|{root_cause}",
        weight=weight,
        supporting_only=supporting_only,
        required_context=required_context,
    )


async def _merge_lineage(session, source_metric: str, target_metric: str) -> None:
    """创建 DERIVED_FROM 血缘关系（fixed weight=1.0, diagnosis/policy disabled）。"""
    await session.run(
        """
        MATCH (src:Metric {entity_code: $source})
        MATCH (tgt:Metric {entity_code: $target})
        MERGE (src)-[rel:DERIVED_FROM]->(tgt)
        SET rel.relation_key = $relation_key,
            rel.relation_type = 'DERIVED_FROM',
            rel.relation_role = 'LINEAGE',
            rel.initial_prior_weight = 1.0,
            rel.effective_weight = 1.0,
            rel.weight_semantics = 'LINEAGE_CERTAINTY',
            rel.fixed_weight = true,
            rel.diagnosis_enabled = false,
            rel.policy_enabled = false,
            rel.weight_calibration_enabled = false,
            rel.enabled = true
        """,
        source=source_metric, target=target_metric,
        relation_key=f"{source_metric}|DERIVED_FROM|{target_metric}",
    )


def _concept_drift_weight(root_cause: str, alert_code: str) -> float:
    """CONCEPT_DRIFT 候选召回统一权重 0.20，其余保持默认 0.10。"""
    if root_cause == "CONCEPT_DRIFT":
        return 0.20
    return 0.10


async def seed() -> None:
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    async with driver.session(database="neo4j") as session:
        # ── Metric 节点 ──
        for code, props in METRICS.items():
            await session.run(
                """
                MERGE (m:Metric {entity_code: $code})
                SET m.name = $name,
                    m.entity_type = 'Metric',
                    m.namespace = 'MONITORING',
                    m.category = $category,
                    m.metric_role = $role,
                    m.enabled = true,
                    m.is_core = $is_core,
                    m.trigger_enabled = $trigger_enabled,
                    m.persistence_count_enabled = coalesce(m.persistence_count_enabled, $trigger_enabled),
                    m.root_cause_enabled = coalesce(m.root_cause_enabled, $root_cause_enabled),
                    m.policy_trigger_enabled = coalesce(m.policy_trigger_enabled, $policy_trigger_enabled),
                    m.kg_calibration_enabled = coalesce(m.kg_calibration_enabled, false),
                    m.direction = coalesce(m.direction, $direction),
                    m.baseline_type = coalesce(m.baseline_type, $baseline_type),
                    m.value_semantics = coalesce(m.value_semantics, $value_semantics),
                    m.window_days = coalesce(m.window_days, $window_days),
                    m.canonical_metric_code = coalesce(m.canonical_metric_code, $canonical_metric_code),
                    m.is_composite = coalesce(m.is_composite, $is_composite),
                    m.summary_only = coalesce(m.summary_only, $summary_only)
                """,
                code=code,
                name=props["name"],
                category=props.get("category", ""),
                role=props.get("metric_role", ""),
                is_core=props.get("is_core", False),
                trigger_enabled=props.get("trigger_enabled", False),
                root_cause_enabled=props.get("root_cause_enabled", True),
                policy_trigger_enabled=props.get("policy_trigger_enabled", True),
                direction=props.get("direction", ""),
                baseline_type=props.get("baseline_type", ""),
                value_semantics=props.get("value_semantics", ""),
                window_days=props.get("window_days", 0),
                canonical_metric_code=props.get("canonical_metric_code", ""),
                is_composite=props.get("is_composite", False),
                summary_only=props.get("summary_only", False),
            )

        # ── Alert 节点 ──
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

        # ── Dimension 节点 ──
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

        # ── RootCause 节点 + BELONGS_TO ──
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

            # FRAUD_PATTERN_SHIFT 特殊属性
            if code == "FRAUD_PATTERN_SHIFT":
                await session.run(
                    """
                    MATCH (r:RootCause {entity_code: $code})
                    SET r.semantic_parent_code = 'CONCEPT_DRIFT',
                        r.semantic_parent_type = 'SPECIALIZED_CONCEPT_DRIFT_PROXY',
                        r.inherit_candidate_weight = false,
                        r.inherit_diagnosis_status = false,
                        r.inherit_policy = false
                    """, code=code)
            if code == "CONCEPT_DRIFT":
                await session.run(
                    """
                    MATCH (r:RootCause {entity_code: $code})
                    SET r.candidate_only = true,
                        r.direct_policy_enabled = false,
                        r.requires_subtype_or_scope_classification = true
                    """, code=code)

        # ── INDICATES (scenario-based) ──
        relation_count = 0
        for scenario_name, root_cause in SCENARIO_TO_ROOT_CAUSE.items():
            for alert_code in _alert_for_scenario(scenario_name):
                await _merge_indicates(
                    session, alert_code, root_cause,
                    _concept_drift_weight(root_cause, alert_code),
                )
                relation_count += 1

        # ── 2026-08-11: 专用关系（非 scenario-based）──

        # PREDICTION_MEAN → TRIGGERS → PREDICTION_MEAN_SHIFT
        await session.run("""
            MATCH (m:Metric {entity_code: 'PREDICTION_MEAN'})
            MATCH (a:Alert {entity_code: 'PREDICTION_MEAN_SHIFT'})
            MERGE (m)-[r:TRIGGERS]->(a)
            SET r.relation_key = 'PREDICTION_MEAN|TRIGGERS|PREDICTION_MEAN_SHIFT',
                r.relation_type = 'TRIGGERS', r.enabled = true
        """)
        relation_count += 1

        # PREDICTION_MEAN_SHIFT → INDICATES → PRIOR_PROBABILITY_SHIFT
        # 预测均值偏移不能单独证明标签先验变化（也可能来自特征漂移/客群变化/管道问题），
        # required_context=prior_probability_evidence 要求 BAD_RATE_SHIFT 同现佐证
        #（诊断服务排除自身告警上下文，见 diagnosis_service._recall_candidates）。
        await _merge_indicates(
            session, "PREDICTION_MEAN_SHIFT", "PRIOR_PROBABILITY_SHIFT", 0.10,
            required_context=["prior_probability_evidence"],
        )
        relation_count += 1

        # BAD_RATE_SHIFT → INDICATES → PRIOR_PROBABILITY_SHIFT (candidate_only, weight=0.30)
        await _merge_indicates(session, "BAD_RATE_SHIFT", "PRIOR_PROBABILITY_SHIFT", 0.30)
        relation_count += 1

        # MAX_FEATURE_PSI_7D → DERIVED_FROM → Metric:FEATURE_PSI (fixed 1.0, lineage only)
        await _merge_lineage(session, "MAX_FEATURE_PSI_7D", "FEATURE_PSI")
        relation_count += 1

        # MAX_FEATURE_PSI_30D → DERIVED_FROM → Metric:FEATURE_PSI (fixed 1.0, lineage only)
        await _merge_lineage(session, "MAX_FEATURE_PSI_30D", "FEATURE_PSI")
        relation_count += 1

        # OUTLIER_RATE → TRIGGERS → OUTLIER_RATE_SPIKE
        await session.run("""
            MATCH (m:Metric {entity_code: 'OUTLIER_RATE'})
            MATCH (a:Alert {entity_code: 'OUTLIER_RATE_SPIKE'})
            MERGE (m)-[r:TRIGGERS]->(a)
            SET r.relation_key = 'OUTLIER_RATE|TRIGGERS|OUTLIER_RATE_SPIKE',
                r.relation_type = 'TRIGGERS', r.enabled = true
        """)
        relation_count += 1

        # OUTLIER_RATE_SPIKE → INDICATES → data_quality_issue (candidate_only, weight=0.10)
        await _merge_indicates(session, "OUTLIER_RATE_SPIKE", "data_quality_issue", 0.10)
        relation_count += 1

        # CONCEPT_DRIFT 候选召回: AUC_DROP/KS_DROP/PR_AUC_DROP/BAD_RECALL_DROP (weight=0.20)
        for alert_code in ("AUC_DROP", "KS_DROP", "PR_AUC_DROP", "BAD_RECALL_DROP"):
            await _merge_indicates(session, alert_code, "CONCEPT_DRIFT", 0.20)
            relation_count += 1

        # ── 2026-08-14 治理收窄 ──

        # SCHEMA_MISMATCH → data_pipeline_issue（强数据契约证据, weight=0.35）
        # required_context=schema_contract_violation 只在告警详情包含真实列级差异时
        # 才被诊断服务授予（payload 证据，不再凭告警码自证）。
        await _merge_indicates(
            session, "SCHEMA_MISMATCH", "data_pipeline_issue", 0.35,
            required_context=["schema_contract_violation"],
        )
        relation_count += 1

        # ── 2026-08-15 治理收窄：HIGH_FEATURE_PSI → feature_drift ──
        # covariate_drift 场景的规范映射（SCENARIO_TO_ROOT_CAUSE），DIRECT 因果边。
        # 诊断 PathRanker 公式（weight×0.6 + evidence×0.4）下，0.10 权重使所有
        # 候选最高只有 0.46 分，L1 自动迭代门槛（iteration.yaml）永远不可达，
        # 自动链路退化为人工作业。0.60 是该告警→该根因的语义先验：
        # "特征 PSI 漂移"本身就是"特征分布漂移"的直接证据（DIRECT 白名单），
        # 业务政策变化/数据管道等竞对候选保持 0.10 弱先验。
        await _merge_indicates(session, "HIGH_FEATURE_PSI", "feature_drift", 0.60)
        relation_count += 1

        # ── 断链告警补齐 ──

        # CALIBRATION_DEGRADE → PRIOR_PROBABILITY_SHIFT（需 BAD_RATE_SHIFT 佐证）
        #                      + CONCEPT_DRIFT（候选）
        await _merge_indicates(
            session, "CALIBRATION_DEGRADE", "PRIOR_PROBABILITY_SHIFT", 0.20,
            required_context=["prior_probability_evidence"],
        )
        relation_count += 1
        await _merge_indicates(session, "CALIBRATION_DEGRADE", "CONCEPT_DRIFT", 0.15)
        relation_count += 1

        # HIGH_SCORE_PSI → 分数漂移的下游根因（全部 INDIRECT，由尾部统一段标记）
        await _merge_indicates(session, "HIGH_SCORE_PSI", "feature_drift", 0.25)
        relation_count += 1
        await _merge_indicates(session, "HIGH_SCORE_PSI", "population_shift", 0.15)
        relation_count += 1
        await _merge_indicates(session, "HIGH_SCORE_PSI", "business_policy_change", 0.10)
        relation_count += 1

        # PERFORMANCE_DECAY 是 AUC/KS 等性能结果的汇总，标记 summary_only，
        # 禁止独立根因召回（召回查询过滤 root_cause_recall_enabled=false）。
        await session.run("""
            MATCH (a:Alert {entity_code: 'PERFORMANCE_DECAY'})
            SET a.summary_only = true,
                a.root_cause_recall_enabled = false
        """)

        # SAMPLE_SIZE_LOW → gate_only，不连 RootCause
        await session.run("""
            MATCH (a:Alert {entity_code: 'SAMPLE_SIZE_LOW'})
            SET a.gate_only = true,
                a.gate_semantics = 'DATA_ELIGIBILITY_BLOCK',
                a.root_cause_recall_enabled = false
        """)

        # AUC_DROP/KS_DROP → PRIOR_PROBABILITY_SHIFT 弱化为 supporting_only (weight=0.03)
        await session.run("""
            MATCH (a:Alert)-[r:INDICATES]->(rc:RootCause {entity_code: 'PRIOR_PROBABILITY_SHIFT'})
            WHERE a.entity_code IN ['AUC_DROP', 'KS_DROP']
            SET r.supporting_only = true,
                r.initial_prior_weight = 0.03,
                r.effective_weight =
                  CASE
                    WHEN r.last_calibrated_at IS NULL THEN 0.03
                    ELSE r.effective_weight
                  END
        """)

        # BAD_RATE_SHIFT → CONCEPT_DRIFT / FRAUD_PATTERN_SHIFT 弱化为辅助关系
        await session.run("""
            MATCH (a:Alert {entity_code: 'BAD_RATE_SHIFT'})-[r:INDICATES]->(rc:RootCause)
            WHERE rc.entity_code IN ['CONCEPT_DRIFT', 'FRAUD_PATTERN_SHIFT']
            SET r.supporting_only = true,
                r.initial_prior_weight = 0.05,
                r.effective_weight =
                  CASE
                    WHEN r.last_calibrated_at IS NULL THEN 0.05
                    ELSE r.effective_weight
                  END
        """)

        # ── causal_distance 全量覆盖 ──
        # 默认 INDIRECT：性能/分布汇总类告警不足以直接证明具体根因。
        await session.run("""
            MATCH ()-[r:INDICATES]->(:RootCause)
            WHERE r.causal_distance IS NULL
            SET r.causal_distance = 'INDIRECT'
        """)
        # DIRECT 白名单：告警本身就是该根因的直接证据。
        await session.run("""
            UNWIND $pairs AS p
            MATCH (a:Alert {entity_code: p[0]})-[r:INDICATES]->(rc:RootCause {entity_code: p[1]})
            SET r.causal_distance = 'DIRECT'
        """, pairs=[
            ["HIGH_FEATURE_PSI", "data_pipeline_issue"],
            ["HIGH_FEATURE_PSI", "data_quality_issue"],
            ["HIGH_FEATURE_PSI", "feature_drift"],
            ["SCHEMA_MISMATCH", "data_pipeline_issue"],
            ["MISSING_RATE_SPIKE", "data_quality_issue"],
            ["OUTLIER_RATE_SPIKE", "data_quality_issue"],
            ["BAD_RATE_SHIFT", "PRIOR_PROBABILITY_SHIFT"],
        ])

        # ═══════════════════════════════════════════════════════════
        # ── A7 策略层（人工确认定稿 + 前后端接口契约 V1.0）──
        # 权威规则：KG 只提供咨询候选（candidate_only/advisory_only），
        # L1（assets/configs/repair_strategies.yaml 决策规则）是最终策略权威。
        # 关系键按契约 §10.2 使用小写根因码：feature_drift|RECOMMENDS|...。
        # A7 的 SEGMENT_DRIFT 对应契约诊断层根因 population_shift（客群结构迁移），
        # 不新增根因节点。正式 A7 Strategy 码恰好 8 个，不得另建同义别名。
        # ═══════════════════════════════════════════════════════════
        # 共同约束（A7 §2）：KG 候选不可直接执行，必须过 L1 校验
        # design_status 单独传参：8 个正式策略统一 HUMAN_CONFIRMED_V2
        _A7_COMMON = {
            "enabled": True,
            "model_task_type": "BINARY_CLASSIFICATION",
            "same_algorithm_family_required": True,
            "max_business_rounds": 2,
            "direct_execution_enabled": False,
            "l1_validation_required": True,
        }
        _strategies = {
            "recent_weighted_retrain": {
                "name": "近期加权重训", "risk_level": "MEDIUM", "stage": "FIRST_ROUND",
                "plan_code": "PLAN_RECENT_WEIGHTED", "executor_code": "MODEL_ITERATION",
                "training_cost_level": "MEDIUM",
                "training_window_ids": ["W2"], "validation_window_ids": ["W3"],
                "description": "真实消费近期样本权重，保持Champion算法族",
            },
            "sliding_window_retrain": {
                "name": "滑动窗口重训", "risk_level": "MEDIUM", "stage": "FIRST_ROUND",
                "plan_code": "PLAN_SLIDING_RECENT", "executor_code": "MODEL_ITERATION",
                "training_cost_level": "MEDIUM",
                "training_window_ids": ["W2"], "validation_window_ids": ["W3"],
                "description": "同算法族在滑动窗口上从头拟合",
            },
            "segment_weighted_retrain": {
                "name": "客群加权重训", "risk_level": "MEDIUM", "stage": "FIRST_ROUND",
                "plan_code": "PLAN_SEGMENT_WEIGHTED", "executor_code": "MODEL_ITERATION",
                "training_cost_level": "MEDIUM",
                "training_window_ids": ["W2"], "validation_window_ids": ["W3"],
                "description": "必须有冻结客群并真实消费客群权重",
            },
            "full_retrain": {
                "name": "全量重训", "risk_level": "HIGH", "stage": "FIRST_ROUND",
                "plan_code": "PLAN_FULL_RETRAIN", "executor_code": "MODEL_ITERATION",
                "training_cost_level": "HIGH",
                "training_window_ids": ["W2"], "validation_window_ids": ["W3"],
                "description": "仅特殊GLOBAL条件且人工批准",
            },
            "regularized_retrain": {
                "name": "正则化重训", "risk_level": "MEDIUM", "stage": "SECOND_ROUND",
                "plan_code": "PLAN_REGULARIZED", "executor_code": "MODEL_ITERATION",
                "training_cost_level": "MEDIUM",
                "training_window_ids": ["W2"], "validation_window_ids": ["W3"],
                "description": "仅W3失败归因确认参数/过拟合限制后使用",
            },
            "feature_reconstruction": {
                "name": "特征重构", "risk_level": "HIGH", "stage": "SECOND_ROUND",
                "plan_code": "PLAN_FEATURE_RECONSTRUCTION", "executor_code": "MODEL_ITERATION",
                "training_cost_level": "HIGH",
                "training_window_ids": ["W2"], "validation_window_ids": ["W3"],
                "description": "仅特征表达不足疑似且人工批准后使用",
            },
            # 任务三硬要求：增量训练与特征筛选两种自适应策略
            "incremental_retrain": {
                "name": "增量重训", "risk_level": "MEDIUM", "stage": "FIRST_ROUND",
                "plan_code": "PLAN_INCREMENTAL", "executor_code": "MODEL_ITERATION",
                "training_cost_level": "MEDIUM",
                "training_window_ids": ["W2"], "validation_window_ids": ["W3"],
                "description": "SUSTAINED_30D 稳定渐变时增量消费近期样本",
            },
            "feature_selection_retrain": {
                "name": "特征筛选重训", "risk_level": "MEDIUM", "stage": "SECOND_ROUND",
                "plan_code": "PLAN_FEATURE_SELECTION", "executor_code": "MODEL_ITERATION",
                "training_cost_level": "MEDIUM",
                "training_window_ids": ["W2"], "validation_window_ids": ["W3"],
                "description": "W3失败归因特征冗余/脆弱后筛选特征并重训",
            },
        }

        # 主训练模式（A7 定稿 §7 正式枚举，沿 Strategy→Candidate→Proposal→
        # TrainingPlan→TrainingJobInput→Worker 传递，禁止从 strategy_tier 猜测）
        _training_modes = {
            "incremental_retrain": "INCREMENTAL_TRAIN",
            "regularized_retrain": "PARAMETER_TUNING",
            "feature_selection_retrain": "FEATURE_SELECTION",
            "feature_reconstruction": "FEATURE_RECONSTRUCTION",
        }
        _l1_training_modes = {
            "champion_replay": "NONE",
            "calibration_only": "NONE",
            "threshold_only": "NONE",
        }

        def _pipeline_stages(code: str) -> list[str]:
            """A7 §3: 各策略的管道阶段组合。"""
            stages = ["DATA_ELIGIBILITY", "MODEL_TRAINING", "MODEL_EVALUATION"]
            if code == "regularized_retrain":
                stages.insert(1, "HYPERPARAMETER_TUNING")
            if code == "feature_selection_retrain":
                stages.insert(1, "FEATURE_SELECTION")
            if code == "feature_reconstruction":
                stages.insert(1, "FEATURE_RECONSTRUCTION")
            return stages

        # ── A7 §2.1: 正式根因码（大写）+ 历史码迁移 ──
        # FEATURE_DRIFT  ← feature_drift
        # SEGMENT_DRIFT  ← population_shift
        # 诊断层继续使用小写历史码；A7 策略边只建在新正式码上，
        # 旧码边退役（不物理删除，不静默改已校准关系键）。
        await session.run("""
            MATCH (d:Dimension {entity_code: 'FEATURE'})
            MERGE (rc:RootCause {entity_code: 'FEATURE_DRIFT'})
            SET rc.name = '特征分布漂移（A7正式码）',
                rc.entity_type = 'RootCause',
                rc.namespace = 'ITERATION',
                rc.enabled = true
            MERGE (rc)-[bel:BELONGS_TO]->(d)
            SET bel.relation_key = 'FEATURE_DRIFT|BELONGS_TO|FEATURE',
                bel.relation_type = 'BELONGS_TO',
                bel.enabled = true
        """)
        await session.run("""
            MATCH (d:Dimension {entity_code: 'BUSINESS'})
            MERGE (rc:RootCause {entity_code: 'SEGMENT_DRIFT'})
            SET rc.name = '客群结构漂移（A7正式码）',
                rc.entity_type = 'RootCause',
                rc.namespace = 'ITERATION',
                rc.enabled = true
            MERGE (rc)-[bel:BELONGS_TO]->(d)
            SET bel.relation_key = 'SEGMENT_DRIFT|BELONGS_TO|BUSINESS',
                bel.relation_type = 'BELONGS_TO',
                bel.enabled = true
        """)

        # ── A7 §4 允许关系（任务三范围：8 策略、6+6 条边）──
        # [(root_cause, strategy, applicability_condition, min_severity,
        #   min_decay_level, min_business_round, required_context)]
        # min_decay_level: NONE=0 / SHORT_TERM_7D=1 / SUSTAINED_30D=2 / SEVERE=3
        # min_business_round: 1=第一轮可召回, 2=仅第二轮（W3 失败归因后）
        # 任务一统一入口：SHORT_TERM_7D → 继续观察；SUSTAINED_30D → 可进 A7；
        # SEVERE → 人工复核，不自动训练。
        _a7_edges = [
            ("FEATURE_DRIFT", "recent_weighted_retrain",
             "SUSTAINED_30D；近期分布稳定；A7数据门禁通过", 0.3, 2, 1, None),
            ("FEATURE_DRIFT", "sliding_window_retrain",
             "SUSTAINED_30D；变化窗口已确认；A7数据门禁通过", 0.4, 2, 1, None),
            ("FEATURE_DRIFT", "incremental_retrain",
             "SUSTAINED_30D + LOCAL/GRADUAL；Champion产物可加载；Schema兼容；"
             "算法支持增量训练", 0.4, 2, 1,
             ["sustained_30d", "champion_artifact_available",
              "schema_compatible", "incremental_algorithm_supported"]),
            ("FEATURE_DRIFT", "full_retrain",
             "SEVERE + GLOBAL；近期新分布稳定；人工明确批准", 0.6, 3, 1, None),
            ("FEATURE_DRIFT", "feature_selection_retrain",
             "不稳定特征子集已确认；有筛选依据；第二轮使用；人工批准", 0.5, 0, 2,
             ["unstable_feature_subset_confirmed",
              "feature_selection_evidence_available", "manual_approval"]),
            ("SEGMENT_DRIFT", "segment_weighted_retrain",
             "合格冻结客群；客群权重可执行；SEVERE时人工批准", 0.3, 0, 1, None),
        ]

        _allowed_relation_keys = {
            f"{rc}|RECOMMENDS|{st}" for rc, st, _c, _m, _d, _b, _x in _a7_edges
        } | {
            f"{st}|MITIGATES|{rc}" for rc, st, _c, _m, _d, _b, _x in _a7_edges
        }

        # ── 退役：非 A7 允许范围的策略边（enabled=false + 退役标记，不物理删除）──
        # 已校准权重/案例数/last_calibrated_at 保留，允许原位恢复；MERGE 原位更新
        # 的 SET 对已校准边使用 CASE 保护，校准历史不会因重跑 seed 丢失。
        # relation_key IS NULL 的历史脏边一并退役。
        _now = datetime.now(timezone.utc).isoformat()
        # 先退役历史根因码边（迁移原因），再处理其余非允许边（保留已有退役原因）
        await session.run("""
            MATCH (:RootCause)-[r:RECOMMENDS]->(:Strategy)
            WHERE r.relation_key STARTS WITH 'feature_drift|'
               OR r.relation_key STARTS WITH 'population_shift|'
            SET r.enabled = false,
                r.deprecated_reason = 'ROOT_CAUSE_CODE_MIGRATED',
                r.deprecated_at = coalesce(r.deprecated_at, $now)
        """, now=_now)
        await session.run("""
            MATCH (:Strategy)-[r:MITIGATES]->(:RootCause)
            WHERE r.relation_key ENDS WITH '|feature_drift'
               OR r.relation_key ENDS WITH '|population_shift'
            SET r.enabled = false,
                r.deprecated_reason = 'ROOT_CAUSE_CODE_MIGRATED',
                r.deprecated_at = coalesce(r.deprecated_at, $now)
        """, now=_now)
        await session.run("""
            MATCH (:RootCause)-[r:RECOMMENDS]->(:Strategy)
            WHERE (r.relation_key IS NULL OR NOT r.relation_key IN $allowed)
            SET r.enabled = false,
                r.deprecated_reason = coalesce(
                    r.deprecated_reason, 'NOT_IN_A7_STRATEGY_V1'),
                r.deprecated_at = coalesce(r.deprecated_at, $now)
        """, allowed=list(_allowed_relation_keys), now=_now)
        await session.run("""
            MATCH (:Strategy)-[r:MITIGATES]->(:RootCause)
            WHERE (r.relation_key IS NULL OR NOT r.relation_key IN $allowed)
            SET r.enabled = false,
                r.deprecated_reason = coalesce(
                    r.deprecated_reason, 'NOT_IN_A7_STRATEGY_V1'),
                r.deprecated_at = coalesce(r.deprecated_at, $now)
        """, allowed=list(_allowed_relation_keys), now=_now)

        # L1 专属策略节点（YAML 规则使用，不进 KG 咨询层）
        _l1_strategies = {
            "champion_replay": {
                "name": "冻结Champion回放", "risk_level": "LOW",
                "plan_code": "PLAN_NO_TRAINING", "executor_code": "DATA_REPAIR",
                "training_cost_level": "LOW",
                "training_window_ids": [], "validation_window_ids": [],
                "algorithm": "",
                "description": "修复数据或流水线后使用冻结 Champion 回放",
            },
            "stable_refit": {
                "name": "同配置稳定重拟合", "risk_level": "LOW",
                "plan_code": "PLAN_STABLE_REFIT", "executor_code": "MODEL_ITERATION",
                "training_cost_level": "LOW",
                "training_window_ids": ["W2"], "validation_window_ids": ["W3"],
                "algorithm": "lightgbm",
                "description": "同算法、同特征、同超参数重新拟合",
            },
            "calibration_only": {
                "name": "仅概率校准", "risk_level": "MEDIUM",
                "plan_code": "PLAN_CALIBRATION_ONLY", "executor_code": "CALIBRATION_ADJUSTMENT",
                "training_cost_level": "LOW",
                "training_window_ids": ["W2"], "validation_window_ids": ["W3"],
                "algorithm": "",
                "description": "排序稳定时仅调整概率校准器",
            },
            "threshold_only": {
                "name": "仅业务阈值调整", "risk_level": "HIGH",
                "plan_code": "PLAN_THRESHOLD_ONLY", "executor_code": "THRESHOLD_ADJUSTMENT",
                "training_cost_level": "LOW",
                "training_window_ids": [], "validation_window_ids": [],
                "algorithm": "",
                "description": "排序与校准正常时调整业务阈值",
            },
        }

        for code, props in _strategies.items():
            await session.run(
                """
                MERGE (s:Strategy {entity_code: $code})
                SET s.name = $name,
                    s.entity_type = 'Strategy',
                    s.namespace = 'ITERATION',
                    s.description = $description,
                    s.risk_level = $risk_level,
                    s.plan_code = $plan_code,
                    s.executor_code = $executor_code,
                    s.training_cost_level = $training_cost_level,
                    s.allowed_training_window_ids = $training_window_ids,
                    s.validation_window_ids = $validation_window_ids,
                    s.algorithm = $algorithm,
                    s.feature_schema_version = $feature_schema_version,
                    s.preprocessing_version = $preprocessing_version,
                    s.label_versions = [],
                    // Neo4j 不支持 Map 属性，以 JSON 字符串存储
                    s.hyperparameters = coalesce(s.hyperparameters, '{}'),
                    s.sample_weight_policy = coalesce(s.sample_weight_policy, '{}'),
                    // A7 共同约束
                    s.model_task_type = $model_task_type,
                    s.same_algorithm_family_required = $same_algorithm_family_required,
                    s.max_business_rounds = $max_business_rounds,
                    s.direct_execution_enabled = $direct_execution_enabled,
                    s.l1_validation_required = $l1_validation_required,
                    s.design_status = $design_status,
                    s.strategy_stage = $stage,
                    s.strategy_scope = 'A7_TRAINING',
                    s.primary_training_mode = $primary_training_mode,
                    // A7 §3 节点合同补充
                    s.allowed_trigger_types = $allowed_trigger_types,
                    s.pipeline_stage_codes = $pipeline_stage_codes,
                    s.supported_algorithms = ['lightgbm'],
                    s.base_model_required = $base_model_required,
                    s.requires_manual_approval = $requires_manual_approval,
                    s.execution_config_json = '{}',
                    s.enabled = true
                """,
                code=code,
                name=props["name"],
                description=props["description"],
                risk_level=props["risk_level"],
                plan_code=props["plan_code"],
                executor_code=props["executor_code"],
                training_cost_level=props["training_cost_level"],
                training_window_ids=props["training_window_ids"],
                validation_window_ids=props["validation_window_ids"],
                algorithm="lightgbm",
                feature_schema_version="FEATURE_SCHEMA_V1",
                preprocessing_version="",
                stage=props["stage"],
                primary_training_mode=_training_modes.get(code, "FULL_RETRAIN"),
                allowed_trigger_types=[
                    "SCHEDULED_TRIGGER", "THRESHOLD_TRIGGER", "ABNORMAL_TRIGGER",
                ],
                pipeline_stage_codes=_pipeline_stages(code),
                base_model_required=(code == "incremental_retrain"),
                requires_manual_approval=(
                    code in {"full_retrain", "feature_selection_retrain",
                             "feature_reconstruction"}
                ),
                **_A7_COMMON,
                # A7 实施定稿 §3：8 个正式策略统一 HUMAN_CONFIRMED_V2
                design_status="HUMAN_CONFIRMED_V2",
            )

        for code, props in _l1_strategies.items():
            await session.run(
                """
                MERGE (s:Strategy {entity_code: $code})
                SET s.name = $name,
                    s.entity_type = 'Strategy',
                    s.namespace = 'ITERATION',
                    s.description = $description,
                    s.risk_level = $risk_level,
                    s.plan_code = $plan_code,
                    s.executor_code = $executor_code,
                    s.training_cost_level = $training_cost_level,
                    s.allowed_training_window_ids = $training_window_ids,
                    s.validation_window_ids = $validation_window_ids,
                    s.algorithm = $algorithm,
                    s.feature_schema_version = '',
                    s.preprocessing_version = '',
                    s.label_versions = [],
                    s.hyperparameters = coalesce(s.hyperparameters, '{}'),
                    s.sample_weight_policy = coalesce(s.sample_weight_policy, '{}'),
                    // L1 专属：不进 KG 咨询层（YAML 决策规则直接使用）
                    s.strategy_scope = 'L1_ONLY',
                    s.design_status = 'YAML_V1',
                    s.direct_execution_enabled = false,
                    s.l1_validation_required = true,
                    s.primary_training_mode = $primary_training_mode,
                    s.enabled = true
                """,
                code=code,
                name=props["name"],
                description=props["description"],
                risk_level=props["risk_level"],
                plan_code=props["plan_code"],
                executor_code=props["executor_code"],
                training_cost_level=props["training_cost_level"],
                training_window_ids=props["training_window_ids"],
                validation_window_ids=props["validation_window_ids"],
                algorithm=props["algorithm"],
                primary_training_mode=_training_modes.get(
                    code,
                    _l1_training_modes.get(code, "FULL_RETRAIN"),
                ),
            )

        for (root_cause, strategy_code, condition, min_severity,
             min_decay_level, min_business_round, required_context) in _a7_edges:
            rkey = f"{root_cause}|RECOMMENDS|{strategy_code}"
            mkey = f"{strategy_code}|MITIGATES|{root_cause}"
            applicable_algorithms = (
                ["lightgbm", "xgboost"]
                if strategy_code == "incremental_retrain"
                else None
            )
            await session.run(
                """
                MATCH (rc:RootCause {entity_code: $root_cause})
                MATCH (s:Strategy {entity_code: $strategy_code})
                MERGE (rc)-[rec:RECOMMENDS]->(s)
                SET rec.relation_key = $rkey,
                    rec.relation_type = 'RECOMMENDS',
                    // A7 §4.1: 咨询候选，不直接生成 TrainingPlan
                    rec.candidate_only = true,
                    rec.advisory_only = true,
                    rec.direct_execution = false,
                    rec.l1_validation_required = true,
                    rec.applicability_condition = $condition,
                    // A7 §6.1: 边级证据门控（L1/运行时校验）
                    rec.required_context = coalesce(rec.required_context, $required_context),
                    // A7 §10 诚实初始化：统一 0.10 先验，不写伪造历史有效率。
                    // 以下字段全部按 last_calibrated_at 保护，已校准边不被覆盖。
                    rec.initial_prior_weight = coalesce(rec.initial_prior_weight, 0.10),
                    rec.effective_weight =
                      CASE
                        WHEN rec.last_calibrated_at IS NULL THEN 0.10
                        ELSE rec.effective_weight
                      END,
                    rec.evidence_case_count = coalesce(rec.evidence_case_count, 0),
                    rec.natural_case_count = coalesce(rec.natural_case_count, 0),
                    rec.support_case_count = coalesce(rec.support_case_count, 0),
                    rec.total_case_count = coalesce(rec.total_case_count, 0),
                    rec.seed_status =
                      CASE
                        WHEN rec.last_calibrated_at IS NULL THEN 'UNVALIDATED_PRIOR'
                        ELSE rec.seed_status
                      END,
                    rec.weight_version =
                      CASE
                        WHEN rec.last_calibrated_at IS NULL THEN 'A7_STRATEGY_SEED_V2'
                        ELSE rec.weight_version
                      END,
                    rec.threshold_status =
                      CASE
                        WHEN rec.last_calibrated_at IS NULL THEN 'PENDING_EMPIRICAL_CALIBRATION'
                        ELSE rec.threshold_status
                      END,
                    // 结构化适用范围（运行时过滤，不再只靠文本条件）
                    rec.min_decay_level =
                      CASE
                        WHEN rec.last_calibrated_at IS NULL THEN $min_decay_level
                        ELSE coalesce(rec.min_decay_level, $min_decay_level)
                      END,
                    rec.min_business_round =
                      CASE
                        WHEN rec.last_calibrated_at IS NULL THEN $min_business_round
                        ELSE coalesce(rec.min_business_round, $min_business_round)
                      END,
                    rec.min_severity =
                      CASE
                        WHEN rec.last_calibrated_at IS NULL THEN $min_severity
                        ELSE coalesce(rec.min_severity, $min_severity)
                      END,
                    rec.max_severity = coalesce(rec.max_severity, 1.0),
                    rec.applicable_algorithms = $applicable_algorithms,
                    rec.strategy_tier = coalesce(rec.strategy_tier, 'full'),
                    rec.enabled = true
                // 允许集合内的边若此前被退役，重新启用时清除退役标记
                REMOVE rec.deprecated_reason, rec.deprecated_at
                MERGE (s)-[mit:MITIGATES]->(rc)
                SET mit.relation_key = $mkey,
                    mit.relation_type = 'MITIGATES',
                    // A7 §4.2: 能力覆盖与结果审计，不是已验证声明、不是授权票
                    mit.relation_semantics = 'CAPABILITY_AND_OUTCOME_AUDIT',
                    mit.mandatory_for_execution = false,
                    mit.missing_edge_policy = 'AUDIT_AND_L1_FALLBACK',
                    mit.effective_weight = coalesce(mit.effective_weight, 0.10),
                    mit.evidence_case_count = coalesce(mit.evidence_case_count, 0),
                    mit.confidence_lower_bound = coalesce(mit.confidence_lower_bound, 0.0),
                    mit.seed_status =
                      CASE
                        WHEN mit.last_calibrated_at IS NULL THEN 'UNVALIDATED_PRIOR'
                        ELSE mit.seed_status
                      END,
                    mit.weight_version =
                      CASE
                        WHEN mit.last_calibrated_at IS NULL THEN 'A7_STRATEGY_SEED_V2'
                        ELSE mit.weight_version
                      END,
                    mit.threshold_status =
                      CASE
                        WHEN mit.last_calibrated_at IS NULL THEN 'PENDING_EMPIRICAL_CALIBRATION'
                        ELSE mit.threshold_status
                      END,
                    mit.enabled = true
                REMOVE mit.deprecated_reason, mit.deprecated_at
                """,
                root_cause=root_cause,
                strategy_code=strategy_code,
                rkey=rkey,
                mkey=mkey,
                condition=condition,
                min_severity=min_severity,
                min_decay_level=min_decay_level,
                min_business_round=min_business_round,
                required_context=required_context,
                applicable_algorithms=applicable_algorithms,
            )
            relation_count += 2

        # A7 §5：historical_effectiveness 与真实结果绑定，无结果时属性必须不存在
        await session.run("""
            MATCH ()-[rec:RECOMMENDS]->(:Strategy)
            WHERE rec.last_calibrated_at IS NULL
            REMOVE rec.historical_effectiveness
        """)
        await session.run("""
            MATCH (:Strategy)-[mit:MITIGATES]->(:RootCause)
            WHERE mit.last_calibrated_at IS NULL
            REMOVE mit.historical_effectiveness
        """)

    await driver.close()
    print(
        "Diagnosis KG seed completed: "
        f"{len(METRICS)} Metric, {len(ALERTS)} Alert, {len(ROOT_CAUSES)} RootCause, "
        f"{len(DIMENSIONS)} Dimension, "
        f"{len(_strategies)} A7 Strategy + {len(_l1_strategies)} L1-only Strategy, "
        f"{relation_count} total relations."
    )


if __name__ == "__main__":
    asyncio.run(seed())
