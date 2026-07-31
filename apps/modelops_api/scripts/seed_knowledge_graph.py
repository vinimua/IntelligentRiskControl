"""Seed 知识图谱 V2 — 严格按《模型风险诊断知识图谱设计整理_V1.1》。

任务一：Feature → Metric → Alert（三层，无冗余 Severity）
任务二+的节点预留结构，暂不创建。

幂等 — 使用 MERGE。
"""

from __future__ import annotations

import asyncio
import sys

from neo4j import AsyncGraphDatabase

from apps.modelops_api.config import settings

# ═══════════════════════════════════════════════════════════
# 实体定义（严格来自设计文档 §4）
# ═══════════════════════════════════════════════════════════

# §4.1 Feature — 34 个入模特征
FEATURES = [
    # credit/risk features
    {"entity_code": "credit_query_times", "name": "征信查询次数", "data_type": "NUMERIC", "feature_group": "credit_risk"},
    {"entity_code": "multi_loan_count", "name": "多头借贷数量", "data_type": "NUMERIC", "feature_group": "credit_risk"},
    {"entity_code": "overdue_history", "name": "逾期历史", "data_type": "NUMERIC", "feature_group": "credit_risk"},
    {"entity_code": "credit_utilization", "name": "信用额度使用率", "data_type": "NUMERIC", "feature_group": "credit_risk"},
    {"entity_code": "credit_length_months", "name": "信用时长(月)", "data_type": "NUMERIC", "feature_group": "credit_risk"},
    {"entity_code": "max_overdue_days", "name": "最大逾期天数", "data_type": "NUMERIC", "feature_group": "credit_risk"},
    # scoring features
    {"entity_code": "social_score", "name": "社交评分", "data_type": "NUMERIC", "feature_group": "scoring"},
    {"entity_code": "telecom_score", "name": "电信评分", "data_type": "NUMERIC", "feature_group": "scoring"},
    {"entity_code": "ecomm_risk_score", "name": "电商风险评分", "data_type": "NUMERIC", "feature_group": "scoring"},
    {"entity_code": "judicial_risk_score", "name": "司法风险评分", "data_type": "NUMERIC", "feature_group": "scoring"},
    {"entity_code": "blacklist_hit", "name": "黑名单命中", "data_type": "NUMERIC", "feature_group": "scoring"},
    {"entity_code": "device_risk_score", "name": "设备风险评分", "data_type": "NUMERIC", "feature_group": "scoring"},
    # behavior features
    {"entity_code": "app_duration", "name": "APP使用时长", "data_type": "NUMERIC", "feature_group": "behavior"},
    {"entity_code": "click_frequency", "name": "点击频率", "data_type": "NUMERIC", "feature_group": "behavior"},
    {"entity_code": "page_depth", "name": "页面深度", "data_type": "NUMERIC", "feature_group": "behavior"},
    {"entity_code": "session_count", "name": "会话次数", "data_type": "NUMERIC", "feature_group": "behavior"},
    {"entity_code": "night_activity_ratio", "name": "夜间活跃占比", "data_type": "NUMERIC", "feature_group": "behavior"},
    {"entity_code": "login_fail_count", "name": "登录失败次数", "data_type": "NUMERIC", "feature_group": "behavior"},
    {"entity_code": "reg_to_apply_days", "name": "注册到申请天数", "data_type": "NUMERIC", "feature_group": "behavior"},
    # device/identity features
    {"entity_code": "ip_change_freq", "name": "IP变更频率", "data_type": "NUMERIC", "feature_group": "device"},
    {"entity_code": "gps_anomaly", "name": "GPS异常", "data_type": "NUMERIC", "feature_group": "device"},
    {"entity_code": "device_type", "name": "设备类型", "data_type": "CATEGORICAL", "feature_group": "device"},
    {"entity_code": "emulator_flag", "name": "模拟器标识", "data_type": "NUMERIC", "feature_group": "device"},
    # demographic features
    {"entity_code": "age", "name": "年龄", "data_type": "NUMERIC", "feature_group": "demographic"},
    {"entity_code": "income_level", "name": "收入水平", "data_type": "NUMERIC", "feature_group": "demographic"},
    {"entity_code": "consumption_level", "name": "消费水平", "data_type": "NUMERIC", "feature_group": "demographic"},
    {"entity_code": "education_level", "name": "教育程度", "data_type": "CATEGORICAL", "feature_group": "demographic"},
    {"entity_code": "job_stability", "name": "工作稳定性", "data_type": "NUMERIC", "feature_group": "demographic"},
    {"entity_code": "marital_status", "name": "婚姻状况", "data_type": "CATEGORICAL", "feature_group": "demographic"},
    {"entity_code": "gender", "name": "性别", "data_type": "CATEGORICAL", "feature_group": "demographic"},
    {"entity_code": "city_tier", "name": "城市等级", "data_type": "CATEGORICAL", "feature_group": "demographic"},
    # financial features
    {"entity_code": "debt_income_ratio", "name": "负债收入比", "data_type": "NUMERIC", "feature_group": "financial"},
    {"entity_code": "loan_amount_request", "name": "申请贷款金额", "data_type": "NUMERIC", "feature_group": "financial"},
    {"entity_code": "repayment_period", "name": "还款周期", "data_type": "CATEGORICAL", "feature_group": "financial"},
]

# §4.2 Metric — 可注册的指标定义（设计文档推荐目录）
METRICS = [
    # 模型性能
    {"entity_code": "AUC", "name": "AUC", "category": "model_performance"},
    {"entity_code": "KS", "name": "KS", "category": "model_performance"},
    {"entity_code": "PR_AUC", "name": "PR-AUC", "category": "model_performance"},
    {"entity_code": "BAD_RECALL", "name": "坏样本召回率", "category": "model_performance"},
    # 概率校准
    {"entity_code": "BRIER", "name": "Brier Score", "category": "calibration"},
    {"entity_code": "ECE", "name": "期望校准误差", "category": "calibration"},
    # 分布稳定
    {"entity_code": "SCORE_PSI", "name": "分数PSI", "category": "distribution"},
    {"entity_code": "FEATURE_PSI", "name": "特征PSI", "category": "distribution"},
    # 数据质量
    {"entity_code": "MISSING_RATE", "name": "缺失率", "category": "data_quality"},
    {"entity_code": "SCHEMA_CONSISTENCY", "name": "模式一致性", "category": "data_quality"},
    {"entity_code": "SAMPLE_SIZE", "name": "样本量", "category": "data_quality"},
    # 标签
    {"entity_code": "BAD_RATE", "name": "坏样本率", "category": "label"},
    {"entity_code": "PREDICTION_MEAN", "name": "预测均值", "category": "distribution"},
    # 退化
    {"entity_code": "PERFORMANCE_DROP_MAX", "name": "最大性能下降", "category": "model_performance"},
    {"entity_code": "MONITOR_STATUS", "name": "监控状态", "category": "monitoring"},
]

# §4.3 Alert — 告警类型（设计文档 §4.3，任务一可能触发的子集）
ALERTS = [
    {"entity_code": "AUC_DROP", "name": "AUC显著下降"},
    {"entity_code": "KS_DROP", "name": "KS显著下降"},
    {"entity_code": "PR_AUC_DROP", "name": "PR-AUC显著下降"},
    {"entity_code": "BAD_RECALL_DROP", "name": "坏样本召回下降"},
    {"entity_code": "CALIBRATION_DEGRADE", "name": "概率校准恶化"},
    {"entity_code": "HIGH_FEATURE_PSI", "name": "特征PSI漂移"},
    {"entity_code": "HIGH_SCORE_PSI", "name": "分数PSI漂移"},
    {"entity_code": "MISSING_RATE_SPIKE", "name": "缺失率异常"},
    {"entity_code": "SCHEMA_MISMATCH", "name": "模式不一致"},
    {"entity_code": "SAMPLE_SIZE_LOW", "name": "样本量不足"},
    {"entity_code": "BAD_RATE_SHIFT", "name": "坏样本率变化"},
    {"entity_code": "PERFORMANCE_DECAY", "name": "性能持续衰退"},
]

# ═══════════════════════════════════════════════════════════
# 关系定义（严格来自设计文档 §5）
# ═══════════════════════════════════════════════════════════

# 5.1 Feature → MONITORED_BY → Metric（无概率权重，纯配置）
#    所有特征通过所有指标监控（全连接）
MONITORED_BY_RELATIONS = []
for f in FEATURES:
    for m in METRICS:
        if m["entity_code"] in ("AUC", "KS", "PR_AUC", "BAD_RECALL",
                                 "BRIER", "ECE", "BAD_RATE",
                                 "PERFORMANCE_DROP_MAX", "MONITOR_STATUS",
                                 "PREDICTION_MEAN"):
            continue  # 这些是模型级指标，不对应具体特征
        MONITORED_BY_RELATIONS.append((f["entity_code"], m["entity_code"]))

# 5.2 Metric → TRIGGERS → Alert（确定性规则，weight=1.0）
TRIGGERS_RELATIONS = [
    # 模型性能 → 告警
    ("AUC", "AUC_DROP"),
    ("KS", "KS_DROP"),
    ("PR_AUC", "PR_AUC_DROP"),
    ("BAD_RECALL", "BAD_RECALL_DROP"),
    ("BRIER", "CALIBRATION_DEGRADE"),
    ("ECE", "CALIBRATION_DEGRADE"),
    # 分布稳定 → 告警
    ("FEATURE_PSI", "HIGH_FEATURE_PSI"),
    ("SCORE_PSI", "HIGH_SCORE_PSI"),
    # 数据质量 → 告警
    ("MISSING_RATE", "MISSING_RATE_SPIKE"),
    ("SCHEMA_CONSISTENCY", "SCHEMA_MISMATCH"),
    ("SAMPLE_SIZE", "SAMPLE_SIZE_LOW"),
    # 标签 → 告警
    ("BAD_RATE", "BAD_RATE_SHIFT"),
    # 性能退化
    ("PERFORMANCE_DROP_MAX", "PERFORMANCE_DECAY"),
]

WEIGHT_VERSION = "kg_weight_v1"


async def seed() -> int:
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    entity_count = 0
    rel_count = 0

    async with driver.session(database="neo4j") as session:
        # ── ① 创建 Feature ──
        for ent in FEATURES:
            await session.run("""
                MERGE (n:Feature {entity_code: $entity_code})
                SET n.name = $name,
                    n.data_type = $data_type,
                    n.feature_group = $feature_group,
                    n.namespace = 'MONITORING',
                    n.is_core = true,
                    n.enabled = true,
                    n.schema_version = 'FEATURE_SCHEMA_V1'
            """, **ent)
            entity_count += 1

        # ── ② 创建 Metric ──
        for ent in METRICS:
            await session.run("""
                MERGE (n:Metric {entity_code: $entity_code})
                SET n.name = $name,
                    n.category = $category,
                    n.namespace = 'MONITORING',
                    n.is_core = true,
                    n.enabled = true
            """, **ent)
            entity_count += 1

        # ── ③ 创建 Alert ──
        for ent in ALERTS:
            await session.run("""
                MERGE (n:Alert {entity_code: $entity_code})
                SET n.name = $name,
                    n.namespace = 'MONITORING',
                    n.is_core = true,
                    n.enabled = true
            """, **ent)
            entity_count += 1

        # ── ④ MONITORED_BY: Feature → Metric ──
        for feat_code, metric_code in MONITORED_BY_RELATIONS:
            rkey = f"{feat_code}|MONITORED_BY|{metric_code}"
            await session.run("""
                MATCH (f:Feature {entity_code: $feat_code})
                MATCH (m:Metric {entity_code: $metric_code})
                MERGE (f)-[r:MONITORED_BY]->(m)
                SET r.relation_key = $rkey,
                    r.relation_type = 'MONITORED_BY',
                    r.enabled = true
            """, feat_code=feat_code, metric_code=metric_code, rkey=rkey)
            rel_count += 1

        # ── ⑤ TRIGGERS: Metric → Alert ──
        for metric_code, alert_code in TRIGGERS_RELATIONS:
            rkey = f"{metric_code}|TRIGGERS|{alert_code}"
            await session.run("""
                MATCH (m:Metric {entity_code: $metric_code})
                MATCH (a:Alert {entity_code: $alert_code})
                MERGE (m)-[r:TRIGGERS]->(a)
                SET r.relation_key = $rkey,
                    r.relation_type = 'TRIGGERS',
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
                    r.weight_version = $wv,
                    r.enabled = true
            """, metric_code=metric_code, alert_code=alert_code, rkey=rkey, wv=WEIGHT_VERSION)
            rel_count += 1

    await driver.close()

    print(
        f"KG Seed V2 完成:\n"
        f"  {len(FEATURES)} Feature + {len(METRICS)} Metric + {len(ALERTS)} Alert "
        f"= {entity_count} 节点\n"
        f"  {len(MONITORED_BY_RELATIONS)} MONITORED_BY + {len(TRIGGERS_RELATIONS)} TRIGGERS "
        f"= {rel_count} 条关系\n"
        f"  符合《模型风险诊断知识图谱设计整理_V1.1》§4 + §5"
    )
    return entity_count


if __name__ == "__main__":
    asyncio.run(seed())
    sys.exit(0)
