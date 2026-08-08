"""完整端到端演示脚本：从监控 → 诊断 → 迭代 → 部署。

用法: python scripts/demo_full_pipeline.py
前置: PostgreSQL、Neo4j、Redis 运行中
"""

import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# 项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apps.modelops_api.config import settings
from apps.modelops_api.database import async_session
from apps.modelops_api.neo4j_db import get_neo4j_driver
from apps.modelops_api.services.knowledge_service import KnowledgeService
from apps.modelops_api.services.monitoring.monitoring_service import MonitoringService
from apps.modelops_api.services.diagnosis.diagnosis_service import DiagnosisService
from apps.modelops_api.services.iteration import (
    RepairDecisionService,
    RiskAssessmentService,
    TrainingPlanBuilder,
    QualificationService,
)
from apps.modelops_api.repositories.iteration_repo import IterationRepo
from apps.modelops_api.repositories.monitoring_repo import MonitoringRepo
from apps.modelops_api.repositories.diagnosis_repo import DiagnosisRepo
from apps.modelops_api.services.iteration.config_loader import load_iteration_config
from apps.modelops_api.services.workflow.graph import (
    _g, _now_iso,
    monitoring_node, diagnosis_node, diagnosis_handoff_node,
    agent_decision_node, iteration_decision_node, training_plan_node,
    qualification_node, deployment_gate_node, event_close_node,
)
from packages.models.common.enums import LifecyclePhase, TriggerType, AgentDecisionAction
from packages.models.workflow.lifecycle_state import ModelLifecycleState


def make_state(**overrides) -> ModelLifecycleState:
    """创建初始 State。"""
    defaults = {
        "lifecycle_run_id": str(uuid.uuid4()),
        "model_id": "credit_model_001",
        "champion_version": "challenger_v1",
        "trigger_type": TriggerType.MANUAL_TRIGGER,
        "current_phase": LifecyclePhase.CREATED,
    }
    defaults.update(overrides)
    return ModelLifecycleState(**defaults)


def make_degraded_data(n_samples: int = 5000, seed: int = 42):
    """生成人造退化数据：特征漂移（device_risk_score PSI=0.35）+ AUC 从 0.75 跌到 0.64。

    设计要点：
    - 标签分布一致（好坏比不变） → 避免被判定为 label_distribution_shift
    - device_risk_score 显著漂移（PSI≈0.35） → 触发 FEATURE_DRIFT
    - AUC 下降 0.11 → 触发 AUC_DROP 告警
    """
    rng = np.random.default_rng(seed)
    N = 2 * n_samples

    # --- baseline (W0) ---
    b_device = rng.normal(0.50, 0.12, N)
    # 用 device_risk_score 生成预测分（让它和标签强相关）
    b_y_true = rng.binomial(1, 0.3, N).astype(float)
    b_noise = rng.normal(0, 0.08, N)
    b_scores = np.clip(b_y_true * 0.40 + b_device * 0.55 + b_noise + 0.05, 0.001, 0.999)

    # --- current (W3) ---
    # device_risk_score 漂移：均值从 0.50 → 0.22，标准差从 0.12 → 0.28（PSI≈0.35）
    c_device = rng.normal(0.22, 0.28, N)
    # 标签分布一致（好坏比不变）
    c_y_true = rng.binomial(1, 0.3, N).astype(float)
    # 模型仍按旧的 device_risk_score 权重打分 → AUC 下降
    c_noise = rng.normal(0, 0.10, N)
    c_scores = np.clip(c_y_true * 0.40 + c_device * 0.35 + c_noise + 0.08, 0.001, 0.999)

    def make_df(scores, y_true, device, prefix, start_date):
        df = pd.DataFrame({
            "y_pred_proba": scores,
            "y_true": y_true,
            "device_risk_score": device,
            "income_score": rng.normal(0.50, 0.15, N),
            "age_score": rng.normal(0.60, 0.10, N),
            "credit_history_score": rng.normal(0.45, 0.18, N),
            "apply_time": pd.date_range(start_date, periods=N, freq="h"),
        })
        df["sample_id"] = [f"{prefix}_{i:06d}" for i in range(N)]
        return df

    baseline = make_df(b_scores, b_y_true, b_device, "W0", "2025-01-01")
    current = make_df(c_scores, c_y_true, c_device, "W3", "2025-12-01")

    return baseline, current


async def demo_full_pipeline():
    print("=" * 60)
    print("信贷风控模型智能监测与自主迭代 — 全链路演示")
    print("=" * 60)

    state = make_state()
    print(f"\n{'▶' * 3} 启动生命周期: {state.lifecycle_run_id}")
    print(f"   模型: {state.model_id}, Champion: {state.champion_version}")

    # ── 任务一：监控 ──
    print(f"\n{'─' * 40}")
    print("任务一：监控 (MonitoringNode)")
    print(f"{'─' * 40}")

    baseline_df, current_df = make_degraded_data()

    async with async_session() as session:
        driver = await get_neo4j_driver()
        knowledge = KnowledgeService(driver)
        service = MonitoringService(session, knowledge)

        result = await service.run(
            model_id=state.model_id,
            champion_version=state.champion_version,
            baseline_data=baseline_df.to_dict(orient="records"),
            current_data=current_df.to_dict(orient="records"),
            baseline_window_id="W0",
            current_window_id="W3",
        )

        await session.commit()

    print(f"   monitoring_run_id: {result.monitoring_run_id}")
    print(f"   has_alerts: {result.has_alerts}")
    print(f"   alert_count: {result.alert_count}")
    print(f"   max_alert_severity: {result.max_alert_severity.value if result.max_alert_severity else 'None'}")
    if result.alerts:
        for a in result.alerts:
            print(f"   alert: code={a.get('alert_code') if isinstance(a, dict) else getattr(a, 'alert_code', '?')} "
                  f"severity={a.get('severity') if isinstance(a, dict) else getattr(a, 'severity', '?')} "
                  f"metric={a.get('metric_code') if isinstance(a, dict) else getattr(a, 'metric_code', '?')}")

    state.current_phase = LifecyclePhase.MONITORING_COMPLETED if result.has_alerts else LifecyclePhase.NO_ALERT

    if not result.has_alerts:
        print("\n   ⚠ 无告警 → 监控通过，模型稳定，无需后续流程")
        return

    # ── 任务二：诊断 ──
    print(f"\n{'─' * 40}")
    print("任务二：诊断 (DiagnosisNode)")
    print(f"{'─' * 40}")

    # 构造 state 含监控结果
    state = make_state(
        lifecycle_run_id=state.lifecycle_run_id,
        monitoring_run_id=result.monitoring_run_id,
        has_alerts=result.has_alerts,
        alert_count=result.alert_count,
        max_alert_severity=result.max_alert_severity.value if result.max_alert_severity else None,
        current_phase=LifecyclePhase.MONITORING_COMPLETED,
    )

    try:
        diag_result = await diagnosis_node(state)
        print(f"   primary_root_cause_code: {diag_result.get('primary_root_cause_code')}")
        print(f"   primary_root_cause_score: {diag_result.get('primary_root_cause_score')}")
        print(f"   recommended_action: {diag_result.get('recommended_action')}")
        print(f"   need_iteration: {diag_result.get('need_iteration')}")
        print(f"   diagnosis_run_id: {diag_result.get('diagnosis_run_id')}")

        if diag_result.get("current_phase") == LifecyclePhase.FAILED.value:
            print(f"   ❌ 诊断失败: {diag_result.get('last_error', {}).get('reason')}")
            return
    except Exception as e:
        print(f"   ❌ 诊断异常: {e}")
        import traceback; traceback.print_exc()
        return

    if not diag_result.get("need_iteration"):
        print("\n   ⚠ 不需要迭代 → 进入持续观察")
        return

    # 合并诊断结果到 state
    for k, v in diag_result.items():
        if v is not None and hasattr(state, k):
            setattr(state, k, v)

    # ── Handoff → Agent → IterationDecision ──
    print(f"\n{'─' * 40}")
    print("任务二续：Handoff → Agent → IterationDecision")
    print(f"{'─' * 40}")

    try:
        handoff = await diagnosis_handoff_node(state)
        print(f"   agent_handoff_status: {handoff.get('agent_handoff_status')}")
        for k, v in handoff.items():
            if v is not None and hasattr(state, k):
                setattr(state, k, v)

        agent = await agent_decision_node(state)
        print(f"   agent_decision_id: {agent.get('agent_decision_id')}")
        print(f"   agent_confidence: {agent.get('agent_confidence')}")
        print(f"   recommended_action: {agent.get('recommended_action')}")
        for k, v in agent.items():
            if v is not None and hasattr(state, k):
                setattr(state, k, v)

        decision = await iteration_decision_node(state)
        print(f"   decision_proposal_id: {decision.get('decision_proposal_id')}")
        print(f"   selected_strategy_code: {decision.get('selected_strategy_code')}")
        print(f"   strategy_tier: {decision.get('strategy_tier')}")
        for k, v in decision.items():
            if v is not None and hasattr(state, k):
                setattr(state, k, v)
    except Exception as e:
        print(f"   ❌ 决策链路异常: {e}")
        import traceback; traceback.print_exc()
        return

    if decision.get("requires_manual_review"):
        print("\n   ⚠ 需要人工复核 → 暂停等待审批")
        print("   (演示模式下跳过人工复核)")

    # ── 任务三：训练计划 ──
    print(f"\n{'─' * 40}")
    print("任务三：训练计划 (TrainingPlanNode)")
    print(f"{'─' * 40}")

    state.manual_review_id = f"demo-review-{str(uuid.uuid4())[:8]}"

    try:
        plan = await training_plan_node(state)
        print(f"   training_plan_id: {plan.get('training_plan_id')}")
        print(f"   experiment_id: {plan.get('experiment_id')}")
        print(f"   iteration_run_id: {plan.get('iteration_run_id')}")
        print(f"   business_round: {plan.get('business_round')}")

        if plan.get("current_phase") == LifecyclePhase.FAILED.value:
            print(f"   ❌ 训练计划失败: {plan.get('last_error', {}).get('reason')}")
            return

        for k, v in plan.items():
            if v is not None and hasattr(state, k):
                setattr(state, k, v)
    except Exception as e:
        print(f"   ❌ 训练计划异常: {e}")
        import traceback; traceback.print_exc()
        return

    # ── 资格验证 ──
    print(f"\n{'─' * 40}")
    print("任务三：资格验证 (QualificationNode)")
    print(f"{'─' * 40}")

    try:
        qual_result = await qualification_node(state)
        print(f"   qualification_run_id: {qual_result.get('qualification_run_id')}")
        print(f"   challenger_qualified: {qual_result.get('challenger_qualified')}")
        print(f"   challenger_version: {qual_result.get('challenger_version')}")

        for k, v in qual_result.items():
            if v is not None and hasattr(state, k):
                setattr(state, k, v)
    except Exception as e:
        print(f"   ❌ 资格验证异常: {e}")
        import traceback; traceback.print_exc()
        return

    if not qual_result.get("challenger_qualified"):
        print("\n   ⚠ Challenger 不合格 → 进入失败分析")
        business_round = state.business_round or 1
        max_rounds = load_iteration_config().iteration.max_iteration_rounds
        print(f"   business_round={business_round}, max_rounds={max_rounds}")
        if business_round < max_rounds:
            print("   → 进入下一轮迭代")
        else:
            print("   → 达到最大轮次，停止自动迭代")
        return

    # ── 任务四：部署 ──
    print(f"\n{'─' * 40}")
    print("任务四：部署 (DeploymentGateNode)")
    print(f"{'─' * 40}")

    try:
        deploy_result = await deployment_gate_node(state)
        print(f"   deployment_id: {deploy_result.get('deployment_id')}")
        print(f"   deployment_stage: {deploy_result.get('deployment_stage')}")
        print(f"   deployment_decision: {deploy_result.get('deployment_decision')}")
        print(f"   current_phase: {deploy_result.get('current_phase')}")

        for k, v in deploy_result.items():
            if v is not None and hasattr(state, k):
                setattr(state, k, v)
    except Exception as e:
        print(f"   ❌ 部署异常: {e}")
        import traceback; traceback.print_exc()
        return

    # ── 事件关闭 ──
    print(f"\n{'─' * 40}")
    print("任务四：事件关闭 (EventCloseNode)")
    print(f"{'─' * 40}")

    try:
        close = await event_close_node(state)
        print(f"   current_phase: {close.get('current_phase')}")
    except Exception as e:
        print(f"   ❌ 关闭异常: {e}")
        import traceback; traceback.print_exc()
        return

    # ── 总结 ──
    print(f"\n{'=' * 60}")
    print("全链路完成!")
    print(f"{'=' * 60}")
    print(f"""
    lifecycle_run_id:    {state.lifecycle_run_id}
    monitoring_run_id:   {state.monitoring_run_id}
    diagnosis_run_id:    {state.diagnosis_run_id}
    root_cause:          {state.primary_root_cause_code} (score={state.primary_root_cause_score})
    strategy:            {state.decision_proposal_id}
    iteration_run_id:    {state.iteration_run_id}
    challenger:          {state.challenger_version}
    qualified:           {state.challenger_qualified}
    deployment_id:       {state.deployment_id}
    deployment_stage:    {state.deployment_stage}
    deployment_decision: {state.deployment_decision}
    final_phase:         {state.current_phase}
    """)


if __name__ == "__main__":
    asyncio.run(demo_full_pipeline())
