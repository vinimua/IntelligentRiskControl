"""测试 IterationDecision → TrainingPlan → Qualification → Deployment 链路。
不动任何项目代码，只构造正确 state 调节点函数。
"""
import asyncio, sys, uuid
sys.path.insert(0, '.')
from sqlalchemy import text
from apps.modelops_api.database import async_session
from apps.modelops_api.services.workflow.graph import (
    iteration_decision_node, training_plan_node,
    qualification_node, deployment_gate_node, event_close_node,
)
from packages.models.workflow.lifecycle_state import ModelLifecycleState
from packages.models.common.enums import LifecyclePhase, TriggerType


async def main():
    lid = str(uuid.uuid4())
    state = ModelLifecycleState(
        lifecycle_run_id=lid, model_id='credit_model_001',
        champion_version='challenger_v1', trigger_type=TriggerType.MANUAL_TRIGGER,
        current_phase=LifecyclePhase.DIAGNOSIS_COMPLETED,
        diagnosis_run_id='685204b5-2d46-4f52-83c2-2458d9b8f147',
        monitoring_run_id='f8c5e971-c8f8-4e85-aa49-5608d5fd0808',
        primary_root_cause_code='FEATURE_DRIFT',
        primary_root_cause_dimension='FEATURE',
        primary_root_cause_score=0.82,
        recommended_action='MODEL_ITERATION',
        need_iteration=True, event_id=str(uuid.uuid4()), business_round=1,
    )

    async with async_session() as s:
        await s.execute(text(
            "INSERT INTO workflow.model_lifecycle_runs"
            " (lifecycle_run_id,model_id,champion_version,current_phase)"
            " VALUES (:id,:mid,:cv,:ph)"
        ), {'id': lid, 'mid': state.model_id, 'cv': state.champion_version,
             'ph': 'DIAGNOSIS_COMPLETED'})
        await s.commit()

    # 1 ── IterationDecisionNode ──
    r = await iteration_decision_node(state)
    pid = r.get('decision_proposal_id')
    print(f"[1] IterationDecision: proposal={pid}, "
          f"strategy={r.get('selected_strategy_code')}, "
          f"training_mode={r.get('training_mode')}, "
          f"phase={r.get('current_phase')}")
    if pid is None:
        print(f"    FAILED: {r.get('last_error', r.get('warnings'))}")
        return
    for k, v in r.items():
        if v is not None and hasattr(state, k):
            setattr(state, k, v)

    state.manual_review_id = str(uuid.uuid4())
    state.requires_manual_review = False
    # 写入人工复核通过记录（TrainingPlanNode 的前置条件）
    async with async_session() as s:
        await s.execute(text(
            "INSERT INTO iteration.manual_review_reports"
            " (review_id, proposal_id, decision, reviewer_id, reason, report_json, reviewed_at)"
            " VALUES (:rid, :pid, 'APPROVE', 'test', 'test approval', CAST('{}' AS JSONB), NOW())"
        ), {'rid': state.manual_review_id, 'pid': pid})
        await s.commit()

    # 2 ── TrainingPlanNode ──
    r = await training_plan_node(state)
    tp_id = r.get('training_plan_id')
    print(f"[2] TrainingPlan: plan={tp_id}, "
          f"experiment={r.get('experiment_id')}, "
          f"iteration={r.get('iteration_run_id')}, "
          f"round={r.get('business_round')}, "
          f"phase={r.get('current_phase')}")
    if r.get('current_phase') != LifecyclePhase.ITERATING.value:
        print(f"    STOP: {r.get('last_error', {}).get('reason', '')}")
        return
    for k, v in r.items():
        if v is not None and hasattr(state, k):
            setattr(state, k, v)

    # 3 ── QualificationNode ──
    r = await qualification_node(state)
    qualified = r.get('challenger_qualified')
    print(f"[3] Qualification: qualified={qualified}, "
          f"challenger={r.get('challenger_version')}, "
          f"phase={r.get('current_phase')}")
    if r.get('current_phase') == LifecyclePhase.FAILED.value:
        print(f"    exit_reason: {r.get('iteration_exit_reason')}")
    for k, v in r.items():
        if v is not None and hasattr(state, k):
            setattr(state, k, v)

    if not qualified:
        print("[4] Deployment: SKIPPED (challenger not qualified)")
        return

    # 4 ── DeploymentGateNode ──
    r = await deployment_gate_node(state)
    print(f"[4] DeploymentGate: id={r.get('deployment_id')}, "
          f"stage={r.get('deployment_stage')}, "
          f"decision={r.get('deployment_decision')}, "
          f"phase={r.get('current_phase')}")

    # 5 ── EventCloseNode ──
    r = await event_close_node(state)
    print(f"[5] EventClose: phase={r.get('current_phase')}")

    print("=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    asyncio.run(main())
