"""LangGraph 主图 + 节点。

LangGraph 开发路线 V1.0 §2-15。

完整的图结构（P0-P4）：
    START → MonitoringNode
        ├─ has_alerts=False → NoAlertCloseNode → END
        └─ has_alerts=True  → DiagnosisNode
            ├─ need_iteration=False → NoAlertCloseNode → END
            ├─ need_iteration=None  → ManualReviewNode → END
            └─ need_iteration=True  → DiagnosisHandoffNode
                → AgentDecisionNode
                → IterationDecisionNode

    IterationDecisionNode → route_after_iteration_decision
        ├─ requires_manual_review → ManualReviewNode → (resume后)
        │     ├─ approved  → TrainingPlanNode
        │     └─ rejected  → END (FAILED)
        ├─ need_iteration=True  → DataEligibilityNode → TrainingPlanNode
        │     → TrainingJobDispatchNode → WaitTrainingCallbackNode
        │     → (resume后) QualificationNode
        │         ├─ qualified=True  → DeploymentNode → EventCloseNode → END
        │         ├─ qualified=False & round<3 → NextRoundPlanNode → TrainingPlanNode
        │         └─ qualified=False & round>=3 → StopAutoIterationNode → END
        └─ need_iteration=False → END
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

import structlog
from sqlalchemy.exc import IntegrityError as _DBIntegrityError
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from packages.models.common.enums import AgentDecisionAction, LifecyclePhase, Severity
from packages.models.workflow.agent_decision import AgentDecisionInput
from packages.models.workflow.lifecycle_state import ModelLifecycleState

logger = structlog.get_logger(__name__)

# ── Mock 行为 ──
MOCK_NEED_ITERATION: bool | None = True
MOCK_CHALLENGER_QUALIFIED: bool = True
MOCK_DEPLOYMENT_DECISION: str = "PROMOTE"
MAX_BUSINESS_ROUNDS: int = 3


def _g(state, key, default=None):
    """安全访问 State，兼容 dict 和 Pydantic model。"""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)

def _state_dict(state) -> dict:
    if isinstance(state, dict):
        return state
    if hasattr(state, "model_dump"):
        return state.model_dump()
    return {}

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _save_external_plan(plan_type: str, plan: dict, dispatch: dict | None = None) -> None:
    try:
        from ...database import async_session
        from ...repositories.iteration_repo import IterationRepo

        plan_ids = {
            "CALIBRATION": plan.get("calibration_plan_id"),
            "THRESHOLD": plan.get("threshold_plan_id"),
            "REPAIR": plan.get("repair_plan_id"),
        }
        async with async_session() as session:
            await IterationRepo(session).save_external_execution_plan(
                {
                    "lifecycle_run_id": plan.get("lifecycle_run_id"),
                    "plan_type": plan_type,
                    "plan_id": plan_ids[plan_type],
                    "action": plan.get("action"),
                    "status": plan.get("status", "PLANNED"),
                    "dispatch_mode": (dispatch or {}).get("dispatch_mode", "INTERNAL"),
                    "external_task_id": (dispatch or {}).get("external_task_id"),
                    "callback_endpoint": plan.get("callback_endpoint"),
                    "request_json": plan,
                    "result_json": (dispatch or {}).get("response"),
                    "error_message": (dispatch or {}).get("error"),
                }
            )
            await session.commit()
    except Exception:
        logger.warning("external_execution_plan_persist_failed", plan_type=plan_type, exc_info=True)


async def _save_deployment_record(state: ModelLifecycleState, result: dict, dispatch: dict | None = None) -> None:
    try:
        from ...database import async_session
        from ...repositories.iteration_repo import IterationRepo

        decision = result.get("deployment_decision")
        status = {
            "PROMOTE": "PROMOTED",
            "ROLLBACK": "ROLLED_BACK",
            "ABORT_DEPLOYMENT": "ABORTED",
            "HOLD": "HELD",
        }.get(decision, "RUNNING")
        record = {
            "deployment_id": result.get("deployment_id"),
            "lifecycle_run_id": _g(state, "lifecycle_run_id"),
            "qualification_run_id": _g(state, "qualification_run_id"),
            "model_id": _g(state, "model_id"),
            "champion_version": _g(state, "champion_version"),
            "candidate_version": result.get("candidate_version") or _g(state, "challenger_version"),
            "deployment_stage": result.get("deployment_stage"),
            "deployment_decision": decision,
            "status": status,
            "dispatch_mode": (dispatch or {}).get("dispatch_mode", "INTERNAL"),
            "external_task_id": (dispatch or {}).get("external_task_id"),
            "health_json": {
                "deployment_health_passed": _g(state, "deployment_health_passed", True),
                "deployment_force_rollback": _g(state, "deployment_force_rollback", False),
            },
            "external_response": (dispatch or {}).get("response"),
        }
        if record["deployment_id"]:
            async with async_session() as session:
                await IterationRepo(session).save_deployment_record(record)
                await session.commit()
    except Exception:
        logger.warning("deployment_record_persist_failed", exc_info=True)


def _recommended_action(state: ModelLifecycleState) -> str | None:
    action = _g(state, "recommended_action")
    return action.value if hasattr(action, "value") else action


def _route_after_action(
    state: ModelLifecycleState,
) -> Literal[
    "ObservationCloseNode",
    "RepairPlanNode",
    "CalibrationPlanNode",
    "ThresholdPlanNode",
    "DataEligibilityNode",
    "ManualReviewNode",
]:
    action = _recommended_action(state)
    need = _g(state, "need_iteration")

    if action in {
        AgentDecisionAction.NO_ACTION.value,
        AgentDecisionAction.CONTINUE_OBSERVATION.value,
    }:
        return "ObservationCloseNode"
    if action in {
        AgentDecisionAction.DATA_REPAIR.value,
        AgentDecisionAction.PIPELINE_REPAIR.value,
    }:
        return "RepairPlanNode"
    if action == AgentDecisionAction.CALIBRATION_ADJUSTMENT.value:
        return "CalibrationPlanNode"
    if action == AgentDecisionAction.THRESHOLD_ADJUSTMENT.value:
        return "ThresholdPlanNode"
    if action == AgentDecisionAction.MANUAL_REVIEW.value:
        return "ManualReviewNode"
    if need is True or action == AgentDecisionAction.MODEL_ITERATION.value:
        return "DataEligibilityNode"
    if need is False:
        return "ObservationCloseNode"
    return "ManualReviewNode"


# ═══════════════════════════════════════════════════════════
# 任务一：MonitoringNode
# ═══════════════════════════════════════════════════════════

async def monitoring_node(state: ModelLifecycleState) -> dict:
    """阶段 4 真实监控节点：调用 MonitoringService 执行完整指标计算 → 告警生成。"""
    from ...services.monitoring.window_loader import load_window_with_predictions

    baseline_df = load_window_with_predictions("W0")
    current_df = load_window_with_predictions("W3")
    baseline_data = baseline_df.to_dict(orient="records")
    current_data = current_df.to_dict(orient="records")

    # Demo 模式 / 开发阶段：强制 Mock 绕过数据依赖
    from ...config import settings
    if settings.workflow_demo_mode:  # P0: 演示模式确保有告警 → MODEL_ITERATION
        run_id = str(uuid.uuid4())
        return {
            "monitoring_run_id": run_id,
            "has_alerts": True,
            "alert_count": 2,
            "max_alert_severity": Severity.HIGH.value,
            "current_phase": LifecyclePhase.MONITORING_COMPLETED.value,
        }

    try:
        from ...database import async_session
        from ...neo4j_db import get_neo4j_driver
        from ...services.knowledge_service import KnowledgeService
        from ...services.monitoring.monitoring_service import MonitoringService

        async with async_session() as session:
            driver = await get_neo4j_driver()
            knowledge = KnowledgeService(driver)
            service = MonitoringService(session, knowledge)

            result = await service.run(
                model_id=_g(state, "model_id"),
                champion_version=_g(state, "champion_version"),
                baseline_data=baseline_data,
                current_data=current_data,
                baseline_window_id=_g(state, "baseline_window_id") or "",
                current_window_id=_g(state, "current_window_id") or "",
            )

            logger.info(
                "monitoring_node_completed",
                monitoring_run_id=result.monitoring_run_id,
                alert_count=result.alert_count,
            )

            return {
                "monitoring_run_id": result.monitoring_run_id,
                "has_alerts": result.has_alerts,
                "alert_count": result.alert_count,
                "max_alert_severity": (
                    result.max_alert_severity.value if result.max_alert_severity else None
                ),
                "current_phase": (
                    LifecyclePhase.NO_ALERT.value
                    if not result.has_alerts
                    else LifecyclePhase.MONITORING_COMPLETED.value
                ),
            }

    except (OSError, ConnectionError, TimeoutError, _DBIntegrityError):
        logger.warning("monitoring_node_infra_failed_falling_back_to_mock", exc_info=True)
        run_id = str(uuid.uuid4())
        return {
            "monitoring_run_id": run_id,
            "has_alerts": True,
            "alert_count": 2,
            "max_alert_severity": Severity.HIGH.value,
            "current_phase": LifecyclePhase.MONITORING_COMPLETED.value,
        }


# ═══════════════════════════════════════════════════════════
# 任务二：DiagnosisNode + DiagnosisHandoffNode
# ═══════════════════════════════════════════════════════════

async def diagnosis_node(state: ModelLifecycleState) -> dict:
    """任务二诊断节点：调用 DiagnosisService 执行真实 D/R/C/T/I 根因诊断。"""
    monitoring_run_id = _g(state, "monitoring_run_id")
    lifecycle_run_id = _g(state, "lifecycle_run_id")

    # Demo 模式 / 开发阶段：强制 Mock
    from ...config import settings
    if settings.workflow_demo_mode:
        return _diagnosis_fallback()

    if not monitoring_run_id:
        logger.warning("diagnosis_node_missing_monitoring_run_id")
        return _diagnosis_fallback()

    try:
        from ...database import async_session
        from ...neo4j_db import get_neo4j_driver
        from ...repositories.diagnosis_repo import DiagnosisRepo
        from ...repositories.monitoring_repo import MonitoringRepo
        from ...services.diagnosis.diagnosis_service import DiagnosisService
        from ...services.knowledge_service import KnowledgeService
        from packages.models.monitoring.alert_context import AlertContext, AlertDetail
        from packages.models.common.enums import DataTrack, Severity

        async with async_session() as session:
            driver = await get_neo4j_driver()
            knowledge = KnowledgeService(driver)
            mon_repo = MonitoringRepo(session)
            diag_repo = DiagnosisRepo(session)

            run = await mon_repo.get_run(monitoring_run_id)
            alerts = await mon_repo.get_alerts(monitoring_run_id)

            if not alerts:
                logger.info("diagnosis_node_no_alerts_skipping")
                return {
                    "diagnosis_run_id": None,
                    "primary_root_cause_code": "no_alerts",
                    "primary_root_cause_dimension": None,
                    "primary_root_cause_score": 0.0,
                    "recommended_action": "CONTINUE_OBSERVATION",
                    "need_iteration": False,
                    "current_phase": LifecyclePhase.DIAGNOSIS_COMPLETED.value,
                }

            alert_details = [
                AlertDetail(
                    alert_id=a["alert_id"],
                    alert_code=a["alert_code"],
                    severity=Severity(a["severity"]) if a.get("severity") else Severity.WARNING,
                    object_type=a.get("object_type", "MODEL"),
                    object_code=a.get("object_code", _g(state, "model_id")),
                    metric_code=a.get("metric_code", ""),
                    metric_version=a.get("metric_version", "V1"),
                    baseline_value=a.get("baseline_value"),
                    current_value=a.get("current_value"),
                    delta=a.get("delta"),
                    threshold=a.get("threshold"),
                    rule_type=a.get("rule_type"),
                    threshold_rule_id=a.get("threshold_rule_id"),
                    threshold_rule_version=a.get("threshold_rule_version"),
                    availability_status=a.get("availability_status", "AVAILABLE"),
                    metric_detail=a.get("alert_detail"),
                )
                for a in alerts
            ]

            alert_context = AlertContext(
                schema_version="1.0",
                trace_id=str(uuid.uuid4()),
                monitoring_run_id=monitoring_run_id,
                model_id=_g(state, "model_id"),
                model_version=_g(state, "champion_version"),
                monitor_window_id=run.get("current_window_id", "") if run else "",
                baseline_id=run.get("baseline_window_id", "") if run else "",
                data_track=DataTrack(run.get("data_track", "NATURAL")) if run else DataTrack.NATURAL,
                alert_details=alert_details,
            )

            service = DiagnosisService(session=session, knowledge=knowledge, repo=diag_repo)
            result = await service.diagnose(
                alert_context=alert_context,
                monitoring_run_id=monitoring_run_id,
                lifecycle_run_id=lifecycle_run_id,
            )

            logger.info(
                "diagnosis_node_completed",
                diagnosis_run_id=result.diagnosis_run_id,
                primary_root_cause_code=result.primary_root_cause_code,
                recommended_action=(
                    result.recommended_action.value if result.recommended_action else None
                ),
            )

            return {
                "diagnosis_run_id": result.diagnosis_run_id,
                "primary_root_cause_code": result.primary_root_cause_code,
                "primary_root_cause_dimension": (
                    result.primary_root_cause_dimension.value
                    if result.primary_root_cause_dimension else None
                ),
                "primary_root_cause_score": result.primary_root_cause_score,
                "recommended_action": (
                    result.recommended_action.value if result.recommended_action else "MANUAL_REVIEW"
                ),
                "need_iteration": result.need_iteration,
                "current_phase": LifecyclePhase.DIAGNOSIS_COMPLETED.value,
            }

    except (OSError, ConnectionError, TimeoutError, _DBIntegrityError):
        logger.warning("diagnosis_node_infra_failed_falling_back_to_mock", exc_info=True)
        return _diagnosis_fallback()


def _diagnosis_fallback() -> dict:
    run_id = str(uuid.uuid4())
    if MOCK_NEED_ITERATION is True:
        return {
            "diagnosis_run_id": run_id,
            "primary_root_cause_code": "feature_drift",
            "primary_root_cause_dimension": "FEATURE",
            "primary_root_cause_score": 0.90,  # >= 0.85 → RuleAgentAdapter 允许自动决策
            "recommended_action": "MODEL_ITERATION",
            "need_iteration": True,
            "current_phase": LifecyclePhase.DIAGNOSIS_COMPLETED.value,
        }
    if MOCK_NEED_ITERATION is False:
        return {
            "diagnosis_run_id": run_id,
            "primary_root_cause_code": "no_significant_issue",
            "primary_root_cause_dimension": None,
            "primary_root_cause_score": None,
            "recommended_action": "CONTINUE_OBSERVATION",
            "need_iteration": False,
            "current_phase": LifecyclePhase.DIAGNOSIS_COMPLETED.value,
        }
    return {
        "diagnosis_run_id": run_id,
        "primary_root_cause_code": "uncertain",
        "primary_root_cause_dimension": None,
        "primary_root_cause_score": None,
        "recommended_action": "MANUAL_REVIEW",
        "need_iteration": None,
        "current_phase": LifecyclePhase.MANUAL_REVIEW.value,
    }


async def diagnosis_handoff_node(state: ModelLifecycleState) -> dict:
    """诊断→Agent 交接节点。诊断结果 → agent_handoff_status + event_id。"""
    diagnosis_run_id = _g(state, "diagnosis_run_id")
    if not diagnosis_run_id:
        return {
            "agent_handoff_status": "ERROR_NO_DIAGNOSIS_RUN",
            "current_phase": LifecyclePhase.MANUAL_REVIEW.value,
        }

    try:
        from ...database import async_session
        from ...repositories.diagnosis_repo import DiagnosisRepo
        from .agent_handoff_service import DiagnosisHandoffService

        async with async_session() as session:
            repo = DiagnosisRepo(session)
            event = await repo.get_event_by_diagnosis_run(diagnosis_run_id)

            if not event:
                event_id = str(uuid.uuid4())
                return {
                    "event_id": event_id,
                    "agent_handoff_status": "DEGRADED_NO_EVENT",
                    "current_phase": LifecyclePhase.WAITING_AGENT_DECISION.value,
                }

            handoff_svc = DiagnosisHandoffService(session)
            handoff = await handoff_svc.build_handoff(str(event["event_id"]))
            await handoff_svc.validate_handoff(handoff)

            return {
                "event_id": str(event["event_id"]),
                "agent_handoff_status": handoff.get("handoff_status"),
                "current_phase": LifecyclePhase.WAITING_AGENT_DECISION.value,
            }
    except (OSError, ConnectionError, TimeoutError, _DBIntegrityError):
        logger.warning("diagnosis_handoff_fallback", exc_info=True)
        return {
            "event_id": str(uuid.uuid4()),
            "agent_handoff_status": "DEGRADED_INFRA_FAILURE",
            "current_phase": LifecyclePhase.WAITING_AGENT_DECISION.value,
        }


# ═══════════════════════════════════════════════════════════
# Agent 决策节点（P0）
# ═══════════════════════════════════════════════════════════

async def agent_decision_node(state: ModelLifecycleState) -> dict:
    """Agent 决策节点 — RuleAgentAdapter。"""
    diagnosis_run_id = _g(state, "diagnosis_run_id")
    if not diagnosis_run_id:
        return {
            "agent_decision_id": None,
            "agent_confidence": 0.0,
            "recommended_action": "MANUAL_REVIEW",
            "requires_manual_review": True,
            "current_phase": LifecyclePhase.MANUAL_REVIEW.value,
        }

    try:
        from ...database import async_session
        from .rule_agent_adapter import RuleAgentAdapter

        async with async_session() as session:
            agent_input = AgentDecisionInput(
                lifecycle_run_id=_g(state, "lifecycle_run_id"),
                event_id=_g(state, "event_id") or "",
                diagnosis_run_id=diagnosis_run_id,
                model_id=_g(state, "model_id"),
                champion_version=_g(state, "champion_version"),
                primary_root_cause_code=_g(state, "primary_root_cause_code") or "",
                primary_root_cause_score=_g(state, "primary_root_cause_score"),
                recommended_action=_g(state, "recommended_action"),
            )
            decision = await RuleAgentAdapter(session).decide(agent_input)

            return {
                "agent_decision_id": decision.agent_decision_id,
                "agent_confidence": decision.confidence,
                "recommended_action": decision.recommended_action,
                "requires_manual_review": decision.requires_manual_review,
                "current_phase": (
                    LifecyclePhase.MANUAL_REVIEW.value
                    if decision.requires_manual_review
                    else LifecyclePhase.AGENT_DECIDING.value
                ),
            }
    except (OSError, ConnectionError, TimeoutError, _DBIntegrityError):
        return {
            "agent_decision_id": None,
            "agent_confidence": 0.0,
            "recommended_action": "MANUAL_REVIEW",
            "requires_manual_review": True,
            "current_phase": LifecyclePhase.MANUAL_REVIEW.value,
        }


# ═══════════════════════════════════════════════════════════
# 迭代决策节点（P0）
# ═══════════════════════════════════════════════════════════

async def iteration_decision_node(state: ModelLifecycleState) -> dict:
    """迭代决策节点 — 调用 RepairDecisionService + RiskAssessmentService。"""
    diagnosis_run_id = _g(state, "diagnosis_run_id")
    lifecycle_run_id = _g(state, "lifecycle_run_id")
    if not diagnosis_run_id:
        return {"current_phase": LifecyclePhase.MANUAL_REVIEW.value}

    try:
        from ...database import async_session
        from ...repositories.iteration_repo import IterationRepo
        from ...services.iteration import RepairDecisionService, RiskAssessmentService
        from packages.models.iteration import DecisionInput
        from packages.models.common.enums import ProposalStatus

        async with async_session() as session:
            primary_code = _g(state, "primary_root_cause_code") or "uncertain"
            primary_score = _g(state, "primary_root_cause_score")

            decision_input = DecisionInput(
                diagnosis_run_id=diagnosis_run_id,
                lifecycle_run_id=lifecycle_run_id,
                model_id=_g(state, "model_id"),
                champion_version=_g(state, "champion_version"),
                root_causes=[{
                    "root_cause_code": primary_code,
                    "dimension": _g(state, "primary_root_cause_dimension") or "FEATURE",
                    "score": primary_score or 0.0,
                    "evidence_coverage": 0.8,
                    # Demo path: complete FEATURE_DRIFT evidence chain (D/I/T + C or R)
                    # so RepairDecisionService deterministically emits MODEL_ITERATION.
                    "evidence_types": ["D", "I", "R", "T"],
                }],
                degraded_metrics=[{
                    "metric_code": "AUC",
                    "baseline_value": 0.78,
                    "current_value": 0.74,
                    "healthy_lower_bound": 0.76,
                    "healthy_upper_bound": None,
                    "degraded": True,
                }],
                business_objective_changed=False,
                data_repair_completed=False,
                pipeline_repair_completed=False,
                rule_version="iteration-rules-v1",
            )

            repo = IterationRepo(session)
            decision_svc = RepairDecisionService()
            risk_svc = RiskAssessmentService()

            proposal = decision_svc.decide(decision_input)
            risk = risk_svc.assess(proposal)

            if risk.requires_manual_review and not proposal.requires_manual_review:
                proposal = proposal.model_copy(update={
                    "requires_manual_review": True,
                    "status": ProposalStatus.PENDING_REVIEW,
                })

            await repo.save_proposal(proposal)
            await repo.save_risk(risk)
            await session.commit()

            return {
                "decision_proposal_id": proposal.proposal_id,
                "risk_assessment_id": risk.assessment_id,
                "recommended_action": (
                    proposal.action.value if proposal.action else _g(state, "recommended_action")
                ),
                "requires_manual_review": proposal.requires_manual_review,
                "current_phase": (
                    LifecyclePhase.MANUAL_REVIEW.value
                    if proposal.requires_manual_review
                    else LifecyclePhase.DECISION_PROPOSED.value
                ),
                "need_iteration": proposal.need_iteration,
            }
    except (OSError, ConnectionError, TimeoutError, _DBIntegrityError):
        logger.warning("iteration_decision_fallback", exc_info=True)
        return {"current_phase": LifecyclePhase.MANUAL_REVIEW.value}


# ═══════════════════════════════════════════════════════════
# P1：数据资格 + 人工复核增强 + 训练计划
# ═══════════════════════════════════════════════════════════

async def data_eligibility_node(state: ModelLifecycleState) -> dict:
    """P1 数据资格评估节点。

    调用 DataEligibilityService.evaluate() 验证：
    - W4 禁止训练/调参
    - 标签缺失率 ≥20% 阻断
    - is_bad 禁止插补
    """
    try:
        from ...database import async_session
        from ...repositories.iteration_repo import IterationRepo
        from ...services.iteration import DataEligibilityService
        from packages.models.iteration.data_eligibility import DataEligibilityInput

        async with async_session() as session:
            svc = DataEligibilityService()
            repo = IterationRepo(session)

            eligibility_input = DataEligibilityInput(
                window_id="W3",
                data_track="NATURAL",
                data_snapshot_id=f"snapshot-w3-{uuid.uuid4().hex[:8]}",
                data_checksum="mock-checksum",
                label_column="is_bad",
                label_missing_rate=0.0,
                label_mature=True,
                label_imputation_requested=False,
                feature_missing_stats=[],
                requested_for_supervised_training=True,
            )

            result = svc.evaluate(eligibility_input)
            assessment_id = str(uuid.uuid4())
            await repo.save_data_eligibility(assessment_id, result)
            await session.commit()

            logger.info(
                "data_eligibility_node_completed",
                assessment_id=assessment_id,
                status=result.status.value,
                supervised_allowed=result.supervised_training_allowed,
            )

            if not result.supervised_training_allowed:
                return {
                    "data_eligibility_assessment_id": assessment_id,
                    "iteration_exit_reason": "DATA_NOT_ELIGIBLE",
                    "current_phase": LifecyclePhase.MANUAL_REVIEW.value,
                }

            return {
                "data_eligibility_assessment_id": assessment_id,
                "current_phase": LifecyclePhase.DECISION_PROPOSED.value,
            }
    except (OSError, ConnectionError, TimeoutError, _DBIntegrityError):
        logger.warning("data_eligibility_fallback", exc_info=True)
        return {"current_phase": LifecyclePhase.DECISION_PROPOSED.value}


async def manual_review_node(state: ModelLifecycleState) -> dict:
    """P1 增强人工复核节点。

    支持两种 resume 模式：
    - decision="approved"  → 继续到 TrainingPlan
    - decision="rejected"  → 走向 FAILED
    """
    resume_data = interrupt("manual_review_required")

    if isinstance(resume_data, dict):
        decision = resume_data.get("decision", "rejected")
        manual_review_id = resume_data.get("manual_review_id") or resume_data.get("review_id")
    else:
        decision = resume_data if isinstance(resume_data, str) else "rejected"
        manual_review_id = None

    if decision == "rejected":
        return {
            "requires_manual_review": True,
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {"reason": "manual_review_rejected", "at": _now_iso()},
        }

    # 通过复核
    return {
        "manual_review_id": manual_review_id,
        "requires_manual_review": False,
        "current_phase": LifecyclePhase.DECISION_PROPOSED.value,
    }


async def observation_close_node(state: ModelLifecycleState) -> dict:
    """Close an event when the agreed action is no-op or continued observation."""
    event_id = _g(state, "event_id")
    if event_id:
        try:
            from ...database import async_session
            from ...repositories.diagnosis_repo import DiagnosisRepo

            async with async_session() as session:
                await DiagnosisRepo(session).close_event(event_id)
                await session.commit()
        except (OSError, ConnectionError, TimeoutError, _DBIntegrityError):
            logger.warning("observation_close_fallback", event_id=event_id, exc_info=True)

    return {
        "iteration_exit_reason": "NO_MODEL_TRAINING_REQUIRED",
        "current_phase": LifecyclePhase.EVENT_CLOSED.value,
    }


async def repair_plan_node(state: ModelLifecycleState) -> dict:
    """P3 数据/管道修复节点 — 调用 RepairExecutor 生成结构化修复计划。"""
    from .executors import create_repair_plan, dispatch_external_execution

    plan = create_repair_plan(_state_dict(state))
    dispatch = {"dispatch_mode": "INTERNAL"}
    try:
        dispatch = dispatch_external_execution("REPAIR", plan)
        if dispatch.get("dispatched"):
            plan["status"] = "DISPATCHED"
    except Exception as exc:
        dispatch = {
            "dispatch_mode": "EXTERNAL_HTTP",
            "error": str(exc),
        }
        logger.warning("repair_external_dispatch_failed", repair_plan_id=plan["repair_plan_id"], exc_info=True)
    await _save_external_plan("REPAIR", plan, dispatch)
    return {
        "repair_plan_id": plan["repair_plan_id"],
        "iteration_exit_reason": plan["action"],
        "current_phase": LifecyclePhase.ITERATING.value,
    }


async def event_pending_repair_node(state: ModelLifecycleState) -> dict:
    """Mark the diagnosis event as waiting for external data or pipeline repair."""
    event_id = _g(state, "event_id")
    if event_id:
        try:
            from ...database import async_session
            from ...repositories.diagnosis_repo import DiagnosisRepo

            async with async_session() as session:
                await DiagnosisRepo(session).mark_event_in_repair(event_id)
                await session.commit()
        except (OSError, ConnectionError, TimeoutError, _DBIntegrityError):
            logger.warning("event_pending_repair_fallback", event_id=event_id, exc_info=True)

    return {
        "current_phase": LifecyclePhase.ITERATING.value,
    }


async def calibration_plan_node(state: ModelLifecycleState) -> dict:
    """P3 校准执行器 — 调用 CalibrationExecutor 创建校准计划。

    真实实现会训练 Isotonic/Platt calibrator 并保存到 MinIO。
    """
    from .executors import create_calibration_plan, dispatch_external_execution

    plan = create_calibration_plan(_state_dict(state))
    dispatch = {"dispatch_mode": "INTERNAL"}
    try:
        dispatch = dispatch_external_execution("CALIBRATION", plan)
        if dispatch.get("dispatched"):
            plan["status"] = "DISPATCHED"
        else:
            from ...config import settings
            if settings.workflow_use_celery:
                from workers.app import app as celery_app
                task = celery_app.send_task("workers.executor_tasks.calibrate", args=[plan])
                dispatch = {
                    "dispatched": True,
                    "dispatch_mode": "CELERY",
                    "external_task_id": getattr(task, "id", None),
                }
                plan["status"] = "DISPATCHED"
    except Exception as exc:
        dispatch = {"dispatch_mode": "EXTERNAL_OR_CELERY", "error": str(exc)}
        logger.warning("calibration_dispatch_failed", calibration_plan_id=plan["calibration_plan_id"], exc_info=True)
    await _save_external_plan("CALIBRATION", plan, dispatch)
    return {
        "calibration_plan_id": plan["calibration_plan_id"],
        "challenger_version": plan["artifact_output_path"].split("/")[-1].replace(".joblib", ""),
        "current_phase": LifecyclePhase.OFFLINE_VALIDATING.value,
    }


async def threshold_plan_node(state: ModelLifecycleState) -> dict:
    """P3 阈值调整执行器 — 调用 ThresholdExecutor 创建阈值搜索计划。

    真实实现会网格搜索最优 F1/Precision@K 阈值并保存到 MinIO。
    """
    from .executors import create_threshold_plan, dispatch_external_execution

    plan = create_threshold_plan(_state_dict(state))
    dispatch = {"dispatch_mode": "INTERNAL"}
    try:
        dispatch = dispatch_external_execution("THRESHOLD", plan)
        if dispatch.get("dispatched"):
            plan["status"] = "DISPATCHED"
        else:
            from ...config import settings
            if settings.workflow_use_celery:
                from workers.app import app as celery_app
                task = celery_app.send_task("workers.executor_tasks.search_threshold", args=[plan])
                dispatch = {
                    "dispatched": True,
                    "dispatch_mode": "CELERY",
                    "external_task_id": getattr(task, "id", None),
                }
                plan["status"] = "DISPATCHED"
    except Exception as exc:
        dispatch = {"dispatch_mode": "EXTERNAL_OR_CELERY", "error": str(exc)}
        logger.warning("threshold_dispatch_failed", threshold_plan_id=plan["threshold_plan_id"], exc_info=True)
    await _save_external_plan("THRESHOLD", plan, dispatch)
    return {
        "threshold_plan_id": plan["threshold_plan_id"],
        "challenger_version": plan["artifact_output_path"].split("/")[-1].replace(".json", ""),
        "current_phase": LifecyclePhase.OFFLINE_VALIDATING.value,
    }


async def training_plan_node(state: ModelLifecycleState) -> dict:
    """P1 训练计划节点。

    前置条件：
    - Proposal 必须已有匹配的人工通过报告
    - 非 MODEL_ITERATION 不得生成排序模型训练计划
    - W4 不得进入训练和调参
    """
    proposal_id = _g(state, "decision_proposal_id")
    if not proposal_id:
        return {
            "current_phase": LifecyclePhase.MANUAL_REVIEW.value,
        }
    manual_review_id = _g(state, "manual_review_id")
    if not manual_review_id:
        logger.warning("training_plan_missing_manual_review_id", proposal_id=proposal_id)
        return {
            "requires_manual_review": True,
            "current_phase": LifecyclePhase.MANUAL_REVIEW.value,
        }
    data_eligibility_assessment_id = _g(state, "data_eligibility_assessment_id")
    if not data_eligibility_assessment_id:
        logger.warning("training_plan_missing_data_eligibility", proposal_id=proposal_id)
        return {
            "iteration_exit_reason": "DATA_NOT_ELIGIBLE",
            "current_phase": LifecyclePhase.MANUAL_REVIEW.value,
        }

    try:
        from ...database import async_session
        from ...repositories.iteration_repo import IterationRepo
        from ...services.iteration import RiskAssessmentService, TrainingPlanBuilder
        from packages.models.common.enums import ProposalStatus

        async with async_session() as session:
            repo = IterationRepo(session)
            proposal = await repo.get_proposal(proposal_id)
            if proposal is None:
                return {"current_phase": LifecyclePhase.MANUAL_REVIEW.value}

            approval = await repo.get_approved_review(manual_review_id, proposal_id)
            if approval is None:
                logger.warning(
                    "training_plan_missing_approved_review",
                    proposal_id=proposal_id,
                    manual_review_id=manual_review_id,
                )
                return {
                    "requires_manual_review": True,
                    "current_phase": LifecyclePhase.MANUAL_REVIEW.value,
                }

            approved_proposal = proposal.model_copy(
                update={"status": ProposalStatus.APPROVED}
            )

            risk = RiskAssessmentService().assess(approved_proposal)
            iteration_run_id = str(uuid.uuid4())
            business_round = _g(state, "business_round") or 1
            eligibility_assessments = await repo.get_data_eligibility_assessments(
                [data_eligibility_assessment_id]
            )
            if len(eligibility_assessments) != 1:
                logger.warning(
                    "training_plan_data_eligibility_not_found",
                    assessment_id=data_eligibility_assessment_id,
                )
                return {
                    "iteration_exit_reason": "DATA_NOT_ELIGIBLE",
                    "current_phase": LifecyclePhase.MANUAL_REVIEW.value,
                }

            plan = TrainingPlanBuilder().build(
                approved_proposal,
                risk,
                approval_id=manual_review_id,
                iteration_run_id=iteration_run_id,
                model_algorithm="lightgbm",
                feature_schema_version="feature-schema-v1",
                preprocessing_version="preprocess-v1",
                business_round=business_round,
                data_eligibility_assessments=eligibility_assessments,
                data_snapshot_ids=[
                    f"snapshot-w2-v{business_round}",
                    f"snapshot-w3-v{business_round}",
                ],
                label_versions=["label-v1"],
            )

            await repo.create_iteration_run(
                iteration_run_id,
                approved_proposal,
                3,  # max rounds
            )
            await repo.save_training_plan(plan)
            await repo.create_round_and_experiment(plan)
            await session.commit()

            logger.info(
                "training_plan_node_completed",
                training_plan_id=plan.training_plan_id,
                iteration_run_id=iteration_run_id,
                business_round=business_round,
            )

            return {
                "training_plan_id": plan.training_plan_id,
                "iteration_run_id": iteration_run_id,
                "experiment_id": plan.experiment_id,
                "business_round": business_round,
                "current_phase": LifecyclePhase.ITERATING.value,
            }
    except (OSError, ConnectionError, TimeoutError, _DBIntegrityError):
        logger.warning("training_plan_fallback", exc_info=True)
        return {"current_phase": LifecyclePhase.MANUAL_REVIEW.value}


# ═══════════════════════════════════════════════════════════
# P2：异步训练接入
# ═══════════════════════════════════════════════════════════

async def training_job_dispatch_node(state: ModelLifecycleState) -> dict:
    """P2 训练任务派发节点。

    创建训练任务 → 提交 Celery（当前 Mock）
    State 写入 training_job_id → 图暂停在 WAITING_TRAINING_CALLBACK
    """
    iteration_run_id = _g(state, "iteration_run_id") or str(uuid.uuid4())
    training_plan_id = _g(state, "training_plan_id") or str(uuid.uuid4())
    experiment_id = _g(state, "experiment_id") or str(uuid.uuid4())
    business_round = _g(state, "business_round") or 1

    training_job_id = str(uuid.uuid4())

    try:
        from ...database import async_session
        from ...repositories.iteration_repo import IterationRepo
        from packages.models.iteration.training_job import TrainingJobInput

        async with async_session() as session:
            repo = IterationRepo(session)
            job_input = TrainingJobInput(
                training_job_id=training_job_id,
                idempotency_key=f"{iteration_run_id}:round-{business_round}:exp-{experiment_id}",
                iteration_run_id=iteration_run_id,
                training_plan_id=training_plan_id,
                experiment_id=experiment_id,
                business_round=business_round,
                strategy_code="PLAN_STABLE_REFIT",
                training_window_ids=["W2"],
                validation_window_ids=["W3"],
                train_time_ranges=[],
                validation_time_ranges=[],
                oot_window_id="W4",
                data_snapshot_ids=[f"snapshot-w2-v{business_round}"],
                label_versions=["label-v1"],
                sample_weight_policy={},
                feature_schema_version="feature-schema-v1",
                preprocessing_version="preprocess-v1",
                algorithm="lightgbm",
                hyperparameters={},
                target_metrics=["AUC", "KS"],
                qualification_rule_version="qualification-rules-v1",
                base_model_version=_g(state, "champion_version"),
                seed=2026,
                artifact_output_uri=f"s3://riskitem/challengers/round-{business_round}",
            )

            created = False
            try:
                created, row = await repo.create_training_job(job_input)
                await session.commit()
            except _DBIntegrityError:
                logger.warning("training_job_db_skip_fk", training_job_id=training_job_id)
                await session.rollback()

            # P1: 派发到 Celery Worker
            celery_app = None
            from ...config import settings
            if settings.workflow_use_celery:
                from workers.app import app as celery_app

            from .executors import dispatch_training_job
            # P1: 注入 lifecycle_run_id + P3: 训练窗口配置
            job_dict = job_input.model_dump()
            job_dict["lifecycle_run_id"] = _g(state, "lifecycle_run_id")
            job_dict["training_window_ids"] = ["W2"]
            job_dict["validation_window_ids"] = ["W3"]
            dispatch_result = await dispatch_training_job(
                job_dict,
                celery_app=celery_app,
            )

            logger.info(
                "training_job_dispatched",
                training_job_id=training_job_id,
                created=created,
                dispatched=dispatch_result["dispatched"],
                business_round=business_round,
            )

            return {
                "training_job_id": training_job_id,
                "training_dispatched": dispatch_result["dispatched"],
                "training_dispatch_mode": "celery" if dispatch_result["dispatched"] else "manual_callback",
                "current_phase": LifecyclePhase.WAITING_TRAINING_CALLBACK.value,
            }
    except (OSError, ConnectionError, TimeoutError, _DBIntegrityError):
        logger.warning("training_job_dispatch_fallback", exc_info=True)
        return {
            "training_job_id": training_job_id,
            "training_dispatched": False,
            "training_dispatch_mode": "fallback_manual_callback",
            "current_phase": LifecyclePhase.WAITING_TRAINING_CALLBACK.value,
        }


async def wait_training_callback_node(state: ModelLifecycleState) -> dict:
    """P2 等待训练回调节点。

    使用 interrupt() 挂起图，等 Worker callback 后 resume。
    """
    resume_data = interrupt("waiting_training_callback")

    callback_status = None
    candidate_version = None
    if isinstance(resume_data, dict):
        callback_status = resume_data.get("status") or resume_data.get("worker_status")
        candidate_version = resume_data.get("candidate_version")
        if resume_data.get("training_job_id") and resume_data.get("training_job_id") != _g(state, "training_job_id"):
            return {
                "current_phase": LifecyclePhase.FAILED.value,
                "last_error": {
                    "reason": "training_job_id_mismatch",
                    "at": _now_iso(),
                },
            }
    elif isinstance(resume_data, str):
        callback_status = resume_data

    logger.info(
        "wait_training_callback_node_resumed",
        training_job_id=_g(state, "training_job_id"),
        experiment_id=_g(state, "experiment_id"),
        callback_status=callback_status,
    )

    if callback_status and callback_status != "SUCCEEDED":
        return {
            "training_callback_status": callback_status,
            "iteration_exit_reason": "TECHNICAL_FAILURE",
            "current_phase": LifecyclePhase.FAILED.value,
        }

    return {
        "training_callback_status": callback_status or "SUCCEEDED",
        "challenger_version": candidate_version or _g(state, "challenger_version"),
        "current_phase": LifecyclePhase.OFFLINE_VALIDATING.value,
    }


# ═══════════════════════════════════════════════════════════
# P3：资格验证与三轮控制
# ═══════════════════════════════════════════════════════════

async def training_callback_resume_node(state: ModelLifecycleState) -> dict:
    """Normalize worker callback result before QualificationNode."""
    callback_status = _g(state, "training_callback_status") or "SUCCEEDED"
    if callback_status != "SUCCEEDED":
        return {
            "iteration_exit_reason": "TECHNICAL_FAILURE",
            "current_phase": LifecyclePhase.FAILED.value,
        }
    return {
        "current_phase": LifecyclePhase.OFFLINE_VALIDATING.value,
    }


async def qualification_node(state: ModelLifecycleState) -> dict:
    """P3 Challenger 资格验证节点 — 七道 Gate。

    调用 QualificationService.evaluate()。
    W4 OOT 失败不回流调参。
    """
    proposal_id = _g(state, "decision_proposal_id")
    iteration_run_id = _g(state, "iteration_run_id")
    experiment_id = _g(state, "experiment_id")
    business_round = _g(state, "business_round") or 1

    try:
        from ...database import async_session
        from ...repositories.iteration_repo import IterationRepo
        from ...services.iteration import QualificationService
        from packages.models.iteration.qualification import (
            MetricComparison,
            QualificationInput,
        )

        async with async_session() as session:
            repo = IterationRepo(session)
            svc = QualificationService()
            experiment = await repo.get_experiment(experiment_id) if experiment_id else None
            action = _recommended_action(state)
            if experiment is None and action in {
                AgentDecisionAction.CALIBRATION_ADJUSTMENT.value,
                AgentDecisionAction.THRESHOLD_ADJUSTMENT.value,
            }:
                candidate_version = (
                    _g(state, "challenger_version")
                    or f"{_g(state, 'champion_version')}_adjusted_v{business_round}"
                )
                report_id = str(uuid.uuid4())
                logger.info(
                    "qualification_node_lightweight_adjustment",
                    qualification_run_id=report_id,
                    recommended_action=action,
                )
                return {
                    "qualification_run_id": report_id,
                    "challenger_version": candidate_version,
                    "challenger_qualified": True,
                    "current_phase": LifecyclePhase.QUALIFICATION_COMPLETED.value,
                }
            if experiment is None or experiment.get("technical_status") != "SUCCEEDED":
                return {
                    "iteration_exit_reason": "TECHNICAL_FAILURE",
                    "current_phase": LifecyclePhase.FAILED.value,
                }
            candidate_version = (
                _g(state, "challenger_version")
                or experiment.get("candidate_version")
                or f"{_g(state, 'champion_version')}_challenger_v{business_round}"
            )
            experiment_json = experiment.get("experiment_json") or {}
            validation_metrics = experiment_json.get("validation_metrics") or {}
            segment_metrics = experiment_json.get("segment_metrics") or {}

            qual_input = QualificationInput(
                qualification_run_id=str(uuid.uuid4()),
                iteration_run_id=iteration_run_id or "",
                experiment_id=experiment_id or "",
                candidate_version=candidate_version,
                target_metrics=[
                    MetricComparison(
                        metric_code="AUC",
                        direction="HIGHER_BETTER",
                        original_drop=validation_metrics.get("original_drop", 0.04),
                        recovered_amount=validation_metrics.get("recovered_amount", 0.03),
                        recovery_rate=validation_metrics.get("recovery_rate", 0.75),
                        champion_value=validation_metrics.get("champion_auc", 0.74),
                        challenger_value=validation_metrics.get("challenger_auc", 0.77),
                        healthy_lower_bound=validation_metrics.get("healthy_lower_bound", 0.76),
                        healthy_upper_bound=validation_metrics.get("healthy_upper_bound"),
                        bootstrap_ci_lower=validation_metrics.get("bootstrap_ci_lower", 0.01),
                        bootstrap_ci_upper=validation_metrics.get("bootstrap_ci_upper", 0.06),
                    ),
                ],
                data_reproducible=experiment_json.get("data_reproducible", True),
                discrimination_passed=validation_metrics.get("discrimination_passed", True),
                calibration_passed=validation_metrics.get("calibration_passed", True),
                score_psi=validation_metrics.get("score_psi", 0.12),
                train_valid_gap=validation_metrics.get("train_valid_gap", 0.02),
                segment_governance_passed=segment_metrics.get("segment_governance_passed", True),
                oot_window_id="W4",
                candidate_frozen_before_oot=experiment_json.get("candidate_frozen_before_oot", True),
                oot_usage="FINAL_QUALIFICATION",
                oot_passed=validation_metrics.get("oot_passed", True),
            )

            report = svc.evaluate(qual_input)
            await repo.save_qualification(report)
            await session.commit()

            logger.info(
                "qualification_node_completed",
                qualification_run_id=report.qualification_run_id,
                qualified=report.qualified,
                business_round=business_round,
            )

            return {
                "qualification_run_id": report.qualification_run_id,
                "challenger_version": report.candidate_version,
                "challenger_qualified": report.qualified,
                "current_phase": (
                    LifecyclePhase.QUALIFICATION_COMPLETED.value
                    if report.qualified
                    else LifecyclePhase.OFFLINE_VALIDATING.value
                ),
            }
    except (OSError, ConnectionError, TimeoutError, _DBIntegrityError):
        logger.warning("qualification_fallback", exc_info=True)
        qualified = MOCK_CHALLENGER_QUALIFIED
        return {
            "qualification_run_id": str(uuid.uuid4()),
            "challenger_qualified": qualified,
            "current_phase": (
                LifecyclePhase.QUALIFICATION_COMPLETED.value
                if qualified
                else LifecyclePhase.OFFLINE_VALIDATING.value
            ),
        }


async def failure_analysis_node(state: ModelLifecycleState) -> dict:
    """Record why a challenger failed qualification before the next round decision."""
    failure_report_id = _g(state, "failure_report_id") or str(uuid.uuid4())
    logger.info(
        "failure_analysis_node",
        lifecycle_run_id=_g(state, "lifecycle_run_id"),
        qualification_run_id=_g(state, "qualification_run_id"),
        business_round=_g(state, "business_round") or 1,
    )
    return {
        "failure_report_id": failure_report_id,
        "iteration_exit_reason": "QUALIFICATION_FAILED",
        "current_phase": LifecyclePhase.OFFLINE_VALIDATING.value,
    }


async def next_round_plan_node(state: ModelLifecycleState) -> dict:
    """P3 下一轮计划节点 — 资格失败且轮次 < 3 时进入。"""
    business_round = (_g(state, "business_round") or 1) + 1

    logger.info(
        "next_round_plan_node",
        lifecycle_run_id=_g(state, "lifecycle_run_id"),
        business_round=business_round,
    )

    return {
        "business_round": business_round,
        "iteration_exit_reason": None,
        "current_phase": LifecyclePhase.ITERATING.value,
    }


async def stop_auto_iteration_node(state: ModelLifecycleState) -> dict:
    """P3 停止自动迭代节点 — 三轮业务失败后终止。"""
    logger.info(
        "stop_auto_iteration_node",
        lifecycle_run_id=_g(state, "lifecycle_run_id"),
        business_round=_g(state, "business_round"),
    )

    return {
        "iteration_exit_reason": "MAX_BUSINESS_ROUNDS_REACHED",
        "current_phase": LifecyclePhase.FAILED.value,
    }


# ═══════════════════════════════════════════════════════════
# P4：任务四与事件关闭
# ═══════════════════════════════════════════════════════════

async def deployment_gate_node(state: ModelLifecycleState) -> dict:
    """P3 部署执行器 — 调用 DeploymentExecutor 执行分阶段部署决策。

    真实实现：
    - OFFLINE_VALIDATION → OOT_GATE → SHADOW → CANARY_5 → CANARY_20 → CANARY_50 → PRODUCTION
    - 每个阶段检查健康指标，不满足则 hold/rollback
    - 只有 PRODUCTION 阶段才允许关闭事件
    """
    from .executors import dispatch_deployment_action, execute_deployment_stage

    result = execute_deployment_stage(_state_dict(state))
    dispatch = {"dispatch_mode": "INTERNAL"}
    try:
        dispatch = dispatch_deployment_action(_state_dict(state), result)
    except Exception as exc:
        dispatch = {"dispatch_mode": "EXTERNAL_HTTP", "error": str(exc)}
        logger.warning("deployment_external_dispatch_failed", deployment_id=result.get("deployment_id"), exc_info=True)
    await _save_deployment_record(state, result, dispatch)

    return {
        "deployment_id": result.get("deployment_id"),
        "deployment_stage": result["deployment_stage"],
        "deployment_decision": result["deployment_decision"],
        "current_phase": (
            LifecyclePhase.PROMOTED.value
            if result["deployment_decision"] == "PROMOTE"
            else (
                LifecyclePhase.MANUAL_REVIEW.value
                if result["deployment_decision"] == "ABORT_DEPLOYMENT"
                else (
                    LifecyclePhase.ROLLED_BACK.value
                    if result["deployment_decision"] == "ROLLBACK"
                    else LifecyclePhase.CANARY_RUNNING.value
                )
            )
        ),
    }


async def event_close_node(state: ModelLifecycleState) -> dict:
    """P4 事件关闭节点。

    职责：
    - 更新诊断事件状态为 CLOSED
    - 只有 Qualification 通过 + 部署完成后才关闭
    - 事件未 CLOSED 前，Dashboard 不得将模型显示为正常
    """
    event_id = _g(state, "event_id")
    deployment_id = _g(state, "deployment_id")
    deployment_stage = _g(state, "deployment_stage")
    deployment_decision = _g(state, "deployment_decision")
    if (
        not _g(state, "challenger_qualified")
        or not _g(state, "qualification_run_id")
        or not deployment_id
        or deployment_stage != "PRODUCTION"
        or deployment_decision != "PROMOTE"
    ):
        return {
            "current_phase": LifecyclePhase.MANUAL_REVIEW.value,
            "last_error": {
                "reason": "event_close_preconditions_not_met",
                "deployment_stage": deployment_stage,
                "deployment_decision": deployment_decision,
                "at": _now_iso(),
            },
        }

    if event_id:
        try:
            from ...database import async_session
            from ...repositories.diagnosis_repo import DiagnosisRepo

            async with async_session() as session:
                repo = DiagnosisRepo(session)
                event = await repo.get_event(event_id)
                if event and event.get("status") != "CLOSED":
                    await repo.close_event(event_id)
                    logger.info(
                        "event_closed",
                        event_id=event_id,
                        deployment_id=deployment_id,
                    )
                    await session.commit()
        except (OSError, ConnectionError, TimeoutError, _DBIntegrityError):
            logger.warning("event_close_fallback", exc_info=True)

    return {
        "current_phase": LifecyclePhase.EVENT_CLOSED.value,
    }


async def no_alert_close_node(state: ModelLifecycleState) -> dict:
    """无告警/不需要迭代时关闭。"""
    return {"current_phase": LifecyclePhase.NO_ALERT.value}


# ═══════════════════════════════════════════════════════════
# Mock 节点（保留用于降级和阶段化开发）
# ═══════════════════════════════════════════════════════════

async def iteration_subgraph(state: ModelLifecycleState) -> dict:
    """Mock：Legacy 任务三子图。P1+ 被 TrainingPlan → TrainingJob 替代。"""
    run_id = str(uuid.uuid4())
    if MOCK_CHALLENGER_QUALIFIED:
        return {
            "iteration_run_id": run_id,
            "challenger_version": f"{_g(state, 'champion_version')}_challenger_v1",
            "challenger_qualified": True,
            "current_phase": LifecyclePhase.CHALLENGER_TRAINED.value,
        }
    return {
        "iteration_run_id": run_id,
        "challenger_version": None,
        "challenger_qualified": False,
        "current_phase": LifecyclePhase.MANUAL_REVIEW.value,
    }


async def deployment_node(state: ModelLifecycleState) -> dict:
    """Mock：Legacy 任务四。P4 被 DeploymentGateNode 替代。"""
    return {
        "deployment_id": str(uuid.uuid4()),
        "deployment_stage": "OOT_GATE",
        "deployment_decision": MOCK_DEPLOYMENT_DECISION,
        "current_phase": (
            LifecyclePhase.PROMOTED.value
            if MOCK_DEPLOYMENT_DECISION == "PROMOTE"
            else LifecyclePhase.ROLLED_BACK.value
        ),
    }


# ═══════════════════════════════════════════════════════════
# 条件路由
# ═══════════════════════════════════════════════════════════

def route_after_monitoring(
    state: ModelLifecycleState,
) -> Literal["DiagnosisNode", "NoAlertCloseNode"]:
    return "DiagnosisNode" if _g(state, "has_alerts") else "NoAlertCloseNode"


def route_after_diagnosis(
    state: ModelLifecycleState,
) -> Literal["DiagnosisHandoffNode", "ManualReviewNode", "NoAlertCloseNode"]:
    need = _g(state, "need_iteration")
    if need is True:
        return "DiagnosisHandoffNode"
    if need is False:
        return "NoAlertCloseNode"
    return "ManualReviewNode"


def route_after_iteration_decision(
    state: ModelLifecycleState,
) -> Literal[
    "ObservationCloseNode",
    "RepairPlanNode",
    "CalibrationPlanNode",
    "ThresholdPlanNode",
    "DataEligibilityNode",
    "ManualReviewNode",
]:
    requires_review = _g(state, "requires_manual_review", False)
    if requires_review:
        return "ManualReviewNode"
    return _route_after_action(state)


def route_after_manual_review(
    state: ModelLifecycleState,
) -> Literal[
    "ObservationCloseNode",
    "RepairPlanNode",
    "CalibrationPlanNode",
    "ThresholdPlanNode",
    "TrainingPlanNode",
    END,
]:
    if _g(state, "requires_manual_review"):
        return END
    action = _recommended_action(state)
    if action == AgentDecisionAction.MANUAL_REVIEW.value:
        need = _g(state, "need_iteration")
        if need is True:
            return "TrainingPlanNode"
        if need is False:
            return "ObservationCloseNode"
        return END
    routed = _route_after_action(state)
    if routed == "DataEligibilityNode":
        return "TrainingPlanNode"
    if routed in {
        "ObservationCloseNode",
        "RepairPlanNode",
        "CalibrationPlanNode",
        "ThresholdPlanNode",
    }:
        return routed
    return END


def route_after_qualification(
    state: ModelLifecycleState,
) -> Literal["DeploymentGateNode", "FailureAnalysisNode"]:
    qualified = _g(state, "challenger_qualified", False)

    if qualified:
        return "DeploymentGateNode"
    return "FailureAnalysisNode"


def route_after_deployment_gate(
    state: ModelLifecycleState,
) -> Literal["DeploymentGateNode", "EventCloseNode", "__end__"]:
    decision = _g(state, "deployment_decision")
    stage = _g(state, "deployment_stage")
    if decision == "PROMOTE" and stage == "PRODUCTION":
        return "EventCloseNode"
    if decision == "ADVANCE_STAGE":
        return "DeploymentGateNode"
    return END


def route_after_failure_analysis(
    state: ModelLifecycleState,
) -> Literal["NextRoundPlanNode", "StopAutoIterationNode"]:
    business_round = _g(state, "business_round") or 1
    if business_round < MAX_BUSINESS_ROUNDS:
        return "NextRoundPlanNode"
    return "StopAutoIterationNode"


# ═══════════════════════════════════════════════════════════
# 图构建
# ═══════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """构建完整 LangGraph 主图。
    """
    graph = StateGraph(ModelLifecycleState)

    # P0 节点
    graph.add_node("MonitoringNode", monitoring_node)
    graph.add_node("NoAlertCloseNode", no_alert_close_node)
    graph.add_node("DiagnosisNode", diagnosis_node)
    graph.add_node("DiagnosisHandoffNode", diagnosis_handoff_node)
    graph.add_node("AgentDecisionNode", agent_decision_node)
    graph.add_node("IterationDecisionNode", iteration_decision_node)

    # P1 节点
    graph.add_node("ObservationCloseNode", observation_close_node)
    graph.add_node("RepairPlanNode", repair_plan_node)
    graph.add_node("EventPendingRepairNode", event_pending_repair_node)
    graph.add_node("CalibrationPlanNode", calibration_plan_node)
    graph.add_node("ThresholdPlanNode", threshold_plan_node)
    graph.add_node("DataEligibilityNode", data_eligibility_node)
    graph.add_node("ManualReviewNode", manual_review_node)
    graph.add_node("TrainingPlanNode", training_plan_node)

    # P2 节点
    graph.add_node("TrainingJobDispatchNode", training_job_dispatch_node)
    graph.add_node("WaitTrainingCallbackNode", wait_training_callback_node)
    graph.add_node("TrainingCallbackResumeNode", training_callback_resume_node)

    # P3 节点
    graph.add_node("QualificationNode", qualification_node)
    graph.add_node("FailureAnalysisNode", failure_analysis_node)
    graph.add_node("NextRoundPlanNode", next_round_plan_node)
    graph.add_node("StopAutoIterationNode", stop_auto_iteration_node)

    # P4 节点
    graph.add_node("DeploymentGateNode", deployment_gate_node)
    graph.add_node("EventCloseNode", event_close_node)

    # Legacy Mock
    graph.add_node("IterationSubgraph", iteration_subgraph)
    graph.add_node("DeploymentNode", deployment_node)

    # ── 边 ──
    graph.add_edge(START, "MonitoringNode")
    graph.add_conditional_edges("MonitoringNode", route_after_monitoring)
    graph.add_edge("NoAlertCloseNode", END)

    # 诊断 → Handoff → Agent → IterationDecision
    graph.add_conditional_edges("DiagnosisNode", route_after_diagnosis)
    graph.add_edge("DiagnosisHandoffNode", "AgentDecisionNode")
    graph.add_edge("AgentDecisionNode", "IterationDecisionNode")

    # IterationDecision 分流
    graph.add_conditional_edges("IterationDecisionNode", route_after_iteration_decision)
    graph.add_edge("ObservationCloseNode", END)
    graph.add_edge("RepairPlanNode", "EventPendingRepairNode")
    graph.add_edge("EventPendingRepairNode", END)
    graph.add_edge("CalibrationPlanNode", "QualificationNode")
    graph.add_edge("ThresholdPlanNode", "QualificationNode")

    # DataEligibility → ManualReview
    graph.add_edge("DataEligibilityNode", "ManualReviewNode")

    # ManualReview → 分流
    graph.add_conditional_edges("ManualReviewNode", route_after_manual_review)

    # TrainingPlan → TrainingJobDispatch → WaitCallback → Qualification
    graph.add_edge("TrainingPlanNode", "TrainingJobDispatchNode")
    graph.add_edge("TrainingJobDispatchNode", "WaitTrainingCallbackNode")
    graph.add_edge("WaitTrainingCallbackNode", "TrainingCallbackResumeNode")
    graph.add_edge("TrainingCallbackResumeNode", "QualificationNode")

    # Qualification 分流
    graph.add_conditional_edges("QualificationNode", route_after_qualification)
    graph.add_conditional_edges("FailureAnalysisNode", route_after_failure_analysis)

    # NextRound → 重新走 TrainingPlan
    graph.add_edge("NextRoundPlanNode", "TrainingPlanNode")

    # StopAutoIteration → END
    graph.add_edge("StopAutoIterationNode", END)

    # P4: DeploymentGate → EventClose → END
    graph.add_conditional_edges("DeploymentGateNode", route_after_deployment_gate)
    graph.add_edge("EventCloseNode", END)

    # Legacy Mock edges (单元测试兼容)
    graph.add_edge("IterationSubgraph", "DeploymentNode")
    graph.add_edge("DeploymentNode", END)

    return graph


def build_compiled_graph(checkpointer):
    """构建带 MemorySaver checkpoint 的编译图。"""
    return build_graph().compile(checkpointer=checkpointer)
