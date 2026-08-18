"""LangGraph 主图 + 节点。

当前监控后的路由口径：
    START → MonitoringNode
        ├─ FAILED → NoAlertCloseNode → END
        ├─ B1 trigger_diagnosis=True → DiagnosisNode
        ├─ B1 requires_manual_review=True → DiagnosisNode
        └─ 否则 → NoAlertCloseNode → END

注意：
    monitoring_alerts 是展示/汇总告警；
    B1 persistence_judgment_json 是是否进入诊断的决策源。
    因此“有展示告警”不等于“一定进入诊断”，
    “B1 触发但 monitoring_alerts 为空”也必须允许进入诊断。

诊断后的路由口径：
    DiagnosisNode
        ├─ requires_manual_review / MANUAL_REVIEW → ManualReviewNode
        ├─ need_iteration=True → DiagnosisHandoffNode → AgentDecisionNode → IterationDecisionNode
        ├─ need_iteration=False → ObservationCloseNode → END
        └─ 无法自动判断 → ManualReviewNode
"""

from __future__ import annotations

import json
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

# route_after_failure_analysis 优先从 iteration.yaml 读取 max_iteration_rounds；
# 此常量仅作为配置加载失败时的 fallback 默认值。
# 最大业务轮次统一为 2（A7 定稿）
MAX_BUSINESS_ROUNDS: int = 2

# B1 持续性等级 → KG 结构化过滤数值（NONE=0 / SHORT_TERM_7D=1 / SUSTAINED_30D=2 / SEVERE=3）
_DECAY_LEVELS: dict[str, int] = {
    "NONE": 0,
    "SHORT_TERM_7D": 1,
    "SUSTAINED_30D": 2,
    "SEVERE": 3,
}


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


async def _save_deployment_record(
    state: ModelLifecycleState | None = None,
    result: dict | None = None,
    dispatch: dict | None = None,
    health_result: dict | None = None,
    record: dict | None = None,
) -> None:
    """持久化部署记录。

    两种调用方式：
    - 旧式: _save_deployment_record(state, result, dispatch, health_result)
    - 新式: _save_deployment_record(record=pre_built_record)
    """
    try:
        from ...database import async_session
        from ...repositories.iteration_repo import IterationRepo

        if record is None and result is not None:
            # 旧式兼容：从 state + result 构建 record
            state_dict = _state_dict(state) if state else {}
            decision = result.get("deployment_decision")
            status = {
                "PROMOTE": "PROMOTED",
                "ROLLBACK": "ROLLED_BACK",
                "ABORT_DEPLOYMENT": "ABORTED",
                "HOLD": "HELD",
            }.get(decision, "RUNNING")
            health_metrics = state_dict.get("validation_metrics") or state_dict.get("training_metrics") or {}
            record = {
                "deployment_id": result.get("deployment_id"),
                "lifecycle_run_id": _g(state, "lifecycle_run_id") if state else None,
                "qualification_run_id": _g(state, "qualification_run_id") if state else None,
                "model_id": _g(state, "model_id") if state else None,
                "champion_version": _g(state, "champion_version") if state else None,
                "candidate_version": result.get("candidate_version") or (_g(state, "challenger_version") if state else None),
                "deployment_stage": result.get("deployment_stage"),
                "deployment_decision": decision,
                "status": status,
                "dispatch_mode": (dispatch or {}).get("dispatch_mode", "INTERNAL"),
                "external_task_id": (dispatch or {}).get("external_task_id"),
                "health_json": {
                    "deployment_health_passed": state_dict.get("deployment_health_passed", True) if state_dict else True,
                    "deployment_force_rollback": _g(state, "deployment_force_rollback", False) if state else False,
                    "health_check": health_result or {},
                },
                "external_response": (dispatch or {}).get("response"),
            }

        if record and record.get("deployment_id"):
            async with async_session() as session:
                await IterationRepo(session).save_deployment_record(record)
                await session.commit()
    except Exception:
        logger.warning("deployment_record_persist_failed", exc_info=True)


def _recommended_action(state: ModelLifecycleState) -> str | None:
    action = _g(state, "recommended_action")
    return action.value if hasattr(action, "value") else action


async def _update_monitoring_diagnosis_status(
    monitoring_run_id: str | None, diagnosis_status: str,
) -> None:
    if not monitoring_run_id:
        return
    try:
        from ...database import async_session
        from ...repositories.monitoring_repo import MonitoringRepo

        async with async_session() as session:
            await MonitoringRepo(session).update_diagnosis_status(
                monitoring_run_id, diagnosis_status,
            )
            await session.commit()
    except Exception:
        logger.warning(
            "monitoring_diagnosis_status_update_failed",
            monitoring_run_id=monitoring_run_id,
            diagnosis_status=diagnosis_status,
            exc_info=True,
        )


def _has_feature_level_issues(state: ModelLifecycleState) -> bool:
    """判断诊断结果中是否存在特征层问题，需要 FeatureReconstruction。

    线性模型更敏感：即使诊断证据为空，也需要走特征重构做 IMPUTE + STANDARDIZE。
    """
    state_dict = _state_dict(state)
    drift = state_dict.get("drift_features", [])
    missing = state_dict.get("high_missing_features", [])
    skew = state_dict.get("skewness", {})
    algorithm = state_dict.get("algorithm") or "lightgbm"

    # 树模型：有诊断证据才需要重构
    if algorithm.lower() in {"lightgbm", "xgboost", "random_forest", "catboost"}:
        return bool(drift or missing or skew)

    # 线性模型：总是需要特征重构（缺失插补 + 标准化 + 交互构造）
    return True


def _route_after_action(
    state: ModelLifecycleState,
) -> Literal[
    "ObservationCloseNode",
    "RepairPlanNode",
    "CalibrationPlanNode",
    "ThresholdPlanNode",
    "FeatureReconstructionNode",
    "FeatureSelectionNode",
    "TrainingPlanNode",
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
        # A7 §7: 路由由 TrainingMode 驱动，禁止按 strategy_tier 猜测训练模式
        training_mode = str(
            _g(state, "training_mode") or "FULL_RETRAIN"
        ).upper()
        if training_mode == "FEATURE_SELECTION":
            # 特征筛选独立执行器：生成冻结特征清单后重训
            return "FeatureSelectionNode"
        if training_mode == "FEATURE_RECONSTRUCTION" or (
            training_mode in {"FULL_RETRAIN", "PARAMETER_TUNING"}
            and _has_feature_level_issues(state)
        ):
            return "FeatureReconstructionNode"
        # INCREMENTAL_TRAIN：必须保持 Champion 特征契约，不走重构
        return "TrainingPlanNode"
    if need is False:
        return "ObservationCloseNode"
    return "ManualReviewNode"


# ═══════════════════════════════════════════════════════════
# 任务一：MonitoringNode
# ═══════════════════════════════════════════════════════════

async def monitoring_node(state: ModelLifecycleState) -> dict:
    """阶段 4 真实监控节点：调用 MonitoringService.run_full_pipeline() 执行 WP02-WP08 完整管道。

    产出：17 个汇总指标 + 诊断时间线 + per-feature 漂移/质量 + 检测器信号 + Sentinel 特征向量。
    """
    from ...services.monitoring.window_loader import (
        load_window_with_predictions,
        resolve_monitoring_window_ids,
    )

    model_id = _g(state, "model_id", "credit_model_001")
    champion_version = _g(state, "champion_version", "champion_v1")
    baseline_window_id, current_window_ids = resolve_monitoring_window_ids(
        model_id, champion_version,
    )
    baseline_df = load_window_with_predictions(baseline_window_id, model_id)
    current_dfs = [
        load_window_with_predictions(window_id, model_id)
        for window_id in current_window_ids
    ]
    w1_df = current_dfs[0]
    w2_df = current_dfs[1] if len(current_dfs) > 1 else current_dfs[0]
    w3_df = current_dfs[-1]

    try:
        from ...database import async_session
        from ...neo4j_db import get_neo4j_driver
        from ...services.knowledge_service import KnowledgeService
        from ...services.monitoring.monitoring_service import MonitoringService

        async with async_session() as session:
            driver = await get_neo4j_driver()
            knowledge = KnowledgeService(driver)
            service = MonitoringService(session, knowledge)

            result = await service.run_full_pipeline(
                model_id=_g(state, "model_id"),
                champion_version=champion_version,
                w0_df=baseline_df,
                w1_df=w1_df,
                w2_df=w2_df,
                w3_df=w3_df,
                baseline_window_id=baseline_window_id,
                current_window_id=current_window_ids[-1],
                current_window_dfs=current_dfs,
            )

            logger.info(
                "monitoring_node_completed",
                monitoring_run_id=result.monitoring_run_id,
                alert_count=result.alert_count,
            )

            # B1 持续性判定：检查是否需要触发诊断
            trigger_diagnosis = False
            decay_degree = "NONE"
            requires_manual_review = False
            status_7d = None
            status_30d = None
            diagnosis_status = None
            persistence_judgment = None
            try:
                from ...repositories.monitoring_repo import MonitoringRepo
                mon_repo = MonitoringRepo(session)
                run_record = await mon_repo.get_run(result.monitoring_run_id)
                judgment_json = run_record.get("persistence_judgment_json") if run_record else None
                diagnosis_status = run_record.get("diagnosis_status") if run_record else None
                if isinstance(judgment_json, str):
                    judgment_json = json.loads(judgment_json)
                if judgment_json and isinstance(judgment_json, dict):
                    persistence_judgment = judgment_json
                    trigger_diagnosis = bool(judgment_json.get("trigger_diagnosis", False))
                    decay_degree = judgment_json.get("decay_degree", "NONE")
                    requires_manual_review = bool(judgment_json.get("requires_manual_review", False))
                    status_7d = judgment_json.get("status_7d")
                    status_30d = judgment_json.get("status_30d")
                    logger.info(
                        "monitoring_persistence_judgment",
                        monitoring_run_id=result.monitoring_run_id,
                        has_alerts=result.has_alerts,
                        trigger_diagnosis=trigger_diagnosis,
                        decay_degree=decay_degree,
                        requires_manual_review=requires_manual_review,
                    )
            except Exception:
                logger.warning("persistence_judgment_load_failed", exc_info=True)

            should_enter_diagnosis = trigger_diagnosis or requires_manual_review

            # A7 §8: 当前生命周期已处于活动状态 → 只记录 trigger_cause
            # （真实信号），不创建新 run。独立监测事件入口（routers/monitoring.py
            # POST /runs）由 TriggerService 创建生命周期。
            # 异常触发读取真实 Sentinel 信号，不用告警严重度代替。
            max_sev = (
                result.max_alert_severity.value
                if result.max_alert_severity else None
            )
            sentinel_triggered = False
            try:
                _pj = persistence_judgment or {}
                sentinel_triggered = bool(
                    (_pj.get("sentinel_evidence") or {}).get("triggered")
                )
            except Exception:
                pass
            trigger_cause = None
            if sentinel_triggered:
                trigger_cause = "SENTINEL_ANOMALY"
            elif (persistence_judgment or {}).get("decay_degree") == "SEVERE":
                trigger_cause = "SEVERE_PERSISTENCE"
            elif max_sev in {"HIGH", "CRITICAL"}:
                trigger_cause = "THRESHOLD_BREACH"
            if trigger_cause:
                logger.info(
                    "trigger_cause_recorded",
                    monitoring_run_id=result.monitoring_run_id,
                    trigger_cause=trigger_cause,
                    sentinel_triggered=sentinel_triggered,
                )

            return {
                "monitoring_run_id": result.monitoring_run_id,
                "has_alerts": result.has_alerts,
                "alert_count": result.alert_count,
                "max_alert_severity": (
                    result.max_alert_severity.value if result.max_alert_severity else None
                ),
                "trigger_diagnosis": trigger_diagnosis,
                "decay_degree": decay_degree,
                "requires_manual_review": requires_manual_review,
                "status_7d": status_7d,
                "status_30d": status_30d,
                "diagnosis_status": diagnosis_status,
                "persistence_judgment": persistence_judgment,
                "trigger_cause": trigger_cause,
                "current_phase": (
                    LifecyclePhase.MONITORING_COMPLETED.value
                    if result.has_alerts or should_enter_diagnosis
                    else LifecyclePhase.NO_ALERT.value
                ),
            }

    except (OSError, ConnectionError, TimeoutError, _DBIntegrityError) as e:
        logger.error("monitoring_node_infra_error", exc_info=True)
        return {
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "monitoring_infra_error",
                "message": f"监控节点基础设施不可用：{e}",
                "at": _now_iso(),
            },
        }
    except Exception as e:
        logger.error("monitoring_node_unexpected_error", exc_info=True)
        return {
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "monitoring_unexpected_error",
                "message": f"监控节点发生未预期错误：{e}",
                "at": _now_iso(),
            },
        }


# ═══════════════════════════════════════════════════════════
# 任务二：DiagnosisNode + DiagnosisHandoffNode
# ═══════════════════════════════════════════════════════════

async def diagnosis_node(state: ModelLifecycleState) -> dict:
    """任务二诊断节点：调用 DiagnosisService 执行真实 D/R/C/T/I 根因诊断。

    诊断完成后从 DB 加载详细证据（drift_features / feature_importance / skewness），
    写入 State 供 FeatureReconstructionNode 使用。
    """
    monitoring_run_id = _g(state, "monitoring_run_id")
    lifecycle_run_id = _g(state, "lifecycle_run_id")

    if not monitoring_run_id:
        logger.error("diagnosis_node_missing_monitoring_run_id — 监控未运行或 State 丢失")
        return {
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "missing_monitoring_run_id",
                "message": "监控节点未写入 monitoring_run_id，无法启动诊断",
                "at": _now_iso(),
            },
            "warnings": ["监控未运行或 State 丢失 monitoring_run_id，请重新启动生命周期"],
        }

    try:
        from ...database import async_session
        from ...neo4j_db import get_neo4j_driver
        from ...repositories.diagnosis_repo import DiagnosisRepo
        from ...repositories.monitoring_repo import MonitoringRepo
        from ...services.diagnosis.diagnosis_service import DiagnosisService
        from ...services.knowledge_service import KnowledgeService
        from packages.models.monitoring.alert_context import AlertContext, AlertDetail
        from packages.models.common.enums import AvailabilityStatus, DataTrack, ObjectType, Severity

        async with async_session() as session:
            driver = await get_neo4j_driver()
            knowledge = KnowledgeService(driver)
            mon_repo = MonitoringRepo(session)
            diag_repo = DiagnosisRepo(session)

            run = await mon_repo.get_run(monitoring_run_id)
            alerts = await mon_repo.get_alerts(monitoring_run_id)

            # monitoring_alerts 里已包含 B1 PERSISTENCE_JUDGMENT 告警（由 _emit_persistence_alerts 写入）
            # 不再需要合成虚拟告警

            if not alerts:
                logger.info("diagnosis_node_no_alerts_skipping")
                await mon_repo.update_diagnosis_status(monitoring_run_id, "SKIPPED")
                await session.commit()
                return {
                    "diagnosis_run_id": None,
                    "primary_root_cause_code": "no_alerts",
                    "primary_root_cause_dimension": None,
                    "primary_root_cause_score": 0.0,
                    "recommended_action": "CONTINUE_OBSERVATION",
                    "need_iteration": False,
                    "current_phase": LifecyclePhase.DIAGNOSIS_COMPLETED.value,
                }

            alert_details = []
            for a in alerts:
                raw_detail = a.get("alert_detail") or {}
                metric_detail = (
                    raw_detail.get("metric_detail", raw_detail)
                    if isinstance(raw_detail, dict)
                    else raw_detail
                )
                alert_details.append(
                    AlertDetail(
                        alert_id=str(a["alert_id"]),
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
                        metric_detail=metric_detail,
                    )
                )

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

            # ── 从 DB 加载诊断详细证据，写入 State ──
            diag_warnings: list[str] = []
            drift_features: list[dict] = []
            high_missing_features: list[dict] = []
            feature_importance: dict[str, float] = {}
            current_feature_names: list[str] = []

            if result.diagnosis_run_id:
                try:
                    # 加载 drift 数据
                    drift_rows = await mon_repo.get_feature_drift_by_run(monitoring_run_id)
                    if drift_rows:
                        drift_features = [
                            {"feature_name": r.get("feature_name", ""),
                             "psi_value": r.get("psi_value", r.get("current_value", 0))}
                            for r in drift_rows
                        ]
                        logger.info("diagnosis_drift_features_loaded count=%d", len(drift_features))
                    else:
                        diag_warnings.append("监控未产出 drift 数据：PSI 验证器可能基于空数据运行")

                    # 加载 feature importance
                    model_id = _g(state, "model_id", "")
                    importance = await service._load_feature_importance(model_id) if model_id else None
                    if importance:
                        feature_importance = importance
                        current_feature_names = list(importance.keys())
                    else:
                        diag_warnings.append("无法加载特征重要性：特征交互和重要性检查将跳过")
                except Exception as evidence_exc:
                    logger.warning("diagnosis_evidence_load_failed err=%s", evidence_exc)

            return_dict: dict = {
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
                "requires_manual_review": result.requires_manual_review or _g(state, "requires_manual_review", False),
                # A7 §4/§5: L1 结构化上下文由诊断输出持久化写入 State
                "impact_scope": getattr(result, "impact_scope", None)
                or _g(state, "impact_scope"),
                "change_pattern": getattr(result, "change_pattern", None)
                or _g(state, "change_pattern"),
                "segment_evidence": getattr(result, "segment_evidence", None)
                or _g(state, "segment_evidence"),
                "drift_features": drift_features,
                "high_missing_features": high_missing_features,
                "feature_importance": feature_importance,
                "feature_names": current_feature_names,
                "current_phase": LifecyclePhase.DIAGNOSIS_COMPLETED.value,
            }
            if diag_warnings:
                return_dict["warnings"] = diag_warnings
            # 优先保留服务返回的 diagnosis_status（如 INSUFFICIENT_DATA），
            # 只有服务正常完成时才按人工复核重新计算。
            diagnosis_status = getattr(result, "diagnosis_status", None) or "COMPLETED"
            if diagnosis_status == "COMPLETED":
                diagnosis_status = (
                    "MANUAL_REVIEW"
                    if return_dict["requires_manual_review"]
                    or return_dict["recommended_action"] == "MANUAL_REVIEW"
                    else "COMPLETED"
                )
            await mon_repo.update_diagnosis_status(monitoring_run_id, diagnosis_status)
            return_dict["diagnosis_status"] = diagnosis_status
            await session.commit()
            return return_dict

    except (OSError, ConnectionError, TimeoutError, _DBIntegrityError) as e:
        logger.error("diagnosis_node_infra_error", exc_info=True)
        await _update_monitoring_diagnosis_status(monitoring_run_id, "FAILED")
        return {
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "diagnosis_infra_error",
                "message": f"诊断节点基础设施不可用：{e}",
                "at": _now_iso(),
            },
        }
    except Exception as e:
        logger.error("diagnosis_node_unexpected_error", exc_info=True)
        await _update_monitoring_diagnosis_status(monitoring_run_id, "FAILED")
        return {
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "diagnosis_unexpected_error",
                "message": f"诊断节点发生未预期错误：{e}",
                "at": _now_iso(),
            },
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
    except Exception as e:
        logger.error("diagnosis_handoff_unexpected_error", exc_info=True)
        return {
            "event_id": str(uuid.uuid4()),
            "agent_handoff_status": "ERROR",
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "diagnosis_handoff_unexpected_error",
                "message": f"诊断交接发生未预期错误：{e}",
                "at": _now_iso(),
            },
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
    except Exception as e:
        logger.error("agent_decision_unexpected_error", exc_info=True)
        return {
            "agent_decision_id": None,
            "agent_confidence": 0.0,
            "recommended_action": "MANUAL_REVIEW",
            "requires_manual_review": True,
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "agent_decision_unexpected_error",
                "message": f"Agent 决策发生未预期错误：{e}",
                "at": _now_iso(),
            },
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
            primary_dimension = _g(state, "primary_root_cause_dimension")
            if not primary_dimension:
                logger.error("iteration_decision_missing_dimension — 诊断未返回根因维度")
                return {
                    "iteration_exit_reason": "MISSING_DIAGNOSIS_DIMENSION",
                    "current_phase": LifecyclePhase.MANUAL_REVIEW.value,
                    "warnings": ["诊断节点未返回根因维度，无法决策——转人工复核"],
                }

            # 从 DB 加载诊断的真实 evidence_types（D/R/C/T/I 字母）
            # 只统计 applicable=True 的证据：不适用的验证器不构成证据链
            #（禁止 fail-open 用占位集合凑数）。
            evidence_types: list[str] = []
            evidence_coverage: float = 0.0
            try:
                from ...repositories.diagnosis_repo import DiagnosisRepo
                diag_repo = DiagnosisRepo(session)
                evidence_records = await diag_repo.get_evidence_for_run(diagnosis_run_id)
                primary_records = [
                    e for e in evidence_records
                    if str(e.get("hypothesis_code") or "").upper()
                    == str(primary_code).upper()
                ]
                applicable_records = [
                    e for e in primary_records if e.get("applicable")
                ]
                evidence_types = sorted({
                    str(e.get("evidence_type", ""))
                    for e in applicable_records
                    if e.get("evidence_type")
                })
                if primary_records:
                    evidence_coverage = round(
                        len(applicable_records) / len(primary_records), 2
                    )
            except Exception:
                logger.warning("iteration_decision_evidence_types_load_failed", exc_info=True)

            # 从 DB 加载监控的退化指标
            degraded_metrics: list[dict] = []
            monitoring_run_id = _g(state, "monitoring_run_id")
            if monitoring_run_id:
                try:
                    from ...repositories.monitoring_repo import MonitoringRepo
                    mon_repo = MonitoringRepo(session)
                    mon_metrics = await mon_repo.get_metrics(monitoring_run_id)
                    degraded_metrics = [
                        {
                            "metric_code": m.get("metric_code", ""),
                            "baseline_value": m.get("baseline_value"),
                            "current_value": m.get("current_value"),
                            "healthy_lower_bound": m.get("healthy_lower_bound"),
                            "healthy_upper_bound": m.get("healthy_upper_bound"),
                            # monitoring_metrics 的触发列是 triggered（阈值评估
                            # 后 update_metric_triggered 写回），不是 degraded
                            "degraded": bool(m.get("triggered", False)),
                        }
                        for m in mon_metrics
                        if m.get("triggered")
                    ]
                except Exception:
                    logger.warning("iteration_decision_metrics_load_failed", exc_info=True)

            # rule_version 从 config 拿
            from ...services.iteration.config_loader import load_iteration_config
            rule_version = load_iteration_config().iteration.rule_version

            # Champion 算法家族（来自 assets bundle manifest；增量策略选择依据）
            champion_family: str | None = None
            try:
                from pathlib import Path as _Path
                manifest_path = (
                    _Path(__file__).resolve().parents[4]
                    / "assets" / "champion_models"
                    / (_g(state, "model_id") or "")
                    / (_g(state, "champion_version") or "champion_v1")
                    / "training_manifest.json"
                )
                if manifest_path.is_file():
                    import json as _json
                    champion_family = _json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    ).get("algorithm_family")
            except Exception:
                logger.warning("iteration_decision_algorithm_family_load_failed", exc_info=True)

            decision_input = DecisionInput(
                diagnosis_run_id=diagnosis_run_id,
                lifecycle_run_id=lifecycle_run_id,
                model_id=_g(state, "model_id"),
                champion_version=_g(state, "champion_version"),
                algorithm_family=champion_family,
                root_causes=[{
                    "root_cause_code": primary_code,
                    "dimension": primary_dimension,
                    "score": primary_score or 0.0,
                    "evidence_coverage": evidence_coverage,
                    "evidence_types": evidence_types,
                }],
                degraded_metrics=degraded_metrics,
                business_objective_changed=_g(state, "business_objective_changed") or False,
                data_repair_completed=_g(state, "data_repair_completed") or False,
                pipeline_repair_completed=_g(state, "pipeline_repair_completed") or False,
                # A7 §4/§5: L1 结构化上下文（KG 不能替 L1 选择）
                monitoring_run_id=_g(state, "monitoring_run_id") or None,
                decay_degree=_g(state, "decay_degree") or None,
                impact_scope=_g(state, "impact_scope") or None,
                change_pattern=_g(state, "change_pattern") or None,
                business_round=int(_g(state, "business_round") or 1),
                manual_approval=bool(_g(state, "manual_approval") or False),
                # A7 §4: 冻结合格客群定义（segment_weighted_retrain 证据）
                segment_evidence=_g(state, "segment_evidence") or None,
                # W3 失败归因证据（feature_selection_retrain 的 L1 门槛）
                failure_report_id=_g(state, "failure_report_id") or None,
                unstable_feature_codes=[
                    str(c).strip()
                    for c in (_g(state, "unstable_feature_codes") or "").split(",")
                    if str(c).strip()
                ],
                feature_evidence_source=_g(state, "feature_evidence_source") or None,
                rule_version=rule_version,
            )

            repo = IterationRepo(session)
            decision_svc = RepairDecisionService()
            risk_svc = RiskAssessmentService()

            # P3 KG: 查询 RootCause → Strategy 候选
            from ...neo4j_db import get_neo4j_driver
            from ...services.knowledge_service import KnowledgeService

            kg_driver = await get_neo4j_driver()
            knowledge = KnowledgeService(kg_driver)

            # KG 查询时传入 severity + algorithm + 结构化条件，
            # 边上的过滤条件自动筛选（持续性等级 / 业务轮次 / 证据上下文）
            raw_algorithm_family = champion_family or _g(state, "algorithm") or None
            kg_algorithm = RepairDecisionService._normalize_algorithm_family(
                raw_algorithm_family
            ) or None
            kg_decay_degree = _g(state, "decay_degree")
            kg_decay_level = _DECAY_LEVELS.get(kg_decay_degree, None)
            kg_business_round = int(_g(state, "business_round") or 0) or None
            # A7 §6.1: 运行时证据上下文（逐边 required_context 校验）。
            # 每个证据码都有真实来源，无法可靠判定的不授予，宁可漏召也不伪造：
            # - sustained_30d: B1 持续性判定
            # - incremental_algorithm_supported: champion 算法族
            # - champion_artifact_available: 监测已完成 = Champion 预测已真实加载
            # - schema_compatible: Sentinel schema hash 校验通过（非 SCHEMA_MISMATCH）
            # - manual_approval: 真实 Review 记录 APPROVED（TrainingPlanNode 写入）
            # - unstable_feature_subset_confirmed / feature_selection_evidence_available:
            #   第二轮 + W3 失败归因报告真实存在
            available_context_codes: list[str] = []
            if kg_decay_degree == "SUSTAINED_30D":
                available_context_codes.append("sustained_30d")
            if RepairDecisionService._supports_incremental(raw_algorithm_family):
                available_context_codes.append("incremental_algorithm_supported")
            if _g(state, "monitoring_run_id"):
                available_context_codes.append("champion_artifact_available")
            judgment = _g(state, "persistence_judgment") or {}
            sentinel_evidence = judgment.get("sentinel_evidence") or {}
            # ACTIVE = Sentinel schema hash 校验通过（真实监控信号）
            if sentinel_evidence.get("sentinel_status") == "ACTIVE":
                available_context_codes.append("schema_compatible")
            if bool(_g(state, "manual_approval") or False):
                available_context_codes.append("manual_approval")
            # 第二轮特征证据：来自真实 W3 失败归因（FailureAttributionService），
            # 不是"存在 ID"占位语义
            if int(_g(state, "business_round") or 1) >= 2:
                if bool(_g(state, "unstable_feature_subset_confirmed") or False):
                    available_context_codes.append(
                        "unstable_feature_subset_confirmed"
                    )
                if (
                    _g(state, "failure_report_id")
                    and _g(state, "failed_gate_codes")
                ):
                    available_context_codes.append(
                        "feature_selection_evidence_available"
                    )
            iteration_ctx = await knowledge.query_iteration_context(
                root_cause_code=primary_code,
                diagnosis_run_id=diagnosis_run_id,
                severity=primary_score,
                algorithm=kg_algorithm,
                decay_level=kg_decay_level,
                business_round=kg_business_round,
                available_context_codes=available_context_codes,
            )

            # 优先用 KG 决策，KG 降级时回退 YAML
            proposal = decision_svc.decide_with_kg(decision_input, iteration_ctx)
            risk = risk_svc.assess(proposal)

            if risk.requires_manual_review and not proposal.requires_manual_review:
                proposal = proposal.model_copy(update={
                    "requires_manual_review": True,
                    "status": ProposalStatus.PENDING_REVIEW,
                })

            await repo.save_proposal(proposal)
            await repo.save_risk(risk)
            await session.commit()

            # 从正式合同字段提取 algorithm / strategy_tier / training_mode：
            # primary_training_mode 是唯一训练模式来源，禁止从 strategy_tier 猜测
            first_selection = proposal.strategies[0] if proposal.strategies else None
            algorithm = (
                first_selection.parameters.get("algorithm")
                if first_selection else None
            )
            strategy_tier = (
                first_selection.parameters.get("strategy_tier", "full")
                if first_selection else "full"
            )
            training_mode = (
                first_selection.primary_training_mode
                if first_selection else "full"
            )
            iteration_warnings: list[str] = []
            if strategy_tier == "full" and first_selection is not None:
                strategy_tier_from_kg = first_selection.parameters.get("strategy_tier")
                if not strategy_tier_from_kg:
                    msg = "KG Strategy 边未配置 strategy_tier，默认使用 full（全量重构+调参+重训）—— 请在 Neo4j 配置该属性以精确控制迭代策略"
                    logger.warning("strategy_tier_defaulted_to_full", msg=msg)
                    iteration_warnings.append(msg)

            return {
                "decision_proposal_id": proposal.proposal_id,
                "warnings": iteration_warnings or None,
                "risk_assessment_id": risk.assessment_id,
                "recommended_action": (
                    proposal.action.value if proposal.action else _g(state, "recommended_action")
                ),
                "requires_manual_review": proposal.requires_manual_review,
                "decision_reasons": proposal.decision_reasons,
                "selected_strategy_code": (
                    proposal.selected_strategy_code
                    or (proposal.strategies[0].strategy_code if proposal.strategies else None)
                ),
                "algorithm": algorithm,
                "strategy_tier": strategy_tier,
                "training_mode": training_mode,
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
    except Exception as e:
        logger.error("iteration_decision_unexpected_error", exc_info=True)
        return {
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "iteration_decision_unexpected_error",
                "message": f"迭代决策发生未预期错误：{e}",
                "at": _now_iso(),
            },
        }


# ═══════════════════════════════════════════════════════════
# P1：数据资格 + 人工复核增强 + 训练计划
# ═══════════════════════════════════════════════════════════

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
    if not event_id:
        logger.warning("observation_close_node_missing_event_id")
    else:
        try:
            from ...database import async_session
            from ...repositories.diagnosis_repo import DiagnosisRepo

            async with async_session() as session:
                await DiagnosisRepo(session).close_event(event_id)
                await session.commit()
        except (OSError, ConnectionError, TimeoutError, _DBIntegrityError):
            logger.warning("observation_close_fallback", event_id=event_id, exc_info=True)
        except Exception as e:
            logger.error("observation_close_unexpected_error", event_id=event_id, exc_info=True)
            return {
                "iteration_exit_reason": "OBSERVATION_CLOSE_ERROR",
                "current_phase": LifecyclePhase.FAILED.value,
                "last_error": {
                    "reason": "observation_close_unexpected_error",
                    "message": f"关闭观察事件发生未预期错误：{e}",
                    "at": _now_iso(),
                },
            }

    return {
        "iteration_exit_reason": "NO_MODEL_TRAINING_REQUIRED",
        "current_phase": LifecyclePhase.EVENT_CLOSED.value,
    }


async def repair_plan_node(state: ModelLifecycleState) -> dict:
    """P3 A3/A4 — 派发真实修复 Worker，等待同窗回放与资格结果。

    派发成功后标记事件待修复，然后 interrupt 等待回调；回调后由
    qualify_repair（action_execution_service）做真实回放资格判定——
    修复是否真的修好了由证据回答，不允许"派发即成功"。
    """
    from .executors import create_repair_plan, dispatch_external_execution
    from ...config import settings

    # A3/A4 修复对象：诊断产出的漂移特征（真实证据，不靠调用方声明）
    state_dict = _state_dict(state)
    if not state_dict.get("affected_features"):
        drift_features = _g(state, "drift_features") or []
        state_dict["affected_features"] = [
            str(f.get("feature_name")) for f in drift_features if f.get("feature_name")
        ]

    try:
        plan = create_repair_plan(state_dict)
    except Exception as e:
        logger.error("repair_plan_creation_failed", exc_info=True)
        return {
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "repair_plan_creation_error",
                "message": f"修复计划创建失败：{e}",
                "at": _now_iso(),
            },
        }

    dispatch: dict = {"dispatch_mode": "INTERNAL"}
    try:
        dispatch = dispatch_external_execution("REPAIR", plan)
        if dispatch.get("dispatched"):
            plan["status"] = "DISPATCHED"
        elif settings.workflow_use_celery:
            try:
                from workers.app import app as celery_app

                task = celery_app.send_task(
                    "workers.executor_tasks.repair_and_replay", args=[plan]
                )
                dispatch = {
                    "dispatched": True,
                    "dispatch_mode": "CELERY",
                    "external_task_id": getattr(task, "id", None),
                }
                plan["status"] = "DISPATCHED"
            except Exception as celery_exc:
                dispatch = {"dispatch_mode": "CELERY", "error": str(celery_exc)}
                logger.warning("repair_celery_dispatch_failed", exc_info=True)
    except Exception as exc:
        dispatch = {
            "dispatch_mode": "EXTERNAL_HTTP",
            "error": str(exc),
        }
        logger.warning("repair_external_dispatch_failed",
                       repair_plan_id=plan["repair_plan_id"], exc_info=True)

    await _save_external_plan("REPAIR", plan, dispatch)

    if not dispatch.get("dispatched"):
        logger.error("repair_plan_dispatch_blocked")
        return {
            "repair_plan_id": plan["repair_plan_id"],
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "repair_dispatch_failed",
                "message": "修复计划派发失败，外部执行器和 Celery 均不可用",
                "at": _now_iso(),
            },
        }

    # 派发成功：标记诊断事件待修复（外部团队可见）
    await _mark_event_pending_repair(state)

    # interrupt 等待 Worker 回调（fail-closed：无 status 不得默认成功）
    resume_data = interrupt("waiting_repair_and_replay_callback")
    if not isinstance(resume_data, dict) or not resume_data.get("status"):
        return {
            "repair_plan_id": plan["repair_plan_id"],
            "repair_qualified": False,
            "iteration_exit_reason": "REPAIR_CALLBACK_CONTRACT_INVALID",
            "current_phase": LifecyclePhase.FAILED.value,
        }
    callback_status = str(resume_data["status"]).upper()
    execution_result = {
        "status": callback_status,
        "artifact_uri": resume_data.get("artifact_uri"),
        "artifact_checksum": resume_data.get("artifact_checksum"),
        "metrics": resume_data.get("metrics") or {},
        "consumption_receipt": resume_data.get("consumption_receipt") or {},
    }
    if callback_status != "SUCCEEDED":
        return {
            "repair_plan_id": plan["repair_plan_id"],
            "repair_qualified": False,
            "repair_execution_result": execution_result,
            "iteration_exit_reason": "REPAIR_WORKER_FAILED",
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "repair_worker_failed",
                "message": resume_data.get("error_message") or "修复 Worker 执行失败",
                "at": _now_iso(),
            },
        }
    from ...services.iteration.action_execution_service import qualify_repair

    qualified, reasons = qualify_repair(execution_result)
    return {
        "repair_plan_id": plan["repair_plan_id"],
        "repair_artifact_uri": execution_result["artifact_uri"],
        "repair_artifact_checksum": execution_result["artifact_checksum"],
        "repair_execution_result": execution_result,
        "repair_qualified": qualified,
        "validation_metrics": execution_result["metrics"],
        "iteration_exit_reason": None if qualified else "REPAIR_REPLAY_QUALIFICATION_FAILED",
        "current_phase": (
            LifecyclePhase.QUALIFICATION_COMPLETED.value
            if qualified
            else LifecyclePhase.FAILED.value
        ),
        "last_error": (
            None
            if qualified
            else {
                "reason": "repair_replay_qualification_failed",
                "message": ",".join(reasons),
                "at": _now_iso(),
            }
        ),
    }


async def _mark_event_pending_repair(state: ModelLifecycleState) -> None:
    """标记诊断事件为待修复（外部数据/管道修复期间的事件状态）。"""
    event_id = _g(state, "event_id")
    if not event_id:
        logger.warning("event_pending_repair_node_missing_event_id")
        return
    try:
        from ...database import async_session
        from ...repositories.diagnosis_repo import DiagnosisRepo

        async with async_session() as session:
            await DiagnosisRepo(session).mark_event_in_repair(event_id)
            await session.commit()
    except (OSError, ConnectionError, TimeoutError, _DBIntegrityError):
        logger.warning("event_pending_repair_fallback", event_id=event_id, exc_info=True)
    except Exception as e:
        logger.error("event_pending_repair_unexpected_error", event_id=event_id, exc_info=True)


async def event_pending_repair_node(state: ModelLifecycleState) -> dict:
    """Mark the diagnosis event as waiting for external data or pipeline repair."""
    event_id = _g(state, "event_id")
    if not event_id:
        logger.warning("event_pending_repair_node_missing_event_id")
    else:
        try:
            from ...database import async_session
            from ...repositories.diagnosis_repo import DiagnosisRepo

            async with async_session() as session:
                await DiagnosisRepo(session).mark_event_in_repair(event_id)
                await session.commit()
        except (OSError, ConnectionError, TimeoutError, _DBIntegrityError):
            logger.warning("event_pending_repair_fallback", event_id=event_id, exc_info=True)
        except Exception as e:
            logger.error("event_pending_repair_unexpected_error", event_id=event_id, exc_info=True)
            return {
                "current_phase": LifecyclePhase.FAILED.value,
                "last_error": {
                    "reason": "event_pending_repair_unexpected_error",
                    "message": f"标记事件待修复发生未预期错误：{e}",
                    "at": _now_iso(),
                },
            }

    return {
        "current_phase": LifecyclePhase.ITERATING.value,
    }


async def calibration_plan_node(state: ModelLifecycleState) -> dict:
    """P3 校准执行器 — 创建校准计划 → 派发 Worker → interrupt 等待回调。"""
    from .executors import create_calibration_plan, dispatch_external_execution
    from ...config import settings

    try:
        plan = create_calibration_plan(_state_dict(state))
    except Exception as e:
        logger.error("calibration_plan_creation_failed", exc_info=True)
        return {
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "calibration_plan_creation_error",
                "message": f"校准计划创建失败：{e}",
                "at": _now_iso(),
            },
        }

    dispatch: dict = {"dispatch_mode": "INTERNAL"}
    try:
        dispatch = dispatch_external_execution("CALIBRATION", plan)
        if dispatch.get("dispatched"):
            plan["status"] = "DISPATCHED"
        elif settings.workflow_use_celery:
            try:
                from workers.app import app as celery_app
                task = celery_app.send_task("workers.executor_tasks.calibrate", args=[plan])
                dispatch = {
                    "dispatched": True,
                    "dispatch_mode": "CELERY",
                    "external_task_id": getattr(task, "id", None),
                }
                plan["status"] = "DISPATCHED"
            except Exception as celery_exc:
                dispatch = {"dispatch_mode": "CELERY", "error": str(celery_exc)}
                logger.warning("calibration_celery_dispatch_failed", exc_info=True)
    except Exception as exc:
        dispatch = {"dispatch_mode": "EXTERNAL_OR_CELERY", "error": str(exc)}
        logger.warning("calibration_external_dispatch_failed",
                       calibration_plan_id=plan["calibration_plan_id"], exc_info=True)

    await _save_external_plan("CALIBRATION", plan, dispatch)

    if not dispatch.get("dispatched"):
        logger.error("calibration_plan_dispatch_blocked")
        return {
            "calibration_plan_id": plan["calibration_plan_id"],
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "calibration_dispatch_failed",
                "message": "校准计划派发失败，外部执行器和 Celery 均不可用",
                "at": _now_iso(),
            },
        }

    # interrupt 等待 Worker 回调（fail-closed：无 status 不得默认成功）
    resume_data = interrupt("waiting_calibration_callback")
    if not isinstance(resume_data, dict) or not resume_data.get("status"):
        return {
            "calibration_plan_id": plan["calibration_plan_id"],
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "calibration_callback_contract_invalid",
                "message": "校准回调缺少 status 字段，按失败处理",
                "at": _now_iso(),
            },
        }
    callback_status = str(resume_data["status"]).upper()
    # Worker 回调字段为 artifact_uri（与包版执行合同一致）
    calibrator_uri = resume_data.get("artifact_uri") or resume_data.get(
        "calibrator_artifact_uri"
    )

    if callback_status != "SUCCEEDED":
        logger.error("calibration_worker_failed", plan_id=plan["calibration_plan_id"])
        return {
            "calibration_plan_id": plan["calibration_plan_id"],
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "calibration_worker_failed",
                "message": (
                    resume_data.get("error_message") or "校准 Worker 执行失败"
                ),
                "at": _now_iso(),
            },
        }

    # 安全解析 challenger_version
    artifact_path: str = str(plan.get("artifact_output_path", ""))
    candidate_version = str(plan.get("champion_version", "v1"))
    if artifact_path:
        try:
            candidate_version = artifact_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        except Exception:
            pass

    adjustment_execution_result = {
        "status": callback_status,
        "artifact_uri": calibrator_uri,
        "artifact_checksum": resume_data.get("artifact_checksum"),
        "metrics": resume_data.get("metrics") or {},
        "consumption_receipt": resume_data.get("consumption_receipt") or {},
    }

    return {
        "calibration_plan_id": plan["calibration_plan_id"],
        "challenger_version": candidate_version,
        "calibrator_artifact_uri": calibrator_uri,
        "adjustment_artifact_checksum": adjustment_execution_result["artifact_checksum"],
        "adjustment_execution_result": adjustment_execution_result,
        "current_phase": LifecyclePhase.OFFLINE_VALIDATING.value,
    }


async def threshold_plan_node(state: ModelLifecycleState) -> dict:
    """P3 阈值调整执行器 — 创建阈值搜索计划 → 派发 Worker → interrupt 等待回调。"""
    from .executors import create_threshold_plan, dispatch_external_execution
    from ...config import settings

    try:
        plan = create_threshold_plan(_state_dict(state))
    except Exception as e:
        logger.error("threshold_plan_creation_failed", exc_info=True)
        return {
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "threshold_plan_creation_error",
                "message": f"阈值计划创建失败：{e}",
                "at": _now_iso(),
            },
        }

    dispatch: dict = {"dispatch_mode": "INTERNAL"}
    try:
        dispatch = dispatch_external_execution("THRESHOLD", plan)
        if dispatch.get("dispatched"):
            plan["status"] = "DISPATCHED"
        elif settings.workflow_use_celery:
            try:
                from workers.app import app as celery_app
                task = celery_app.send_task("workers.executor_tasks.search_threshold", args=[plan])
                dispatch = {
                    "dispatched": True,
                    "dispatch_mode": "CELERY",
                    "external_task_id": getattr(task, "id", None),
                }
                plan["status"] = "DISPATCHED"
            except Exception as celery_exc:
                dispatch = {"dispatch_mode": "CELERY", "error": str(celery_exc)}
                logger.warning("threshold_celery_dispatch_failed", exc_info=True)
    except Exception as exc:
        dispatch = {"dispatch_mode": "EXTERNAL_OR_CELERY", "error": str(exc)}
        logger.warning("threshold_external_dispatch_failed",
                       threshold_plan_id=plan["threshold_plan_id"], exc_info=True)

    await _save_external_plan("THRESHOLD", plan, dispatch)

    if not dispatch.get("dispatched"):
        logger.error("threshold_plan_dispatch_blocked")
        return {
            "threshold_plan_id": plan["threshold_plan_id"],
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "threshold_dispatch_failed",
                "message": "阈值计划派发失败，外部执行器和 Celery 均不可用",
                "at": _now_iso(),
            },
        }

    # interrupt 等待 Worker 回调（fail-closed：无 status 不得默认成功）
    resume_data = interrupt("waiting_threshold_callback")
    if not isinstance(resume_data, dict) or not resume_data.get("status"):
        return {
            "threshold_plan_id": plan["threshold_plan_id"],
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "threshold_callback_contract_invalid",
                "message": "阈值回调缺少 status 字段，按失败处理",
                "at": _now_iso(),
            },
        }
    callback_status = str(resume_data["status"]).upper()
    # Worker 回调字段为 artifact_uri（与包版执行合同一致）
    threshold_uri = resume_data.get("artifact_uri") or resume_data.get(
        "threshold_artifact_uri"
    )

    if callback_status != "SUCCEEDED":
        logger.error("threshold_worker_failed", plan_id=plan["threshold_plan_id"])
        return {
            "threshold_plan_id": plan["threshold_plan_id"],
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "threshold_worker_failed",
                "message": (
                    resume_data.get("error_message") or "阈值 Worker 执行失败"
                ),
                "at": _now_iso(),
            },
        }

    # 安全解析 challenger_version
    artifact_path: str = str(plan.get("artifact_output_path", ""))
    candidate_version = str(plan.get("champion_version", "v1"))
    if artifact_path:
        try:
            candidate_version = artifact_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        except Exception:
            pass

    adjustment_execution_result = {
        "status": callback_status,
        "artifact_uri": threshold_uri,
        "artifact_checksum": resume_data.get("artifact_checksum"),
        "metrics": resume_data.get("metrics") or {},
        "consumption_receipt": resume_data.get("consumption_receipt") or {},
    }

    return {
        "threshold_plan_id": plan["threshold_plan_id"],
        "challenger_version": candidate_version,
        "threshold_artifact_uri": threshold_uri,
        "adjustment_artifact_checksum": adjustment_execution_result["artifact_checksum"],
        "adjustment_execution_result": adjustment_execution_result,
        "current_phase": LifecyclePhase.OFFLINE_VALIDATING.value,
    }


# ═══════════════════════════════════════════════════════════
# T3-GAP-01：特征重构节点
# ═══════════════════════════════════════════════════════════

async def feature_reconstruction_node(state: ModelLifecycleState) -> dict:
    """T3-GAP-01: 特征重构节点。

    在 TrainingPlan 之前执行，根据诊断结果决定增/删/改特征。
    产出 feature_schema_version / feature_snapshot_id / transform_artifact_uri，
    供 TrainingPlanNode 使用。

    流程：
    1. 读取诊断证据（PSI 漂移、缺失率）
    2. FeatureReconstructionService 生成 Plan
    3. Demo 模式：内联执行变换
    4. Celery 模式：派发到 Worker
    """
    from ...services.iteration.feature_reconstruction_service import FeatureReconstructionService
    from ...config import settings

    state_dict = _state_dict(state)
    diagnosis_run_id = _g(state, "diagnosis_run_id")
    model_id = _g(state, "model_id", "")
    lifecycle_run_id = _g(state, "lifecycle_run_id")

    # 1. 从诊断结果提取证据
    drift_features: list[dict] = state_dict.get("drift_features", [])
    high_missing_features: list[dict] = state_dict.get("high_missing_features", [])
    current_feature_names: list[str] = state_dict.get("feature_names", [])
    feature_importance: dict[str, float] = state_dict.get("feature_importance", {})
    skewness: dict[str, float] = state_dict.get("skewness", {})

    # 如果没有 drift 信息但有 diagnosis_run_id，尝试从 DB 加载
    if not drift_features and diagnosis_run_id:
        try:
            from ...database import async_session
            from ...repositories.diagnosis_repo import DiagnosisRepo

            async with async_session() as session:
                repo = DiagnosisRepo(session)
                drift_records = await repo.get_drift_features(diagnosis_run_id)
                if drift_records:
                    drift_features = [
                        {"feature_name": r.get("feature_name", ""), "psi_value": r.get("psi_value", 0)}
                        for r in drift_records
                    ]
                    logger.info(
                        "feature_recon_loaded_drift",
                        diagnosis_run_id=diagnosis_run_id,
                        count=len(drift_features),
                    )
        except Exception:
            logger.warning("feature_recon_drift_load_failed", exc_info=True)

    # 2. 生成重构计划
    svc = FeatureReconstructionService()
    current_schema = _g(state, "feature_schema_version") or "v1"
    # 从 IterationDecisionNode 获取 KG 推荐的算法族
    algorithm = _g(state, "algorithm") or "lightgbm"
    plan = svc.build_plan(
        model_id=model_id,
        lifecycle_run_id=lifecycle_run_id,
        diagnosis_run_id=diagnosis_run_id,
        current_schema_version=current_schema,
        drift_features=drift_features,
        high_missing_features=high_missing_features,
        current_feature_names=current_feature_names,
        feature_importance=feature_importance,
        skewness=skewness,
        algorithm=algorithm,
    )

    logger.info(
        "feature_reconstruction_plan_created",
        plan_id=plan.plan_id,
        transforms=len(plan.transforms),
        before=plan.expected_feature_count_before,
        after=plan.expected_feature_count_after,
    )

    try:
        from ...database import async_session
        from ...repositories.iteration_repo import IterationRepo

        async with async_session() as session:
            await IterationRepo(session).save_feature_reconstruction_plan(plan)
            await session.commit()
    except Exception:
        logger.warning(
            "feature_reconstruction_plan_persist_failed",
            plan_id=plan.plan_id,
            exc_info=True,
        )

    # 3. 执行（Demo: 内联；Celery: 派发）
    worker_dispatched = False
    if settings.workflow_use_celery and plan.transforms:
        try:
            from workers.app import app as celery_app
            celery_app.send_task(
                "workers.feature_tasks.reconstruct_features",
                args=[{
                    "plan_id": plan.plan_id,
                    "lifecycle_run_id": lifecycle_run_id,
                    "model_id": model_id,
                    "transforms": [t.model_dump() for t in plan.transforms],
                    "current_schema_version": plan.current_schema_version,
                    "target_schema_version": plan.target_schema_version,
                    "window_ids": ["W2", "W3"],
                }],
            )
            worker_dispatched = True
            logger.info("feature_recon_dispatched_to_celery", plan_id=plan.plan_id)
        except Exception:
            logger.warning("feature_recon_celery_dispatch_failed", exc_info=True)

    if worker_dispatched:
        return {
            "feature_reconstruction_plan_id": plan.plan_id,
            "feature_reconstruction_status": "DISPATCHED",
            "feature_reconstruction_dispatched": True,
            "feature_transform_count": len(plan.transforms),
            "current_phase": LifecyclePhase.WAITING_FEATURE_RECONSTRUCTION.value,
        }

    # Celery 不可用且有变换需要执行 → FAILED
    if plan.transforms:
        logger.error(
            "feature_reconstruction_blocked_no_celery — Celery 不可用，无法执行特征重构"
        )
        return {
            "feature_reconstruction_plan_id": plan.plan_id,
            "feature_reconstruction_status": "FAILED",
            "feature_reconstruction_dispatched": False,
            "feature_transform_count": len(plan.transforms),
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "feature_reconstruction_worker_unavailable",
                "message": "Celery Worker 不可用，特征重构无法执行",
                "at": _now_iso(),
            },
        }

    # 无变换 → 跳过
    return {
        "feature_reconstruction_plan_id": plan.plan_id,
        "feature_reconstruction_status": "SKIPPED_NO_TRANSFORMS",
        "feature_reconstruction_dispatched": False,
        "feature_schema_version": plan.current_schema_version,
        "feature_transform_count": 0,
        "current_phase": LifecyclePhase.ITERATING.value,
    }


async def wait_feature_reconstruction_node(state: ModelLifecycleState) -> dict:
    """Wait for the feature reconstruction worker callback before training planning."""
    resume_data = interrupt("waiting_feature_reconstruction")

    if not isinstance(resume_data, dict):
        return {
            "feature_reconstruction_status": "FAILED",
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "invalid_feature_reconstruction_resume_payload",
                "at": _now_iso(),
            },
        }

    callback_status = str(resume_data.get("status") or "SUCCEEDED").upper()
    plan_id = resume_data.get("feature_reconstruction_plan_id")
    if plan_id and plan_id != _g(state, "feature_reconstruction_plan_id"):
        return {
            "feature_reconstruction_status": "FAILED",
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "feature_reconstruction_plan_id_mismatch",
                "at": _now_iso(),
            },
        }

    if callback_status != "SUCCEEDED":
        return {
            "feature_reconstruction_status": callback_status,
            "iteration_exit_reason": "TECHNICAL_FAILURE",
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "feature_reconstruction_failed",
                "message": resume_data.get("error_message"),
                "at": _now_iso(),
            },
        }

    return {
        "feature_reconstruction_status": "SUCCEEDED",
        "feature_reconstruction_dispatched": False,
        "feature_schema_version": resume_data.get("feature_schema_version") or _g(state, "feature_schema_version"),
        "feature_snapshot_id": resume_data.get("feature_snapshot_id") or _g(state, "feature_snapshot_id"),
        "transform_artifact_uri": resume_data.get("transform_artifact_uri") or _g(state, "transform_artifact_uri"),
        "current_phase": LifecyclePhase.ITERATING.value,
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
        # A7 定稿 §3 节点合同 requires_manual_approval=false：
        # 低风险提案（L1 + 风险评估均未要求人工复核）由系统自动批准，
        # 生成 AUTO_RULE 批准记录（可审计）；高风险（full_retrain/SEVERE/
        # 需要人工的提案）在路由层已去 ManualReviewNode，不会走到这里。
        if _g(state, "requires_manual_review", False):
            logger.warning(
                "training_plan_missing_manual_review_id", proposal_id=proposal_id
            )
            return {
                "requires_manual_review": True,
                "current_phase": LifecyclePhase.MANUAL_REVIEW.value,
            }
        logger.info(
            "training_plan_auto_approval",
            proposal_id=proposal_id,
            message="低风险提案自动批准（A7 定稿 requires_manual_approval=false）",
        )
    try:
        from ...database import async_session
        from ...repositories.iteration_repo import IterationRepo
        from ...services.iteration import RiskAssessmentService, TrainingPlanBuilder
        from ...services.iteration.config_loader import load_iteration_config
        from packages.models.common.enums import ProposalStatus
        from packages.models.iteration import ManualReviewReport
        from packages.models.common.enums import ReviewDecision

        async with async_session() as session:
            repo = IterationRepo(session)
            proposal = await repo.get_proposal(proposal_id)
            if proposal is None:
                return {"current_phase": LifecyclePhase.MANUAL_REVIEW.value}

            if not manual_review_id:
                # 自动批准：系统生成批准记录（reviewer=system-auto-approver，
                # decision=APPROVE，全程落库可审计）
                auto_report = ManualReviewReport(
                    review_id=str(uuid.uuid4()),
                    proposal_id=proposal_id,
                    reviewer_id="system-auto-approver",
                    decision=ReviewDecision.APPROVE,
                    reason=(
                        "AUTO_APPROVAL:L1_AND_RISK_ASSESSMENT_NO_MANUAL_REVIEW_"
                        "REQUIRED;A7_REQUIRES_MANUAL_APPROVAL_FALSE"
                    ),
                    reviewed_at=datetime.now(timezone.utc),
                )
                await repo.save_review(auto_report)
                manual_review_id = auto_report.review_id
                await session.commit()
                logger.info(
                    "training_plan_auto_approval_created",
                    proposal_id=proposal_id,
                    review_id=manual_review_id,
                )

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
            iteration_config = load_iteration_config()
            # A7 §5: 第二轮复用同一 iteration_run_id（业务轮次切换，非新迭代）
            iteration_run_id = _g(state, "iteration_run_id") or str(uuid.uuid4())
            business_round = _g(state, "business_round") or 1

            # T3-GAP-01: 使用特征重构产出的 schema_version
            recon_schema = _g(state, "feature_schema_version")
            recon_snapshot = _g(state, "feature_snapshot_id")

            plan_warnings: list[str] = []
            snapshot_ids = []
            if recon_snapshot:
                snapshot_ids = [recon_snapshot]
            else:
                # 无重构快照 → Worker 将回退到 load_window() 直接读窗口数据
                plan_warnings.append(
                    "无特征重构快照，Worker 将直接加载 W2/W3 窗口数据训练"
                )

            # 从 State 提取 IterationDecisionNode 写入的 algorithm
            algorithm = _g(state, "algorithm") or None
            # 从 config 拿 max_rounds，不用硬编码 3
            max_rounds = iteration_config.iteration.max_iteration_rounds

            plan = TrainingPlanBuilder().build(
                approved_proposal,
                risk,
                approval_id=manual_review_id,
                iteration_run_id=iteration_run_id,
                feature_schema_version=recon_schema,
                model_algorithm=algorithm,
                business_round=business_round,
                data_snapshot_ids=snapshot_ids if snapshot_ids else None,
                # A7 阶段四：特征筛选产物（FEATURE_SELECTION 模式）
                unstable_feature_codes=[
                    str(c).strip()
                    for c in (_g(state, "unstable_feature_codes") or "").split(",")
                    if str(c).strip()
                ],
                selected_feature_codes=[
                    str(c).strip()
                    for c in (_g(state, "selected_feature_codes") or "").split(",")
                    if str(c).strip()
                ],
                feature_selection_artifact_uri=(
                    _g(state, "feature_selection_artifact_uri")
                ),
            )

            if business_round == 1:
                await repo.create_iteration_run(
                    iteration_run_id,
                    approved_proposal,
                    max_rounds,
                )
            else:
                # 第二轮复用同一 iteration_run_id：只更新当前 Proposal 引用，
                # 不重复 INSERT（否则主键冲突），也不覆盖首轮冻结信息
                await repo.update_iteration_run_proposal(
                    iteration_run_id, approved_proposal.proposal_id,
                )
            await repo.save_training_plan(plan)
            await repo.create_round_and_experiment(plan)
            await session.commit()

            logger.info(
                "training_plan_node_completed",
                training_plan_id=plan.training_plan_id,
                iteration_run_id=iteration_run_id,
                business_round=business_round,
                algorithm=algorithm,
                feature_schema_version=recon_schema,
                max_rounds=max_rounds,
            )

            return_dict: dict = {
                "training_plan_id": plan.training_plan_id,
                "iteration_run_id": iteration_run_id,
                "experiment_id": plan.experiment_id,
                "business_round": business_round,
                # A7 §5: 人工批准由真实 Review 记录推导（此处必已 APPROVED）
                "manual_approval": True,
                "current_phase": LifecyclePhase.ITERATING.value,
            }
            if plan_warnings:
                return_dict["warnings"] = plan_warnings
            return return_dict
    except (OSError, ConnectionError, TimeoutError, _DBIntegrityError):
        logger.warning("training_plan_fallback", exc_info=True)
        return {"current_phase": LifecyclePhase.MANUAL_REVIEW.value}
    except Exception as e:
        logger.error("training_plan_unexpected_error", exc_info=True)
        return {
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "training_plan_unexpected_error",
                "message": f"训练计划生成发生未预期错误：{e}",
                "at": _now_iso(),
            },
        }


# ═══════════════════════════════════════════════════════════
# T3-GAP-02：超参优化节点
# ═══════════════════════════════════════════════════════════

async def hyperparameter_tuning_node(state: ModelLifecycleState) -> dict:
    """T3-GAP-02: 超参优化节点。

    在 TrainingPlan 之后、TrainingJobDispatch 之前执行。
    生成 N 组候选超参 → Worker 并行训练 trial → 选出 best_params。
    """
    from ...services.iteration.hyperparameter_tuning_service import HyperparameterTuningService
    from ...config import settings

    state_dict = _state_dict(state)
    model_id = _g(state, "model_id", "")
    lifecycle_run_id = _g(state, "lifecycle_run_id")
    training_plan_id = _g(state, "training_plan_id")
    algorithm: str | None = state_dict.get("algorithm")
    base_params: dict = {}
    seed = int(state_dict.get("seed") or 2026)
    training_window_ids = ["W2"]
    validation_window_ids = ["W3"]

    try:
        from ...database import async_session
        from ...repositories.iteration_repo import IterationRepo
        from packages.models.iteration.training_plan import TrainingPlan

        async with async_session() as session:
            plan_payload = await IterationRepo(session).get_training_plan(training_plan_id)
            if plan_payload:
                training_plan = TrainingPlan.model_validate(plan_payload)
                algorithm = training_plan.algorithm or algorithm
                base_params = training_plan.hyperparameter_space or {}
                seed = training_plan.random_seed
                training_window_ids = training_plan.windows.training_window_ids
                validation_window_ids = training_plan.windows.validation_window_ids
    except Exception:
        logger.warning(
            "tuning_load_training_plan_failed",
            training_plan_id=training_plan_id,
            exc_info=True,
        )

    if not algorithm:
        logger.error("hyperparameter_tuning_missing_algorithm — State 和 TrainingPlan 均未提供 algorithm")
        return {
            "iteration_exit_reason": "MISSING_ALGORITHM_FOR_TUNING",
            "current_phase": LifecyclePhase.FAILED.value,
            "warnings": ["State 和 TrainingPlan 中均无 algorithm 字段，无法生成超参搜索计划"],
        }

    # 1. 生成搜索计划
    # 按 strategy_tier 调整调参强度: full=10, light=5, minimal/其他=3
    strategy_tier = state_dict.get("strategy_tier", "full")
    tier_trials: dict[str, int] = {"full": 10, "light": 5, "minimal": 3, "tune_only": 8}
    num_trials = tier_trials.get(strategy_tier, 5) if strategy_tier else 5

    svc = HyperparameterTuningService()
    plan = svc.build_plan(
        model_id=model_id,
        lifecycle_run_id=lifecycle_run_id,
        training_plan_id=training_plan_id,
        algorithm=algorithm,
        num_trials=num_trials,
        seed=seed,
        base_params=base_params,
    )

    logger.info(
        "hyperparameter_tuning_plan_created",
        plan_id=plan.plan_id,
        algorithm=algorithm,
        num_trials=plan.num_trials,
        strategy_tier=strategy_tier,
    )

    # 2. 派发到 Celery Worker
    worker_dispatched = False
    if settings.workflow_use_celery:
        try:
            from workers.app import app as celery_app
            try:
                from ...database import async_session
                from ...repositories.iteration_repo import IterationRepo
                async with async_session() as session:
                    await IterationRepo(session).save_tuning_plan(plan)
                    await session.commit()
            except Exception:
                logger.warning("tuning_plan_persist_failed", plan_id=plan.plan_id, exc_info=True)
            celery_app.send_task(
                "workers.tuning_tasks.run_tuning",
                args=[{
                    "plan_id": plan.plan_id,
                    "lifecycle_run_id": lifecycle_run_id,
                    "training_plan_id": training_plan_id,
                    "algorithm": algorithm,
                    "training_window_ids": training_window_ids,
                    "validation_window_ids": validation_window_ids,
                    "trials": [t.model_dump() for t in plan.trials],
                    "seed": seed,
                }],
            )
            worker_dispatched = True
            logger.info("tuning_dispatched_to_celery", plan_id=plan.plan_id)
        except Exception:
            logger.warning("tuning_celery_dispatch_failed", exc_info=True)

    # 3. Celery 不可用 → 不造假，直接 FAILED
    if not worker_dispatched:
        logger.error(
            "hyperparameter_tuning_blocked_no_celery — Celery 不可用，无法执行真实超参搜索"
        )
        return {
            "hyperparameter_tuning_plan_id": plan.plan_id,
            "tuning_dispatched": False,
            "tuning_completed": False,
            "iteration_exit_reason": "TUNING_WORKER_UNAVAILABLE",
            "current_phase": LifecyclePhase.FAILED.value,
            "warnings": [
                "Celery Worker 不可用，超参优化无法执行。"
                "请启动 Celery Worker（tuning_tasks）或设置 WORKFLOW_USE_CELERY=true"
            ],
        }

    # Celery 模式：等待 Worker 回调
    return {
        "hyperparameter_tuning_plan_id": plan.plan_id,
        "tuning_dispatched": True,
        "tuning_completed": False,
        "current_phase": LifecyclePhase.ITERATING.value,
    }


async def wait_tuning_callback_node(state: ModelLifecycleState) -> dict:
    """等待超参优化 Worker 回调。"""
    resume_data = interrupt("waiting_tuning_callback")
    result: dict = {}
    if isinstance(resume_data, dict):
        result = resume_data
    elif isinstance(resume_data, str):
        result = {"status": resume_data}

    logger.info(
        "wait_tuning_callback_resumed",
        plan_id=_g(state, "hyperparameter_tuning_plan_id"),
        status=result.get("status"),
    )
    return {
        "best_hyperparameters": result.get("best_hyperparameters", {}),
        "best_tuning_metric": result.get("best_val_auc"),
        "tuning_completed": True,
        "current_phase": LifecyclePhase.ITERATING.value,
    }


# ═══════════════════════════════════════════════════════════
# P2：异步训练接入
# ═══════════════════════════════════════════════════════════

async def training_job_dispatch_node(state: ModelLifecycleState) -> dict:
    """P2 训练任务派发节点。

    创建训练任务 → 提交 Celery Worker 真实训练
    State 写入 training_job_id → 图暂停在 WAITING_TRAINING_CALLBACK
    """
    iteration_run_id = _g(state, "iteration_run_id")
    training_plan_id = _g(state, "training_plan_id")
    experiment_id = _g(state, "experiment_id")
    business_round = _g(state, "business_round") or 1

    missing = []
    if not iteration_run_id:
        missing.append("iteration_run_id")
    if not training_plan_id:
        missing.append("training_plan_id")
    if not experiment_id:
        missing.append("experiment_id")
    if missing:
        return {
            "iteration_exit_reason": f"MISSING_UPSTREAM_IDS: {', '.join(missing)}",
            "current_phase": LifecyclePhase.FAILED.value,
            "warnings": [f"上游 training_plan_node 未写入: {', '.join(missing)}，条件边可能路由错误"],
        }

    training_job_id = str(uuid.uuid4())

    try:
        from ...database import async_session
        from ...repositories.iteration_repo import IterationRepo
        from packages.models.iteration.training_plan import TrainingPlan
        from packages.models.iteration.training_job import TrainingJobInput

        async with async_session() as session:
            repo = IterationRepo(session)
            plan_payload = await repo.get_training_plan(training_plan_id)
            plan = TrainingPlan.model_validate(plan_payload) if plan_payload else None

            tuned_hyperparameters = _g(state, "best_hyperparameters") or {}
            plan_hyperparameters = plan.hyperparameter_space if plan else {}
            final_hyperparameters = tuned_hyperparameters or plan_hyperparameters or {}

            job_input = TrainingJobInput(
                training_job_id=training_job_id,
                idempotency_key=f"{iteration_run_id}:round-{business_round}:exp-{experiment_id}",
                model_id=plan.model_id if plan else _g(state, "model_id", ""),
                iteration_run_id=iteration_run_id,
                training_plan_id=training_plan_id,
                experiment_id=experiment_id,
                business_round=business_round,
                strategy_code=plan.strategy_code if plan else "PLAN_STABLE_REFIT",
                training_window_ids=plan.windows.training_window_ids if plan else ["W2"],
                validation_window_ids=plan.windows.validation_window_ids if plan else ["W3"],
                train_time_ranges=[],
                validation_time_ranges=[],
                oot_window_id=plan.windows.oot_window_id if plan else "W4",
                data_snapshot_ids=plan.data_snapshot_ids if plan else [f"snapshot-w2-v{business_round}"],
                label_versions=plan.label_versions if plan else ["label-v1"],
                sample_weight_policy=plan.sample_weight_policy if plan else {},
                feature_schema_version=plan.feature_schema_version if plan else "feature-schema-v1",
                preprocessing_version=plan.preprocessing_version if plan else "preprocess-v1",
                algorithm=plan.algorithm if plan else "lightgbm",
                hyperparameters=final_hyperparameters,
                target_metrics=plan.target_metric_codes if plan else ["AUC", "KS"],
                qualification_rule_version=plan.qualification_rule_version if plan else "qualification-rules-v1",
                base_model_version=plan.frozen_champion_version if plan else _g(state, "champion_version"),
                seed=plan.random_seed if plan else 2026,
                training_mode=_g(state, "training_mode") or "FULL_RETRAIN",
                # A7 阶段四：特征筛选合同传递（TrainingPlan → TrainingJob）
                unstable_feature_codes=(
                    plan.unstable_feature_codes if plan else []
                ),
                selected_feature_codes=(
                    plan.selected_feature_codes if plan else []
                ),
                feature_selection_artifact_uri=(
                    plan.feature_selection_artifact_uri if plan else None
                ),
                artifact_output_uri=(
                    f"s3://riskitem/challengers/{plan.model_id if plan else _g(state, 'model_id', 'unknown')}"
                    f"/{iteration_run_id}/round-{business_round}"
                ),
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

            if not dispatch_result["dispatched"]:
                logger.error(
                    "training_job_dispatch_blocked_no_celery — Celery 不可用，无法派发训练任务"
                )
                return {
                    "training_job_id": training_job_id,
                    "training_dispatched": False,
                    "training_dispatch_mode": "none",
                    "current_phase": LifecyclePhase.FAILED.value,
                    "iteration_exit_reason": "TRAINING_DISPATCH_FAILED",
                    "last_error": {
                        "reason": "training_dispatch_worker_unavailable",
                        "message": "Celery Worker 不可用，训练任务无法派发",
                        "at": _now_iso(),
                    },
                }

            return {
                "training_job_id": training_job_id,
                "training_dispatched": dispatch_result["dispatched"],
                "training_dispatch_mode": "celery",
                "current_phase": LifecyclePhase.WAITING_TRAINING_CALLBACK.value,
            }
    except (OSError, ConnectionError, TimeoutError, _DBIntegrityError) as e:
        logger.error("training_job_dispatch_infra_error", exc_info=True)
        return {
            "training_job_id": training_job_id,
            "training_dispatched": False,
            "training_dispatch_mode": "none",
            "current_phase": LifecyclePhase.FAILED.value,
            "iteration_exit_reason": "TRAINING_DISPATCH_INFRA_ERROR",
            "last_error": {
                "reason": "training_dispatch_infra_error",
                "message": f"数据库或基础设施不可用，训练任务无法派发：{e}",
                "at": _now_iso(),
            },
        }
    except Exception as e:
        logger.error("training_job_dispatch_unexpected_error", exc_info=True)
        return {
            "training_job_id": training_job_id,
            "training_dispatched": False,
            "training_dispatch_mode": "none",
            "current_phase": LifecyclePhase.FAILED.value,
            "iteration_exit_reason": "TRAINING_DISPATCH_ERROR",
            "last_error": {
                "reason": "training_dispatch_unexpected_error",
                "message": f"训练任务派发发生未预期错误：{e}",
                "at": _now_iso(),
            },
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
    """Normalize worker callback result before QualificationNode.成功 → 进入资格验证，失败 → 标记流程失败"""
    callback_status = _g(state, "training_callback_status")
    if not callback_status:
        return {
            "iteration_exit_reason": "TRAINING_CALLBACK_STATUS_MISSING",
            "current_phase": LifecyclePhase.FAILED.value,
        }
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
            QualificationInput,
        )

        async with async_session() as session:
            repo = IterationRepo(session)
            svc = QualificationService()
            experiment = await repo.get_experiment(experiment_id) if experiment_id else None
            action = _recommended_action(state)
            #逻辑链：校准器/阈值调整 ≠ 换模型 → 不需要验证 Challenger 的排序能力 → 直接通过。
            #KG 判断根因是"概率失真"或"阈值偏移"，不需要重新训练模型——只需调整校准器或阈值。这两种情况没有 experiment（没跑训练）
            if experiment is None and action in {
                AgentDecisionAction.CALIBRATION_ADJUSTMENT.value,
                AgentDecisionAction.THRESHOLD_ADJUSTMENT.value,
            }:
                # 轻量调整作用于现有 champion，不产生新模型 artifact。
                # 资格由真实执行产物判定（qualify_adjustment）：没有执行结果
                # 不得直接通过（fail-closed，杜绝"假通过"）。
                champion_version = _g(state, "champion_version")
                if not champion_version:
                    return {
                        "iteration_exit_reason": "MISSING_CHAMPION_VERSION",
                        "current_phase": LifecyclePhase.FAILED.value,
                    }
                candidate_version = _g(state, "challenger_version") or champion_version
                report_id = str(uuid.uuid4())
                adjustment_result = _g(state, "adjustment_execution_result") or {}
                if not adjustment_result:
                    logger.warning(
                        "qualification_node_adjustment_result_missing",
                        action=action,
                    )
                    return {
                        "qualification_run_id": report_id,
                        "challenger_version": candidate_version,
                        "challenger_qualified": False,
                        "iteration_exit_reason": "ADJUSTMENT_EXECUTION_RESULT_MISSING",
                        "current_phase": LifecyclePhase.QUALIFICATION_COMPLETED.value,
                    }
                from ...services.iteration.action_execution_service import (
                    qualify_adjustment,
                )
                qualified, reasons = qualify_adjustment(
                    action, adjustment_result
                )
                logger.info(
                    "qualification_node_lightweight_adjustment",
                    qualification_run_id=report_id,
                    recommended_action=action,
                    candidate_version=candidate_version,
                    qualified=qualified,
                    reasons=reasons,
                )
                return {
                    "qualification_run_id": report_id,
                    "challenger_version": candidate_version,
                    "challenger_qualified": qualified,
                    "iteration_exit_reason": (
                        None if qualified else "ADJUSTMENT_QUALIFICATION_FAILED"
                    ),
                    "current_phase": LifecyclePhase.QUALIFICATION_COMPLETED.value,
                }
            if experiment is None or experiment.get("technical_status") != "SUCCEEDED":
                return {
                    "iteration_exit_reason": "TECHNICAL_FAILURE",
                    "current_phase": LifecyclePhase.FAILED.value,
                }
            candidate_version = _g(state, "challenger_version") or experiment.get("candidate_version")
            if not candidate_version:
                return {
                    "iteration_exit_reason": "MISSING_CANDIDATE_VERSION",
                    "current_phase": LifecyclePhase.FAILED.value,
                    "warnings": ["State 和 experiment 中均无 candidate_version，无法进行资格验证——Worker 回调可能未正确写入"],
                }
            experiment_json = experiment.get("experiment_json") or {}
            validation_metrics = experiment_json.get("validation_metrics") or {}
            segment_metrics = experiment_json.get("segment_metrics") or {}

            # 从 validation_metrics 提取真实值，字段缺失时不使用虚构默认值
            recovery_auc = validation_metrics.get("recovery_auc")
            recovery_ks = validation_metrics.get("recovery_ks")
            score_psi = validation_metrics.get("score_psi")

            # 核心性能 Gate: AUC/KS 恢复率 + PSI 稳定性全部满足
            core_perf_passed = (
                (recovery_auc is not None and recovery_auc >= 1.0)
                or (recovery_ks is not None and recovery_ks >= 1.0)
            ) if (recovery_auc is not None or recovery_ks is not None) else False

            # 核心性能未达标 → 直接不合格，不进入七道 Gate。
            # 仍落一份真实资格报告（TARGET_RECOVERY FAILED），
            # 失败归因节点需要它才能生成第二轮证据——不落报告会
            # 导致 failure_analysis 报 QUALIFICATION_REPORT_NOT_FOUND。
            if not core_perf_passed:
                report_id = str(uuid.uuid4())
                logger.warning(
                    "qualification_node_core_perf_failed",
                    qualification_run_id=report_id,
                    recovery_auc=recovery_auc,
                    recovery_ks=recovery_ks,
                    score_psi=score_psi,
                )
                from packages.models.common.enums import (
                    QualificationGateCode,
                    QualificationStatus,
                )
                from packages.models.iteration.qualification import (
                    QualificationGateResult,
                    QualificationReport,
                )
                from ...services.iteration.config_loader import (
                    load_iteration_config,
                )

                try:
                    rule_version = load_iteration_config().qualification.rule_version
                except Exception:
                    rule_version = "qualification-rules-v2"
                await repo.save_qualification(
                    QualificationReport(
                        qualification_run_id=report_id,
                        iteration_run_id=iteration_run_id or "",
                        experiment_id=experiment_id or "",
                        candidate_version=candidate_version,
                        status=QualificationStatus.FAILED,
                        qualified=False,
                        gate_results=[
                            QualificationGateResult(
                                gate_code=QualificationGateCode.TARGET_RECOVERY,
                                gate_order=1,
                                status=QualificationStatus.FAILED,
                                required=True,
                                actual={},
                                metrics={
                                    "AUC": {
                                        "recovery_rate": recovery_auc,
                                        "healthy_range_reached": False,
                                    },
                                    "KS": {
                                        "recovery_rate": recovery_ks,
                                        "healthy_range_reached": False,
                                    },
                                },
                                reasons=[
                                    f"RECOVERY_RATE_FAILED:{code}"
                                    for code, value in (
                                        ("AUC", recovery_auc),
                                        ("KS", recovery_ks),
                                    )
                                    if value is not None
                                ],
                            )
                        ],
                        failed_gate_codes=[
                            QualificationGateCode.TARGET_RECOVERY,
                        ],
                        rule_version=rule_version,
                        qualification_stage="PRE_OOT",
                        allow_w4=False,
                    )
                )
                await session.commit()
                return {
                    "qualification_run_id": report_id,
                    "challenger_version": candidate_version,
                    "challenger_qualified": False,
                    "iteration_exit_reason": "CORE_PERFORMANCE_GATE_FAILED",
                    "current_phase": LifecyclePhase.OFFLINE_VALIDATING.value,
                }

            # 从 validation_metrics 构建真实 MetricComparison 列表
            # A7 §5: 特征级 PSI 来自真实监测漂移数据，
            # 供 STABILITY 门生成结构化 unstable_feature_codes
            feature_psi: dict[str, float] = {}
            if _g(state, "monitoring_run_id"):
                from ...repositories.monitoring_repo import MonitoringRepo
                drift_rows = await MonitoringRepo(session).get_feature_drift_by_run(
                    _g(state, "monitoring_run_id")
                )
                for row in drift_rows:
                    fname = row.get("feature_name")
                    psi = row.get("psi")
                    if fname and psi is not None:
                        feature_psi[str(fname)] = max(
                            feature_psi.get(str(fname), 0.0), float(psi)
                        )

            # 共享构建入口：Graph 与外部资格端点使用同一套逻辑，
            # 必填证据缺失时拒绝评估（禁止 fail-open）
            from ...services.iteration.qualification_service import (
                QualificationEvidenceIncompleteError,
                build_qualification_input,
            )
            try:
                qual_input = build_qualification_input(
                    qualification_run_id=str(uuid.uuid4()),
                    iteration_run_id=iteration_run_id or "",
                    experiment_id=experiment_id or "",
                    candidate_version=candidate_version,
                    experiment_json=experiment_json,
                    feature_psi=feature_psi,
                    include_oot=False,  # W3 预资格：OOT 门在 W4 完成后单独重跑
                )
            except QualificationEvidenceIncompleteError as evidence_exc:
                logger.warning(
                    "qualification_evidence_incomplete",
                    missing_fields=evidence_exc.missing_fields,
                )
                return {
                    "qualification_run_id": str(uuid.uuid4()),
                    "challenger_qualified": False,
                    "iteration_exit_reason": "QUALIFICATION_EVIDENCE_INCOMPLETE",
                    "current_phase": LifecyclePhase.OFFLINE_VALIDATING.value,
                    "last_error": {
                        "reason": "QUALIFICATION_EVIDENCE_INCOMPLETE",
                        "message": str(evidence_exc),
                        "at": _now_iso(),
                    },
                }

            report = svc.evaluate(qual_input, include_oot=False)
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
                "validation_metrics": validation_metrics,
                "current_phase": (
                    LifecyclePhase.QUALIFICATION_COMPLETED.value
                    if report.qualified
                    else LifecyclePhase.OFFLINE_VALIDATING.value
                ),
            }
    except (OSError, ConnectionError, TimeoutError, _DBIntegrityError) as e:
        logger.error("qualification_node_infra_error", exc_info=True)
        return {
            "qualification_run_id": str(uuid.uuid4()),
            "challenger_qualified": False,
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "qualification_infra_error",
                "message": f"数据库不可用，资格验证无法执行：{e}",
                "at": _now_iso(),
            },
        }
    except Exception as e:
        logger.error("qualification_node_unexpected_error", exc_info=True)
        return {
            "qualification_run_id": str(uuid.uuid4()),
            "challenger_qualified": False,
            "current_phase": LifecyclePhase.FAILED.value,
            "last_error": {
                "reason": "qualification_unexpected_error",
                "message": f"资格验证发生未预期错误：{e}",
                "at": _now_iso(),
            },
        }


async def failure_analysis_node(state: ModelLifecycleState) -> dict:
    """W3 失败归因：调用真实 FailureAttributionService，持久化失败报告。

    归因报告是第二轮证据（unstable_feature_subset_confirmed /
    feature_selection_evidence_available）的真实来源，不能用随机 ID 代替。
    """
    failure_report_id = _g(state, "failure_report_id")
    failed_gate_codes: list[str] = []
    unstable_feature_codes: list[str] = []
    feature_evidence_source: str | None = None
    feature_related_failure = False
    attribution_error: str | None = None

    try:
        from ...database import async_session
        from ...repositories.iteration_repo import IterationRepo
        from ...services.iteration.failure_attribution import (
            FailureAttributionService,
        )
        from packages.models.iteration.qualification import QualificationReport

        experiment_id = _g(state, "experiment_id")
        proposal_id = _g(state, "decision_proposal_id") or ""
        if experiment_id:
            async with async_session() as session:
                repo = IterationRepo(session)
                payload = await repo.get_experiment_qualification(experiment_id)
                if payload:
                    report = QualificationReport.model_validate(payload)
                    attribution = FailureAttributionService().from_qualification(
                        proposal_id, report,
                    )
                    if attribution is not None:
                        failure_report_id = attribution.failure_report_id
                        failed_gate_codes = list(attribution.failed_gate_codes)
                        unstable_feature_codes = list(
                            attribution.unstable_feature_codes
                        )
                        feature_evidence_source = (
                            attribution.feature_evidence_source
                        )
                        # 必须存在经归因确认的特征，才授予不稳定特征子集证据
                        feature_related_failure = bool(unstable_feature_codes)
                        await repo.save_failure(attribution)
                        await session.commit()
                    else:
                        attribution_error = "QUALIFICATION_PASSED_NO_FAILURE"
        # 归因失败时不生成随机 failure_report_id：没有真实报告，
        # 不允许基于该证据的自动决策（第二轮特征筛选证据不授予）
        if failure_report_id is None:
            attribution_error = attribution_error or "QUALIFICATION_REPORT_NOT_FOUND"
    except Exception as exc:
        logger.warning("failure_attribution_failed err=%s", exc)
        attribution_error = f"ATTRIBUTION_ERROR: {exc}"

    logger.info(
        "failure_analysis_node",
        lifecycle_run_id=_g(state, "lifecycle_run_id"),
        qualification_run_id=_g(state, "qualification_run_id"),
        business_round=_g(state, "business_round") or 1,
        failure_report_id=failure_report_id,
        failed_gate_codes=failed_gate_codes,
        feature_related_failure=feature_related_failure,
    )
    return {
        "failure_report_id": failure_report_id,
        "failed_gate_codes": ",".join(failed_gate_codes) if failed_gate_codes else None,
        "unstable_feature_codes": (
            ",".join(unstable_feature_codes) if unstable_feature_codes else None
        ),
        "feature_evidence_source": feature_evidence_source,
        # 第二轮特征筛选证据的真实来源：至少一个经归因确认的特征
        "unstable_feature_subset_confirmed": feature_related_failure,
        "iteration_exit_reason": "QUALIFICATION_FAILED",
        "current_phase": LifecyclePhase.OFFLINE_VALIDATING.value,
        "last_error": (
            {"reason": attribution_error, "at": _now_iso()}
            if attribution_error else None
        ),
    }


async def next_round_plan_node(state: ModelLifecycleState) -> dict:
    """P3 下一轮计划节点 — 资格失败且轮次 < max 时进入。

    P0 修复：进入下一轮必须清理首轮部署与 W4 状态，否则第二轮 OOT
    成功后 route_after_deployment_gate 会发现
    final_qualification_completed=True，跳过新 Challenger 的最终七门资格。
    """
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
        # ── 下一轮状态重置（部署与 W4 状态属于上一轮 Challenger）──
        "final_qualification_completed": False,
        "deployment_stage": "OFFLINE_VALIDATION",
        "deployment_decision": None,
        "deployment_id": None,
        "oot_validation_completed": False,
        "oot_validation_run_id": None,
        "w4_available": False,
        "oot_passed": None,
        "candidate_frozen_before_oot": False,
        "lifecycle_terminal": False,
        "challenger_qualified": None,
        "challenger_version": None,
        "qualification_run_id": None,
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

# ═══════════════════════════════════════════════════════════
# P4：任务四 — DeploymentSubgraph
# ═══════════════════════════════════════════════════════════

async def deployment_gate_node(state: ModelLifecycleState) -> dict:
    """P3 部署子图入口 — observe → knowledge → gatekeeper → action → record。

    内部 Pipeline：
    1. Observe: 健康检查 + 生成 DeploymentAlert
    2. Knowledge: KG 查询 DeploymentAlert → Risk → Strategy
    3. Gatekeeper: 综合 KG + health → 最终决策
    4. Action: 执行决策 (traffic_ratio / promote / rollback)
    5. Record: 持久化部署记录 + KG 上下文
    """
    state_dict = _state_dict(state)
    current_stage = _g(state, "deployment_stage") or "OFFLINE_VALIDATION"
    model_id = _g(state, "model_id", "")
    lifecycle_run_id = _g(state, "lifecycle_run_id")
    deployment_id = _g(state, "deployment_id") or str(uuid.uuid4())

    # ── Step 1: Observe ──
    health_metrics = state_dict.get("validation_metrics") or state_dict.get("training_metrics") or {}
    health_result, alerts, oot_evidence = await _deployment_observe(
        state, current_stage, health_metrics, lifecycle_run_id, deployment_id
    )

    logger.info(
        "deployment_subgraph_observe",
        stage=current_stage,
        passed=health_result["passed"],
        alert_count=len(alerts),
        failures=health_result.get("failures", []),
    )

    # ── Step 2: Knowledge (KG query) ──
    alert_codes = [a.alert_code for a in alerts]
    kg_context = await _deployment_knowledge(
        alert_codes, alerts, current_stage, model_id, lifecycle_run_id, deployment_id
    )

    logger.info(
        "deployment_subgraph_knowledge",
        risk_count=len(kg_context.deployment_risks),
        degraded=kg_context.retrieval_degraded,
    )

    # ── Step 3: Gatekeeper ──
    gatekeeper_decision = _deployment_gatekeeper(
        current_stage, health_result, kg_context,
        challenger_qualified=_g(state, "challenger_qualified", True),
    )

    logger.info(
        "deployment_subgraph_gatekeeper",
        decision=gatekeeper_decision.decision,
        reasons=gatekeeper_decision.decision_reasons,
        kg_strategy=gatekeeper_decision.selected_strategy_code,
    )

    # ── Step 4: Action ──
    challenger = _g(state, "challenger_version") or f"{_g(state, 'champion_version', 'v1')}_challenger_v1"
    champion = _g(state, "champion_version", "v1")
    action_result = await _deployment_action(
        state, gatekeeper_decision, current_stage, model_id,
        champion, challenger, deployment_id,
    )

    # ── Step 5: Record ──
    await _deployment_record(
        state, deployment_id, current_stage, gatekeeper_decision,
        health_result, alerts, kg_context, action_result,
    )

    # 合并结果返回
    return _deployment_subgraph_result(
        deployment_id, gatekeeper_decision, action_result, oot_evidence,
    )


# ── P1: DeploymentObserve ──────────────────────────────────────

async def _deployment_observe(
    state: ModelLifecycleState,
    stage: str,
    health_metrics: dict,
    lifecycle_run_id: str | None,
    deployment_id: str,
) -> tuple[dict, list]:
    """Step 1: 健康检查 + 生成 DeploymentAlert。"""
    from ...services.iteration.deployment_safety_service import DeploymentSafetyService
    from ...services.deployment.deployment_observe_service import build_deployment_alerts

    state_dict = _state_dict(state)

    # 健康检查
    # W4 完成证据（A7 §10 NATURAL 校准门槛）：
    # 从 State 继承已有证据，OOT_GATE 真实执行时更新；
    # 后续 Canary/Production 阶段不得清空（否则 PROMOTE/ROLLBACK
    # 会拿不到 W4 证据，NATURAL 观测全部丢失）
    oot_validation_completed = bool(_g(state, "oot_validation_completed"))
    oot_w4_available = bool(_g(state, "w4_available"))
    oot_candidate_frozen = bool(_g(state, "candidate_frozen_before_oot"))
    oot_passed = _g(state, "oot_passed")
    oot_lifecycle = (
        _g(state, "oot_validation_run_id")
        or _g(state, "lifecycle_run_id", "")
    )
    health_passed_explicit = state_dict.get("deployment_health_passed")
    force_rollback = bool(state_dict.get("deployment_force_rollback"))
    if force_rollback:
        health_result = {
            "passed": False,
            "failures": ["injected_force_rollback"],
            "warnings": [],
            "rollback_recommended": True,
            "rollback_reasons": ["injected_force_rollback"],
        }
    elif health_passed_explicit is False:
        health_result = {
            "passed": False,
            "failures": ["injected_health_failure"],
            "warnings": [],
            "rollback_recommended": False,
            "rollback_reasons": [],
        }
    elif stage == "OOT_GATE":
        # ── OOT_GATE: Task 4 独立 W4 盲测 ──
        # 阶段判断必须优先于 health_metrics：资格验证完成后 validation_metrics
        # 通常非空，若先命中 health_metrics 分支，真实 W4 验证会被跳过
        challenger_version = _g(state, "challenger_version", "")
        candidate_version_oot = state_dict.get("challenger_version") or challenger_version
        oot_model_id = _g(state, "model_id", "")
        oot_lifecycle = _g(state, "lifecycle_run_id", "")

        try:
            from ...services.deployment.deployment_oot_service import (
                load_frozen_challenger,
                run_oot_validation,
            )
            frozen = load_frozen_challenger(oot_model_id, oot_lifecycle, candidate_version_oot)
            if not frozen["loaded"]:
                health_result = {
                    "passed": False,
                    "failures": [f"FROZEN_CHALLENGER_LOAD_FAILED: {frozen.get('load_errors', [])}"],
                    "warnings": [],
                    "rollback_recommended": True,
                    "rollback_reasons": ["challenger_frozen_package_not_available"],
                }
            else:
                oot_result = run_oot_validation(
                    frozen["model"],
                    frozen["feature_cols"],
                    model_id=oot_model_id,
                    lifecycle_run_id=oot_lifecycle,
                    candidate_version=candidate_version_oot,
                )
                oot_health = {
                    "challenger_auc": oot_result["oot_auc"],
                    "challenger_ks": oot_result["oot_ks"],
                    "score_psi": oot_result.get("oot_psi"),
                    "w4_available": oot_result["w4_available"],
                    "oot_passed": oot_result["oot_passed"],
                }
                health_result = DeploymentSafetyService.check_stage_health(stage, oot_health)
                if not oot_result["oot_passed"]:
                    health_result["passed"] = False
                    if "failures" not in health_result:
                        health_result["failures"] = []
                    health_result["failures"].append(
                        f"OOT_VALIDATION_FAILED: auc={oot_result['oot_auc']} ks={oot_result['oot_ks']}"
                    )
                logger.info(
                    "deployment_oot_gate_completed model=%s candidate=%s oot_passed=%s auc=%.4f ks=%.4f w4=%s",
                    oot_model_id, candidate_version_oot, oot_result["oot_passed"],
                    oot_result.get("oot_auc", -1), oot_result.get("oot_ks", -1),
                    oot_result["w4_available"],
                )
                # W4 完成证据写入 State（观测层据此判定 NATURAL 门槛）
                oot_validation_completed = bool(oot_result["w4_available"])
                oot_w4_available = bool(oot_result["w4_available"])
                oot_candidate_frozen = True
                oot_passed = bool(oot_result["oot_passed"])

                # ── OOT 结果回写到 experiment，供 qualification_node Gate 6 读取 ──
                oot_experiment_id = _g(state, "experiment_id", "")
                if oot_experiment_id:
                    try:
                        from ...database import async_session
                        from sqlalchemy import text as _sql_text
                        async with async_session() as oot_session:
                            await oot_session.execute(
                                _sql_text("""
                                    UPDATE iteration.experiments
                                    SET experiment_json = experiment_json || CAST(:payload AS JSONB),
                                        updated_at = NOW()
                                    WHERE experiment_id = :eid
                                """),
                                {
                                    "eid": oot_experiment_id,
                                    "payload": json.dumps({
                                        "oot_passed": oot_result["oot_passed"],
                                        "oot_auc": oot_result["oot_auc"],
                                        "oot_ks": oot_result["oot_ks"],
                                        "oot_psi": oot_result.get("oot_psi"),
                                        "candidate_frozen_before_oot": True,
                                        "w4_available": oot_result["w4_available"],
                                    }),
                                },
                            )
                            await oot_session.commit()
                        logger.info("oot_result_written_back_to_experiment eid=%s", oot_experiment_id)
                    except Exception as wb_exc:
                        logger.warning("oot_writeback_failed eid=%s err=%s", oot_experiment_id, wb_exc)

        except Exception as oot_exc:
            logger.warning("oot_gate_service_unavailable err=%s", oot_exc)
            health_result = {
                "passed": False,
                "failures": [f"OOT_SERVICE_UNAVAILABLE: {oot_exc}"],
                "warnings": [],
                "rollback_recommended": False,
                "rollback_reasons": [],
            }
    elif health_metrics:
        health_result = DeploymentSafetyService.check_stage_health(stage, health_metrics)
    elif stage in {"OFFLINE_VALIDATION"}:
        health_result = {"passed": True, "failures": [], "warnings": ["no_health_metrics_provided"]}
    else:
        health_result = {
            "passed": False,
            "failures": ["health_metrics_required_for_canary_or_production"],
            "warnings": [],
            "rollback_recommended": False,
            "rollback_reasons": [],
        }

    # 生成告警
    alerts = build_deployment_alerts(
        stage=stage,
        health_metrics=health_metrics,
        health_result=health_result,
        lifecycle_run_id=lifecycle_run_id,
        deployment_id=deployment_id,
    )

    oot_evidence = {
        "completed": oot_validation_completed,
        "lifecycle": oot_lifecycle,
        "w4_available": oot_w4_available,
        "candidate_frozen_before_oot": oot_candidate_frozen,
        "oot_passed": oot_passed,
    }
    return health_result, alerts, oot_evidence


# ── P0: DeploymentKnowledge ────────────────────────────────────

async def _deployment_knowledge(
    alert_codes: list[str],
    alerts: list,
    stage: str,
    model_id: str,
    lifecycle_run_id: str | None,
    deployment_id: str,
):
    """Step 2: KG 查询 DeploymentAlert → DeploymentRisk → DeploymentStrategy。"""
    from packages.models.deployment.deployment_context import DeploymentContext

    if not alert_codes:
        return DeploymentContext(
            context_pack_id=f"ctx-{deployment_id}",
            model_id=model_id,
            stage=stage,
            retrieval_degraded=False,
        )

    try:
        from ...neo4j_db import get_neo4j_driver
        driver = await get_neo4j_driver()
        from ...services.knowledge_service import KnowledgeService
        svc = KnowledgeService(driver)
        ctx = await svc.query_deployment_context(
            alert_codes=alert_codes,
            alert_payloads=[
                a.model_dump(mode="json") if hasattr(a, "model_dump") else dict(a)
                for a in alerts
            ],
            stage=stage,
            model_id=model_id,
        )
        return ctx
    except Exception:
        logger.warning("deployment_kg_query_failed", exc_info=True)
        from packages.models.deployment.deployment_context import DeploymentContext
        return DeploymentContext(
            context_pack_id=f"ctx-{deployment_id}",
            model_id=model_id,
            stage=stage,
            retrieval_degraded=True,
            degradation_reason="Neo4j connection failed",
        )


# ── P2: DeploymentGatekeeper ───────────────────────────────────

def _deployment_gatekeeper(
    stage: str,
    health_result: dict,
    kg_context,
    *,
    challenger_qualified: bool = True,
):
    """Step 3: 综合 KG + health → 最终决策。"""
    from ...services.deployment.deployment_gatekeeper_service import DeploymentGatekeeperService

    gk = DeploymentGatekeeperService()
    return gk.decide(
        stage=stage,
        health_result=health_result,
        deployment_context=kg_context,
        challenger_qualified=challenger_qualified,
    )


# ── P3: DeploymentAction ───────────────────────────────────────

async def _deployment_action(
    state: ModelLifecycleState,
    decision,
    stage: str,
    model_id: str,
    champion: str,
    challenger: str,
    deployment_id: str,
) -> dict:
    """Step 4: 执行部署决策 (traffic_ratio / promote / rollback)。"""
    d = decision.decision
    result = {
        "deployment_id": deployment_id,
        "deployment_stage": stage,
        "deployment_decision": d,
        "candidate_version": challenger,
    }

    if d in ("ABORT_DEPLOYMENT", "HOLD"):
        return result

    try:
        from ...database import async_session
        async with async_session() as session:
            from ...services.iteration.deployment_safety_service import (
                DeploymentSafetyService, STAGE_TRAFFIC_RATIO,
            )
            svc = DeploymentSafetyService(session)

            if d == "ADVANCE_STAGE":
                from .executors import DEPLOYMENT_STAGES
                try:
                    idx = DEPLOYMENT_STAGES.index(stage)
                    next_stage = DEPLOYMENT_STAGES[idx + 1] if idx + 1 < len(DEPLOYMENT_STAGES) else "PRODUCTION"
                except ValueError:
                    next_stage = "OFFLINE_VALIDATION"
                result["deployment_stage"] = next_stage
                if model_id:
                    await svc.update_traffic_ratio(
                        model_id,
                        next_stage,
                        champion_version=champion,
                        challenger_version=challenger,
                    )
                logger.info("deployment_action_advance", stage=stage, next=next_stage)

            elif d == "PROMOTE":
                result["deployment_stage"] = "PRODUCTION"
                if model_id:
                    await svc.promote_to_champion({
                        "deployment_id": deployment_id,
                        "model_id": model_id,
                        "champion_version": champion,
                        "candidate_version": challenger,
                    })
                logger.info("deployment_action_promote", challenger=challenger)

            elif d == "ROLLBACK":
                await svc.rollback(
                    deployment={
                        "deployment_id": deployment_id,
                        "model_id": model_id,
                        "champion_version": champion,
                        "current_stage": stage,
                    },
                    reason=f"gatekeeper_decision:{d}",
                    rollback_target=decision.rollback_target,
                )
                result["rollback_target"] = decision.rollback_target or champion
                logger.info("deployment_action_rollback", target=result["rollback_target"])

            await session.commit()
    except Exception as exc:
        logger.warning("deployment_action_failed", exc_info=True)
        result["action_failed"] = True
        result["action_error"] = str(exc)

    return result


# ── P4: DeploymentRecord ───────────────────────────────────────

async def _deployment_record(
    state: ModelLifecycleState,
    deployment_id: str,
    stage: str,
    gatekeeper_decision,
    health_result: dict,
    alerts: list,
    kg_context,
    action_result: dict,
) -> None:
    """Step 5: 持久化部署记录 + KG 上下文 + Gatekeeper 决策。"""
    decision = action_result.get("deployment_decision", gatekeeper_decision.decision)
    status = {
        "PROMOTE": "PROMOTED",
        "ROLLBACK": "ROLLED_BACK",
        "ABORT_DEPLOYMENT": "ABORTED",
        "HOLD": "HELD",
    }.get(decision, "RUNNING")

    kg_json = {}
    try:
        kg_json = kg_context.model_dump(mode="json") if hasattr(kg_context, "model_dump") else {}
    except Exception:
        pass

    alerts_json = []
    try:
        alerts_json = [a.model_dump(mode="json") for a in alerts] if alerts else []
    except Exception:
        pass

    record = {
        "deployment_id": deployment_id,
        "lifecycle_run_id": _g(state, "lifecycle_run_id"),
        "qualification_run_id": _g(state, "qualification_run_id"),
        "model_id": _g(state, "model_id"),
        "champion_version": _g(state, "champion_version"),
        "candidate_version": action_result.get("candidate_version") or _g(state, "challenger_version"),
        "deployment_stage": action_result.get("deployment_stage", stage),
        "deployment_decision": decision,
        "status": status,
        "health_json": {
            "health_check": health_result,
            "deployment_alerts": alerts_json,
            "deployment_context": kg_json,
            "gatekeeper_decision": {
                "decision": gatekeeper_decision.decision,
                "selected_strategy_code": gatekeeper_decision.selected_strategy_code,
                "decision_reasons": gatekeeper_decision.decision_reasons,
                "gatekeeper_rule_refs": gatekeeper_decision.gatekeeper_rule_refs,
            },
            "deployment_action": action_result,
        },
    }
    await _save_deployment_record(state=state, record=record)


def _deployment_subgraph_result(
    deployment_id: str,
    gatekeeper_decision,
    action_result: dict,
    oot_evidence: dict | None = None,
) -> dict:
    """构建子图返回值。"""
    decision = action_result.get("deployment_decision", gatekeeper_decision.decision)
    oot_evidence = oot_evidence or {}
    return {
        "deployment_id": deployment_id,
        "deployment_stage": action_result.get("deployment_stage", ""),
        "deployment_decision": decision,
        "gatekeeper_decision": gatekeeper_decision.decision,
        "gatekeeper_reasons": gatekeeper_decision.decision_reasons,
        "selected_deployment_strategy": gatekeeper_decision.selected_strategy_code,
        # A7 §10: W4 FINAL-OOT 完成证据（观测层 NATURAL 门槛）
        "oot_validation_completed": bool(oot_evidence.get("completed")),
        "oot_validation_run_id": (
            oot_evidence.get("lifecycle")
            if oot_evidence.get("completed") else None
        ),
        "w4_available": bool(oot_evidence.get("w4_available")),
        "candidate_frozen_before_oot": bool(
            oot_evidence.get("candidate_frozen_before_oot")
        ),
        "oot_passed": oot_evidence.get("oot_passed"),
        "lifecycle_terminal": decision in {"PROMOTE", "ROLLBACK"},
        "last_error": (
            {
                "reason": "deployment_action_failed",
                "message": action_result.get("action_error"),
                "at": _now_iso(),
            }
            if action_result.get("action_failed")
            else None
        ),
        "current_phase": (
            LifecyclePhase.PROMOTED.value if decision == "PROMOTE"
            else LifecyclePhase.MANUAL_REVIEW.value if decision == "ABORT_DEPLOYMENT"
            else LifecyclePhase.ROLLED_BACK.value if decision == "ROLLBACK"
            else LifecyclePhase.CANARY_RUNNING.value
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


def _build_correlation_matrix(
    state: ModelLifecycleState, feature_names: list[str],
) -> dict[str, dict[str, float]]:
    """从真实训练窗口数据（W2/W3）计算特征共线性矩阵（第三路筛选证据）。"""
    import pandas as pd

    from ...services.monitoring.window_loader import load_window

    frames = []
    for wid in ("W2", "W3"):
        try:
            frames.append(load_window(wid))
        except Exception:
            continue
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    cols = [c for c in feature_names if c in df.columns]
    if len(cols) < 2:
        return {}
    corr = df[cols].fillna(0).corr()
    matrix: dict[str, dict[str, float]] = {}
    for a in cols:
        row: dict[str, float] = {}
        for b in cols:
            if a == b:
                continue
            value = corr.loc[a, b]
            if pd.notna(value):
                row[b] = float(value)
        if row:
            matrix[a] = row
    return matrix


def _save_selection_report_artifact(
    state: ModelLifecycleState, report_json: str,
) -> str | None:
    """筛选报告持久化到 MinIO，返回 artifact URI；不可用时返回 None。"""
    import json as _json

    try:
        from minio import Minio

        client = Minio(
            "localhost:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False,
        )
        object_name = (
            f"feature-selection/{_g(state, 'lifecycle_run_id')}"
            f"/selection-report.json"
        )
        data = _json.dumps(
            _json.loads(report_json), ensure_ascii=False,
        ).encode("utf-8")
        client.put_object(
            "riskitem", object_name, __import__("io").BytesIO(data), len(data),
        )
        return f"s3://riskitem/{object_name}"
    except Exception as exc:
        logger.warning("feature_selection_artifact_save_failed err=%s", exc)
        return None


async def feature_selection_node(state: ModelLifecycleState) -> dict:
    """A7 阶段四：特征筛选节点（FEATURE_SELECTION 模式）。

    基于真实归因证据（unstable_feature_codes + 特征重要性）生成
    冻结特征清单，写入 State 供 TrainingPlan / TrainingJob / Worker 消费。
    不简单删除所有漂移特征：只剔除经归因确认的不稳定特征 +
    低重要性 + 高共线性（保留更重要者）。
    """
    unstable = [
        str(c).strip()
        for c in (_g(state, "unstable_feature_codes") or "").split(",")
        if str(c).strip()
    ]
    feature_importance = _g(state, "feature_importance") or {}
    feature_names = _g(state, "feature_names") or []
    if not feature_names:
        return {
            "requires_manual_review": True,
            "current_phase": LifecyclePhase.MANUAL_REVIEW.value,
            "last_error": {
                "reason": "FEATURE_SELECTION_NO_FEATURE_NAMES",
                "message": "诊断输出缺少 feature_names，无法执行特征筛选",
                "at": _now_iso(),
            },
        }
    try:
        from ...services.iteration.feature_selection_service import (
            select_features,
            serialize_selection,
        )
        # 第三路证据：共线性矩阵来自真实训练窗口数据（W2/W3）
        correlation_matrix = _build_correlation_matrix(state, feature_names)
        result = select_features(
            list(feature_names),
            unstable_feature_codes=unstable,
            feature_importance={
                str(k): float(v) for k, v in feature_importance.items()
            },
            correlation_matrix=correlation_matrix,
        )
        # 筛选报告持久化到 MinIO，生成真实 artifact URI
        artifact_uri = _save_selection_report_artifact(
            state, serialize_selection(result),
        )
        result.feature_selection_artifact_uri = artifact_uri
        logger.info(
            "feature_selection_completed",
            lifecycle_run_id=_g(state, "lifecycle_run_id"),
            selected_count=len(result.selected_feature_codes),
            dropped_count=len(result.dropped_feature_codes),
            drop_reasons=result.drop_reasons,
            artifact_uri=artifact_uri,
        )
        return {
            "selected_feature_codes": ",".join(result.selected_feature_codes),
            "feature_selection_report": serialize_selection(result),
            "feature_selection_artifact_uri": artifact_uri,
            "warnings": [
                f"FEATURE_SELECTION_DROPPED:{code}:{reason}"
                for code, reason in result.drop_reasons.items()
            ] or None,
        }
    except Exception as exc:
        logger.warning("feature_selection_failed err=%s", exc)
        return {
            "requires_manual_review": True,
            "current_phase": LifecyclePhase.MANUAL_REVIEW.value,
            "last_error": {
                "reason": "FEATURE_SELECTION_FAILED",
                "message": str(exc),
                "at": _now_iso(),
            },
        }


async def final_qualification_node(state: ModelLifecycleState) -> dict:
    """A7 资格时序：W4 OOT 完成后的最终资格（Gate 6 + 汇总前六道门）。

    OOT_GATE 已把 oot_passed / candidate_frozen_before_oot / w4_available
    回写 experiment_json，此处用共享构建器重跑完整七道门并保存最终报告。
    """
    experiment_id = _g(state, "experiment_id")
    if not experiment_id:
        return {
            "challenger_qualified": False,
            "final_qualification_completed": True,
            "iteration_exit_reason": "FINAL_QUALIFICATION_NO_EXPERIMENT",
            "current_phase": LifecyclePhase.OFFLINE_VALIDATING.value,
        }
    try:
        from ...database import async_session
        from ...repositories.iteration_repo import IterationRepo
        from ...repositories.monitoring_repo import MonitoringRepo
        from ...services.iteration.config_loader import load_iteration_config
        from ...services.iteration.qualification_service import (
            QualificationEvidenceIncompleteError,
            QualificationService,
            build_qualification_input,
        )

        async with async_session() as session:
            repo = IterationRepo(session)
            experiment = await repo.get_experiment(experiment_id)
            experiment_json = (
                experiment.get("experiment_json") or {} if experiment else {}
            )
            feature_psi: dict[str, float] = {}
            if _g(state, "monitoring_run_id"):
                drift_rows = await MonitoringRepo(session).get_feature_drift_by_run(
                    _g(state, "monitoring_run_id")
                )
                for row in drift_rows:
                    fname = row.get("feature_name")
                    psi = row.get("psi")
                    if fname and psi is not None:
                        feature_psi[str(fname)] = max(
                            feature_psi.get(str(fname), 0.0), float(psi)
                        )
            qual_input = build_qualification_input(
                qualification_run_id=str(uuid.uuid4()),
                iteration_run_id=_g(state, "iteration_run_id") or "",
                experiment_id=experiment_id,
                candidate_version=_g(state, "challenger_version") or "",
                experiment_json=experiment_json,
                feature_psi=feature_psi,
            )
            report = QualificationService(load_iteration_config()).evaluate(
                qual_input,
            )
            await repo.save_qualification(report)
            await session.commit()

            logger.info(
                "final_qualification_completed",
                qualification_run_id=report.qualification_run_id,
                qualified=report.qualified,
                failed_gates=[g.value for g in report.failed_gate_codes],
            )
            return {
                "qualification_run_id": report.qualification_run_id,
                "challenger_qualified": report.qualified,
                "final_qualification_completed": True,
                "current_phase": (
                    LifecyclePhase.QUALIFICATION_COMPLETED.value
                    if report.qualified
                    else LifecyclePhase.OFFLINE_VALIDATING.value
                ),
            }
    except QualificationEvidenceIncompleteError as evidence_exc:
        logger.warning(
            "final_qualification_evidence_incomplete",
            missing_fields=evidence_exc.missing_fields,
        )
        return {
            "challenger_qualified": False,
            "final_qualification_completed": True,
            "iteration_exit_reason": "QUALIFICATION_EVIDENCE_INCOMPLETE",
            "current_phase": LifecyclePhase.OFFLINE_VALIDATING.value,
            "last_error": {
                "reason": "QUALIFICATION_EVIDENCE_INCOMPLETE",
                "message": str(evidence_exc),
                "at": _now_iso(),
            },
        }
    except (OSError, ConnectionError, TimeoutError, _DBIntegrityError):
        logger.warning("final_qualification_infra_error", exc_info=True)
        return {
            "challenger_qualified": False,
            "final_qualification_completed": True,
            "current_phase": LifecyclePhase.FAILED.value,
        }


def route_after_final_qualification(
    state: ModelLifecycleState,
) -> Literal["DeploymentGateNode", "FailureAnalysisNode"]:
    if _g(state, "challenger_qualified", False):
        return "DeploymentGateNode"
    return "FailureAnalysisNode"


async def deployment_outcome_node(state: ModelLifecycleState) -> dict:
    """P4 部署结果节点（A7 §10）。

    PROMOTE / ROLLBACK 都经过本节点写入 KG 观测（NATURAL 门槛由
    KnowledgeObservationService 校验 W4 完成证据 + 终态），之后：
    - PROMOTE → EventCloseNode（关闭诊断事件）
    - ROLLBACK → END
    """
    decision = _g(state, "deployment_decision")

    try:
        from ...database import async_session
        from ...repositories.kg_repo import KnowledgeObservationRepo
        from ...services.knowledge_observation_service import KnowledgeObservationService

        async with async_session() as session:
            repo = KnowledgeObservationRepo(session)
            svc = KnowledgeObservationService()
            observations = svc.build_observations(_state_dict(state))
            if observations:
                ids = await repo.write_observations_batch(observations)
                await session.commit()
                logger.info(
                    "kg_observations_written",
                    deployment_decision=decision,
                    count=len(ids),
                )
    except Exception:
        logger.warning("kg_observation_write_failed", exc_info=True)

    # ROLLBACK 保留 ROLLED_BACK 终态；只有生产 PROMOTE 关闭事件后才 EVENT_CLOSED
    if decision == "PROMOTE":
        return {"current_phase": LifecyclePhase.EVENT_CLOSED.value}
    return {"current_phase": LifecyclePhase.ROLLED_BACK.value}


async def no_alert_close_node(state: ModelLifecycleState) -> dict:
    """无告警/不需要迭代时关闭。如果上游已是 FAILED 则保留原状态不覆盖。"""
    if _g(state, "current_phase") == LifecyclePhase.FAILED.value:
        return {}
    return {"current_phase": LifecyclePhase.NO_ALERT.value}


# ═══════════════════════════════════════════════════════════
# Mock 节点（保留用于降级和阶段化开发）
# ═══════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════
# 条件路由
# ═══════════════════════════════════════════════════════════

def route_after_monitoring(
    state: ModelLifecycleState,
) -> Literal["DiagnosisNode", "NoAlertCloseNode"]:
    if _g(state, "current_phase") == LifecyclePhase.FAILED.value:
        return "NoAlertCloseNode"
    # B1 持续性判定优先：trigger_diagnosis 决定是否进诊断
    trigger_diag = _g(state, "trigger_diagnosis", False)
    if trigger_diag:
        return "DiagnosisNode"
    # SEVERE / requires_manual_review：即使 trigger_diagnosis=false 也进诊断
    if _g(state, "requires_manual_review", False):
        return "DiagnosisNode"
    return "NoAlertCloseNode"


def route_after_diagnosis(
    state: ModelLifecycleState,
) -> Literal["DiagnosisHandoffNode", "ManualReviewNode", "NoAlertCloseNode", "ObservationCloseNode"]:
    if _g(state, "current_phase") == LifecyclePhase.FAILED.value:
        return "NoAlertCloseNode"
    action = _g(state, "recommended_action")
    if _g(state, "requires_manual_review", False) or action == "MANUAL_REVIEW":
        return "ManualReviewNode"
    need = _g(state, "need_iteration")
    if need is True:
        return "DiagnosisHandoffNode"
    if need is False:
        return "ObservationCloseNode"
    return "ManualReviewNode"


def route_after_iteration_decision(
    state: ModelLifecycleState,
) -> Literal[
    "ObservationCloseNode",
    "RepairPlanNode",
    "CalibrationPlanNode",
    "ThresholdPlanNode",
    "FeatureReconstructionNode",
    "FeatureSelectionNode",
    "TrainingPlanNode",
    "ManualReviewNode",
]:
    if _g(state, "current_phase") == LifecyclePhase.FAILED.value:
        return "ManualReviewNode"
    requires_review = _g(state, "requires_manual_review", False)
    if requires_review:
        return "ManualReviewNode"
    return _route_after_action(state)


def route_after_repair_plan(
    state: ModelLifecycleState,
) -> Literal["ObservationCloseNode", "StopAutoIterationNode"]:
    """A3/A4 修复回放资格：合格 → 观察关闭；不合格 → 停止自动迭代。"""
    if _g(state, "repair_qualified") is True:
        return "ObservationCloseNode"
    return "StopAutoIterationNode"


def route_after_manual_review(
    state: ModelLifecycleState,
) -> Literal[
    "ObservationCloseNode",
    "RepairPlanNode",
    "CalibrationPlanNode",
    "ThresholdPlanNode",
    "FeatureReconstructionNode",
    "FeatureSelectionNode",
    "TrainingPlanNode",
    END,
]:
    if _g(state, "requires_manual_review"):
        return END
    action = _recommended_action(state)
    if action == AgentDecisionAction.MANUAL_REVIEW.value:
        need = _g(state, "need_iteration")
        if need is True:
            if _has_feature_level_issues(state):
                return "FeatureReconstructionNode"
            return "TrainingPlanNode"
        if need is False:
            return "ObservationCloseNode"
        return END
    routed = _route_after_action(state)
    if routed in {
        "ObservationCloseNode",
        "RepairPlanNode",
        "CalibrationPlanNode",
        "ThresholdPlanNode",
        "FeatureReconstructionNode",
        "TrainingPlanNode",
    }:
        return routed
    return END


def route_after_feature_reconstruction(
    state: ModelLifecycleState,
) -> Literal["WaitFeatureReconstructionNode", "TrainingPlanNode", "FailureAnalysisNode"]:
    if _g(state, "current_phase") == LifecyclePhase.FAILED.value:
        return "FailureAnalysisNode"
    if _g(state, "feature_reconstruction_dispatched"):
        return "WaitFeatureReconstructionNode"
    return "TrainingPlanNode"


def route_after_training_plan(
    state: ModelLifecycleState,
) -> Literal["HyperparameterTuningNode", "TrainingJobDispatchNode"]:
    """TrainingPlan 之后：按 TrainingMode 决定是否需要超参调优（A7 §7）。

    PARAMETER_TUNING → HyperparameterTuningNode
    其余（FULL_RETRAIN / INCREMENTAL_TRAIN / FEATURE_*）→ 直接派发训练
    """
    training_mode = str(
        _g(state, "training_mode") or "FULL_RETRAIN"
    ).upper()
    if training_mode == "PARAMETER_TUNING":
        return "HyperparameterTuningNode"
    return "TrainingJobDispatchNode"


def route_after_hyperparameter_tuning(
    state: ModelLifecycleState,
) -> Literal["WaitTuningCallbackNode", "TrainingJobDispatchNode", "FailureAnalysisNode"]:
    if _g(state, "current_phase") == LifecyclePhase.FAILED.value:
        return "FailureAnalysisNode"
    if _g(state, "tuning_dispatched") and not _g(state, "tuning_completed"):
        return "WaitTuningCallbackNode"
    return "TrainingJobDispatchNode"


def route_after_qualification(
    state: ModelLifecycleState,
) -> Literal["DeploymentGateNode", "FailureAnalysisNode"]:
    qualified = _g(state, "challenger_qualified", False)

    if qualified:
        return "DeploymentGateNode"
    return "FailureAnalysisNode"


def route_after_deployment_gate(
    state: ModelLifecycleState,
) -> Literal[
    "DeploymentGateNode", "FinalQualificationNode", "DeploymentOutcomeNode", "__end__",
]:
    decision = _g(state, "deployment_decision")
    if decision in {"PROMOTE", "ROLLBACK"}:
        # OOT_GATE 阶段的失败是资格失败，不是部署结果：
        # 先走最终资格（OOT 门 FAILED → FailureAnalysisNode 归因进入第二轮），
        # 不写 NATURAL 部署观测；Canary/Production 的 ROLLBACK 才进
        # DeploymentOutcomeNode 写观测。
        if (
            decision == "ROLLBACK"
            and _g(state, "deployment_stage") == "OOT_GATE"
        ):
            return "FinalQualificationNode"
        return "DeploymentOutcomeNode"
    if decision == "ADVANCE_STAGE":
        # A7 资格时序：OOT_GATE 完成（W4 证据已回写）且最终资格未跑 →
        # 先跑 FinalQualificationNode（Gate 6 + 汇总），再进入下一部署阶段
        if (
            _g(state, "oot_validation_completed")
            and not _g(state, "final_qualification_completed")
        ):
            return "FinalQualificationNode"
        return "DeploymentGateNode"
    # HOLD / ABORT → END
    return END


def route_after_deployment_outcome(
    state: ModelLifecycleState,
) -> Literal["EventCloseNode", "__end__"]:
    decision = _g(state, "deployment_decision")
    stage = _g(state, "deployment_stage")
    if decision == "PROMOTE" and stage == "PRODUCTION":
        return "EventCloseNode"
    # ROLLBACK（含 Canary 回滚）→ END
    return END


def route_after_failure_analysis(
    state: ModelLifecycleState,
) -> Literal["NextRoundPlanNode", "StopAutoIterationNode"]:
    business_round = _g(state, "business_round") or 1
    try:
        from ...services.iteration.config_loader import load_iteration_config
        max_rounds = load_iteration_config().iteration.max_iteration_rounds
    except Exception:
        max_rounds = MAX_BUSINESS_ROUNDS
    if business_round < max_rounds:
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
    graph.add_node("ManualReviewNode", manual_review_node)
    graph.add_node("FeatureReconstructionNode", feature_reconstruction_node)
    graph.add_node("WaitFeatureReconstructionNode", wait_feature_reconstruction_node)
    graph.add_node("FeatureSelectionNode", feature_selection_node)
    graph.add_node("TrainingPlanNode", training_plan_node)

    # T3-GAP-02: 超参优化
    graph.add_node("HyperparameterTuningNode", hyperparameter_tuning_node)
    graph.add_node("WaitTuningCallbackNode", wait_tuning_callback_node)

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
    graph.add_node("FinalQualificationNode", final_qualification_node)
    graph.add_node("DeploymentOutcomeNode", deployment_outcome_node)
    graph.add_node("EventCloseNode", event_close_node)

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
    graph.add_conditional_edges(
        "RepairPlanNode",
        route_after_repair_plan,
        {"ObservationCloseNode": "ObservationCloseNode", "StopAutoIterationNode": "StopAutoIterationNode"},
    )
    graph.add_edge("CalibrationPlanNode", "QualificationNode")
    graph.add_edge("ThresholdPlanNode", "QualificationNode")

    # ManualReview → 分流
    graph.add_conditional_edges("ManualReviewNode", route_after_manual_review)

    # FeatureReconstruction → TrainingPlan
    graph.add_conditional_edges("FeatureReconstructionNode", route_after_feature_reconstruction)
    graph.add_edge("WaitFeatureReconstructionNode", "TrainingPlanNode")
    graph.add_edge("FeatureSelectionNode", "TrainingPlanNode")

    # T3-GAP-02: TrainingPlan → HyperparameterTuning → WaitTuning → TrainingJobDispatch
    graph.add_conditional_edges("TrainingPlanNode", route_after_training_plan)
    graph.add_conditional_edges("HyperparameterTuningNode", route_after_hyperparameter_tuning)
    graph.add_edge("WaitTuningCallbackNode", "TrainingJobDispatchNode")

    # TrainingJobDispatch → WaitCallback → Qualification
    graph.add_edge("TrainingJobDispatchNode", "WaitTrainingCallbackNode")
    graph.add_edge("WaitTrainingCallbackNode", "TrainingCallbackResumeNode")
    graph.add_edge("TrainingCallbackResumeNode", "QualificationNode")

    # Qualification 分流
    graph.add_conditional_edges("QualificationNode", route_after_qualification)
    graph.add_conditional_edges("FailureAnalysisNode", route_after_failure_analysis)

    # NextRound → 重新进入决策（L1 依据 business_round=2 + 失败归因重新选择策略）
    graph.add_edge("NextRoundPlanNode", "IterationDecisionNode")

    # StopAutoIteration → END
    graph.add_edge("StopAutoIterationNode", END)

    # P4: DeploymentGate → OOT 后最终资格 → DeploymentOutcome（写观测）→ EventClose / END
    graph.add_conditional_edges("DeploymentGateNode", route_after_deployment_gate)
    graph.add_conditional_edges(
        "FinalQualificationNode", route_after_final_qualification,
    )
    graph.add_conditional_edges("DeploymentOutcomeNode", route_after_deployment_outcome)
    graph.add_edge("EventCloseNode", END)

    return graph


def build_compiled_graph(checkpointer):
    """构建带 MemorySaver checkpoint 的编译图。"""
    return build_graph().compile(checkpointer=checkpointer)
