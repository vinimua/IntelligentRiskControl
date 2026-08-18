"""P3 真实执行器 — 训练 / 校准 / 阈值 / 修复 / 部署。

每个执行器被对应的 graph node 调用。
计划必须携带可验证的真实模型和快照身份；Worker 缺任何输入时失败关闭。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone

import structlog

logger = structlog.get_logger(__name__)


def _post_external(url: str | None, payload: dict) -> dict:
    if not url:
        return {"dispatched": False, "dispatch_mode": "INTERNAL"}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body) if body else {}
            return {
                "dispatched": True,
                "dispatch_mode": "EXTERNAL_HTTP",
                "external_task_id": parsed.get("task_id") or parsed.get("id"),
                "response": parsed,
            }
    except urllib.error.HTTPError as exc:
        logger.warning("external_executor_http_failed", url=url, code=exc.code)
        raise


# ═══════════════════════════════════════════
# 1. 训练执行器
# ═══════════════════════════════════════════

async def dispatch_training_job(
    job_input: dict,
    celery_app=None,
) -> dict:
    """派发训练任务到 Celery Worker。

    接入真实 Celery：
        from celery import Celery
        app = Celery("riskitem")
        app.send_task("workers.training_tasks.train_model", args=[job_input])
    """
    training_job_id = job_input.get("training_job_id", str(uuid.uuid4()))
    dispatched = False
    error: str | None = None

    try:
        if celery_app is not None:
            async_result = celery_app.send_task(
                "workers.training_tasks.train_model",
                args=[job_input],
            )
            dispatched = True
            logger.info(
                "training_job_dispatched_to_celery",
                training_job_id=training_job_id,
                celery_task_id=getattr(async_result, "id", None),
            )
        else:
            # Celery 不可用 → 需前端手动提交训练回调
            logger.warning(
                "training_job_not_dispatched_celery_unavailable",
                training_job_id=training_job_id,
            )
            dispatched = False
    except Exception as exc:
        error = str(exc)
        logger.error(
            "training_job_dispatch_failed",
            training_job_id=training_job_id,
            error=error,
        )

    return {
        "training_job_id": training_job_id,
        "dispatched": dispatched,
        "error": error,
    }


# ═══════════════════════════════════════════
# 2. 校准执行器
# ═══════════════════════════════════════════

def create_calibration_plan(state: dict) -> dict:
    """创建校准调整计划。

    正式口径：W2 拟合，W3 验证，W1 提供健康基线；Worker 必须重新消费
    Champion 原始 risk score，禁止使用随机分数或校准后分数拟合。
    """
    from ...config import settings

    state = {**state, **dict(state.get("action_execution_context") or {})}

    model_id = str(state.get("model_id") or "")
    champion_version = str(state.get("champion_version") or "")
    if not model_id or not champion_version:
        raise ValueError("CALIBRATION_MODEL_ID_AND_VERSION_REQUIRED")
    business_round = state.get("business_round", 1)
    plan_id = state.get("calibration_plan_id") or str(uuid.uuid4())
    lifecycle_run_id = state.get("lifecycle_run_id")

    return {
        "calibration_plan_id": plan_id,
        "lifecycle_run_id": lifecycle_run_id,
        "model_id": model_id,
        "champion_version": champion_version,
        "champion_bundle_uri": state.get("champion_bundle_uri"),
        "champion_model_checksum": state.get("champion_model_checksum"),
        "fit_snapshot_id": state.get("fit_snapshot_id") or "W2",
        "fit_snapshot_uri": state.get("fit_snapshot_uri") or "assets/data/windows/W2/data.parquet",
        "fit_snapshot_checksum": state.get("fit_snapshot_checksum"),
        "validation_snapshot_id": state.get("validation_snapshot_id") or "W3",
        "validation_snapshot_uri": state.get("validation_snapshot_uri") or "assets/data/windows/W3/data.parquet",
        "validation_snapshot_checksum": state.get("validation_snapshot_checksum"),
        "healthy_snapshot_id": state.get("healthy_snapshot_id") or "W1",
        "healthy_snapshot_uri": state.get("healthy_snapshot_uri") or "assets/data/windows/W1/data.parquet",
        "healthy_snapshot_checksum": state.get("healthy_snapshot_checksum"),
        "calibrator_type": state.get("calibrator_type") or settings.calibration_calibrator_type,
        "calibration_metrics": state.get("calibration_metrics") or settings.calibration_metrics,
        "artifact_output_path": state.get("calibration_artifact_output_path") or (
            f"{settings.calibration_artifact_prefix}/"
            f"{model_id}/{champion_version}_calibrated_v{business_round}.joblib"
        ),
        "status": "PLANNED",
        "callback_endpoint": (
            f"/api/internal/iteration/executions/CALIBRATION/{plan_id}/callback"
        ),
    }


# ═══════════════════════════════════════════
# 3. 阈值执行器
# ═══════════════════════════════════════════

def create_threshold_plan(state: dict) -> dict:
    """创建阈值调整计划。

    A6 是人工批准的预留能力。缺真实业务目标变化信号或批准号时，计划创建
    即失败；不得仅因离线指标可能提高而自动调阈值。
    """
    from ...config import settings

    state = {**state, **dict(state.get("action_execution_context") or {})}

    model_id = str(state.get("model_id") or "")
    champion_version = str(state.get("champion_version") or "")
    if not model_id or not champion_version:
        raise ValueError("THRESHOLD_MODEL_ID_AND_VERSION_REQUIRED")
    if not bool(state.get("business_objective_changed")):
        raise ValueError("A6_REAL_BUSINESS_OBJECTIVE_CHANGE_REQUIRED")
    authorization_id = state.get("authorization_id") or state.get("manual_review_id")
    if not authorization_id:
        raise ValueError("A6_MANUAL_AUTHORIZATION_REQUIRED")
    business_round = state.get("business_round", 1)
    plan_id = state.get("threshold_plan_id") or str(uuid.uuid4())
    lifecycle_run_id = state.get("lifecycle_run_id")

    search_range = state.get("search_range") or {
        "min": settings.threshold_search_min,
        "max": settings.threshold_search_max,
        "step": settings.threshold_search_step,
    }

    return {
        "threshold_plan_id": plan_id,
        "lifecycle_run_id": lifecycle_run_id,
        "model_id": model_id,
        "champion_version": champion_version,
        "champion_bundle_uri": state.get("champion_bundle_uri"),
        "champion_model_checksum": state.get("champion_model_checksum"),
        "fit_snapshot_id": state.get("fit_snapshot_id") or "W2",
        "fit_snapshot_uri": state.get("fit_snapshot_uri") or "assets/data/windows/W2/data.parquet",
        "fit_snapshot_checksum": state.get("fit_snapshot_checksum"),
        "validation_snapshot_id": state.get("validation_snapshot_id") or "W3",
        "validation_snapshot_uri": state.get("validation_snapshot_uri") or "assets/data/windows/W3/data.parquet",
        "validation_snapshot_checksum": state.get("validation_snapshot_checksum"),
        "business_objective_changed": True,
        "authorization_id": authorization_id,
        "current_threshold": state.get("current_threshold"),
        "search_metric": state.get("search_metric") or settings.threshold_search_metric,
        "search_range": search_range,
        "artifact_output_path": state.get("threshold_artifact_output_path") or (
            f"{settings.threshold_artifact_prefix}/"
            f"{model_id}/{champion_version}_threshold_v{business_round}.json"
        ),
        "status": "PLANNED",
        "callback_endpoint": (
            f"/api/internal/iteration/executions/THRESHOLD/{plan_id}/callback"
        ),
    }


# ═══════════════════════════════════════════
# 4. 数据修复执行器
# ═══════════════════════════════════════════

def create_repair_plan(state: dict) -> dict:
    """创建数据/管道修复计划。

    当前产出结构化修复指令（供外部 data/pipeline 团队执行）。
    修复完成后通过 API 回调标记完成。
    """
    from ...config import settings
    from ...services.iteration.config_loader import load_iteration_config

    state = {**state, **dict(state.get("action_execution_context") or {})}
    action = state.get("recommended_action", "DATA_REPAIR")
    if action not in {"DATA_REPAIR", "PIPELINE_REPAIR"}:
        raise ValueError(f"REPAIR_ACTION_UNSUPPORTED:{action}")
    model_id = str(state.get("model_id") or "")
    champion_version = str(state.get("champion_version") or "")
    if not model_id or not champion_version:
        raise ValueError("REPAIR_MODEL_ID_AND_VERSION_REQUIRED")
    affected_features = [str(name) for name in state.get("affected_features") or []]
    if not affected_features:
        raise ValueError("REPAIR_AFFECTED_FEATURES_REQUIRED")
    diagnosis_run_id = state.get("diagnosis_run_id", "")
    plan_id = state.get("repair_plan_id") or str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    try:
        iter_config = load_iteration_config().iteration
        default_training = iter_config.default_training_window_ids
    except Exception:
        default_training = ["W2", "W3"]

    repair_items = []
    if action == "DATA_REPAIR":
        repair_items = [
            {
                "type": "FEATURE_BACKFILL",
                "description": "回填特征源数据到正确版本",
                "target_windows": state.get("training_window_ids") or default_training,
                "priority": "HIGH",
            }
        ]
    elif action == "PIPELINE_REPAIR":
        repair_items = [
            {
                "type": "PIPELINE_FIX",
                "description": "修复数据管道处理逻辑",
                "affected_stages": state.get("affected_stages") or ["WP03", "WP04"],
                "priority": "CRITICAL",
            }
        ]

    return {
        "repair_plan_id": plan_id,
        "lifecycle_run_id": state.get("lifecycle_run_id"),
        "action": action,
        "model_id": model_id,
        "champion_version": champion_version,
        "champion_bundle_uri": state.get("champion_bundle_uri"),
        "champion_model_checksum": state.get("champion_model_checksum"),
        "diagnosis_run_id": diagnosis_run_id,
        "source_snapshot_id": state.get("source_snapshot_id") or "W3_DEGRADED",
        "source_snapshot_uri": state.get("source_snapshot_uri"),
        "source_snapshot_checksum": state.get("source_snapshot_checksum"),
        "reference_snapshot_id": state.get("reference_snapshot_id") or (
            "W2" if action == "DATA_REPAIR" else "W3_TRUSTED"
        ),
        "reference_snapshot_uri": state.get("reference_snapshot_uri"),
        "reference_snapshot_checksum": state.get("reference_snapshot_checksum"),
        "healthy_snapshot_id": state.get("healthy_snapshot_id") or "W1",
        "healthy_snapshot_uri": state.get("healthy_snapshot_uri") or "assets/data/windows/W1/data.parquet",
        "healthy_snapshot_checksum": state.get("healthy_snapshot_checksum"),
        "affected_features": affected_features,
        "repair_items": repair_items,
        "artifact_output_path": state.get("repair_artifact_output_path") or (
            f"artifacts/repairs/{model_id}/{plan_id}/repaired.parquet"
        ),
        "created_at": now,
        "status": "PLANNED",
        "callback_endpoint": f"/api/internal/iteration/executions/REPAIR/{plan_id}/callback",
    }


def dispatch_external_execution(plan_type: str, plan: dict) -> dict:
    from ...config import settings

    urls = {
        "CALIBRATION": settings.calibration_executor_url,
        "THRESHOLD": settings.threshold_executor_url,
        "REPAIR": settings.repair_executor_url,
    }
    url = urls.get(plan_type)
    payload = {"plan_type": plan_type, **plan}
    return _post_external(url, payload)


def dispatch_deployment_action(state: dict, result: dict) -> dict:
    from ...config import settings

    payload = {
        "lifecycle_run_id": state.get("lifecycle_run_id"),
        "qualification_run_id": state.get("qualification_run_id"),
        "model_id": state.get("model_id"),
        "champion_version": state.get("champion_version"),
        **result,
    }
    return _post_external(settings.deployment_executor_url, payload)


# ═══════════════════════════════════════════
# 5. 部署执行器
# ═══════════════════════════════════════════

DEPLOYMENT_STAGES = [
    "OFFLINE_VALIDATION",
    "OOT_GATE",
    "SHADOW",
    "CANARY_5",
    "CANARY_20",
    "CANARY_50",
    "PRODUCTION",
]


def execute_deployment_stage(state: dict) -> dict:
    """执行一个部署阶段决策。

    真实实现会：
    1. 检查当前 stage 的健康指标
    2. 满足条件 → advance 到下一 stage
    3. 不满足 → hold / reduce_traffic / rollback
    """
    current_stage = state.get("deployment_stage") or "OFFLINE_VALIDATION"
    challenger_qualified = state.get("challenger_qualified", False)
    champion_version = state.get("champion_version", "v1")
    challenger_version = state.get("challenger_version") or f"{champion_version}_challenger_v1"
    deployment_id = state.get("deployment_id") or str(uuid.uuid4())
    health_passed = state.get("deployment_health_passed", True)
    force_rollback = state.get("deployment_force_rollback", False)

    if not challenger_qualified:
        return {
            "deployment_id": state.get("deployment_id"),
            "deployment_stage": current_stage,
            "deployment_decision": "ABORT_DEPLOYMENT",
        }

    if force_rollback:
        return {
            "deployment_id": deployment_id,
            "deployment_stage": current_stage,
            "deployment_decision": "ROLLBACK",
            "rollback_target": champion_version,
            "candidate_version": challenger_version,
        }

    if not health_passed:
        return {
            "deployment_id": deployment_id,
            "deployment_stage": current_stage,
            "deployment_decision": "HOLD",
            "hold_reason": "DEPLOYMENT_HEALTH_CHECK_FAILED",
            "candidate_version": challenger_version,
        }

    # 找到当前阶段索引，推进到下一阶段
    try:
        idx = DEPLOYMENT_STAGES.index(current_stage)
        if idx + 1 >= len(DEPLOYMENT_STAGES):
            # 已经在 PRODUCTION
            return {
                "deployment_id": deployment_id,
                "deployment_stage": "PRODUCTION",
                "deployment_decision": "PROMOTE",
                "candidate_version": challenger_version,
            }
        next_stage = DEPLOYMENT_STAGES[idx + 1]
    except ValueError:
        # 未知阶段 → 从头开始
        next_stage = "OFFLINE_VALIDATION"
        idx = -1

    if next_stage == "PRODUCTION":
        return {
            "deployment_id": deployment_id,
            "deployment_stage": "PRODUCTION",
            "deployment_decision": "PROMOTE",
            "candidate_version": challenger_version,
        }

    return {
        "deployment_id": deployment_id,
        "deployment_stage": next_stage,
        "deployment_decision": "ADVANCE_STAGE",
        "candidate_version": challenger_version,
    }
