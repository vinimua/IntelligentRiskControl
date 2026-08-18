"""P4 执行任务 — 校准 / 阈值 / 修复。

每个任务：
1. 接收计划 → 执行 → 保存产物
2. 回调对应 lifecycle 端点
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from celery.utils.log import get_task_logger

from .app import app

logger = get_task_logger(__name__)

API_BASE = os.getenv("MODELOPS_API_BASE", "http://127.0.0.1:8000")


def _api_post(path: str, body: dict) -> dict:
    """同步 HTTP POST，Windows 兼容。"""
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        logger.warning("api_post_failed url=%s code=%s", url, exc.code)
        raise


# ═══════════════════════════════════════════
# 1. 校准执行器
# ═══════════════════════════════════════════

@app.task(bind=True, name="workers.executor_tasks.calibrate", max_retries=1)
def calibrate(self, plan: dict):
    """执行 Isotonic/Platt 校准训练。

    输入: {calibration_plan_id, champion_version, lifecycle_run_id}
    输出: calibrator artifact → MinIO → callback
    """
    plan_id = str(plan.get("calibration_plan_id") or "")
    if not plan_id:
        raise ValueError("CALIBRATION_PLAN_ID_REQUIRED")
    run_id = plan.get("lifecycle_run_id", "")

    logger.info("calibrate_started plan=%s", plan_id)

    try:
        from apps.modelops_api.services.iteration.action_execution_service import (
            execute_calibration,
        )

        result = execute_calibration(plan)

        logger.info("calibrate_done plan=%s", plan_id)

        # 回调
        callback_payload = {
            "artifact_uri": result["artifact_uri"],
            "artifact_checksum": result["artifact_checksum"],
            "status": result["status"],
            "metrics": result["metrics"],
            "consumption_receipt": result["consumption_receipt"],
            "external_task_id": getattr(self.request, "id", None),
            "resume_lifecycle": bool(run_id),
        }
        _api_post(
            f"/api/internal/iteration/executions/CALIBRATION/{plan_id}/callback",
            callback_payload,
        )

        return result

    except Exception as exc:
        logger.error("calibrate_failed plan=%s err=%s", plan_id, exc)
        try:
            _api_post(
                f"/api/internal/iteration/executions/CALIBRATION/{plan_id}/callback",
                {
                    "status": "FAILED",
                    "error_message": str(exc),
                    "external_task_id": getattr(self.request, "id", None),
                    "resume_lifecycle": bool(run_id),
                },
            )
        except Exception:
            logger.error("calibrate_failure_callback_failed plan=%s", plan_id)
        raise self.retry(exc=exc)


# ═══════════════════════════════════════════
# 2. 阈值搜索执行器
# ═══════════════════════════════════════════

@app.task(bind=True, name="workers.executor_tasks.search_threshold", max_retries=1)
def search_threshold(self, plan: dict):
    """网格搜索最优决策阈值。

    输入: {threshold_plan_id, champion_version, lifecycle_run_id}
    输出: threshold.json → MinIO → callback
    """
    plan_id = str(plan.get("threshold_plan_id") or "")
    if not plan_id:
        raise ValueError("THRESHOLD_PLAN_ID_REQUIRED")
    run_id = plan.get("lifecycle_run_id", "")

    logger.info("threshold_search_started plan=%s", plan_id)

    try:
        from apps.modelops_api.services.iteration.action_execution_service import (
            execute_threshold_search,
        )

        result = execute_threshold_search(plan)
        best = result["metrics"]["after"]

        logger.info("threshold_search_done plan=%s best=%.2f", plan_id, best["threshold"])

        # 回调
        callback_payload = {
            "artifact_uri": result["artifact_uri"],
            "artifact_checksum": result["artifact_checksum"],
            "status": result["status"],
            "metrics": result["metrics"],
            "consumption_receipt": result["consumption_receipt"],
            "external_task_id": getattr(self.request, "id", None),
            "resume_lifecycle": bool(run_id),
        }
        _api_post(
            f"/api/internal/iteration/executions/THRESHOLD/{plan_id}/callback",
            callback_payload,
        )

        return result

    except Exception as exc:
        logger.error("threshold_search_failed plan=%s err=%s", plan_id, exc)
        try:
            _api_post(
                f"/api/internal/iteration/executions/THRESHOLD/{plan_id}/callback",
                {
                    "status": "FAILED",
                    "error_message": str(exc),
                    "external_task_id": getattr(self.request, "id", None),
                    "resume_lifecycle": bool(run_id),
                },
            )
        except Exception:
            logger.error("threshold_failure_callback_failed plan=%s", plan_id)
        raise self.retry(exc=exc)


# ═══════════════════════════════════════════
# 3. 数据/管道修复 + 同窗回放
# ═══════════════════════════════════════════

@app.task(bind=True, name="workers.executor_tasks.repair_and_replay", max_retries=1)
def repair_and_replay(self, plan: dict):
    plan_id = str(plan.get("repair_plan_id") or "")
    if not plan_id:
        raise ValueError("REPAIR_PLAN_ID_REQUIRED")
    run_id = plan.get("lifecycle_run_id", "")
    logger.info("repair_and_replay_started plan=%s", plan_id)
    try:
        from apps.modelops_api.services.iteration.action_execution_service import (
            execute_repair_and_replay,
        )

        result = execute_repair_and_replay(plan)
        _api_post(
            f"/api/internal/iteration/executions/REPAIR/{plan_id}/callback",
            {
                "artifact_uri": result["artifact_uri"],
                "artifact_checksum": result["artifact_checksum"],
                "status": result["status"],
                "metrics": result["metrics"],
                "consumption_receipt": result["consumption_receipt"],
                "external_task_id": getattr(self.request, "id", None),
                "resume_lifecycle": bool(run_id),
            },
        )
        return result
    except Exception as exc:
        logger.error("repair_and_replay_failed plan=%s err=%s", plan_id, exc)
        try:
            _api_post(
                f"/api/internal/iteration/executions/REPAIR/{plan_id}/callback",
                {
                    "status": "FAILED",
                    "error_message": str(exc),
                    "external_task_id": getattr(self.request, "id", None),
                    "resume_lifecycle": bool(run_id),
                },
            )
        except Exception:
            logger.error("repair_failure_callback_failed plan=%s", plan_id)
        raise self.retry(exc=exc)


# ═══════════════════════════════════════════
# 4. 兼容旧外部修复通知（不再作为正式 A3/A4 成功证据）
# ═══════════════════════════════════════════

@app.task(bind=True, name="workers.executor_tasks.notify_repair_complete", max_retries=3)
def notify_repair_complete(self, repair_info: dict):
    """通知修复完成（外部修复团队调用）。

    输入: {repair_plan_id, lifecycle_run_id, status}
    """
    plan_id = repair_info.get("repair_plan_id", "")
    run_id = repair_info.get("lifecycle_run_id", "")
    status = str(repair_info.get("status") or "").upper()
    if status not in {"SUCCEEDED", "FAILED"}:
        raise ValueError("REPAIR_STATUS_EXPLICIT_REQUIRED")

    logger.info("repair_notify plan=%s status=%s", plan_id, status)

    try:
        _api_post(
            f"/api/internal/iteration/repair/{plan_id}/complete?lifecycle_run_id={run_id}",
            {
                **repair_info,
                "status": status,
                "repair_plan_id": plan_id,
            },
        )
        return {"status": status, "lifecycle_resumed": True}
    except Exception as exc:
        logger.error("repair_notify_failed plan=%s err=%s", plan_id, exc)
        raise self.retry(exc=exc)
