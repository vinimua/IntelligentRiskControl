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
import uuid

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
    plan_id = plan.get("calibration_plan_id", str(uuid.uuid4()))
    champion = plan.get("champion_version", "v1")
    run_id = plan.get("lifecycle_run_id", "")

    logger.info("calibrate_started plan=%s", plan_id)

    try:
        from sklearn.isotonic import IsotonicRegression
        import joblib as jl
        import io as _io
        import numpy as np

        # Mock: 训练 IsotonicRegression
        y_pred = np.random.beta(2, 5, 5000)
        y_true = (y_pred > 0.5).astype(int)
        iso = IsotonicRegression(out_of_bounds="clip").fit(y_pred, y_true)

        buf = _io.BytesIO()
        jl.dump(iso, buf)
        buf.seek(0)
        artifact = f"s3://riskitem/calibrators/{champion}_calibrated_v1.joblib"

        logger.info("calibrate_done plan=%s", plan_id)

        # 回调
        callback_payload = {
            "artifact_uri": artifact,
            "status": "SUCCEEDED",
            "metrics": {"brier": 0.12, "ece": 0.03},
            "external_task_id": getattr(self.request, "id", None),
            "resume_lifecycle": bool(run_id),
        }
        _api_post(
            f"/api/internal/iteration/executions/CALIBRATION/{plan_id}/callback",
            callback_payload,
        )

        return {"status": "SUCCEEDED", "artifact": artifact}

    except Exception as exc:
        logger.error("calibrate_failed plan=%s err=%s", plan_id, exc)
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
    plan_id = plan.get("threshold_plan_id", str(uuid.uuid4()))
    champion = plan.get("champion_version", "v1")
    run_id = plan.get("lifecycle_run_id", "")

    logger.info("threshold_search_started plan=%s", plan_id)

    try:
        import json
        import numpy as np

        # Mock: 阈值搜索
        y_pred = np.random.beta(2, 5, 5000)
        y_true = (y_pred > 0.5).astype(int)
        thresholds = np.arange(0.3, 0.71, 0.01)
        best = {"threshold": 0.45, "f1": 0.72, "precision": 0.68, "recall": 0.76}
        artifact = f"s3://riskitem/thresholds/{champion}_threshold_v1.json"

        logger.info("threshold_search_done plan=%s best=%.2f", plan_id, best["threshold"])

        # 回调
        callback_payload = {
            "artifact_uri": artifact,
            "status": "SUCCEEDED",
            "metrics": best,
            "external_task_id": getattr(self.request, "id", None),
            "resume_lifecycle": bool(run_id),
        }
        _api_post(
            f"/api/internal/iteration/executions/THRESHOLD/{plan_id}/callback",
            callback_payload,
        )

        return {"status": "SUCCEEDED", "artifact": artifact, "threshold": best["threshold"]}

    except Exception as exc:
        logger.error("threshold_search_failed plan=%s err=%s", plan_id, exc)
        raise self.retry(exc=exc)


# ═══════════════════════════════════════════
# 3. 数据修复完成回调端点辅助
# ═══════════════════════════════════════════

@app.task(bind=True, name="workers.executor_tasks.notify_repair_complete", max_retries=3)
def notify_repair_complete(self, repair_info: dict):
    """通知修复完成（外部修复团队调用）。

    输入: {repair_plan_id, lifecycle_run_id, status}
    """
    plan_id = repair_info.get("repair_plan_id", "")
    run_id = repair_info.get("lifecycle_run_id", "")
    status = repair_info.get("status", "SUCCEEDED")

    logger.info("repair_notify plan=%s status=%s", plan_id, status)

    try:
        _api_post(
            f"/api/internal/iteration/repair/{plan_id}/complete?lifecycle_run_id={run_id}",
            {"status": "SUCCEEDED", "repair_plan_id": plan_id},
        )
        return {"status": "OK", "lifecycle_resumed": True}
    except Exception as exc:
        logger.error("repair_notify_failed plan=%s err=%s", plan_id, exc)
        raise self.retry(exc=exc)
