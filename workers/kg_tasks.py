"""KG 权重校准 Celery 定时任务。

定时执行：
- kg_calibrate: 聚合观测 → Beta-Binomial 收缩 → 权重快照
- kg_sync_to_neo4j: 将最新快照同步到 Neo4j
"""

from __future__ import annotations

import json
import os
import urllib.request
import uuid

from celery.utils.log import get_task_logger

from .app import app

logger = get_task_logger(__name__)

API_BASE = os.getenv("MODELOPS_API_BASE", "http://127.0.0.1:8000")


def _api_post(path: str, body: dict) -> dict:
    """同步 HTTP POST，兼容 Windows ProactorEventLoop。"""
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        logger.warning("api_post_failed url=%s code=%s", url, exc.code)
        raise


def _api_get(path: str) -> dict:
    """同步 HTTP GET。"""
    url = f"{API_BASE}{path}"
    req = urllib.request.Request(url, method="GET")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


@app.task(bind=True, name="workers.kg_tasks.kg_calibrate", max_retries=1, default_retry_delay=300)
def kg_calibrate(
    self,
    data_track: str = "NATURAL",
    rule_version: str = "BETA_BINOMIAL_V2",
    weight_version: str | None = None,
):
    """定时执行 KG 权重校准。

    参数：
    - data_track: NATURAL 或 SCENARIO
    - rule_version: 校准算法版本
    - weight_version: 权重版本标签，不传则自动生成
    """
    if weight_version is None:
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        weight_version = f"KG_WEIGHT_CELERY_{ts}"

    logger.info(
        "kg_calibrate_started data_track=%s rule=%s weight=%s",
        data_track, rule_version, weight_version,
    )

    try:
        result = _api_post("/api/kg/calibration-runs", {
            "data_track": data_track,
            "rule_version": rule_version,
            "weight_version": weight_version,
        })
        run_id = result.get("data", {}).get("calibration_run_id", "unknown")
        logger.info("kg_calibrate_completed calibration_run_id=%s", run_id)
        return {"status": "SUCCEEDED", "calibration_run_id": run_id, "weight_version": weight_version}
    except Exception as exc:
        logger.error("kg_calibrate_failed error=%s", str(exc))
        raise self.retry(exc=exc)


@app.task(bind=True, name="workers.kg_tasks.kg_sync_to_neo4j", max_retries=1, default_retry_delay=120)
def kg_sync_to_neo4j(
    self,
    calibration_run_id: str | None = None,
    weight_version: str | None = None,
):
    """将最新的权重快照同步到 Neo4j。

    参数：
    - calibration_run_id: 指定校准运行 ID
    - weight_version: 指定权重版本
    都不传则同步每个 relation_key 的最新快照。
    """
    logger.info(
        "kg_sync_to_neo4j_started calibration_run_id=%s weight_version=%s",
        calibration_run_id, weight_version,
    )

    try:
        if not calibration_run_id:
            # 找最新的 SUCCEEDED 校准运行
            resp = _api_get("/api/kg/calibration-runs?status=SUCCEEDED&limit=1")
            items = resp.get("data", {}).get("items", [])
            if items:
                calibration_run_id = items[0]["calibration_run_id"]
            else:
                logger.warning("kg_sync_to_neo4j_no_calibration_run")
                return {"status": "SKIPPED", "reason": "no calibration run found"}

        result = _api_post(
            f"/api/kg/calibration-runs/{calibration_run_id}/apply-to-neo4j",
            {"weight_version": weight_version} if weight_version else {},
        )
        applied = result.get("data", {}).get("applied", 0)
        logger.info("kg_sync_to_neo4j_completed applied=%d", applied)
        return {"status": "SUCCEEDED", "applied_count": applied}
    except Exception as exc:
        logger.error("kg_sync_to_neo4j_failed error=%s", str(exc))
        raise self.retry(exc=exc)


@app.task(bind=True, name="workers.kg_tasks.kg_full_pipeline", max_retries=0)
def kg_full_pipeline(self, data_track: str = "NATURAL"):
    """一键执行 校准 → 同步 完整链路。"""
    logger.info("kg_full_pipeline_started data_track=%s", data_track)

    # Step 1: 校准
    cal_result = kg_calibrate(data_track=data_track)
    if cal_result.get("status") != "SUCCEEDED":
        logger.error("kg_full_pipeline_calibrate_failed")
        return {"status": "FAILED", "step": "calibrate"}

    # Step 2: 同步
    sync_result = kg_sync_to_neo4j(
        calibration_run_id=cal_result.get("calibration_run_id"),
        weight_version=cal_result.get("weight_version"),
    )

    logger.info(
        "kg_full_pipeline_completed applied=%d",
        sync_result.get("applied_count", 0),
    )
    return {
        "status": "SUCCEEDED",
        "calibration_run_id": cal_result.get("calibration_run_id"),
        "applied_count": sync_result.get("applied_count", 0),
    }
