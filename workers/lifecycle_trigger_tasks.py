"""Celery Beat 生命周期触发任务（A7 定稿 §8）。

定时触发只能启动"监测 → 诊断 → 决策 → 人工复核"生命周期，
禁止定时直接重训。幂等/冷却/去重/并发锁由 LifecycleTriggerService 保证。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from celery.utils.log import get_task_logger

from apps.modelops_api.services.lifecycle_triggers import (
    ScheduledLifecycleTriggerService,
)
from workers.app import app

logger = get_task_logger(__name__)


def _enabled_model_ids() -> list[str]:
    """从 models.yaml 读取启用模型（champion_fleet.enabled_model_ids）。"""
    from pathlib import Path

    import yaml

    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "assets" / "configs" / "models.yaml"
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        fleet = (cfg or {}).get("champion_fleet", {})
        ids = fleet.get("enabled_model_ids", [])
        return [str(i) for i in ids]
    except Exception:
        logger.exception("model_ids_load_failed", path=str(config_path))
        return []


async def _run_scheduled_trigger() -> dict:
    from apps.modelops_api.database import async_session
    from apps.modelops_api.services.lifecycle_triggers import (
        ScheduledLifecycleTriggerService,
    )

    window_key = datetime.now(timezone.utc).strftime("D%Y-%m-%d")
    started: list[str] = []
    skipped: list[str] = []

    model_ids = _enabled_model_ids()
    if not model_ids:
        return {"started": [], "skipped": [], "reason": "NO_ENABLED_MODELS"}

    async with async_session() as session:
        service = ScheduledLifecycleTriggerService(session)
        for model_id in model_ids:
            try:
                decision = await service.evaluate(
                    model_id=model_id,
                    champion_version="champion_v1",
                    fingerprint_detail=window_key,
                )
                if decision.allowed:
                    started.append(decision.lifecycle_run_id)
                else:
                    skipped.append(f"{model_id}:{decision.reason}")
            except Exception:
                logger.exception(
                    "scheduled_trigger_failed", model_id=model_id,
                )
                skipped.append(f"{model_id}:ERROR")
    return {"started": started, "skipped": skipped}


@app.task(name="workers.lifecycle_trigger_tasks.scheduled_lifecycle_trigger")
def scheduled_lifecycle_trigger() -> dict:
    """定时触发任务一监测生命周期（Beat 入口）。"""
    logger.info("scheduled_lifecycle_trigger_started")
    result = asyncio.run(_run_scheduled_trigger())
    logger.info(
        "scheduled_lifecycle_trigger_done",
        started_count=len(result.get("started", [])),
        skipped_count=len(result.get("skipped", [])),
    )
    return result
