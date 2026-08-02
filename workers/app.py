"""Celery Worker 入口"""

from __future__ import annotations

from celery import Celery
from celery.schedules import crontab
from celery.utils.log import get_task_logger

from apps.modelops_api.config import settings

logger = get_task_logger(__name__)

app = Celery("riskitem")

app.config_from_object(
    {
        "broker_url": settings.celery_broker_url,
        "result_backend": settings.celery_result_backend,
        "task_serializer": "json",
        "accept_content": ["json"],
        "result_serializer": "json",
        "enable_utc": True,
        "task_track_started": True,
        # ── Celery Beat 定时任务 ──
        "beat_schedule": {
            # 每 6 小时执行一次 KG 权重校准
            "kg-calibrate-every-6h": {
                "task": "workers.kg_tasks.kg_calibrate",
                "schedule": crontab(minute="17", hour="*/6"),
                "kwargs": {"data_track": "NATURAL", "rule_version": "BETA_BINOMIAL_V2"},
            },
            # 校准后 30 分钟同步到 Neo4j
            "kg-sync-to-neo4j-every-6h": {
                "task": "workers.kg_tasks.kg_sync_to_neo4j",
                "schedule": crontab(minute="47", hour="*/6"),
                "kwargs": {},
            },
        },
    }
)

# 注册 task 模块
app.autodiscover_tasks(["workers"], force=True)
app.conf.imports = (
    "workers.training_tasks",
    "workers.executor_tasks",
    "workers.kg_tasks",
    "workers.feature_tasks",
    "workers.tuning_tasks",
)


@app.task(bind=True, name="workers.app.test_task")
def test_task(self, msg: str = "hello"):
    """测试任务 — 确认 Celery 能正常调度和执行。"""
    logger.info(f"test_task received: msg={msg}, task_id={self.request.id}")
    return {"status": "ok", "msg": msg, "task_id": self.request.id}
