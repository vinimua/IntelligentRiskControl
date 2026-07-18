"""Celery Worker 入口"""

from __future__ import annotations

from celery import Celery
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
    }
)

# 自动发现 tasks 模块
app.autodiscover_tasks(["workers"], force=True)


@app.task(bind=True, name="workers.app.test_task")
def test_task(self, msg: str = "hello"):
    """测试任务 — 确认 Celery 能正常调度和执行。"""
    logger.info(f"test_task received: msg={msg}, task_id={self.request.id}")
    return {"status": "ok", "msg": msg, "task_id": self.request.id}
