"""Seed W0–W4/OOT 数据窗口

运行方式：
    python -m apps.modelops_api.scripts.seed_data_windows
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from apps.modelops_api.database import async_session
from apps.modelops_api.repositories.data_window_repo import DataWindowRepo

WINDOWS = [
    {
        "window_id": "W0_20250101_20250331",
        "window_name": "W0-初始Champion训练",
        "start_time": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "end_time": datetime(2025, 3, 31, tzinfo=timezone.utc),
        "purpose": "INITIAL_TRAINING",
        "allows_training": True,
        "allows_monitoring_label": True,
        "allows_diagnosis_label": True,
        "allows_iteration_label": True,
        "allows_deployment_label": False,
        "is_frozen": False,
    },
    {
        "window_id": "W1_20250401_20250407",
        "window_name": "W1-健康基准",
        "start_time": datetime(2025, 4, 1, tzinfo=timezone.utc),
        "end_time": datetime(2025, 4, 7, tzinfo=timezone.utc),
        "purpose": "MONITORING_BASELINE",
        "allows_training": False,
        "allows_monitoring_label": True,
        "allows_diagnosis_label": True,
        "allows_iteration_label": True,
        "allows_deployment_label": False,
        "is_frozen": False,
    },
    {
        "window_id": "W2_20250408_20250414",
        "window_name": "W2-滚动监控",
        "start_time": datetime(2025, 4, 8, tzinfo=timezone.utc),
        "end_time": datetime(2025, 4, 14, tzinfo=timezone.utc),
        "purpose": "MONITORING",
        "allows_training": False,
        "allows_monitoring_label": True,
        "allows_diagnosis_label": True,
        "allows_iteration_label": True,
        "allows_deployment_label": False,
        "is_frozen": False,
    },
    {
        "window_id": "W3_20250415_20250421",
        "window_name": "W3-近期监控与训练",
        "start_time": datetime(2025, 4, 15, tzinfo=timezone.utc),
        "end_time": datetime(2025, 4, 21, tzinfo=timezone.utc),
        "purpose": "MONITORING_AND_TRAINING",
        "allows_training": True,
        "allows_monitoring_label": True,
        "allows_diagnosis_label": True,
        "allows_iteration_label": True,
        "allows_deployment_label": False,
        "is_frozen": False,
    },
    {
        "window_id": "W4_20250422_20250428",
        "window_name": "W4-OOT独立准入",
        "start_time": datetime(2025, 4, 22, tzinfo=timezone.utc),
        "end_time": datetime(2025, 4, 28, tzinfo=timezone.utc),
        "purpose": "OOT_GATE",
        "allows_training": False,
        "allows_monitoring_label": False,
        "allows_diagnosis_label": False,
        "allows_iteration_label": False,
        "allows_deployment_label": True,
        "is_frozen": True,
    },
]


async def seed():
    async with async_session() as session:
        repo = DataWindowRepo(session)
        for w in WINDOWS:
            await repo.insert_window(**w)
        await session.commit()
        print(f"已写入 {len(WINDOWS)} 个数据窗口")


if __name__ == "__main__":
    asyncio.run(seed())
