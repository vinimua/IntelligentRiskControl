"""model_registry.data_windows 数据访问"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class DataWindowRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def insert_window(
        self,
        window_id: str,
        window_name: str,
        start_time: datetime,
        end_time: datetime,
        purpose: str,
        allows_training: bool = False,
        allows_monitoring_label: bool = False,
        allows_diagnosis_label: bool = False,
        allows_iteration_label: bool = False,
        allows_deployment_label: bool = False,
        is_frozen: bool = False,
    ) -> dict:
        await self.session.execute(
            text("""
                INSERT INTO model_registry.data_windows
                    (window_id, window_name, start_time, end_time, purpose,
                     allows_training, allows_monitoring_label,
                     allows_diagnosis_label, allows_iteration_label,
                     allows_deployment_label, is_frozen)
                VALUES (:wid, :name, :st, :et, :purpose,
                        :at, :am, :ad, :ai, :adep, :frozen)
            """),
            {
                "wid": window_id, "name": window_name,
                "st": start_time, "et": end_time, "purpose": purpose,
                "at": allows_training, "am": allows_monitoring_label,
                "ad": allows_diagnosis_label, "ai": allows_iteration_label,
                "adep": allows_deployment_label, "frozen": is_frozen,
            },
        )
        return {"window_id": window_id}

    async def get_window(self, window_id: str) -> dict | None:
        result = await self.session.execute(
            text("SELECT * FROM model_registry.data_windows WHERE window_id = :wid"),
            {"wid": window_id},
        )
        row = result.mappings().first()
        return dict(row) if row else None

    async def list_windows(self) -> list[dict]:
        result = await self.session.execute(
            text("SELECT * FROM model_registry.data_windows ORDER BY start_time")
        )
        return [dict(row) for row in result.mappings()]
