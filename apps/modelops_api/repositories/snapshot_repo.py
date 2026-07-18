"""model_registry.dataset_snapshots 数据访问"""

from __future__ import annotations

import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class SnapshotRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def insert_snapshot(
        self,
        storage_uri: str,
        content_hash: str,
        model_id: str | None = None,
        window_id: str | None = None,
        data_track: str = "NATURAL",
        row_count: int | None = None,
        column_count: int | None = None,
        feature_schema_version: str | None = None,
        label_maturity_time=None,
    ) -> dict:
        new_id = str(uuid.uuid4())
        await self.session.execute(
            text("""
                INSERT INTO model_registry.dataset_snapshots
                    (dataset_snapshot_id, model_id, window_id, data_track,
                     storage_uri, content_hash, row_count, column_count,
                     feature_schema_version, label_maturity_time)
                VALUES (:id, :mid, :wid, :track, :uri, :hash,
                        :rows, :cols, :fsv, :lmt)
            """),
            {
                "id": new_id, "mid": model_id, "wid": window_id,
                "track": data_track, "uri": storage_uri, "hash": content_hash,
                "rows": row_count, "cols": column_count,
                "fsv": feature_schema_version, "lmt": label_maturity_time,
            },
        )
        return {"dataset_snapshot_id": new_id}

    async def get_by_hash(self, content_hash: str) -> dict | None:
        result = await self.session.execute(
            text("""
                SELECT * FROM model_registry.dataset_snapshots
                WHERE content_hash = :hash
            """),
            {"hash": content_hash},
        )
        row = result.mappings().first()
        return dict(row) if row else None

    async def get_by_id(self, snapshot_id: str) -> dict | None:
        result = await self.session.execute(
            text("""
                SELECT * FROM model_registry.dataset_snapshots
                WHERE dataset_snapshot_id = :id
            """),
            {"id": snapshot_id},
        )
        row = result.mappings().first()
        return dict(row) if row else None
