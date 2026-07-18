"""model_registry.models + model_versions 数据访问"""

from __future__ import annotations

import json
import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class ModelRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def insert_model(
        self, model_id: str, model_name: str, model_type: str = "CREDIT_RISK"
    ) -> dict:
        await self.session.execute(
            text("""
                INSERT INTO model_registry.models (model_id, model_name, model_type)
                VALUES (:model_id, :model_name, :model_type)
            """),
            {"model_id": model_id, "model_name": model_name, "model_type": model_type},
        )
        return {"model_id": model_id}

    async def get_model(self, model_id: str) -> dict | None:
        result = await self.session.execute(
            text("SELECT * FROM model_registry.models WHERE model_id = :mid"),
            {"mid": model_id},
        )
        row = result.mappings().first()
        return dict(row) if row else None

    async def list_models(
        self,
        status: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict]:
        sql = "SELECT * FROM model_registry.models"
        params: dict = {}
        if status:
            sql += " WHERE status = :status"
            params["status"] = status
        sql += " ORDER BY created_at DESC"
        if limit is not None:
            sql += " LIMIT :limit OFFSET :offset"
            params["limit"] = limit
            params["offset"] = offset
        result = await self.session.execute(text(sql), params)
        return [dict(row) for row in result.mappings()]

    async def count_models(self, status: str | None = None) -> int:
        sql = "SELECT COUNT(*) FROM model_registry.models"
        params: dict = {}
        if status:
            sql += " WHERE status = :status"
            params["status"] = status
        result = await self.session.execute(text(sql), params)
        return int(result.scalar_one())

    async def set_champion(self, model_id: str, version_code: str) -> None:
        await self.session.execute(
            text("""
                UPDATE model_registry.models
                SET current_champion_version = :ver, updated_at = NOW()
                WHERE model_id = :mid
            """),
            {"ver": version_code, "mid": model_id},
        )

    async def set_stable(self, model_id: str, version_code: str) -> None:
        await self.session.execute(
            text("""
                UPDATE model_registry.models
                SET stable_version = :ver, updated_at = NOW()
                WHERE model_id = :mid
            """),
            {"ver": version_code, "mid": model_id},
        )

    async def insert_version(
        self,
        model_id: str,
        version_code: str,
        role: str = "CHALLENGER",
        base_version_code: str | None = None,
        mlflow_run_id: str | None = None,
        artifact_uri: str | None = None,
        metrics_json: dict | None = None,
    ) -> dict:
        new_id = str(uuid.uuid4())
        await self.session.execute(
            text("""
                INSERT INTO model_registry.model_versions
                    (model_version_id, model_id, version_code, role, base_version_code,
                     mlflow_run_id, artifact_uri, metrics_json)
                VALUES (:id, :mid, :ver, :role, :base, :mlflow, :uri, :metrics::jsonb)
            """),
            {
                "id": new_id,
                "mid": model_id,
                "ver": version_code,
                "role": role,
                "base": base_version_code,
                "mlflow": mlflow_run_id,
                "uri": artifact_uri,
                "metrics": json.dumps(metrics_json or {}, ensure_ascii=False, default=str),
            },
        )
        return {"model_version_id": new_id}

    async def get_version(self, model_id: str, version_code: str) -> dict | None:
        result = await self.session.execute(
            text("""
                SELECT * FROM model_registry.model_versions
                WHERE model_id = :mid AND version_code = :ver
            """),
            {"mid": model_id, "ver": version_code},
        )
        row = result.mappings().first()
        return dict(row) if row else None

    async def list_versions(self, model_id: str) -> list[dict]:
        result = await self.session.execute(
            text("""
                SELECT * FROM model_registry.model_versions
                WHERE model_id = :mid ORDER BY created_at DESC
            """),
            {"mid": model_id},
        )
        return [dict(row) for row in result.mappings()]
