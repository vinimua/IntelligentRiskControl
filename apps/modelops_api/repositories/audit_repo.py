"""audit.data_access_violations 数据访问"""

from __future__ import annotations

import json
import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class AuditRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def log_violation(
        self,
        task_phase: str,
        violation_code: str,
        model_id: str | None = None,
        window_id: str | None = None,
        attempted_operation: str | None = None,
        detail_json: dict | None = None,
    ) -> dict:
        new_id = str(uuid.uuid4())
        await self.session.execute(
            text("""
                INSERT INTO audit.data_access_violations
                    (violation_id, model_id, task_phase, window_id,
                     violation_code, attempted_operation, detail_json)
                VALUES (:id, :mid, :task, :wid, :code, :op, :detail::jsonb)
            """),
            {
                "id": new_id, "mid": model_id, "task": task_phase,
                "wid": window_id, "code": violation_code,
                "op": attempted_operation,
                "detail": json.dumps(detail_json or {}, ensure_ascii=False, default=str),
            },
        )
        return {"violation_id": new_id}
