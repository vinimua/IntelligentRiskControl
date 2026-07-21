"""workflow schema 数据访问"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class WorkflowRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    # ── lifecycle_runs ──

    async def create_run(
        self, model_id: str, champion_version: str, trigger_type: str = "SCHEDULED_TRIGGER"
    ) -> dict:
        new_id = str(uuid.uuid4())
        await self.session.execute(
            text("""
                INSERT INTO workflow.model_lifecycle_runs
                    (lifecycle_run_id, model_id, champion_version, trigger_type)
                VALUES (:id, :mid, :ver, :trigger)
            """),
            {"id": new_id, "mid": model_id, "ver": champion_version, "trigger": trigger_type},
        )
        return {"lifecycle_run_id": new_id}

    async def get_run(self, lifecycle_run_id: str) -> dict | None:
        result = await self.session.execute(
            text("SELECT * FROM workflow.model_lifecycle_runs WHERE lifecycle_run_id = :id"),
            {"id": lifecycle_run_id},
        )
        row = result.mappings().first()
        return dict(row) if row else None

    async def update_phase(self, lifecycle_run_id: str, phase: str, state_json: dict | None = None) -> None:
        sets = ["current_phase = :phase", "updated_at = NOW()"]
        params: dict = {"id": lifecycle_run_id, "phase": phase}
        if state_json is not None:
            sets.append("state_json = :state::jsonb")
            params["state"] = json.dumps(state_json, ensure_ascii=False, default=str)
        await self.session.execute(
            text(f"UPDATE workflow.model_lifecycle_runs SET {', '.join(sets)} WHERE lifecycle_run_id = :id"),
            params,
        )

    async def complete_run(self, lifecycle_run_id: str) -> None:
        await self.session.execute(
            text("""
                UPDATE workflow.model_lifecycle_runs
                SET completed_at = NOW(), updated_at = NOW()
                WHERE lifecycle_run_id = :id
            """),
            {"id": lifecycle_run_id},
        )

    async def set_manual_review(self, lifecycle_run_id: str, value: bool = True) -> None:
        await self.session.execute(
            text("""
                UPDATE workflow.model_lifecycle_runs
                SET requires_manual_review = :val, updated_at = NOW()
                WHERE lifecycle_run_id = :id
            """),
            {"id": lifecycle_run_id, "val": value},
        )

    # ── action_logs ──

    async def log_action(
        self,
        lifecycle_run_id: str,
        node_name: str,
        phase: str,
        action: str,
        status: str = "COMPLETED",
        duration_ms: int | None = None,
        summary: dict | None = None,
        error: dict | None = None,
    ) -> None:
        await self.session.execute(
            text("""
                INSERT INTO workflow.workflow_action_logs
                    (lifecycle_run_id, node_name, phase, action, status, duration_ms, summary_json, error_json)
                VALUES (:rid, :node, :phase, :action, :status, :dur, :summary::jsonb, :error::jsonb)
            """),
            {
                "rid": lifecycle_run_id,
                "node": node_name,
                "phase": phase,
                "action": action,
                "status": status,
                "dur": duration_ms,
                "summary": json.dumps(summary or {}, ensure_ascii=False, default=str),
                "error": json.dumps(error or {}, ensure_ascii=False, default=str) if error else None,
            },
        )

    # ── manual_review ──

    async def create_review_task(
        self, lifecycle_run_id: str, node_name: str, reason: str
    ) -> dict:
        new_id = str(uuid.uuid4())
        await self.session.execute(
            text("""
                INSERT INTO workflow.manual_review_tasks
                    (review_task_id, lifecycle_run_id, node_name, review_reason)
                VALUES (:id, :rid, :node, :reason)
            """),
            {"id": new_id, "rid": lifecycle_run_id, "node": node_name, "reason": reason},
        )
        return {"review_task_id": new_id}

    async def resolve_review_task(self, review_task_id: str, decision: str, reviewer: str = "system") -> None:
        await self.session.execute(
            text("""
                UPDATE workflow.manual_review_tasks
                SET status = 'RESOLVED', decision = :decision, reviewer = :reviewer, resolved_at = NOW()
                WHERE review_task_id = :id
            """),
            {"id": review_task_id, "decision": decision, "reviewer": reviewer},
        )

    # ── outbox ──

    async def push_outbox(
        self, lifecycle_run_id: str, event_type: str, payload: dict, idempotency_key: str
    ) -> dict:
        new_id = str(uuid.uuid4())
        await self.session.execute(
            text("""
                INSERT INTO workflow.outbox_events
                    (outbox_id, lifecycle_run_id, event_type, payload_json, idempotency_key)
                VALUES (:id, :rid, :type, :payload::jsonb, :key)
            """),
            {
                "id": new_id,
                "rid": lifecycle_run_id,
                "type": event_type,
                "payload": json.dumps(payload, ensure_ascii=False, default=str),
                "key": idempotency_key,
            },
        )
        return {"outbox_id": new_id}

    # ── inbox (idempotent dedup) ──

    async def receive_inbox(
        self, lifecycle_run_id: str, event_type: str, payload: dict, idempotency_key: str
    ) -> bool:
        """返回 True = 新事件已处理；False = 重复事件（幂等跳过）。"""
        result = await self.session.execute(
            text("SELECT inbox_id FROM workflow.inbox_events WHERE idempotency_key = :key"),
            {"key": idempotency_key},
        )
        if result.fetchone():
            return False  # 已处理过

        new_id = str(uuid.uuid4())
        await self.session.execute(
            text("""
                INSERT INTO workflow.inbox_events
                    (inbox_id, lifecycle_run_id, idempotency_key, event_type, payload_json, processed)
                VALUES (:id, :rid, :key, :type, :payload::jsonb, TRUE)
            """),
            {
                "id": new_id,
                "rid": lifecycle_run_id,
                "key": idempotency_key,
                "type": event_type,
                "payload": json.dumps(payload, ensure_ascii=False, default=str),
            },
        )
        return True
