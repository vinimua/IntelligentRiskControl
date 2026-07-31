"""diagnosis schema 数据访问 — diagnosis_runs / diagnosis_candidates / diagnosis_evidence"""

from __future__ import annotations

import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class DiagnosisRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_run(
        self,
        monitoring_run_id: str,
        lifecycle_run_id: str | None = None,
        alert_count: int = 0,
        event_id: str | None = None,
        logic_version: str = "V2_EVENT_TIME",
    ) -> dict:
        new_id = str(uuid.uuid4())
        await self.session.execute(
            text("""
                INSERT INTO diagnosis.diagnosis_runs
                    (diagnosis_run_id, lifecycle_run_id, monitoring_run_id,
                     alert_count, event_id, logic_version, status)
                VALUES (:id, :lid, :mid, :cnt, :event_id, :logic_version, 'RUNNING')
            """),
            {"id": new_id, "lid": lifecycle_run_id, "mid": monitoring_run_id,
             "cnt": alert_count, "event_id": event_id,
             "logic_version": logic_version},
        )
        return {"diagnosis_run_id": new_id}

    async def get_active_event(
        self, model_id: str, model_version: str
    ) -> dict | None:
        result = await self.session.execute(
            text("""
                SELECT * FROM diagnosis.diagnosis_events
                WHERE model_id = :model_id AND model_version = :model_version
                  AND status NOT IN ('CLOSED', 'CANCELLED')
                ORDER BY event_time, created_at LIMIT 1
            """),
            {"model_id": model_id, "model_version": model_version},
        )
        row = result.mappings().first()
        return dict(row) if row else None

    async def get_event(self, event_id: str) -> dict | None:
        result = await self.session.execute(
            text("SELECT * FROM diagnosis.diagnosis_events WHERE event_id = :event_id"),
            {"event_id": event_id},
        )
        row = result.mappings().first()
        return dict(row) if row else None

    async def create_event(
        self, model_id: str, model_version: str, monitoring_run_id: str,
        event_time, alert_ids: list[str],
    ) -> dict:
        if not alert_ids:
            raise ValueError("a diagnosis event requires at least one alert")
        event_id = str(uuid.uuid4())
        await self.session.execute(
            text("""
                INSERT INTO diagnosis.diagnosis_events
                    (event_id, model_id, model_version, monitoring_run_id,
                     event_time, primary_alert_id, status)
                VALUES (:event_id, :model_id, :model_version, :run_id,
                        :event_time, :primary_alert_id, 'OPEN')
            """),
            {"event_id": event_id, "model_id": model_id,
             "model_version": model_version, "run_id": monitoring_run_id,
             "event_time": event_time, "primary_alert_id": alert_ids[0]},
        )
        for alert_id in alert_ids:
            await self.session.execute(
                text("""
                    INSERT INTO diagnosis.diagnosis_event_alerts(event_id, alert_id)
                    VALUES (:event_id, :alert_id)
                    ON CONFLICT (alert_id) DO NOTHING
                """),
                {"event_id": event_id, "alert_id": alert_id},
            )
        return {"event_id": event_id, "status": "OPEN"}

    async def get_event_alert_ids(self, event_id: str) -> list[str]:
        result = await self.session.execute(
            text("""
                SELECT alert_id FROM diagnosis.diagnosis_event_alerts
                WHERE event_id = :event_id ORDER BY attached_at, alert_id
            """),
            {"event_id": event_id},
        )
        return [str(row["alert_id"]) for row in result.mappings()]

    async def mark_event_diagnosed(self, event_id: str) -> None:
        await self.session.execute(
            text("""
                UPDATE diagnosis.diagnosis_events
                SET status = 'WAITING_AGENT_DECISION', updated_at = NOW()
                WHERE event_id = :event_id AND status = 'OPEN'
            """),
            {"event_id": event_id},
        )

    async def update_event_status(self, event_id: str, status: str) -> None:
        await self.session.execute(
            text("""
                UPDATE diagnosis.diagnosis_events
                SET status = :status, updated_at = NOW()
                WHERE event_id = :event_id
            """),
            {"event_id": event_id, "status": status},
        )

    async def mark_event_in_repair(self, event_id: str) -> None:
        await self.update_event_status(event_id, "IN_REPAIR")

    async def close_event(self, event_id: str) -> None:
        await self.update_event_status(event_id, "CLOSED")

    async def complete_run(
        self,
        diagnosis_run_id: str,
        primary_root_cause_code: str | None = None,
        primary_root_cause_dimension: str | None = None,
        primary_root_cause_score: float | None = None,
        recommended_action: str | None = None,
        need_iteration: bool | None = None,
        status: str = "COMPLETED",
    ) -> None:
        await self.session.execute(
            text("""
                UPDATE diagnosis.diagnosis_runs
                SET primary_root_cause_code = :rc,
                    primary_root_cause_dimension = :dim,
                    primary_root_cause_score = :score,
                    recommended_action = :action,
                    need_iteration = :ni,
                    status = :status,
                    completed_at = NOW()
                WHERE diagnosis_run_id = :id
            """),
            {"id": diagnosis_run_id, "rc": primary_root_cause_code,
             "dim": primary_root_cause_dimension, "score": primary_root_cause_score,
             "action": recommended_action, "ni": need_iteration, "status": status},
        )

    async def batch_insert_candidates(
        self, diagnosis_run_id: str, candidates: list[dict]
    ) -> dict[str, str]:
        """Persist candidates and return root-cause-code -> candidate-id."""
        candidate_ids: dict[str, str] = {}
        for c in candidates:
            cid = str(uuid.uuid4())
            await self.session.execute(
                text("""
                    INSERT INTO diagnosis.diagnosis_candidates
                        (candidate_id, diagnosis_run_id, alert_code,
                         root_cause_code, dimension_code, relation_key,
                         effective_weight_snapshot, evidence_case_count_snapshot,
                         confidence_lower_bound_snapshot,
                         ranked_score, rank_no, is_primary)
                    VALUES (:cid, :rid, :alert, :rc, :dim, :rkey,
                            :w, :ec, :cl, :score, :rank, :primary)
                """),
                {
                    "cid": cid, "rid": diagnosis_run_id,
                    "alert": c["alert_code"], "rc": c["root_cause_code"],
                    "dim": c["dimension_code"], "rkey": c["relation_key"],
                    "w": c.get("effective_weight", 0),
                    "ec": c.get("evidence_case_count", 0),
                    "cl": c.get("confidence_lower_bound", 0),
                    "score": c.get("ranked_score"),
                    "rank": c.get("rank_no"),
                    "primary": c.get("is_primary", False),
                },
            )
            candidate_ids[c["root_cause_code"]] = cid
        return candidate_ids

    async def insert_evidence(self, evidence: dict) -> str:
        eid = str(uuid.uuid4())
        await self.session.execute(
            text("""
                INSERT INTO diagnosis.diagnosis_evidence
                    (evidence_id, diagnosis_run_id, candidate_id,
                     hypothesis_code, evidence_type, method_code,
                     normalized_score, direction, applicable, evidence_detail_json)
                VALUES (:eid, :rid, :cid, :hyp, :etype, :method,
                        :score, :dir, :app, :det)
            """),
            {
                "eid": eid, "rid": evidence["diagnosis_run_id"],
                "cid": evidence["candidate_id"],
                "hyp": evidence.get("hypothesis_code"),
                "etype": evidence["evidence_type"],
                "method": evidence["method_code"],
                "score": evidence.get("normalized_score"),
                "dir": evidence.get("direction"),
                "app": evidence.get("applicable", True),
                "det": evidence.get("evidence_detail_json", "{}"),
            },
        )
        return eid

    async def get_run(self, diagnosis_run_id: str) -> dict | None:
        result = await self.session.execute(
            text("SELECT * FROM diagnosis.diagnosis_runs WHERE diagnosis_run_id = :id"),
            {"id": diagnosis_run_id},
        )
        row = result.mappings().first()
        return dict(row) if row else None

    async def list_runs(self, limit: int = 20) -> list[dict]:
        result = await self.session.execute(
            text("SELECT * FROM diagnosis.diagnosis_runs ORDER BY created_at DESC LIMIT :lim"),
            {"lim": limit},
        )
        return [dict(row) for row in result.mappings()]

    async def get_run_by_monitoring(self, monitoring_run_id: str) -> dict | None:
        """查询某个监控运行对应的最新诊断运行。"""
        result = await self.session.execute(
            text("""
                SELECT * FROM diagnosis.diagnosis_runs
                WHERE monitoring_run_id = :mid
                  AND status <> 'LEGACY_INVALID'
                ORDER BY created_at DESC LIMIT 1
            """),
            {"mid": monitoring_run_id},
        )
        row = result.mappings().first()
        return dict(row) if row else None

    async def get_run_by_event(self, event_id: str) -> dict | None:
        result = await self.session.execute(
            text("""
                SELECT * FROM diagnosis.diagnosis_runs
                WHERE event_id = :event_id AND status <> 'LEGACY_INVALID'
                ORDER BY created_at DESC LIMIT 1
            """),
            {"event_id": event_id},
        )
        row = result.mappings().first()
        return dict(row) if row else None

    async def get_event_by_diagnosis_run(self, diagnosis_run_id: str) -> dict | None:
        """通过诊断运行 ID 查找关联的诊断事件。

        先从 diagnosis_runs 中读取 event_id，再查询 diagnosis_events。
        """
        run = await self.get_run(diagnosis_run_id)
        if not run or not run.get("event_id"):
            return None
        return await self.get_event(str(run["event_id"]))

    async def get_candidates(self, diagnosis_run_id: str) -> list[dict]:
        """查询某个诊断运行的所有候选根因（按 rank_no 排序）。"""
        result = await self.session.execute(
            text("""
                SELECT * FROM diagnosis.diagnosis_candidates
                WHERE diagnosis_run_id = :did
                ORDER BY rank_no ASC
            """),
            {"did": diagnosis_run_id},
        )
        return [dict(row) for row in result.mappings()]

    async def get_evidence_for_run(self, diagnosis_run_id: str) -> list[dict]:
        """查询某个诊断运行的所有证据项。"""
        result = await self.session.execute(
            text("""
                SELECT * FROM diagnosis.diagnosis_evidence
                WHERE diagnosis_run_id = :did
                ORDER BY created_at
            """),
            {"did": diagnosis_run_id},
        )
        return [dict(row) for row in result.mappings()]
