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
    ) -> dict:
        new_id = str(uuid.uuid4())
        await self.session.execute(
            text("""
                INSERT INTO diagnosis.diagnosis_runs
                    (diagnosis_run_id, lifecycle_run_id, monitoring_run_id,
                     alert_count, status)
                VALUES (:id, :lid, :mid, :cnt, 'RUNNING')
            """),
            {"id": new_id, "lid": lifecycle_run_id, "mid": monitoring_run_id,
             "cnt": alert_count},
        )
        return {"diagnosis_run_id": new_id}

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
    ) -> int:
        inserted = 0
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
            inserted += 1
        return inserted

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
                ORDER BY created_at DESC LIMIT 1
            """),
            {"mid": monitoring_run_id},
        )
        row = result.mappings().first()
        return dict(row) if row else None

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
