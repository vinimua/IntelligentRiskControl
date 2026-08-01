"""KG 观测与校准数据访问 — knowledge schema 下的三张表 + kg_sync_jobs。

PostgreSQL 是权重真相源；Neo4j 是投影。
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from packages.models.knowledge.kg_entity import KgRelationObservation
from packages.models.knowledge.calibration import (
    CalibrationRun,
    RelationWeightSnapshot,
)


class KnowledgeObservationRepo:
    """kg_relation_observations 的读写。"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def write_observation(
        self,
        relation_key: str,
        source_domain: str,
        source_record_id: str,
        direction: str,
        *,
        lifecycle_run_id: str | None = None,
        evidence_score: float | None = None,
        quality_weight: float = 1.0,
        data_track: str = "NATURAL",
        evidence_detail: dict | None = None,
    ) -> str:
        """幂等写入一条观测。重复写入同一 relation+domain+record 会 UPDATE 而非报错。"""
        obs_id = str(uuid.uuid4())
        weighted = (
            round(evidence_score * quality_weight, 4)
            if evidence_score is not None
            else None
        )
        now = datetime.now(timezone.utc)
        await self.session.execute(
            text("""
                INSERT INTO knowledge.kg_relation_observations
                    (observation_id, relation_key, source_domain, source_record_id,
                     lifecycle_run_id, direction, evidence_score, quality_weight,
                     weighted_strength, data_track, evidence_detail, observed_at, created_at)
                VALUES
                    (:id, :rk, :sd, :srid, :lrid, :dir, :score, :qw,
                     :ws, :track, :det, :now, :now)
                ON CONFLICT (relation_key, source_domain, source_record_id)
                DO UPDATE SET
                    direction         = EXCLUDED.direction,
                    evidence_score    = EXCLUDED.evidence_score,
                    quality_weight    = EXCLUDED.quality_weight,
                    weighted_strength = EXCLUDED.weighted_strength,
                    evidence_detail   = EXCLUDED.evidence_detail,
                    observed_at       = EXCLUDED.observed_at
            """),
            {
                "id": obs_id, "rk": relation_key, "sd": source_domain,
                "srid": source_record_id, "lrid": lifecycle_run_id,
                "dir": direction, "score": evidence_score, "qw": quality_weight,
                "ws": weighted, "track": data_track,
                "det": evidence_detail or {}, "now": now,
            },
        )
        return obs_id

    async def write_observations_batch(
        self, observations: list[KgRelationObservation],
    ) -> list[str]:
        """批量写入观测。"""
        ids: list[str] = []
        for obs in observations:
            oid = await self.write_observation(
                relation_key=obs.relation_key,
                source_domain=obs.source_domain,
                source_record_id=obs.source_record_id,
                direction=obs.direction,
                lifecycle_run_id=obs.lifecycle_run_id,
                evidence_score=obs.evidence_score,
                quality_weight=obs.quality_weight,
                data_track=obs.data_track,
                evidence_detail=obs.evidence_detail,
            )
            ids.append(oid)
        return ids

    async def get_observations_for_run(
        self, lifecycle_run_id: str,
    ) -> list[dict]:
        result = await self.session.execute(
            text("""
                SELECT * FROM knowledge.kg_relation_observations
                WHERE lifecycle_run_id = :lrid
                ORDER BY observed_at
            """),
            {"lrid": lifecycle_run_id},
        )
        return [dict(row) for row in result.mappings()]


class KgCalibrationRepo:
    """kg_calibration_runs 和 kg_relation_weight_snapshots 的读写。"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_run(
        self,
        data_track: str,
        rule_version: str,
        weight_version: str,
    ) -> str:
        run_id = str(uuid.uuid4())
        await self.session.execute(
            text("""
                INSERT INTO knowledge.kg_calibration_runs
                    (calibration_run_id, data_track, calibration_rule_version,
                     target_weight_version, status, started_at)
                VALUES (:id, :track, :rule, :wv, 'RUNNING', NOW())
            """),
            {"id": run_id, "track": data_track, "rule": rule_version, "wv": weight_version},
        )
        return run_id

    async def write_snapshot(self, snapshot: dict) -> str:
        sid = str(uuid.uuid4())
        await self.session.execute(
            text("""
                INSERT INTO knowledge.kg_relation_weight_snapshots (
                    snapshot_id, calibration_run_id, relation_key,
                    old_effective_weight, new_effective_weight,
                    confidence_lower_bound, confidence_upper_bound,
                    evidence_case_count, natural_case_count, scenario_case_count,
                    support_count, against_count, neutral_count,
                    support_strength, against_strength,
                    weight_version, snapshot_detail
                ) VALUES (
                    :sid, :rid, :rk,
                    :old_w, :new_w,
                    :clb, :cub,
                    :ec, :nc, :sc,
                    :sup, :ag, :neu,
                    :sup_str, :ag_str,
                    :wv, :det
                )
                ON CONFLICT (calibration_run_id, relation_key) DO NOTHING
            """),
            snapshot | {"sid": sid},
        )
        return sid

    async def mark_run_completed(self, run_id: str, relation_count: int, observation_count: int) -> None:
        await self.session.execute(
            text("""
                UPDATE knowledge.kg_calibration_runs
                SET status = 'SUCCEEDED',
                    relation_count = :rc,
                    observation_count = :oc,
                    completed_at = NOW()
                WHERE calibration_run_id = :id
            """),
            {"id": run_id, "rc": relation_count, "oc": observation_count},
        )
