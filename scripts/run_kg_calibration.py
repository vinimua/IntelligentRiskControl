"""Aggregate KG relation observations into weight snapshots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from apps.modelops_api.config import settings


def _dsn() -> str:
    return settings.database_url_sync.replace("postgresql+psycopg://", "postgresql://")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _scenario_weight(support_strength: float, against_strength: float, neutral_count: int) -> float:
    total_strength = support_strength + against_strength
    if total_strength <= 0 and neutral_count > 0:
        return 0.10
    if total_strength <= 0:
        return 0.10

    net = (support_strength - against_strength) / total_strength
    return round(_clamp(0.10 + 0.25 * net, 0.03, 0.35), 4)


def run_calibration(
    data_track: str,
    rule_version: str,
    weight_version: str,
) -> str:
    data_track = data_track.upper()
    if data_track not in {"NATURAL", "SCENARIO"}:
        raise ValueError("data_track must be NATURAL or SCENARIO")

    with psycopg.connect(_dsn(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO knowledge.kg_calibration_runs (
                    data_track, calibration_rule_version, target_weight_version,
                    status, started_at
                )
                VALUES (%s, %s, %s, 'RUNNING', NOW())
                RETURNING calibration_run_id
                """,
                (data_track, rule_version, weight_version),
            )
            calibration_run_id = str(cur.fetchone()["calibration_run_id"])

            cur.execute(
                """
                SELECT
                    relation_key,
                    COUNT(*) AS evidence_case_count,
                    COUNT(*) FILTER (WHERE data_track = 'NATURAL') AS natural_case_count,
                    COUNT(*) FILTER (WHERE data_track = 'SCENARIO') AS scenario_case_count,
                    COUNT(*) FILTER (WHERE direction = 'SUPPORT') AS support_count,
                    COUNT(*) FILTER (WHERE direction = 'AGAINST') AS against_count,
                    COUNT(*) FILTER (WHERE direction = 'NEUTRAL') AS neutral_count,
                    COALESCE(SUM(weighted_strength) FILTER (WHERE direction = 'SUPPORT'), 0.0) AS support_strength,
                    COALESCE(SUM(weighted_strength) FILTER (WHERE direction = 'AGAINST'), 0.0) AS against_strength
                FROM knowledge.kg_relation_observations
                WHERE data_track = %s
                GROUP BY relation_key
                ORDER BY relation_key
                """,
                (data_track,),
            )
            rows = cur.fetchall()

            for row in rows:
                new_weight = _scenario_weight(
                    float(row["support_strength"]),
                    float(row["against_strength"]),
                    int(row["neutral_count"]),
                )
                confidence_lower = 0.0
                confidence_upper = min(1.0, new_weight + 0.15)
                cur.execute(
                    """
                    INSERT INTO knowledge.kg_relation_weight_snapshots (
                        calibration_run_id, relation_key,
                        old_effective_weight, new_effective_weight,
                        confidence_lower_bound, confidence_upper_bound,
                        evidence_case_count, natural_case_count, scenario_case_count,
                        support_count, against_count, neutral_count,
                        support_strength, against_strength,
                        weight_version, snapshot_detail
                    )
                    VALUES (
                        %(calibration_run_id)s, %(relation_key)s,
                        NULL, %(new_effective_weight)s,
                        %(confidence_lower_bound)s, %(confidence_upper_bound)s,
                        %(evidence_case_count)s, %(natural_case_count)s, %(scenario_case_count)s,
                        %(support_count)s, %(against_count)s, %(neutral_count)s,
                        %(support_strength)s, %(against_strength)s,
                        %(weight_version)s, %(snapshot_detail)s
                    )
                    ON CONFLICT (calibration_run_id, relation_key) DO NOTHING
                    """,
                    {
                        "calibration_run_id": calibration_run_id,
                        "relation_key": row["relation_key"],
                        "new_effective_weight": new_weight,
                        "confidence_lower_bound": confidence_lower,
                        "confidence_upper_bound": confidence_upper,
                        "evidence_case_count": row["evidence_case_count"],
                        "natural_case_count": row["natural_case_count"],
                        "scenario_case_count": row["scenario_case_count"],
                        "support_count": row["support_count"],
                        "against_count": row["against_count"],
                        "neutral_count": row["neutral_count"],
                        "support_strength": row["support_strength"],
                        "against_strength": row["against_strength"],
                        "weight_version": weight_version,
                        "snapshot_detail": Jsonb({"rule": "weak_scenario_prior_v1"}),
                    },
                )

            cur.execute(
                """
                UPDATE knowledge.kg_calibration_runs
                SET status = 'SUCCEEDED',
                    relation_count = %s,
                    observation_count = (
                        SELECT COUNT(*) FROM knowledge.kg_relation_observations
                        WHERE data_track = %s
                    ),
                    completed_at = NOW()
                WHERE calibration_run_id = %s
                """,
                (len(rows), data_track, calibration_run_id),
            )
        conn.commit()

    return calibration_run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-track", default="SCENARIO", choices=["SCENARIO", "NATURAL"])
    parser.add_argument("--rule-version", default="KG_CALIBRATION_WEAK_PRIOR_V1")
    parser.add_argument("--weight-version", default="SCENARIO_INIT_V1")
    args = parser.parse_args()

    run_id = run_calibration(args.data_track, args.rule_version, args.weight_version)
    print(f"KG calibration completed: calibration_run_id={run_id}")


if __name__ == "__main__":
    main()
