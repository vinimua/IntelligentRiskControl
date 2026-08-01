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


def _bayesian_shrinkage(
    support_count: int,
    against_count: int,
    neutral_count: int,
    support_strength: float,
    against_strength: float,
    *,
    prior_alpha: float = 2.0,
    prior_beta: float = 8.0,
) -> dict:
    """贝叶斯 Beta-Binomial 收缩。

    先验 Beta(2, 8) → 均值 0.20（弱有效先验）。
    后验 Beta(α + support_strength, β + against_strength)。
    中性观测不改变后验均值但增加精度。
    """
    total = support_count + against_count + neutral_count
    if total <= 0:
        return {
            "new_weight": round(prior_alpha / (prior_alpha + prior_beta), 4),
            "alpha_post": prior_alpha,
            "beta_post": prior_beta,
            "confidence_lower": 0.0,
            "confidence_upper": 1.0,
        }

    # 强度加权：1 个 SUPPORT 证据 = 1.0 强度贡献
    alpha_post = prior_alpha + support_strength
    beta_post = prior_beta + against_strength

    # 后验均值
    posterior_mean = alpha_post / (alpha_post + beta_post)

    # 后验方差 → 95% 近似置信区间
    posterior_var = (alpha_post * beta_post) / (
        (alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1)
    )
    posterior_std = posterior_var ** 0.5
    confidence_lower = max(0.0, posterior_mean - 1.96 * posterior_std)
    confidence_upper = min(1.0, posterior_mean + 1.96 * posterior_std)

    # 缩水到 [0.03, 0.85] — 永远不极端
    new_weight = round(_clamp(posterior_mean, 0.03, 0.85), 4)

    return {
        "new_weight": new_weight,
        "alpha_post": round(alpha_post, 4),
        "beta_post": round(beta_post, 4),
        "confidence_lower": round(confidence_lower, 4),
        "confidence_upper": round(confidence_upper, 4),
    }


def _scenario_weight(support_strength: float, against_strength: float, neutral_count: int) -> float:
    """Deprecated: 旧版弱先验启发式，保留兼容。

    新版 _bayesian_shrinkage 使用 Beta-Binomial 收缩。
    """
    total_strength = support_strength + against_strength
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
                    obs.relation_key,
                    obs.evidence_case_count,
                    obs.natural_case_count,
                    obs.scenario_case_count,
                    obs.support_count,
                    obs.against_count,
                    obs.neutral_count,
                    obs.support_strength,
                    obs.against_strength,
                    prev.new_effective_weight AS old_effective_weight
                FROM (
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
                ) obs
                LEFT JOIN LATERAL (
                    SELECT new_effective_weight
                    FROM knowledge.kg_relation_weight_snapshots prev
                    WHERE prev.relation_key = obs.relation_key
                    ORDER BY prev.created_at DESC
                    LIMIT 1
                ) prev ON true
                ORDER BY obs.relation_key
                """,
                (data_track,),
            )
            rows = cur.fetchall()

            for row in rows:
                bayes = _bayesian_shrinkage(
                    support_count=int(row["support_count"]),
                    against_count=int(row["against_count"]),
                    neutral_count=int(row["neutral_count"]),
                    support_strength=float(row["support_strength"]),
                    against_strength=float(row["against_strength"]),
                )
                new_weight = bayes["new_weight"]
                confidence_lower = bayes["confidence_lower"]
                confidence_upper = bayes["confidence_upper"]
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
                        %(old_effective_weight)s, %(new_effective_weight)s,
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
                        "old_effective_weight": row["old_effective_weight"],
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
                        "snapshot_detail": Jsonb({
                            "rule": "BETA_BINOMIAL_V2",
                            "prior_alpha": 2.0,
                            "prior_beta": 8.0,
                            "posterior_alpha": bayes["alpha_post"],
                            "posterior_beta": bayes["beta_post"],
                        }),
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
    parser.add_argument("--data-track", default="NATURAL", choices=["SCENARIO", "NATURAL"])
    parser.add_argument("--rule-version", default="BETA_BINOMIAL_V2")
    parser.add_argument("--weight-version", default="KG_WEIGHT_BETA_V2")
    args = parser.parse_args()

    run_id = run_calibration(args.data_track, args.rule_version, args.weight_version)
    print(f"KG calibration completed: calibration_run_id={run_id}")


if __name__ == "__main__":
    main()
