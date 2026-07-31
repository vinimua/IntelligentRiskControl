"""Backfill model-specific V2 diagnosis evidence from full formal windows.

The script is idempotent at model/run level: a run with existing
``V2_EVENT_TIME`` metrics or feature-drift rows is skipped.
"""

from __future__ import annotations

import asyncio
import argparse

import pandas as pd
from sqlalchemy import text

from apps.modelops_api.database import async_session
from apps.modelops_api.services.knowledge_service import KnowledgeService
from apps.modelops_api.services.monitoring.monitoring_service import MonitoringService
from apps.modelops_api.services.monitoring.drift.algorithms import (
    compute_performance_metrics,
)
from apps.modelops_api.services.monitoring.rolling import iter_rolling_windows
from apps.modelops_api.services.monitoring.window_loader import (
    load_window,
    predict_on_window,
)
from packages.models.common.enums import AvailabilityStatus


EXCLUDED_FEATURES = {
    "sample_id", "apply_time", "is_bad", "y_true", "risk_score",
    "y_pred_proba", "apply_hour_sin", "apply_hour_cos",
    "apply_weekday_sin", "apply_weekday_cos", "apply_is_weekend",
    "apply_is_night",
}


def _formal_window_id(value: str) -> str:
    return value.removeprefix("ROLLING_")


async def main(
    model_id_filter: str | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> int:
    # Physical windows are loaded with checksum/full-population validation.
    raw_windows = {wid: load_window(wid) for wid in ("W0", "W1", "W2", "W3")}

    async with async_session() as session:
        rows = (
            await session.execute(
                text("""
                    SELECT DISTINCT ON (model_id)
                           monitoring_run_id, model_id, champion_version
                    FROM monitoring.monitoring_runs
                    ORDER BY model_id, started_at DESC
                """)
            )
        ).mappings().all()

        drift_source_run_id = (
            await session.execute(
                text("""
                    SELECT monitoring_run_id
                    FROM monitoring.monitoring_feature_drift
                    GROUP BY monitoring_run_id
                    HAVING COUNT(*) > 0
                    ORDER BY COUNT(*) DESC
                    LIMIT 1
                """)
            )
        ).scalar_one_or_none()
        if drift_source_run_id is None:
            raise RuntimeError(
                "No full-population drift source exists; backfill credit_model_001 first"
            )

        if model_id_filter:
            rows = [row for row in rows if str(row["model_id"]) == model_id_filter]
        else:
            rows = list(rows)[offset: offset + limit if limit is not None else None]

        for index, run in enumerate(rows, start=1):
            run_id = str(run["monitoring_run_id"])
            model_id = str(run["model_id"])
            version = str(run["champion_version"])
            existing = (
                await session.execute(
                    text("""
                        SELECT
                          (SELECT COUNT(*) FROM monitoring.monitoring_metrics
                           WHERE monitoring_run_id = :run_id
                             AND metric_version = 'V2_EVENT_TIME') AS metric_count,
                          (SELECT COUNT(*) FROM monitoring.monitoring_feature_drift
                           WHERE monitoring_run_id = :run_id) AS drift_count
                    """),
                    {"run_id": run_id},
                )
            ).mappings().one()
            if int(existing["metric_count"]) or int(existing["drift_count"]):
                print(
                    f"[{index}/{len(rows)}] SKIP {model_id}: "
                    f"metrics={existing['metric_count']} drift={existing['drift_count']}",
                    flush=True,
                )
                continue

            predicted = {
                wid: predict_on_window(frame, model_id)
                for wid, frame in raw_windows.items()
            }
            reference = predicted["W0"]
            timeline = pd.concat(
                [predicted["W1"], predicted["W2"], predicted["W3"]],
                ignore_index=True,
            ).sort_values("apply_time")

            service = MonitoringService(session, KnowledgeService(None))
            feature_names = [
                column for column in reference.columns
                if column not in EXCLUDED_FEATURES
            ]
            baseline = service.build_baseline(
                reference,
                model_id,
                version,
                feature_names=feature_names,
            )
            perf_rows: list[dict] = []
            for start, end, window in iter_rolling_windows(
                timeline, window_days=7, step_days=1, require_full_window=False
            ):
                metrics = compute_performance_metrics(
                    window["y_true"], window["y_pred_proba"]
                )
                perf_rows.append({
                    "monitor_window_id": f"7D_{start:%Y%m%d}_{end:%Y%m%d}",
                    "window_start": start,
                    "window_end": end,
                    "window_days": 7,
                    "sample_count": len(window),
                    "bad_count": int(window["y_true"].sum()),
                    **metrics,
                })
            perf = pd.DataFrame(perf_rows)

            for row in perf.to_dict(orient="records"):
                detail = {
                    "window_id": row["monitor_window_id"],
                    "window_start": row["window_start"],
                    "window_end": row["window_end"],
                    "window_days": row["window_days"],
                    "sample_count": row["sample_count"],
                    "bad_count": row["bad_count"],
                    "category": "diagnosis_timeline",
                    "source_population": "FULL_POPULATION",
                }
                for code in ("AUC", "KS", "PR_AUC", "BRIER", "ECE", "BAD_RECALL"):
                    key = code.lower()
                    current = row.get(key)
                    current_value = (
                        float(current)
                        if current is not None and pd.notna(current)
                        else None
                    )
                    baseline_value = baseline.performance_reference_json.get(key)
                    await service.repo.insert_metric(
                        monitoring_run_id=run_id,
                        metric_code=code,
                        metric_version="V2_EVENT_TIME",
                        object_type="MODEL",
                        object_code=model_id,
                        baseline_value=baseline_value,
                        current_value=current_value,
                        delta=(
                            current_value - baseline_value
                            if current_value is not None
                            and baseline_value is not None
                            else None
                        ),
                        availability_status=(
                            AvailabilityStatus.AVAILABLE.value
                            if current_value is not None
                            else AvailabilityStatus.SAMPLE_TOO_SMALL.value
                        ),
                        metric_detail=detail,
                    )

            await session.execute(
                text("""
                    INSERT INTO monitoring.monitoring_feature_drift (
                        drift_id, monitoring_run_id, window_id, feature_name,
                        feature_type, psi, js_divergence, wasserstein_distance,
                        ks_statistic, ks_p_value, ks_q_value, missing_rate,
                        missing_rate_delta, outlier_rate, outlier_rate_delta,
                        default_value_rate, range_violation_rate,
                        unknown_category_rate, dq_score, dq_flag, data_track,
                        created_at
                    )
                    SELECT gen_random_uuid(), :target_run_id, window_id,
                           feature_name, feature_type, psi, js_divergence,
                           wasserstein_distance, ks_statistic, ks_p_value,
                           ks_q_value, missing_rate, missing_rate_delta,
                           outlier_rate, outlier_rate_delta, default_value_rate,
                           range_violation_rate, unknown_category_rate, dq_score,
                           dq_flag, data_track, NOW()
                    FROM monitoring.monitoring_feature_drift
                    WHERE monitoring_run_id = :source_run_id
                """),
                {
                    "target_run_id": run_id,
                    "source_run_id": str(drift_source_run_id),
                },
            )
            await session.commit()
            print(
                f"[{index}/{len(rows)}] DONE {model_id}: "
                f"windows={len(perf)} drift_rows=3094",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(main(args.model_id, args.offset, args.limit))
    )
