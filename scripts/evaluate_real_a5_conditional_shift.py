"""Build and evaluate real-row conditional calibration-drift scenarios.

Rows, feature values and labels all come from the competition W2/W3 snapshots.
The injection only performs deterministic stratified selection.  Within frozen
Champion score strata it changes P(is_bad | score) while keeping the marginal
score-stratum distribution.  This is intentionally different from a pure prior
shift, which the formal policy must route to A2 instead of A5.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from apps.modelops_api.services.iteration.action_execution_service import (
    _apply_calibrator,
    _bundle,
    _raw_scores,
    execute_calibration,
    qualify_adjustment,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "formal_a1_a7" / "a5_conditional_formal_model"
SEED = 20260815
MODEL_ID = "credit_formal_logistic_a5"
VERSION = "test_v1"
BUNDLE_URI = ROOT / "assets/test_models/formal_a1_a7" / MODEL_ID / VERSION
SLOPES = (0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04)
TARGET_BAD_RATE = 0.05
QUANTILE_LEVELS = tuple(np.linspace(0.0, 1.0, 11))


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-5, 1 - 1e-5)
    return np.log(clipped / (1 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _solve_intercept(base_rates: np.ndarray, weights: np.ndarray, slope: float) -> float:
    target = TARGET_BAD_RATE
    lo, hi = -8.0, 8.0
    logits = _logit(base_rates)
    for _ in range(100):
        mid = (lo + hi) / 2
        current = float(np.sum(_sigmoid(slope * logits + mid) * weights))
        if current < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _assign_bins(scores: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(scores, edges[1:-1], right=True), 0, len(edges) - 2)


def _select_real_rows(
    frame: pd.DataFrame,
    bins: np.ndarray,
    target_rates: np.ndarray,
    controlled: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    source = frame.copy()
    source["__score_bin"] = bins
    groups = [source.loc[source["__score_bin"] == index] for index in range(len(target_rates))]
    effective_rates = []
    capacities = []
    for group, rate, is_controlled in zip(groups, target_rates, controlled, strict=True):
        bads = int(group["is_bad"].sum())
        goods = len(group) - bads
        if not is_controlled:
            effective_rates.append(float(group["is_bad"].mean()))
            capacities.append(1.0)
            continue
        # A stratum with no observed member of one class cannot synthesize that
        # class.  Keep its observed class composition instead of relabeling.
        rate = 0.0 if bads == 0 else (1.0 if goods == 0 else float(rate))
        effective_rates.append(rate)
        bad_capacity = bads / (len(group) * rate) if rate > 0 else 1.0
        good_capacity = goods / (len(group) * (1 - rate)) if rate < 1 else 1.0
        capacities.append(min(bad_capacity, good_capacity, 1.0))
    fraction = 0.90 * min(capacities)
    if fraction <= 0:
        raise ValueError("CONDITIONAL_SCENARIO_HAS_NO_REAL_ROW_CAPACITY")

    selected = []
    for index, (group, rate, is_controlled) in enumerate(
        zip(groups, effective_rates, controlled, strict=True)
    ):
        total = max(2, int(np.floor(len(group) * fraction)))
        if not is_controlled:
            selected.append(group.sample(total, random_state=seed + index * 2))
            continue
        wanted_bad = max(0, min(total, int(round(total * rate))))
        wanted_good = total - wanted_bad
        bad = group.loc[group["is_bad"].astype(int) == 1]
        good = group.loc[group["is_bad"].astype(int) == 0]
        if wanted_bad > len(bad) or wanted_good > len(good):
            raise ValueError(
                f"CONDITIONAL_SCENARIO_CAPACITY_ERROR:bin={index}:"
                f"want={wanted_bad}/{wanted_good}:have={len(bad)}/{len(good)}"
            )
        selected.append(bad.sample(wanted_bad, random_state=seed + index * 2))
        selected.append(good.sample(wanted_good, random_state=seed + index * 2 + 1))
    return (
        pd.concat(selected, ignore_index=True)
        .drop(columns=["__score_bin"])
        .sort_values(["apply_time", "sample_id"], kind="stable")
        .reset_index(drop=True)
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frames = {
        name: pd.read_parquet(ROOT / f"assets/data/windows/{name}/data.parquet")
        for name in ("W1", "W2", "W3")
    }
    bundle = _bundle(
        {
            "model_id": MODEL_ID,
            "champion_version": VERSION,
            "champion_bundle_uri": str(BUNDLE_URI),
        }
    )
    # Use continuous Champion raw scores for strata.  The frozen isotonic output
    # contains legitimate ties and therefore cannot define five unique quantiles.
    scores = {name: _raw_scores(bundle, frame) for name, frame in frames.items()}
    edges = np.unique(np.quantile(scores["W1"], QUANTILE_LEVELS))
    if len(edges) < 4:
        raise ValueError("HEALTHY_SCORE_HAS_TOO_FEW_UNIQUE_STRATA")
    bins = {name: _assign_bins(value, edges) for name, value in scores.items()}
    healthy = frames["W1"].copy()
    healthy["__score_bin"] = bins["W1"]
    summary_table = healthy.groupby("__score_bin")["is_bad"].agg(["count", "sum"])
    # Laplace smoothing prevents a zero-rate stratum from making the profile undefined.
    base_rates = ((summary_table["sum"] + 1) / (summary_table["count"] + 2)).to_numpy()
    weights = (summary_table["count"] / summary_table["count"].sum()).to_numpy()

    records = []
    for index, slope in enumerate(SLOPES):
        intercept = _solve_intercept(base_rates, weights, slope)
        target_rates = _sigmoid(slope * _logit(base_rates) + intercept)
        # The Champion is extremely discriminative: the lowest-score strata
        # contain fewer than 20 bad rows.  They are statistically ineligible for
        # a calibration claim, so keep their observed rate and inject the
        # conditional change only in adequately labelled strata.
        controlled = summary_table["sum"].to_numpy() >= 20
        target_rates[~controlled] = base_rates[~controlled]
        scenario_dir = OUTPUT / f"slope_{slope:.2f}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        fit = _select_real_rows(
            frames["W2"], bins["W2"], target_rates, controlled, SEED + index * 100
        )
        validation = _select_real_rows(
            frames["W3"],
            bins["W3"],
            target_rates,
            controlled,
            SEED + index * 100 + 50,
        )
        fit_path = scenario_dir / "W2_real_rows.parquet"
        validation_path = scenario_dir / "W3_real_rows.parquet"
        fit.to_parquet(fit_path, index=False)
        validation.to_parquet(validation_path, index=False)
        if len(fit) < 1000 or int(fit["is_bad"].sum()) < 20:
            records.append(
                {
                    "slope": slope,
                    "intercept": intercept,
                    "target_bad_rates_by_score_bin": target_rates.tolist(),
                    "fit_sample_count": len(fit),
                    "validation_sample_count": len(validation),
                    "qualification": {
                        "qualified": False,
                        "reasons": ["CALIBRATION_SAMPLE_INSUFFICIENT"],
                    },
                    "status": "INELIGIBLE",
                }
            )
            continue
        result = execute_calibration(
            {
                "calibration_plan_id": f"a5-real-conditional-{slope:.2f}",
                "model_id": MODEL_ID,
                "champion_version": VERSION,
                "champion_bundle_uri": str(BUNDLE_URI),
                "fit_snapshot_id": f"W2_CONDITIONAL_{slope:.2f}",
                "fit_snapshot_uri": str(fit_path),
                "validation_snapshot_id": f"W3_CONDITIONAL_{slope:.2f}",
                "validation_snapshot_uri": str(validation_path),
                "healthy_snapshot_id": "W1",
                "healthy_snapshot_uri": "assets/data/windows/W1/data.parquet",
                "calibrator_type": "isotonic",
                "artifact_output_path": str(scenario_dir / "calibrator.joblib"),
            }
        )
        qualified, reasons = qualify_adjustment("CALIBRATION_ADJUSTMENT", result)
        metrics = result["metrics"]
        before, healthy_metrics = metrics["before"], metrics["healthy_w1"]
        trigger = {
            "brier_delta_vs_w1": before["brier"] - healthy_metrics["brier"],
            "ece_delta_vs_w1": before["ece"] - healthy_metrics["ece"],
            "brier_warning": before["brier"] - healthy_metrics["brier"] >= 0.003,
            "ece_warning": before["ece"] - healthy_metrics["ece"] >= 0.003,
            "auc_stable": abs(before["auc"] - healthy_metrics["auc"]) < 0.02,
            "ks_stable": abs(before["ks"] - healthy_metrics["ks"]) < 0.02,
        }
        record = {
            "slope": slope,
            "intercept": intercept,
            "target_bad_rates_by_score_bin": target_rates.tolist(),
            "fit_sample_count": len(fit),
            "validation_sample_count": len(validation),
            "fit_bad_rate": float(fit["is_bad"].mean()),
            "validation_bad_rate": float(validation["is_bad"].mean()),
            "trigger": trigger,
            "qualification": {"qualified": qualified, "reasons": reasons},
            "result": result,
        }
        records.append(record)
        (scenario_dir / "report.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    output = {
        "seed": SEED,
        "row_policy": "REAL_COMPETITION_ROWS_ONLY_NO_RELABELING",
        "scenario_track": "SCENARIO",
        "target_bad_rate": TARGET_BAD_RATE,
        "candidates": records,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            [
                {
                    "slope": item["slope"],
                    "n": item["validation_sample_count"],
                    "trigger": item.get("trigger"),
                    "qualified": item["qualification"]["qualified"],
                    "reasons": item["qualification"]["reasons"],
                    "old_score_psi": (item.get("result") or {}).get("metrics", {}).get("old_score_psi"),
                    "new_score_psi": (item.get("result") or {}).get("metrics", {}).get("new_score_psi"),
                    "before": (item.get("result") or {}).get("metrics", {}).get("before"),
                    "after": (item.get("result") or {}).get("metrics", {}).get("after"),
                }
                for item in records
            ],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
