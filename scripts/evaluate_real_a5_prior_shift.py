"""Empirically select a real-row prior-shift scenario for formal A5 testing.

The scenario never invents features or labels.  It deterministically keeps all
observed bad samples and downsamples observed good samples in W2/W3.  This
changes only the population prior, which preserves the original class-conditional
records and is suitable for testing probability recalibration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from apps.modelops_api.services.iteration.action_execution_service import (
    execute_calibration,
    qualify_adjustment,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "formal_a1_a7" / "a5_empirical"
TARGET_BAD_RATES = (0.045, 0.05, 0.06, 0.07, 0.08, 0.10)
SEED = 20260814


def _real_row_prior_shift(source: Path, target_bad_rate: float, seed: int) -> pd.DataFrame:
    frame = pd.read_parquet(source)
    bad = frame.loc[frame["is_bad"].astype(int) == 1]
    good = frame.loc[frame["is_bad"].astype(int) == 0]
    wanted_good = round(len(bad) * (1.0 - target_bad_rate) / target_bad_rate)
    if wanted_good > len(good):
        raise ValueError(f"TARGET_RATE_BELOW_NATURAL_RATE:{target_bad_rate}")
    sampled_good = good.sample(n=wanted_good, replace=False, random_state=seed)
    return (
        pd.concat([bad, sampled_good], ignore_index=True)
        .sort_values(["apply_time", "sample_id"], kind="stable")
        .reset_index(drop=True)
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for index, target_rate in enumerate(TARGET_BAD_RATES):
        scenario_dir = OUTPUT / f"bad_rate_{target_rate:.3f}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        fit_path = scenario_dir / "W2_real_rows.parquet"
        validation_path = scenario_dir / "W3_real_rows.parquet"
        fit = _real_row_prior_shift(
            ROOT / "assets/data/windows/W2/data.parquet", target_rate, SEED + index
        )
        validation = _real_row_prior_shift(
            ROOT / "assets/data/windows/W3/data.parquet", target_rate, SEED + 100 + index
        )
        fit.to_parquet(fit_path, index=False)
        validation.to_parquet(validation_path, index=False)
        plan = {
            "calibration_plan_id": f"a5-real-prior-{target_rate:.3f}",
            "model_id": "credit_model_001",
            "champion_version": "champion_v1",
            "fit_snapshot_id": f"W2_PRIOR_{target_rate:.3f}",
            "fit_snapshot_uri": str(fit_path),
            "validation_snapshot_id": f"W3_PRIOR_{target_rate:.3f}",
            "validation_snapshot_uri": str(validation_path),
            "healthy_snapshot_id": "W1",
            "healthy_snapshot_uri": "assets/data/windows/W1/data.parquet",
            "calibrator_type": "isotonic",
            "artifact_output_path": str(scenario_dir / "calibrator.joblib"),
        }
        result = execute_calibration(plan)
        qualified, reasons = qualify_adjustment("CALIBRATION_ADJUSTMENT", result)
        metrics = result["metrics"]
        healthy = metrics["healthy_w1"]
        before = metrics["before"]
        record = {
            "target_bad_rate": target_rate,
            "actual_fit_bad_rate": float(fit["is_bad"].mean()),
            "actual_validation_bad_rate": float(validation["is_bad"].mean()),
            "fit_sample_count": len(fit),
            "validation_sample_count": len(validation),
            "trigger": {
                "brier_delta_vs_w1": before["brier"] - healthy["brier"],
                "ece_delta_vs_w1": before["ece"] - healthy["ece"],
                "brier_warning": before["brier"] - healthy["brier"] >= 0.003,
                "ece_warning": before["ece"] - healthy["ece"] >= 0.003,
                "auc_delta_vs_w1": before["auc"] - healthy["auc"],
                "ks_delta_vs_w1": before["ks"] - healthy["ks"],
            },
            "qualification": {"qualified": qualified, "reasons": reasons},
            "result": result,
        }
        records.append(record)
        (scenario_dir / "report.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    summary = {"seed": SEED, "row_policy": "REAL_ROWS_ONLY", "candidates": records}
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    compact = [
        {
            "rate": item["target_bad_rate"],
            "n": item["validation_sample_count"],
            "trigger": item["trigger"],
            "qualified": item["qualification"]["qualified"],
            "reasons": item["qualification"]["reasons"],
            "before": item["result"]["metrics"]["before"],
            "after": item["result"]["metrics"]["after"],
        }
        for item in records
    ]
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
