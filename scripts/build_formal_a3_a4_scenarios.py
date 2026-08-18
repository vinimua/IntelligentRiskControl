"""Create deterministic A3/A4 scenarios from real competition W3 rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.modelops_api.services.iteration.action_execution_service import (
    execute_repair_and_replay,
    qualify_repair,
)

OUTPUT = ROOT / "artifacts/formal_a1_a7/repair_scenarios"
SEED = 20260814


def _run(name: str, plan: dict) -> dict:
    result = execute_repair_and_replay(plan)
    qualified, reasons = qualify_repair(result)
    report = {"scenario": name, "qualification": {"qualified": qualified, "reasons": reasons}, "result": result}
    path = OUTPUT / name / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    clean = pd.read_parquet(ROOT / "assets/data/windows/W3/data.parquet")
    bundle = ROOT / "assets/champion_models/credit_model_001/champion_v1"
    common = {
        "model_id": "credit_model_001",
        "champion_version": "champion_v1",
        "champion_bundle_uri": str(bundle),
        "healthy_snapshot_id": "W1",
        "healthy_snapshot_uri": str(ROOT / "assets/data/windows/W1/data.parquet"),
    }

    a3_dir = OUTPUT / "a3_missing_age"
    a3_dir.mkdir(parents=True, exist_ok=True)
    a3 = clean.copy()
    selected = a3.sample(frac=0.10, random_state=SEED).index
    a3.loc[selected, "age"] = np.nan
    a3_source = a3_dir / "W3_missing.parquet"
    a3.to_parquet(a3_source, index=False)
    a3_report = _run(
        "a3_missing_age",
        {
            **common,
            "repair_plan_id": "formal-a3-real-001",
            "action": "DATA_REPAIR",
            "source_snapshot_id": "W3_MISSING_AGE",
            "source_snapshot_uri": str(a3_source),
            "reference_snapshot_id": "W2",
            "reference_snapshot_uri": str(ROOT / "assets/data/windows/W2/data.parquet"),
            "affected_features": ["age"],
            "artifact_output_path": str(a3_dir / "W3_repaired.parquet"),
        },
    )

    a4_dir = OUTPUT / "a4_pipeline_column_misalignment"
    a4_dir.mkdir(parents=True, exist_ok=True)
    a4 = clean.copy()
    # This reproduces a pipeline join/order bug: values remain genuine W3 values
    # but are attached to the wrong sample IDs.  No label or row is fabricated.
    a4["login_fail_count"] = (
        a4["login_fail_count"].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    )
    a4_source = a4_dir / "W3_pipeline_corrupted.parquet"
    trusted = a4_dir / "W3_trusted.parquet"
    a4.to_parquet(a4_source, index=False)
    clean.to_parquet(trusted, index=False)
    a4_report = _run(
        "a4_pipeline_column_misalignment",
        {
            **common,
            "repair_plan_id": "formal-a4-real-001",
            "action": "PIPELINE_REPAIR",
            "source_snapshot_id": "W3_PIPELINE_CORRUPTED",
            "source_snapshot_uri": str(a4_source),
            "reference_snapshot_id": "W3_TRUSTED",
            "reference_snapshot_uri": str(trusted),
            "affected_features": ["login_fail_count"],
            "artifact_output_path": str(a4_dir / "W3_repaired.parquet"),
        },
    )
    a4_lgbm_report = _run(
        "a4_lightgbm_027",
        {
            "repair_plan_id": "formal-a4-lightgbm-027",
            "action": "PIPELINE_REPAIR",
            "model_id": "credit_model_027",
            "champion_version": "champion_v1",
            "champion_bundle_uri": str(
                ROOT / "assets/champion_models/credit_model_027/champion_v1"
            ),
            "healthy_snapshot_id": "W1",
            "healthy_snapshot_uri": str(ROOT / "assets/data/windows/W1/data.parquet"),
            "source_snapshot_id": "W3_PIPELINE_CORRUPTED",
            "source_snapshot_uri": str(a4_source),
            "reference_snapshot_id": "W3_TRUSTED",
            "reference_snapshot_uri": str(trusted),
            "affected_features": ["login_fail_count"],
            "artifact_output_path": str(
                OUTPUT / "a4_lightgbm_027/W3_repaired.parquet"
            ),
        },
    )
    a4_model_052_report = _run(
        "a4_lightgbm_052",
        {
            "repair_plan_id": "formal-a4-lightgbm-052",
            "action": "PIPELINE_REPAIR",
            "model_id": "credit_test_lightgbm",
            "champion_version": "test_v1",
            "champion_bundle_uri": str(
                ROOT / "assets/test_models/a1_a7/credit_test_lightgbm/test_v1"
            ),
            "healthy_snapshot_id": "W1",
            "healthy_snapshot_uri": str(ROOT / "assets/data/windows/W1/data.parquet"),
            "source_snapshot_id": "W3_PIPELINE_CORRUPTED",
            "source_snapshot_uri": str(a4_source),
            "reference_snapshot_id": "W3_TRUSTED",
            "reference_snapshot_uri": str(trusted),
            "affected_features": ["login_fail_count"],
            "artifact_output_path": str(
                OUTPUT / "a4_lightgbm_052/W3_repaired.parquet"
            ),
        },
    )
    summary = {
        "seed": SEED,
        "row_policy": "REAL_COMPETITION_ROWS_AND_LABELS_ONLY",
        "w4_read_count": 0,
        "a3": a3_report,
        "a4": a4_report,
        "a4_lightgbm_027": a4_lgbm_report,
        "a4_lightgbm_052": a4_model_052_report,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: {
                    "qualification": value["qualification"],
                    "metrics": value["result"]["metrics"],
                }
                for key, value in (
                    ("a3", a3_report),
                    ("a4", a4_report),
                    ("a4_lightgbm_027", a4_lgbm_report),
                    ("a4_lightgbm_052", a4_model_052_report),
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
