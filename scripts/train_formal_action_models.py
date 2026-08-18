"""Train isolated real-data models needed by the formal A1-A7 action tests.

These assets are never registered as production Champions automatically.  Each
bundle has the same immutable model/calibrator/threshold contract as a Champion
and can therefore be consumed by the production action workers without a test
adapter.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "assets" / "test_models" / "formal_a1_a7"
MODEL_ID = "credit_formal_logistic_a5"
VERSION = "test_v1"
FEATURES = ["max_overdue_days"]
SEED = 20260814


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _ks(labels, scores) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.max(tpr - fpr))


def _threshold(labels, scores) -> dict:
    candidates = np.arange(0.01, 1.0, 0.01)
    evaluated = [
        {
            "threshold": float(value),
            "f1": float(f1_score(labels, scores >= value, zero_division=0)),
        }
        for value in candidates
    ]
    return max(evaluated, key=lambda item: (item["f1"], -item["threshold"]))


def main() -> None:
    w0 = pd.read_parquet(ROOT / "assets/data/windows/W0/data.parquet").sort_values(
        ["apply_time", "sample_id"], kind="stable"
    )
    w1 = pd.read_parquet(ROOT / "assets/data/windows/W1/data.parquet")
    first = int(len(w0) * 0.60)
    second = int(len(w0) * 0.80)
    train, calibration, threshold_frame = w0.iloc[:first], w0.iloc[first:second], w0.iloc[second:]
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=SEED,
                ),
            ),
        ]
    )
    model.fit(train[FEATURES], train["is_bad"])
    calibration_raw = model.predict_proba(calibration[FEATURES])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        calibration_raw, calibration["is_bad"]
    )
    threshold_scores = calibrator.predict(model.predict_proba(threshold_frame[FEATURES])[:, 1])
    selected_threshold = _threshold(threshold_frame["is_bad"], threshold_scores)
    w1_scores = calibrator.predict(model.predict_proba(w1[FEATURES])[:, 1])
    healthy = {
        "sample_count": len(w1),
        "bad_count": int(w1["is_bad"].sum()),
        "auc": float(roc_auc_score(w1["is_bad"], w1_scores)),
        "ks": _ks(w1["is_bad"], w1_scores),
    }
    if healthy["auc"] < 0.80:
        raise RuntimeError(f"FORMAL_MODEL_HEALTH_GATE_FAILED:{healthy}")

    output = ASSET_ROOT / MODEL_ID / VERSION
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "model.joblib")
    joblib.dump(calibrator, output / "calibrator.joblib")
    schema = {
        "schema_version": "formal_action_test_feature_schema/1.0",
        "model_id": MODEL_ID,
        "model_version": VERSION,
        "ordered_features": FEATURES,
        "fields": [{"name": FEATURES[0], "kind": "numeric", "nullable": True}],
        "forbidden_model_inputs": ["sample_id", "apply_time", "is_bad"],
    }
    (output / "feature_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    threshold_artifact = {
        "threshold_id": f"{MODEL_ID}_w0_threshold_v1",
        "model_id": MODEL_ID,
        "model_version": VERSION,
        "score_field": "calibrated_pd",
        "comparison": ">=",
        **selected_threshold,
    }
    (output / "decision_threshold.json").write_text(
        json.dumps(threshold_artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "model_id": MODEL_ID,
        "model_version": VERSION,
        "algorithm_family": "LogisticRegression",
        "worker_adapter": "logistic_regression",
        "scope": "FORMAL_TEST_ONLY_NOT_PRODUCTION_CHAMPION",
        "random_seed": SEED,
        "ordered_features": FEATURES,
        "fit_boundary": "W0_FIRST_60_PERCENT",
        "calibration_boundary": "W0_NEXT_20_PERCENT",
        "threshold_boundary": "W0_LAST_20_PERCENT",
        "healthy_confirmation_boundary": "W1",
        "healthy_metrics": healthy,
        "w4_read_count": 0,
    }
    (output / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    checksums = {
        name: _sha256(output / name)
        for name in (
            "model.joblib",
            "calibrator.joblib",
            "feature_schema.json",
            "decision_threshold.json",
            "training_manifest.json",
        )
    }
    report = {"model_id": MODEL_ID, "version": VERSION, "healthy": healthy, "checksums": checksums}
    (output / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
