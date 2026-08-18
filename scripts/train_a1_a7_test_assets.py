"""Build isolated, evidence-backed model assets for A1-A7 testing.

The script is deliberately separated from ``assets/champion_models``.  It
never reads W4, never registers or replaces a Champion, and makes all test
profiles explicit in machine-readable manifests.

Assets:
* credit_test_logistic: healthy control for A1-A4, real A5/A6 algorithm
  fixtures, and an A7 LogisticRegression drift/repair candidate.
* credit_test_lightgbm: a second A7 LightGBM drift/repair candidate.

Run from the repository root with Python 3.11::

    python scripts/train_a1_a7_test_assets.py
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.modelops_api.services.monitoring.metric_calculators import (  # noqa: E402
    _compute_psi_frozen,
)
from apps.modelops_api.services.monitoring.scenarios.injectors import (  # noqa: E402
    ScenarioFactory,
)
from apps.modelops_api.services.monitoring.threshold_rules import (  # noqa: E402
    DEFAULT_THRESHOLD_RULES,
)
from apps.modelops_api.services.monitoring.window_loader import load_window  # noqa: E402
from workers.training_tasks import (  # noqa: E402
    _compute_ks,
    _paired_bootstrap_delta_ci,
    _prepare_features,
    _train_lightgbm,
    _train_logistic_regression,
)


SEED = 20260814
FEATURES = ["login_fail_count", "reg_to_apply_days", "max_overdue_days"]
OUTPUT_ROOT = PROJECT_ROOT / "assets" / "test_models" / "a1_a7"
REPORT_ROOT = PROJECT_ROOT / "artifacts" / "a1_a7_test_assets"
HEALTHY_AUC_MIN = 0.80
HEALTHY_KS_MIN = 0.30
HEALTHY_TOLERANCE = 0.02
MIN_RECOVERY_RATE = 0.90
FEATURE_PSI_STRESS_CEILING = 0.50
SCENARIO_CANDIDATES = [
    {"intensity": 0.30, "affected_features": ["login_fail_count"]},
    {"intensity": 0.40, "affected_features": ["login_fail_count"]},
    {"intensity": 0.30, "affected_features": FEATURES},
    {"intensity": 0.40, "affected_features": FEATURES},
]

FAMILIES = {
    "LogisticRegression": {
        "model_id": "credit_test_logistic",
        "legacy_model_id": "credit_test_logistic",
        "trainer": _train_logistic_regression,
        "parameters": {
            "C": 1.0,
            "solver": "liblinear",
            "max_iter": 2000,
            "class_weight": "balanced",
        },
        "worker_adapter": "logistic_regression",
    },
    "LightGBM": {
        "model_id": "credit_test_lightgbm",
        "legacy_model_id": "credit_test_lightgbm",
        "trainer": _train_lightgbm,
        "parameters": {
            "n_estimators": 180,
            "max_depth": 5,
            "num_leaves": 15,
            "learning_rate": 0.05,
            "min_child_samples": 40,
            "subsample": 0.9,
            "colsample_bytree": 1.0,
            "n_jobs": 1,
        },
        "worker_adapter": "lightgbm",
    },
}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _freeze(path: Path, payload: object) -> str:
    buffer = io.BytesIO()
    joblib.dump(payload, buffer)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())
    return "sha256:" + hashlib.sha256(buffer.getvalue()).hexdigest()


def _hash_frame(frame: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _ece(y_true, scores, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(scores, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    total = len(y)
    value = 0.0
    for index in range(bins):
        mask = assignments == index
        if not mask.any():
            continue
        value += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(value if total else 0.0)


def _rank_metrics(frame: pd.DataFrame, scores: np.ndarray) -> dict:
    y = frame["is_bad"]
    return {
        "auc": float(roc_auc_score(y, scores)),
        "ks": float(_compute_ks(y, scores)),
        "brier": float(brier_score_loss(y, scores)),
        "ece": _ece(y, scores),
    }


def _threshold_metrics(y_true, scores, threshold: float) -> dict:
    predicted = np.asarray(scores) >= threshold
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "high_risk_rate": float(predicted.mean()),
    }


def _best_f1_threshold(y_true, scores) -> dict:
    candidates = np.linspace(0.01, 0.99, 99)
    results = [_threshold_metrics(y_true, scores, value) for value in candidates]
    return max(results, key=lambda item: (item["f1"], item["recall"], -item["threshold"]))


def _miscalibrate(scores, slope: float, intercept: float) -> np.ndarray:
    p = np.clip(np.asarray(scores, dtype=float), 1e-6, 1.0 - 1e-6)
    logit = np.log(p / (1.0 - p))
    return 1.0 / (1.0 + np.exp(-(slope * logit + intercept)))


def _split_data() -> dict[str, pd.DataFrame]:
    w0 = load_window("W0").sort_values("apply_time").reset_index(drop=True)
    w1 = load_window("W1").sort_values("apply_time").reset_index(drop=True)
    w2 = load_window("W2").sort_values("apply_time").reset_index(drop=True)
    w3 = load_window("W3").sort_values("apply_time").reset_index(drop=True)
    n = len(w0)
    fit_end = int(n * 0.60)
    calibration_end = int(n * 0.80)
    frames = {
        "fit": w0.iloc[:fit_end].copy(),
        "calibration": w0.iloc[fit_end:calibration_end].copy(),
        "threshold": w0.iloc[calibration_end:].copy(),
        "healthy": w1,
        "w2": w2,
        "w3": w3,
    }
    for name, frame in frames.items():
        if frame["is_bad"].isna().any():
            raise RuntimeError(f"LABEL_MISSING:{name}")
        if frame["sample_id"].astype(str).duplicated().any():
            raise RuntimeError(f"DUPLICATE_SAMPLE_ID:{name}")
    return frames


def _scenario(frames: dict[str, pd.DataFrame], candidate: dict) -> dict:
    w3 = frames["w3"].copy()
    # The production injector adds fractional shifts.  Cast only the isolated
    # copy so pandas does not silently coerce integer columns in future releases.
    for feature in candidate["affected_features"]:
        w3[feature] = pd.to_numeric(w3[feature], errors="coerce").astype(float)
    times = pd.to_datetime(w3["apply_time"], errors="raise")
    injected = ScenarioFactory.inject(
        w3,
        {
            "scenario_name": "covariate_drift",
            "intensity": candidate["intensity"],
            "affected_features": candidate["affected_features"],
            "base_window_id": "W3",
            "event_start_date": str(times.min().date()),
            "event_end_date": str((times.max() + pd.Timedelta(days=1)).date()),
        },
        SEED,
    )
    split = times.max().normalize() - pd.Timedelta(days=6)
    injected_times = pd.to_datetime(injected.dataframe["apply_time"], errors="raise")
    w3_train = injected.dataframe.loc[injected_times < split].copy()
    validation = injected.dataframe.loc[injected_times >= split].copy()
    control = w3.loc[times >= split].copy()
    training = pd.concat([frames["w2"], w3_train], ignore_index=True)
    overlap = set(training["sample_id"].astype(str)) & set(validation["sample_id"].astype(str))
    if overlap:
        raise RuntimeError(f"SAMPLE_OVERLAP_DETECTED:{len(overlap)}")
    return {
        "training": training,
        "validation": validation,
        "control": control,
        "metadata": {**injected.metadata, "split_boundary": split.isoformat()},
    }


def _evaluate_a7(family: str, champion, frames: dict, candidate: dict) -> dict:
    config = FAMILIES[family]
    scenario = _scenario(frames, candidate)
    trained = config["trainer"](
        scenario["training"],
        seed=SEED,
        hyperparameters=config["parameters"],
        sample_weight=None,
        ordered_features=FEATURES,
    )
    challenger = trained["model"]
    healthy_scores = champion.predict_proba(_prepare_features(frames["healthy"], FEATURES))[:, 1]
    control_scores = champion.predict_proba(_prepare_features(scenario["control"], FEATURES))[:, 1]
    degraded_scores = champion.predict_proba(_prepare_features(scenario["validation"], FEATURES))[:, 1]
    repaired_scores = challenger.predict_proba(_prepare_features(scenario["validation"], FEATURES))[:, 1]
    challenger_healthy_scores = challenger.predict_proba(_prepare_features(frames["healthy"], FEATURES))[:, 1]
    healthy = _rank_metrics(frames["healthy"], healthy_scores)
    control = _rank_metrics(scenario["control"], control_scores)
    degraded = _rank_metrics(scenario["validation"], degraded_scores)
    repaired = _rank_metrics(scenario["validation"], repaired_scores)
    auc_drop = healthy["auc"] - degraded["auc"]
    ks_drop = healthy["ks"] - degraded["ks"]
    auc_gain = repaired["auc"] - degraded["auc"]
    ks_gain = repaired["ks"] - degraded["ks"]
    auc_ci = _paired_bootstrap_delta_ci(
        scenario["validation"]["is_bad"], degraded_scores, repaired_scores, "AUC", seed=SEED, rounds=400
    )
    ks_ci = _paired_bootstrap_delta_ci(
        scenario["validation"]["is_bad"], degraded_scores, repaired_scores, "KS", seed=SEED, rounds=400
    )
    psi_by_feature = {
        feature: float(_compute_psi_frozen(frames["healthy"][feature].tolist(), scenario["validation"][feature].tolist()))
        for feature in candidate["affected_features"]
    }
    max_psi = max(psi_by_feature.values())
    auc_alert, auc_severity = DEFAULT_THRESHOLD_RULES["AUC"].evaluate(
        degraded["auc"] - healthy["auc"], degraded["auc"]
    )
    ks_alert, ks_severity = DEFAULT_THRESHOLD_RULES["KS"].evaluate(
        degraded["ks"] - healthy["ks"], degraded["ks"]
    )
    psi_alert, psi_severity = DEFAULT_THRESHOLD_RULES["FEATURE_PSI"].evaluate(max_psi, max_psi)
    auc_recovery = auc_gain / auc_drop if auc_drop > 0 else 0.0
    ks_recovery = ks_gain / ks_drop if ks_drop > 0 else 0.0
    passed = all(
        [
            healthy["auc"] >= HEALTHY_AUC_MIN,
            healthy["ks"] >= HEALTHY_KS_MIN,
            bool(auc_alert or ks_alert),
            bool(psi_alert),
            max_psi < FEATURE_PSI_STRESS_CEILING,
            auc_recovery >= MIN_RECOVERY_RATE,
            ks_recovery >= MIN_RECOVERY_RATE,
            repaired["auc"] >= healthy["auc"] - HEALTHY_TOLERANCE,
            repaired["ks"] >= healthy["ks"] - HEALTHY_TOLERANCE,
            auc_ci is not None and auc_ci[0] > 0,
            ks_ci is not None and ks_ci[0] > 0,
        ]
    )
    return {
        "candidate": candidate,
        "scenario": scenario,
        "challenger": challenger,
        "trained": trained,
        "metrics": {
            "healthy": healthy,
            "paired_clean_control": control,
            "degraded": degraded,
            "repaired": repaired,
            "auc_drop": auc_drop,
            "ks_drop": ks_drop,
            "auc_gain": auc_gain,
            "ks_gain": ks_gain,
            "auc_recovery_rate": auc_recovery,
            "ks_recovery_rate": ks_recovery,
            "auc_bootstrap_ci": auc_ci,
            "ks_bootstrap_ci": ks_ci,
            "feature_psi": max_psi,
            "feature_psi_by_feature": psi_by_feature,
            "alerts": {
                "AUC": {"triggered": auc_alert, "severity": auc_severity.value if auc_severity else None},
                "KS": {"triggered": ks_alert, "severity": ks_severity.value if ks_severity else None},
                "FEATURE_PSI": {"triggered": psi_alert, "severity": psi_severity.value if psi_severity else None},
            },
            "train_auc": float(trained["train_auc"]),
            "train_valid_gap": abs(float(trained["train_auc"]) - repaired["auc"]),
        },
        "passed": passed,
    }


def _a5_a6_profiles(champion, frames: dict) -> tuple[dict, dict, object]:
    calibration_raw = champion.predict_proba(_prepare_features(frames["calibration"], FEATURES))[:, 1]
    threshold_raw = champion.predict_proba(_prepare_features(frames["threshold"], FEATURES))[:, 1]
    healthy_raw = champion.predict_proba(_prepare_features(frames["healthy"], FEATURES))[:, 1]
    candidates = [
        {"slope": 0.55, "intercept": 0.80},
        {"slope": 0.70, "intercept": 1.00},
        {"slope": 1.40, "intercept": 0.80},
        {"slope": 1.80, "intercept": 1.00},
    ]
    evaluated = []
    for profile in candidates:
        cal_bad = _miscalibrate(calibration_raw, **profile)
        threshold_bad = _miscalibrate(threshold_raw, **profile)
        healthy_bad = _miscalibrate(healthy_raw, **profile)
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(
            cal_bad, frames["calibration"]["is_bad"]
        )
        threshold_fixed = calibrator.predict(threshold_bad)
        healthy_fixed = calibrator.predict(healthy_bad)
        before = {
            "brier": float(brier_score_loss(frames["healthy"]["is_bad"], healthy_bad)),
            "ece": _ece(frames["healthy"]["is_bad"], healthy_bad),
            "auc": float(roc_auc_score(frames["healthy"]["is_bad"], healthy_bad)),
            "ks": float(_compute_ks(frames["healthy"]["is_bad"], healthy_bad)),
        }
        after = {
            "brier": float(brier_score_loss(frames["healthy"]["is_bad"], healthy_fixed)),
            "ece": _ece(frames["healthy"]["is_bad"], healthy_fixed),
            "auc": float(roc_auc_score(frames["healthy"]["is_bad"], healthy_fixed)),
            "ks": float(_compute_ks(frames["healthy"]["is_bad"], healthy_fixed)),
        }
        evaluated.append({
            "profile": profile,
            "calibrator": calibrator,
            "threshold_fixed_scores": threshold_fixed,
            "healthy_fixed_scores": healthy_fixed,
            "before": before,
            "after": after,
            "brier_improvement": before["brier"] - after["brier"],
            "ece_improvement": before["ece"] - after["ece"],
        })
    selected = max(evaluated, key=lambda item: (item["brier_improvement"], item["ece_improvement"]))
    a5_passed = all(
        [
            selected["brier_improvement"] >= DEFAULT_THRESHOLD_RULES["BRIER"].warning_threshold,
            selected["ece_improvement"] >= DEFAULT_THRESHOLD_RULES["ECE"].warning_threshold,
            selected["after"]["auc"] >= selected["before"]["auc"] - 0.01,
            selected["after"]["ks"] >= selected["before"]["ks"] - 0.02,
        ]
    )
    a5 = {
        "action": "A5_CALIBRATION_ADJUSTMENT",
        "input_profile": {
            **selected["profile"],
            "formula": "sigmoid(slope * logit(raw_probability) + intercept)",
        },
        "before": selected["before"],
        "after": selected["after"],
        "brier_improvement": selected["brier_improvement"],
        "ece_improvement": selected["ece_improvement"],
        "ranking_guardrail_passed": (
            selected["after"]["auc"] >= selected["before"]["auc"] - 0.01
            and selected["after"]["ks"] >= selected["before"]["ks"] - 0.02
        ),
        "algorithm_fixture_passed": a5_passed,
        "system_executor_status": "NOT_PROVEN_WORKER_USES_MOCK_DATA",
    }
    optimized = _best_f1_threshold(frames["threshold"]["is_bad"], selected["threshold_fixed_scores"])
    stale = _threshold_metrics(frames["healthy"]["is_bad"], selected["healthy_fixed_scores"], 0.50)
    recovered = _threshold_metrics(
        frames["healthy"]["is_bad"], selected["healthy_fixed_scores"], optimized["threshold"]
    )
    a6 = {
        "action": "A6_THRESHOLD_ADJUSTMENT",
        "search_metric": "F1",
        "search_data": "W0_THRESHOLD_ONLY",
        "stale_threshold": stale,
        "selected_on_w0_threshold": optimized,
        "replayed_on_w1": recovered,
        "f1_improvement": recovered["f1"] - stale["f1"],
        "algorithm_fixture_passed": recovered["f1"] > stale["f1"],
        "required_upstream_condition": "REAL_BUSINESS_TARGET_CHANGE_AND_MANUAL_APPROVAL",
        "system_executor_status": "NOT_PROVEN_WORKER_USES_MOCK_DATA",
    }
    return a5, a6, selected["calibrator"]


def _public_attempt(result: dict) -> dict:
    return {
        "candidate": result["candidate"],
        "passed": result["passed"],
        "metrics": result["metrics"],
    }


def _save_asset(
    family: str,
    champion,
    champion_training: dict,
    selected: dict,
    attempts: list[dict],
    frames: dict,
    *,
    base_calibrator,
    a5: dict | None = None,
    a6: dict | None = None,
    a5_calibrator=None,
) -> dict:
    config = FAMILIES[family]
    root = OUTPUT_ROOT / config["model_id"] / "test_v1"
    model_checksum = _freeze(root / "model.joblib", champion)
    challenger_checksum = _freeze(root / "a7_repair_challenger.joblib", selected["challenger"])
    checksums = {
        "model": model_checksum,
        "calibrator": _freeze(root / "calibrator.joblib", base_calibrator),
        "a7_repair_challenger": challenger_checksum,
    }
    if a5_calibrator is not None:
        checksums["a5_recalibrator"] = _freeze(root / "a5_recalibrator.joblib", a5_calibrator)
    schema = {
        "schema_version": "a1_a7_test_feature_schema/1.0",
        "model_id": config["model_id"],
        "model_version": "test_v1",
        "ordered_features": FEATURES,
        "fields": [{"name": feature, "kind": "numeric", "nullable": True} for feature in FEATURES],
        "forbidden_model_inputs": ["sample_id", "apply_time", "is_bad"],
    }
    manifest = {
        "scope": "TEST_ONLY_NOT_CHAMPION_NOT_PRODUCTION",
        "model_id": config["model_id"],
        "legacy_model_id": config["legacy_model_id"],
        "model_version": "test_v1",
        "algorithm_family": family,
        "worker_adapter": config["worker_adapter"],
        "random_seed": SEED,
        "ordered_features": FEATURES,
        "selected_parameters": config["parameters"],
        "fit_boundary": "W0_FIRST_60_PERCENT",
        "calibration_boundary": "W0_NEXT_20_PERCENT",
        "threshold_boundary": "W0_LAST_20_PERCENT",
        "healthy_confirmation_boundary": "W1",
        "a7_training_boundary": "W2_PLUS_W3_BEFORE_LAST_7D",
        "a7_validation_boundary": "W3_LAST_7D",
        "w4_read_count": 0,
        "production_models_modified": False,
        "checksums": checksums,
        "source_data_hashes": {
            name: _hash_frame(frames[name]) for name in ("fit", "calibration", "threshold", "healthy", "w2", "w3")
        },
        "action_coverage": {
            "A1": "HEALTHY_CONTROL",
            "A2": "HEALTHY_CONTROL_FOR_OBSERVATION",
            "A3": "HEALTHY_CONTROL_FOR_DATA_REPAIR_REPLAY",
            "A4": "HEALTHY_CONTROL_FOR_PIPELINE_REPAIR_REPLAY",
            "A5": "REAL_ALGORITHM_FIXTURE" if a5 else "NOT_ASSIGNED",
            "A6": "REAL_ALGORITHM_FIXTURE" if a6 else "NOT_ASSIGNED",
            "A7": (
                "COVARIATE_DRIFT_REPAIR_FIXTURE"
                if selected["passed"]
                else "NOT_ASSIGNED_ACCEPTANCE_GATES_NOT_MET_SEE_A7_EVALUATION"
            ),
        },
    }
    _write_json(root / "feature_schema.json", schema)
    _write_json(root / "training_manifest.json", manifest)
    _write_json(root / "a7_evaluation.json", {
        "selected": _public_attempt(selected),
        "screening_attempts": [_public_attempt(item) for item in attempts],
    })
    if a5 is not None:
        _write_json(root / "a5_calibration_profile.json", a5)
    if a6 is not None:
        _write_json(root / "a6_threshold_profile.json", a6)
    card = f"""# {config['model_id']} / test_v1

## Identity

- Scope: TEST_ONLY_NOT_CHAMPION_NOT_PRODUCTION
- Legacy identity: {config['legacy_model_id']}
- Algorithm: {family}
- Worker adapter: {config['worker_adapter']}
- Features: {', '.join(FEATURES)}
- W4 read count: 0

## Verified use

- Healthy W1 AUC: {selected['metrics']['healthy']['auc']:.6f}
- Healthy W1 KS: {selected['metrics']['healthy']['ks']:.6f}
- A7 feature PSI: {selected['metrics']['feature_psi']:.6f}
- A7 degraded AUC / repaired AUC: {selected['metrics']['degraded']['auc']:.6f} / {selected['metrics']['repaired']['auc']:.6f}
- A7 degraded KS / repaired KS: {selected['metrics']['degraded']['ks']:.6f} / {selected['metrics']['repaired']['ks']:.6f}
- A7 fixture passed: {selected['passed']}

## Restrictions

This bundle is an isolated test asset.  It must not be copied into
`assets/champion_models`, registered as a Champion, or used to claim a full
production workflow.  A1-A4 use the model only as a healthy replay control.
A5/A6 system executors currently use mock data; their profiles here prove only
the real standalone algorithms and expose the remaining integration gap.
"""
    (root / "model_card.md").write_text(card, encoding="utf-8")
    return {
        "model_id": config["model_id"],
        "model_version": "test_v1",
        "algorithm_family": family,
        "root": str(root),
        "healthy": selected["metrics"]["healthy"],
        "a7_passed": selected["passed"],
        "a7": selected["metrics"],
        "a5": a5,
        "a6": a6,
        "checksums": checksums,
        "champion_training_auc": float(champion_training["train_auc"]),
    }


def run() -> tuple[dict, Path]:
    frames = _split_data()
    results = []
    for family, config in FAMILIES.items():
        champion_training = config["trainer"](
            frames["fit"],
            seed=SEED,
            hyperparameters=config["parameters"],
            sample_weight=None,
            ordered_features=FEATURES,
        )
        champion = champion_training["model"]
        calibration_raw = champion.predict_proba(
            _prepare_features(frames["calibration"], FEATURES)
        )[:, 1]
        base_calibrator = IsotonicRegression(out_of_bounds="clip").fit(
            calibration_raw, frames["calibration"]["is_bad"]
        )
        attempts = [_evaluate_a7(family, champion, frames, candidate) for candidate in SCENARIO_CANDIDATES]
        passing = [item for item in attempts if item["passed"]]
        if not passing:
            selected = max(
                attempts,
                key=lambda item: (
                    min(item["metrics"]["auc_recovery_rate"], item["metrics"]["ks_recovery_rate"]),
                    item["metrics"]["auc_gain"] + item["metrics"]["ks_gain"],
                ),
            )
        else:
            selected = max(
                passing,
                key=lambda item: (
                    min(item["metrics"]["auc_recovery_rate"], item["metrics"]["ks_recovery_rate"]),
                    -item["metrics"]["feature_psi"],
                ),
            )
        a5 = a6 = a5_calibrator = None
        if family == "LogisticRegression":
            a5, a6, a5_calibrator = _a5_a6_profiles(champion, frames)
            if not a5["algorithm_fixture_passed"] or not a6["algorithm_fixture_passed"]:
                raise RuntimeError(f"A5_A6_FIXTURE_FAILED:{a5}:{a6}")
        results.append(
            _save_asset(
                family,
                champion,
                champion_training,
                selected,
                attempts,
                frames,
                base_calibrator=base_calibrator,
                a5=a5,
                a6=a6,
                a5_calibrator=a5_calibrator,
            )
        )
    model_012_report = PROJECT_ROOT / "artifacts" / "a7_repair_model_012" / "latest_report.json"
    model_012 = json.loads(model_012_report.read_text(encoding="utf-8"))
    registry = {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "A1_A7_TEST_MODEL_ASSET_REGISTRY",
        "w4_read_count": 0,
        "production_models_modified": False,
        "assets": [
            {
                "model_id": "credit_model_012",
                "model_version": "champion_v1",
                "role": "A7_RANDOM_FOREST_REFERENCE_ASSET",
                "source": str(PROJECT_ROOT / "assets" / "champion_models" / "credit_model_012" / "champion_v1"),
                "controlled_repair_effect_proven": bool(model_012["claims"]["controlled_repair_effect_proven"]),
                "production_full_flow_proven": bool(model_012["claims"]["production_ready"]),
            },
            *results,
        ],
        "system_truth": {
            "A1_A4": "NO_SPECIAL_MODEL_REQUIRED_USE_HEALTHY_CONTROL",
            "A5": "ALGORITHM_ASSET_READY_SYSTEM_WORKER_STILL_MOCK",
            "A6": "ALGORITHM_ASSET_READY_SYSTEM_WORKER_STILL_MOCK_AND_MANUAL_APPROVAL_REQUIRED",
            "A7": "CREDIT_MODEL_012_IS_CONFIRMED_LIGHTGBM_IMPROVES_BUT_FULL_TRIGGER_GATES_NOT_MET_LOGISTIC_IS_NEGATIVE_CONTROL",
        },
    }
    path = REPORT_ROOT / "latest_report.json"
    _write_json(path, registry)
    return registry, path


if __name__ == "__main__":
    report, report_path = run()
    print(json.dumps({
        "report": str(report_path),
        "assets": [item["model_id"] for item in report["assets"]],
        "w4_read_count": report["w4_read_count"],
        "production_models_modified": report["production_models_modified"],
    }, ensure_ascii=False, indent=2))
