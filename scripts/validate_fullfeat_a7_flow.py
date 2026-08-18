"""对完整特征(40)的各算法族健康模型跑 A7 异常→修复全链验证。

复用 train_a1_a7_test_assets.py 的评估逻辑，但特征口径换成完整 40 特征
（34 原始 + 6 时间衍生），对齐 F01 / credit_model_027。

验证目标：找出能"表现出异常检出 + 修复恢复"的模型。
- 健康模型：上一轮 train_fullfeat_healthy_models.py 产出
- 异常注入：covariate_drift（特征漂移）
- 修复：同族重训 challenger，在 W3 验证集对比 degraded vs repaired
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.modelops_api.services.monitoring.scenarios.injectors import ScenarioFactory  # noqa: E402
from apps.modelops_api.services.monitoring.threshold_rules import DEFAULT_THRESHOLD_RULES  # noqa: E402
from apps.modelops_api.services.monitoring.window_loader import load_window  # noqa: E402
from workers.training_tasks import (  # noqa: E402
    _compute_ks,
    _paired_bootstrap_delta_ci,
    _prepare_features,
    _train_lightgbm,
    _train_logistic_regression,
    _train_random_forest,
)


SEED = 20260815
RAW_FEATURES = [
    "credit_query_times", "multi_loan_count", "overdue_history", "credit_utilization",
    "credit_length_months", "max_overdue_days", "social_score", "telecom_score",
    "ecomm_risk_score", "judicial_risk_score", "blacklist_hit", "app_duration",
    "click_frequency", "page_depth", "session_count", "night_activity_ratio",
    "login_fail_count", "reg_to_apply_days", "device_risk_score", "ip_change_freq",
    "gps_anomaly", "device_type", "emulator_flag", "age", "income_level",
    "consumption_level", "education_level", "job_stability", "marital_status",
    "gender", "city_tier", "debt_income_ratio", "loan_amount_request",
    "repayment_period",
]
TIME_FEATURES = [
    "apply_hour_sin", "apply_hour_cos", "apply_weekday_sin", "apply_weekday_cos",
    "apply_is_weekend", "apply_is_night",
]
FEATURES = RAW_FEATURES + TIME_FEATURES

HEALTHY_AUC_MIN = 0.80
HEALTHY_KS_MIN = 0.30
HEALTHY_TOLERANCE = 0.02
MIN_RECOVERY_RATE = 0.90
FEATURE_PSI_STRESS_CEILING = 0.50

FAMILIES = {
    "LogisticRegression": {
        "model_id": "credit_formal_logistic_fullfeat",
        "trainer": _train_logistic_regression,
        "parameters": {"C": 1.0, "solver": "lbfgs", "max_iter": 2000, "class_weight": "balanced"},
    },
    "RandomForest": {
        "model_id": "credit_formal_rf_fullfeat",
        "trainer": _train_random_forest,
        "parameters": {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 40, "class_weight": "balanced_subsample"},
    },
    "LightGBM": {
        "model_id": "credit_formal_lgbm_fullfeat",
        "trainer": _train_lightgbm,
        "parameters": {"n_estimators": 400, "num_leaves": 12, "max_depth": 4, "learning_rate": 0.05, "min_child_samples": 10, "feature_fraction": 0.6, "subsample": 0.9},
    },
}


def _ece(y_true, scores, bins=10):
    y = np.asarray(y_true, dtype=float); p = np.asarray(scores, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    total = len(y); value = 0.0
    for i in range(bins):
        mask = assignments == i
        if not mask.any(): continue
        value += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(value if total else 0.0)


def _rank_metrics(frame, scores):
    y = frame["is_bad"]
    return {"auc": float(roc_auc_score(y, scores)), "ks": float(_compute_ks(y, scores)),
            "brier": float(brier_score_loss(y, scores)), "ece": _ece(y, scores)}


def _split_data():
    w0 = load_window("W0").sort_values("apply_time").reset_index(drop=True)
    w1 = load_window("W1").sort_values("apply_time").reset_index(drop=True)
    w2 = load_window("W2").sort_values("apply_time").reset_index(drop=True)
    w3 = load_window("W3").sort_values("apply_time").reset_index(drop=True)
    n = len(w0); fit_end = int(n * 0.60); cal_end = int(n * 0.80)
    return {
        "fit": w0.iloc[:fit_end].copy(), "calibration": w0.iloc[fit_end:cal_end].copy(),
        "threshold": w0.iloc[cal_end:].copy(), "healthy": w1, "w2": w2, "w3": w3,
    }


def _scenario(frames, candidate):
    w3 = frames["w3"].copy()
    for f in candidate["affected_features"]:
        if f in w3.columns:
            w3[f] = pd.to_numeric(w3[f], errors="coerce").astype(float)
    times = pd.to_datetime(w3["apply_time"], errors="raise")
    scenario_cfg = {
        "scenario_name": candidate["scenario_name"],
        "intensity": candidate["intensity"],
        "affected_features": candidate["affected_features"],
        "base_window_id": "W3",
        "event_start_date": str(times.min().date()),
        "event_end_date": str((times.max() + pd.Timedelta(days=1)).date()),
    }
    injected = ScenarioFactory.inject(w3, scenario_cfg, SEED)
    split = times.max().normalize() - pd.Timedelta(days=6)
    itimes = pd.to_datetime(injected.dataframe["apply_time"], errors="raise")
    w3_train = injected.dataframe.loc[itimes < split].copy()
    validation = injected.dataframe.loc[itimes >= split].copy()
    control = w3.loc[times >= split].copy()
    training = pd.concat([frames["w2"], w3_train], ignore_index=True)
    return {"training": training, "validation": validation, "control": control}


def evaluate(family):
    config = FAMILIES[family]
    bundle = PROJECT_ROOT / "assets" / "test_models" / "formal_a1_a7" / config["model_id"] / "test_v1"
    champion = joblib.load(bundle / "model.joblib")
    calibrator = joblib.load(bundle / "calibrator.joblib")
    frames = _split_data()

    candidate = {"scenario_name": "concept_drift", "intensity": 0.40, "affected_features": []}
    scenario = _scenario(frames, candidate)

    trained = config["trainer"](
        scenario["training"], seed=SEED, hyperparameters=config["parameters"],
        sample_weight=None, ordered_features=FEATURES,
    )
    challenger = trained["model"]

    healthy_raw = champion.predict_proba(_prepare_features(frames["healthy"], FEATURES))[:, 1]
    healthy_scores = calibrator.predict(healthy_raw)
    control_raw = champion.predict_proba(_prepare_features(scenario["control"], FEATURES))[:, 1]
    control_scores = calibrator.predict(control_raw)
    degraded_raw = champion.predict_proba(_prepare_features(scenario["validation"], FEATURES))[:, 1]
    degraded_scores = calibrator.predict(degraded_raw)
    repaired_scores = challenger.predict_proba(_prepare_features(scenario["validation"], FEATURES))[:, 1]

    healthy = _rank_metrics(frames["healthy"], healthy_scores)
    control = _rank_metrics(scenario["control"], control_scores)
    degraded = _rank_metrics(scenario["validation"], degraded_scores)
    repaired = _rank_metrics(scenario["validation"], repaired_scores)

    auc_drop = healthy["auc"] - degraded["auc"]
    ks_drop = healthy["ks"] - degraded["ks"]
    auc_gain = repaired["auc"] - degraded["auc"]
    ks_gain = repaired["ks"] - degraded["ks"]
    auc_ci = _paired_bootstrap_delta_ci(scenario["validation"]["is_bad"], degraded_scores, repaired_scores, "AUC", seed=SEED, rounds=400)
    ks_ci = _paired_bootstrap_delta_ci(scenario["validation"]["is_bad"], degraded_scores, repaired_scores, "KS", seed=SEED, rounds=400)

    from apps.modelops_api.services.monitoring.metric_calculators import _compute_psi_frozen
    if candidate["affected_features"]:
        psi_by_feature = {f: float(_compute_psi_frozen(frames["healthy"][f].tolist(), scenario["validation"][f].tolist())) for f in candidate["affected_features"]}
        max_psi = max(psi_by_feature.values())
    else:
        # concept_drift 不涉及特征漂移，用整体分数 PSI 近似（标签被置换，特征不变）
        psi_by_feature = {}
        max_psi = 0.0
    auc_alert, auc_sev = DEFAULT_THRESHOLD_RULES["AUC"].evaluate(degraded["auc"] - healthy["auc"], degraded["auc"])
    ks_alert, ks_sev = DEFAULT_THRESHOLD_RULES["KS"].evaluate(degraded["ks"] - healthy["ks"], degraded["ks"])
    psi_alert, psi_sev = DEFAULT_THRESHOLD_RULES["FEATURE_PSI"].evaluate(max_psi, max_psi)

    auc_recovery = auc_gain / auc_drop if auc_drop > 0 else 0.0
    ks_recovery = ks_gain / ks_drop if ks_drop > 0 else 0.0
    passed = all([
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
    ])

    return {
        "family": family, "model_id": config["model_id"], "passed": passed,
        "healthy": healthy, "degraded": degraded, "repaired": repaired,
        "auc_drop": auc_drop, "ks_drop": ks_drop, "auc_gain": auc_gain, "ks_gain": ks_gain,
        "auc_recovery_rate": auc_recovery, "ks_recovery_rate": ks_recovery,
        "auc_bootstrap_ci": auc_ci, "ks_bootstrap_ci": ks_ci,
        "feature_psi": max_psi,
        "alerts": {"AUC": {"triggered": auc_alert, "severity": auc_sev.value if auc_sev else None},
                   "KS": {"triggered": ks_alert, "severity": ks_sev.value if ks_sev else None},
                   "FEATURE_PSI": {"triggered": psi_alert, "severity": psi_sev.value if psi_sev else None}},
    }


def main():
    results = []
    for family in FAMILIES:
        print(f"=== A7 异常→修复验证: {family} ===", flush=True)
        try:
            r = evaluate(family)
            results.append(r)
            print(json.dumps({k: v for k, v in r.items() if k != "challenger"}, ensure_ascii=False, indent=2, default=str), flush=True)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            results.append({"family": family, "error": str(exc)})

    out = PROJECT_ROOT / "artifacts" / "fullfeat_a7_validation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n报告已写: {out}")


if __name__ == "__main__":
    main()
