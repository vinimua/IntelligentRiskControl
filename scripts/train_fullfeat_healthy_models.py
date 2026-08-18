"""训练完整特征（40 = 34 原始 + 6 时间衍生）的各算法族健康基础模型。

用途：为 A1-A7 正式流程提供特征口径完整（对齐 F01/027 的 40 特征）的健康基础模型，
供"异常 → 修复"全链路跑通。样本边界沿用正式 Champion 的 W0 三段切分，特征不缩水。

- 不读 W4（盲测）
- 不覆盖 assets/champion_models 下任何生产模型
- 输出到 assets/test_models/formal_a1_a7/<model_id>/test_v1/

健康门槛：W1 AUC >= 0.80（允许预测效果在 0.8-0.9 区间，但特征必须完整）。
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "assets" / "test_models" / "formal_a1_a7"

# 完整特征口径：34 个原始特征 + 6 个时间衍生特征（顺序对齐 F01 / credit_model_027）
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
ORDERED_FEATURES = RAW_FEATURES + TIME_FEATURES

SEED = 20260815
HEALTHY_AUC_MIN = 0.80


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _ks(labels, scores) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.max(tpr - fpr))


def _bad_recall_at_top20(y_true, scores) -> float:
    y = np.asarray(y_true)
    s = np.asarray(scores)
    bad = int((y == 1).sum())
    if bad == 0:
        return 0.0
    cutoff = float(np.quantile(s, 0.8))
    return float(((y == 1) & (s >= cutoff)).sum() / bad)


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    ts = pd.to_datetime(data["apply_time"], errors="raise")
    hour = ts.dt.hour + ts.dt.minute / 60.0
    weekday = ts.dt.weekday
    data["apply_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["apply_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["apply_weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    data["apply_weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    data["apply_is_weekend"] = (weekday >= 5).astype(float)
    data["apply_is_night"] = ((ts.dt.hour < 6) | (ts.dt.hour >= 22)).astype(float)
    return data


def _make_model(family: str):
    if family == "LogisticRegression":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000,
                solver="lbfgs", random_state=SEED)),
        ])
    if family == "RandomForest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=40,
                class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
        ])
    if family == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=400, num_leaves=12, max_depth=4, learning_rate=0.05,
            min_child_samples=10, feature_fraction=0.6, subsample=0.9,
            class_weight="balanced", n_jobs=-1, random_state=SEED,
            verbosity=-1,
        )
    raise ValueError(f"UNSUPPORTED_FAMILY:{family}")


def _threshold(labels, scores) -> dict:
    candidates = np.arange(0.01, 1.0, 0.01)
    best = max(
        ({"threshold": float(v), "f1": float(f1_score(labels, scores >= v, zero_division=0))}
         for v in candidates),
        key=lambda item: (item["f1"], -item["threshold"]),
    )
    return best


def train_one(family: str, model_id: str) -> dict:
    w0 = pd.read_parquet(ROOT / "assets/data/windows/W0/data.parquet").sort_values(
        ["apply_time", "sample_id"], kind="stable"
    ).reset_index(drop=True)
    w1 = pd.read_parquet(ROOT / "assets/data/windows/W1/data.parquet").reset_index(drop=True)

    first = int(len(w0) * 0.60)
    second = int(len(w0) * 0.80)
    train = _add_time_features(w0.iloc[:first])
    calibration = _add_time_features(w0.iloc[first:second])
    threshold_frame = _add_time_features(w0.iloc[second:])
    w1_full = _add_time_features(w1)

    model = _make_model(family)
    model.fit(train[ORDERED_FEATURES], train["is_bad"])

    calib_raw = model.predict_proba(calibration[ORDERED_FEATURES])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(calib_raw, calibration["is_bad"])

    thr_scores = calibrator.predict(model.predict_proba(threshold_frame[ORDERED_FEATURES])[:, 1])
    selected_threshold = _threshold(threshold_frame["is_bad"], thr_scores)

    w1_scores = calibrator.predict(model.predict_proba(w1_full[ORDERED_FEATURES])[:, 1])
    healthy = {
        "sample_count": len(w1),
        "bad_count": int(w1["is_bad"].sum()),
        "auc": float(roc_auc_score(w1["is_bad"], w1_scores)),
        "ks": _ks(w1["is_bad"], w1_scores),
        "bad_recall_at_top20": _bad_recall_at_top20(w1["is_bad"], w1_scores),
    }
    if healthy["auc"] < HEALTHY_AUC_MIN:
        raise RuntimeError(f"HEALTH_GATE_FAILED:{model_id}:{healthy}")

    out = OUT_ROOT / model_id / "test_v1"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    joblib.dump(calibrator, out / "calibrator.joblib")

    schema = {
        "schema_version": "champion_feature_schema/1.0",
        "model_id": model_id,
        "model_version": "test_v1",
        "ordered_features": ORDERED_FEATURES,
        "forbidden_model_inputs": ["sample_id", "apply_time", "is_bad"],
    }
    (out / "feature_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    threshold_artifact = {
        "threshold_id": f"{model_id}_w0_threshold_v1",
        "model_id": model_id,
        "model_version": "test_v1",
        "score_field": "calibrated_pd",
        "comparison": ">=",
        **selected_threshold,
    }
    (out / "decision_threshold.json").write_text(json.dumps(threshold_artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "model_id": model_id,
        "model_version": "test_v1",
        "algorithm_family": family,
        "worker_adapter": {
            "LogisticRegression": "logistic_regression",
            "RandomForest": "random_forest",
            "LightGBM": "lightgbm",
        }[family],
        "scope": "FORMAL_TEST_ONLY_NOT_PRODUCTION_CHAMPION",
        "random_seed": SEED,
        "feature_count": len(ORDERED_FEATURES),
        "ordered_features": ORDERED_FEATURES,
        "fit_boundary": "W0_FIRST_60_PERCENT",
        "calibration_boundary": "W0_NEXT_20_PERCENT",
        "threshold_boundary": "W0_LAST_20_PERCENT",
        "healthy_confirmation_boundary": "W1",
        "healthy_metrics": healthy,
        "w4_read_count": 0,
    }
    (out / "training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    checksums = {n: _sha256(out / n) for n in ("model.joblib", "calibrator.joblib", "feature_schema.json", "decision_threshold.json", "training_manifest.json")}
    report = {"model_id": model_id, "algorithm_family": family, "healthy": healthy, "checksums": checksums}
    (out / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    jobs = [
        ("LogisticRegression", "credit_formal_logistic_fullfeat"),
        ("RandomForest", "credit_formal_rf_fullfeat"),
        ("LightGBM", "credit_formal_lgbm_fullfeat"),
    ]
    reports = []
    for family, model_id in jobs:
        print(f"=== 训练 {model_id} ({family}, {len(ORDERED_FEATURES)} 特征) ===", flush=True)
        try:
            r = train_one(family, model_id)
            reports.append(r)
            print(json.dumps(r, ensure_ascii=False, indent=2), flush=True)
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
    print("\n=== 完成 ===")
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
