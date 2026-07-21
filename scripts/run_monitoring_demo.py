"""监控演示脚本 — 端到端跑一次完整 WP02-WP08 监控链路。

不依赖 Docker/PostgreSQL/API，直接使用本地 Parquet 数据 + Champion 模型。
输出：性能指标、漂移报告、数据质量、检测器信号、告警事件。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# 确保项目根在 path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from apps.modelops_api.services.monitoring.baseline import build_monitoring_baseline
from apps.modelops_api.services.monitoring.drift.algorithms import (
    benjamini_hochberg,
    categorical_drift,
    compute_performance_metrics,
    continuous_drift,
    feature_quality,
)
from apps.modelops_api.services.monitoring.drift.output_monitor import output_metrics
from apps.modelops_api.services.monitoring.detectors.runner import run_detectors
from apps.modelops_api.services.monitoring.rolling import iter_rolling_windows
from apps.modelops_api.services.monitoring.sentinel.feature_builder import (
    build_monitor_feature_vector,
    select_canonical_sentinel_rows,
)
from apps.modelops_api.services.monitoring.trend_features import trailing_slope
from apps.modelops_api.services.monitoring.window_loader import (
    load_window,
    load_window_with_predictions,
)


def _json_safe(obj):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def main():
    print("=" * 70)
    print("  RiskItem - Task 1 WP02-WP08 Monitoring Demo")
    print(f"  {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    # === Step 1: Load Data ===
    print("\n[WP02] Loading data + model predictions")
    w0 = load_window_with_predictions("W0")
    w1 = load_window_with_predictions("W1")
    w2 = load_window_with_predictions("W2")
    w3 = load_window_with_predictions("W3")

    for label, df in [("W0", w0), ("W1", w1), ("W2", w2), ("W3", w3)]:
        bad = df["is_bad"].sum()
        print(f"  {label}: {len(df):,} rows, is_bad={bad}, "
              f"pred mean={df['y_pred_proba'].mean():.4f}")

    # === ② 构建基线 ===
    print("\n> WP02 构建 W0 监控基线（冻结分箱 + 性能基准）")
    feature_names = [c for c in w0.columns
                     if c not in ("sample_id", "apply_time", "is_bad",
                                  "risk_score", "y_pred_proba",
                                  "apply_hour_sin", "apply_hour_cos",
                                  "apply_weekday_sin", "apply_weekday_cos",
                                  "apply_is_weekend", "apply_is_night")]
    categorical = {
        "device_type": [0, 1],
        "education_level": [1, 2, 3, 4, 5],
        "marital_status": [0, 1],
        "gender": [0, 1],
        "city_tier": [1, 2, 3, 4],
        "repayment_period": [6, 12, 24, 36],
    }

    baseline = build_monitoring_baseline(
        w0, model_id="credit_model_001", model_version="champion_v1",
        feature_names=feature_names, categorical_features=categorical,
    )

    perf_ref = baseline.performance_reference_json
    print(f"  W0 基准: AUC={perf_ref.get('auc')}, KS={perf_ref.get('ks')}, "
          f"bad_rate={perf_ref.get('bad_rate')}")
    print(f"  特征数: {len(baseline.feature_names)}")
    print(f"  分箱规则数: {len(baseline.binning_rules_json)}")

    # === ③ 多窗口滚动监控 ===
    print("\n> WP04-WP05 滚动窗口监控（7天窗口）")

    all_data = pd.concat([w1, w2, w3], ignore_index=True).sort_values("apply_time")
    reference_scores = w0["y_pred_proba"]

    perf_rows = []
    qual_rows = []
    drift_rows = []

    for start, end, window in iter_rolling_windows(all_data, window_days=7, step_days=1):
        window_id = f"7D_{start:%Y%m%d}_{end:%Y%m%d}"
        sample_count = len(window)
        bad_count = int(window["is_bad"].sum())

        # 性能
        perf = compute_performance_metrics(window["is_bad"], window["y_pred_proba"])

        # 输出分布
        out = output_metrics(window["y_pred_proba"], reference_scores, baseline.score_edges)

        common = {
            "window_id": window_id, "start": start, "end": end,
            "sample_count": sample_count, "bad_count": bad_count,
        }
        perf_rows.append({**common, **perf, **out})

        # 特征漂移 + 质量
        p_positions = []
        p_values = []
        for fname in baseline.feature_names:
            rule = baseline.binning_rules_json.get(fname)
            if rule is None or fname not in window.columns:
                continue

            # 质量
            if fname in baseline.feature_profiles:
                quality = feature_quality(
                    window[fname], pd.Series(baseline.feature_profiles[fname]),
                    rule["feature_type"],
                )
                qual_rows.append({**common, "feature_name": fname, **quality})

            # 漂移
            if rule["feature_type"] == "categorical":
                drift = categorical_drift(w0[fname], window[fname], rule["categories"])
                row = {"feature_name": fname, "feature_type": "categorical", **drift,
                       "wasserstein_distance": None, "ks_statistic": None,
                       "ks_p_value": None, "ks_q_value": None}
            else:
                drift = continuous_drift(w0[fname], window[fname], rule["edges"])
                row = {"feature_name": fname, "feature_type": "continuous", **drift,
                       "category_share_change": None, "unknown_category_rate": 0.0,
                       "ks_q_value": None}
                if row.get("ks_p_value") is not None:
                    p_positions.append(len(drift_rows))
                    p_values.append(row["ks_p_value"])
            drift_rows.append({**common, **row})

        for pos, q_val in zip(p_positions, benjamini_hochberg(p_values)):
            drift_rows[pos]["ks_q_value"] = q_val

    perf_df = pd.DataFrame(perf_rows)
    qual_df = pd.DataFrame(qual_rows)
    drift_df = pd.DataFrame(drift_rows)

    print(f"  窗口数: {len(perf_df)}")
    print(f"  漂移记录数: {len(drift_df)}")

    # === ④ 漂移报告 ===
    print("\n> WP05 特征漂移 TOP-10")

    if not drift_df.empty:
        latest_start = perf_df["start"].max()
        latest_drift = drift_df[drift_df["start"] == latest_start]
        top_psi = latest_drift.nlargest(10, "psi")[
            ["feature_name", "psi", "js_divergence", "wasserstein_distance", "ks_statistic"]
        ]
        for _, row in top_psi.iterrows():
            psi_val = row["psi"] if pd.notna(row["psi"]) else 0
            flag = "!!" if psi_val > 0.25 else "!" if psi_val > 0.1 else "OK"
            print(f"  {flag} {row['feature_name']:<25s} PSI={psi_val:.4f}  "
                  f"JS={row['js_divergence']:.4f}  KS={row['ks_statistic']}")

    # === ⑤ 数据质量 ===
    print("\n> WP05 数据质量异常")

    if not qual_df.empty:
        latest_qual = qual_df[qual_df["start"] == latest_start]
        alerts = latest_qual[latest_qual["dq_flag"].isin(["ALERT", "WARN"])]
        if not alerts.empty:
            for _, row in alerts.iterrows():
                print(f"  [{row['dq_flag']}] {row['feature_name']}: "
                      f"score={row['dq_score']:.3f}  "
                      f"missing_delta={row['missing_rate_delta']:.4f}  "
                      f"outlier_delta={row['outlier_rate_delta']:.4f}")
        else:
            print("  OK 所有特征数据质量正常")

    # === ⑥ 检测器 ===
    print("\n> WP07 流式检测器")

    features_input = perf_df[["window_id", "start"]].copy()
    features_input["model_id"] = "credit_model_001"
    features_input["model_version"] = "champion_v1"
    features_input["data_track"] = "NATURAL"
    features_input["monitor_window_id"] = features_input["window_id"]
    features_input["auc"] = features_input["window_id"].map(
        perf_df.set_index("window_id")["auc"]
    )
    features_input["ks"] = features_input["window_id"].map(
        perf_df.set_index("window_id")["ks"]
    )
    features_input["prediction_mean"] = features_input["window_id"].map(
        perf_df.set_index("window_id")["prediction_mean"]
    )
    if not drift_df.empty:
        max_psi_per_window = drift_df.groupby("window_id")["psi"].max()
        features_input["max_feature_psi_7d"] = features_input["window_id"].map(max_psi_per_window)
    if not qual_df.empty:
        max_missing = qual_df.groupby("window_id")["missing_rate_delta"].apply(
            lambda x: x.abs().max()
        )
        max_outlier = qual_df.groupby("window_id")["outlier_rate_delta"].apply(
            lambda x: x.abs().max()
        )
        features_input["missing_rate_max_delta"] = features_input["window_id"].map(max_missing)
        features_input["outlier_rate_max_delta"] = features_input["window_id"].map(max_outlier)

    for col in ["auc", "ks", "prediction_mean", "max_feature_psi_7d",
                "missing_rate_max_delta", "outlier_rate_max_delta"]:
        if col not in features_input:
            features_input[col] = None

    detector_df = run_detectors(features_input)
    alarm_count = int(detector_df["alarm_flag"].sum()) if not detector_df.empty else 0
    print(f"  检测器告警总数: {alarm_count}")

    if not detector_df.empty:
        alarm_by_detector = (detector_df[detector_df["alarm_flag"]]
                             .groupby("detector_name").size())
        for det, count in alarm_by_detector.items():
            print(f"    {det}: {count} 次")
        if alarm_count == 0:
            print("    OK 四个检测器均未触发告警")

    # === ⑦ W3 最新窗口综合报告 ===
    print("\n> W3 最新窗口综合报告")
    print("=" * 70)

    latest = perf_df[perf_df["start"] == latest_start]
    if not latest.empty:
        row = latest.iloc[-1]
        print(f"  窗口: {row['window_id']}")
        print(f"  样本数: {row['sample_count']:,}  |  坏样本: {row['bad_count']}")
        print(f"  AUC: {row['auc']}  |  KS: {row['ks']}")
        print(f"  Prediction PSI: {row['prediction_psi']}")
        print(f"  Prediction Mean: {row['prediction_mean']:.4f}")

    if not drift_df.empty:
        psi_vals = [r["psi"] for r in drift_rows
                    if r.get("psi") is not None and r["start"] == latest_start]
        if psi_vals:
            max_psi = max(psi_vals)
            mean_psi = sum(psi_vals) / len(psi_vals)
            print(f"  特征 PSI (均值): {mean_psi:.4f}  |  最大: {max_psi:.4f}")
            if max_psi > 0.25:
                print(f"  !! 严重漂移！max PSI > 0.25")
            elif max_psi > 0.1:
                print(f"  ! 轻微漂移。max PSI > 0.1")
            else:
                print(f"  OK 特征分布稳定。max PSI < 0.1")

    # 趋势
    if len(perf_df) >= 5:
        auc_slope = trailing_slope([r["auc"] for r in perf_rows if r.get("auc") is not None])
        psi_slope = trailing_slope(
            [r["psi"] for r in drift_rows if r.get("psi") is not None]
        ) if drift_rows else None
        print(f"\n  趋势:")
        print(f"    AUC 斜率: {auc_slope}  {'v 恶化' if auc_slope and auc_slope < -0.001 else '^ 改善' if auc_slope and auc_slope > 0.001 else '- 稳定'}")
        if psi_slope:
            print(f"    PSI 斜率: {psi_slope:.6f}  {'^ 恶化' if psi_slope > 0.0001 else '- 稳定'}")

    print("\n" + "=" * 70)
    print("  监控链路演示完成。")
    print("=" * 70)


if __name__ == "__main__":
    main()
