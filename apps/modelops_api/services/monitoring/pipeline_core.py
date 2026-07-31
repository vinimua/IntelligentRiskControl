"""交接包 pipeline.py 核心函数 — 逐行移植，逻辑完全一致。

包含 _monitor_one 和 _natural_monitoring。
属性名与交接包 MonitoringBaselinePackage 一致：
  baseline.performance_reference_json
  baseline.binning_rules_json
  baseline.feature_profile_uri
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .drift.algorithms import (
    benjamini_hochberg,
    categorical_drift,
    compute_performance_metrics as performance_metrics,
    continuous_drift,
    feature_quality,
)
from .drift.output_monitor import output_metrics
from .rolling import iter_rolling_windows
from .trend_features import trailing_slope


def _monitor_one(
    source: pd.DataFrame,
    predictions: pd.DataFrame,
    monitor_window_id: str,
    context,   # MonitoringBaseline 对象
    baseline,
    reference: pd.DataFrame,
    reference_scores: pd.Series,
    baseline_profile: pd.DataFrame,
    data_track: str,
    trace_id: str,
    scenario_id: str | None = None,
    scenario_instance_id: str | None = None,
    anomaly_label: int | None = None,
    window_days: int | None = None,
    window_start: pd.Timestamp | None = None,
    window_end: pd.Timestamp | None = None,
    min_samples: int = 2000,
    min_bad: int = 50,
) -> tuple[dict, list[dict], list[dict]]:
    """交接包原版 _monitor_one — 逐行移植。

    参数签名与交接包完全一致。context 和 baseline 的属性名
    已对齐为交接包 MonitoringBaselinePackage 的命名。
    """
    now = datetime.now(timezone.utc)
    merged = source.merge(predictions[["sample_id", "risk_score"]], on="sample_id", how="inner")
    sample_count = len(merged)
    bad_count = int(merged["is_bad"].sum()) if "is_bad" in merged else None
    if "is_bad" not in merged:
        status, reason = "PENDING_LABEL", "y_true unavailable"
    elif sample_count < min_samples or bad_count is None or bad_count < min_bad:
        status, reason = "INSUFFICIENT_LABEL", f"sample_count={sample_count}, bad_count={bad_count}"
    else:
        status, reason = "READY", None
    metrics = performance_metrics(merged["is_bad"], merged["risk_score"]) if status == "READY" else {k: None for k in ("auc", "ks", "pr_auc", "brier", "ece", "bad_recall")}
    metric_deltas = {key: (None if metrics[key] is None or baseline.performance_reference_json.get(key) is None else float(metrics[key]) - float(baseline.performance_reference_json[key])) for key in metrics}
    adverse_changes = [
        -float(metric_deltas[key]) if key in {"auc", "ks", "pr_auc", "bad_recall"} else float(metric_deltas[key])
        for key in metrics if metric_deltas[key] is not None
    ]
    bad_rate = None if bad_count is None or sample_count == 0 else float(bad_count / sample_count)
    baseline_bad_rate = baseline.performance_reference_json.get("bad_rate")
    score_rule = baseline.binning_rules_json["__risk_score__"]
    outputs = output_metrics(merged["risk_score"], reference_scores, score_rule["edges"])
    common = {
        "trace_id": trace_id, "model_id": context.model_id, "model_version": context.model_version,
        "baseline_id": baseline.baseline_id, "baseline_version": baseline.baseline_version,
        "monitor_window_id": monitor_window_id, "window_start": window_start if window_start is not None else pd.to_datetime(source["apply_time"]).min(),
        "window_end": window_end if window_end is not None else pd.to_datetime(source["apply_time"]).max(), "sample_count": sample_count,
        "bad_count": bad_count, "data_track": data_track, "scenario_id": scenario_id,
        "scenario_instance_id": scenario_instance_id,
        "window_days": window_days,
    }
    performance = {**common, "status": status, "reason": reason, **metrics, **outputs,
        "bad_rate": bad_rate,
        "bad_rate_delta": None if bad_rate is None or baseline_bad_rate is None else bad_rate - float(baseline_bad_rate),
        "performance_drop_max": max(0.0, max(adverse_changes, default=0.0)),
        "metric_deltas": metric_deltas,
        "metric_slopes": {}, "anomaly_label": anomaly_label, "created_at": now,
    }
    quality_rows: list[dict] = []
    drift_rows: list[dict] = []
    p_positions: list[int] = []
    p_values: list[float | None] = []
    profile_index = baseline_profile.set_index("feature_name")
    for feature, rule in baseline.binning_rules_json.items():
        if feature == "__risk_score__" or feature not in source or feature not in reference:
            continue
        quality = feature_quality(source[feature], profile_index.loc[feature], rule["feature_type"])
        quality_rows.append({**common, "feature_name": feature, **quality, "created_at": now, "anomaly_label": anomaly_label})
        if rule["feature_type"] == "categorical":
            drift = categorical_drift(reference[feature], source[feature], rule["categories"])
            row = {**common, "feature_name": feature, "feature_type": "categorical", **drift, "wasserstein_distance": None, "ks_statistic": None, "ks_p_value": None, "ks_q_value": None, "created_at": now, "anomaly_label": anomaly_label}
        else:
            drift = continuous_drift(reference[feature], source[feature], rule["edges"])
            row = {**common, "feature_name": feature, "feature_type": "continuous", **drift, "category_share_change": None, "unknown_category_rate": 0.0, "ks_q_value": None, "created_at": now, "anomaly_label": anomaly_label}
            p_positions.append(len(drift_rows)); p_values.append(row["ks_p_value"])
        drift_rows.append(row)
    for position, q_value in zip(p_positions, benjamini_hochberg(p_values)):
        drift_rows[position]["ks_q_value"] = q_value
    return performance, quality_rows, drift_rows


def natural_monitoring(
    source_df: pd.DataFrame,    # W1+W2+W3 合并数据（含 sample_id, apply_time, is_bad, risk_score, y_pred_proba）
    predictions: pd.DataFrame,  # 预测结果（含 sample_id, risk_score）
    context,                    # MonitoringBaseline (baseline 自己也是 context)
    baseline,                   # MonitoringBaseline
    reference: pd.DataFrame,    # W0 数据
    trace_id: str,
    window_days: int = 7,
    step_days: int = 1,
    min_samples: int = 2000,
    min_bad: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """交接包 _natural_monitoring 简化版 — 单粒度滚动窗口。

    交接包原版支持多粒度（7D+30D），这里简化为单粒度。
    其他逻辑完全一致。
    """
    reference_scores = pd.Series(reference["y_pred_proba"])
    baseline_profile = pd.read_parquet(Path(baseline.feature_profile_uri))

    performance_rows: list[dict] = []
    quality_rows: list[dict] = []
    drift_rows: list[dict] = []
    history: dict[str, list[float | None]] = {
        "auc": [], "ks": [], "prediction_mean": [], "prediction_psi": [],
    }

    for start, end, source in iter_rolling_windows(
        source_df, window_days, step_days, require_full_window=False,
    ):
        window_id = f"NAT_{window_days}D_{start:%Y%m%d}_{end:%Y%m%d}"
        ids = set(source["sample_id"])
        window_predictions = predictions[predictions["sample_id"].isin(ids)]

        performance, quality, drift = _monitor_one(
            source, window_predictions, window_id,
            context, baseline,
            reference, reference_scores, baseline_profile,
            "NATURAL", trace_id,
            window_days=window_days, window_start=start, window_end=end,
            min_samples=min_samples, min_bad=min_bad,
        )

        for signal in history:
            history[signal].append(performance.get(signal))
            performance["metric_slopes"][signal] = trailing_slope(
                history[signal], 5,
            )

        performance_rows.append(performance)
        quality_rows.extend(quality)
        drift_rows.extend(drift)

    return (
        pd.DataFrame(performance_rows),
        pd.DataFrame(quality_rows),
        pd.DataFrame(drift_rows),
    )
