"""Sentinel 特征向量构建 — 基于交接包 monitor_feature_builder.py。

将性能指标 + 数据质量 + 漂移指标 + 检测器信号
聚合成每个模型-窗口一行、Sentinel 可直接推理的特征矩阵。
"""

from __future__ import annotations

import json

import pandas as pd


# 检测器输出列映射
_DETECTOR_COLUMNS = {
    "ADWIN": "adwin_alarm_count",
    "PAGE_HINKLEY": "ph_alarm_count",
    "KSWIN": "kswin_alarm_count",
    "ROBUST_Z": "robust_z_alarm_count",
}

# 行标识键
_ROW_KEYS = [
    "trace_id", "model_id", "model_version", "baseline_id", "baseline_version",
    "monitor_window_id", "window_start", "window_end", "window_days", "data_track",
    "scenario_id", "scenario_instance_id", "anomaly_label",
]


def _present(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """返回 frame 中实际存在的列。"""
    return [c for c in columns if c in frame.columns]


def _group_keys(frame: pd.DataFrame) -> list[str]:
    """返回聚合分组键。"""
    return [c for c in _present(frame, _ROW_KEYS) if not frame[c].isna().all()]


def _maximum_category_share_delta(values: pd.Series) -> float:
    """从 category_share_change JSON 中提取最大绝对变化。"""
    maximum = 0.0
    for value in values.dropna():
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                continue
        if isinstance(value, dict) and value:
            maximum = max(maximum, max(abs(float(v)) for v in value.values()))
    return float(maximum)


def _add_horizon_features(result: pd.DataFrame) -> pd.DataFrame:
    """将 7D/30D 两个粒度的 PSI 值并列到同一行。

    7D 行作为主行，30D 的 prediction_psi 和 max_feature_psi
    作为额外列（_7d / _30d）加入。
    """
    if "window_days" not in result or "window_end" not in result:
        return result

    join_keys = _present(result, [
        "model_id", "model_version", "baseline_id", "baseline_version",
        "data_track", "scenario_id", "scenario_instance_id", "window_end",
    ])

    for horizon in (7, 30):
        horizon_rows = result[
            pd.to_numeric(result["window_days"], errors="coerce") == horizon
        ]
        for source in ("prediction_psi", "max_feature_psi"):
            target = f"{source}_{horizon}d"
            if source not in result:
                continue
            values = (
                horizon_rows[join_keys + [source]]
                .drop_duplicates(join_keys, keep="last")
                .rename(columns={source: target})
            )
            result = result.merge(values, on=join_keys, how="left")
            scenario_mask = result["data_track"].eq("SCENARIO")
            result.loc[scenario_mask, target] = result.loc[scenario_mask, target].fillna(
                result.loc[scenario_mask, source]
            )

    return result


def build_monitor_feature_vector(
    performance: pd.DataFrame,
    quality: pd.DataFrame,
    drift: pd.DataFrame,
    detectors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """聚合 DS05-DS08 为 Sentinel 特征矩阵。

    每行 = 一个模型-窗口的完整监控特征向量。

    Args:
        performance: 性能监控 DataFrame（performance_monitor.parquet）。
        quality: 数据质量 DataFrame（data_quality_monitor.parquet）。
        drift: 漂移监控 DataFrame（drift_monitor.parquet）。
        detectors: 检测器信号 DataFrame（detector_signals.parquet），可选。

    Returns:
        每行一个监测窗口的特征 DataFrame。
    """
    present_key = _present(performance, _ROW_KEYS)
    perf_columns = present_key + [
        c for c in [
            "auc", "ks", "pr_auc", "brier", "ece", "bad_recall",
            "bad_rate", "bad_rate_delta", "performance_drop_max",
            "prediction_mean", "prediction_std", "prediction_psi",
        ]
        if c in performance.columns
    ]
    result = performance[perf_columns].copy()

    # 数据质量聚合
    if not quality.empty:
        quality_keys = _group_keys(quality)
        aggregations = {
            "missing_rate_max_delta": ("missing_rate_delta", "max"),
            "outlier_rate_max_delta": ("outlier_rate_delta", "max"),
            "dq_score_min": ("dq_score", "min"),
            "range_violation_rate_max": ("range_violation_rate", "max"),
            "unknown_category_rate_max": ("unknown_category_rate", "max"),
        }
        if "default_value_rate_delta" in quality:
            aggregations["default_value_rate_max_delta"] = (
                "default_value_rate_delta", "max"
            )
        dq = (
            quality.groupby(quality_keys, dropna=False)
            .agg(**aggregations)
            .reset_index()
        )
        result = result.merge(dq, on=quality_keys, how="left")

    # 漂移聚合
    if not drift.empty:
        drift_keys = _group_keys(drift)
        aggregations = {
            "max_feature_psi": ("psi", "max"),
            "max_feature_js": ("js_divergence", "max"),
            "min_ks_q_value": ("ks_q_value", "min"),
        }
        if "ks_statistic" in drift:
            aggregations["max_feature_ks_statistic"] = ("ks_statistic", "max")
        if "category_share_change" in drift:
            aggregations["max_segment_share_delta"] = (
                "category_share_change",
                _maximum_category_share_delta,
            )
        dr = (
            drift.groupby(drift_keys, dropna=False)
            .agg(**aggregations)
            .reset_index()
        )
        result = result.merge(dr, on=drift_keys, how="left")

    # 7D/30D 多尺度特征
    result = _add_horizon_features(result)

    # 检测器信号聚合
    for col in _DETECTOR_COLUMNS.values():
        result[col] = 0

    if detectors is not None and not detectors.empty:
        detector_keys = _present(detectors, [
            "model_id", "model_version", "monitor_window_id",
            "data_track", "scenario_id", "scenario_instance_id",
        ])
        alarms = (
            detectors.groupby(detector_keys + ["detector_name"], dropna=False)["alarm_flag"]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )
        for det, col in _DETECTOR_COLUMNS.items():
            if det not in alarms:
                alarms[det] = 0
            alarms = alarms.rename(columns={det: col})

        result = result.drop(
            columns=list(_DETECTOR_COLUMNS.values()), errors="ignore"
        ).merge(
            alarms[detector_keys + list(_DETECTOR_COLUMNS.values())],
            on=detector_keys,
            how="left",
        )
        for col in _DETECTOR_COLUMNS.values():
            result[col] = result[col].fillna(0).astype(int)

        result["detector_vote_ratio"] = (
            result[list(_DETECTOR_COLUMNS.values())].gt(0).mean(axis=1)
        )
    else:
        result["detector_vote_ratio"] = 0.0

    return result


def select_canonical_sentinel_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """选择 Sentinel 的标准决策行：只保留 7 天窗口行。

    7 天行已通过 _add_horizon_features 携带了 30 天 PSI 特征。
    """
    if "window_days" not in frame:
        return frame.copy()
    window_days = pd.to_numeric(frame["window_days"], errors="coerce")
    keep = window_days.eq(7) | window_days.isna()
    return frame.loc[keep].reset_index(drop=True)
