"""W0 监控基线构建 — 基于交接包 baseline_builder.py。

基线包 = 分箱规则 + 特征概要 + 性能基准 + 分数分箱。
基线只在 W0 上构建一次，之后冻结不变。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .drift.algorithms import compute_performance_metrics


# ═══════════════════════════════════════════════════════════════
# 基线包数据结构
# ═══════════════════════════════════════════════════════════════


@dataclass
class MonitoringBaseline:
    """W0 监控基线包 — 属性名与交接包 MonitoringBaselinePackage 完全一致。"""

    baseline_id: str
    model_id: str
    model_version: str
    baseline_version: str

    # 与交接包 MonitoringBaselinePackage 一致的属性名
    performance_reference_json: dict[str, float | None] = field(default_factory=dict)
    binning_rules_json: dict[str, dict] = field(default_factory=dict)
    feature_profile_uri: str = ""  # feature_profile.parquet 路径
    feature_profiles: dict[str, dict] = field(default_factory=dict)

    # 额外属性（交接包也有或用到的）
    raw_performance_reference_json: dict[str, float | None] = field(default_factory=dict)
    score_edges: list[float] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    created_at: str = ""


# ═══════════════════════════════════════════════════════════════
# 分箱构建
# ═══════════════════════════════════════════════════════════════


def _build_edges(series: pd.Series, n_bins: int = 10) -> list[float]:
    """对连续变量构建等频分箱边界（基于 10 等分位）。

    首尾边界微微外扩，确保极值不掉出箱外。
    如果数据退化（唯一值 < 2），则用中心 ±1 兜底。
    """
    values = pd.to_numeric(series, errors="coerce").dropna()
    quantiles = np.unique(values.quantile(np.linspace(0, 1, n_bins + 1)).to_numpy(dtype=float))

    if len(quantiles) < 2:
        center = float(values.iloc[0]) if not values.empty else 0.0
        quantiles = np.array([center - 1.0, center + 1.0])

    span = max(float(quantiles[-1] - quantiles[0]), 1.0)
    quantiles[0] = float(quantiles[0] - span * 1e-6 - 1e-9)
    quantiles[-1] = float(quantiles[-1] + span * 1e-6 + 1e-9)

    return [float(v) for v in quantiles]


# ═══════════════════════════════════════════════════════════════
# 基线构建
# ═══════════════════════════════════════════════════════════════


def build_monitoring_baseline(
    w0_data: pd.DataFrame,
    model_id: str,
    model_version: str,
    baseline_id: str | None = None,
    baseline_version: str = "V1",
    feature_names: list[str] | None = None,
    categorical_features: dict[str, list[str]] | None = None,
    score_column: str = "y_pred_proba",
    label_column: str = "y_true",
) -> MonitoringBaseline:
    """在 W0 参照数据上构建监控基线包。

    Args:
        w0_data: W0 窗口的完整数据（含特征列 + 标签 + 预测分）。
        model_id: 模型标识。
        model_version: 模型版本。
        baseline_id: 基线 ID（默认自动生成）。
        baseline_version: 基线版本。
        feature_names: 特征列名列表（如不提供则自动从 feature_ 前缀检测）。
        categorical_features: {feature_name: [category_values]} 映射。
        score_column: 预测分列名。
        label_column: 标签列名。

    Returns:
        MonitoringBaseline 对象。
    """
    baseline_id = baseline_id or f"BASELINE_{model_id}_{model_version}_{baseline_version}"

    # 自动检测特征列
    if feature_names is None:
        feature_names = sorted(
            [c for c in w0_data.columns if c.startswith("feature_")]
        )

    categorical_features = categorical_features or {}

    # 构建分箱规则 + 特征概要（变量名与交接包 MonitoringBaselinePackage 一致）
    binning_rules_json: dict[str, dict] = {}
    feature_profiles: dict[str, dict] = {}

    for name in feature_names:
        if name not in w0_data.columns:
            continue

        values = pd.to_numeric(w0_data[name], errors="coerce")
        cats = categorical_features.get(name)

        median = float(values.median())
        mad = float((values - median).abs().median())
        fallback_mad = max(
            1e-12,
            float(values.quantile(0.75)) - float(values.quantile(0.25)),
        ) / 1.349
        outlier_mad = mad if mad > 1e-12 else fallback_mad
        baseline_outlier_rate = (
            0.0
            if cats
            else float(
                ((values - median).abs() > 3.0 * outlier_mad).fillna(False).mean()
            )
        )

        feature_profiles[name] = {
            "missing_rate": float(w0_data[name].isna().mean()),
            "median": median,
            "mad": mad,
            "q1": float(values.quantile(0.25)),
            "q3": float(values.quantile(0.75)),
            "outlier_rate": baseline_outlier_rate,
            "default_value_rate": float(values.eq(0).mean()),
            "allowed_min": None,
            "allowed_max": None,
            "categories": json.dumps(cats or []),
        }

        binning_rules_json[name] = {
            "feature_type": "categorical" if cats else "continuous",
            "edges": [] if cats else _build_edges(values),
            "categories": [str(v) for v in (cats or [])],
            "baseline_profile": feature_profiles[name],
        }

    # 分数分箱
    if score_column in w0_data.columns:
        scores = pd.to_numeric(w0_data[score_column], errors="coerce").dropna()
        score_edges = _build_edges(scores)
    else:
        score_edges = _build_edges(pd.Series(np.linspace(0, 1, 101)))
        scores = pd.Series(dtype=float)

    binning_rules_json["__risk_score__"] = {
        "feature_type": "continuous",
        "edges": score_edges,
        "categories": [],
    }

    # 保存 feature_profile.parquet（交接包 _monitor_one 需要从文件读取）
    profile_rows = [{"feature_name": name, **profile} for name, profile in feature_profiles.items()]
    profile_df = pd.DataFrame(profile_rows)
    profile_path = f"assets/baselines/{model_id}/{model_version}/{baseline_version}/feature_profile.parquet"
    import os
    os.makedirs(os.path.dirname(profile_path), exist_ok=True)
    profile_df.to_parquet(profile_path, index=False)

    # 性能基准（校准分数）
    if label_column in w0_data.columns and score_column in w0_data.columns:
        perf = compute_performance_metrics(w0_data[label_column], w0_data[score_column])
        perf["bad_rate"] = float(pd.to_numeric(w0_data[label_column], errors="coerce").mean())
    else:
        perf = {k: None for k in ("auc","ks","pr_auc","brier","ece","bad_recall","bad_rate")}

    # 性能基准（原始分数）— 排序指标用
    if label_column in w0_data.columns and "risk_score" in w0_data.columns:
        raw_perf = compute_performance_metrics(w0_data[label_column], w0_data["risk_score"])
    else:
        raw_perf = {k: None for k in ("auc","ks","pr_auc","brier","ece","bad_recall")}

    return MonitoringBaseline(
        baseline_id=baseline_id,
        model_id=model_id,
        model_version=model_version,
        baseline_version=baseline_version,
        binning_rules_json=binning_rules_json,
        feature_profiles=feature_profiles,
        performance_reference_json=perf,
        raw_performance_reference_json=raw_perf,
        feature_profile_uri=profile_path,
        score_edges=score_edges,
        feature_names=feature_names,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
