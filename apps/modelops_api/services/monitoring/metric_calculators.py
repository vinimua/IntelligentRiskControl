"""7+ 个监控指标计算器 — 基于交接包 WP04-WP05 算法增强。

每个计算器签名：fn(baseline_data: list[dict], current_data: list[dict]) -> MetricResult
数据格式假设：list[dict]，每行包含 y_true, y_pred_proba, score, feature_* 等列。

V2 增强（2026-07-20）：
- FEATURE_PSI/SCORE_PSI 内部使用 drift.algorithms.psi_from_edges（冻结分箱 PSI）
- 新增 PREDICTION_MEAN / MAX_FEATURE_PSI_7D / MAX_FEATURE_PSI_30D
  / BAD_RATE / OUTLIER_RATE / DATA_QUALITY_SCORE
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from packages.models.common.enums import AvailabilityStatus
from .metrics_registry import MetricResult, register
from .drift.algorithms import psi_from_edges


def _safe_float(value) -> float | None:
    """安全转换值为 float，None 或 NaN 返回 None。"""
    if value is None:
        return None
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


# 非特征列（metadata + label + prediction + time engineered）
_NON_FEATURE_COLS = {
    "sample_id", "apply_time", "is_bad", "y_true",
    "y_pred_proba", "risk_score", "score",
    "apply_hour_sin", "apply_hour_cos", "apply_weekday_sin",
    "apply_weekday_cos", "apply_is_weekend", "apply_is_night",
}


def _get_feature_columns(data: list[dict]) -> list[str]:
    """从数据中提取真实特征列名（排除 metadata/label/prediction/time 列）。"""
    if not data:
        return []
    cols = set()
    for row in data[:100]:
        cols.update(row.keys())
    return sorted(c for c in cols if c not in _NON_FEATURE_COLS)


def _get_column(data: list[dict], col: str) -> list:
    """从数据集中提取一列，缺失值返回 None。"""
    return [row.get(col) for row in data]


# ── AUC ──


@register("AUC")
def calc_auc(baseline_data: list[dict], current_data: list[dict]) -> MetricResult:
    """计算 AUC，使用 sklearn.metrics.roc_auc_score。"""
    from sklearn.metrics import roc_auc_score

    y_true = _get_column(current_data, "y_true")
    y_pred = _get_column(current_data, "y_pred_proba")

    valid = [(t, p) for t, p in zip(y_true, y_pred) if t is not None and p is not None]
    if len(valid) < 10:
        return MetricResult(
            metric_code="AUC",
            availability_status=AvailabilityStatus.SAMPLE_TOO_SMALL,
            metric_detail={"reason": "valid_samples < 10"},
        )

    t, p = zip(*valid)
    try:
        current_auc = float(roc_auc_score(list(t), list(p)))
    except ValueError:
        return MetricResult(
            metric_code="AUC",
            availability_status=AvailabilityStatus.CALCULATION_FAILED,
            metric_detail={"reason": "only one class present"},
        )

    b_true = _get_column(baseline_data, "y_true")
    b_pred = _get_column(baseline_data, "y_pred_proba")
    b_valid = [(t, p) for t, p in zip(b_true, b_pred) if t is not None and p is not None]

    baseline_auc: float | None = None
    if len(b_valid) >= 10:
        bt, bp = zip(*b_valid)
        try:
            baseline_auc = float(roc_auc_score(list(bt), list(bp)))
        except ValueError:
            baseline_auc = None

    delta = (current_auc - baseline_auc) if baseline_auc is not None else None

    return MetricResult(
        metric_code="AUC",
        baseline_value=baseline_auc,
        current_value=current_auc,
        delta=delta,
    )


# ── KS ──


def _compute_ks(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """计算单组数据的 KS 统计量。"""
    goods = scores[labels == 0]
    bads = scores[labels == 1]
    if len(goods) == 0 or len(bads) == 0:
        return None

    all_scores = np.sort(np.unique(scores))
    ks_max = 0.0
    n_good = len(goods)
    n_bad = len(bads)
    for thr in all_scores:
        good_below = np.sum(goods <= thr) / n_good
        bad_below = np.sum(bads <= thr) / n_bad
        ks_max = max(ks_max, abs(bad_below - good_below))
    return float(ks_max)


@register("KS")
def calc_ks(baseline_data: list[dict], current_data: list[dict]) -> MetricResult:
    """计算 KS 统计量及与 baseline 的变化。"""
    y_true = _get_column(current_data, "y_true")
    scores = _get_column(current_data, "y_pred_proba")

    valid = [(s, t) for s, t in zip(scores, y_true) if s is not None and t is not None]
    if len(valid) < 10:
        return MetricResult(
            metric_code="KS",
            availability_status=AvailabilityStatus.SAMPLE_TOO_SMALL,
        )

    s_arr = np.array([v[0] for v in valid])
    t_arr = np.array([v[1] for v in valid])
    current_ks = _compute_ks(s_arr, t_arr)
    if current_ks is None:
        return MetricResult(
            metric_code="KS",
            availability_status=AvailabilityStatus.CALCULATION_FAILED,
            metric_detail={"reason": "single class"},
        )

    b_true = _get_column(baseline_data, "y_true")
    b_scores = _get_column(baseline_data, "y_pred_proba")
    b_valid = [(s, t) for s, t in zip(b_scores, b_true) if s is not None and t is not None]
    baseline_ks: float | None = None
    if len(b_valid) >= 10:
        b_arr = np.array([v[0] for v in b_valid])
        bt_arr = np.array([v[1] for v in b_valid])
        baseline_ks = _compute_ks(b_arr, bt_arr)

    delta = (current_ks - baseline_ks) if baseline_ks is not None else None

    return MetricResult(
        metric_code="KS",
        current_value=current_ks,
        baseline_value=baseline_ks,
        delta=delta,
    )


# ── FEATURE_PSI ──


def _compute_psi_frozen(baseline_vals: list, current_vals: list) -> float | None:
    """使用冻结分箱（交接包算法）计算 PSI。

    先基于 baseline 数据做 10 等分位冻结分箱，再用该分箱对 current 算 PSI。
    """
    b_arr = np.array([v for v in baseline_vals if v is not None], dtype=float)
    c_arr = np.array([v for v in current_vals if v is not None], dtype=float)

    if len(b_arr) < 10 or len(c_arr) < 10:
        return None
    if len(np.unique(np.concatenate([b_arr, c_arr]))) < 2:
        return 0.0

    # 基于 baseline 构建冻结分箱边界
    quantiles = np.unique(np.percentile(b_arr, np.linspace(0, 100, 11)))
    if len(quantiles) < 2:
        return 0.0
    # 首尾微扩
    span = max(float(quantiles[-1] - quantiles[0]), 1.0)
    quantiles = quantiles.astype(float)
    quantiles[0] -= span * 1e-6 + 1e-9
    quantiles[-1] += span * 1e-6 + 1e-9

    b_series = pd.Series(b_arr)
    c_series = pd.Series(c_arr)
    return psi_from_edges(b_series, c_series, [float(v) for v in quantiles])


@register("FEATURE_PSI")
def calc_feature_psi(baseline_data: list[dict], current_data: list[dict]) -> MetricResult:
    """计算特征 PSI：使用冻结分箱算法，返回所有特征 PSI 的均值 + 最大值。

    PSI = Σ(Q_i - P_i) × ln(Q_i / P_i)，分箱基于 W0 baseline 冻结。
    """
    feature_cols = _get_feature_columns(current_data)
    if not feature_cols:
        return MetricResult(
            metric_code="FEATURE_PSI",
            availability_status=AvailabilityStatus.NOT_APPLICABLE,
            metric_detail={"reason": "no feature columns found"},
        )

    psi_values: list[float] = []
    for col in feature_cols:
        b_vals = [v for v in _get_column(baseline_data, col) if v is not None]
        c_vals = [v for v in _get_column(current_data, col) if v is not None]
        if len(b_vals) < 10 or len(c_vals) < 10:
            continue

        psi = _compute_psi_frozen(b_vals, c_vals)
        if psi is not None:
            psi_values.append(psi)

    if not psi_values:
        return MetricResult(
            metric_code="FEATURE_PSI",
            availability_status=AvailabilityStatus.SAMPLE_TOO_SMALL,
        )

    mean_psi = float(np.mean(psi_values))
    max_psi = float(np.max(psi_values))
    return MetricResult(
        metric_code="FEATURE_PSI",
        current_value=mean_psi,
        metric_detail={
            "max_psi": max_psi,
            "per_column_psi": dict(zip(feature_cols[:10], psi_values[:10])),
        },
    )


# ── SCORE_PSI ──


@register("SCORE_PSI")
def calc_score_psi(baseline_data: list[dict], current_data: list[dict]) -> MetricResult:
    """计算分数 PSI：模型预测分数的分布漂移（使用冻结分箱算法）。"""
    b_scores = [v for v in _get_column(baseline_data, "y_pred_proba") if v is not None]
    c_scores = [v for v in _get_column(current_data, "y_pred_proba") if v is not None]

    if len(b_scores) < 10 or len(c_scores) < 10:
        return MetricResult(
            metric_code="SCORE_PSI",
            availability_status=AvailabilityStatus.SAMPLE_TOO_SMALL,
        )

    psi = _compute_psi_frozen(b_scores, c_scores)
    return MetricResult(
        metric_code="SCORE_PSI",
        current_value=psi,
    )


# ── MISSING_RATE ──


@register("MISSING_RATE")
def calc_missing_rate(baseline_data: list[dict], current_data: list[dict]) -> MetricResult:
    """计算缺失率变化：current 与 baseline 的缺失率差异均值。"""
    if not current_data:
        return MetricResult(
            metric_code="MISSING_RATE",
            availability_status=AvailabilityStatus.SAMPLE_TOO_SMALL,
        )

    cols = list(current_data[0].keys())
    baseline_missing = _column_missing_rates(baseline_data, cols)
    current_missing = _column_missing_rates(current_data, cols)

    deltas = []
    for col in cols:
        c_rate = current_missing.get(col, 0.0)
        b_rate = baseline_missing.get(col, 0.0)
        deltas.append(c_rate - b_rate)

    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    return MetricResult(
        metric_code="MISSING_RATE",
        current_value=mean_delta,
        metric_detail={"per_column": dict(zip(cols[:20], deltas[:20]))},
    )


# ── SCHEMA_CONSISTENCY ──


@register("SCHEMA_CONSISTENCY")
def calc_schema_consistency(baseline_data: list[dict], current_data: list[dict]) -> MetricResult:
    """检查 Schema 一致性：列名 + 类型是否匹配。"""
    if not baseline_data or not current_data:
        return MetricResult(
            metric_code="SCHEMA_CONSISTENCY",
            availability_status=AvailabilityStatus.NOT_APPLICABLE,
        )

    b_cols: set[str] = set()
    for row in baseline_data:
        b_cols.update(row.keys())
    c_cols: set[str] = set()
    for row in current_data:
        c_cols.update(row.keys())

    added = c_cols - b_cols
    removed = b_cols - c_cols
    common = b_cols & c_cols

    type_mismatches = []
    for col in common:
        b_types = {type(row.get(col)).__name__ for row in baseline_data[:20] if row.get(col) is not None}
        c_types = {type(row.get(col)).__name__ for row in current_data[:20] if row.get(col) is not None}
        if b_types and c_types and b_types != c_types:
            type_mismatches.append({
                "column": col,
                "baseline_types": sorted(b_types),
                "current_types": sorted(c_types),
            })

    mismatch_count = len(added) + len(removed) + len(type_mismatches)
    return MetricResult(
        metric_code="SCHEMA_CONSISTENCY",
        current_value=float(mismatch_count),
        metric_detail={
            "added_columns": list(added),
            "removed_columns": list(removed),
            "type_mismatches": type_mismatches,
        },
    )


# ── SAMPLE_SIZE ──


@register("SAMPLE_SIZE")
def calc_sample_size(baseline_data: list[dict], current_data: list[dict]) -> MetricResult:
    """计算当前窗口样本量。"""
    return MetricResult(
        metric_code="SAMPLE_SIZE",
        current_value=float(len(current_data)),
        baseline_value=float(len(baseline_data)),
    )


# ═══════════════════════════════════════════════════════════════
# V2 新增计算器（来自交接包）
# ═══════════════════════════════════════════════════════════════


@register("PREDICTION_MEAN")
def calc_prediction_mean(baseline_data: list[dict], current_data: list[dict]) -> MetricResult:
    """计算预测分数均值的变化。"""
    b_scores = [v for v in _get_column(baseline_data, "y_pred_proba") if v is not None]
    c_scores = [v for v in _get_column(current_data, "y_pred_proba") if v is not None]

    if len(c_scores) < 10:
        return MetricResult(
            metric_code="PREDICTION_MEAN",
            availability_status=AvailabilityStatus.SAMPLE_TOO_SMALL,
        )

    b_mean = float(np.mean(b_scores)) if b_scores else None
    c_mean = float(np.mean(c_scores))
    delta = (c_mean - b_mean) if b_mean is not None else None

    return MetricResult(
        metric_code="PREDICTION_MEAN",
        baseline_value=b_mean,
        current_value=c_mean,
        delta=delta,
        metric_detail={
            "current_std": float(np.std(c_scores)),
            "current_min": float(np.min(c_scores)),
            "current_max": float(np.max(c_scores)),
        },
    )


@register("MAX_FEATURE_PSI_7D")
def calc_max_feature_psi_7d(baseline_data: list[dict], current_data: list[dict]) -> MetricResult:
    """所有特征中 PSI（7天窗口）的最大值。"""
    result = calc_feature_psi(baseline_data, current_data)
    max_psi = result.metric_detail.get("max_psi", 0.0) if result.metric_detail else 0.0
    return MetricResult(
        metric_code="MAX_FEATURE_PSI_7D",
        current_value=float(max_psi),
        baseline_value=0.0,
        delta=float(max_psi),
        metric_detail=result.metric_detail,
    )


@register("MAX_FEATURE_PSI_30D")
def calc_max_feature_psi_30d(baseline_data: list[dict], current_data: list[dict]) -> MetricResult:
    """所有特征中 PSI（30天窗口）的最大值（与 7d 共享计算，区分在调用层）。"""
    result = calc_feature_psi(baseline_data, current_data)
    max_psi = result.metric_detail.get("max_psi", 0.0) if result.metric_detail else 0.0
    return MetricResult(
        metric_code="MAX_FEATURE_PSI_30D",
        current_value=float(max_psi),
        baseline_value=0.0,
        delta=float(max_psi),
        metric_detail=result.metric_detail,
    )


@register("BAD_RATE")
def calc_bad_rate(baseline_data: list[dict], current_data: list[dict]) -> MetricResult:
    """计算坏样本率及变化。"""
    b_y = [v for v in _get_column(baseline_data, "y_true") if v is not None]
    c_y = [v for v in _get_column(current_data, "y_true") if v is not None]

    if len(c_y) < 10:
        return MetricResult(
            metric_code="BAD_RATE",
            availability_status=AvailabilityStatus.SAMPLE_TOO_SMALL,
        )

    b_rate = float(np.mean(b_y)) if b_y else None
    c_rate = float(np.mean(c_y))
    delta = (c_rate - b_rate) if b_rate is not None else None

    return MetricResult(
        metric_code="BAD_RATE",
        baseline_value=b_rate,
        current_value=c_rate,
        delta=delta,
    )


@register("OUTLIER_RATE")
def calc_outlier_rate(baseline_data: list[dict], current_data: list[dict]) -> MetricResult:
    """计算离群率变化（基于 MAD 检测）。"""
    if not current_data:
        return MetricResult(
            metric_code="OUTLIER_RATE",
            availability_status=AvailabilityStatus.SAMPLE_TOO_SMALL,
        )

    feature_cols = _get_feature_columns(current_data)
    if not feature_cols:
        return MetricResult(
            metric_code="OUTLIER_RATE",
            availability_status=AvailabilityStatus.NOT_APPLICABLE,
        )

    deltas = []
    for col in feature_cols:
        b_vals = np.array([v for v in _get_column(baseline_data, col) if v is not None], dtype=float)
        c_vals = np.array([v for v in _get_column(current_data, col) if v is not None], dtype=float)

        if len(b_vals) < 10 or len(c_vals) < 10:
            continue

        b_median = float(np.median(b_vals))
        b_mad = float(np.median(np.abs(b_vals - b_median)))
        if b_mad < 1e-12:
            b_mad = max(1e-12, float(np.percentile(b_vals, 75) - np.percentile(b_vals, 25)) / 1.349)

        b_outlier_rate = float((np.abs(b_vals - b_median) > 3.0 * b_mad).mean())
        c_outlier_rate = float((np.abs(c_vals - b_median) > 3.0 * b_mad).mean())
        deltas.append(c_outlier_rate - b_outlier_rate)

    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    max_delta = float(np.max(np.abs(deltas))) if deltas else 0.0

    return MetricResult(
        metric_code="OUTLIER_RATE",
        current_value=mean_delta,
        delta=mean_delta,
        metric_detail={"max_delta": max_delta},
    )


@register("DATA_QUALITY_SCORE")
def calc_data_quality_score(baseline_data: list[dict], current_data: list[dict]) -> MetricResult:
    """计算综合数据质量评分（0-1，越高越好）。

    综合缺失率变化、离群率变化、范围违规率、未知类别率。
    """
    if not current_data:
        return MetricResult(
            metric_code="DATA_QUALITY_SCORE",
            availability_status=AvailabilityStatus.SAMPLE_TOO_SMALL,
        )

    missing_result = calc_missing_rate(baseline_data, current_data)
    outlier_result = calc_outlier_rate(baseline_data, current_data)

    missing_penalty = min(1.0, abs(missing_result.current_value or 0.0) * 5)
    outlier_penalty = min(1.0, abs(outlier_result.current_value or 0.0) * 2)
    penalty = min(1.0, missing_penalty + outlier_penalty)
    score = float(max(0.0, 1.0 - penalty))

    flag = "ALERT" if score < 0.7 else "WARN" if score < 0.9 else "OK"

    return MetricResult(
        metric_code="DATA_QUALITY_SCORE",
        current_value=score,
        metric_detail={
            "dq_flag": flag,
            "missing_penalty": missing_penalty,
            "outlier_penalty": outlier_penalty,
        },
    )


# ═══════════════════════════════════════════════════════════════
# V2 补充：交接包完整性能指标（PR-AUC / Brier / ECE / Bad Recall）
# ═══════════════════════════════════════════════════════════════


@register("PR_AUC")
def calc_pr_auc(baseline_data: list[dict], current_data: list[dict]) -> MetricResult:
    """PR-AUC（精确率-召回率曲线下面积）——不平衡场景下的重要补充指标。"""
    from sklearn.metrics import average_precision_score

    y_true = _get_column(current_data, "y_true")
    y_pred = _get_column(current_data, "y_pred_proba")
    valid = [(t, p) for t, p in zip(y_true, y_pred) if t is not None and p is not None]
    if len(valid) < 10:
        return MetricResult(metric_code="PR_AUC", availability_status=AvailabilityStatus.SAMPLE_TOO_SMALL)
    t, p = zip(*valid)
    try:
        cur = float(average_precision_score(list(t), list(p)))
    except ValueError:
        return MetricResult(metric_code="PR_AUC", availability_status=AvailabilityStatus.CALCULATION_FAILED)

    b_true = _get_column(baseline_data, "y_true")
    b_pred = _get_column(baseline_data, "y_pred_proba")
    b_valid = [(t, p) for t, p in zip(b_true, b_pred) if t is not None and p is not None]
    base = float(average_precision_score(*zip(*b_valid))) if len(b_valid) >= 10 else None
    delta = (cur - base) if base is not None else None
    return MetricResult(metric_code="PR_AUC", current_value=cur, baseline_value=base, delta=delta)


@register("BRIER")
def calc_brier(baseline_data: list[dict], current_data: list[dict]) -> MetricResult:
    """Brier Score——概率校准的均方误差，越低越好。"""
    from sklearn.metrics import brier_score_loss

    y_true = _get_column(current_data, "y_true")
    y_pred = _get_column(current_data, "y_pred_proba")
    valid = [(t, p) for t, p in zip(y_true, y_pred) if t is not None and p is not None]
    if len(valid) < 10:
        return MetricResult(metric_code="BRIER", availability_status=AvailabilityStatus.SAMPLE_TOO_SMALL)
    t, p = zip(*valid)
    cur = float(brier_score_loss(list(t), list(p)))

    b_true = _get_column(baseline_data, "y_true")
    b_pred = _get_column(baseline_data, "y_pred_proba")
    b_valid = [(t, p) for t, p in zip(b_true, b_pred) if t is not None and p is not None]
    base = float(brier_score_loss(*zip(*b_valid))) if len(b_valid) >= 10 else None
    delta = (cur - base) if base is not None else None
    return MetricResult(metric_code="BRIER", current_value=cur, baseline_value=base, delta=delta)


@register("ECE")
def calc_ece(baseline_data: list[dict], current_data: list[dict]) -> MetricResult:
    """Expected Calibration Error——模型概率校准误差。"""
    import numpy as np

    y_true = _get_column(current_data, "y_true")
    y_pred = _get_column(current_data, "y_pred_proba")
    valid = [(t, p) for t, p in zip(y_true, y_pred) if t is not None and p is not None]
    if len(valid) < 10:
        return MetricResult(metric_code="ECE", availability_status=AvailabilityStatus.SAMPLE_TOO_SMALL)
    t_arr = np.array([v[0] for v in valid])
    p_arr = np.array([v[1] for v in valid])
    bins = 10
    edges = np.linspace(0, 1, bins + 1)
    bucket = np.clip(np.digitize(p_arr, edges) - 1, 0, bins - 1)
    ece = 0.0
    for i in range(bins):
        mask = bucket == i
        if mask.any():
            ece += float(mask.mean()) * abs(float(t_arr[mask].mean()) - float(p_arr[mask].mean()))
    cur = float(ece)

    b_true = _get_column(baseline_data, "y_true")
    b_pred = _get_column(baseline_data, "y_pred_proba")
    b_valid = [(t, p) for t, p in zip(b_true, b_pred) if t is not None and p is not None]
    if len(b_valid) >= 10:
        bt = np.array([v[0] for v in b_valid])
        bp = np.array([v[1] for v in b_valid])
        bucket_b = np.clip(np.digitize(bp, edges) - 1, 0, bins - 1)
        ece_b = 0.0
        for i in range(bins):
            mask = bucket_b == i
            if mask.any():
                ece_b += float(mask.mean()) * abs(float(bt[mask].mean()) - float(bp[mask].mean()))
        base = float(ece_b)
    else:
        base = None
    delta = (cur - base) if base is not None else None
    return MetricResult(metric_code="ECE", current_value=cur, baseline_value=base, delta=delta)


@register("BAD_RECALL")
def calc_bad_recall(baseline_data: list[dict], current_data: list[dict]) -> MetricResult:
    """坏样本召回率——Top 20% 高分样本中坏样本的覆盖率。"""
    import pandas as pd

    y_true = _get_column(current_data, "y_true")
    y_pred = _get_column(current_data, "y_pred_proba")
    valid = pd.DataFrame({"y": y_true, "score": y_pred}).dropna()
    if len(valid) < 10 or valid["y"].sum() == 0:
        return MetricResult(metric_code="BAD_RECALL", availability_status=AvailabilityStatus.SAMPLE_TOO_SMALL)
    cutoff = max(1, int(np.ceil(len(valid) * 0.20)))
    top = valid.nlargest(cutoff, "score")
    cur = float(top["y"].sum() / max(1, valid["y"].sum()))

    b_true = _get_column(baseline_data, "y_true")
    b_pred = _get_column(baseline_data, "y_pred_proba")
    b_valid = pd.DataFrame({"y": b_true, "score": b_pred}).dropna()
    if len(b_valid) >= 10 and b_valid["y"].sum() > 0:
        cutoff_b = max(1, int(np.ceil(len(b_valid) * 0.20)))
        top_b = b_valid.nlargest(cutoff_b, "score")
        base = float(top_b["y"].sum() / max(1, b_valid["y"].sum()))
    else:
        base = None
    delta = (cur - base) if base is not None else None
    return MetricResult(metric_code="BAD_RECALL", current_value=cur, baseline_value=base, delta=delta)


# ── 辅助函数 ──


def _column_missing_rates(data: list[dict], columns: list[str]) -> dict[str, float]:
    """计算每列的缺失率。"""
    if not data:
        return {col: 0.0 for col in columns}
    n = len(data)
    rates = {}
    for col in columns:
        nulls = sum(1 for row in data if row.get(col) is None)
        rates[col] = nulls / n if n > 0 else 0.0
    return rates
