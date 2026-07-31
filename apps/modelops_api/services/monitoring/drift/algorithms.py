"""漂移检测与数据质量算法 — 基于交接包 WP02-WP08 完整实现。

本模块从 risk_inquiry_agent 的以下文件移植：
- drift_metrics.py: PSI / JS / KS / Wasserstein / Benjamini-Hochberg
- data_quality.py: 缺失率 / 范围违规 / 未知类别
- performance_metrics.py: AUC / KS / PR-AUC / Brier / ECE / Bad Recall

所有函数均为纯函数，输入 pandas Series/DataFrame，输出 dict/float。
分箱边界由调用方提供（通常由基线构建阶段从 W0 冻结）。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


# ═══════════════════════════════════════════════════════════════
# 内部辅助
# ═══════════════════════════════════════════════════════════════


def _normalized(counts: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """将计数转换为归一化比例，加 epsilon 平滑防除零。"""
    values = counts.astype(float) + epsilon
    return values / values.sum()


def _closed_bins(edges: list[float]) -> np.ndarray:
    """将分箱边界首尾扩展为 ±∞，确保极值不掉出箱外。

    W0 基线分箱的边界是有限值。如果 W3 数据超出该范围，
    np.histogram 会静默丢弃超界值——这正是监控要检测的漂移。
    扩展 ±∞ 兜底保留了所有观测值，同时不改变内部箱边界。
    """
    bins = np.asarray(edges, dtype=float)
    if bins.ndim != 1 or len(bins) < 2 or np.any(np.diff(bins) <= 0):
        raise ValueError("Drift edges must be a strictly increasing one-dimensional sequence")
    bins = bins.copy()
    bins[0] = -np.inf
    bins[-1] = np.inf
    return bins


# ═══════════════════════════════════════════════════════════════
# PSI（群体稳定性指标）— 冻结分箱
# ═══════════════════════════════════════════════════════════════


def psi_from_edges(
    reference: pd.Series,
    current: pd.Series,
    edges: list[float],
    epsilon: float = 1e-6,
) -> float:
    """基于冻结分箱边界计算 PSI。

    PSI = Σ (Q_i - P_i) × ln(Q_i / P_i)

    Args:
        reference: 参照组数据（W0 基线）。
        current: 当前组数据（W3 监测窗口）。
        edges: 冻结分箱边界（从 W0 基线一次性构建）。
        epsilon: 微小平滑值，防止 ln(0)。

    Returns:
        PSI 值，≥ 0。值越大表示分布漂移越严重。
    """
    bins = _closed_bins(edges)
    ref = np.histogram(pd.to_numeric(reference, errors="coerce").dropna(), bins=bins)[0]
    cur = np.histogram(pd.to_numeric(current, errors="coerce").dropna(), bins=bins)[0]
    p, q = _normalized(ref, epsilon), _normalized(cur, epsilon)
    return float(np.sum((q - p) * np.log(q / p)))


# ═══════════════════════════════════════════════════════════════
# 类别变量漂移
# ═══════════════════════════════════════════════════════════════


def categorical_drift(
    reference: pd.Series,
    current: pd.Series,
    categories: list[str],
) -> dict[str, object]:
    """计算类别变量的漂移指标。

    Returns:
        dict with keys: psi, js_divergence, category_share_change, unknown_category_rate
    """
    ref = reference.astype("string").fillna("__MISSING__")
    cur = current.astype("string").fillna("__MISSING__")
    universe = list(dict.fromkeys([*categories, "__MISSING__", *ref.unique().tolist(), *cur.unique().tolist()]))
    p = _normalized(np.array([(ref == v).sum() for v in universe]))
    q = _normalized(np.array([(cur == v).sum() for v in universe]))
    shares = {v: float(q[i] - p[i]) for i, v in enumerate(universe)}
    unknown = float((~cur.isin(categories + ["__MISSING__"])).mean())
    return {
        "psi": float(np.sum((q - p) * np.log(q / p))),
        "js_divergence": float(jensenshannon(p, q) ** 2),
        "category_share_change": shares,
        "unknown_category_rate": unknown,
    }


# ═══════════════════════════════════════════════════════════════
# 连续变量漂移
# ═══════════════════════════════════════════════════════════════


def continuous_drift(
    reference: pd.Series,
    current: pd.Series,
    edges: list[float],
) -> dict[str, float | None]:
    """计算连续变量的完整漂移指标。

    Returns:
        dict with keys: psi, js_divergence, wasserstein_distance, ks_statistic, ks_p_value
    """
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty:
        return {
            "psi": 0.0,
            "js_divergence": 0.0,
            "wasserstein_distance": None,
            "ks_statistic": None,
            "ks_p_value": None,
        }
    bins = _closed_bins(edges)
    ref_counts = np.histogram(ref, bins=bins)[0]
    cur_counts = np.histogram(cur, bins=bins)[0]
    p, q = _normalized(ref_counts), _normalized(cur_counts)
    ks = ks_2samp(ref, cur)
    return {
        "psi": float(np.sum((q - p) * np.log(q / p))),
        "js_divergence": float(jensenshannon(p, q) ** 2),
        "wasserstein_distance": float(wasserstein_distance(ref, cur)),
        "ks_statistic": float(ks.statistic),
        "ks_p_value": float(ks.pvalue),
    }


# ═══════════════════════════════════════════════════════════════
# Benjamini-Hochberg 多重检验校正
# ═══════════════════════════════════════════════════════════════


def benjamini_hochberg(p_values: list[float | None]) -> list[float | None]:
    """对多个 KS p 值做 BH FDR 校正。

    控制假发现率（FDR），防止对几十个特征同时做 KS 检验时
    出现大量假阳性。

    Args:
        p_values: 原始 p 值列表，允许 None。

    Returns:
        调整后的 q 值列表（与输入等长），None 位置保持 None。
    """
    valid = [(i, float(p)) for i, p in enumerate(p_values) if p is not None and np.isfinite(p)]
    result: list[float | None] = [None] * len(p_values)
    if not valid:
        return result
    ordered = sorted(valid, key=lambda item: item[1])
    m = len(ordered)
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        running = min(running, ordered[rank - 1][1] * m / rank)
        adjusted[rank - 1] = running
    for (original, _), value in zip(ordered, adjusted):
        result[original] = float(min(1.0, value))
    return result


# ═══════════════════════════════════════════════════════════════
# 数据质量
# ═══════════════════════════════════════════════════════════════


def feature_quality(
    current: pd.Series,
    baseline_row: pd.Series,
    feature_type: str,
) -> dict[str, float | int | str]:
    """计算单特征的数据质量指标。

    所有指标与 W0 基线对比，不依赖标签。

    Args:
        current: 当前窗口的特征值。
        baseline_row: W0 基线的特征概要行（含 missing_rate, median, mad, categories 等）。
        feature_type: "continuous" 或 "categorical"。

    Returns:
        dict with keys: missing_rate, missing_rate_delta, outlier_rate, outlier_rate_delta,
        default_value_rate, default_value_rate_delta, range_violation_rate,
        unknown_category_rate, unique_count, dq_score, dq_flag
    """
    missing = float(current.isna().mean())
    missing_delta = missing - float(baseline_row["missing_rate"])

    numeric = pd.to_numeric(current, errors="coerce")
    median = float(baseline_row["median"])
    mad = float(baseline_row["mad"])
    mad = mad if mad > 1e-12 else max(1e-12, float(baseline_row["q3"]) - float(baseline_row["q1"])) / 1.349

    outlier = (
        float(((numeric - median).abs() > 3.0 * mad).fillna(False).mean())
        if feature_type == "continuous"
        else 0.0
    )

    low = baseline_row.get("allowed_min")
    high = baseline_row.get("allowed_max")
    violations = pd.Series(False, index=current.index)
    if pd.notna(low):
        violations |= numeric < float(low)
    if pd.notna(high):
        violations |= numeric > float(high)
    range_rate = float(violations.fillna(False).mean())

    categories_raw = baseline_row.get("categories")
    if categories_raw is not None:
        try:
            cats = [str(v) for v in json.loads(str(categories_raw))]
        except (json.JSONDecodeError, TypeError):
            cats = []
    else:
        cats = []
    unknown = float((~current.astype("string").isin(cats) & current.notna()).mean()) if cats else 0.0

    baseline_outlier = float(baseline_row.get("outlier_rate", 0.0))
    outlier_delta = outlier - baseline_outlier

    default_value_rate = float(numeric.eq(0).fillna(False).mean())
    baseline_default = float(baseline_row.get("default_value_rate", 0.0))
    default_delta = default_value_rate - baseline_default

    # 综合质量评分
    penalty = min(
        1.0,
        abs(missing_delta) * 5
        + abs(outlier_delta) * 2
        + range_rate * 5
        + unknown * 5
        + max(0.0, default_delta) * 3,
    )
    score = float(max(0.0, 1.0 - penalty))
    flag = "ALERT" if score < 0.7 else "WARN" if score < 0.9 else "OK"

    return {
        "missing_rate": missing,
        "missing_rate_delta": missing_delta,
        "outlier_rate": outlier,
        "outlier_rate_delta": outlier_delta,
        "default_value_rate": default_value_rate,
        "default_value_rate_delta": default_delta,
        "range_violation_rate": range_rate,
        "unknown_category_rate": unknown,
        "unique_count": int(current.nunique(dropna=True)),
        "dq_score": score,
        "dq_flag": flag,
    }


# ═══════════════════════════════════════════════════════════════
# 性能指标（AUC / KS / PR-AUC / Brier / ECE / Bad Recall）
# ═══════════════════════════════════════════════════════════════


def expected_calibration_error(y_true: pd.Series, scores: pd.Series, bins: int = 10) -> float:
    """期望校准误差（ECE）——概率校准度。"""
    edges = np.linspace(0, 1, bins + 1)
    bucket = np.clip(np.digitize(scores, edges) - 1, 0, bins - 1)
    ece = 0.0
    for i in range(bins):
        mask = bucket == i
        if mask.any():
            ece += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(scores[mask].mean()))
    return float(ece)


def compute_performance_metrics(
    y_true: pd.Series,
    scores: pd.Series,
    bad_recall_cutoff: float = 0.20,
) -> dict[str, float | None]:
    """计算全部 6 个性能指标。

    Args:
        y_true: 真实标签（0/1）。
        scores: 模型预测概率值。
        bad_recall_cutoff: 坏样本召回率的分位切点（默认 Top 20%）。

    Returns:
        dict with keys: auc, ks, pr_auc, brier, ece, bad_recall。
        如果数据不足返回全部 None。
    """
    valid = pd.DataFrame({"y": y_true, "score": scores}).dropna()
    if valid.empty or valid["y"].nunique() < 2:
        return {k: None for k in ("auc", "ks", "pr_auc", "brier", "ece", "bad_recall")}

    y = valid["y"].astype(int)
    score = valid["score"].astype(float)

    fpr, tpr, _ = roc_curve(y, score)
    cutoff = max(1, int(np.ceil(len(valid) * bad_recall_cutoff)))
    top = valid.nlargest(cutoff, "score")
    bad_recall = float(top["y"].sum() / max(1, valid["y"].sum()))

    return {
        "auc": float(roc_auc_score(y, score)),
        "ks": float(np.max(tpr - fpr)),
        "pr_auc": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, score)),
        "ece": expected_calibration_error(y, score),
        "bad_recall": bad_recall,
    }


# ═══════════════════════════════════════════════════════════════
# 综合漂移汇总（便捷函数）
# ═══════════════════════════════════════════════════════════════


def compute_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    binning_rules: dict[str, dict],
    feature_names: list[str],
) -> tuple[list[dict], list[dict]]:
    """对全部特征批量计算漂移 + 数据质量，并做 BH 校正。

    Args:
        reference: W0 参照数据（含特征列）。
        current: 当前窗口数据（含特征列）。
        binning_rules: 从基线包加载的分箱规则，格式: {feature_name: {feature_type, edges, categories}}。
        feature_names: 要计算的特征名列表（不含 __risk_score__）。

    Returns:
        (quality_rows, drift_rows): 每个特征一条记录。
    """
    quality_rows: list[dict] = []
    drift_rows: list[dict] = []
    p_positions: list[int] = []
    p_values: list[float | None] = []

    for feature in feature_names:
        rule = binning_rules.get(feature)
        if rule is None or feature not in current or feature not in reference:
            continue

        feat_type = rule["feature_type"]

        # 数据质量
        quality = feature_quality(
            current[feature],
            pd.Series(rule.get("baseline_profile", {})),
            feat_type,
        )
        quality["feature_name"] = feature
        quality_rows.append(quality)

        # 漂移
        if feat_type == "categorical":
            drift = categorical_drift(reference[feature], current[feature], rule["categories"])
            row = {
                "feature_name": feature,
                "feature_type": "categorical",
                **drift,
                "wasserstein_distance": None,
                "ks_statistic": None,
                "ks_p_value": None,
                "ks_q_value": None,
            }
        else:
            drift = continuous_drift(reference[feature], current[feature], rule["edges"])
            row = {
                "feature_name": feature,
                "feature_type": "continuous",
                **drift,
                "category_share_change": None,
                "unknown_category_rate": 0.0,
                "ks_q_value": None,
            }
            p_positions.append(len(drift_rows))
            p_values.append(row["ks_p_value"])

        drift_rows.append(row)

    # BH 校正
    for pos, q_val in zip(p_positions, benjamini_hochberg(p_values)):
        drift_rows[pos]["ks_q_value"] = q_val

    return quality_rows, drift_rows
