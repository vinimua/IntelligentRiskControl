"""场景质量验收 — 基于交接包 scenario_acceptance.py。

注入后的场景必须通过验收才能作为 Sentinel 训练数据：
- 异常信号确实可被监控指标捕获（与对照组配对比较）
- 信号强度在合理范围内（不能太微弱、不能过度应力）
- CLEAN_CONTROL 必须确认为正常

输出三态标签：ACCEPTED_NORMAL(0) / ACCEPTED_ANOMALY(1) / UNCERTAIN(None)
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════


def _maximum(frame: pd.DataFrame, column: str, *, absolute: bool = False) -> float:
    """DataFrame 中某列的最大值（可选绝对值）。"""
    if frame.empty or column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return 0.0
    if absolute:
        values = values.abs()
    return float(values.max())


def _maximum_share_shift(frame: pd.DataFrame) -> float:
    """类别占比变化的最大绝对值。"""
    if frame.empty or "category_share_change" not in frame:
        return 0.0
    maximum = 0.0
    for value in frame["category_share_change"].dropna():
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                continue
        if isinstance(value, dict) and value:
            maximum = max(maximum, max(abs(float(v)) for v in value.values()))
    return maximum


def _paired_adverse(
    performance: pd.DataFrame,
    control: pd.DataFrame,
    fallback: dict[str, Any],
) -> dict[str, float]:
    """与未修改的对照组配对比较，度量注入造成的性能退化。"""
    result: dict[str, float] = {}
    candidate = performance.copy()
    reference = control.copy()
    if "window_days" in candidate and "window_days" in reference:
        candidate = candidate.set_index("window_days", drop=False)
        reference = reference.set_index("window_days", drop=False)

    for metric in ("auc", "ks", "pr_auc", "bad_recall", "brier", "ece"):
        changes: list[float] = []
        if metric in candidate and metric in reference:
            for key in candidate.index.intersection(reference.index):
                cur = pd.to_numeric(pd.Series([candidate.loc[key, metric]]), errors="coerce").iloc[0]
                nor = pd.to_numeric(pd.Series([reference.loc[key, metric]]), errors="coerce").iloc[0]
                if pd.notna(cur) and pd.notna(nor):
                    changes.append(
                        float(nor - cur) if metric in {"auc", "ks", "pr_auc", "bad_recall"}
                        else float(cur - nor)
                    )
        if not changes:
            vals = (
                pd.to_numeric(performance.get(metric), errors="coerce").dropna()
                if metric in performance
                else pd.Series(dtype=float)
            )
            normal = fallback.get(metric)
            if normal is not None and not vals.empty:
                changes = (
                    [float(normal) - float(vals.min())]
                    if metric in {"auc", "ks", "pr_auc", "bad_recall"}
                    else [float(vals.max()) - float(normal)]
                )
        suffix = "drop" if metric in {"auc", "ks", "pr_auc", "bad_recall"} else "increase"
        result[f"{metric}_{suffix}"] = max(0.0, max(changes, default=0.0))
    return result


# ═══════════════════════════════════════════════════════════════
# 验收入口
# ═══════════════════════════════════════════════════════════════


def evaluate_scenario_acceptance(
    performance: pd.DataFrame,
    quality: pd.DataFrame,
    drift: pd.DataFrame,
    *,
    control_performance: pd.DataFrame,
    control_quality: pd.DataFrame,
    control_drift: pd.DataFrame,
    baseline_performance: dict[str, Any],
    scenario_name: str,
    anomaly_scope: str,
    thresholds: dict[str, float],
    scenario_category: str = "OPERATIONAL_ANOMALY",
    drift_type: str = "UNKNOWN",
) -> dict[str, Any]:
    """评估注入场景是否通过验收。

    通过配对比较（注入副本 vs 未修改副本 + 基线基准）判断：
    - 异常信号是否可检测（超过告警阈值）
    - 信号是否在合理范围（未过度应力）
    - CLEAN_CONTROL 是否确认正常

    Args:
        performance: 注入场景的性能监控 DataFrame。
        quality: 注入场景的数据质量 DataFrame。
        drift: 注入场景的漂移 DataFrame。
        control_performance: 同一时间窗口未注入的对照组性能。
        control_quality: 对照组数据质量。
        control_drift: 对照组漂移。
        baseline_performance: W0 基线性能基准（含 auc/ks/pr_auc/bad_recall/brier/ece）。
        scenario_name: 场景名称（如 "clean_control" / "covariate_drift"）。
        anomaly_scope: 预期影响范围（DATA_QUALITY/DISTRIBUTION/MODEL_PERFORMANCE/MULTI）。
        thresholds: 验收阈值字典（psi_alert_min / auc_drop_alert / missing_delta_alert 等）。
        scenario_category: 场景类别（BUSINESS_DRIFT/OPERATIONAL_ANOMALY/COMPOUND/CONTROL）。
        drift_type: 漂移类型。

    Returns:
        {
            "scenario_acceptance_status": "ACCEPTED_NORMAL" | "ACCEPTED_ANOMALY" | "UNCERTAIN",
            "anomaly_label": 0 | 1 | None,
            "achieved_effects": {详细的效应指标},
        }
    """
    # ── 注入效应度量 ──
    max_feature_psi = _maximum(drift, "psi")
    control_feature_psi = _maximum(control_drift, "psi")
    max_feature_js = _maximum(drift, "js_divergence")
    max_feature_ks = _maximum(drift, "ks_statistic")
    max_segment_share_delta = _maximum_share_shift(drift)
    score_psi = _maximum(performance, "prediction_psi")
    control_score_psi = _maximum(control_performance, "prediction_psi")

    missing_delta = _maximum(quality, "missing_rate_delta", absolute=True)
    control_missing_delta = _maximum(control_quality, "missing_rate_delta", absolute=True)
    range_rate = _maximum(quality, "range_violation_rate")
    control_range_rate = _maximum(control_quality, "range_violation_rate")
    unknown_rate = _maximum(quality, "unknown_category_rate")
    control_unknown_rate = _maximum(control_quality, "unknown_category_rate")
    default_rate_delta = _maximum(quality, "default_value_rate_delta", absolute=True)
    control_default_rate_delta = _maximum(control_quality, "default_value_rate_delta", absolute=True)

    injected_missing_delta = max(0.0, missing_delta - control_missing_delta)
    injected_range_rate = max(0.0, range_rate - control_range_rate)
    injected_unknown_rate = max(0.0, unknown_rate - control_unknown_rate)
    injected_default_rate = max(0.0, default_rate_delta - control_default_rate_delta)

    adverse = _paired_adverse(performance, control_performance, baseline_performance)

    bad_rate = _maximum(performance, "bad_rate")
    control_bad_rate = _maximum(control_performance, "bad_rate")
    bad_rate_delta = abs(bad_rate - control_bad_rate)

    # ── 阈值判定 ──
    psi_peak = max(max_feature_psi, score_psi)
    distribution_hit = (
        psi_peak >= float(thresholds["psi_alert_min"])
        or max_feature_ks >= float(thresholds.get("ks_effect_alert", 1.0))
        or max_segment_share_delta >= float(thresholds.get("segment_share_delta_alert", 1.0))
    )
    distribution_warning = (
        psi_peak >= float(thresholds["psi_warning_min"])
        or max_feature_ks >= float(thresholds.get("ks_effect_alert", 1.0)) / 2
    )
    distribution_stress = psi_peak > float(thresholds["psi_stress_max"])

    data_quality_hit = (
        injected_missing_delta >= float(thresholds["missing_delta_alert"])
        or injected_default_rate >= float(thresholds.get("default_value_delta_alert", 1.0))
        or injected_range_rate >= float(thresholds["range_rate_alert"])
        or injected_unknown_rate >= float(thresholds["unknown_rate_alert"])
    )

    performance_hit = (
        adverse["auc_drop"] >= float(thresholds["auc_drop_alert"])
        or adverse["ks_drop"] >= float(thresholds["ks_drop_alert"])
        or adverse["pr_auc_drop"] >= float(thresholds["pr_auc_drop_alert"])
        or adverse["bad_recall_drop"] >= float(thresholds["bad_recall_drop_alert"])
        or adverse["brier_increase"] >= float(thresholds["brier_increase_alert"])
        or adverse["ece_increase"] >= float(thresholds["ece_increase_alert"])
    )

    domains = {
        "DATA_QUALITY": data_quality_hit,
        "DISTRIBUTION": distribution_hit,
        "MODEL_PERFORMANCE": performance_hit,
    }
    domain_hit_count = sum(domains.values())
    prior_probability_hit = bad_rate_delta >= float(thresholds.get("bad_rate_delta_alert", 1.0))

    # ── 三态判定 ──
    if scenario_name == "clean_control":
        normal = (
            not data_quality_hit
            and not performance_hit
            and psi_peak < float(thresholds["psi_warning_min"])
        )
        status, label = ("ACCEPTED_NORMAL", 0) if normal else ("UNCERTAIN", None)
    else:
        if anomaly_scope == "DATA_QUALITY":
            accepted = data_quality_hit
        elif anomaly_scope == "DISTRIBUTION":
            accepted = distribution_hit and not distribution_stress
        elif anomaly_scope == "MODEL_PERFORMANCE":
            accepted = performance_hit or (
                drift_type == "PRIOR_PROBABILITY" and prior_probability_hit
            )
        elif anomaly_scope == "MULTI":
            accepted = domain_hit_count >= 2 and (
                scenario_category == "OPERATIONAL_ANOMALY" or not distribution_stress
            )
        else:
            accepted = False
        status, label = ("ACCEPTED_ANOMALY", 1) if accepted else ("UNCERTAIN", None)

    # ── 效应汇总 ──
    effects = {
        "max_feature_psi": max_feature_psi,
        "control_max_feature_psi": control_feature_psi,
        "injected_feature_psi_increment": max(0.0, max_feature_psi - control_feature_psi),
        "max_feature_js": max_feature_js,
        "max_feature_ks_statistic": max_feature_ks,
        "max_segment_share_delta": max_segment_share_delta,
        "score_psi": score_psi,
        "control_score_psi": control_score_psi,
        "injected_score_psi_increment": max(0.0, score_psi - control_score_psi),
        "missing_rate_delta_max_abs": missing_delta,
        "injected_missing_rate_delta": injected_missing_delta,
        "range_violation_rate_max": range_rate,
        "injected_range_violation_rate": injected_range_rate,
        "unknown_category_rate_max": unknown_rate,
        "injected_unknown_category_rate": injected_unknown_rate,
        "default_value_rate_delta_max_abs": default_rate_delta,
        "injected_default_value_rate_delta": injected_default_rate,
        "bad_rate": bad_rate,
        "control_bad_rate": control_bad_rate,
        "bad_rate_delta": bad_rate_delta,
        "prior_probability_hit": prior_probability_hit,
        **adverse,
        "distribution_warning": distribution_warning,
        "distribution_hit": distribution_hit,
        "distribution_stress": distribution_stress,
        "data_quality_hit": data_quality_hit,
        "performance_hit": performance_hit,
        "evidence_domain_count": domain_hit_count,
        "data_quality_passed": not data_quality_hit,
        "scenario_category": scenario_category,
        "drift_type": drift_type,
        "comparison_mode": "PAIRED_UNMODIFIED_WINDOW_COPY",
    }

    return {
        "scenario_acceptance_status": status,
        "anomaly_label": label,
        "achieved_effects": effects,
    }
