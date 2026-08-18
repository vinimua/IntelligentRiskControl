"""Alert-to-metric binding helpers shared by diagnosis validators."""

from __future__ import annotations


_ALERT_METRIC_PREFIXES: tuple[tuple[str, str], ...] = (
    ("PR_AUC_DROP", "PR_AUC"),
    ("AUC_DROP", "AUC"),
    ("KS_DROP", "KS"),
    ("RECALL_DROP", "RECALL"),
    ("PRECISION_DROP", "PRECISION"),
    ("F1_DROP", "F1"),
    ("BRIER_RISE", "BRIER"),
    ("BRIER_INCREASE", "BRIER"),
    ("ECE_RISE", "ECE"),
    ("ECE_INCREASE", "ECE"),
    ("BAD_RATE_DELTA", "BAD_RATE"),
)

_HIGHER_IS_BETTER = {
    "AUC",
    "PR_AUC",
    "KS",
    "RECALL",
    "PRECISION",
    "F1",
}
_LOWER_IS_BETTER = {"BRIER", "ECE"}
_DEVIATION_IS_BAD = {"BAD_RATE"}


def resolve_alert_metric_code(alert_code: str) -> str | None:
    """Return the one metric that the alert explicitly represents.

    No fallback to a different performance metric is allowed: using KS as
    evidence for an AUC alert (or vice versa) makes the evidence chain invalid.
    """

    normalized = (alert_code or "").upper()
    for prefix, metric_code in _ALERT_METRIC_PREFIXES:
        if normalized.startswith(prefix):
            return metric_code
    return None


def degradation_from_delta(metric_code: str, delta: float) -> float | None:
    """Convert a metric delta to positive-is-worse degradation severity."""

    code = (metric_code or "").upper()
    value = float(delta)
    if code in _HIGHER_IS_BETTER:
        return -value
    if code in _LOWER_IS_BETTER:
        return value
    if code in _DEVIATION_IS_BAD:
        return abs(value)
    return None


_RANKING_METRICS = {"AUC", "KS", "PR_AUC", "BAD_RECALL"}
_DEGRADATION_SIGNIFICANCE = 0.02  # 与 L1 WARNING 阈值一致


def has_ranking_degradation(metrics: list[dict] | None) -> bool:
    """是否存在需要解释的排序性能退化（≥0.02 的 AUC/KS/PR_AUC/BAD_RECALL 下降）。

    没有退化时"修复能否恢复性能 / 漂移特征是否重要"没有评估对象——
    反事实与重要性证据不构成对任何候选的 SUPPORT 或 AGAINST。
    """
    for m in metrics or []:
        code = str(m.get("metric_code", "")).upper()
        if code not in _RANKING_METRICS:
            continue
        if m.get("degraded"):
            return True
        try:
            delta = float(m.get("delta"))
        except (TypeError, ValueError):
            continue
        degradation = degradation_from_delta(code, delta)
        if degradation is not None and degradation >= _DEGRADATION_SIGNIFICANCE:
            return True
    return False


def resolve_metric_from_supporting_alerts(
    alert_code: str,
    supporting_alert_codes: list[str] | None = None,
) -> str | None:
    """绑定候选告警集里唯一可绑定的性能指标。

    候选主告警（如 HIGH_FEATURE_PSI）本身绑定不到性能指标时，
    从同候选的 supporting alert codes（AUC_DROP/KS_DROP/...）中找
    第一个可绑定的排序性能告警——漂移候选与性能告警同现时，
    时序/相关性验证以该性能指标为退化参照，不做 AUC/KS 混用。
    """
    bound = resolve_alert_metric_code(alert_code)
    if bound is not None:
        return bound
    for code in supporting_alert_codes or []:
        bound = resolve_alert_metric_code(code)
        if bound is not None:
            return bound
    return None
