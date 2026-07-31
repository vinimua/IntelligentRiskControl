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
