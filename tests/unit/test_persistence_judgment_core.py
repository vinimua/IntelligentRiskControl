"""B1 持续性判定：OUTLIER_RATE 退出核心 SEVERE 判定（任务三落地口径）。"""

import pytest

from apps.modelops_api.services.monitoring.persistence_judgment import (
    CORE_METRICS,
    PersistenceJudgmentService,
)


def _svc() -> PersistenceJudgmentService:
    return PersistenceJudgmentService(session=None)


def _window_alerts(
    metric_code: str, severity: str, n: int = 3, window_days: int = 7,
) -> list[dict]:
    return [
        {
            "metric_code": metric_code,
            "severity": severity,
            "window_id": f"2025-12-{i + 1:02d}",
            "window_days": window_days,
            "current_value": 0.1,
            "baseline_value": 0.0,
            "delta": 0.1,
        }
        for i in range(n)
    ]


def test_outlier_rate_not_in_core_metrics():
    """OUTLIER_RATE 是诊断证据，不是自动训练链路的刹车。"""
    assert "OUTLIER_RATE" not in CORE_METRICS
    assert "AUC" in CORE_METRICS
    assert "KS" in CORE_METRICS
    assert "FEATURE_PSI" in CORE_METRICS


def test_outlier_critical_with_sustained_feature_psi_is_not_severe():
    """OUTLIER_RATE CRITICAL + FEATURE_PSI WARNING 持续性 → 不得判 SEVERE。"""
    svc = _svc()
    alerts = (
        _window_alerts("OUTLIER_RATE", "CRITICAL", n=10)
        + _window_alerts("FEATURE_PSI", "WARNING", n=10)
    )
    counts_7d = svc._count_consecutive(alerts, window_days=7)
    counts_30d = svc._count_cumulative(alerts, window_days=30)
    is_severe, reason = svc._check_severe(counts_7d, counts_30d, alerts, [])
    assert is_severe is False, reason


def test_auc_ks_critical_both_windows_still_severe():
    """AUC/KS 双窗口 CRITICAL 仍然必须判 SEVERE（排序能力严重衰退）。"""
    svc = _svc()
    alerts = (
        _window_alerts("AUC", "CRITICAL", n=10, window_days=7)
        + _window_alerts("KS", "CRITICAL", n=10, window_days=7)
        + _window_alerts("AUC", "CRITICAL", n=10, window_days=30)
        + _window_alerts("KS", "CRITICAL", n=10, window_days=30)
    )
    counts_7d = svc._count_consecutive(alerts, window_days=7)
    counts_30d = svc._count_cumulative(alerts, window_days=30)
    is_severe, reason = svc._check_severe(counts_7d, counts_30d, alerts, [])
    assert is_severe is True
    assert "core_critical_both_windows" in reason


def test_smooth_drift_sustained_30d_combines_to_sustained():
    """平滑漂移（FEATURE_PSI WARNING 持续 30D）→ SUSTAINED_30D 组合判定，
    可进入自动迭代策略（不再被 OUTLIER 覆盖为 SEVERE）。"""
    svc = _svc()
    alerts = (
        _window_alerts("FEATURE_PSI", "WARNING", n=10, window_days=7)
        + _window_alerts("FEATURE_PSI", "WARNING", n=10, window_days=30)
    )
    counts_7d = svc._count_consecutive(alerts, window_days=7)
    counts_30d = svc._count_cumulative(alerts, window_days=30)
    status_7d = svc._compute_window_status(counts_7d, window_days=7)
    status_30d = svc._compute_window_status(counts_30d, window_days=30)
    decay_degree, trigger_diag = svc._combine_status(status_7d, status_30d)

    assert status_7d == "TRIGGERED"
    assert status_30d == "TRIGGERED"
    assert decay_degree == "SUSTAINED_30D"
    assert trigger_diag is True

