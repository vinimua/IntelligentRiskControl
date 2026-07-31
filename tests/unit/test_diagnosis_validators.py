from __future__ import annotations

import pytest

from apps.modelops_api.services.diagnosis.validators.drift_group_regression import (
    MIN_CORRELATION_WINDOWS,
    drift_group_regression,
)
from apps.modelops_api.services.diagnosis.validators.metric_binding import (
    resolve_alert_metric_code,
)
from apps.modelops_api.services.diagnosis.validators.temporal_precedence_check import (
    temporal_precedence_check,
)
from packages.models.common.enums import AvailabilityStatus, EvidenceDirection


def _drift(window_ids: list[str], psi_values: list[float]) -> dict[str, list[dict]]:
    return {
        window_id: [{"feature_name": "x", "psi": psi}]
        for window_id, psi in zip(window_ids, psi_values, strict=True)
    }


def _metric(metric_code: str, window_id: str, delta: float) -> dict:
    return {
        "metric_code": metric_code,
        "window_id": window_id,
        "delta": delta,
    }


def test_alert_metric_binding_is_strict() -> None:
    assert resolve_alert_metric_code("AUC_DROP") == "AUC"
    assert resolve_alert_metric_code("AUC_DROP_P50") == "AUC"
    assert resolve_alert_metric_code("KS_DROP") == "KS"
    assert resolve_alert_metric_code("PR_AUC_DROP_P50") == "PR_AUC"
    assert resolve_alert_metric_code("HIGH_FEATURE_PSI") is None


@pytest.mark.asyncio
async def test_regression_uses_only_the_alert_metric() -> None:
    windows = ["W1", "W2", "W3", "W4"]
    drift = _drift(windows, [0.01, 0.02, 0.03, 0.04])
    metrics = []
    for index, window_id in enumerate(windows):
        metrics.extend(
            [
                _metric("AUC", window_id, -0.01 * index),
                _metric("KS", window_id, 0.20 - 0.05 * index),
            ]
        )

    evidence = await drift_group_regression(
        [],
        "AUC_DROP",
        multi_window_drift=drift,
        metrics=metrics,
    )

    assert evidence.applicable is True
    assert evidence.direction == EvidenceDirection.SUPPORT
    assert evidence.evidence_detail_json["target_metric_code"] == "AUC"
    assert evidence.evidence_detail_json["per_window_delta"] == {
        "W1": 0.0,
        "W2": -0.01,
        "W3": -0.02,
        "W4": -0.03,
    }


@pytest.mark.asyncio
async def test_regression_with_too_few_windows_does_not_affect_ranking() -> None:
    windows = ["W1", "W2"]
    evidence = await drift_group_regression(
        [],
        "AUC_DROP",
        multi_window_drift=_drift(windows, [0.01, 0.02]),
        metrics=[
            _metric("AUC", "W1", -0.01),
            _metric("AUC", "W2", -0.02),
        ],
    )

    assert MIN_CORRELATION_WINDOWS == 4
    assert evidence.applicable is False
    assert evidence.normalized_score is None
    assert evidence.availability_status == AvailabilityStatus.SAMPLE_TOO_SMALL


@pytest.mark.asyncio
async def test_temporal_check_rejects_overlapping_horizons() -> None:
    windows = ["7D_20251225_20251231", "30D_20251202_20251231"]
    evidence = await temporal_precedence_check(
        [],
        "KS_DROP",
        multi_window_drift=_drift(windows, [0.01, 0.04]),
        metrics=[
            _metric("AUC", windows[0], -0.50),
            _metric("AUC", windows[1], -0.60),
            _metric("KS", windows[0], -0.01),
            _metric("KS", windows[1], -0.02),
        ],
    )

    assert evidence.applicable is False
    assert evidence.availability_status == AvailabilityStatus.NOT_APPLICABLE
    assert evidence.evidence_detail_json["target_metric_code"] == "KS"
    assert evidence.evidence_detail_json["per_window_delta"] == {
        windows[0]: -0.01,
        windows[1]: -0.02,
    }
    assert "时间重叠" in evidence.evidence_detail_json["message"]


@pytest.mark.asyncio
async def test_temporal_check_supports_ordered_legacy_windows() -> None:
    windows = ["W1", "W2", "W3"]
    evidence = await temporal_precedence_check(
        [],
        "AUC_DROP",
        multi_window_drift=_drift(windows, [0.30, 0.10, 0.05]),
        metrics=[
            _metric("AUC", "W1", 0.0),
            _metric("AUC", "W2", -0.01),
            _metric("AUC", "W3", -0.05),
        ],
    )

    assert evidence.applicable is True
    assert evidence.direction == EvidenceDirection.SUPPORT
    assert evidence.evidence_detail_json["peak_psi_window"] == "W1"
    assert evidence.evidence_detail_json["peak_degradation_window"] == "W3"
