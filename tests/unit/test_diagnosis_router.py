from __future__ import annotations

from apps.modelops_api.routers.diagnosis import (
    _build_alert_context,
    _diagnosis_source,
)
from apps.modelops_api.services.diagnosis.diagnosis_service import DiagnosisService


def _run(alert_context_json: dict) -> dict:
    return {
        "monitoring_run_id": "539e6e72-3a9f-4f31-ae23-35edfd212021",
        "model_id": "credit_model_001",
        "champion_version": "champion_v1",
        "baseline_window_id": "W1",
        "current_window_id": "W3",
        "data_track": "NATURAL",
        "trace_id": "trace-from-run",
        "alert_context_json": alert_context_json,
    }


def test_build_alert_context_preserves_canonical_contract() -> None:
    raw = {
        "schema_version": "V2-WP08",
        "trace_id": "canonical-trace",
        "monitoring_run_id": "run-1",
        "model_id": "credit_model_001",
        "model_version": "champion_v1",
        "monitor_window_id": "W3",
        "baseline_id": "BASELINE-1",
        "data_track": "NATURAL",
        "alert_details": [],
    }

    context = _build_alert_context(_run(raw), [])

    assert context.trace_id == "canonical-trace"
    assert context.monitor_window_id == "W3"
    assert context.alert_details == []


def test_build_alert_context_adapts_sentinel_summary_from_alert_rows() -> None:
    sentinel_summary = {
        "schema_version": "W1_FULL_V2",
        "trace_id": "sentinel-trace",
        "model_id": "credit_model_001",
        "model_version": "champion_v1",
        "baseline_id": "BASELINE-1",
        "classification": "WATCH",
        "monitoring_decisions": [{"state": "WATCHING_RECOVERY"}],
    }
    alerts = [
        {
            "alert_id": "5378a76e-dd65-4ea4-a0b4-02a72029491e",
            "alert_code": "FORMAL_DISCRIMINATION_AUC",
            "severity": "WARNING",
            "object_type": "MODEL",
            "object_code": "credit_model_001",
            "metric_code": "AUC",
            "metric_version": "W1_FULL_V2",
            "availability_status": "AVAILABLE",
            "alert_detail": {"window_id": "W3"},
        }
    ]

    context = _build_alert_context(_run(sentinel_summary), alerts)

    assert context.monitoring_run_id == _run({})["monitoring_run_id"]
    assert context.monitor_window_id == "W3"
    assert context.data_track.value == "NATURAL"
    assert len(context.alert_details) == 1
    assert context.alert_details[0].alert_code == "AUC_DROP"
    assert context.alert_details[0].metric_detail == {
        "window_id": "W3",
        "source_alert_code": "FORMAL_DISCRIMINATION_AUC",
    }


def test_diagnosis_source_exposes_model_specific_trigger() -> None:
    alerts = [
        {
            "alert_code": "FORMAL_DISCRIMINATION_KS",
            "delta": -0.0226,
        },
        {
            "alert_code": "FORMAL_DISCRIMINATION_KS",
            "delta": -0.0255,
        },
    ]

    source = _diagnosis_source(_run({}), alerts)

    assert source["model_id"] == "credit_model_001"
    assert source["diagnosis_alert_codes"] == ["KS_DROP"]
    assert source["alert_count"] == 2
    assert source["largest_drop"] == -0.0255


async def test_legacy_aggregate_metrics_never_impersonate_feature_drift(monkeypatch) -> None:
    metrics = [
        {
            "metric_code": "FEATURE_PSI",
            "current_value": 0.12,
            "metric_detail": {
                "window_id": "7D_20251224_20251231",
                "feature_name": "income_level",
            },
        },
        {
            "metric_code": "AUC",
            "delta": -0.02,
            "metric_detail": {"window_id": "7D_20251224_20251231"},
        },
    ]

    class FakeMonitoringRepo:
        def __init__(self, _session):
            pass

        async def get_feature_drift_by_run(self, _run_id):
            return []

        async def get_metrics(self, _run_id):
            return metrics

    monkeypatch.setattr(
        "apps.modelops_api.services.diagnosis.diagnosis_service.MonitoringRepo",
        FakeMonitoringRepo,
    )
    service = DiagnosisService(session=None, knowledge=None, repo=None)

    drift = await service._load_drift_data("run-1")
    normalized = await service._load_metrics("run-1")

    assert drift == []
    assert normalized[1]["window_id"] == "7D_20251224_20251231"
