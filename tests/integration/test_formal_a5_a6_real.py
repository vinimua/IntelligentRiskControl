"""Formal real-data Worker tests for A5 and A6.

No metric, score or label is mocked.  Only the callback HTTP transport is
captured because this test invokes the Celery task function in-process.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from apps.modelops_api.services.iteration.action_execution_service import (
    qualify_adjustment,
)
from workers.executor_tasks import calibrate, search_threshold


ROOT = Path(__file__).resolve().parents[2]


def test_a5_real_worker_healthy_trigger_repair_and_qualification(tmp_path):
    bundle = (
        ROOT
        / "assets/test_models/formal_a1_a7/credit_formal_logistic_a5/test_v1"
    )
    scenario = ROOT / "artifacts/formal_a1_a7/a5_conditional_formal_model/slope_0.99"
    report = json.loads((scenario / "report.json").read_text(encoding="utf-8"))
    assert report["trigger"]["brier_warning"] is True
    assert report["trigger"]["ece_warning"] is True
    assert report["trigger"]["auc_stable"] is True
    assert report["trigger"]["ks_stable"] is True

    plan = {
        "calibration_plan_id": "formal-a5-worker-integration",
        "model_id": "credit_formal_logistic_a5",
        "champion_version": "test_v1",
        "champion_bundle_uri": str(bundle),
        "fit_snapshot_id": "W2_CONDITIONAL_0.99",
        "fit_snapshot_uri": str(scenario / "W2_real_rows.parquet"),
        "validation_snapshot_id": "W3_CONDITIONAL_0.99",
        "validation_snapshot_uri": str(scenario / "W3_real_rows.parquet"),
        "healthy_snapshot_id": "W1",
        "healthy_snapshot_uri": str(ROOT / "assets/data/windows/W1/data.parquet"),
        "calibrator_type": "isotonic",
        "artifact_output_path": str(tmp_path / "calibrator.joblib"),
    }
    callbacks = []
    with patch("workers.executor_tasks._api_post", side_effect=lambda path, body: callbacks.append((path, body)) or {}):
        result = calibrate.run(plan)

    qualified, reasons = qualify_adjustment("CALIBRATION_ADJUSTMENT", result)
    assert qualified is True, reasons
    assert result["consumption_receipt"]["w4_read_count"] == 0
    assert result["consumption_receipt"]["sample_overlap_count"] == 0
    assert result["metrics"]["brier_improvement"] > 0
    assert result["metrics"]["ece_improvement"] > 0
    assert callbacks[0][1]["artifact_checksum"] == result["artifact_checksum"]


def test_a6_real_worker_requires_approval_and_improves_f1(tmp_path):
    plan = {
        "threshold_plan_id": "formal-a6-worker-integration",
        "model_id": "credit_model_001",
        "champion_version": "champion_v1",
        "fit_snapshot_id": "W2",
        "fit_snapshot_uri": str(ROOT / "assets/data/windows/W2/data.parquet"),
        "validation_snapshot_id": "W3",
        "validation_snapshot_uri": str(ROOT / "assets/data/windows/W3/data.parquet"),
        "business_objective_changed": True,
        "authorization_id": "human-confirmed-formal-test",
        "search_metric": "F1",
        "search_range": {"min": 0.01, "max": 0.99, "step": 0.01},
        "artifact_output_path": str(tmp_path / "threshold.json"),
    }
    callbacks = []
    with patch("workers.executor_tasks._api_post", side_effect=lambda path, body: callbacks.append((path, body)) or {}):
        result = search_threshold.run(plan)

    qualified, reasons = qualify_adjustment("THRESHOLD_ADJUSTMENT", result)
    assert qualified is True, reasons
    assert result["metrics"]["f1_improvement"] > 0
    assert result["consumption_receipt"]["w4_read_count"] == 0
    assert result["consumption_receipt"]["sample_overlap_count"] == 0
    assert callbacks[0][1]["artifact_checksum"] == result["artifact_checksum"]
