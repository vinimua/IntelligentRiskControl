"""Validate the persisted production-Worker A7 run without retraining per test."""

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("model_suffix,algorithm_family", [("012", "RandomForest"), ("027", "LightGBM")])
def test_formal_worker_recovered_and_passed_pre_oot(model_suffix, algorithm_family):
    report_path = ROOT / f"artifacts/formal_a1_a7/a7_worker_model_{model_suffix}/latest_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    callback = report["callback"]
    metrics = callback["validation_metrics"]
    receipt = callback["consumption_receipt"]
    assert report["formal_worker_passed"] is True
    assert report["models_scores_labels_mocked"] is False
    assert callback["status"] == "SUCCEEDED"
    assert metrics["pre_oot_qualified"] is True
    assert metrics["recovery_auc"] >= 0.90
    assert metrics["recovery_ks"] >= 0.90
    assert metrics["auc_bootstrap_ci_lower"] > 0
    assert metrics["ks_bootstrap_ci_lower"] > 0
    assert receipt["actual_algorithm_family"] == algorithm_family
    assert receipt["sample_overlap_count"] == 0
    assert receipt["w4_read_count"] == 0


def test_credit_model_027_anomaly_detected_and_repaired_within_stress_ceiling():
    report_path = ROOT / "artifacts/a7_repair_model_027/latest_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    feature_psi = report["monitoring"]["feature_psi"]
    assert feature_psi["triggered"] is True
    assert feature_psi["severity"] == "CRITICAL"
    assert report["monitoring"]["ks"]["triggered"] is True
    assert report["strict_flow_audit"]["scenario_stress_valid"] is True
    assert report["pre_oot"]["status"] == "PASSED"
    assert report["repair"]["auc_bootstrap_ci"][0] > 0
    assert report["repair"]["ks_bootstrap_ci"][0] > 0
    assert report["w4_accessed"] is False
