"""Formal real-data Worker tests for A3 data repair and A4 pipeline repair."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from apps.modelops_api.services.iteration.action_execution_service import qualify_repair
from workers.executor_tasks import repair_and_replay


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "assets/champion_models/credit_model_001/champion_v1"
SCENARIOS = ROOT / "artifacts/formal_a1_a7/repair_scenarios"


def _run_worker(plan: dict):
    callbacks = []
    with patch(
        "workers.executor_tasks._api_post",
        side_effect=lambda path, body: callbacks.append((path, body)) or {},
    ):
        result = repair_and_replay.run(plan)
    qualified, reasons = qualify_repair(result)
    assert qualified is True, reasons
    assert callbacks[0][1]["artifact_checksum"] == result["artifact_checksum"]
    return result


def test_a3_real_missing_data_is_repaired_without_model_or_label_change(tmp_path):
    scenario = SCENARIOS / "a3_missing_age"
    result = _run_worker(
        {
            "repair_plan_id": "formal-a3-worker-integration",
            "action": "DATA_REPAIR",
            "model_id": "credit_model_001",
            "champion_version": "champion_v1",
            "champion_bundle_uri": str(BUNDLE),
            "source_snapshot_id": "W3_MISSING_AGE",
            "source_snapshot_uri": str(scenario / "W3_missing.parquet"),
            "reference_snapshot_id": "W2",
            "reference_snapshot_uri": str(ROOT / "assets/data/windows/W2/data.parquet"),
            "healthy_snapshot_id": "W1",
            "healthy_snapshot_uri": str(ROOT / "assets/data/windows/W1/data.parquet"),
            "affected_features": ["age"],
            "artifact_output_path": str(tmp_path / "a3_repaired.parquet"),
        }
    )
    assert result["metrics"]["missing_rate_before"] >= 0.10
    assert result["metrics"]["missing_rate_after"] == 0
    assert result["metrics"]["labels_unchanged"] is True
    assert result["consumption_receipt"]["w4_read_count"] == 0


def test_a4_real_pipeline_misalignment_recovers_frozen_champion(tmp_path):
    scenario = SCENARIOS / "a4_pipeline_column_misalignment"
    result = _run_worker(
        {
            "repair_plan_id": "formal-a4-worker-integration",
            "action": "PIPELINE_REPAIR",
            "model_id": "credit_model_001",
            "champion_version": "champion_v1",
            "champion_bundle_uri": str(BUNDLE),
            "source_snapshot_id": "W3_PIPELINE_CORRUPTED",
            "source_snapshot_uri": str(scenario / "W3_pipeline_corrupted.parquet"),
            "reference_snapshot_id": "W3_TRUSTED",
            "reference_snapshot_uri": str(scenario / "W3_trusted.parquet"),
            "healthy_snapshot_id": "W1",
            "healthy_snapshot_uri": str(ROOT / "assets/data/windows/W1/data.parquet"),
            "affected_features": ["login_fail_count"],
            "artifact_output_path": str(tmp_path / "a4_repaired.parquet"),
        }
    )
    metrics = result["metrics"]
    assert metrics["healthy_w1"]["auc"] - metrics["degraded"]["auc"] > 0.05
    assert metrics["auc_recovery_rate"] >= 0.90
    assert metrics["ks_recovery_rate"] >= 0.90
    assert metrics["labels_unchanged"] is True
    assert result["consumption_receipt"]["w4_read_count"] == 0


@pytest.mark.parametrize(
    "model_id,version,bundle",
    [
        (
            "credit_model_027",
            "champion_v1",
            ROOT / "assets/champion_models/credit_model_027/champion_v1",
        ),
        (
            "credit_test_lightgbm",
            "test_v1",
            ROOT / "assets/test_models/a1_a7/credit_test_lightgbm/test_v1",
        ),
    ],
)
def test_a4_lightgbm_family_uses_same_real_repair_contract(
    tmp_path, model_id, version, bundle
):
    scenario = SCENARIOS / "a4_pipeline_column_misalignment"
    result = _run_worker(
        {
            "repair_plan_id": "formal-a4-lightgbm-worker-integration",
            "action": "PIPELINE_REPAIR",
            "model_id": model_id,
            "champion_version": version,
            "champion_bundle_uri": str(bundle),
            "source_snapshot_id": "W3_PIPELINE_CORRUPTED",
            "source_snapshot_uri": str(scenario / "W3_pipeline_corrupted.parquet"),
            "reference_snapshot_id": "W3_TRUSTED",
            "reference_snapshot_uri": str(scenario / "W3_trusted.parquet"),
            "healthy_snapshot_id": "W1",
            "healthy_snapshot_uri": str(ROOT / "assets/data/windows/W1/data.parquet"),
            "affected_features": ["login_fail_count"],
            "artifact_output_path": str(tmp_path / "a4_lgbm_repaired.parquet"),
        }
    )
    assert result["metrics"]["auc_recovery_rate"] >= 0.90
    assert result["metrics"]["ks_recovery_rate"] >= 0.90
