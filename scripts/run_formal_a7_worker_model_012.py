"""Run an existing Champion through the production A7 training Worker contract.

The immutable scenario snapshots are produced by
``run_a7_repair_model_012_fixture.py``. This runner does not reuse that
fixture's trained model: it sends the snapshots through ``train_model`` and
captures only the callback transport locally.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workers.training_tasks import train_model  # noqa: E402


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main(model_id: str = "credit_model_012") -> None:
    suffix = model_id.removeprefix("credit_model_")
    source_report = ROOT / f"artifacts/a7_repair_model_{suffix}/latest_report.json"
    training_snapshot = ROOT / f"artifacts/a7_repair_model_{suffix}/snapshots/training_snapshot.parquet"
    validation_snapshot = ROOT / f"artifacts/a7_repair_model_{suffix}/snapshots/validation_snapshot.parquet"
    output = ROOT / f"artifacts/formal_a1_a7/a7_worker_model_{suffix}"
    for path in (source_report, training_snapshot, validation_snapshot):
        if not path.is_file():
            raise FileNotFoundError(
                f"A7_SCENARIO_INPUT_MISSING:{path}; run scripts/run_a7_repair_model_012_fixture.py first"
            )
    source = json.loads(source_report.read_text(encoding="utf-8"))
    plan = source["plan"]
    windows = plan["windows"]
    training_checksum = _sha256(training_snapshot)
    validation_checksum = _sha256(validation_snapshot)
    snapshot_ids = ["W2", "W3_TRAIN_SPLIT", "W3_VALIDATION_SPLIT"]
    snapshot_uris = {
        "W2": str(training_snapshot),
        "W3_TRAIN_SPLIT": str(training_snapshot),
        "W3_VALIDATION_SPLIT": str(validation_snapshot),
    }
    snapshot_checksums = {
        "W2": training_checksum,
        "W3_TRAIN_SPLIT": training_checksum,
        "W3_VALIDATION_SPLIT": validation_checksum,
    }
    job = {
        "training_job_id": f"formal-a7-worker-model-{suffix}",
        "idempotency_key": f"formal-a7-worker-model-{suffix}:round-1",
        "model_id": model_id,
        "lifecycle_run_id": f"formal-a7-model-{suffix}",
        "iteration_run_id": plan["iteration_run_id"],
        "training_plan_id": plan["training_plan_id"],
        "experiment_id": f"formal-a7-worker-model-{suffix}-experiment",
        "business_round": 1,
        "strategy_code": plan["strategy_code"],
        "execution_mode": plan["execution_mode"],
        "training_data_mode": plan["training_data_mode"],
        "training_window_ids": windows["training_window_ids"],
        "validation_window_ids": windows["validation_window_ids"],
        "train_time_ranges": windows["training_time_ranges"],
        "validation_time_ranges": windows["validation_time_ranges"],
        "oot_window_id": "W4",
        "data_snapshot_ids": snapshot_ids,
        "data_snapshot_checksums": snapshot_checksums,
        "data_snapshot_uris": snapshot_uris,
        "label_versions": plan["label_versions"],
        "sample_weight_policy": plan["sample_weight_policy"],
        "sample_weight_required": plan["sample_weight_required"],
        "affected_segments": plan.get("affected_segments") or [],
        "change_point": plan.get("change_point"),
        "feature_schema_version": plan["feature_schema_version"],
        "ordered_features": plan["ordered_features"],
        "ordered_features_hash": plan["ordered_features_hash"],
        "preprocessing_version": plan["preprocessing_version"],
        "preprocessing_hash": plan["preprocessing_hash"],
        "algorithm": plan["algorithm"],
        "algorithm_family": plan["algorithm_family"],
        "champion_artifact_checksum": plan["champion_artifact_checksum"],
        "hyperparameters": plan["hyperparameter_space"],
        "target_metrics": plan["target_metric_codes"],
        "qualification_rule_version": plan["qualification_rule_version"],
        "base_model_version": plan["frozen_champion_version"],
        "seed": plan["random_seed"],
        "artifact_output_uri": f"s3://riskitem/challengers/formal-a7-model-{suffix}/",
        "training_mode": "full",
    }
    callbacks = []
    with patch(
        "workers.training_tasks._api_post",
        side_effect=lambda path, body: callbacks.append((path, body)) or {},
    ):
        worker_result = train_model.run(job)
    if not callbacks:
        raise RuntimeError("A7_WORKER_CALLBACK_MISSING")
    callback = callbacks[-1][1]
    metrics = callback.get("validation_metrics") or {}
    passed = bool(
        worker_result.get("status") == "SUCCEEDED"
        and callback.get("status") == "SUCCEEDED"
        and metrics.get("pre_oot_qualified") is True
        and (callback.get("consumption_receipt") or {}).get("w4_read_count") == 0
        and (callback.get("consumption_receipt") or {}).get("sample_overlap_count") == 0
    )
    report = {
        "formal_worker_passed": passed,
        "job_input": job,
        "worker_result": worker_result,
        "callback": callback,
        "transport_mocked_only": True,
        "models_scores_labels_mocked": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "latest_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({"formal_worker_passed": passed, "worker_result": worker_result, "validation_metrics": metrics, "receipt": callback.get("consumption_receipt")}, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError("A7_FORMAL_WORKER_QUALIFICATION_FAILED")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="credit_model_012")
    args = parser.parse_args()
    main(args.model_id)
