"""Validate formal numeric identities for the former credit_test assets."""

import hashlib
import json
from pathlib import Path

import joblib
import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "model_id,legacy_model_id,algorithm_family",
    [
        ("credit_test_logistic", "credit_test_logistic", "LogisticRegression"),
        ("credit_test_lightgbm", "credit_test_lightgbm", "LightGBM"),
    ],
)
def test_formal_numeric_test_asset_identity_and_checksums(
    model_id, legacy_model_id, algorithm_family
):
    bundle = ROOT / "assets/test_models/a1_a7" / model_id / "test_v1"
    manifest = json.loads((bundle / "training_manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((bundle / "feature_schema.json").read_text(encoding="utf-8"))
    evaluation = json.loads((bundle / "a7_evaluation.json").read_text(encoding="utf-8"))
    registration = json.loads(
        (bundle / "formal_action_registration.json").read_text(encoding="utf-8")
    )

    assert manifest["model_id"] == schema["model_id"] == model_id
    assert manifest["legacy_model_id"] == legacy_model_id
    assert manifest["algorithm_family"] == algorithm_family
    assert manifest["scope"] == "TEST_ONLY_NOT_CHAMPION_NOT_PRODUCTION"
    assert manifest["w4_read_count"] == 0
    assert evaluation["selected"]["passed"] is False
    assert registration["model_id"] == model_id
    if model_id == "credit_test_logistic":
        assert registration["primary_repair_action"] == "A5_CALIBRATION_ADJUSTMENT"
        assert registration["a5_algorithm_fixture_passed"] is True
        assert registration["a6_algorithm_fixture_passed"] is True
    else:
        assert registration["primary_repair_action"] == "A4_PIPELINE_REPAIR"
        assert registration["qualification"]["qualified"] is True

    for filename, checksum_key in (
        ("model.joblib", "model"),
        ("a7_repair_challenger.joblib", "a7_repair_challenger"),
    ):
        path = bundle / filename
        actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == manifest["checksums"][checksum_key]
        assert callable(getattr(joblib.load(path), "predict_proba", None))
