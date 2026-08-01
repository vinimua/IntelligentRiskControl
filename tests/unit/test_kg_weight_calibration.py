from packages.models.common.enums import DataTrack, EvidenceDirection

from apps.modelops_api.services.knowledge_observation_service import (
    KnowledgeObservationService,
)
from scripts import apply_kg_weights_to_neo4j, run_kg_calibration


def test_lifecycle_observations_include_persisted_context_fields():
    observations = KnowledgeObservationService.build_observations(
        {
            "lifecycle_run_id": "11111111-1111-1111-1111-111111111111",
            "diagnosis_run_id": "diag-1",
            "primary_root_cause_code": "FEATURE_DRIFT",
            "primary_root_cause_score": 0.85,
            "decision_proposal_id": "proposal-1",
            "selected_strategy_code": "recent_weighted_retrain",
            "decision_reasons": [
                "KG_STRATEGY:recent_weighted_retrain",
                "SUPPORT_CASES:25",
            ],
            "deployment_id": "deploy-1",
            "deployment_decision": "PROMOTE",
            "challenger_qualified": True,
            "max_alert_severity": "HIGH",
        }
    )

    assert len(observations) == 4
    assert {obs.relation_key for obs in observations} == {
        "HIGH_FEATURE_PSI|INDICATES|FEATURE_DRIFT",
        "FEATURE_DRIFT|RECOMMENDS|recent_weighted_retrain",
        "recent_weighted_retrain|MITIGATES|FEATURE_DRIFT",
    }
    assert all(
        obs.lifecycle_run_id == "11111111-1111-1111-1111-111111111111"
        for obs in observations
    )
    assert all(obs.evidence_detail for obs in observations)
    assert observations[0].direction == EvidenceDirection.SUPPORT
    assert observations[0].data_track == DataTrack.NATURAL


def test_bayesian_shrinkage_replaces_weak_prior_rule():
    result = run_kg_calibration._bayesian_shrinkage(
        support_count=25,
        against_count=2,
        neutral_count=3,
        support_strength=21.0,
        against_strength=1.0,
    )

    assert result["new_weight"] > 0.65
    assert result["alpha_post"] == 23.0
    assert result["beta_post"] == 9.0
    assert result["confidence_lower"] < result["new_weight"] < result["confidence_upper"]


def test_calibration_cli_defaults_to_beta_v2(monkeypatch):
    captured = {}

    def fake_run_calibration(data_track: str, rule_version: str, weight_version: str) -> str:
        captured["data_track"] = data_track
        captured["rule_version"] = rule_version
        captured["weight_version"] = weight_version
        return "calibration-1"

    monkeypatch.setattr(run_kg_calibration, "run_calibration", fake_run_calibration)
    monkeypatch.setattr("sys.argv", ["run_kg_calibration.py"])

    run_kg_calibration.main()

    assert captured == {
        "data_track": "NATURAL",
        "rule_version": "BETA_BINOMIAL_V2",
        "weight_version": "KG_WEIGHT_BETA_V2",
    }


def test_apply_filters_supported_relation_templates():
    snapshots = [
        {
            "calibration_run_id": "11111111-1111-1111-1111-111111111111",
            "relation_key": "HIGH_FEATURE_PSI|INDICATES|FEATURE_DRIFT",
            "weight_version": "KG_WEIGHT_BETA_V2",
        },
        {
            "calibration_run_id": "11111111-1111-1111-1111-111111111111",
            "relation_key": "FEATURE_DRIFT|RECOMMENDS|recent_weighted_retrain",
            "weight_version": "KG_WEIGHT_BETA_V2",
        },
        {
            "calibration_run_id": "11111111-1111-1111-1111-111111111111",
            "relation_key": "unknown|IGNORED|target",
            "weight_version": "KG_WEIGHT_BETA_V2",
        },
    ]

    supported = apply_kg_weights_to_neo4j._supported_snapshots(snapshots)

    assert [item["relation_key"] for item in supported] == [
        "HIGH_FEATURE_PSI|INDICATES|FEATURE_DRIFT",
        "FEATURE_DRIFT|RECOMMENDS|recent_weighted_retrain",
    ]
    assert apply_kg_weights_to_neo4j._job_key(supported[1]) == (
        "11111111-1111-1111-1111-111111111111",
        "RECOMMENDS",
        "KG_WEIGHT_BETA_V2",
    )
