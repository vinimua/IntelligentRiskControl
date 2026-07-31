import pytest

from apps.modelops_api.services.knowledge_observation_mapper import (
    enrich_scenario_observation,
    validate_mapped_observation,
)


def test_enrich_scenario_observation_maps_to_alert_root_cause_relation():
    raw = {
        "source_entity": "loan_amount_request",
        "source_type": "Feature",
        "relation": "MAY_CAUSE",
        "target_entity": "AUC_DROP",
        "target_type": "Alert",
        "direction": "AGAINST",
        "evidence_score": 0.1,
        "data_track": "SCENARIO",
        "scenario": "covariate_drift",
        "model_id": "credit_model_045",
        "auc_drop": 0.0,
        "feature_psi": 0.2246,
    }

    enriched = enrich_scenario_observation(raw)

    assert enriched["mapped_relation_key"] == "AUC_DROP|INDICATES|feature_drift"
    assert enriched["mapped_source_entity"] == "AUC_DROP"
    assert enriched["mapped_source_type"] == "Alert"
    assert enriched["mapped_relation"] == "INDICATES"
    assert enriched["mapped_target_entity"] == "feature_drift"
    assert enriched["mapped_target_type"] == "RootCause"
    assert enriched["source_entity"] == "loan_amount_request"

    validate_mapped_observation(enriched)


def test_enrich_scenario_observation_rejects_unknown_scenario():
    with pytest.raises(ValueError, match="Unknown scenario"):
        enrich_scenario_observation({
            "target_entity": "AUC_DROP",
            "direction": "SUPPORT",
            "data_track": "SCENARIO",
            "scenario": "unknown_scenario",
        })
