"""Scenario observation mapping for KG relation evidence.

This keeps scenario-analysis output compatible with the formal KG path:
Alert -[:INDICATES]-> RootCause.
"""

from __future__ import annotations

from typing import Any

VALID_DIRECTIONS = {"SUPPORT", "AGAINST", "NEUTRAL"}
VALID_DATA_TRACKS = {"NATURAL", "SCENARIO"}

SCENARIO_TO_ROOT_CAUSE: dict[str, str] = {
    "bad_rate_shift": "label_distribution_shift",
    "concept_drift": "label_distribution_shift",
    "covariate_drift": "feature_drift",
    "customer_mix_shift": "population_shift",
    "feature_staleness": "data_pipeline_issue",
    "fraud_pattern_shift": "label_distribution_shift",
    "key_feature_failure": "feature_failure",
    "missing_rate_anomaly": "data_quality_issue",
    "multi_root_cause": "data_quality_issue",
    "numeric_scaling_anomaly": "data_pipeline_issue",
    "policy_selection_shift": "business_policy_change",
    "preprocessing_version_mismatch": "data_pipeline_issue",
}


def mapped_relation_for_scenario(alert_code: str, scenario_name: str) -> dict[str, str]:
    """Map a scenario observation to the formal Alert -> RootCause relation."""
    root_cause = SCENARIO_TO_ROOT_CAUSE.get(scenario_name)
    if not root_cause:
        raise ValueError(f"Unknown scenario for KG mapping: {scenario_name}")

    return {
        "mapped_relation_key": f"{alert_code}|INDICATES|{root_cause}",
        "mapped_source_entity": alert_code,
        "mapped_source_type": "Alert",
        "mapped_relation": "INDICATES",
        "mapped_target_entity": root_cause,
        "mapped_target_type": "RootCause",
    }


def enrich_scenario_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Add mapped_* fields while preserving the original scenario output."""
    required = ("target_entity", "scenario", "direction", "data_track")
    missing = [key for key in required if not observation.get(key)]
    if missing:
        raise ValueError(f"Missing required scenario observation fields: {missing}")

    direction = str(observation["direction"])
    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"Invalid evidence direction: {direction}")

    data_track = str(observation["data_track"])
    if data_track not in VALID_DATA_TRACKS:
        raise ValueError(f"Invalid data_track: {data_track}")

    mapped = mapped_relation_for_scenario(
        alert_code=str(observation["target_entity"]),
        scenario_name=str(observation["scenario"]),
    )
    return {**observation, **mapped}


def validate_mapped_observation(observation: dict[str, Any]) -> None:
    """Validate that mapped_* fields are internally consistent."""
    expected_key = (
        f"{observation.get('mapped_source_entity')}|"
        f"{observation.get('mapped_relation')}|"
        f"{observation.get('mapped_target_entity')}"
    )
    if observation.get("mapped_relation_key") != expected_key:
        raise ValueError("mapped_relation_key does not match mapped triplet")

    if observation.get("mapped_source_type") != "Alert":
        raise ValueError("mapped_source_type must be Alert")
    if observation.get("mapped_relation") != "INDICATES":
        raise ValueError("mapped_relation must be INDICATES")
    if observation.get("mapped_target_type") != "RootCause":
        raise ValueError("mapped_target_type must be RootCause")
