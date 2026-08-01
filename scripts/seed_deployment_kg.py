"""Seed deployment KG nodes and relations for Task 4.

The seeded path is:
DeploymentAlert -> DeploymentRisk -> DeploymentStrategy,
with DeploymentStage and DeploymentPolicy constraints.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from neo4j import AsyncGraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from apps.modelops_api.config import settings


DEPLOYMENT_ALERTS: dict[str, dict] = {
    "CHALLENGER_AUC_REGRESSION": {
        "name": "Challenger AUC regression",
        "metric_code": "challenger_auc",
        "severity": "HIGH",
    },
    "CHALLENGER_KS_REGRESSION": {
        "name": "Challenger KS regression",
        "metric_code": "challenger_ks",
        "severity": "HIGH",
    },
    "HIGH_DEPLOYMENT_SCORE_PSI": {
        "name": "High deployment score PSI",
        "metric_code": "score_psi",
        "severity": "HIGH",
    },
    "BAD_RATE_DRIFT_HIGH": {
        "name": "High bad-rate drift during canary",
        "metric_code": "bad_rate_drift",
        "severity": "HIGH",
    },
    "TRAIN_VALID_GAP_LARGE": {
        "name": "Large train-validation performance gap",
        "metric_code": "train_valid_gap",
        "severity": "WARNING",
    },
    "RECOVERY_RATE_LOW": {
        "name": "Low recovery rate",
        "metric_code": "recovery_rate",
        "severity": "HIGH",
    },
    "OOT_DEPLOYMENT_RISK": {
        "name": "OOT validation failed before deployment",
        "metric_code": "oot_passed",
        "severity": "HIGH",
    },
    "DISCRIMINATION_GATE_FAILED": {
        "name": "Discrimination validation gate failed",
        "metric_code": "discrimination_passed",
        "severity": "HIGH",
    },
    "CALIBRATION_GATE_FAILED": {
        "name": "Calibration validation gate failed",
        "metric_code": "calibration_passed",
        "severity": "HIGH",
    },
}


DEPLOYMENT_RISKS: dict[str, dict] = {
    "MODEL_PERFORMANCE_REGRESSION_RISK": {
        "name": "Model performance regression risk",
        "severity": "HIGH",
    },
    "ONLINE_SCORE_DISTRIBUTION_RISK": {
        "name": "Online score distribution drift risk",
        "severity": "HIGH",
    },
    "BUSINESS_BAD_RATE_RISK": {
        "name": "Business bad-rate drift risk",
        "severity": "HIGH",
    },
    "OVERFITTING_GENERALIZATION_RISK": {
        "name": "Overfitting or poor generalization risk",
        "severity": "MEDIUM",
    },
    "RECOVERY_INSUFFICIENT_RISK": {
        "name": "Insufficient recovery risk",
        "severity": "HIGH",
    },
    "OOT_STABILITY_RISK": {
        "name": "Out-of-time stability risk",
        "severity": "HIGH",
    },
    "MODEL_VALIDATION_GATE_RISK": {
        "name": "Model validation gate risk",
        "severity": "HIGH",
    },
}


DEPLOYMENT_STRATEGIES: dict[str, dict] = {
    "pause_canary_and_review": {
        "name": "Pause canary and require review",
        "action_type": "PAUSE_CANARY",
        "allowed_stages": ["CANARY_5", "CANARY_20", "CANARY_50"],
        "parameters": {"requires_manual_review": True},
    },
    "rollback_to_stable": {
        "name": "Rollback to stable champion",
        "action_type": "ROLLBACK",
        "allowed_stages": ["CANARY_20", "CANARY_50", "PRODUCTION"],
        "parameters": {"rollback_target": "stable_version"},
    },
    "reduce_to_previous_canary": {
        "name": "Reduce traffic to previous canary stage",
        "action_type": "REDUCE_TRAFFIC",
        "allowed_stages": ["CANARY_20", "CANARY_50"],
        "parameters": {"fallback_stage": "previous_canary"},
    },
    "hold_for_oot_investigation": {
        "name": "Hold deployment for OOT investigation",
        "action_type": "HOLD",
        "allowed_stages": ["OOT_GATE", "OFFLINE_VALIDATION"],
        "parameters": {"requires_oot_report": True},
    },
    "advance_with_close_monitoring": {
        "name": "Advance with close monitoring",
        "action_type": "ADVANCE_STAGE",
        "allowed_stages": ["OFFLINE_VALIDATION", "OOT_GATE", "SHADOW", "CANARY_5"],
        "parameters": {"monitoring_window": "short"},
    },
}


DEPLOYMENT_STAGES: dict[str, dict] = {
    "OFFLINE_VALIDATION": {"name": "Offline validation", "traffic_ratio": 0.0},
    "OOT_GATE": {"name": "Out-of-time validation gate", "traffic_ratio": 0.0},
    "SHADOW": {"name": "Shadow traffic", "traffic_ratio": 0.0},
    "CANARY_5": {"name": "5 percent canary", "traffic_ratio": 0.05},
    "CANARY_20": {"name": "20 percent canary", "traffic_ratio": 0.20},
    "CANARY_50": {"name": "50 percent canary", "traffic_ratio": 0.50},
    "PRODUCTION": {"name": "Production", "traffic_ratio": 1.0},
}


DEPLOYMENT_POLICIES: dict[str, dict] = {
    "CANARY_SAFETY_POLICY_V1": {
        "name": "Canary safety policy",
        "description": "Block or rollback canary rollout when live risk metrics fail.",
    },
    "PRODUCTION_PROMOTION_POLICY_V1": {
        "name": "Production promotion policy",
        "description": "Promote only after all health and KG risk checks pass.",
    },
    "OOT_STABILITY_POLICY_V1": {
        "name": "OOT stability policy",
        "description": "Never continue when OOT stability gates fail.",
    },
}


ALERT_RISK_RELATIONS: list[dict] = [
    {
        "alert": "CHALLENGER_AUC_REGRESSION",
        "risk": "MODEL_PERFORMANCE_REGRESSION_RISK",
        "weight": 0.82,
        "confidence": 0.64,
        "cases": 18,
    },
    {
        "alert": "CHALLENGER_KS_REGRESSION",
        "risk": "MODEL_PERFORMANCE_REGRESSION_RISK",
        "weight": 0.78,
        "confidence": 0.60,
        "cases": 16,
    },
    {
        "alert": "HIGH_DEPLOYMENT_SCORE_PSI",
        "risk": "ONLINE_SCORE_DISTRIBUTION_RISK",
        "weight": 0.80,
        "confidence": 0.62,
        "cases": 14,
    },
    {
        "alert": "BAD_RATE_DRIFT_HIGH",
        "risk": "BUSINESS_BAD_RATE_RISK",
        "weight": 0.86,
        "confidence": 0.70,
        "cases": 25,
    },
    {
        "alert": "TRAIN_VALID_GAP_LARGE",
        "risk": "OVERFITTING_GENERALIZATION_RISK",
        "weight": 0.72,
        "confidence": 0.55,
        "cases": 11,
    },
    {
        "alert": "RECOVERY_RATE_LOW",
        "risk": "RECOVERY_INSUFFICIENT_RISK",
        "weight": 0.84,
        "confidence": 0.66,
        "cases": 19,
    },
    {
        "alert": "OOT_DEPLOYMENT_RISK",
        "risk": "OOT_STABILITY_RISK",
        "weight": 0.88,
        "confidence": 0.72,
        "cases": 21,
    },
    {
        "alert": "DISCRIMINATION_GATE_FAILED",
        "risk": "MODEL_VALIDATION_GATE_RISK",
        "weight": 0.86,
        "confidence": 0.70,
        "cases": 17,
    },
    {
        "alert": "CALIBRATION_GATE_FAILED",
        "risk": "MODEL_VALIDATION_GATE_RISK",
        "weight": 0.82,
        "confidence": 0.64,
        "cases": 15,
    },
]


RISK_STRATEGY_RELATIONS: list[dict] = [
    {
        "risk": "BUSINESS_BAD_RATE_RISK",
        "strategy": "rollback_to_stable",
        "weight": 0.76,
        "confidence": 0.63,
        "cases": 12,
        "natural_cases": 20,
    },
    {
        "risk": "BUSINESS_BAD_RATE_RISK",
        "strategy": "pause_canary_and_review",
        "weight": 0.70,
        "confidence": 0.58,
        "cases": 15,
        "natural_cases": 25,
    },
    {
        "risk": "ONLINE_SCORE_DISTRIBUTION_RISK",
        "strategy": "pause_canary_and_review",
        "weight": 0.68,
        "confidence": 0.54,
        "cases": 10,
        "natural_cases": 14,
    },
    {
        "risk": "ONLINE_SCORE_DISTRIBUTION_RISK",
        "strategy": "reduce_to_previous_canary",
        "weight": 0.62,
        "confidence": 0.50,
        "cases": 8,
        "natural_cases": 12,
    },
    {
        "risk": "MODEL_PERFORMANCE_REGRESSION_RISK",
        "strategy": "pause_canary_and_review",
        "weight": 0.66,
        "confidence": 0.52,
        "cases": 9,
        "natural_cases": 13,
    },
    {
        "risk": "OVERFITTING_GENERALIZATION_RISK",
        "strategy": "hold_for_oot_investigation",
        "weight": 0.69,
        "confidence": 0.53,
        "cases": 9,
        "natural_cases": 13,
    },
    {
        "risk": "RECOVERY_INSUFFICIENT_RISK",
        "strategy": "rollback_to_stable",
        "weight": 0.72,
        "confidence": 0.57,
        "cases": 10,
        "natural_cases": 16,
    },
    {
        "risk": "OOT_STABILITY_RISK",
        "strategy": "hold_for_oot_investigation",
        "weight": 0.74,
        "confidence": 0.60,
        "cases": 11,
        "natural_cases": 17,
    },
    {
        "risk": "MODEL_VALIDATION_GATE_RISK",
        "strategy": "hold_for_oot_investigation",
        "weight": 0.70,
        "confidence": 0.55,
        "cases": 10,
        "natural_cases": 15,
    },
]


STAGE_STRATEGY_ALLOWS: list[tuple[str, str]] = [
    (stage, strategy)
    for strategy, payload in DEPLOYMENT_STRATEGIES.items()
    for stage in payload["allowed_stages"]
]


POLICY_STRATEGY_CONSTRAINTS: list[tuple[str, str]] = [
    ("CANARY_SAFETY_POLICY_V1", "pause_canary_and_review"),
    ("CANARY_SAFETY_POLICY_V1", "reduce_to_previous_canary"),
    ("CANARY_SAFETY_POLICY_V1", "rollback_to_stable"),
    ("PRODUCTION_PROMOTION_POLICY_V1", "rollback_to_stable"),
    ("OOT_STABILITY_POLICY_V1", "hold_for_oot_investigation"),
]


def _relation_payload(relation_key: str, relation_type: str, weight: float, confidence: float, cases: int) -> dict:
    return {
        "relation_key": relation_key,
        "relation_type": relation_type,
        "initial_prior_weight": weight,
        "prior_strength": 1.0,
        "effective_weight": weight,
        "confidence_lower_bound": confidence,
        "confidence_upper_bound": min(0.99, weight + 0.12),
        "evidence_case_count": cases,
        "natural_case_count": cases,
        "scenario_case_count": 0,
        "support_count": cases,
        "against_count": 0,
        "neutral_count": 0,
        "support_strength": float(cases),
        "against_strength": 0.0,
        "weight_version": "DEPLOYMENT_SEED_V1",
        "enabled": True,
    }


async def seed() -> None:
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    async with driver.session(database="neo4j") as session:
        for code, payload in DEPLOYMENT_ALERTS.items():
            await session.run(
                """
                MERGE (a:DeploymentAlert {entity_code: $code})
                SET a.name = $name,
                    a.entity_type = 'DeploymentAlert',
                    a.namespace = 'DEPLOYMENT',
                    a.metric_code = $metric_code,
                    a.severity = $severity,
                    a.enabled = true
                """,
                code=code,
                **payload,
            )

        for code, payload in DEPLOYMENT_RISKS.items():
            await session.run(
                """
                MERGE (r:DeploymentRisk {entity_code: $code})
                SET r.name = $name,
                    r.entity_type = 'DeploymentRisk',
                    r.namespace = 'DEPLOYMENT',
                    r.severity = $severity,
                    r.enabled = true
                """,
                code=code,
                **payload,
            )

        for code, payload in DEPLOYMENT_STRATEGIES.items():
            await session.run(
                """
                MERGE (s:DeploymentStrategy {entity_code: $code})
                SET s.name = $name,
                    s.entity_type = 'DeploymentStrategy',
                    s.namespace = 'DEPLOYMENT',
                    s.action_type = $action_type,
                    s.allowed_stages = $allowed_stages,
                    s.parameters = $strategy_parameters,
                    s.enabled = true
                """,
                code=code,
                name=payload["name"],
                action_type=payload["action_type"],
                allowed_stages=payload["allowed_stages"],
                strategy_parameters=json.dumps(payload["parameters"], ensure_ascii=False),
            )

        for code, payload in DEPLOYMENT_STAGES.items():
            await session.run(
                """
                MERGE (s:DeploymentStage {entity_code: $code})
                SET s.name = $name,
                    s.entity_type = 'DeploymentStage',
                    s.namespace = 'DEPLOYMENT',
                    s.traffic_ratio = $traffic_ratio,
                    s.enabled = true
                """,
                code=code,
                **payload,
            )

        for code, payload in DEPLOYMENT_POLICIES.items():
            await session.run(
                """
                MERGE (p:DeploymentPolicy {entity_code: $code})
                SET p.name = $name,
                    p.description = $description,
                    p.entity_type = 'DeploymentPolicy',
                    p.namespace = 'DEPLOYMENT',
                    p.enabled = true
                """,
                code=code,
                **payload,
            )

        for item in ALERT_RISK_RELATIONS:
            relation_key = f"{item['alert']}|INDICATES|{item['risk']}"
            await session.run(
                """
                MATCH (a:DeploymentAlert {entity_code: $alert})
                MATCH (r:DeploymentRisk {entity_code: $risk})
                MERGE (a)-[rel:INDICATES]->(r)
                SET rel += $payload,
                    rel.source_entity_code = $alert,
                    rel.target_entity_code = $risk
                """,
                alert=item["alert"],
                risk=item["risk"],
                payload=_relation_payload(
                    relation_key,
                    "INDICATES",
                    item["weight"],
                    item["confidence"],
                    item["cases"],
                ),
            )

        for item in RISK_STRATEGY_RELATIONS:
            relation_key = f"{item['risk']}|RECOMMENDS|{item['strategy']}"
            await session.run(
                """
                MATCH (r:DeploymentRisk {entity_code: $risk})
                MATCH (s:DeploymentStrategy {entity_code: $strategy})
                MERGE (r)-[rel:RECOMMENDS]->(s)
                SET rel += $payload,
                    rel.source_entity_code = $risk,
                    rel.target_entity_code = $strategy,
                    rel.natural_case_count = $natural_cases
                """,
                risk=item["risk"],
                strategy=item["strategy"],
                natural_cases=item["natural_cases"],
                payload=_relation_payload(
                    relation_key,
                    "RECOMMENDS",
                    item["weight"],
                    item["confidence"],
                    item["cases"],
                ),
            )

            mitigates_key = f"{item['strategy']}|MITIGATES|{item['risk']}"
            await session.run(
                """
                MATCH (s:DeploymentStrategy {entity_code: $strategy})
                MATCH (r:DeploymentRisk {entity_code: $risk})
                MERGE (s)-[rel:MITIGATES]->(r)
                SET rel += $payload,
                    rel.source_entity_code = $strategy,
                    rel.target_entity_code = $risk,
                    rel.natural_case_count = $natural_cases
                """,
                strategy=item["strategy"],
                risk=item["risk"],
                natural_cases=item["natural_cases"],
                payload=_relation_payload(
                    mitigates_key,
                    "MITIGATES",
                    item["weight"],
                    item["confidence"],
                    item["cases"],
                ),
            )

        for stage, strategy in STAGE_STRATEGY_ALLOWS:
            relation_key = f"{stage}|ALLOWS|{strategy}"
            await session.run(
                """
                MATCH (stage:DeploymentStage {entity_code: $stage})
                MATCH (strategy:DeploymentStrategy {entity_code: $strategy})
                MERGE (stage)-[rel:ALLOWS]->(strategy)
                SET rel.relation_key = $relation_key,
                    rel.relation_type = 'ALLOWS',
                    rel.source_entity_code = $stage,
                    rel.target_entity_code = $strategy,
                    rel.enabled = true
                """,
                stage=stage,
                strategy=strategy,
                relation_key=relation_key,
            )

        for policy, strategy in POLICY_STRATEGY_CONSTRAINTS:
            relation_key = f"{policy}|CONSTRAINS|{strategy}"
            await session.run(
                """
                MATCH (policy:DeploymentPolicy {entity_code: $policy})
                MATCH (strategy:DeploymentStrategy {entity_code: $strategy})
                MERGE (policy)-[rel:CONSTRAINS]->(strategy)
                SET rel.relation_key = $relation_key,
                    rel.relation_type = 'CONSTRAINS',
                    rel.source_entity_code = $policy,
                    rel.target_entity_code = $strategy,
                    rel.enabled = true
                """,
                policy=policy,
                strategy=strategy,
                relation_key=relation_key,
            )

    await driver.close()


def print_summary() -> None:
    print("Deployment KG seed data:")
    print(f"- DeploymentAlert: {len(DEPLOYMENT_ALERTS)}")
    print(f"- DeploymentRisk: {len(DEPLOYMENT_RISKS)}")
    print(f"- DeploymentStrategy: {len(DEPLOYMENT_STRATEGIES)}")
    print(f"- DeploymentStage: {len(DEPLOYMENT_STAGES)}")
    print(f"- DeploymentPolicy: {len(DEPLOYMENT_POLICIES)}")
    print(f"- INDICATES: {len(ALERT_RISK_RELATIONS)}")
    print(f"- RECOMMENDS: {len(RISK_STRATEGY_RELATIONS)}")
    print(f"- MITIGATES: {len(RISK_STRATEGY_RELATIONS)}")
    print(f"- ALLOWS: {len(STAGE_STRATEGY_ALLOWS)}")
    print(f"- CONSTRAINS: {len(POLICY_STRATEGY_CONSTRAINTS)}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Task 4 deployment KG data.")
    parser.add_argument("--dry-run", action="store_true", help="Only print seed summary.")
    args = parser.parse_args()

    print_summary()
    if args.dry_run:
        return

    await seed()
    print("Deployment KG seed completed.")


if __name__ == "__main__":
    asyncio.run(main())
