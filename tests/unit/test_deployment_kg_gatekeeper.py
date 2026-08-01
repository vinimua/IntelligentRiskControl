from packages.models.deployment.deployment_context import (
    DeploymentContext,
    DeploymentRisk,
    DeploymentStrategyCandidate,
)
from apps.modelops_api.services.deployment.deployment_gatekeeper_service import (
    DeploymentGatekeeperService,
)
from apps.modelops_api.services.deployment.deployment_observe_service import (
    build_deployment_alerts,
)


def test_build_deployment_alerts_uses_metric_thresholds_not_failure_text_only():
    alerts = build_deployment_alerts(
        "CANARY_20",
        {"challenger_auc": 0.75, "bad_rate_drift": 0.12},
        {
            "passed": False,
            "failures": ["Bad rate drift 0.1200 > 0.08"],
            "warnings": [],
            "rollback_recommended": False,
            "rollback_reasons": [],
        },
        lifecycle_run_id="run-001",
        deployment_id="deploy-001",
    )

    assert len(alerts) == 1
    assert alerts[0].alert_code == "BAD_RATE_DRIFT_HIGH"
    assert alerts[0].metric_code == "bad_rate_drift"
    assert alerts[0].threshold == 0.08
    assert alerts[0].evidence_detail["stage"] == "CANARY_20"


def test_gatekeeper_accepts_plain_rollback_action_type_from_kg():
    ctx = DeploymentContext(
        context_pack_id="ctx-001",
        stage="CANARY_20",
        deployment_risks=[
            DeploymentRisk(
                risk_code="BAD_RATE_RISK",
                relation_key="BAD_RATE_DRIFT_HIGH|INDICATES|BAD_RATE_RISK",
                effective_weight_snapshot=0.8,
                confidence_lower_bound_snapshot=0.6,
                strategy_candidates=[
                    DeploymentStrategyCandidate(
                        strategy_code="rollback_when_bad_rate_spikes",
                        relation_key="BAD_RATE_RISK|RECOMMENDS|rollback_when_bad_rate_spikes",
                        effective_weight_snapshot=0.72,
                        confidence_lower_bound_snapshot=0.6,
                        support_case_count=5,
                        action_type="ROLLBACK",
                    )
                ],
            )
        ],
    )

    result = DeploymentGatekeeperService().decide(
        "CANARY_20",
        {"passed": True, "failures": [], "warnings": []},
        ctx,
    )

    assert result.decision == "ROLLBACK"
    assert result.selected_strategy_code == "rollback_when_bad_rate_spikes"


def test_gatekeeper_ignores_strategy_not_allowed_in_current_stage():
    ctx = DeploymentContext(
        context_pack_id="ctx-001",
        stage="CANARY_5",
        deployment_risks=[
            DeploymentRisk(
                risk_code="PRODUCTION_ONLY_RISK",
                relation_key="A|INDICATES|R",
                effective_weight_snapshot=0.8,
                confidence_lower_bound_snapshot=0.6,
                strategy_candidates=[
                    DeploymentStrategyCandidate(
                        strategy_code="production_rollback_policy",
                        relation_key="R|RECOMMENDS|S",
                        effective_weight_snapshot=0.9,
                        confidence_lower_bound_snapshot=0.7,
                        support_case_count=10,
                        action_type="ROLLBACK",
                        allowed_stages=["PRODUCTION"],
                    )
                ],
            )
        ],
    )

    result = DeploymentGatekeeperService().decide(
        "CANARY_5",
        {"passed": True, "failures": [], "warnings": []},
        ctx,
    )

    assert result.decision == "ADVANCE_STAGE"
