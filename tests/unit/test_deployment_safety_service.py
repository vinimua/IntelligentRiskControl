from apps.modelops_api.services.iteration.deployment_safety_service import (
    DeploymentSafetyService,
)


class FakeSession:
    def __init__(self):
        self.calls = []

    async def execute(self, statement, params=None):
        self.calls.append((str(statement), params or {}))


def test_canary_bad_rate_drift_is_blocking_failure():
    result = DeploymentSafetyService.check_stage_health(
        "CANARY_20",
        {"challenger_auc": 0.75, "bad_rate_drift": 0.12},
    )

    assert result["passed"] is False
    assert any("Bad rate drift" in item for item in result["failures"])
    assert result["rollback_recommended"] is False


def test_severe_canary_bad_rate_drift_recommends_rollback():
    result = DeploymentSafetyService.check_stage_health(
        "CANARY_20",
        {"challenger_auc": 0.75, "bad_rate_drift": 0.18},
    )

    assert result["passed"] is False
    assert result["rollback_recommended"] is True
    assert any("severe_bad_rate_drift" in item for item in result["rollback_reasons"])


def test_oot_train_valid_gap_is_blocking_failure():
    result = DeploymentSafetyService.check_stage_health(
        "OOT_GATE",
        {"challenger_auc": 0.75, "train_valid_gap": 0.08},
    )

    assert result["passed"] is False
    assert any("Train/valid gap" in item for item in result["failures"])
    assert result["rollback_recommended"] is False


async def test_update_traffic_ratio_writes_routing_versions():
    session = FakeSession()
    service = DeploymentSafetyService(session)

    ratio = await service.update_traffic_ratio(
        "credit_model_001",
        "CANARY_20",
        champion_version="champion_v1",
        challenger_version="challenger_v2",
    )

    assert ratio == 0.20
    sql, params = session.calls[0]
    assert "active_version_code" in sql
    assert "stable_version_code" in sql
    assert "challenger_version_code" in sql
    assert params["active"] == "champion_v1"
    assert params["stable"] == "champion_v1"
    assert params["challenger"] == "challenger_v2"


async def test_promote_updates_model_registry_current_champion():
    session = FakeSession()
    service = DeploymentSafetyService(session)

    await service.promote_to_champion(
        {
            "deployment_id": "deploy-001",
            "model_id": "credit_model_001",
            "champion_version": "champion_v1",
            "candidate_version": "challenger_v2",
        }
    )

    statements = "\n".join(sql for sql, _ in session.calls)
    assert "UPDATE model_registry.models" in statements
    assert "current_champion_version = :challenger" in statements


async def test_rollback_updates_model_registry_current_champion():
    session = FakeSession()
    service = DeploymentSafetyService(session)

    await service.rollback(
        {
            "deployment_id": "deploy-001",
            "model_id": "credit_model_001",
            "champion_version": "champion_v1",
            "current_stage": "CANARY_20",
        },
        rollback_target="stable_v1",
    )

    statements = "\n".join(sql for sql, _ in session.calls)
    assert "UPDATE model_registry.models" in statements
    assert "current_champion_version = :target" in statements
