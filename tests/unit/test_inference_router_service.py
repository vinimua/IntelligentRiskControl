import pytest

from apps.modelops_api.services.inference.inference_router_service import (
    InferenceRouterService,
    LoadedModel,
    _stable_hash,
)


class FakeResult:
    def __init__(self, row=None):
        self.row = row

    def mappings(self):
        return self

    def first(self):
        return self.row


class FakeSession:
    def __init__(self, row=None):
        self.row = row
        self.calls = []

    async def execute(self, statement, params=None):
        self.calls.append((str(statement), params or {}))
        return FakeResult(self.row)


class SequencedFakeSession:
    def __init__(self, rows):
        self.rows = list(rows)
        self.calls = []

    async def execute(self, statement, params=None):
        self.calls.append((str(statement), params or {}))
        row = self.rows.pop(0) if self.rows else None
        return FakeResult(row)


class FakeProbabilityModel:
    feature_names_in_ = ["age", "income", "debt_ratio"]

    def __init__(self):
        self.last_columns = None

    def predict_proba(self, frame):
        self.last_columns = list(frame.columns)
        return [[0.77, 0.23]]


def test_stable_hash_is_deterministic():
    first = _stable_hash("request-001", "credit_model_001")
    second = _stable_hash("request-001", "credit_model_001")

    assert first == second
    assert 0 <= first < 1


@pytest.mark.asyncio
async def test_get_routing_state_filters_environment():
    session = FakeSession()
    service = InferenceRouterService(session)

    state = await service.get_routing_state("credit_model_001", environment="PROD")

    sql, params = session.calls[0]
    assert "environment = :env" in sql
    assert params == {"mid": "credit_model_001", "env": "PROD"}
    assert state["environment"] == "PROD"
    assert state["challenger_traffic_ratio"] == 0.0


@pytest.mark.asyncio
async def test_route_is_sticky_for_same_request_id():
    session = FakeSession(
        {
            "model_id": "credit_model_001",
            "active_version_code": "champion_v1",
            "stable_version_code": "champion_v0",
            "challenger_version_code": "challenger_v2",
            "challenger_traffic_ratio": 0.2,
        }
    )
    service = InferenceRouterService(session)

    first = await service.route("credit_model_001", "same-user")
    second = await service.route("credit_model_001", "same-user")

    assert first["chosen_version"] == second["chosen_version"]
    assert first["chosen_role"] == second["chosen_role"]
    assert first["hash_value"] == second["hash_value"]


@pytest.mark.asyncio
async def test_predict_loads_artifact_and_uses_model_predict_proba(monkeypatch):
    fake_model = FakeProbabilityModel()
    session = SequencedFakeSession([
        {
            "model_id": "credit_model_001",
            "active_version_code": "champion_v1",
            "stable_version_code": "champion_v0",
            "challenger_version_code": None,
            "challenger_traffic_ratio": 0.0,
        },
        {
            "artifact_uri": "s3://riskitem/challengers/champion_v1/model.joblib",
            "metrics_json": {"val_auc": 0.81},
        },
    ])
    service = InferenceRouterService(session)
    monkeypatch.setattr(
        service,
        "_load_model_artifact",
        lambda uri: LoadedModel(model=fake_model, artifact_uri=uri, loader="joblib"),
    )

    result = await service.predict(
        "credit_model_001",
        "request-001",
        {"age": 30, "income": 12000, "unused": 1},
    )

    assert result["prediction"]["score"] == 0.23
    assert result["prediction"]["score_source"] == "real_model_predict_proba"
    assert result["artifact"]["artifact_source"] == "model_registry.model_versions"
    assert result["feature_schema"]["missing_features_filled_with_zero"] == ["debt_ratio"]
    assert result["feature_schema"]["extra_features_ignored"] == ["unused"]
    assert fake_model.last_columns == ["age", "income", "debt_ratio"]
