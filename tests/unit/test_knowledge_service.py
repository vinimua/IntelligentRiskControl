"""KnowledgeService 单元测试 — Mock Neo4j 驱动，验证降级和查询逻辑。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neo4j import AsyncDriver

from apps.modelops_api.services.knowledge_service import (
    KnowledgeService,
    AlertResult,
    _DEFAULT_METRIC_ALERT_MAP,
    _SUPPORTED_TRAINING_ALGORITHMS,
)
from workers.training_tasks import TRAINERS
from packages.models.common.enums import Severity


@pytest.fixture
def mock_driver() -> AsyncDriver:
    """创建一个 Mock Neo4j 异步驱动。"""
    driver = MagicMock(spec=AsyncDriver)
    driver.session = MagicMock()
    return driver


class TestDefaultMappingFallback:
    """Neo4j 不可用时，验证降级到内置默认映射。"""

    async def test_resolve_alert_falls_back_when_neo4j_unavailable(self, mock_driver):
        mock_driver.session.side_effect = Exception("Connection refused")
        svc = KnowledgeService(mock_driver)

        result = await svc.resolve_alert("FEATURE_PSI")
        assert result is not None
        assert result.alert_code == "HIGH_FEATURE_PSI"
        assert result.severity == Severity.HIGH
        assert result.from_neo4j is False

    async def test_resolve_alert_returns_none_for_unknown_metric(self, mock_driver):
        mock_driver.session.side_effect = Exception("Connection refused")
        svc = KnowledgeService(mock_driver)

        result = await svc.resolve_alert("NONEXISTENT_METRIC")
        assert result is None

    async def test_all_seven_default_metrics_have_valid_mapping(self):
        """验证所有 7 个内置默认映射包含必要的字段。"""
        for metric_code, mapping in _DEFAULT_METRIC_ALERT_MAP.items():
            assert isinstance(mapping["alert_code"], str), f"{metric_code}: alert_code 不是 str"
            assert isinstance(mapping["severity"], Severity), f"{metric_code}: severity 不是 Severity"
            assert isinstance(mapping["description"], str), f"{metric_code}: description 不是 str"
            assert len(mapping["description"]) > 0, f"{metric_code}: description 为空"


class TestAlertResult:
    def test_alert_result_fields(self):
        result = AlertResult(
            alert_code="HIGH_FEATURE_PSI",
            metric_code="FEATURE_PSI",
            severity=Severity.HIGH,
            effective_weight=0.95,
            description="特征漂移",
            from_neo4j=True,
        )
        assert result.alert_code == "HIGH_FEATURE_PSI"
        assert result.metric_code == "FEATURE_PSI"
        assert result.severity == Severity.HIGH
        assert result.from_neo4j is True

    def test_alert_result_from_default(self):
        result = AlertResult(
            alert_code="AUC_DROP",
            metric_code="AUC",
            severity=Severity.WARNING,
            description="AUC下降",
            from_neo4j=False,
        )
        assert result.from_neo4j is False
        assert result.effective_weight == 1.0


class TestKnowledgeServiceInit:
    def test_service_accepts_driver(self, mock_driver):
        svc = KnowledgeService(mock_driver)
        assert svc.driver is mock_driver


def test_kg_training_algorithm_whitelist_matches_worker_registry():
    assert _SUPPORTED_TRAINING_ALGORITHMS == set(TRAINERS)


class _AsyncRecordResult:
    def __init__(self, records):
        self.records = records

    def __aiter__(self):
        self._iter = iter(self.records)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class _DeploymentSession:
    def __init__(self):
        self.params = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def run(self, query, **params):
        self.query = query
        self.params = params
        return _AsyncRecordResult(
            [
                {
                    "alert_code": "BAD_RATE_DRIFT_HIGH",
                    "risk_code": "BAD_RATE_RISK",
                    "risk_name": "Bad rate deployment risk",
                    "risk_relation_key": "BAD_RATE_DRIFT_HIGH|INDICATES|BAD_RATE_RISK",
                    "risk_weight": 0.8,
                    "risk_confidence": 0.6,
                    "strategy_code": "pause_canary",
                    "action_type": "PAUSE_CANARY",
                    "strategy_parameters": "{\"traffic_ratio\": 0.05}",
                    "strategy_relation_key": "BAD_RATE_RISK|RECOMMENDS|pause_canary",
                    "strategy_weight": 0.7,
                    "strategy_confidence": 0.5,
                    "support_case_count": 12,
                    "natural_case_count": 20,
                    "mitigates_relation_key": "pause_canary|MITIGATES|BAD_RATE_RISK",
                    "mitigates_weight": 0.7,
                    "allowed_stages": ["CANARY_20"],
                    "policy_refs": ["CANARY_POLICY_V1"],
                }
            ]
        )


class _DeploymentDriver:
    def __init__(self):
        self.session_obj = _DeploymentSession()

    def session(self, **_kwargs):
        return self.session_obj


async def test_query_deployment_context_keeps_alert_payload_and_stage_param():
    driver = _DeploymentDriver()
    svc = KnowledgeService(driver)

    ctx = await svc.query_deployment_context(
        alert_codes=["BAD_RATE_DRIFT_HIGH"],
        alert_payloads=[
            {
                "alert_code": "BAD_RATE_DRIFT_HIGH",
                "metric_code": "bad_rate_drift",
                "stage": "CANARY_20",
                "value": 0.12,
            }
        ],
        stage="CANARY_20",
        model_id="credit_model_001",
    )

    assert driver.session_obj.params["stage"] == "CANARY_20"
    assert ctx.deployment_alerts[0]["metric_code"] == "bad_rate_drift"
    assert ctx.deployment_risks[0].evidence_detail["alerts"][0]["value"] == 0.12
    assert ctx.deployment_risks[0].strategy_candidates[0].allowed_stages == ["CANARY_20"]
    assert ctx.deployment_risks[0].strategy_candidates[0].parameters == {"traffic_ratio": 0.05}
