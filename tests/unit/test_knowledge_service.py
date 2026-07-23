"""KnowledgeService 单元测试 — Mock Neo4j 驱动，验证降级和查询逻辑。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neo4j import AsyncDriver

from apps.modelops_api.services.knowledge_service import (
    KnowledgeService,
    AlertResult,
    _DEFAULT_METRIC_ALERT_MAP,
)
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
