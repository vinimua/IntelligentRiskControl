"""MonitoringService 单元测试 — Mock DB + Mock KnowledgeService 验证核心流程。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apps.modelops_api.services.monitoring.monitoring_service import MonitoringService, MonitoringRunResult
from apps.modelops_api.services.knowledge_service import AlertTypeResult
from packages.models.common.enums import Severity, AvailabilityStatus


def _make_healthy_data(n: int = 500) -> list[dict]:
    """生成健康数据集（无漂移）。"""
    import random as _random
    import math
    rng = _random.Random(42)
    data = []
    for _ in range(n):
        f1 = rng.gauss(0, 1)
        f2 = rng.gauss(0, 1)
        z = 0.7 * f1 + 0.4 * f2 + rng.gauss(0, 0.1)
        proba = 1.0 / (1.0 + math.exp(-z))
        y_true = 1 if rng.random() < proba else 0
        data.append({
            "y_true": y_true,
            "y_pred_proba": round(proba, 6),
            "score": int(300 + 200 * proba),
            "feature_f1": round(f1, 4),
            "feature_f2": round(f2, 4),
            "feature_f3": round(rng.gauss(5, 2), 4),
        })
    return data


def _make_drift_data(n: int = 500) -> list[dict]:
    """生成漂移数据集。"""
    import random as _random
    import math
    rng = _random.Random(123)
    data = []
    for _ in range(n):
        f1 = rng.gauss(0.5, 1)  # 均值偏移 0.5
        f2 = rng.gauss(0.3, 1)
        f3 = rng.gauss(4.5, 2)
        z = 0.7 * f1 + 0.4 * f2 + rng.gauss(0, 0.1)
        proba = 1.0 / (1.0 + math.exp(-z))
        y_true = 1 if rng.random() < proba else 0
        data.append({
            "y_true": y_true,
            "y_pred_proba": round(proba, 6),
            "score": int(300 + 200 * proba),
            "feature_f1": round(f1, 4),
            "feature_f2": round(f2, 4),
            "feature_f3": round(f3, 4),
        })
    return data


@pytest.fixture
def mock_knowledge() -> MagicMock:
    """Mock KnowledgeService 返回固定 AlertTypeResult。"""
    ks = MagicMock()
    ks.resolve_alert = AsyncMock(return_value=AlertTypeResult(
        alert_code="HIGH_FEATURE_PSI",
        metric_code="FEATURE_PSI",
        severity=Severity.HIGH,
        effective_weight=1.0,
        description="特征漂移",
        from_neo4j=False,
    ))
    return ks


@pytest.fixture
def mock_session() -> MagicMock:
    """Mock AsyncSession with proper execute → mappings → first chain。"""
    s = MagicMock()
    s.commit = AsyncMock()
    s.rollback = AsyncMock()

    # Mock the execute → mappings → first() chain (returns None = insert OK)
    mock_result = MagicMock()
    mock_mappings = MagicMock()
    mock_mappings.first = MagicMock(return_value=None)
    mock_result.mappings = MagicMock(return_value=mock_mappings)
    s.execute = AsyncMock(return_value=mock_result)
    return s


class TestMonitoringServiceHealthy:
    """健康场景：所有指标正常，无告警。"""

    async def test_healthy_no_alerts(self, mock_session, mock_knowledge):
        svc = MonitoringService(mock_session, mock_knowledge)
        baseline = _make_healthy_data(500)
        current = _make_healthy_data(500)

        result = await svc.run(
            model_id="test_model",
            champion_version="v1",
            baseline_data=baseline,
            current_data=current,
        )

        assert isinstance(result, MonitoringRunResult)
        assert result.has_alerts is False
        assert result.alert_count == 0
        assert result.max_alert_severity is None
        assert result.monitoring_run_id is not None


class TestMonitoringServiceDrift:
    """漂移场景：FEATURE_PSI 超阈值 → 产生告警。"""

    async def test_drift_triggers_alert(self, mock_session, mock_knowledge):
        svc = MonitoringService(mock_session, mock_knowledge)
        baseline = _make_healthy_data(500)
        current = _make_drift_data(500)

        result = await svc.run(
            model_id="test_model",
            champion_version="v1",
            baseline_data=baseline,
            current_data=current,
        )

        assert result.has_alerts is True
        assert result.alert_count >= 1
        # 应包含 FEATURE_PSI 告警
        alert_codes = [a.alert_code for a in result.alerts]
        assert any("FEATURE_PSI" in code or "PSI" in code for code in alert_codes)


class TestMonitoringServiceSmallSample:
    """小样本场景：SAMPLE_SIZE < 50 → SAMPLE_SIZE_LOW。"""

    async def test_small_sample_triggers_alert(self, mock_session, mock_knowledge):
        svc = MonitoringService(mock_session, mock_knowledge)
        # SAMPLE_SIZE critical threshold is 50
        data = _make_healthy_data(30)

        result = await svc.run(
            model_id="test_model",
            champion_version="v1",
            baseline_data=_make_healthy_data(500),
            current_data=data,
        )

        # SAMPLE_SIZE = 30 < 50 (critical threshold)
        sample_alerts = [a for a in result.alerts if a.metric_code == "SAMPLE_SIZE"]
        assert len(sample_alerts) >= 1
        # severity 来自 KnowledgeService mock（固定返回 HIGH），真实场景由图谱查询决定
        assert sample_alerts[0].severity.value in ("CRITICAL", "WARNING", "HIGH")


class TestMonitoringServiceSchemaChange:
    """Schema 变化场景：列不一致 → SCHEMA_CHANGE。"""

    async def test_schema_change_triggers_alert(self, mock_session, mock_knowledge):
        svc = MonitoringService(mock_session, mock_knowledge)
        baseline = [{"a": 1, "b": 2.0}] * 500
        current = [{"a": 1, "b": 2.0, "c": 3.0}] * 500

        result = await svc.run(
            model_id="test_model",
            champion_version="v1",
            baseline_data=baseline,
            current_data=current,
        )

        schema_alerts = [a for a in result.alerts if a.metric_code == "SCHEMA_CONSISTENCY"]
        assert len(schema_alerts) >= 1


class TestMonitoringServiceResultFields:
    """验证 MonitoringRunResult 字段完整性。"""

    async def test_result_has_all_fields(self, mock_session, mock_knowledge):
        svc = MonitoringService(mock_session, mock_knowledge)
        data = _make_healthy_data(200)

        result = await svc.run(
            model_id="m1",
            champion_version="v1",
            baseline_data=data,
            current_data=data,
        )

        assert len(result.metrics) == 17  # 7 original + 6 V2 + 4 performance
        metric_codes = {m["metric_code"] for m in result.metrics}
        assert "AUC" in metric_codes
        assert "KS" in metric_codes
        assert "SAMPLE_SIZE" in metric_codes
