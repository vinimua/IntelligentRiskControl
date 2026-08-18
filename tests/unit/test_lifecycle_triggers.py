"""任务三自动触发服务专项测试（A7 定稿 §8）。"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from apps.modelops_api.services.lifecycle_triggers import (
    AnomalyLifecycleTriggerService,
    ScheduledLifecycleTriggerService,
    ThresholdLifecycleTriggerService,
    _TriggerDecision,
)


def _fake_session(rows: dict = None) -> MagicMock:
    """构造假 AsyncSession：execute 按 SQL 关键字返回预设结果。"""
    rows = rows or {}
    session = MagicMock()
    session.commit = AsyncMock()

    async def _execute(stmt, params):
        sql = str(stmt)
        if "pg_advisory_xact_lock" in sql:
            return MagicMock()
        result = MagicMock()
        if "COUNT(*)" in sql:
            result.scalar_one.return_value = rows.get("count", 0)
            return result
        if "state_json->>'idempotency_key'" in sql:
            dup = rows.get("duplicate")
            result.mappings.return_value.first.return_value = dup
            return result
        if "current_phase = ANY(:terminal)" in sql:
            active = rows.get("active")
            result.mappings.return_value.first.return_value = active
            return result
        # create_run 等其它查询
        return MagicMock()

    session.execute = AsyncMock(side_effect=_execute)
    return session


def test_fingerprint_is_deterministic():
    svc = ScheduledLifecycleTriggerService(MagicMock())
    assert svc._fingerprint("m1", "SCHEDULED_TRIGGER", "D2026-08-14") == (
        "m1|SCHEDULED_TRIGGER|D2026-08-14"
    )


def test_trigger_type_and_cooldown_defaults():
    assert ScheduledLifecycleTriggerService.trigger_type == "SCHEDULED_TRIGGER"
    assert ThresholdLifecycleTriggerService.trigger_type == "THRESHOLD_TRIGGER"
    assert AnomalyLifecycleTriggerService.trigger_type == "ABNORMAL_TRIGGER"
    assert ScheduledLifecycleTriggerService.cooldown_seconds >= 24 * 3600
    assert ThresholdLifecycleTriggerService.cooldown_seconds >= 6 * 3600


@pytest.mark.asyncio
async def test_active_lifecycle_blocks_trigger():
    session = _fake_session({
        "active": {"lifecycle_run_id": "run-active", "current_phase": "TRAINING"},
    })
    decision = await ScheduledLifecycleTriggerService(session).evaluate(
        "m1", "champion_v1", "D2026-08-14",
    )
    assert decision.allowed is False
    assert decision.reason.startswith("active_lifecycle_exists:run-active")


@pytest.mark.asyncio
async def test_duplicate_fingerprint_blocks_trigger():
    session = _fake_session({
        "active": None,
        "duplicate": {"lifecycle_run_id": "run-dup"},
    })
    decision = await ThresholdLifecycleTriggerService(session).evaluate(
        "m1", "champion_v1", "mon-1",
    )
    assert decision.allowed is False
    assert decision.reason.startswith("idempotency_duplicate:run-dup")


@pytest.mark.asyncio
async def test_cooldown_blocks_trigger():
    session = _fake_session({"active": None, "duplicate": None, "count": 1})
    decision = await ScheduledLifecycleTriggerService(session).evaluate(
        "m1", "champion_v1", "D2026-08-14",
    )
    assert decision.allowed is False
    assert decision.reason == "cooldown_active"


def test_trigger_decision_shape():
    ok = _TriggerDecision(True, "started", "run-1")
    assert ok.allowed and ok.lifecycle_run_id == "run-1"
    skipped = _TriggerDecision(False, "cooldown_active")
    assert not skipped.allowed and skipped.lifecycle_run_id == ""
