"""任务三：生命周期自动触发服务。

三类触发：
- ScheduledLifecycleTriggerService   —— 定时触发，只能启动监测生命周期
- ThresholdLifecycleTriggerService   —— 阈值突破触发（监控指标越限）
- AnomalyLifecycleTriggerService     —— 异常检测触发（Sentinel 告警）

统一保证：
- 幂等键（idempotency_key = model_id|trigger_type|fingerprint）
- 冷却时间（同一模型在冷却窗口内不重复触发）
- 同模型活动生命周期去重（有未完成 run 时不触发）
- model_id 级并发锁（pg_advisory_xact_lock，事务结束自动释放）

红线：任何触发服务都只能启动"监测 → 诊断 → 决策 → 人工复核"生命周期，
禁止定时触发直接重训。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from packages.models.common.enums import TriggerType

logger = structlog.get_logger(__name__)

# 冷却窗口默认值（秒）
COOLDOWN_SCHEDULED = 24 * 3600
COOLDOWN_THRESHOLD = 6 * 3600
COOLDOWN_ANOMALY = 3600

# 未完成生命周期阶段（活动 run 判定）
_ACTIVE_PHASES = (
    "CREATED", "MONITORING", "MONITORED", "DIAGNOSIS", "DIAGNOSED",
    "ITERATION_DECISION", "DECISION_PROPOSED", "TRAINING", "TRAINED",
    "QUALIFICATION", "DEPLOYMENT", "EVENT_CLOSE", "RUNNING",
)

_TERMINAL_PHASES = ("COMPLETED", "CLOSED", "FAILED", "CANCELLED", "MANUAL_REVIEW")


class _TriggerDecision:
    __slots__ = ("allowed", "reason", "lifecycle_run_id")

    def __init__(self, allowed: bool, reason: str, lifecycle_run_id: str = ""):
        self.allowed = allowed
        self.reason = reason
        self.lifecycle_run_id = lifecycle_run_id


class LifecycleTriggerService:
    """触发服务基类：幂等 + 冷却 + 去重 + 并发锁。"""

    trigger_type: str = TriggerType.SCHEDULED_TRIGGER.value
    cooldown_seconds: int = COOLDOWN_SCHEDULED

    def __init__(self, session: AsyncSession):
        self.session = session

    @staticmethod
    def _lock_key(model_id: str) -> str:
        return f"lifecycle_trigger:{model_id}"

    @staticmethod
    def _fingerprint(model_id: str, trigger_type: str, detail: str) -> str:
        return f"{model_id}|{trigger_type}|{detail}"

    async def _acquire_model_lock(self, model_id: str) -> None:
        """model_id 级并发锁：事务级 advisory lock，commit/rollback 自动释放。"""
        await self.session.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:key))"),
            {"key": self._lock_key(model_id)},
        )

    async def _find_active_run(self, model_id: str) -> dict | None:
        result = await self.session.execute(
            text("""
                SELECT lifecycle_run_id, current_phase
                FROM workflow.model_lifecycle_runs
                WHERE model_id = :mid
                  AND (completed_at IS NULL
                       OR NOT (current_phase = ANY(:terminal)))
                ORDER BY created_at DESC
                LIMIT 1
            """),
            {"mid": model_id, "terminal": list(_TERMINAL_PHASES)},
        )
        row = result.mappings().first()
        return dict(row) if row else None

    async def _find_duplicate_fingerprint(
        self, model_id: str, fingerprint: str
    ) -> dict | None:
        result = await self.session.execute(
            text("""
                SELECT lifecycle_run_id
                FROM workflow.model_lifecycle_runs
                WHERE model_id = :mid
                  AND state_json->>'idempotency_key' = :fp
                ORDER BY created_at DESC
                LIMIT 1
            """),
            {"mid": model_id, "fp": fingerprint},
        )
        row = result.mappings().first()
        return dict(row) if row else None

    async def _within_cooldown(self, model_id: str) -> bool:
        since = datetime.now(timezone.utc) - timedelta(seconds=self.cooldown_seconds)
        result = await self.session.execute(
            text("""
                SELECT COUNT(*) AS c
                FROM workflow.model_lifecycle_runs
                WHERE model_id = :mid
                  AND trigger_type = :trigger
                  AND created_at >= :since
            """),
            {"mid": model_id, "trigger": self.trigger_type, "since": since},
        )
        count = result.scalar_one()
        return bool(count and count > 0)

    async def _start_lifecycle(
        self,
        model_id: str,
        champion_version: str,
        fingerprint: str,
        trigger_type: str | None = None,
    ) -> str:
        """创建生命周期 run 并立即写入幂等键（state_json）。

        只创建 run（当前阶段 CREATED，流程从监测节点开始），
        不在此处推进任何训练动作。
        """
        import uuid as _uuid

        from .workflow.workflow_service import WorkflowService
        from ..repositories.workflow_repo import WorkflowRepo

        repo = WorkflowRepo(self.session)
        run = await repo.create_run(
            model_id=model_id,
            champion_version=champion_version,
            trigger_type=trigger_type or self.trigger_type,
        )
        lifecycle_run_id = run["lifecycle_run_id"]
        await repo.update_phase(
            lifecycle_run_id,
            "CREATED",
            {"idempotency_key": fingerprint, "trigger_type": self.trigger_type},
        )
        await self.session.commit()

        service = WorkflowService(self.session, None)
        await service.run_existing(
            lifecycle_run_id=lifecycle_run_id,
            model_id=model_id,
            champion_version=champion_version,
            trigger_type=self.trigger_type,
        )
        return lifecycle_run_id

    async def evaluate(
        self,
        model_id: str,
        champion_version: str,
        fingerprint_detail: str,
    ) -> _TriggerDecision:
        """完整触发决策：锁 → 活动去重 → 幂等 → 冷却 → 启动。"""
        await self._acquire_model_lock(model_id)

        active = await self._find_active_run(model_id)
        if active is not None:
            reason = (
                f"active_lifecycle_exists:{active.get('lifecycle_run_id')}:"
                f"{active.get('current_phase')}"
            )
            logger.info("trigger_skipped_active_run", model_id=model_id, reason=reason)
            return _TriggerDecision(False, reason)

        fingerprint = self._fingerprint(model_id, self.trigger_type, fingerprint_detail)
        duplicate = await self._find_duplicate_fingerprint(model_id, fingerprint)
        if duplicate is not None:
            reason = f"idempotency_duplicate:{duplicate['lifecycle_run_id']}"
            logger.info("trigger_skipped_duplicate", model_id=model_id, reason=reason)
            return _TriggerDecision(False, reason)

        if await self._within_cooldown(model_id):
            logger.info(
                "trigger_skipped_cooldown",
                model_id=model_id,
                cooldown_seconds=self.cooldown_seconds,
            )
            return _TriggerDecision(False, "cooldown_active")

        lifecycle_run_id = await self._start_lifecycle(
            model_id, champion_version, fingerprint,
        )
        logger.info(
            "trigger_started",
            model_id=model_id,
            trigger_type=self.trigger_type,
            lifecycle_run_id=lifecycle_run_id,
        )
        return _TriggerDecision(True, "started", lifecycle_run_id)


class ScheduledLifecycleTriggerService(LifecycleTriggerService):
    """定时触发：只启动监测生命周期，禁止定时直接重训。

    fingerprint_detail 使用调度窗口标识（如 'D2026-08-14'），同一窗口幂等。
    """

    trigger_type = TriggerType.SCHEDULED_TRIGGER.value
    cooldown_seconds = COOLDOWN_SCHEDULED


class ThresholdLifecycleTriggerService(LifecycleTriggerService):
    """阈值突破触发：监控指标越过阈值后启动生命周期。

    fingerprint_detail 使用 monitoring_run_id，同一次监控运行只触发一次。
    """

    trigger_type = TriggerType.THRESHOLD_TRIGGER.value
    cooldown_seconds = COOLDOWN_THRESHOLD


class AnomalyLifecycleTriggerService(LifecycleTriggerService):
    """异常检测触发：Sentinel 告警后启动生命周期。

    fingerprint_detail 使用 monitoring_run_id（或 alert_run 标识），
    同一异常事件只触发一次。
    """

    trigger_type = TriggerType.ABNORMAL_TRIGGER.value
    cooldown_seconds = COOLDOWN_ANOMALY
