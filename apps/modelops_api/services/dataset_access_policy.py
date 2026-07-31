"""DatasetAccessPolicy — 数据窗口访问控制

强制 W4/OOT 隔离规则：
- 任务一至任务三不得读取 W4/OOT 标签
- W4/OOT 只能由任务四读取
- W4/OOT 的所有训练、迭代标签权限必须为 false
- 违规访问抛出 DATASET_POLICY_VIOLATION 并写入 audit.data_access_violations
- 标签未成熟时抛出 LABEL_NOT_MATURE（对应 AvailabilityStatus.LABEL_NOT_MATURE）
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.exceptions import ConflictError, ForbiddenError
from ..repositories.audit_repo import AuditRepo

logger = structlog.get_logger(__name__)

# W4/OOT 硬约束
W4_REQUIRED_FLAGS = {
    "allows_training": False,
    "allows_iteration_label": False,
    "allows_deployment_label": True,
    "is_frozen": True,
}

TASK_LABEL_FIELD = {
    "TASK_1": "allows_monitoring_label",
    "TASK_2": "allows_diagnosis_label",
    "TASK_3": "allows_iteration_label",
    "TASK_4": "allows_deployment_label",
}


class DatasetAccessPolicy:
    """数据集访问策略 — 对应 model_registry.data_windows 权限字段。"""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.audit_repo = AuditRepo(session)

    def can_read_labels(self, window: dict, task: str) -> bool:
        """任务是否可以读取该窗口的标签。"""
        field = TASK_LABEL_FIELD.get(task)
        if field is None:
            return False
        return bool(window.get(field))

    def can_use_for_training(self, window: dict, task: str) -> bool:
        """任务是否可以用该窗口做训练。"""
        if task == "TASK_4":
            return False  # 任务四不训练
        if task == "TASK_3":
            return bool(window.get("allows_training"))
        return False  # 任务一、任务二不做训练

    async def validate_access(
        self, window: dict, task: str, usage: str = "READ_LABEL"
    ) -> None:
        """验证窗口访问权限；违规写入审计表并抛出 DATASET_POLICY_VIOLATION。"""
        if usage == "TRAINING":
            allowed = self.can_use_for_training(window, task)
            if not allowed:
                await self._violation(window, task, usage, "W4_TRAINING_VIOLATION")
                raise ForbiddenError(
                    f"任务 {task} 不允许使用窗口 {window['window_id']} 进行训练 "
                    f"(allows_training={window.get('allows_training')})",
                    code="DATASET_POLICY_VIOLATION",
                )
        elif usage == "READ_LABEL":
            allowed = self.can_read_labels(window, task)
            if not allowed:
                await self._violation(window, task, usage, "W4_LABEL_ACCESS_VIOLATION")
                raise ForbiddenError(
                    f"任务 {task} 不允许读取窗口 {window['window_id']} 的标签",
                    code="DATASET_POLICY_VIOLATION",
                )

    def validate_label_maturity(
        self,
        label_maturity_time: datetime | None,
        task: str,
        as_of: datetime | None = None,
    ) -> None:
        """校验标签是否已成熟；未成熟抛出 LABEL_NOT_MATURE。

        label_maturity_time 为 None 视为无成熟时间要求（例如无标签数据集）。
        """
        if label_maturity_time is None:
            return
        now = as_of or datetime.now(timezone.utc)
        if label_maturity_time.tzinfo is None:
            label_maturity_time = label_maturity_time.replace(tzinfo=timezone.utc)
        if now < label_maturity_time:
            raise ConflictError(
                f"任务 {task} 请求的标签尚未成熟 "
                f"(label_maturity_time={label_maturity_time.isoformat()})",
                code="LABEL_NOT_MATURE",
            )

    def validate_w4_flags(self, window: dict) -> None:
        """确认 W4/OOT 窗口的硬约束字段设置正确。"""
        if window.get("window_name", "").startswith("W4"):
            for flag, expected in W4_REQUIRED_FLAGS.items():
                actual = window.get(flag)
                if actual != expected:
                    raise ForbiddenError(
                        f"W4/OOT 窗口 {window['window_id']} 的 {flag} 必须为 {expected}，当前为 {actual}",
                        code="DATASET_POLICY_VIOLATION",
                    )

    async def _violation(
        self, window: dict, task: str, operation: str, violation_code: str
    ) -> None:
        """写入违规审计记录。"""
        try:
            await self.audit_repo.log_violation(
                task_phase=task,
                violation_code=violation_code,
                window_id=window.get("window_id"),
                attempted_operation=operation,
                detail_json={
                    "window_name": window.get("window_name"),
                    "allows_training": window.get("allows_training"),
                    "allows_iteration_label": window.get("allows_iteration_label"),
                    "is_frozen": window.get("is_frozen"),
                },
            )
            await self.session.flush()
        except Exception:
            logger.exception("failed_to_write_violation_audit")
