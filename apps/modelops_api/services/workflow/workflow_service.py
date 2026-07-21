"""WorkflowService — LangGraph 生命周期管理。

职责：
- start:  创建 lifecycle_run → 推进图直到 END 或 interrupt
- resume: 从 interrupt 恢复继续推进
- get_state: 查询当前状态
- cancel: 取消运行
"""

from __future__ import annotations

import structlog
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession

from ...repositories.workflow_repo import WorkflowRepo
from packages.models.workflow.lifecycle_state import ModelLifecycleState
from .graph import build_graph

logger = structlog.get_logger(__name__)


class WorkflowService:
    def __init__(self, session: AsyncSession, checkpointer: AsyncPostgresSaver):
        self.session = session
        self.checkpointer = checkpointer
        self.graph = build_graph()

    async def start(
        self,
        model_id: str,
        champion_version: str,
        trigger_type: str = "SCHEDULED_TRIGGER",
    ) -> dict:
        """创建并推进一个新生命周期。"""
        repo = WorkflowRepo(self.session)

        # ① 先创建 DB 记录并提交（确保 run 持久化后再推进 Checkpointer）
        run = await repo.create_run(model_id, champion_version, trigger_type)
        lifecycle_run_id = run["lifecycle_run_id"]
        await self.session.commit()

        # 初始 State
        initial_state = ModelLifecycleState(
            lifecycle_run_id=lifecycle_run_id,
            model_id=model_id,
            champion_version=champion_version,
            trigger_type=trigger_type,
        )

        # ② 推进 LangGraph 图（interrupt 处自动暂停）
        config = {"configurable": {"thread_id": lifecycle_run_id}}
        compiled = self.graph.compile(checkpointer=self.checkpointer)

        final_state = None
        try:
            async for event in compiled.astream(
                initial_state.model_dump(), config, stream_mode="values"
            ):
                final_state = event
        except GraphInterrupt:
            checkpoint = await compiled.aget_state(config)
            if checkpoint and checkpoint.values:
                final_state = checkpoint.values
            logger.info("workflow_interrupted", lifecycle_run_id=lifecycle_run_id)

        # ③ 同步 state 到 DB（独立事务；失败时 Checkpointer 仍是真相源）
        if final_state:
            try:
                await repo.update_phase(
                    lifecycle_run_id,
                    final_state.get("current_phase", "CREATED"),
                    final_state,
                )
                if final_state.get("current_phase") in (
                    "NO_ALERT", "COMPLETED", "PROMOTED", "ROLLED_BACK", "FAILED"
                ):
                    await repo.complete_run(lifecycle_run_id)
                if final_state.get("requires_manual_review"):
                    await repo.set_manual_review(lifecycle_run_id)

                await self.session.commit()
            except Exception:
                await self.session.rollback()
                logger.warning(
                    "workflow_db_sync_failed",
                    lifecycle_run_id=lifecycle_run_id,
                    exc_info=True,
                )

        logger.info("workflow_started", lifecycle_run_id=lifecycle_run_id)

        return {
            "lifecycle_run_id": lifecycle_run_id,
            "current_phase": final_state.get("current_phase") if final_state else "CREATED",
            "state": final_state,
        }

    async def get_state(self, lifecycle_run_id: str) -> dict | None:
        """从 checkpointer 获取最新状态（优先），回退到 DB。"""
        config = {"configurable": {"thread_id": lifecycle_run_id}}
        compiled = self.graph.compile(checkpointer=self.checkpointer)
        checkpoint = await compiled.aget_state(config)
        if checkpoint is None or checkpoint.values is None:
            repo = WorkflowRepo(self.session)
            run = await repo.get_run(lifecycle_run_id)
            if run is None:
                return None
            return {
                "lifecycle_run_id": lifecycle_run_id,
                "current_phase": run.get("current_phase"),
                "state": run.get("state_json") or {},
            }
        return {
            "lifecycle_run_id": lifecycle_run_id,
            "current_phase": checkpoint.values.get("current_phase"),
            "state": checkpoint.values,
        }

    async def resume(self, lifecycle_run_id: str, decision: str = "approved") -> dict:
        """从 ManualReview interrupt 恢复，继续推进图。

        decision: "approved" → 图继续走向 END
                  "rejected" → 图走向 FAILED → END
        """
        repo = WorkflowRepo(self.session)

        config = {"configurable": {"thread_id": lifecycle_run_id}}
        compiled = self.graph.compile(checkpointer=self.checkpointer)

        final_state = None
        async for event in compiled.astream(
            Command(resume=decision), config, stream_mode="values"
        ):
            final_state = event

        if final_state:
            try:
                await repo.update_phase(
                    lifecycle_run_id,
                    final_state.get("current_phase", "MANUAL_REVIEW"),
                    final_state,
                )
                # 同步 requires_manual_review DB 列（approved 清除，rejected 保持）
                await repo.set_manual_review(
                    lifecycle_run_id,
                    value=bool(final_state.get("requires_manual_review")),
                )

                await self.session.commit()
            except Exception:
                await self.session.rollback()
                logger.warning(
                    "workflow_resume_db_sync_failed",
                    lifecycle_run_id=lifecycle_run_id,
                    exc_info=True,
                )

        logger.info("workflow_resumed", lifecycle_run_id=lifecycle_run_id)

        return {
            "lifecycle_run_id": lifecycle_run_id,
            "current_phase": final_state.get("current_phase") if final_state else "MANUAL_REVIEW",
            "state": final_state,
        }

    async def cancel(self, lifecycle_run_id: str) -> None:
        """取消运行。"""
        repo = WorkflowRepo(self.session)
        await repo.update_phase(lifecycle_run_id, "FAILED", {"cancelled": True})
        await repo.complete_run(lifecycle_run_id)
        await self.session.commit()
        logger.info("workflow_cancelled", lifecycle_run_id=lifecycle_run_id)
