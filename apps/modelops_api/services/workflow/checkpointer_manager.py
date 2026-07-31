"""Checkpointer 生命周期管理器。

P2: 支持 MemorySaver (dev) 和 AsyncPostgresSaver (production)。
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.checkpoint.memory import MemorySaver

import structlog

logger = structlog.get_logger(__name__)

# Module-level singleton
_manager: CheckpointerManager | None = None


class CheckpointerManager:
    """管理 checkpointer 的创建、复用和销毁。"""

    def __init__(self, mode: str, dsn: str):
        self.mode = mode  # "memory" | "postgres"
        self.dsn = dsn    # PostgreSQL key=value connection string
        self._checkpointer = None
        self._ctx = None

    async def start(self):
        if self.mode == "memory":
            from langgraph.checkpoint.memory import MemorySaver
            self._checkpointer = MemorySaver()
            logger.info("checkpointer_memory_ready")
            return self._checkpointer

        # PostgreSQL mode
        if sys.platform == "win32":
            import asyncio
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        self._ctx = AsyncPostgresSaver.from_conn_string(self.dsn)
        self._checkpointer = await self._ctx.__aenter__()
        await self._checkpointer.setup()
        logger.info("checkpointer_postgres_ready")
        return self._checkpointer

    async def stop(self):
        if self._ctx is not None:
            await self._ctx.__aexit__(None, None, None)
            self._ctx = None
            logger.info("checkpointer_postgres_closed")
        self._checkpointer = None

    def get(self):
        return self._checkpointer


async def init_checkpointer(settings) -> BaseCheckpointSaver:
    """初始化全局 checkpointer（在 FastAPI lifespan 中调用）。"""
    global _manager

    mode = getattr(settings, "workflow_checkpointer", "memory")
    dsn = (
        f"host={settings.postgres_host} port={settings.postgres_port} "
        f"dbname={settings.postgres_db} "
        f"user={settings.postgres_user} password={settings.postgres_password}"
    )
    _manager = CheckpointerManager(mode, dsn)
    return await _manager.start()


async def shutdown_checkpointer():
    """关闭 checkpointer（在 FastAPI shutdown 中调用）。"""
    global _manager
    if _manager:
        await _manager.stop()
        _manager = None


def get_checkpointer():
    """获取当前 checkpointer 实例。"""
    global _manager
    if _manager is None:
        # 降级：MemorySaver
        from langgraph.checkpoint.memory import MemorySaver
        logger.warning("checkpointer_not_initialized_falling_back_to_memory")
        return MemorySaver()
    return _manager.get()
