"""工作流 API 路由 — 生命周期管理

契约依据：doc/前后端接口契约文档_V1.0.md §2（序号 1–7）。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, Request
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..core.exceptions import NotFoundError, request_trace_id
from ..database import get_db
from ..services.workflow.workflow_service import WorkflowService

router = APIRouter(prefix="/api/lifecycle-runs", tags=["workflow"])


class StartRunRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=100)
    champion_version: str = Field(min_length=1, max_length=100)
    trigger_type: str = "SCHEDULED_TRIGGER"


class ResumeRequest(BaseModel):
    decision: str = "approved"


@asynccontextmanager
async def _get_checkpointer() -> AsyncGenerator[AsyncPostgresSaver, None]:
    """获取 PostgreSQL checkpointer（每次请求新建连接）。"""
    saver = AsyncPostgresSaver.from_conn_string(settings.asyncpg_dsn)
    await saver.setup()
    try:
        yield saver
    finally:
        await saver.aclose()


def _envelope(request: Request, data, message: str = "success") -> dict:
    return {
        "success": True,
        "code": "OK",
        "message": message,
        "data": data,
        "trace_id": request_trace_id(request),
    }


@router.post("")
async def start_run(
    request: Request,
    body: StartRunRequest,
    db: AsyncSession = Depends(get_db),
):
    async with _get_checkpointer() as checkpointer:
        service = WorkflowService(db, checkpointer)
        result = await service.start(
            model_id=body.model_id,
            champion_version=body.champion_version,
            trigger_type=body.trigger_type,
        )
    return _envelope(request, result, message="lifecycle started")


@router.get("/{lifecycle_run_id}")
async def get_run(
    lifecycle_run_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    async with _get_checkpointer() as checkpointer:
        service = WorkflowService(db, checkpointer)
        result = await service.get_state(lifecycle_run_id)
    if not result:
        raise NotFoundError(f"生命周期 {lifecycle_run_id} 不存在")
    return _envelope(request, result)


@router.post("/{lifecycle_run_id}/resume")
async def resume_run(
    lifecycle_run_id: str,
    request: Request,
    body: ResumeRequest = ResumeRequest(),
    db: AsyncSession = Depends(get_db),
):
    async with _get_checkpointer() as checkpointer:
        service = WorkflowService(db, checkpointer)
        result = await service.resume(lifecycle_run_id, decision=body.decision)
    return _envelope(request, result, message="lifecycle resumed")


@router.post("/{lifecycle_run_id}/cancel")
async def cancel_run(
    lifecycle_run_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    async with _get_checkpointer() as checkpointer:
        service = WorkflowService(db, checkpointer)
        await service.cancel(lifecycle_run_id)
    return _envelope(request, {"lifecycle_run_id": lifecycle_run_id}, message="lifecycle cancelled")
