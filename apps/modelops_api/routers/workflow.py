"""工作流 API 路由 — 生命周期管理

契约依据：doc/前后端接口契约文档_V1.0.md §2（序号 1–7）。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..core.exceptions import NotFoundError, request_trace_id
from ..database import async_session, get_db
from ..repositories.workflow_repo import WorkflowRepo
from ..services.workflow.workflow_service import WorkflowService
from ..services.workflow.checkpointer_manager import get_checkpointer

router = APIRouter(prefix="/api/lifecycle-runs", tags=["workflow"])
logger = structlog.get_logger(__name__)


class StartRunRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=100)
    champion_version: str = Field(min_length=1, max_length=100)
    trigger_type: str = "SCHEDULED_TRIGGER"


class ResumeRequest(BaseModel):
    decision: str = "approved"
    resume_type: str | None = None
    manual_review_id: str | None = None
    review_id: str | None = None
    plan_type: str | None = None
    plan_id: str | None = None
    repair_plan_id: str | None = None
    artifact_uri: str | None = None
    metrics: dict[str, Any] | None = None
    best_threshold: float | None = None
    training_job_id: str | None = None
    status: str | None = None
    worker_status: str | None = None
    candidate_version: str | None = None
    experiment_id: str | None = None

    def to_resume_payload(self) -> dict[str, Any] | str:
        payload = self.model_dump(exclude_none=True)
        if set(payload.keys()) == {"decision"}:
            return self.decision
        return payload


def _envelope(request: Request, data, message: str = "success") -> dict:
    return {
        "success": True,
        "code": "OK",
        "message": message,
        "data": data,
        "trace_id": request_trace_id(request),
    }


async def _advance_lifecycle_background(
    lifecycle_run_id: str,
    model_id: str,
    champion_version: str,
    trigger_type: str,
) -> None:
    async with async_session() as session:
        service = WorkflowService(session, get_checkpointer())
        try:
            await service.run_existing(
                lifecycle_run_id=lifecycle_run_id,
                model_id=model_id,
                champion_version=champion_version,
                trigger_type=trigger_type,
            )
        except Exception as exc:
            await session.rollback()
            logger.error(
                "workflow_background_advance_failed",
                lifecycle_run_id=lifecycle_run_id,
                exc_info=True,
            )
            repo = WorkflowRepo(session)
            await repo.update_phase(
                lifecycle_run_id,
                "FAILED",
                {
                    "lifecycle_run_id": lifecycle_run_id,
                    "model_id": model_id,
                    "champion_version": champion_version,
                    "trigger_type": trigger_type,
                    "current_phase": "FAILED",
                    "last_error": {
                        "reason": "workflow_background_advance_failed",
                        "message": str(exc),
                    },
                },
            )
            await repo.complete_run(lifecycle_run_id)
            await session.commit()


@router.post("")
async def start_run(
    request: Request,
    body: StartRunRequest,
    background_tasks: BackgroundTasks,
    wait: bool = Query(True, description="true=同步跑完；false=立即返回并后台推进"),
    db: AsyncSession = Depends(get_db),
):
    checkpointer = get_checkpointer()
    service = WorkflowService(db, checkpointer)
    if not wait:
        result = await service.create_run_only(
            model_id=body.model_id,
            champion_version=body.champion_version,
            trigger_type=body.trigger_type,
        )
        background_tasks.add_task(
            _advance_lifecycle_background,
            result["lifecycle_run_id"],
            body.model_id,
            body.champion_version,
            body.trigger_type,
        )
        return _envelope(request, result, message="lifecycle queued")

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
    checkpointer = get_checkpointer()
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
    checkpointer = get_checkpointer()
    service = WorkflowService(db, checkpointer)
    result = await service.resume(
            lifecycle_run_id,
            decision=body.decision,
            resume_payload=body.to_resume_payload(),
        )
    return _envelope(request, result, message="lifecycle resumed")


@router.post("/{lifecycle_run_id}/cancel")
async def cancel_run(
    lifecycle_run_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    checkpointer = get_checkpointer()
    service = WorkflowService(db, checkpointer)
    await service.cancel(lifecycle_run_id)
    return _envelope(request, {"lifecycle_run_id": lifecycle_run_id}, message="lifecycle cancelled")


# ═══════════════════════════════════════════════════════════════════════════════
# P4: 50 模型并行管控
# ═══════════════════════════════════════════════════════════════════════════════

class BatchStartRequest(BaseModel):
    models: list[dict] = Field(
        min_length=1,
        max_length=50,
        description="批量启动列表，每项包含 model_id, champion_version, trigger_type",
    )

    max_concurrency: int = Field(default=10, ge=1, le=50)


class BatchStartResult(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: list[dict]


@router.post("/batch")
async def batch_start_runs(
    request: Request,
    body: BatchStartRequest,
):
    """批量启动 lifecycle runs — 最多 50 个模型并行启动。

    每个模型独立创建 lifecycle_run，互不影响。
    """
    checkpointer = get_checkpointer()
    semaphore = asyncio.Semaphore(body.max_concurrency)

    async def start_one(entry: dict) -> dict:
        model_id = entry.get("model_id", "")
        champion_version = entry.get("champion_version", "")
        trigger_type = entry.get("trigger_type", "SCHEDULED_TRIGGER")
        try:
            async with semaphore:
                async with async_session() as session:
                    service = WorkflowService(session, checkpointer)
                    result = await service.start(
                        model_id=model_id,
                        champion_version=champion_version,
                        trigger_type=trigger_type,
                    )
            return {
                "model_id": model_id,
                "status": "started",
                "lifecycle_run_id": result.get("lifecycle_run_id"),
            }
        except Exception as exc:
            return {
                "model_id": model_id,
                "status": "failed",
                "error": str(exc),
            }

    results = await asyncio.gather(*(start_one(entry) for entry in body.models))
    succeeded = sum(1 for item in results if item["status"] == "started")
    failed = len(results) - succeeded

    return _envelope(
        request,
        {
            "total": len(body.models),
            "succeeded": succeeded,
            "failed": failed,
            "results": results,
        },
        f"batch start completed: {succeeded}/{len(body.models)} succeeded",
    )


@router.get("")
async def list_lifecycle_runs(
    request: Request,
    model_id: str | None = None,
    current_phase: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """列出 lifecycle runs — 支持按 model_id / phase 过滤。

    用于 50 模型并行管控：批量查看各模型当前阶段。
    """
    from sqlalchemy import text

    where = ["1=1"]
    params: dict = {"limit": limit, "offset": offset}

    if model_id:
        where.append("r.model_id = :mid")
        params["mid"] = model_id
    if current_phase:
        where.append("r.current_phase = :phase")
        params["phase"] = current_phase

    result = await db.execute(
        text(f"""
            SELECT r.lifecycle_run_id, r.model_id, r.champion_version,
                   r.current_phase, r.created_at, r.updated_at,
                   r.state_json AS state
            FROM workflow.model_lifecycle_runs r
            WHERE {' AND '.join(where)}
            ORDER BY r.created_at DESC
            LIMIT :limit OFFSET :offset
        """),
        params,
    )
    rows = []
    for r in result.mappings():
        d = dict(r)
        # 解析 checkpoint state
        if d.get("state"):
            try:
                d["state"] = json.loads(d["state"]) if isinstance(d["state"], str) else d["state"]
            except (json.JSONDecodeError, TypeError):
                d["state"] = None
        rows.append(d)

    count_result = await db.execute(
        text(f"""
            SELECT COUNT(*) AS cnt FROM workflow.model_lifecycle_runs r
            WHERE {' AND '.join(where)}
        """),
        {k: v for k, v in params.items() if k not in ("limit", "offset")},
    )
    total = count_result.scalar()

    return _envelope(request, {"items": rows, "total": total})
