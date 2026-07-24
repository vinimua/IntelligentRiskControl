"""诊断 API 路由 — 任务二：四维根因诊断"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from pydantic import BaseModel, Field

from ..core.exceptions import NotFoundError, request_trace_id
from ..database import get_db
from ..neo4j_db import get_neo4j_driver
from ..repositories.diagnosis_repo import DiagnosisRepo
from ..repositories.monitoring_repo import MonitoringRepo
from ..services.knowledge_service import KnowledgeService
from ..services.diagnosis.diagnosis_service import DiagnosisService

router = APIRouter(prefix="/api/diagnosis", tags=["diagnosis"])


class TriggerDiagnosisRequest(BaseModel):
    monitoring_run_id: str = Field(min_length=1, max_length=100)
    lifecycle_run_id: str | None = None


def _envelope(request: Request, data, message: str = "success") -> dict:
    return {
        "success": True, "code": "OK", "message": message,
        "data": data, "trace_id": request_trace_id(request),
    }


@router.get("/runs")
async def list_runs(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    repo = DiagnosisRepo(db)
    runs = await repo.list_runs()
    return _envelope(request, {"items": runs})


@router.get("/runs/{diagnosis_run_id}")
async def get_run(
    diagnosis_run_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    repo = DiagnosisRepo(db)
    run = await repo.get_run(diagnosis_run_id)
    if not run:
        raise NotFoundError(f"诊断运行 {diagnosis_run_id} 不存在")
    candidates = await repo.get_candidates(diagnosis_run_id)
    evidence = await repo.get_evidence_for_run(diagnosis_run_id)
    return _envelope(request, {
        "run": run,
        "candidates": candidates,
        "evidence": evidence,
    })


@router.get("/runs/by-monitoring/{monitoring_run_id}")
async def get_diagnosis_by_monitoring(
    monitoring_run_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """查询某个监控运行对应的最新诊断结果（含候选排序 + 证据详情）。"""
    repo = DiagnosisRepo(db)
    run = await repo.get_run_by_monitoring(monitoring_run_id)
    if not run:
        raise NotFoundError(f"监控运行 {monitoring_run_id} 尚未执行诊断")
    candidates = await repo.get_candidates(run["diagnosis_run_id"])
    evidence = await repo.get_evidence_for_run(run["diagnosis_run_id"])
    return _envelope(request, {
        "run": run,
        "candidates": candidates,
        "evidence": evidence,
    })


@router.post("/trigger")
async def trigger_diagnosis(
    request: Request,
    body: TriggerDiagnosisRequest,
    db: AsyncSession = Depends(get_db),
):
    """手动触发一次诊断（用于场景注入后的根因分析）。"""
    driver = await get_neo4j_driver()
    knowledge = KnowledgeService(driver)
    repo = DiagnosisRepo(db)
    mon_repo = MonitoringRepo(db)
    service = DiagnosisService(db, knowledge, repo)

    # 加载 AlertContext
    run = await mon_repo.get_run(body.monitoring_run_id)
    if not run:
        raise NotFoundError(f"监控运行 {body.monitoring_run_id} 不存在")

    from packages.models.monitoring.alert_context import AlertContext
    alert_ctx = run.get("alert_context_json") or {}
    if isinstance(alert_ctx, str):
        import json
        alert_ctx = json.loads(alert_ctx)
    alert_context = AlertContext(**alert_ctx) if alert_ctx else AlertContext(
        schema_version="V2-WP08", trace_id="manual",
        monitoring_run_id=body.monitoring_run_id,
        model_id=run.get("model_id", "unknown"),
        model_version=run.get("champion_version", "v1"),
        monitor_window_id=run.get("current_window_id", "W3"),
        data_track="NATURAL", alert_details=[],
    )

    result = await service.diagnose(
        alert_context=alert_context,
        monitoring_run_id=body.monitoring_run_id,
        lifecycle_run_id=body.lifecycle_run_id,
    )
    await db.commit()

    return _envelope(request, {
        "diagnosis_run_id": result.diagnosis_run_id,
        "primary_root_cause_code": result.primary_root_cause_code,
        "primary_root_cause_score": result.primary_root_cause_score,
        "recommended_action": result.recommended_action.value if result.recommended_action else None,
        "need_iteration": result.need_iteration,
    })
