"""诊断 API 路由 — 任务二：四维根因诊断"""

from __future__ import annotations

import json
import uuid
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.exceptions import NotFoundError, request_trace_id
from ..database import get_db
from ..neo4j_db import get_neo4j_driver
from ..repositories.diagnosis_repo import DiagnosisRepo
from ..repositories.monitoring_repo import MonitoringRepo
from ..services.knowledge_service import KnowledgeService
from ..services.diagnosis.diagnosis_service import DiagnosisService
from ..services.diagnosis.event_timeline import (
    alert_event_time,
    alerts_at_first_event_time,
    four_non_overlapping_windows,
)

router = APIRouter(prefix="/api/diagnosis", tags=["diagnosis"])

_DIAGNOSIS_ALERT_CODE_MAP = {
    "FORMAL_DISCRIMINATION_AUC": "AUC_DROP",
    "FORMAL_DISCRIMINATION_KS": "KS_DROP",
    "FORMAL_DISCRIMINATION_PR_AUC": "PR_AUC_DROP",
    "FORMAL_DISCRIMINATION_BAD_RECALL": "BAD_RECALL_DROP",
    "FORMAL_CALIBRATION_BRIER": "CALIBRATION_DEGRADE",
    "FORMAL_CALIBRATION_ECE": "CALIBRATION_DEGRADE",
    "FORMAL_FEATURE_PSI": "HIGH_FEATURE_PSI",
    "FORMAL_SCORE_PSI": "HIGH_SCORE_PSI",
    "FORMAL_DATA_QUALITY_MISSING_RATE": "MISSING_RATE_SPIKE",
}


def _json_object(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _event_payload(event: dict | None) -> dict | None:
    if not event:
        return None
    payload = dict(event)
    event_time = payload.get("event_time")
    if event_time is not None:
        if getattr(event_time, "tzinfo", None) is None:
            event_time = event_time.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
        else:
            event_time = event_time.astimezone(ZoneInfo("Asia/Shanghai"))
        payload["event_time"] = event_time.isoformat()
    return payload


def _build_alert_context(run: dict, alert_rows: list[dict]):
    """Adapt monitoring output variants to Task-2's strict AlertContext contract."""
    from packages.models.monitoring.alert_context import AlertContext, AlertDetail

    raw_context = _json_object(run.get("alert_context_json"))
    if raw_context:
        try:
            return AlertContext.model_validate(raw_context)
        except ValidationError:
            # Sentinel/replay monitoring stores a decision summary rather than the
            # Task-2 AlertContext contract. Rebuild it from persisted alert rows.
            pass

    alert_details = []
    for row in alert_rows:
        source_alert_code = row["alert_code"]
        alert_code = _DIAGNOSIS_ALERT_CODE_MAP.get(
            source_alert_code, source_alert_code
        )
        metric_detail = _json_object(row.get("alert_detail"))
        if alert_code != source_alert_code:
            metric_detail = {
                **metric_detail,
                "source_alert_code": source_alert_code,
            }

        alert_details.append(
            AlertDetail(
                alert_id=str(row["alert_id"]),
                alert_code=alert_code,
                severity=row.get("severity") or "WARNING",
                object_type=row.get("object_type") or "MODEL",
                object_code=row.get("object_code") or run.get("model_id", "unknown"),
                metric_code=row.get("metric_code") or "",
                metric_version=row.get("metric_version") or "V1",
                baseline_value=row.get("baseline_value"),
                current_value=row.get("current_value"),
                delta=row.get("delta"),
                threshold=row.get("threshold"),
                rule_type=row.get("rule_type"),
                threshold_rule_id=row.get("threshold_rule_id"),
                threshold_rule_version=row.get("threshold_rule_version"),
                availability_status=row.get("availability_status") or "AVAILABLE",
                metric_detail=metric_detail,
                created_at=row.get("created_at"),
            )
        )

    return AlertContext(
        schema_version=str(raw_context.get("schema_version") or "V2-WP08"),
        trace_id=str(
            raw_context.get("trace_id")
            or run.get("trace_id")
            or uuid.uuid4()
        ),
        monitoring_run_id=str(run["monitoring_run_id"]),
        model_id=str(run.get("model_id") or raw_context.get("model_id") or "unknown"),
        model_version=str(
            run.get("champion_version")
            or raw_context.get("model_version")
            or "unknown"
        ),
        monitor_window_id=str(
            raw_context.get("monitor_window_id")
            or run.get("current_window_id")
            or "W3"
        ),
        baseline_id=str(
            raw_context.get("baseline_id")
            or run.get("baseline_window_id")
            or "W1"
        ),
        data_track=run.get("data_track") or raw_context.get("data_track") or "NATURAL",
        scenario_id=raw_context.get("scenario_id"),
        anomaly_probability=raw_context.get("anomaly_probability"),
        top_signals=list(raw_context.get("top_signals") or []),
        alert_details=alert_details,
    )


def _diagnosis_source(run: dict, alert_rows: list[dict]) -> dict:
    canonical_codes = [
        _DIAGNOSIS_ALERT_CODE_MAP.get(row["alert_code"], row["alert_code"])
        for row in alert_rows
    ]
    deltas = [
        float(row["delta"])
        for row in alert_rows
        if row.get("delta") is not None
    ]
    return {
        "monitoring_run_id": str(run["monitoring_run_id"]),
        "model_id": str(run.get("model_id") or ""),
        "model_version": str(run.get("champion_version") or ""),
        "source_alert_codes": sorted({row["alert_code"] for row in alert_rows}),
        "diagnosis_alert_codes": sorted(set(canonical_codes)),
        "alert_count": len(alert_rows),
        "largest_drop": min(deltas) if deltas else None,
    }


def _supporting_documents_from_evidence(evidence: list[dict]) -> list[dict]:
    docs: list[dict] = []
    seen: set[str] = set()
    for row in evidence:
        if row.get("method_code") != "rag_supporting_document_search":
            continue
        detail = _json_object(row.get("evidence_detail_json"))
        for doc in detail.get("documents") or []:
            if not isinstance(doc, dict):
                continue
            chunk_id = str(doc.get("chunk_id") or "")
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            docs.append(doc)
    return docs


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


@router.get("/events/{event_id}/agent-handoff")
async def get_agent_handoff(
    event_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Read-only Agent handoff contract. Delegates to DiagnosisHandoffService."""
    from ..services.workflow.agent_handoff_service import DiagnosisHandoffService

    handoff_svc = DiagnosisHandoffService(db)
    handoff = await handoff_svc.build_handoff(event_id)
    return _envelope(request, handoff)


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
        "supporting_documents": _supporting_documents_from_evidence(evidence),
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
    event = (
        await repo.get_event(str(run["event_id"]))
        if run.get("event_id")
        else None
    )
    mon_repo = MonitoringRepo(db)
    monitoring_run = await mon_repo.get_run(monitoring_run_id)
    alerts = await mon_repo.get_alerts(monitoring_run_id)
    if event:
        event_alert_ids = set(
            await repo.get_event_alert_ids(str(event["event_id"]))
        )
        alerts = [
            alert for alert in alerts
            if str(alert["alert_id"]) in event_alert_ids
        ]
    return _envelope(request, {
        "run": run,
        "candidates": candidates,
        "evidence": evidence,
        "supporting_documents": _supporting_documents_from_evidence(evidence),
        "event": _event_payload(event),
        "agent_handoff": {
            "next_stage": "AGENT_DECISION",
            "agent_connected": False,
            "handoff_status": "READY_NOT_DISPATCHED",
        } if event else None,
        "source": _diagnosis_source(monitoring_run, alerts)
        if monitoring_run
        else None,
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

    alerts = await mon_repo.get_alerts(body.monitoring_run_id)
    active_event = await repo.get_active_event(
        str(run["model_id"]), str(run["champion_version"])
    )
    if active_event and active_event["status"] != "OPEN":
        existing = await repo.get_run_by_event(str(active_event["event_id"]))
        if existing:
            return _envelope(request, {
                "diagnosis_run_id": str(existing["diagnosis_run_id"]),
                "event_id": str(active_event["event_id"]),
                "event_status": active_event["status"],
                "message": "当前事件尚未由后续决策/修复流程关闭，未创建重复诊断",
            })

    if active_event:
        event_alert_ids = set(
            await repo.get_event_alert_ids(str(active_event["event_id"]))
        )
        event_alerts = [
            alert for alert in alerts if str(alert["alert_id"]) in event_alert_ids
        ]
        event_id = str(active_event["event_id"])
        event_time = active_event["event_time"]
    else:
        unassigned = await mon_repo.get_unassigned_alerts(body.monitoring_run_id)
        event_alerts = alerts_at_first_event_time(unassigned)
        if not event_alerts:
            raise NotFoundError("没有尚未处理的告警事件")
        event_time = alert_event_time(event_alerts[0])
        created = await repo.create_event(
            model_id=str(run["model_id"]),
            model_version=str(run["champion_version"]),
            monitoring_run_id=body.monitoring_run_id,
            event_time=event_time,
            alert_ids=[str(alert["alert_id"]) for alert in event_alerts],
        )
        event_id = str(created["event_id"])

    window_days = int(
        (_json_object(event_alerts[0].get("alert_detail")).get("window_days") or 7)
    )
    evidence_windows = four_non_overlapping_windows(event_time, window_days)
    evidence_window_ids = [window["window_id"] for window in evidence_windows]
    alert_context = _build_alert_context(run, event_alerts)

    result = await service.diagnose(
        alert_context=alert_context,
        monitoring_run_id=body.monitoring_run_id,
        lifecycle_run_id=body.lifecycle_run_id,
        event_id=event_id,
        evidence_window_ids=evidence_window_ids,
    )
    await repo.mark_event_diagnosed(event_id)
    await db.commit()

    return _envelope(request, {
        "diagnosis_run_id": result.diagnosis_run_id,
        "event_id": event_id,
        "event_time": event_time,
        "evidence_window_ids": evidence_window_ids,
        "primary_root_cause_code": result.primary_root_cause_code,
        "primary_root_cause_score": result.primary_root_cause_score,
        "recommended_action": result.recommended_action.value if result.recommended_action else None,
        "need_iteration": result.need_iteration,
        "supporting_documents": [
            doc.model_dump(mode="json") for doc in result.supporting_documents
        ],
    })
