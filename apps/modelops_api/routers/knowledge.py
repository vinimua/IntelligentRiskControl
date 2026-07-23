"""知识图谱查询路由 — 阶段 4 调试/只读端点

供 MonitoringService 开发和调试使用。
生产环境可禁用或限制为内部访问。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from neo4j import AsyncDriver as Neo4jAsyncDriver

from ..core.exceptions import NotFoundError, request_trace_id
from ..neo4j_db import get_neo4j_driver
from ..services.knowledge_service import KnowledgeService

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


def _envelope(request: Request, data, message: str = "success") -> dict:
    return {
        "success": True,
        "code": "OK",
        "message": message,
        "data": data,
        "trace_id": request_trace_id(request),
    }


@router.get("/resolve-alert")
async def resolve_alert(
    request: Request,
    metric_code: str = Query(..., description="指标代码，如 FEATURE_PSI"),
    driver: Neo4jAsyncDriver = Depends(get_neo4j_driver),
):
    """查询 Metric → Alert 映射。"""
    svc = KnowledgeService(driver)
    result = await svc.resolve_alert(metric_code)
    if not result:
        raise NotFoundError(f"指标 {metric_code} 没有告警映射")
    return _envelope(
        request,
        {
            "alert_code": result.alert_code,
            "metric_code": result.metric_code,
            "severity": result.severity.value,
            "effective_weight": result.effective_weight,
            "description": result.description,
            "from_neo4j": result.from_neo4j,
        },
    )


@router.get("/entities/{entity_code}")
async def get_entity(
    request: Request,
    entity_code: str,
    driver: Neo4jAsyncDriver = Depends(get_neo4j_driver),
):
    """查询单个知识实体。"""
    svc = KnowledgeService(driver)
    result = await svc.get_entity(entity_code)
    if not result:
        raise NotFoundError(f"实体 {entity_code} 不存在")
    return _envelope(request, result.model_dump())
