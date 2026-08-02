"""T4-GAP-05: 推理路由 API — 业务无感切换。

提供预测请求路由和路由状态查询接口。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.exceptions import NotFoundError, request_trace_id
from ..database import get_db

router = APIRouter(prefix="/api/inference", tags=["inference"])


class PredictRequest(BaseModel):
    request_id: str = Field(default="", min_length=0, max_length=100, description="留空则自动生成")
    features: dict = Field(default_factory=dict, description="特征字典")


def _envelope(request: Request, data, message: str = "success") -> dict:
    return {
        "success": True, "code": "OK", "message": message,
        "data": data, "trace_id": request_trace_id(request),
    }


@router.get("/{model_id}/routing-state")
async def get_routing_state(
    model_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """查询模型当前路由配置 — 哪个版本接收多少流量。"""
    from ..services.inference.inference_router_service import InferenceRouterService

    svc = InferenceRouterService(db)
    state = await svc.get_routing_state(model_id)
    return _envelope(request, state)


@router.post("/{model_id}/predict")
async def predict(
    model_id: str,
    request: Request,
    body: PredictRequest = PredictRequest(),
    db: AsyncSession = Depends(get_db),
):
    """模拟预测 — 根据请求 ID 的稳定哈希分流到 champion 或 challenger。

    同一 request_id 始终路由到相同版本（确保用户体验一致）。
    """
    from ..services.inference.inference_router_service import InferenceRouterService

    req_id = body.request_id.strip() or _generate_request_id()
    svc = InferenceRouterService(db)
    result = await svc.predict(model_id, req_id, body.features)
    return _envelope(request, result, f"routed to {result['chosen_role']}: {result['chosen_version']}")


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(min_length=1, max_length=100)


@router.post("/{model_id}/batch-predict")
async def batch_predict(
    model_id: str,
    request: Request,
    body: BatchPredictRequest,
    db: AsyncSession = Depends(get_db),
):
    """批量模拟预测 — 展示流量分流效果。

    发送 100 个请求，可以看到 challenger 实际分到的比例。
    """
    from ..services.inference.inference_router_service import InferenceRouterService

    svc = InferenceRouterService(db)
    results = []
    champion_count = 0
    challenger_count = 0

    items = body.items
    for i, item in enumerate(items):
        req_id = item.request_id.strip() or f"batch-req-{i:04d}"
        r = await svc.predict(model_id, req_id, item.features)
        results.append({
            "request_id": req_id,
            "chosen_version": r["chosen_version"],
            "chosen_role": r["chosen_role"],
            "hash_value": r["hash_value"],
            "score": r["prediction"]["score"],
            "decision": r["prediction"]["decision"],
            "artifact_uri": r["artifact"]["artifact_uri"],
        })
        if r["chosen_role"] == "CHALLENGER":
            challenger_count += 1
        else:
            champion_count += 1

    total = len(items)
    return _envelope(request, {
        "model_id": model_id,
        "total": total,
        "champion_count": champion_count,
        "challenger_count": challenger_count,
        "actual_challenger_ratio": round(challenger_count / total, 4) if total > 0 else 0,
        "results": results,
    }, f"batch routed: {champion_count} champion / {challenger_count} challenger")


def _generate_request_id() -> str:
    import uuid
    return str(uuid.uuid4())[:12]
