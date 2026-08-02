"""模型注册 API 路由

契约依据：doc/前后端接口契约文档_V1.0.md §8（序号 50–52）。
POST 两端点为开发/内部端点，已在 contracts/api_inventory.yaml 登记为 internal。
"""

from __future__ import annotations

import math

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.exceptions import NotFoundError, request_trace_id
from ..database import get_db
from ..repositories.data_window_repo import DataWindowRepo
from ..services.model_registry_service import ModelRegistryService

router = APIRouter(prefix="/api/models", tags=["models"])


class RegisterModelRequest(BaseModel):
    """POST /api/models 请求体。"""

    model_id: str = Field(min_length=1, max_length=100)
    model_name: str = Field(min_length=1, max_length=255)
    model_type: str = "CREDIT_RISK"


class RegisterVersionRequest(BaseModel):
    """POST /api/models/{model_id}/versions 请求体。"""

    version_code: str = Field(min_length=1, max_length=100)
    role: str = "CHALLENGER"


def _envelope(request: Request, data, message: str = "success") -> dict:
    return {
        "success": True,
        "code": "OK",
        "message": message,
        "data": data,
        "trace_id": request_trace_id(request),
    }


def _window_to_contract(row: dict) -> dict:
    """DB 行（start_time/end_time TIMESTAMPTZ）→ 契约 §8.3 形状（start_date/end_date DATE）。"""
    return {
        "window_id": row["window_id"],
        "window_name": row["window_name"],
        "start_date": row["start_time"].date().isoformat() if row.get("start_time") else None,
        "end_date": row["end_time"].date().isoformat() if row.get("end_time") else None,
        "allows_training": row["allows_training"],
        "allows_monitoring_label": row["allows_monitoring_label"],
        "allows_diagnosis_label": row["allows_diagnosis_label"],
        "allows_iteration_label": row["allows_iteration_label"],
        "allows_deployment_label": row["allows_deployment_label"],
        "is_frozen": row["is_frozen"],
    }


@router.get("")
async def list_models(
    request: Request,
    status: str | None = Query(default=None, pattern="^(ACTIVE|INACTIVE|RETIRED)$"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    service = ModelRegistryService(db)
    total = await service.repo.count_models(status=status)
    models = await service.repo.list_models(
        status=status, limit=page_size, offset=(page - 1) * page_size
    )
    return _envelope(
        request,
        {
            "items": models,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": math.ceil(total / page_size) if total else 0,
            },
        },
    )


@router.post("")
async def register_model(
    request: Request,
    body: RegisterModelRequest,
    db: AsyncSession = Depends(get_db),
):
    service = ModelRegistryService(db)
    result = await service.register_model(
        model_id=body.model_id,
        model_name=body.model_name,
        model_type=body.model_type,
    )
    return _envelope(request, result, message="model registered")


@router.get("/{model_id}")
async def get_model(
    model_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    service = ModelRegistryService(db)
    result = await service.get_model_detail(model_id)
    if not result:
        raise NotFoundError(f"模型 {model_id} 不存在")
    return _envelope(request, result)


@router.post("/{model_id}/versions")
async def register_version(
    model_id: str,
    request: Request,
    body: RegisterVersionRequest,
    db: AsyncSession = Depends(get_db),
):
    service = ModelRegistryService(db)
    result = await service.register_version(
        model_id=model_id,
        version_code=body.version_code,
        role=body.role,
    )
    return _envelope(request, result, message="version registered")


@router.get("/{model_id}/data-windows")
async def list_data_windows(
    model_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    repo = DataWindowRepo(db)
    windows = await repo.list_windows()
    return _envelope(
        request,
        {"model_id": model_id, "windows": [_window_to_contract(w) for w in windows]},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# T4-GAP-02: 灰度路由配置 API
# ═══════════════════════════════════════════════════════════════════════════════

class RoutingSwitchRequest(BaseModel):
    """手动切换路由配置。"""
    challenger_traffic_ratio: float = Field(ge=0.0, le=1.0, description="challenger 流量比例 0-1")
    challenger_version_code: str | None = None
    environment: str = "PROD"
    updated_by: str = "admin"


@router.get("/{model_id}/routing")
async def get_model_routing(
    model_id: str,
    request: Request,
    environment: str = "PROD",
    db: AsyncSession = Depends(get_db),
):
    """T4-GAP-02: 查询模型当前路由配置 — 灰度发布核心查询。

    返回 champion/challenger 版本和流量分配比例。
    """
    from ..repositories.iteration_repo import IterationRepo

    repo = IterationRepo(db)
    state = await repo.get_model_deployment_state(model_id, environment)

    if not state:
        return _envelope(request, {
            "model_id": model_id,
            "environment": environment,
            "active_version_code": None,
            "stable_version_code": None,
            "challenger_version_code": None,
            "challenger_traffic_ratio": 0.0,
            "message": "尚未部署 — 无路由配置",
        })

    return _envelope(request, {
        "model_id": model_id,
        "environment": environment,
        **{k: str(v) if not isinstance(v, (int, float, type(None))) else v
           for k, v in state.items()},
    })


@router.post("/{model_id}/routing/switch")
async def switch_model_routing(
    model_id: str,
    request: Request,
    body: RoutingSwitchRequest,
    db: AsyncSession = Depends(get_db),
):
    """T4-GAP-02: 手动切换路由 — 灰度比例调整。

    直接更新 model_deployment_state 的 challenger_traffic_ratio。
    0.0 = 全部 champion，1.0 = 全部 challenger（晋升）。
    """
    from ..repositories.iteration_repo import IterationRepo

    repo = IterationRepo(db)
    current = await repo.get_model_deployment_state(model_id, body.environment)

    active_version = current.get("active_version_code") if current else None
    stable_version = current.get("stable_version_code") if current else active_version
    challenger_version = (
        body.challenger_version_code
        or (current.get("challenger_version_code") if current else None)
    )
    ratio = body.challenger_traffic_ratio

    if ratio >= 1.0 and challenger_version:
        next_active = challenger_version
        next_stable = active_version or stable_version
        next_challenger = None
        next_ratio = 0.0
        action = "promoted_to_champion"
    elif ratio <= 0.0:
        next_active = stable_version or active_version
        next_stable = stable_version or active_version
        next_challenger = None
        next_ratio = 0.0
        action = "rolled_back_to_champion"
    else:
        next_active = active_version
        next_stable = stable_version
        next_challenger = challenger_version
        next_ratio = ratio
        action = f"traffic_ratio_set_to_{ratio}"

    record = {
        "model_id": model_id,
        "environment": body.environment,
        "active_version_code": next_active,
        "stable_version_code": next_stable,
        "challenger_version_code": next_challenger,
        "challenger_traffic_ratio": next_ratio,
        "state_version": (current.get("state_version", 0) + 1) if current else 1,
        "updated_by": body.updated_by,
    }

    await repo.upsert_model_deployment_state(record)
    await db.commit()

    return _envelope(request, {
        "model_id": model_id,
        "action": action,
        "active_version_code": next_active,
        "stable_version_code": next_stable,
        "challenger_version_code": next_challenger,
        "challenger_traffic_ratio": next_ratio,
    }, f"routing switched: {action}")
