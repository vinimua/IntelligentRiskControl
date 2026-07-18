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
