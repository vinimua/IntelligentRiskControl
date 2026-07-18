"""健康检查路由"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from minio import Minio

from ..config import settings

router = APIRouter(tags=["health"])

@router.get("/health/live")
async def health_live(request: Request):
    """存活检查 — 进程是否在运行。"""
    trace_id = request.state.trace_id
    return {
        "success": True,
        "code": "OK",
        "message": "alive",
        "data": {"status": "ok"},
        "trace_id": trace_id,
    }


@router.get("/health/ready")
async def health_ready(request: Request):
    """就绪检查 — 依赖服务是否可用。"""
    trace_id = request.state.trace_id
    checks: dict[str, dict] = {}

    # PostgreSQL
    try:
        import asyncpg

        conn = await asyncpg.connect(settings.asyncpg_dsn, timeout=5)
        await conn.execute("SELECT 1")
        await conn.close()
        checks["postgresql"] = {"status": "ok"}
    except Exception as e:
        checks["postgresql"] = {"status": "unavailable", "error": str(e)}

    # Redis
    try:
        import redis.asyncio as redis_asyncio

        r = redis_asyncio.from_url(settings.celery_broker_url)
        await r.ping()
        await r.aclose()
        checks["redis"] = {"status": "ok"}
    except Exception as e:
        checks["redis"] = {"status": "unavailable", "error": str(e)}

    # MinIO
    try:
        client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        client.list_buckets()
        checks["minio"] = {"status": "ok"}
    except Exception as e:
        checks["minio"] = {"status": "unavailable", "error": str(e)}

    all_ok = all(v["status"] == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "success": all_ok,
            "code": "OK" if all_ok else "SERVICE_UNAVAILABLE",
            "message": "ready" if all_ok else "some services unavailable",
            "data": {"services": checks},
            "trace_id": trace_id,
        },
    )
