"""FastAPI 主应用入口"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from .config import settings
from .core.exceptions import register_exception_handlers
from .core.logging_config import configure_logging, trace_id_var
from .routers import dashboard, diagnosis, health, knowledge, models, monitoring, workflow


class TraceIdMiddleware(BaseHTTPMiddleware):
    """为每个请求设置 trace_id 到 contextvars。"""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        trace_id = request.headers.get("X-Trace-Id", str(uuid.uuid4()))
        request.state.trace_id = trace_id
        trace_id_var.set(trace_id)
        response = await call_next(request)
        response.headers["X-Trace-Id"] = trace_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动/关闭。"""
    configure_logging(log_level=settings.log_level, log_format=settings.log_format)
    import structlog

    logger = structlog.get_logger(__name__)
    logger.info("modelops_api_starting", env=settings.env)
    yield
    from .neo4j_db import close_neo4j_driver

    await close_neo4j_driver()
    logger.info("modelops_api_stopping")


def create_app() -> FastAPI:
    app = FastAPI(
        title="ModelOps API — 信贷风控模型智能监测与自主迭代",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Trace ID
    app.add_middleware(TraceIdMiddleware)

    # Exception handlers
    register_exception_handlers(app)

    # Routers
    app.include_router(dashboard.router)
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(workflow.router)
    app.include_router(knowledge.router)
    app.include_router(diagnosis.router)
    app.include_router(monitoring.router)

    return app


app = create_app()
