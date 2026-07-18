from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from apps.modelops_api.core.exceptions import register_exception_handlers
from apps.modelops_api.main import TraceIdMiddleware


@pytest.mark.asyncio
async def test_request_validation_error_uses_standard_error_response():
    app = FastAPI()
    app.add_middleware(TraceIdMiddleware)
    register_exception_handlers(app)

    @app.get("/values/{value}")
    async def get_value(value: int):
        return {"value": value}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/values/not-an-integer")

    body = response.json()
    assert response.status_code == 422
    assert body["code"] == "VALIDATION_ERROR"
    assert body["trace_id"] == response.headers["X-Trace-Id"]
    assert body["details"]["errors"]
