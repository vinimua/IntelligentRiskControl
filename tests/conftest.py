"""共享测试工具与 fixtures"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """创建 FastAPI 测试客户端。"""
    os.environ.setdefault("ENV", "test")
    os.environ.setdefault("SKIP_INTEGRATION", "true")

    from apps.modelops_api.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def test_settings():
    """覆盖环境变量用于测试。"""
    os.environ["ENV"] = "test"
    os.environ["POSTGRES_DB"] = "riskitem_test"
    from apps.modelops_api.config import Settings

    return Settings()
