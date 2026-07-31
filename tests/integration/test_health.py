"""健康检查集成测试"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_live(async_client: AsyncClient):
    """GET /health/live 返回 200。"""
    response = await async_client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["status"] == "ok"
    assert "trace_id" in data
    assert data["trace_id"] == response.headers["X-Trace-Id"]


@pytest.mark.asyncio
async def test_health_ready_returns_200_when_dependencies_are_available(
    async_client: AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    """GET /health/ready 在依赖可用时返回 200。"""
    import asyncpg
    import redis.asyncio as redis_asyncio

    from apps.modelops_api.routers import health as health_module

    class FakePostgresConnection:
        async def execute(self, statement: str):
            assert statement == "SELECT 1"

        async def close(self):
            return None

    async def fake_connect(dsn: str, timeout: int):
        assert dsn.startswith("postgresql://")
        return FakePostgresConnection()

    class FakeRedis:
        async def ping(self):
            return True

        async def aclose(self):
            return None

    class FakeMinio:
        def __init__(self, *args, **kwargs):
            pass

        def list_buckets(self):
            return []

    monkeypatch.setattr(asyncpg, "connect", fake_connect)
    monkeypatch.setattr(redis_asyncio, "from_url", lambda *args, **kwargs: FakeRedis())
    monkeypatch.setattr(health_module, "Minio", FakeMinio)

    response = await async_client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["trace_id"] == response.headers["X-Trace-Id"]
    assert "services" in data["data"]
    assert all(
        service["status"] == "ok" for service in data["data"]["services"].values()
    )


@pytest.mark.asyncio
async def test_health_ready_returns_503_when_postgresql_is_unavailable(
    async_client: AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    import asyncpg
    import redis.asyncio as redis_asyncio

    from apps.modelops_api.routers import health as health_module

    async def failed_connect(*args, **kwargs):
        raise ConnectionError("postgres unavailable")

    class FakeRedis:
        async def ping(self):
            return True

        async def aclose(self):
            return None

    class FakeMinio:
        def __init__(self, *args, **kwargs):
            pass

        def list_buckets(self):
            return []

    monkeypatch.setattr(asyncpg, "connect", failed_connect)
    monkeypatch.setattr(redis_asyncio, "from_url", lambda *args, **kwargs: FakeRedis())
    monkeypatch.setattr(health_module, "Minio", FakeMinio)

    response = await async_client.get("/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["success"] is False
    assert body["data"]["services"]["postgresql"]["status"] == "unavailable"
