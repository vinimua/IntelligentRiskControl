from __future__ import annotations

import io
import os
import time
import uuid

import pytest


RUN_INFRA_TESTS = os.environ.get("RUN_INFRA_TESTS", "false").lower() == "true"
pytestmark = pytest.mark.skipif(
    not RUN_INFRA_TESTS,
    reason="需要设置 RUN_INFRA_TESTS=true 并启动 Docker 基础设施",
)


@pytest.mark.asyncio
async def test_redis_ping():
    import redis.asyncio as redis_asyncio

    from apps.modelops_api.config import settings

    client = redis_asyncio.from_url(settings.celery_broker_url)
    try:
        assert await client.ping() is True
    finally:
        await client.aclose()


def test_minio_upload_and_read():
    from minio import Minio

    from apps.modelops_api.config import settings

    client = Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )
    object_name = f"acceptance/{uuid.uuid4()}.txt"
    payload = b"stage-1-acceptance"
    client.put_object(
        settings.minio_bucket,
        object_name,
        io.BytesIO(payload),
        length=len(payload),
        content_type="text/plain",
    )
    response = client.get_object(settings.minio_bucket, object_name)
    try:
        assert response.read() == payload
    finally:
        response.close()
        response.release_conn()
        client.remove_object(settings.minio_bucket, object_name)


@pytest.mark.asyncio
async def test_mlflow_create_run():
    import httpx

    from apps.modelops_api.config import settings

    experiment_name = f"stage1-acceptance-{uuid.uuid4()}"
    async with httpx.AsyncClient(base_url=settings.mlflow_tracking_uri, timeout=10) as client:
        experiment = await client.post(
            "/api/2.0/mlflow/experiments/create", json={"name": experiment_name}
        )
        experiment.raise_for_status()
        experiment_id = experiment.json()["experiment_id"]
        run = await client.post(
            "/api/2.0/mlflow/runs/create",
            json={"experiment_id": experiment_id, "start_time": int(time.time() * 1000)},
        )
        run.raise_for_status()
        assert run.json()["run"]["info"]["run_id"]


@pytest.mark.asyncio
async def test_neo4j_merge_and_query():
    import httpx

    from apps.modelops_api.config import settings

    marker = str(uuid.uuid4())
    payload = {
        "statements": [
            {
                "statement": (
                    "MERGE (n:Stage1Acceptance {marker: $marker}) "
                    "RETURN n.marker AS marker"
                ),
                "parameters": {"marker": marker},
            }
        ]
    }
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            "http://localhost:7474/db/neo4j/tx/commit",
            auth=(settings.neo4j_user, settings.neo4j_password),
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
        assert body["errors"] == []
        assert body["results"][0]["data"][0]["row"][0] == marker
        await client.post(
            "http://localhost:7474/db/neo4j/tx/commit",
            auth=(settings.neo4j_user, settings.neo4j_password),
            json={
                "statements": [
                    {
                        "statement": "MATCH (n:Stage1Acceptance {marker: $marker}) DELETE n",
                        "parameters": {"marker": marker},
                    }
                ]
            },
        )


@pytest.mark.asyncio
async def test_qdrant_create_and_delete_collection():
    import httpx

    from apps.modelops_api.config import settings

    collection = f"stage1_acceptance_{uuid.uuid4().hex}"
    base_url = f"http://{settings.qdrant_host}:{settings.qdrant_port}"
    async with httpx.AsyncClient(base_url=base_url, timeout=10) as client:
        create = await client.put(
            f"/collections/{collection}",
            json={"vectors": {"size": 4, "distance": "Cosine"}},
        )
        create.raise_for_status()
        assert create.json()["result"] is True
        delete = await client.delete(f"/collections/{collection}")
        delete.raise_for_status()
        assert delete.json()["result"] is True


def test_celery_worker_executes_test_task():
    from workers.app import test_task

    result = test_task.apply_async(args=["acceptance"])
    payload = result.get(timeout=15)
    assert payload["status"] == "ok"
    assert payload["msg"] == "acceptance"
