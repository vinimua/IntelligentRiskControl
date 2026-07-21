"""Neo4j 知识图谱集成测试 — 需要 Docker Neo4j + RUN_INFRA_TESTS=true"""

from __future__ import annotations

import os

import pytest

RUN_INFRA_TESTS = os.environ.get("RUN_INFRA_TESTS", "false").lower() == "true"
pytestmark = pytest.mark.skipif(
    not RUN_INFRA_TESTS,
    reason="需要设置 RUN_INFRA_TESTS=true 并启动 Docker Neo4j",
)


@pytest.mark.asyncio
async def test_neo4j_connectivity():
    """验证 Neo4j 驱动可连接。"""
    from apps.modelops_api.neo4j_db import verify_neo4j_connectivity

    ok = await verify_neo4j_connectivity()
    assert ok is True


@pytest.mark.asyncio
async def test_seed_script_is_idempotent():
    """验证种子脚本可重复运行，不产生重复节点。"""
    from apps.modelops_api.config import settings
    from neo4j import AsyncGraphDatabase

    from apps.modelops_api.scripts.seed_knowledge_graph import seed

    # 运行种子脚本
    count1 = await seed()

    # 再次运行种子脚本（幂等）
    count2 = await seed()

    # 两次运行实体数应相同
    assert count1 == count2

    # 验证实体数 = 7 Metric + 7 AlertType + 4 Severity = 18
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    async with driver.session(database="neo4j") as session:
        result = await session.run(
            "MATCH (n) WHERE n.namespace = 'MONITORING' AND n.enabled = true RETURN count(n) AS cnt"
        )
        record = await result.single()
        assert record is not None
        assert record["cnt"] == 18

        # 验证关系数 = 7 BREACHES_THRESHOLD + 7 HAS_SEVERITY = 14
        result2 = await session.run(
            "MATCH ()-[r]->() WHERE r.weight_version = 'seed_v1' AND r.enabled = true RETURN count(r) AS cnt"
        )
        record2 = await result2.single()
        assert record2 is not None
        assert record2["cnt"] == 14

    await driver.close()


@pytest.mark.asyncio
async def test_resolve_alert_from_neo4j():
    """验证 KnowledgeService 能从 Neo4j 正确查询 Metric→Alert 映射。"""
    from apps.modelops_api.config import settings
    from neo4j import AsyncGraphDatabase

    from apps.modelops_api.services.knowledge_service import KnowledgeService

    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    svc = KnowledgeService(driver)

    # 查询 FEATURE_PSI 的告警映射
    result = await svc.resolve_alert("FEATURE_PSI")
    assert result is not None
    assert result.alert_code == "HIGH_FEATURE_PSI"
    assert result.from_neo4j is True
    assert result.effective_weight >= 0.3  # PRODUCTION_MONITORING min weight

    # 查询未知指标应返回 None（不是异常）
    unknown = await svc.resolve_alert("NONEXISTENT")
    assert unknown is None

    await driver.close()
