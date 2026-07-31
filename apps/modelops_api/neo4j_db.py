"""Neo4j 异步驱动单例与 FastAPI 依赖注入。

遵循 database.py 的模块级单例模式。
阶段 4–5：READ 模式（KnowledgeService 只读查询）。
阶段 8：WRITE 模式（校准脚本通过显式 session 参数写入）。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import structlog
from neo4j import AsyncGraphDatabase
from neo4j import AsyncDriver as Neo4jAsyncDriver
from neo4j import AsyncSession as Neo4jAsyncSession

from .config import settings

logger = structlog.get_logger(__name__)

_driver: Neo4jAsyncDriver | None = None


async def get_neo4j_driver() -> Neo4jAsyncDriver:
    """获取或创建模块级 Neo4j 异步驱动单例。"""
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_lifetime=3600,
            max_connection_pool_size=5,
            connection_acquisition_timeout=5.0,
        )
        logger.info("neo4j_driver_created", uri=settings.neo4j_uri)
    return _driver


async def close_neo4j_driver() -> None:
    """优雅关闭 Neo4j 驱动（应用关闭时调用）。"""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
        logger.info("neo4j_driver_closed")


async def get_neo4j_session() -> AsyncGenerator[Neo4jAsyncSession, None]:
    """FastAPI 依赖注入：每个请求新建一个 Neo4j READ 会话。

    阶段 4–5 强制 READ 模式，确保单次请求不会在线修改图谱。
    如需 WRITE 模式（阶段 8 校准），直接使用 get_neo4j_driver()。
    """
    driver = await get_neo4j_driver()
    async with driver.session(database="neo4j", default_access_mode="READ") as session:
        try:
            yield session
        finally:
            pass  # session 由 async with 自动关闭


async def verify_neo4j_connectivity() -> bool:
    """健康检查：验证 Neo4j 是否可达。"""
    try:
        driver = await get_neo4j_driver()
        await driver.verify_connectivity()
        return True
    except Exception:
        logger.warning("neo4j_connectivity_failed", exc_info=True)
        return False
