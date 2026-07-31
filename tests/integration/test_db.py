"""数据库连接集成测试"""

import os
import subprocess
import sys

import pytest


RUN_INFRA_TESTS = os.environ.get("RUN_INFRA_TESTS", "false").lower() == "true"


def test_alembic_config_exists():
    """alembic.ini 文件存在。"""
    assert os.path.exists("alembic.ini")


def test_migrations_directory_exists():
    """migrations 目录和版本文件存在。"""
    assert os.path.isdir("migrations")
    assert os.path.isdir("migrations/versions")
    assert os.path.exists("migrations/versions/V001_initial_schemas.py")


@pytest.mark.skipif(
    not RUN_INFRA_TESTS,
    reason="需要设置 RUN_INFRA_TESTS=true 并启动 Docker 基础设施",
)
@pytest.mark.asyncio
async def test_database_connectivity():
    """数据库可连接（需要 Docker）。"""
    import asyncpg

    from apps.modelops_api.config import settings

    conn = await asyncpg.connect(settings.asyncpg_dsn, timeout=5)
    result = await conn.fetchval("SELECT 1")
    await conn.close()
    assert result == 1


@pytest.mark.skipif(
    not RUN_INFRA_TESTS,
    reason="需要设置 RUN_INFRA_TESTS=true 并启动 Docker 基础设施",
)
def test_migration_upgrade_rollback_and_reupgrade():
    """在专用验收数据库中验证升级、回滚和重新升级。"""
    import psycopg
    from psycopg import sql

    from apps.modelops_api.config import settings

    database_name = "riskitem_stage1_acceptance"
    admin_dsn = (
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/postgres"
    )
    migration_url = (
        f"postgresql+psycopg://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{database_name}"
    )

    def recreate_database() -> None:
        with psycopg.connect(admin_dsn, autocommit=True) as connection:
            connection.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (database_name,),
            )
            connection.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(database_name)))
            connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))

    recreate_database()
    env = {**os.environ, "DATABASE_URL_SYNC": migration_url}
    try:
        for command in (("upgrade", "head"), ("downgrade", "base"), ("upgrade", "head")):
            result = subprocess.run(
                [sys.executable, "-m", "alembic", *command],
                cwd=os.getcwd(),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            assert result.returncode == 0, result.stderr
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as connection:
            connection.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (database_name,),
            )
            connection.execute(
                sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(database_name))
            )
