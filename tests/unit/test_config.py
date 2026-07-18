"""配置加载单元测试"""

import os

import pytest
from pydantic import ValidationError


def test_default_settings_load():
    """默认环境变量应能正常加载。"""
    os.environ["ENV"] = "test"
    from apps.modelops_api.config import Settings

    s = Settings()
    assert s.env == "test"
    assert s.api_port == 8000
    assert s.postgres_db == "riskitem"
    assert "localhost" in s.database_url


def test_database_url_property():
    """DATABASE_URL 属性应正确拼接。"""
    os.environ["ENV"] = "test"
    from apps.modelops_api.config import Settings

    s = Settings(postgres_host="db.example.com", postgres_port=9999, postgres_db="mydb")
    url = s.database_url
    assert "db.example.com" in url
    assert "9999" in url
    assert "mydb" in url


def test_database_url_sync_property():
    """DATABASE_URL_SYNC 不使用 asyncpg 驱动。"""
    os.environ["ENV"] = "test"
    from apps.modelops_api.config import Settings

    s = Settings()
    sync_url = s.database_url_sync
    assert "asyncpg" not in sync_url
    assert sync_url.startswith("postgresql+psycopg://")


def test_asyncpg_dsn_uses_native_postgresql_scheme():
    from apps.modelops_api.config import Settings

    s = Settings()
    assert s.asyncpg_dsn.startswith("postgresql://")
    assert "+asyncpg" not in s.asyncpg_dsn
