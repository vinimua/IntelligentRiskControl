"""Alembic 迁移环境配置

从 DATABASE_URL 环境变量读取数据库连接。
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

load_dotenv()

# Alembic Config 对象
config = context.config

# 日志配置
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 从环境变量覆盖 sqlalchemy.url
database_url = os.getenv(
    "DATABASE_URL_SYNC",
    "postgresql+psycopg://riskitem:riskitem_dev@localhost:5432/riskitem",
)
config.set_main_option("sqlalchemy.url", database_url)

# 目标元数据 — V1 阶段不自动生成，使用显式 migration
target_metadata = None


def run_migrations_offline() -> None:
    """离线模式：生成 SQL 脚本。"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式：连接数据库执行。"""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
