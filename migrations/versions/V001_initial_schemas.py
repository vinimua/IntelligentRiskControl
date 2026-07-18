"""V001：初始 Schema 与扩展

创建 10 个 Schema + pgcrypto 扩展 + 每 Schema 一个占位表。
来源：技术开发文档 V1.4.2 §12.1, §12.12
"""

from collections.abc import Sequence

from alembic import op

revision: str = "V001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMAS = [
    "model_registry",
    "workflow",
    "monitoring",
    "diagnosis",
    "iteration",
    "deployment",
    "document_store",
    "knowledge",
    "audit",
    "langgraph",
]


def upgrade() -> None:
    # pgcrypto 扩展
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    for schema in SCHEMAS:
        op.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

        # 占位表：证明 Schema 可用，后续阶段替换为正式表
        op.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema}._placeholder (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)


def downgrade() -> None:
    for schema in reversed(SCHEMAS):
        op.execute(f"DROP TABLE IF EXISTS {schema}._placeholder")
        op.execute(f"DROP SCHEMA IF EXISTS {schema}")

    op.execute("DROP EXTENSION IF EXISTS pgcrypto")
