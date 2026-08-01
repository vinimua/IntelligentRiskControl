"""V009：kg_sync_jobs — KG 权重批量同步任务。

每次校准完成后创建一个 sync job，记录哪些 snapshot 被写入 Neo4j。
"""
from collections.abc import Sequence

from alembic import op

revision: str = "V010"
down_revision: str | None = "V009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE knowledge.kg_sync_jobs (
            sync_job_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            calibration_run_id      UUID NOT NULL,
            idempotency_key         VARCHAR(500) NOT NULL UNIQUE,
            relation_type           VARCHAR(100) NOT NULL DEFAULT 'INDICATES',
            status                  VARCHAR(50) NOT NULL DEFAULT 'PENDING',
            snapshot_count          INTEGER NOT NULL DEFAULT 0 CHECK (snapshot_count >= 0),
            applied_count           INTEGER NOT NULL DEFAULT 0 CHECK (applied_count >= 0),
            error_message           TEXT,
            weight_version          VARCHAR(100) NOT NULL,
            applied_to_neo4j        BOOLEAN NOT NULL DEFAULT FALSE,
            neo4j_applied_at        TIMESTAMPTZ,
            started_at              TIMESTAMPTZ,
            completed_at            TIMESTAMPTZ,
            created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX idx_kg_sync_jobs_status
            ON knowledge.kg_sync_jobs (status, created_at DESC);
        CREATE INDEX idx_kg_sync_jobs_calibration
            ON knowledge.kg_sync_jobs (calibration_run_id, relation_type);
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS knowledge.kg_sync_jobs")
