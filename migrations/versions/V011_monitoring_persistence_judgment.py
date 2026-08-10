"""V011：monitoring_runs 新增 persistence_judgment_json + diagnosis_status。

B1 持续性判定服务输出 trigger_diagnosis / decay_degree / persistence_evidence 等字段，
写入 monitoring_runs 持久化，供下游 graph 路由和前端展示。
"""
from collections.abc import Sequence

from alembic import op

revision: str = "V011"
down_revision: str | None = "V010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE monitoring.monitoring_runs
        ADD COLUMN IF NOT EXISTS persistence_judgment_json JSONB;

        ALTER TABLE monitoring.monitoring_runs
        ADD COLUMN IF NOT EXISTS diagnosis_status VARCHAR DEFAULT 'SKIPPED';

        COMMENT ON COLUMN monitoring.monitoring_runs.persistence_judgment_json
            IS 'B1持续性判定完整结果（trigger_diagnosis/decay_degree/persistence_evidence等）';
        COMMENT ON COLUMN monitoring.monitoring_runs.diagnosis_status
            IS '诊断状态：PENDING/IN_PROGRESS/COMPLETED/SKIPPED';
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE monitoring.monitoring_runs
        DROP COLUMN IF EXISTS persistence_judgment_json;
        ALTER TABLE monitoring.monitoring_runs
        DROP COLUMN IF EXISTS diagnosis_status;
    """)
