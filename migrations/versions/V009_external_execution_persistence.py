"""V009: P4/P5 external execution persistence."""

from collections.abc import Sequence

from alembic import op

revision: str = "V009"
down_revision: str | None = "V008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE iteration.external_execution_plans (
            execution_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            lifecycle_run_id  UUID REFERENCES workflow.model_lifecycle_runs(lifecycle_run_id),
            plan_type         VARCHAR(30) NOT NULL,
            plan_id           UUID NOT NULL UNIQUE,
            action            VARCHAR(50),
            status            VARCHAR(30) NOT NULL DEFAULT 'PLANNED',
            dispatch_mode     VARCHAR(30) NOT NULL DEFAULT 'INTERNAL',
            external_task_id  VARCHAR(255),
            callback_endpoint TEXT,
            request_json      JSONB NOT NULL,
            result_json       JSONB,
            error_message     TEXT,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at      TIMESTAMPTZ
        )
    """)
    op.execute("""
        CREATE INDEX idx_external_execution_lifecycle
        ON iteration.external_execution_plans (lifecycle_run_id, created_at DESC)
    """)
    op.execute("""
        CREATE INDEX idx_external_execution_type_status
        ON iteration.external_execution_plans (plan_type, status, created_at DESC)
    """)

    op.execute("""
        CREATE TABLE iteration.deployment_records (
            deployment_id        UUID PRIMARY KEY,
            lifecycle_run_id     UUID REFERENCES workflow.model_lifecycle_runs(lifecycle_run_id),
            qualification_run_id UUID,
            model_id             VARCHAR(100),
            champion_version     VARCHAR(100),
            candidate_version    VARCHAR(100),
            current_stage        VARCHAR(50) NOT NULL,
            decision             VARCHAR(50) NOT NULL,
            status               VARCHAR(30) NOT NULL,
            dispatch_mode        VARCHAR(30) NOT NULL DEFAULT 'INTERNAL',
            external_task_id     VARCHAR(255),
            record_json          JSONB NOT NULL,
            created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at         TIMESTAMPTZ
        )
    """)
    op.execute("""
        CREATE INDEX idx_deployment_records_lifecycle
        ON iteration.deployment_records (lifecycle_run_id, created_at DESC)
    """)

    op.execute("""
        CREATE TABLE iteration.deployment_stage_records (
            stage_record_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            deployment_id   UUID NOT NULL REFERENCES iteration.deployment_records(deployment_id),
            stage           VARCHAR(50) NOT NULL,
            decision        VARCHAR(50) NOT NULL,
            status          VARCHAR(30) NOT NULL,
            health_json     JSONB NOT NULL DEFAULT '{}'::JSONB,
            result_json     JSONB NOT NULL DEFAULT '{}'::JSONB,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("""
        CREATE INDEX idx_deployment_stage_records_deployment
        ON iteration.deployment_stage_records (deployment_id, created_at)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS iteration.deployment_stage_records")
    op.execute("DROP TABLE IF EXISTS iteration.deployment_records")
    op.execute("DROP TABLE IF EXISTS iteration.external_execution_plans")
