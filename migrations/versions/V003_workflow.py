"""V003：workflow 首批表 + outbox/inbox + manual_review_tasks"""

from collections.abc import Sequence

from alembic import op

revision: str = "V003"
down_revision: str | None = "V002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── workflow.model_lifecycle_runs ──
    op.execute("""
        CREATE TABLE workflow.model_lifecycle_runs (
            lifecycle_run_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_id            VARCHAR(100) NOT NULL
                                REFERENCES model_registry.models(model_id),
            champion_version    VARCHAR(100) NOT NULL,
            trigger_type        VARCHAR(50)  NOT NULL DEFAULT 'SCHEDULED_TRIGGER',
            current_phase       VARCHAR(50)  NOT NULL DEFAULT 'CREATED',
            requires_manual_review BOOLEAN   NOT NULL DEFAULT FALSE,
            state_json          JSONB        NOT NULL DEFAULT '{}'::JSONB,
            created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            completed_at        TIMESTAMPTZ
        )
    """)
    op.execute("CREATE INDEX idx_lifecycle_model ON workflow.model_lifecycle_runs (model_id, created_at DESC)")

    # ── workflow.workflow_action_logs ──
    op.execute("""
        CREATE TABLE workflow.workflow_action_logs (
            action_log_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            lifecycle_run_id    UUID NOT NULL
                                REFERENCES workflow.model_lifecycle_runs(lifecycle_run_id),
            node_name           VARCHAR(100) NOT NULL,
            phase               VARCHAR(50)  NOT NULL,
            action              VARCHAR(50)  NOT NULL,
            status              VARCHAR(30)  NOT NULL DEFAULT 'COMPLETED',
            duration_ms         INTEGER,
            summary_json        JSONB        NOT NULL DEFAULT '{}'::JSONB,
            error_json          JSONB,
            created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX idx_action_logs_run ON workflow.workflow_action_logs (lifecycle_run_id, created_at)")

    # ── workflow.manual_review_tasks ──
    op.execute("""
        CREATE TABLE workflow.manual_review_tasks (
            review_task_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            lifecycle_run_id    UUID NOT NULL
                                REFERENCES workflow.model_lifecycle_runs(lifecycle_run_id),
            node_name           VARCHAR(100) NOT NULL,
            review_reason       VARCHAR(200) NOT NULL,
            status              VARCHAR(30)  NOT NULL DEFAULT 'PENDING',
            reviewer            VARCHAR(100),
            decision            VARCHAR(100),
            comment_json        JSONB        NOT NULL DEFAULT '{}'::JSONB,
            created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            resolved_at         TIMESTAMPTZ
        )
    """)

    # ── workflow.outbox_events ──
    op.execute("""
        CREATE TABLE workflow.outbox_events (
            outbox_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            lifecycle_run_id    UUID NOT NULL,
            event_type          VARCHAR(100) NOT NULL,
            payload_json        JSONB        NOT NULL DEFAULT '{}'::JSONB,
            idempotency_key     VARCHAR(255) NOT NULL,
            status              VARCHAR(30)  NOT NULL DEFAULT 'PENDING',
            retry_count         INTEGER      NOT NULL DEFAULT 0,
            max_retries         INTEGER      NOT NULL DEFAULT 5,
            created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            processed_at        TIMESTAMPTZ,
            UNIQUE (idempotency_key)
        )
    """)

    # ── workflow.inbox_events ──
    op.execute("""
        CREATE TABLE workflow.inbox_events (
            inbox_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            lifecycle_run_id    UUID NOT NULL,
            idempotency_key     VARCHAR(255) NOT NULL,
            event_type          VARCHAR(100) NOT NULL,
            payload_json        JSONB        NOT NULL DEFAULT '{}'::JSONB,
            processed           BOOLEAN      NOT NULL DEFAULT FALSE,
            created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            UNIQUE (idempotency_key)
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS workflow.inbox_events")
    op.execute("DROP TABLE IF EXISTS workflow.outbox_events")
    op.execute("DROP TABLE IF EXISTS workflow.manual_review_tasks")
    op.execute("DROP TABLE IF EXISTS workflow.workflow_action_logs")
    op.execute("DROP TABLE IF EXISTS workflow.model_lifecycle_runs")
