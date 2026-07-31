"""V008: chronological diagnosis events and versioned diagnosis results."""

from collections.abc import Sequence

from alembic import op

revision: str = "V008"
down_revision: str | None = "V007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE diagnosis.diagnosis_events (
            event_id UUID PRIMARY KEY,
            model_id VARCHAR(100) NOT NULL,
            model_version VARCHAR(100) NOT NULL,
            monitoring_run_id UUID NOT NULL
                REFERENCES monitoring.monitoring_runs(monitoring_run_id),
            event_time TIMESTAMPTZ NOT NULL,
            status VARCHAR(30) NOT NULL DEFAULT 'OPEN',
            primary_alert_id UUID NOT NULL
                REFERENCES monitoring.monitoring_alerts(alert_id),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            closed_at TIMESTAMPTZ,
            UNIQUE (model_id, model_version, primary_alert_id)
        );
        CREATE INDEX idx_diagnosis_event_timeline
            ON diagnosis.diagnosis_events
            (model_id, model_version, event_time, created_at);

        CREATE TABLE diagnosis.diagnosis_event_alerts (
            event_id UUID NOT NULL
                REFERENCES diagnosis.diagnosis_events(event_id),
            alert_id UUID NOT NULL
                REFERENCES monitoring.monitoring_alerts(alert_id),
            attached_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (event_id, alert_id),
            UNIQUE (alert_id)
        );

        ALTER TABLE diagnosis.diagnosis_runs
            ADD COLUMN event_id UUID
                REFERENCES diagnosis.diagnosis_events(event_id),
            ADD COLUMN logic_version VARCHAR(50) NOT NULL DEFAULT 'V1_LEGACY';

        UPDATE diagnosis.diagnosis_runs
        SET status = 'LEGACY_INVALID',
            logic_version = 'V1_LEGACY'
        WHERE event_id IS NULL;

        CREATE INDEX idx_diagnosis_runs_event
            ON diagnosis.diagnosis_runs(event_id, created_at DESC);
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS diagnosis.idx_diagnosis_runs_event")
    op.execute(
        """
        ALTER TABLE diagnosis.diagnosis_runs
            DROP COLUMN IF EXISTS logic_version,
            DROP COLUMN IF EXISTS event_id;
        DROP TABLE IF EXISTS diagnosis.diagnosis_event_alerts;
        DROP TABLE IF EXISTS diagnosis.diagnosis_events;
        """
    )
