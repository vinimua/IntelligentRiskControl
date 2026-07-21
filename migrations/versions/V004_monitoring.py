"""V004：monitoring 首批三张表 — monitoring_runs / monitoring_metrics / monitoring_alerts"""

from collections.abc import Sequence

from alembic import op

revision: str = "V004"
down_revision: str | None = "V003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── monitoring.monitoring_runs ──
    op.execute("""
        CREATE TABLE monitoring.monitoring_runs (
            monitoring_run_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_id            VARCHAR(100) NOT NULL
                                REFERENCES model_registry.models(model_id),
            champion_version    VARCHAR(100) NOT NULL,
            baseline_window_id  VARCHAR(100) NOT NULL
                                REFERENCES model_registry.data_windows(window_id),
            current_window_id   VARCHAR(100) NOT NULL
                                REFERENCES model_registry.data_windows(window_id),
            data_track          VARCHAR(50)  NOT NULL DEFAULT 'NATURAL',
            overall_status      VARCHAR(50)  NOT NULL DEFAULT 'RUNNING',
            alert_count         INTEGER      NOT NULL DEFAULT 0,
            max_alert_severity  VARCHAR(20),
            alert_context_json  JSONB        NOT NULL DEFAULT '{}'::JSONB,
            trace_id            VARCHAR(100),
            started_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            completed_at        TIMESTAMPTZ
        )
    """)
    op.execute("""
        CREATE INDEX idx_monitoring_runs_model
        ON monitoring.monitoring_runs (model_id, started_at DESC)
    """)

    # ── monitoring.monitoring_metrics ──
    op.execute("""
        CREATE TABLE monitoring.monitoring_metrics (
            metric_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            monitoring_run_id   UUID NOT NULL
                                REFERENCES monitoring.monitoring_runs(monitoring_run_id),
            metric_code         VARCHAR(100) NOT NULL,
            metric_version      VARCHAR(50)  NOT NULL DEFAULT 'V1',
            object_type         VARCHAR(50)  NOT NULL DEFAULT 'MODEL',
            object_code         VARCHAR(100),
            baseline_value      DOUBLE PRECISION,
            current_value       DOUBLE PRECISION,
            delta               DOUBLE PRECISION,
            threshold           DOUBLE PRECISION,
            rule_type           VARCHAR(50),
            threshold_rule_id   VARCHAR(100),
            triggered           BOOLEAN      NOT NULL DEFAULT FALSE,
            availability_status VARCHAR(50)  NOT NULL DEFAULT 'AVAILABLE',
            metric_detail       JSONB        NOT NULL DEFAULT '{}'::JSONB,
            created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("""
        CREATE INDEX idx_monitoring_metrics_run
        ON monitoring.monitoring_metrics (monitoring_run_id, metric_code)
    """)

    # ── monitoring.monitoring_alerts ──
    op.execute("""
        CREATE TABLE monitoring.monitoring_alerts (
            alert_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            monitoring_run_id   UUID NOT NULL
                                REFERENCES monitoring.monitoring_runs(monitoring_run_id),
            metric_id           UUID
                                REFERENCES monitoring.monitoring_metrics(metric_id),
            alert_code          VARCHAR(100) NOT NULL,
            severity            VARCHAR(20)  NOT NULL,
            object_type         VARCHAR(50)  NOT NULL DEFAULT 'MODEL',
            object_code         VARCHAR(100),
            metric_code         VARCHAR(100) NOT NULL,
            metric_version      VARCHAR(50)  NOT NULL DEFAULT 'V1',
            baseline_value      DOUBLE PRECISION,
            current_value       DOUBLE PRECISION,
            delta               DOUBLE PRECISION,
            threshold           DOUBLE PRECISION,
            rule_type           VARCHAR(50),
            threshold_rule_id   VARCHAR(100),
            threshold_rule_version VARCHAR(20),
            availability_status VARCHAR(50)  NOT NULL DEFAULT 'AVAILABLE',
            alert_detail        JSONB        NOT NULL DEFAULT '{}'::JSONB,
            created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("""
        CREATE INDEX idx_monitoring_alerts_run
        ON monitoring.monitoring_alerts (monitoring_run_id, alert_code)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS monitoring.monitoring_alerts")
    op.execute("DROP TABLE IF EXISTS monitoring.monitoring_metrics")
    op.execute("DROP TABLE IF EXISTS monitoring.monitoring_runs")
