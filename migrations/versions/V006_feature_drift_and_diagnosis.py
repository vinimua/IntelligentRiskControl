"""V006：per-feature drift 持久化 + diagnosis 表"""

from collections.abc import Sequence

from alembic import op

revision: str = "V006"
down_revision: str | None = "V005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── 1. Per-feature drift + quality（监控运行后持久化） ──
    op.execute("""
        CREATE TABLE monitoring.monitoring_feature_drift (
            drift_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            monitoring_run_id   UUID NOT NULL
                                REFERENCES monitoring.monitoring_runs(monitoring_run_id),
            window_id           VARCHAR(100) NOT NULL,
            feature_name        VARCHAR(200) NOT NULL,
            feature_type        VARCHAR(50) NOT NULL DEFAULT 'continuous',
            -- drift metrics
            psi                 DOUBLE PRECISION,
            js_divergence       DOUBLE PRECISION,
            wasserstein_distance DOUBLE PRECISION,
            ks_statistic        DOUBLE PRECISION,
            ks_p_value          DOUBLE PRECISION,
            ks_q_value          DOUBLE PRECISION,
            -- quality metrics
            missing_rate        DOUBLE PRECISION,
            missing_rate_delta  DOUBLE PRECISION,
            outlier_rate        DOUBLE PRECISION,
            outlier_rate_delta  DOUBLE PRECISION,
            default_value_rate  DOUBLE PRECISION,
            range_violation_rate DOUBLE PRECISION,
            unknown_category_rate DOUBLE PRECISION,
            dq_score            DOUBLE PRECISION,
            dq_flag             VARCHAR(10),
            -- metadata
            data_track          VARCHAR(50) NOT NULL DEFAULT 'NATURAL',
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX idx_feature_drift_run
            ON monitoring.monitoring_feature_drift (monitoring_run_id, window_id);
        CREATE INDEX idx_feature_drift_feature
            ON monitoring.monitoring_feature_drift (feature_name, window_id);
    """)

    # ── 2. Diagnosis schema + 表 ──
    op.execute("CREATE SCHEMA IF NOT EXISTS diagnosis;")

    op.execute("""
        CREATE TABLE diagnosis.diagnosis_runs (
            diagnosis_run_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            lifecycle_run_id            UUID,
            monitoring_run_id           UUID NOT NULL
                                        REFERENCES monitoring.monitoring_runs(monitoring_run_id),
            alert_count                 INTEGER NOT NULL DEFAULT 0,
            primary_root_cause_code     VARCHAR(100),
            primary_root_cause_dimension VARCHAR(50),
            primary_root_cause_score    DOUBLE PRECISION,
            recommended_action          VARCHAR(100),
            need_iteration              BOOLEAN,
            status                      VARCHAR(50) NOT NULL DEFAULT 'RUNNING',
            context_pack_id             VARCHAR(100),
            created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at                TIMESTAMPTZ
        );
    """)

    op.execute("""
        CREATE TABLE diagnosis.diagnosis_candidates (
            candidate_id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            diagnosis_run_id                UUID NOT NULL
                                            REFERENCES diagnosis.diagnosis_runs(diagnosis_run_id),
            alert_code                      VARCHAR(100) NOT NULL,
            root_cause_code                 VARCHAR(100) NOT NULL,
            dimension_code                  VARCHAR(50) NOT NULL,
            relation_key                    VARCHAR(500) NOT NULL,
            effective_weight_snapshot       DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            evidence_case_count_snapshot    INTEGER NOT NULL DEFAULT 0,
            confidence_lower_bound_snapshot DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            ranked_score                    DOUBLE PRECISION,
            rank_no                         INTEGER,
            is_primary                      BOOLEAN NOT NULL DEFAULT FALSE,
            created_at                      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)

    op.execute("""
        CREATE TABLE diagnosis.diagnosis_evidence (
            evidence_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            diagnosis_run_id        UUID NOT NULL
                                    REFERENCES diagnosis.diagnosis_runs(diagnosis_run_id),
            candidate_id            UUID NOT NULL
                                    REFERENCES diagnosis.diagnosis_candidates(candidate_id),
            hypothesis_code         VARCHAR(100),
            evidence_type           VARCHAR(10) NOT NULL,
            method_code             VARCHAR(100) NOT NULL,
            normalized_score        DOUBLE PRECISION,
            direction               VARCHAR(20),
            p_value                 DOUBLE PRECISION,
            applicable              BOOLEAN NOT NULL DEFAULT TRUE,
            availability_status     VARCHAR(50) NOT NULL DEFAULT 'AVAILABLE',
            evidence_detail_json    JSONB NOT NULL DEFAULT '{}'::JSONB,
            created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS diagnosis.diagnosis_evidence;")
    op.execute("DROP TABLE IF EXISTS diagnosis.diagnosis_candidates;")
    op.execute("DROP TABLE IF EXISTS diagnosis.diagnosis_runs;")
    op.execute("DROP TABLE IF EXISTS monitoring.monitoring_feature_drift;")
    op.execute("DROP SCHEMA IF EXISTS diagnosis;")
