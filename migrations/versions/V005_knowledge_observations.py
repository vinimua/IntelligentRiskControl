"""V005：knowledge 关系观测与校准表"""

from collections.abc import Sequence

from alembic import op

revision: str = "V005"
down_revision: str | None = "V004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Evidence ledger: one row per support/against/neutral observation.
    op.execute("""
        CREATE TABLE knowledge.kg_relation_observations (
            observation_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            relation_key        VARCHAR(500) NOT NULL,
            source_domain       VARCHAR(100) NOT NULL,
            source_record_id    VARCHAR(500) NOT NULL,
            lifecycle_run_id    UUID,
            direction           VARCHAR(20) NOT NULL
                                CHECK (direction IN ('SUPPORT', 'AGAINST', 'NEUTRAL')),
            evidence_score      DOUBLE PRECISION
                                CHECK (evidence_score IS NULL OR evidence_score BETWEEN 0.0 AND 1.0),
            quality_weight      DOUBLE PRECISION NOT NULL DEFAULT 1.0
                                CHECK (quality_weight >= 0.0),
            weighted_strength   DOUBLE PRECISION,
            data_track          VARCHAR(50) NOT NULL DEFAULT 'NATURAL'
                                CHECK (data_track IN ('NATURAL', 'SCENARIO')),
            evidence_detail     JSONB NOT NULL DEFAULT '{}'::JSONB,
            observed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (relation_key, source_domain, source_record_id)
        )
    """)
    op.execute("""
        CREATE INDEX idx_kg_relation_observations_relation_track
        ON knowledge.kg_relation_observations (relation_key, data_track, observed_at DESC)
    """)
    op.execute("""
        CREATE INDEX idx_kg_relation_observations_lifecycle
        ON knowledge.kg_relation_observations (lifecycle_run_id, source_domain)
    """)

    # One row per offline calibration job.
    op.execute("""
        CREATE TABLE knowledge.kg_calibration_runs (
            calibration_run_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            data_track               VARCHAR(50) NOT NULL
                                     CHECK (data_track IN ('NATURAL', 'SCENARIO')),
            observed_from            TIMESTAMPTZ,
            observed_to              TIMESTAMPTZ,
            calibration_rule_version VARCHAR(100) NOT NULL,
            target_weight_version    VARCHAR(100) NOT NULL,
            status                   VARCHAR(50) NOT NULL DEFAULT 'PENDING',
            relation_count           INTEGER NOT NULL DEFAULT 0 CHECK (relation_count >= 0),
            observation_count        INTEGER NOT NULL DEFAULT 0 CHECK (observation_count >= 0),
            error_message            TEXT,
            started_at               TIMESTAMPTZ,
            completed_at             TIMESTAMPTZ,
            created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("""
        CREATE INDEX idx_kg_calibration_runs_status
        ON knowledge.kg_calibration_runs (status, created_at DESC)
    """)

    # Per-relation output from a calibration run.
    op.execute("""
        CREATE TABLE knowledge.kg_relation_weight_snapshots (
            snapshot_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            calibration_run_id       UUID NOT NULL
                                      REFERENCES knowledge.kg_calibration_runs(calibration_run_id)
                                      ON DELETE CASCADE,
            relation_key             VARCHAR(500) NOT NULL,
            old_effective_weight     DOUBLE PRECISION,
            new_effective_weight     DOUBLE PRECISION NOT NULL
                                      CHECK (new_effective_weight BETWEEN 0.0 AND 1.0),
            confidence_lower_bound   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            confidence_upper_bound   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            evidence_case_count      INTEGER NOT NULL DEFAULT 0,
            natural_case_count       INTEGER NOT NULL DEFAULT 0,
            scenario_case_count      INTEGER NOT NULL DEFAULT 0,
            support_count            INTEGER NOT NULL DEFAULT 0,
            against_count            INTEGER NOT NULL DEFAULT 0,
            neutral_count            INTEGER NOT NULL DEFAULT 0,
            support_strength         DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            against_strength         DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            weight_version           VARCHAR(100) NOT NULL,
            applied_to_neo4j         BOOLEAN NOT NULL DEFAULT FALSE,
            snapshot_detail          JSONB NOT NULL DEFAULT '{}'::JSONB,
            created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (calibration_run_id, relation_key)
        )
    """)
    op.execute("""
        CREATE INDEX idx_kg_relation_weight_snapshots_relation
        ON knowledge.kg_relation_weight_snapshots (relation_key, created_at DESC)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS knowledge.kg_relation_weight_snapshots")
    op.execute("DROP TABLE IF EXISTS knowledge.kg_calibration_runs")
    op.execute("DROP TABLE IF EXISTS knowledge.kg_relation_observations")
