"""V002：model_registry 首批表 + audit.data_access_violations

创建 7 张正式表，删除 V001 占位表。
来源：PostgreSQL DDL V1.1 §model_registry, §audit
"""

from collections.abc import Sequence

from alembic import op

revision: str = "V002"
down_revision: str | None = "V001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── 删除占位表 ──
    for schema in (
        "model_registry", "workflow", "monitoring", "diagnosis",
        "iteration", "deployment", "document_store", "knowledge",
        "audit", "langgraph",
    ):
        op.execute(f"DROP TABLE IF EXISTS {schema}._placeholder")

    # ── model_registry.models ──
    op.execute("""
        CREATE TABLE model_registry.models (
            model_id        VARCHAR(100) PRIMARY KEY,
            model_name      VARCHAR(255) NOT NULL,
            model_type      VARCHAR(50)  NOT NULL DEFAULT 'CREDIT_RISK',
            target_name     VARCHAR(100),
            owner           VARCHAR(100),
            business_line   VARCHAR(100),
            description     TEXT,
            status          VARCHAR(30)  NOT NULL DEFAULT 'ACTIVE'
                            CHECK (status IN ('ACTIVE', 'INACTIVE', 'RETIRED')),
            current_champion_version VARCHAR(100),
            stable_version  VARCHAR(100),
            attributes_json JSONB        NOT NULL DEFAULT '{}'::JSONB,
            created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("""
        CREATE OR REPLACE FUNCTION model_registry.tg_models_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """)
    op.execute("""
        CREATE TRIGGER trg_models_updated_at
        BEFORE UPDATE ON model_registry.models
        FOR EACH ROW EXECUTE FUNCTION model_registry.tg_models_updated_at()
    """)

    # ── model_registry.feature_schemas ──
    op.execute("""
        CREATE TABLE model_registry.feature_schemas (
            feature_schema_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_id            VARCHAR(100) NOT NULL
                                REFERENCES model_registry.models(model_id),
            schema_version      VARCHAR(100) NOT NULL,
            feature_schema_json JSONB        NOT NULL,
            content_hash        VARCHAR(128) NOT NULL,
            status              VARCHAR(30)  NOT NULL DEFAULT 'ACTIVE'
                                CHECK (status IN ('ACTIVE', 'SUPERSEDED', 'RETIRED')),
            created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            UNIQUE (model_id, schema_version)
        )
    """)

    # ── model_registry.data_windows ──
    op.execute("""
        CREATE TABLE model_registry.data_windows (
            window_id                VARCHAR(100) PRIMARY KEY,
            window_name              VARCHAR(255) NOT NULL,
            start_time               TIMESTAMPTZ  NOT NULL,
            end_time                 TIMESTAMPTZ  NOT NULL,
            purpose                  VARCHAR(50)  NOT NULL,
            allows_training          BOOLEAN      NOT NULL DEFAULT FALSE,
            allows_monitoring_label  BOOLEAN      NOT NULL DEFAULT FALSE,
            allows_diagnosis_label   BOOLEAN      NOT NULL DEFAULT FALSE,
            allows_iteration_label   BOOLEAN      NOT NULL DEFAULT FALSE,
            allows_deployment_label  BOOLEAN      NOT NULL DEFAULT FALSE,
            is_frozen                BOOLEAN      NOT NULL DEFAULT FALSE,
            attributes_json          JSONB        NOT NULL DEFAULT '{}'::JSONB,
            created_at               TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            CHECK (end_time > start_time)
        )
    """)

    # ── model_registry.dataset_snapshots ──
    op.execute("""
        CREATE TABLE model_registry.dataset_snapshots (
            dataset_snapshot_id     UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
            model_id                VARCHAR(100) REFERENCES model_registry.models(model_id),
            window_id               VARCHAR(100) REFERENCES model_registry.data_windows(window_id),
            data_track              VARCHAR(20)  NOT NULL DEFAULT 'NATURAL'
                                    CHECK (data_track IN ('NATURAL', 'SCENARIO')),
            scenario_id             VARCHAR(100),
            storage_uri             TEXT         NOT NULL,
            content_hash            VARCHAR(128) NOT NULL,
            row_count               BIGINT,
            column_count            INTEGER,
            feature_schema_version  VARCHAR(100),
            label_maturity_time     TIMESTAMPTZ,
            metadata_json           JSONB        NOT NULL DEFAULT '{}'::JSONB,
            created_at              TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            UNIQUE (content_hash)
        )
    """)

    # ── model_registry.model_versions ──
    op.execute("""
        CREATE TABLE model_registry.model_versions (
            model_version_id        UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
            model_id                VARCHAR(100) NOT NULL
                                    REFERENCES model_registry.models(model_id),
            version_code            VARCHAR(100) NOT NULL,
            role                    VARCHAR(30)  NOT NULL DEFAULT 'CHALLENGER'
                                    CHECK (role IN ('CHAMPION', 'CHALLENGER', 'STABLE', 'ARCHIVED')),
            status                  VARCHAR(30)  NOT NULL DEFAULT 'REGISTERED'
                                    CHECK (status IN ('REGISTERED', 'VALIDATED', 'DEPLOYED', 'REJECTED', 'ARCHIVED')),
            base_version_code       VARCHAR(100),
            feature_schema_version  VARCHAR(100),
            training_snapshot_id    UUID         REFERENCES model_registry.dataset_snapshots(dataset_snapshot_id),
            mlflow_run_id           VARCHAR(255),
            artifact_uri            TEXT,
            code_version            VARCHAR(100),
            random_seed             INTEGER,
            metrics_json            JSONB        NOT NULL DEFAULT '{}'::JSONB,
            governance_json         JSONB        NOT NULL DEFAULT '{}'::JSONB,
            created_by              VARCHAR(100),
            created_at              TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            updated_at              TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            UNIQUE (model_id, version_code)
        )
    """)
    op.execute("""
        CREATE INDEX idx_model_versions_model_created
        ON model_registry.model_versions (model_id, created_at DESC)
    """)
    op.execute("""
        CREATE OR REPLACE FUNCTION model_registry.tg_model_versions_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """)
    op.execute("""
        CREATE TRIGGER trg_model_versions_updated_at
        BEFORE UPDATE ON model_registry.model_versions
        FOR EACH ROW EXECUTE FUNCTION model_registry.tg_model_versions_updated_at()
    """)

    # ── model_registry.model_deployment_state ──
    op.execute("""
        CREATE TABLE model_registry.model_deployment_state (
            deployment_state_id      UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
            model_id                 VARCHAR(100) NOT NULL
                                     REFERENCES model_registry.models(model_id),
            environment              VARCHAR(30)  NOT NULL DEFAULT 'TEST'
                                     CHECK (environment IN ('DEV', 'TEST', 'STAGING', 'PROD')),
            active_version_code      VARCHAR(100),
            stable_version_code      VARCHAR(100),
            challenger_version_code  VARCHAR(100),
            challenger_traffic_ratio NUMERIC(6,5) NOT NULL DEFAULT 0
                                     CHECK (challenger_traffic_ratio BETWEEN 0 AND 1),
            state_version            BIGINT       NOT NULL DEFAULT 1,
            updated_by               VARCHAR(100),
            updated_at               TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            UNIQUE (model_id, environment)
        )
    """)

    # ── audit.data_access_violations ──
    op.execute("""
        CREATE TABLE audit.data_access_violations (
            violation_id         UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
            lifecycle_run_id     UUID,
            model_id             VARCHAR(100),
            task_phase           VARCHAR(50)  NOT NULL,
            dataset_snapshot_id  UUID         REFERENCES model_registry.dataset_snapshots(dataset_snapshot_id),
            window_id            VARCHAR(100),
            violation_code       VARCHAR(100) NOT NULL,
            attempted_operation  VARCHAR(100),
            detail_json          JSONB        NOT NULL DEFAULT '{}'::JSONB,
            created_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW()
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS audit.data_access_violations")
    op.execute("DROP TABLE IF EXISTS model_registry.model_deployment_state")
    op.execute("DROP TABLE IF EXISTS model_registry.model_versions")
    op.execute("DROP TABLE IF EXISTS model_registry.dataset_snapshots")
    op.execute("DROP TABLE IF EXISTS model_registry.data_windows")
    op.execute("DROP TABLE IF EXISTS model_registry.feature_schemas")
    op.execute("DROP TABLE IF EXISTS model_registry.models")
    op.execute("DROP FUNCTION IF EXISTS model_registry.tg_models_updated_at() CASCADE")
    op.execute("DROP FUNCTION IF EXISTS model_registry.tg_model_versions_updated_at() CASCADE")

    # 恢复占位表
    for schema in (
        "model_registry", "workflow", "monitoring", "diagnosis",
        "iteration", "deployment", "document_store", "knowledge",
        "audit", "langgraph",
    ):
        op.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema}._placeholder (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
