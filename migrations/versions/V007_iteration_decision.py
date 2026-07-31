"""V007：任务三根因驱动修复决策、训练计划与资格门。"""

from collections.abc import Sequence

from alembic import op

revision: str = "V007"
down_revision: str | None = "V006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE iteration.data_eligibility_assessments (
            assessment_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            window_id          VARCHAR(100) NOT NULL,
            status             VARCHAR(50) NOT NULL,
            supervised_training_allowed BOOLEAN NOT NULL,
            result_json        JSONB NOT NULL,
            rule_version       VARCHAR(100) NOT NULL,
            created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("""
        CREATE INDEX idx_iteration_data_gate_window
        ON iteration.data_eligibility_assessments (window_id, created_at DESC)
    """)

    op.execute("""
        CREATE TABLE iteration.decision_proposals (
            proposal_id        UUID PRIMARY KEY,
            proposal_version   INTEGER NOT NULL DEFAULT 1,
            parent_proposal_id UUID,
            lifecycle_run_id   UUID NOT NULL
                               REFERENCES workflow.model_lifecycle_runs(lifecycle_run_id),
            monitoring_run_id  UUID,
            diagnosis_run_id   VARCHAR(100) NOT NULL,
            model_id           VARCHAR(100) NOT NULL
                               REFERENCES model_registry.models(model_id),
            champion_version   VARCHAR(100) NOT NULL,
            primary_root_cause_code VARCHAR(100) NOT NULL,
            action             VARCHAR(50) NOT NULL,
            need_iteration     BOOLEAN NOT NULL,
            confidence         VARCHAR(20) NOT NULL,
            status             VARCHAR(30) NOT NULL DEFAULT 'DRAFT',
            requires_manual_review BOOLEAN NOT NULL DEFAULT FALSE,
            proposal_json      JSONB NOT NULL,
            rule_version       VARCHAR(100) NOT NULL,
            created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("""
        CREATE UNIQUE INDEX uq_iteration_proposal_version
        ON iteration.decision_proposals (proposal_id, proposal_version)
    """)
    op.execute("""
        CREATE INDEX idx_iteration_proposals_model
        ON iteration.decision_proposals (model_id, created_at DESC)
    """)

    op.execute("""
        CREATE TABLE iteration.risk_assessments (
            assessment_id      UUID PRIMARY KEY,
            proposal_id        UUID NOT NULL
                               REFERENCES iteration.decision_proposals(proposal_id),
            risk_level         VARCHAR(20) NOT NULL,
            risk_score         INTEGER NOT NULL,
            requires_manual_review BOOLEAN NOT NULL,
            assessment_json    JSONB NOT NULL,
            rule_version       VARCHAR(100) NOT NULL,
            created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    op.execute("""
        CREATE TABLE iteration.manual_review_reports (
            review_id          UUID PRIMARY KEY,
            proposal_id        UUID NOT NULL
                               REFERENCES iteration.decision_proposals(proposal_id),
            reviewer_id        VARCHAR(100) NOT NULL,
            decision           VARCHAR(30) NOT NULL,
            reason             TEXT NOT NULL,
            report_json        JSONB NOT NULL,
            reviewed_at        TIMESTAMPTZ NOT NULL,
            created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    op.execute("""
        CREATE TABLE iteration.iteration_runs (
            iteration_run_id   UUID PRIMARY KEY,
            proposal_id        UUID NOT NULL
                               REFERENCES iteration.decision_proposals(proposal_id),
            model_id           VARCHAR(100) NOT NULL
                               REFERENCES model_registry.models(model_id),
            frozen_champion_version VARCHAR(100) NOT NULL,
            current_business_round INTEGER NOT NULL DEFAULT 1,
            max_business_rounds INTEGER NOT NULL DEFAULT 3,
            status             VARCHAR(30) NOT NULL DEFAULT 'CREATED',
            exit_reason        VARCHAR(100),
            created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at       TIMESTAMPTZ
        )
    """)

    op.execute("""
        CREATE TABLE iteration.training_plans (
            training_plan_id   UUID PRIMARY KEY,
            proposal_id        UUID NOT NULL
                               REFERENCES iteration.decision_proposals(proposal_id),
            approval_id        UUID NOT NULL
                               REFERENCES iteration.manual_review_reports(review_id),
            iteration_run_id   UUID NOT NULL
                               REFERENCES iteration.iteration_runs(iteration_run_id),
            experiment_id      UUID NOT NULL UNIQUE,
            business_round     INTEGER NOT NULL CHECK (business_round BETWEEN 1 AND 3),
            strategy_code      VARCHAR(100) NOT NULL,
            status             VARCHAR(30) NOT NULL,
            plan_json          JSONB NOT NULL,
            rule_version       VARCHAR(100) NOT NULL,
            created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    op.execute("""
        CREATE TABLE iteration.iteration_rounds (
            round_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            iteration_run_id    UUID NOT NULL
                                REFERENCES iteration.iteration_runs(iteration_run_id),
            round_no            INTEGER NOT NULL CHECK (round_no BETWEEN 1 AND 3),
            strategy_code       VARCHAR(100) NOT NULL,
            experiment_id       UUID NOT NULL UNIQUE,
            status              VARCHAR(30) NOT NULL DEFAULT 'PLANNED',
            failure_report_id   UUID,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at        TIMESTAMPTZ,
            UNIQUE (iteration_run_id, round_no)
        )
    """)

    op.execute("""
        CREATE TABLE iteration.experiments (
            experiment_id       UUID PRIMARY KEY,
            iteration_run_id    UUID NOT NULL
                                REFERENCES iteration.iteration_runs(iteration_run_id),
            training_plan_id    UUID NOT NULL
                                REFERENCES iteration.training_plans(training_plan_id),
            round_no            INTEGER NOT NULL CHECK (round_no BETWEEN 1 AND 3),
            strategy_code       VARCHAR(100) NOT NULL,
            frozen_champion_version VARCHAR(100) NOT NULL,
            candidate_version   VARCHAR(100),
            technical_status    VARCHAR(30) NOT NULL DEFAULT 'PENDING',
            qualification_status VARCHAR(30) NOT NULL DEFAULT 'PENDING',
            experiment_json     JSONB NOT NULL DEFAULT '{}'::JSONB,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    op.execute("""
        CREATE TABLE iteration.training_jobs (
            training_job_id     UUID PRIMARY KEY,
            idempotency_key     VARCHAR(255) NOT NULL UNIQUE,
            iteration_run_id    UUID NOT NULL
                                REFERENCES iteration.iteration_runs(iteration_run_id),
            training_plan_id    UUID NOT NULL
                                REFERENCES iteration.training_plans(training_plan_id),
            experiment_id       UUID NOT NULL
                                REFERENCES iteration.experiments(experiment_id),
            round_no            INTEGER NOT NULL CHECK (round_no BETWEEN 1 AND 3),
            status              VARCHAR(30) NOT NULL DEFAULT 'PENDING',
            technical_retry_count INTEGER NOT NULL DEFAULT 0,
            request_json        JSONB NOT NULL,
            result_json         JSONB,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at        TIMESTAMPTZ
        )
    """)

    op.execute("""
        CREATE TABLE iteration.qualification_reports (
            qualification_run_id UUID PRIMARY KEY,
            iteration_run_id   UUID NOT NULL
                               REFERENCES iteration.iteration_runs(iteration_run_id),
            experiment_id      UUID NOT NULL,
            candidate_version  VARCHAR(100) NOT NULL,
            status             VARCHAR(30) NOT NULL,
            qualified          BOOLEAN NOT NULL,
            report_json        JSONB NOT NULL,
            rule_version       VARCHAR(100) NOT NULL,
            created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
            ,UNIQUE (experiment_id, rule_version)
        )
    """)

    op.execute("""
        CREATE TABLE iteration.qualification_checks (
            qualification_check_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            qualification_run_id UUID NOT NULL
                                 REFERENCES iteration.qualification_reports(
                                     qualification_run_id
                                 ),
            gate_code           VARCHAR(100) NOT NULL,
            gate_order          INTEGER NOT NULL,
            required            BOOLEAN NOT NULL,
            status              VARCHAR(30) NOT NULL,
            check_json          JSONB NOT NULL,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (qualification_run_id, gate_code)
        )
    """)

    op.execute("""
        CREATE TABLE iteration.failure_reports (
            failure_report_id  UUID PRIMARY KEY,
            iteration_run_id   UUID NOT NULL
                               REFERENCES iteration.iteration_runs(iteration_run_id),
            experiment_id      UUID,
            proposal_id        UUID NOT NULL
                               REFERENCES iteration.decision_proposals(proposal_id),
            failure_code       VARCHAR(100) NOT NULL,
            retryable          BOOLEAN NOT NULL DEFAULT FALSE,
            report_json        JSONB NOT NULL,
            created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    op.execute("""
        CREATE TABLE iteration.repair_case_records (
            case_id            UUID PRIMARY KEY,
            data_track         VARCHAR(30) NOT NULL,
            model_id           VARCHAR(100) NOT NULL
                               REFERENCES model_registry.models(model_id),
            diagnosis_run_id   VARCHAR(100) NOT NULL,
            proposal_id        UUID NOT NULL
                               REFERENCES iteration.decision_proposals(proposal_id),
            iteration_run_id   UUID,
            primary_root_cause_code VARCHAR(100) NOT NULL,
            action             VARCHAR(50) NOT NULL,
            outcome            VARCHAR(50) NOT NULL,
            qualified          BOOLEAN,
            failure_report_id  UUID
                               REFERENCES iteration.failure_reports(failure_report_id),
            case_json          JSONB NOT NULL,
            created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("""
        CREATE INDEX idx_iteration_cases_lookup
        ON iteration.repair_case_records
        (data_track, primary_root_cause_code, action, created_at DESC)
    """)

    op.execute("""
        CREATE TABLE iteration.data_incidents (
            data_incident_id    UUID PRIMARY KEY,
            canonical_snapshot_id VARCHAR(100) NOT NULL,
            incident_code       VARCHAR(100) NOT NULL,
            status              VARCHAR(30) NOT NULL DEFAULT 'OPEN',
            affected_model_ids  JSONB NOT NULL DEFAULT '[]'::JSONB,
            incident_json       JSONB NOT NULL,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            resolved_at         TIMESTAMPTZ
        )
    """)

    op.execute("""
        CREATE TABLE iteration.derived_data_views (
            derived_view_id     UUID PRIMARY KEY,
            data_incident_id    UUID NOT NULL
                                REFERENCES iteration.data_incidents(data_incident_id),
            canonical_snapshot_id VARCHAR(100) NOT NULL,
            derivation_rule_version VARCHAR(100) NOT NULL,
            model_id            VARCHAR(100),
            label_imputation_forbidden BOOLEAN NOT NULL DEFAULT TRUE
                                      CHECK (label_imputation_forbidden),
            view_uri            TEXT NOT NULL,
            checksum            VARCHAR(255) NOT NULL,
            evaluation_json     JSONB NOT NULL DEFAULT '{}'::JSONB,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (canonical_snapshot_id, derivation_rule_version, model_id)
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS iteration.derived_data_views")
    op.execute("DROP TABLE IF EXISTS iteration.data_incidents")
    op.execute("DROP TABLE IF EXISTS iteration.repair_case_records")
    op.execute("DROP TABLE IF EXISTS iteration.failure_reports")
    op.execute("DROP TABLE IF EXISTS iteration.qualification_checks")
    op.execute("DROP TABLE IF EXISTS iteration.qualification_reports")
    op.execute("DROP TABLE IF EXISTS iteration.training_jobs")
    op.execute("DROP TABLE IF EXISTS iteration.experiments")
    op.execute("DROP TABLE IF EXISTS iteration.iteration_rounds")
    op.execute("DROP TABLE IF EXISTS iteration.training_plans")
    op.execute("DROP TABLE IF EXISTS iteration.iteration_runs")
    op.execute("DROP TABLE IF EXISTS iteration.manual_review_reports")
    op.execute("DROP TABLE IF EXISTS iteration.risk_assessments")
    op.execute("DROP TABLE IF EXISTS iteration.decision_proposals")
    op.execute("DROP TABLE IF EXISTS iteration.data_eligibility_assessments")
