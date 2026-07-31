"""任务三 PostgreSQL 数据访问。"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from packages.models.iteration import (
    DataEligibilityResult,
    DecisionProposal,
    FailureReport,
    ManualReviewReport,
    QualificationReport,
    RepairCaseRecord,
    RiskAssessment,
    TrainingPlan,
)
from packages.models.callbacks.training_callback import TrainingCallback
from packages.models.iteration.training_job import TrainingJobInput


def _json(model: Any) -> str:
    return json.dumps(model.model_dump(mode="json"), ensure_ascii=False)


class IterationRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_data_eligibility(
        self, assessment_id: str, result: DataEligibilityResult
    ) -> None:
        await self.session.execute(
            text("""
                INSERT INTO iteration.data_eligibility_assessments
                    (assessment_id, window_id, status,
                     supervised_training_allowed, result_json, rule_version)
                VALUES (:id, :window, :status, :allowed, :payload, :version)
            """),
            {
                "id": assessment_id,
                "window": result.window_id,
                "status": result.status.value,
                "allowed": result.supervised_training_allowed,
                "payload": _json(result),
                "version": result.rule_version,
            },
        )

    async def get_data_eligibility_assessments(
        self, assessment_ids: list[str]
    ) -> list[tuple[str, DataEligibilityResult]]:
        if not assessment_ids:
            return []
        result = await self.session.execute(
            text("""
                SELECT assessment_id, result_json
                FROM iteration.data_eligibility_assessments
                WHERE assessment_id = ANY(CAST(:ids AS UUID[]))
            """),
            {"ids": assessment_ids},
        )
        return [
            (
                str(row["assessment_id"]),
                DataEligibilityResult.model_validate(row["result_json"]),
            )
            for row in result.mappings()
        ]

    async def save_proposal(self, proposal: DecisionProposal) -> None:
        await self.session.execute(
            text("""
                INSERT INTO iteration.decision_proposals
                    (proposal_id, proposal_version, parent_proposal_id,
                     lifecycle_run_id, monitoring_run_id, diagnosis_run_id, model_id,
                     champion_version, primary_root_cause_code, action,
                     need_iteration, confidence, status, requires_manual_review,
                     proposal_json, rule_version)
                VALUES (:id, :proposal_version, :parent_proposal,
                        :lifecycle, :monitoring, :diagnosis, :model, :champion, :root,
                        :action, :need, :confidence, :status, :review, :payload,
                        :version)
            """),
            {
                "id": proposal.proposal_id,
                "proposal_version": proposal.proposal_version,
                "parent_proposal": proposal.parent_proposal_id,
                "lifecycle": proposal.lifecycle_run_id,
                "monitoring": proposal.monitoring_run_id,
                "diagnosis": proposal.diagnosis_run_id,
                "model": proposal.model_id,
                "champion": proposal.champion_version,
                "root": proposal.primary_root_cause_code,
                "action": proposal.action.value,
                "need": proposal.need_iteration,
                "confidence": proposal.confidence.value,
                "status": proposal.status.value,
                "review": proposal.requires_manual_review,
                "payload": _json(proposal),
                "version": proposal.rule_version,
            },
        )

    async def save_risk(self, risk: RiskAssessment) -> None:
        await self.session.execute(
            text("""
                INSERT INTO iteration.risk_assessments
                    (assessment_id, proposal_id, risk_level, risk_score,
                     requires_manual_review, assessment_json, rule_version)
                VALUES (:id, :proposal, :level, :score, :review, :payload, :version)
            """),
            {
                "id": risk.assessment_id,
                "proposal": risk.proposal_id,
                "level": risk.risk_level.value,
                "score": risk.risk_score,
                "review": risk.requires_manual_review,
                "payload": _json(risk),
                "version": risk.rule_version,
            },
        )

    async def get_risk_for_proposal(self, proposal_id: str) -> dict | None:
        result = await self.session.execute(
            text("""
                SELECT assessment_json
                FROM iteration.risk_assessments
                WHERE proposal_id = :id
                ORDER BY created_at DESC
                LIMIT 1
            """),
            {"id": proposal_id},
        )
        return result.scalar_one_or_none()

    async def get_proposal(self, proposal_id: str) -> DecisionProposal | None:
        result = await self.session.execute(
            text("""
                SELECT proposal_json
                FROM iteration.decision_proposals
                WHERE proposal_id = :id
            """),
            {"id": proposal_id},
        )
        payload = result.scalar_one_or_none()
        return DecisionProposal.model_validate(payload) if payload else None

    async def list_proposals(
        self, model_id: str | None = None, limit: int = 20
    ) -> list[dict]:
        sql = """
            SELECT proposal_id, lifecycle_run_id, diagnosis_run_id, model_id,
                   champion_version, primary_root_cause_code, action,
                   need_iteration, confidence, status, requires_manual_review,
                   rule_version, created_at, updated_at
            FROM iteration.decision_proposals
        """
        params: dict = {"limit": limit}
        if model_id:
            sql += " WHERE model_id = :model_id"
            params["model_id"] = model_id
        sql += " ORDER BY created_at DESC LIMIT :limit"
        result = await self.session.execute(text(sql), params)
        return [dict(row) for row in result.mappings()]

    async def save_review(self, report: ManualReviewReport) -> None:
        await self.session.execute(
            text("""
                INSERT INTO iteration.manual_review_reports
                    (review_id, proposal_id, reviewer_id, decision, reason,
                     report_json, reviewed_at)
                VALUES (:id, :proposal, :reviewer, :decision, :reason,
                        :payload, :reviewed_at)
            """),
            {
                "id": report.review_id,
                "proposal": report.proposal_id,
                "reviewer": report.reviewer_id,
                "decision": report.decision.value,
                "reason": report.reason,
                "payload": _json(report),
                "reviewed_at": report.reviewed_at,
            },
        )
        status = "APPROVED" if report.decision.value == "APPROVE" else "REJECTED"
        await self.session.execute(
            text("""
                UPDATE iteration.decision_proposals
                SET status = :status,
                    proposal_json = jsonb_set(
                        proposal_json, '{status}', to_jsonb(CAST(:status_json AS TEXT))
                    ),
                    updated_at = NOW()
                WHERE proposal_id = :proposal
            """),
            {"status": status, "status_json": status, "proposal": report.proposal_id},
        )

    async def get_approved_review(
        self, review_id: str, proposal_id: str
    ) -> ManualReviewReport | None:
        result = await self.session.execute(
            text("""
                SELECT report_json
                FROM iteration.manual_review_reports
                WHERE review_id = :id
                  AND proposal_id = :proposal
                  AND decision = 'APPROVE'
            """),
            {"id": review_id, "proposal": proposal_id},
        )
        payload = result.scalar_one_or_none()
        return ManualReviewReport.model_validate(payload) if payload else None

    async def create_iteration_run(
        self,
        iteration_run_id: str,
        proposal: DecisionProposal,
        max_business_rounds: int,
    ) -> None:
        await self.session.execute(
            text("""
                INSERT INTO iteration.iteration_runs
                    (iteration_run_id, proposal_id, model_id,
                     frozen_champion_version, max_business_rounds)
                VALUES (:id, :proposal, :model, :champion, :rounds)
            """),
            {
                "id": iteration_run_id,
                "proposal": proposal.proposal_id,
                "model": proposal.model_id,
                "champion": proposal.champion_version,
                "rounds": max_business_rounds,
            },
        )

    async def save_training_plan(self, plan: TrainingPlan) -> None:
        await self.session.execute(
            text("""
                INSERT INTO iteration.training_plans
                    (training_plan_id, proposal_id, approval_id, iteration_run_id,
                     experiment_id, business_round, strategy_code, status,
                     plan_json, rule_version)
                VALUES (:id, :proposal, :approval, :run, :experiment, :round,
                        :strategy, :status, :payload, :version)
            """),
            {
                "id": plan.training_plan_id,
                "proposal": plan.proposal_id,
                "approval": plan.approval_id,
                "run": plan.iteration_run_id,
                "experiment": plan.experiment_id,
                "round": plan.business_round,
                "strategy": plan.strategy_code,
                "status": plan.status.value,
                "payload": _json(plan),
                "version": plan.rule_version,
            },
        )

    async def create_round_and_experiment(self, plan: TrainingPlan) -> None:
        await self.session.execute(
            text("""
                INSERT INTO iteration.iteration_rounds
                    (iteration_run_id, round_no, strategy_code, experiment_id)
                VALUES (:run, :round, :strategy, :experiment)
            """),
            {
                "run": plan.iteration_run_id,
                "round": plan.business_round,
                "strategy": plan.strategy_code,
                "experiment": plan.experiment_id,
            },
        )
        await self.session.execute(
            text("""
                INSERT INTO iteration.experiments
                    (experiment_id, iteration_run_id, training_plan_id, round_no,
                     strategy_code, frozen_champion_version, experiment_json)
                VALUES (:id, :run, :plan, :round, :strategy, :champion, :payload)
            """),
            {
                "id": plan.experiment_id,
                "run": plan.iteration_run_id,
                "plan": plan.training_plan_id,
                "round": plan.business_round,
                "strategy": plan.strategy_code,
                "champion": plan.frozen_champion_version,
                "payload": json.dumps(
                    {
                        "experiment_id": plan.experiment_id,
                        "training_plan_id": plan.training_plan_id,
                        "business_round": plan.business_round,
                        "technical_status": "PENDING",
                        "qualification_status": "PENDING",
                    }
                ),
            },
        )

    async def create_training_job(self, job: TrainingJobInput) -> tuple[bool, dict]:
        result = await self.session.execute(
            text("""
                INSERT INTO iteration.training_jobs
                    (training_job_id, idempotency_key, iteration_run_id,
                     training_plan_id, experiment_id, round_no, request_json)
                VALUES (:id, :key, :run, :plan, :experiment, :round, :payload)
                ON CONFLICT (idempotency_key) DO NOTHING
                RETURNING training_job_id
            """),
            {
                "id": job.training_job_id,
                "key": job.idempotency_key,
                "run": job.iteration_run_id,
                "plan": job.training_plan_id,
                "experiment": job.experiment_id,
                "round": job.business_round,
                "payload": _json(job),
            },
        )
        created = result.scalar_one_or_none() is not None
        existing = await self.session.execute(
            text("""
                SELECT training_job_id, idempotency_key, status, request_json,
                       result_json, technical_retry_count
                FROM iteration.training_jobs
                WHERE idempotency_key = :key
            """),
            {"key": job.idempotency_key},
        )
        return created, dict(existing.mappings().one())

    async def save_training_callback(
        self, callback: TrainingCallback
    ) -> tuple[bool, dict]:
        current = await self.session.execute(
            text("""
                SELECT *
                FROM iteration.training_jobs
                WHERE training_job_id = :id
                  AND idempotency_key = :key
            """),
            {
                "id": callback.training_job_id,
                "key": callback.idempotency_key,
            },
        )
        current_row = current.mappings().first()
        if current_row is None:
            return False, {}
        if current_row["result_json"] is not None:
            return False, dict(current_row)

        terminal = callback.status.value in {
            "SUCCEEDED",
            "FAILED",
            "CANCELLED",
            "DEAD_LETTER",
            "LOST",
        }
        await self.session.execute(
            text("""
                UPDATE iteration.training_jobs
                SET status = :status,
                    result_json = CASE
                        WHEN :terminal THEN CAST(:payload AS JSONB)
                        ELSE result_json END,
                    technical_retry_count = :retry_count,
                    updated_at = NOW(),
                    completed_at = CASE
                        WHEN :terminal
                        THEN NOW() ELSE completed_at END
                WHERE training_job_id = :id
            """),
            {
                "id": callback.training_job_id,
                "status": callback.status.value,
                "payload": _json(callback),
                "terminal": terminal,
                "retry_count": callback.technical_retry_count,
            },
        )
        await self.session.execute(
            text("""
                UPDATE iteration.experiments
                SET technical_status = :status,
                    candidate_version = :candidate,
                    experiment_json = experiment_json || CAST(:payload AS JSONB),
                    updated_at = NOW()
                WHERE experiment_id = :experiment
            """),
            {
                "experiment": callback.experiment_id,
                "status": callback.status.value,
                "candidate": callback.candidate_version,
                "payload": _json(callback),
            },
        )
        return True, dict(current_row)

    async def get_training_plan(self, training_plan_id: str) -> dict | None:
        result = await self.session.execute(
            text("""
                SELECT plan_json
                FROM iteration.training_plans
                WHERE training_plan_id = :id
            """),
            {"id": training_plan_id},
        )
        return result.scalar_one_or_none()

    async def get_iteration_run(self, iteration_run_id: str) -> dict | None:
        result = await self.session.execute(
            text("""
                SELECT *
                FROM iteration.iteration_runs
                WHERE iteration_run_id = :id
            """),
            {"id": iteration_run_id},
        )
        row = result.mappings().first()
        return dict(row) if row else None

    async def get_iteration_rounds(self, iteration_run_id: str) -> list[dict]:
        result = await self.session.execute(
            text("""
                SELECT *
                FROM iteration.iteration_rounds
                WHERE iteration_run_id = :id
                ORDER BY round_no
            """),
            {"id": iteration_run_id},
        )
        return [dict(row) for row in result.mappings()]

    async def get_experiment(self, experiment_id: str) -> dict | None:
        result = await self.session.execute(
            text("""
                SELECT *
                FROM iteration.experiments
                WHERE experiment_id = :id
            """),
            {"id": experiment_id},
        )
        row = result.mappings().first()
        return dict(row) if row else None

    async def get_experiment_qualification(
        self, experiment_id: str
    ) -> dict | None:
        result = await self.session.execute(
            text("""
                SELECT report_json
                FROM iteration.qualification_reports
                WHERE experiment_id = :id
                ORDER BY created_at DESC
                LIMIT 1
            """),
            {"id": experiment_id},
        )
        return result.scalar_one_or_none()

    async def get_failures(self, iteration_run_id: str) -> list[dict]:
        result = await self.session.execute(
            text("""
                SELECT report_json
                FROM iteration.failure_reports
                WHERE iteration_run_id = :id
                ORDER BY created_at
            """),
            {"id": iteration_run_id},
        )
        return [row["report_json"] for row in result.mappings()]

    async def save_qualification(self, report: QualificationReport) -> None:
        await self.session.execute(
            text("""
                INSERT INTO iteration.qualification_reports
                    (qualification_run_id, iteration_run_id, experiment_id,
                     candidate_version, status, qualified, report_json, rule_version)
                VALUES (:id, :run, :experiment, :candidate, :status, :qualified,
                        :payload, :version)
            """),
            {
                "id": report.qualification_run_id,
                "run": report.iteration_run_id,
                "experiment": report.experiment_id,
                "candidate": report.candidate_version,
                "status": report.status.value,
                "qualified": report.qualified,
                "payload": _json(report),
                "version": report.rule_version,
            },
        )
        for gate in report.gate_results:
            await self.session.execute(
                text("""
                    INSERT INTO iteration.qualification_checks
                        (qualification_run_id, gate_code, gate_order, required,
                         status, check_json)
                    VALUES (:run, :code, :gate_order, :required, :status, :payload)
                """),
                {
                    "run": report.qualification_run_id,
                    "code": gate.gate_code.value,
                    "gate_order": gate.gate_order,
                    "required": gate.required,
                    "status": gate.status.value,
                    "payload": _json(gate),
                },
            )
        await self.session.execute(
            text("""
                UPDATE iteration.experiments
                SET qualification_status = :status,
                    experiment_json = experiment_json || CAST(:payload AS JSONB),
                    updated_at = NOW()
                WHERE experiment_id = :experiment
            """),
            {
                "experiment": report.experiment_id,
                "status": report.status.value,
                "payload": json.dumps(
                    {
                        "qualification_run_id": report.qualification_run_id,
                        "qualification_status": report.status.value,
                        "qualified": report.qualified,
                    }
                ),
            },
        )
        await self.session.execute(
            text("""
                UPDATE iteration.iteration_rounds
                SET status = :status,
                    completed_at = NOW()
                WHERE experiment_id = :experiment
            """),
            {
                "experiment": report.experiment_id,
                "status": "QUALIFIED" if report.qualified else "FAILED",
            },
        )

    async def save_failure(self, report: FailureReport) -> None:
        await self.session.execute(
            text("""
                INSERT INTO iteration.failure_reports
                    (failure_report_id, iteration_run_id, experiment_id,
                     proposal_id, failure_code, retryable, report_json, created_at)
                VALUES (:id, :run, :experiment, :proposal, :code, :retryable,
                        :payload, :created_at)
            """),
            {
                "id": report.failure_report_id,
                "run": report.iteration_run_id,
                "experiment": report.experiment_id,
                "proposal": report.proposal_id,
                "code": report.failure_code.value,
                "retryable": report.retryable,
                "payload": _json(report),
                "created_at": report.created_at,
            },
        )

    async def save_case(self, record: RepairCaseRecord) -> None:
        await self.session.execute(
            text("""
                INSERT INTO iteration.repair_case_records
                    (case_id, data_track, model_id, diagnosis_run_id, proposal_id,
                     iteration_run_id, primary_root_cause_code, action, outcome,
                     qualified, failure_report_id, case_json, created_at)
                VALUES (:id, :track, :model, :diagnosis, :proposal, :run, :root,
                        :action, :outcome, :qualified, :failure, :payload, :created_at)
            """),
            {
                "id": record.case_id,
                "track": record.data_track.value,
                "model": record.model_id,
                "diagnosis": record.diagnosis_run_id,
                "proposal": record.proposal_id,
                "run": record.iteration_run_id,
                "root": record.primary_root_cause_code,
                "action": record.action,
                "outcome": record.outcome,
                "qualified": record.qualified,
                "failure": record.failure_report_id,
                "payload": _json(record),
                "created_at": record.created_at,
            },
        )

    async def save_external_execution_plan(self, plan: dict) -> None:
        await self.session.execute(
            text("""
                INSERT INTO iteration.external_execution_plans
                    (lifecycle_run_id, plan_type, plan_id, action, status,
                     dispatch_mode, external_task_id, callback_endpoint,
                     request_json, result_json, error_message, completed_at)
                VALUES (:lifecycle, :type, :plan, :action, :status, :mode,
                        :task, :callback, CAST(:request AS JSONB),
                        CASE WHEN :result IS NULL THEN NULL ELSE CAST(:result AS JSONB) END,
                        :error,
                        CASE WHEN :completed THEN NOW() ELSE NULL END)
                ON CONFLICT (plan_id) DO UPDATE
                SET status = EXCLUDED.status,
                    dispatch_mode = EXCLUDED.dispatch_mode,
                    external_task_id = EXCLUDED.external_task_id,
                    callback_endpoint = EXCLUDED.callback_endpoint,
                    request_json = EXCLUDED.request_json,
                    result_json = COALESCE(EXCLUDED.result_json, iteration.external_execution_plans.result_json),
                    error_message = EXCLUDED.error_message,
                    updated_at = NOW(),
                    completed_at = COALESCE(EXCLUDED.completed_at, iteration.external_execution_plans.completed_at)
            """),
            {
                "lifecycle": plan.get("lifecycle_run_id"),
                "type": plan["plan_type"],
                "plan": plan["plan_id"],
                "action": plan.get("action"),
                "status": plan.get("status", "PLANNED"),
                "mode": plan.get("dispatch_mode", "INTERNAL"),
                "task": plan.get("external_task_id"),
                "callback": plan.get("callback_endpoint"),
                "request": json.dumps(plan.get("request_json", plan), ensure_ascii=False),
                "result": (
                    json.dumps(plan.get("result_json"), ensure_ascii=False)
                    if plan.get("result_json") is not None
                    else None
                ),
                "error": plan.get("error_message"),
                "completed": plan.get("status") in {"SUCCEEDED", "FAILED", "CANCELLED"},
            },
        )

    async def save_external_execution_callback(
        self,
        plan_id: str,
        status: str,
        payload: dict,
    ) -> dict | None:
        result = await self.session.execute(
            text("""
                UPDATE iteration.external_execution_plans
                SET status = :status,
                    result_json = CAST(:payload AS JSONB),
                    error_message = :error,
                    updated_at = NOW(),
                    completed_at = CASE
                        WHEN :terminal THEN NOW() ELSE completed_at END
                WHERE plan_id = :plan
                RETURNING *
            """),
            {
                "plan": plan_id,
                "status": status,
                "payload": json.dumps(payload, ensure_ascii=False),
                "error": payload.get("error_message"),
                "terminal": status in {"SUCCEEDED", "FAILED", "CANCELLED"},
            },
        )
        row = result.mappings().first()
        return dict(row) if row else None

    async def get_external_execution_plan(self, plan_id: str) -> dict | None:
        result = await self.session.execute(
            text("""
                SELECT *
                FROM iteration.external_execution_plans
                WHERE plan_id = :plan
            """),
            {"plan": plan_id},
        )
        row = result.mappings().first()
        return dict(row) if row else None

    async def save_deployment_record(self, record: dict) -> None:
        await self.session.execute(
            text("""
                INSERT INTO iteration.deployment_records
                    (deployment_id, lifecycle_run_id, qualification_run_id,
                     model_id, champion_version, candidate_version, current_stage,
                     decision, status, dispatch_mode, external_task_id, record_json,
                     completed_at)
                VALUES (:deployment, :lifecycle, :qualification, :model, :champion,
                        :candidate, :stage, :decision, :status, :mode, :task,
                        CAST(:payload AS JSONB),
                        CASE WHEN :completed THEN NOW() ELSE NULL END)
                ON CONFLICT (deployment_id) DO UPDATE
                SET current_stage = EXCLUDED.current_stage,
                    decision = EXCLUDED.decision,
                    status = EXCLUDED.status,
                    dispatch_mode = EXCLUDED.dispatch_mode,
                    external_task_id = EXCLUDED.external_task_id,
                    record_json = EXCLUDED.record_json,
                    updated_at = NOW(),
                    completed_at = CASE
                        WHEN EXCLUDED.completed_at IS NOT NULL
                        THEN EXCLUDED.completed_at
                        ELSE iteration.deployment_records.completed_at END
            """),
            {
                "deployment": record["deployment_id"],
                "lifecycle": record.get("lifecycle_run_id"),
                "qualification": record.get("qualification_run_id"),
                "model": record.get("model_id"),
                "champion": record.get("champion_version"),
                "candidate": record.get("candidate_version"),
                "stage": record["deployment_stage"],
                "decision": record["deployment_decision"],
                "status": record.get("status", "RUNNING"),
                "mode": record.get("dispatch_mode", "INTERNAL"),
                "task": record.get("external_task_id"),
                "payload": json.dumps(record, ensure_ascii=False),
                "completed": record.get("status") in {"PROMOTED", "ROLLED_BACK", "ABORTED"},
            },
        )
        await self.session.execute(
            text("""
                INSERT INTO iteration.deployment_stage_records
                    (deployment_id, stage, decision, status, health_json, result_json)
                VALUES (:deployment, :stage, :decision, :status,
                        CAST(:health AS JSONB), CAST(:result AS JSONB))
            """),
            {
                "deployment": record["deployment_id"],
                "stage": record["deployment_stage"],
                "decision": record["deployment_decision"],
                "status": record.get("status", "RUNNING"),
                "health": json.dumps(record.get("health_json", {}), ensure_ascii=False),
                "result": json.dumps(record, ensure_ascii=False),
            },
        )
