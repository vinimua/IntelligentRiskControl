"""任务三：根因驱动修复决策 API。

本路由只提供确定性规则、持久化和跨模块合同，不负责 Agent 或 LangGraph。
"""

from datetime import UTC, datetime
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, Query, Request

logger = structlog.get_logger(__name__)
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from packages.models.common.enums import DataTrack, ProposalStatus, ReviewDecision
from packages.models.iteration import (
    DecisionInput,
    ManualReviewReport,
    ManualReviewSubmission,
    QualificationInput,
    RepairCaseRecord,
)
from packages.models.callbacks.training_callback import TrainingCallback
from packages.models.iteration.training_job import TrainingJobInput

from ..core.exceptions import (
    ConflictError,
    NotFoundError,
    ValidationAppError,
    request_trace_id,
)
from ..database import get_db
from ..repositories.iteration_repo import IterationRepo
from ..services.iteration import (
    FailureAttributionService,
    QualificationService,
    RepairDecisionService,
    RiskAssessmentService,
    TrainingPlanBuilder,
    load_iteration_config,
)

router = APIRouter(prefix="/api/iteration", tags=["iteration"])
internal_router = APIRouter(
    prefix="/api/internal/iteration", tags=["iteration-internal"]
)


class PlanBuildRequest(BaseModel):
    approval_id: str
    model_algorithm: str = Field(min_length=1)
    feature_schema_version: str = Field(min_length=1)
    preprocessing_version: str = Field(min_length=1)
    business_round: int = Field(default=1, ge=1, le=3)
    data_snapshot_ids: list[str] = Field(min_length=1)
    label_versions: list[str] = Field(min_length=1)


class QualificationRequest(QualificationInput):
    data_track: DataTrack = DataTrack.NATURAL


class ExternalExecutionCallbackRequest(BaseModel):
    status: str = Field(default="SUCCEEDED", min_length=1)
    artifact_uri: str | None = None
    metrics: dict = Field(default_factory=dict)
    external_task_id: str | None = None
    error_message: str | None = None
    resume_lifecycle: bool = True


class RepairCompleteRequest(BaseModel):
    status: str = Field(default="SUCCEEDED", min_length=1)
    repair_plan_id: str | None = None
    metrics: dict = Field(default_factory=dict)
    artifact_uri: str | None = None
    error_message: str | None = None


class DeploymentCallbackRequest(BaseModel):
    lifecycle_run_id: str | None = None
    qualification_run_id: str | None = None
    model_id: str | None = None
    champion_version: str | None = None
    candidate_version: str | None = None
    deployment_stage: str = Field(min_length=1)
    deployment_decision: str = Field(min_length=1)
    status: str = Field(default="RUNNING", min_length=1)
    external_task_id: str | None = None
    health_json: dict = Field(default_factory=dict)
    result_json: dict = Field(default_factory=dict)


def _envelope(request: Request, data, message: str = "success") -> dict:
    return {
        "success": True,
        "code": "OK",
        "message": message,
        "data": data,
        "trace_id": request_trace_id(request),
    }


@router.post("/decisions")
async def create_decision(
    request: Request,
    body: DecisionInput,
    db: AsyncSession = Depends(get_db),
):
    proposal = RepairDecisionService().decide(body)
    risk = RiskAssessmentService().assess(proposal)
    if risk.requires_manual_review and not proposal.requires_manual_review:
        proposal = proposal.model_copy(
            update={
                "requires_manual_review": True,
                "status": ProposalStatus.PENDING_REVIEW,
            }
        )
    repo = IterationRepo(db)
    await repo.save_proposal(proposal)
    await repo.save_risk(risk)
    return _envelope(
        request,
        {
            "proposal": proposal.model_dump(mode="json"),
            "risk_assessment": risk.model_dump(mode="json"),
        },
        "decision proposal created",
    )


@router.get("/decisions")
async def list_decisions(
    request: Request,
    model_id: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    items = await IterationRepo(db).list_proposals(model_id=model_id, limit=limit)
    return _envelope(request, {"items": items})


@router.get("/decisions/{proposal_id}")
async def get_decision(
    proposal_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    proposal = await IterationRepo(db).get_proposal(proposal_id)
    if proposal is None:
        raise NotFoundError("修复决策建议不存在")
    return _envelope(request, proposal.model_dump(mode="json"))


@router.get("/proposals/{proposal_id}")
async def get_proposal_contract(
    proposal_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    return await get_decision(proposal_id, request, db)


@router.get("/proposals/{proposal_id}/risk")
async def get_proposal_risk(
    proposal_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    payload = await IterationRepo(db).get_risk_for_proposal(proposal_id)
    if payload is None:
        raise NotFoundError("风险评估不存在")
    return _envelope(request, payload)


@router.get("/plans/{training_plan_id}")
async def get_training_plan(
    training_plan_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    payload = await IterationRepo(db).get_training_plan(training_plan_id)
    if payload is None:
        raise NotFoundError("训练计划不存在")
    return _envelope(request, payload)


@router.get("/runs/{iteration_run_id}")
async def get_iteration_run(
    iteration_run_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    payload = await IterationRepo(db).get_iteration_run(iteration_run_id)
    if payload is None:
        raise NotFoundError("迭代运行不存在")
    return _envelope(request, payload)


@router.get("/runs/{iteration_run_id}/rounds")
async def get_iteration_rounds(
    iteration_run_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    payload = await IterationRepo(db).get_iteration_rounds(iteration_run_id)
    return _envelope(request, {"items": payload})


@router.get("/runs/{iteration_run_id}/failures")
async def get_iteration_failures(
    iteration_run_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    payload = await IterationRepo(db).get_failures(iteration_run_id)
    return _envelope(request, {"items": payload})


@router.get("/executions/{plan_id}")
async def get_external_execution(
    plan_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    payload = await IterationRepo(db).get_external_execution_plan(plan_id)
    if payload is None:
        raise NotFoundError("外部执行计划不存在")
    return _envelope(request, payload)


@router.get("/experiments/{experiment_id}")
async def get_experiment(
    experiment_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    payload = await IterationRepo(db).get_experiment(experiment_id)
    if payload is None:
        raise NotFoundError("训练实验不存在")
    return _envelope(request, payload)


@router.get("/experiments/{experiment_id}/qualification")
async def get_experiment_qualification(
    experiment_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    payload = await IterationRepo(db).get_experiment_qualification(experiment_id)
    if payload is None:
        raise NotFoundError("候选模型资格报告不存在")
    return _envelope(request, payload)


@router.get("/config/{model_id}")
async def get_iteration_config(model_id: str, request: Request):
    config = load_iteration_config()
    return _envelope(
        request,
        {
            "model_id": model_id,
            "iteration": config.iteration.model_dump(mode="json"),
            "qualification": config.qualification.model_dump(mode="json"),
            "risk": config.risk.model_dump(mode="json"),
            "strategy_rule_version": config.strategies.rule_version,
        },
    )


@router.post("/decisions/{proposal_id}/reviews")
async def submit_review(
    proposal_id: str,
    request: Request,
    body: ManualReviewSubmission,
    db: AsyncSession = Depends(get_db),
):
    if body.proposal_id != proposal_id:
        raise ValidationAppError(
            "PROPOSAL_ID_MISMATCH", "路径与人工复核报告中的 proposal_id 不一致"
        )
    repo = IterationRepo(db)
    proposal = await repo.get_proposal(proposal_id)
    if proposal is None:
        raise NotFoundError("修复决策建议不存在")
    if proposal.status in {ProposalStatus.APPROVED, ProposalStatus.REJECTED}:
        raise ConflictError("该决策建议已经完成复核")

    report = ManualReviewReport(
        review_id=str(uuid4()),
        proposal_id=proposal_id,
        reviewer_id=body.reviewer_id,
        decision=body.decision,
        reason=body.reason,
        rejection_reason_codes=body.rejection_reason_codes,
        adjustment_instructions=body.adjustment_instructions,
        forbidden_adjustments=body.forbidden_adjustments,
        expected_evidence=body.expected_evidence,
        reviewed_at=body.reviewed_at,
    )
    await repo.save_review(report)
    if body.decision == ReviewDecision.REJECT:
        case = RepairCaseRecord(
            case_id=str(uuid4()),
            data_track=DataTrack.NATURAL,
            model_id=proposal.model_id,
            diagnosis_run_id=proposal.diagnosis_run_id,
            proposal_id=proposal_id,
            primary_root_cause_code=proposal.primary_root_cause_code,
            action=proposal.action.value,
            strategy_codes=[item.strategy_code for item in proposal.strategies],
            outcome="REVIEW_REJECTED",
            qualified=None,
            created_at=datetime.now(UTC),
        )
        await repo.save_case(case)
    return _envelope(
        request,
        report.model_dump(mode="json"),
        (
            "review approved"
            if body.decision == ReviewDecision.APPROVE
            else "review rejected with adjustment report"
        ),
    )


@router.post("/decisions/{proposal_id}/training-plans")
async def create_training_plan(
    proposal_id: str,
    request: Request,
    body: PlanBuildRequest,
    db: AsyncSession = Depends(get_db),
):
    repo = IterationRepo(db)
    proposal = await repo.get_proposal(proposal_id)
    if proposal is None:
        raise NotFoundError("修复决策建议不存在")
    approval = await repo.get_approved_review(body.approval_id, proposal_id)
    if approval is None:
        raise ValidationAppError(
            "APPROVAL_REQUIRED", "生成训练计划前必须具有匹配的人工通过报告"
        )

    approved_proposal = proposal.model_copy(update={"status": ProposalStatus.APPROVED})
    risk = RiskAssessmentService().assess(approved_proposal)
    iteration_run_id = str(uuid4())
    try:
        plan = TrainingPlanBuilder().build(
            approved_proposal,
            risk,
            approval_id=body.approval_id,
            iteration_run_id=iteration_run_id,
            model_algorithm=body.model_algorithm,
            feature_schema_version=body.feature_schema_version,
            preprocessing_version=body.preprocessing_version,
            business_round=body.business_round,
            data_snapshot_ids=body.data_snapshot_ids,
            label_versions=body.label_versions,
        )
    except ValueError as exc:
        raise ValidationAppError("TRAINING_PLAN_REJECTED", str(exc)) from exc

    config = load_iteration_config()
    await repo.create_iteration_run(
        iteration_run_id,
        approved_proposal,
        config.iteration.max_iteration_rounds,
    )
    await repo.save_training_plan(plan)
    await repo.create_round_and_experiment(plan)
    return _envelope(
        request,
        plan.model_dump(mode="json"),
        "approved training plan created",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# T3-GAP-01: 特征重构 API
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureReconstructionTriggerRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=100)
    lifecycle_run_id: str | None = None
    diagnosis_run_id: str | None = None
    current_schema_version: str = "v1"
    drift_features: list[dict] = Field(default_factory=list)
    high_missing_features: list[dict] = Field(default_factory=list)
    current_feature_names: list[str] = Field(default_factory=list)
    feature_importance: dict[str, float] = Field(default_factory=dict)
    skewness: dict[str, float] = Field(default_factory=dict)


@router.post("/features/reconstruction-plans")
async def create_feature_reconstruction_plan(
    request: Request,
    body: FeatureReconstructionTriggerRequest,
    db: AsyncSession = Depends(get_db),
):
    """T3-GAP-01: 创建特征重构计划。

    根据诊断证据（PSI 漂移、缺失率、偏度）生成增/删/改特征的计划。
    """
    from ..services.iteration.feature_reconstruction_service import FeatureReconstructionService

    svc = FeatureReconstructionService()
    plan = svc.build_plan(
        model_id=body.model_id,
        lifecycle_run_id=body.lifecycle_run_id,
        diagnosis_run_id=body.diagnosis_run_id,
        current_schema_version=body.current_schema_version,
        drift_features=body.drift_features,
        high_missing_features=body.high_missing_features,
        current_feature_names=body.current_feature_names,
        feature_importance=body.feature_importance,
        skewness=body.skewness,
    )

    # 持久化
    await IterationRepo(db).save_feature_reconstruction_plan(plan)
    await db.commit()

    return _envelope(
        request,
        plan.model_dump(mode="json"),
        f"feature reconstruction plan created: {plan.plan_id}",
    )


@router.get("/features/reconstruction-plans/{plan_id}")
async def get_feature_reconstruction_plan(
    plan_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """获取特征重构计划及其执行状态。"""
    repo = IterationRepo(db)
    plan = await repo.get_feature_reconstruction_plan(plan_id)
    if plan is None:
        raise NotFoundError("特征重构计划不存在")
    return _envelope(request, plan)


@internal_router.post("/features/{plan_id}/callback")
async def feature_reconstruction_callback(
    plan_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Worker 特征重构完成回调。"""
    from packages.models.iteration.feature_reconstruction import FeatureReconstructionResult

    body = await request.json()
    result = FeatureReconstructionResult(**body)

    repo = IterationRepo(db)
    await repo.save_feature_reconstruction_result(plan_id, result)
    await db.commit()

    # 自动 resume lifecycle
    lifecycle_resumed = False
    if result.lifecycle_run_id and result.status == "SUCCEEDED":
        try:
            from ..services.workflow.checkpointer_manager import get_checkpointer
            from ..services.workflow.workflow_service import WorkflowService

            service = WorkflowService(db, get_checkpointer())
            resume_result = await service.resume(
                result.lifecycle_run_id,
                decision="approved",
                resume_payload={
                    "decision": "approved",
                    "resume_type": "FEATURE_RECONSTRUCTION_COMPLETE",
                    "feature_reconstruction_plan_id": plan_id,
                    "status": result.status,
                    "feature_schema_version": result.feature_schema_version,
                    "feature_snapshot_id": result.feature_snapshot_id,
                    "transform_artifact_uri": result.transform_artifact_uri,
                    "error_message": result.error_message,
                },
            )
            lifecycle_resumed = True
        except Exception:
            pass

    return _envelope(
        request,
        {
            "plan_id": plan_id,
            "status": result.status,
            "lifecycle_resumed": lifecycle_resumed,
        },
        "feature reconstruction callback recorded",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# T3-GAP-02: 超参优化 API
# ═══════════════════════════════════════════════════════════════════════════════

class TuningTriggerRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=100)
    lifecycle_run_id: str | None = None
    training_plan_id: str | None = None
    algorithm: str = "lightgbm"
    num_trials: int = Field(default=5, ge=2, le=20)


@router.post("/tuning-runs")
async def create_tuning_run(
    request: Request,
    body: TuningTriggerRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create a hyperparameter tuning run."""
    from ..services.iteration.hyperparameter_tuning_service import HyperparameterTuningService

    svc = HyperparameterTuningService()
    plan = svc.build_plan(
        model_id=body.model_id,
        lifecycle_run_id=body.lifecycle_run_id,
        training_plan_id=body.training_plan_id,
        algorithm=body.algorithm,
        num_trials=body.num_trials,
    )
    await IterationRepo(db).save_tuning_plan(plan)
    await db.commit()
    return _envelope(request, plan.model_dump(mode="json"), f"tuning run created: {plan.plan_id}")


@router.get("/tuning-runs/{plan_id}")
async def get_tuning_run(
    plan_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Get a hyperparameter tuning run and its result."""
    repo = IterationRepo(db)
    plan = await repo.get_tuning_plan(plan_id)
    if plan is None:
        raise NotFoundError("tuning run not found")
    return _envelope(request, plan)


@router.get("/tuning-runs/{plan_id}/trials")
async def get_tuning_trials(
    plan_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Get all trial results for a tuning run."""
    repo = IterationRepo(db)
    plan = await repo.get_tuning_plan(plan_id)
    if plan is None:
        raise NotFoundError("tuning run not found")
    request_json = plan.get("request_json") if isinstance(plan.get("request_json"), dict) else {}
    result_json = plan.get("result_json") if isinstance(plan.get("result_json"), dict) else {}
    return _envelope(request, {
        "plan_id": plan_id,
        "status": plan.get("status"),
        "trials": result_json.get("trials", request_json.get("trials", [])),
        "best_trial_index": result_json.get("best_trial_index"),
        "best_hyperparameters": result_json.get("best_hyperparameters"),
        "best_val_auc": result_json.get("best_val_auc"),
    })


@internal_router.post("/tuning-runs/{plan_id}/callback")
async def tuning_callback(
    plan_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Worker callback for a completed hyperparameter tuning run."""
    body = await request.json()
    repo = IterationRepo(db)
    await repo.save_tuning_result(plan_id, body)
    await db.commit()

    lifecycle_resumed = False
    lifecycle_run_id = body.get("lifecycle_run_id")
    if lifecycle_run_id and str(body.get("status", "")).upper() == "SUCCEEDED":
        try:
            from ..services.workflow.checkpointer_manager import get_checkpointer
            from ..services.workflow.workflow_service import WorkflowService

            service = WorkflowService(db, get_checkpointer())
            await service.resume(
                str(lifecycle_run_id),
                decision="approved",
                resume_payload={
                    "decision": "approved",
                    "resume_type": "TUNING_COMPLETE",
                    "hyperparameter_tuning_plan_id": plan_id,
                    "status": body.get("status"),
                    "best_hyperparameters": body.get("best_hyperparameters", {}),
                    "best_val_auc": body.get("best_val_auc"),
                    "trials": body.get("trials", []),
                },
            )
            lifecycle_resumed = True
        except Exception:
            logger.warning(
                "tuning_callback_auto_resume_failed",
                plan_id=plan_id,
                lifecycle_run_id=lifecycle_run_id,
                exc_info=True,
            )

    return _envelope(
        request,
        {"plan_id": plan_id, "status": body.get("status"), "lifecycle_resumed": lifecycle_resumed},
        "tuning callback recorded",
    )


@router.post("/decisions/{proposal_id}/qualifications")
async def evaluate_qualification(
    proposal_id: str,
    request: Request,
    body: QualificationRequest,
    db: AsyncSession = Depends(get_db),
):
    repo = IterationRepo(db)
    proposal = await repo.get_proposal(proposal_id)
    if proposal is None:
        raise NotFoundError("修复决策建议不存在")

    experiment = await repo.get_experiment(body.experiment_id)
    if experiment is None:
        raise NotFoundError("训练实验不存在")
    if (
        str(experiment["iteration_run_id"]) != body.iteration_run_id
        or experiment["technical_status"] != "SUCCEEDED"
        or experiment["candidate_version"] != body.candidate_version
    ):
        raise ValidationAppError(
            "EXPERIMENT_NOT_READY_FOR_QUALIFICATION",
            "实验必须技术成功，且 iteration_run_id 与 candidate_version 完全匹配",
        )

    qualification_input = QualificationInput.model_validate(
        body.model_dump(exclude={"data_track"})
    )
    report = QualificationService().evaluate(qualification_input)
    await repo.save_qualification(report)

    failure = FailureAttributionService().from_qualification(proposal_id, report)
    if failure:
        await repo.save_failure(failure)

    case = RepairCaseRecord(
        case_id=str(uuid4()),
        data_track=body.data_track,
        model_id=proposal.model_id,
        diagnosis_run_id=proposal.diagnosis_run_id,
        proposal_id=proposal_id,
        iteration_run_id=report.iteration_run_id,
        primary_root_cause_code=proposal.primary_root_cause_code,
        action=proposal.action.value,
        strategy_codes=[item.strategy_code for item in proposal.strategies],
        outcome="QUALIFIED" if report.qualified else "QUALIFICATION_FAILED",
        qualified=report.qualified,
        failure_report_id=failure.failure_report_id if failure else None,
        created_at=datetime.now(UTC),
    )
    await repo.save_case(case)
    return _envelope(
        request,
        {
            "qualification_report": report.model_dump(mode="json"),
            "failure_report": (
                failure.model_dump(mode="json") if failure is not None else None
            ),
            "case_record_id": case.case_id,
        },
        "challenger qualification evaluated",
    )


@internal_router.post("/jobs")
async def create_training_job(
    request: Request,
    body: TrainingJobInput,
    db: AsyncSession = Depends(get_db),
):
    repo = IterationRepo(db)
    created, row = await repo.create_training_job(body)
    stored = row.get("request_json")
    if not created and stored != body.model_dump(mode="json"):
        raise ConflictError(
            "同一 idempotency_key 已用于不同训练参数",
            code="TRAINING_JOB_IDEMPOTENCY_CONFLICT",
        )
    return _envelope(
        request,
        {
            "training_job_id": str(row["training_job_id"]),
            "created": created,
            "status": row["status"],
        },
        "training job accepted",
    )


@internal_router.post("/jobs/{training_job_id}/callback")
async def training_job_callback(
    training_job_id: str,
    request: Request,
    body: TrainingCallback,
    db: AsyncSession = Depends(get_db),
):
    if body.training_job_id != training_job_id:
        raise ValidationAppError(
            "TRAINING_JOB_ID_MISMATCH",
            "路径与回调中的 training_job_id 不一致",
        )
    applied, existing = await IterationRepo(db).save_training_callback(body)
# 第 753 行：把 FAILED 状态写入 PostgreSQL iteration.training_jobs 表
# → DB 里 training_job.status 现在是 "FAILED"
# → applied = True（新回调被接受）
    if not existing:
        raise NotFoundError("训练任务不存在")
    if not applied:
        stored_result = existing.get("result_json")
        if stored_result != body.model_dump(mode="json"):
            raise ConflictError(
                "训练任务已经收到不同结果的终态回调",
                code="TRAINING_CALLBACK_CONFLICT",
            )
    # P1: 自动 resume lifecycle
    if applied:
        await db.commit()
    lifecycle_resumed = False
    lifecycle_run_id = body.lifecycle_run_id
    if lifecycle_run_id and applied:
        try:
            from ..services.workflow.workflow_service import WorkflowService
            from ..services.workflow.checkpointer_manager import get_checkpointer

            checkpointer = get_checkpointer()
            service = WorkflowService(db, checkpointer)
            resume_result = await service.resume(
                lifecycle_run_id,
                decision="approved",
                resume_payload={
                    "decision": "approved",
                    "resume_type": "TRAINING_CALLBACK",
                    "training_job_id": training_job_id,
                    "status": body.status,
                    "candidate_version": body.candidate_version,
                    "experiment_id": body.experiment_id,
                },
            )
            lifecycle_resumed = True
            logger.info(
                "training_callback_auto_resumed",
                lifecycle_run_id=lifecycle_run_id,
                new_phase=resume_result.get("current_phase"),
            )
        except Exception:
            logger.warning(
                "training_callback_auto_resume_failed",
                lifecycle_run_id=lifecycle_run_id,
                exc_info=True,
            )

    return _envelope(
        request,
        {
            "training_job_id": training_job_id,
            "callback_applied": applied,
            "qualification_status": "PENDING",
            "lifecycle_resumed": lifecycle_resumed,
        },
        "technical callback recorded; qualification remains pending",
    )


# ── P4: 数据修复完成回调 ──

@internal_router.post("/executions/{plan_type}/{plan_id}/callback")
async def external_execution_callback(
    plan_type: str,
    plan_id: str,
    request: Request,
    body: ExternalExecutionCallbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """External calibration/threshold/repair executors report completion."""
    payload = {
        "plan_type": plan_type.upper(),
        "plan_id": plan_id,
        **body.model_dump(mode="json"),
    }
    row = await IterationRepo(db).save_external_execution_callback(
        plan_id,
        body.status.upper(),
        payload,
    )
    if row is None:
        raise NotFoundError("外部执行计划不存在")
    await db.commit()

    lifecycle_resumed = False
    lifecycle_run_id = row.get("lifecycle_run_id")
    if lifecycle_run_id and body.resume_lifecycle and body.status.upper() == "SUCCEEDED":
        try:
            from ..services.workflow.checkpointer_manager import get_checkpointer
            from ..services.workflow.workflow_service import WorkflowService

            service = WorkflowService(db, get_checkpointer())
            result = await service.resume(
                str(lifecycle_run_id),
                decision="approved",
                resume_payload={
                    "decision": "approved",
                    "resume_type": f"{plan_type.upper()}_COMPLETE",
                    "plan_type": plan_type.upper(),
                    "plan_id": plan_id,
                    "artifact_uri": body.artifact_uri,
                    "metrics": body.metrics,
                },
            )
            lifecycle_resumed = True
            payload["current_phase"] = result.get("current_phase")
        except Exception:
            logger.warning(
                "external_execution_auto_resume_failed",
                plan_type=plan_type,
                plan_id=plan_id,
                exc_info=True,
            )

    return _envelope(
        request,
        {
            "plan_type": plan_type.upper(),
            "plan_id": plan_id,
            "status": body.status.upper(),
            "lifecycle_resumed": lifecycle_resumed,
            "current_phase": payload.get("current_phase"),
        },
        "external execution callback recorded",
    )


@internal_router.post("/repair/{repair_plan_id}/complete")
async def repair_complete(
    repair_plan_id: str,
    request: Request,
    body: RepairCompleteRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """外部数据/管道修复完成回调。

    修复团队完成修复后调用此端点，通知 lifecycle 继续。
    """
    try:
        from ..services.workflow.checkpointer_manager import get_checkpointer
        from ..services.workflow.workflow_service import WorkflowService

        lifecycle_run_id = request.query_params.get("lifecycle_run_id", "")
        body = body or RepairCompleteRequest(repair_plan_id=repair_plan_id)
        await IterationRepo(db).save_external_execution_callback(
            repair_plan_id,
            body.status.upper(),
            {
                "plan_type": "REPAIR",
                "plan_id": repair_plan_id,
                **body.model_dump(mode="json"),
            },
        )

        await db.commit()
        checkpointer = get_checkpointer()
        service = WorkflowService(db, checkpointer)

        if lifecycle_run_id:
            result = await service.resume(
                lifecycle_run_id,
                decision="approved",
                resume_payload={
                    "decision": "approved",
                    "resume_type": "REPAIR_COMPLETE",
                    "repair_plan_id": repair_plan_id,
                },
            )
            return _envelope(
                request,
                {
                    "repair_plan_id": repair_plan_id,
                    "lifecycle_resumed": True,
                    "current_phase": result.get("current_phase"),
                },
                "repair completed and lifecycle resumed",
            )

        return _envelope(
            request,
            {"repair_plan_id": repair_plan_id, "lifecycle_resumed": False},
            "repair recorded (no lifecycle to resume)",
        )
    except Exception as exc:
        logger.warning("repair_complete_failed plan=%s err=%s", repair_plan_id, exc)
        return _envelope(
            request,
            {"repair_plan_id": repair_plan_id, "error": str(exc)},
            "repair recorded with warning",
        )


@internal_router.post("/deployment/{deployment_id}/callback")
async def deployment_callback(
    deployment_id: str,
    request: Request,
    body: DeploymentCallbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """External deployment system reports a stage result."""
    record = {
        "deployment_id": deployment_id,
        **body.model_dump(mode="json"),
        "dispatch_mode": "EXTERNAL_HTTP_CALLBACK",
        "external_response": body.result_json,
    }
    await IterationRepo(db).save_deployment_record(record)
    return _envelope(
        request,
        {
            "deployment_id": deployment_id,
            "deployment_stage": body.deployment_stage,
            "deployment_decision": body.deployment_decision,
            "status": body.status,
        },
        "deployment callback recorded",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# P0: 部署记录查询接口
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/deployments")
async def list_deployments(
    request: Request,
    model_id: str | None = None,
    status: str | None = None,
    current_stage: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """列出部署记录。支持按 model_id / status / current_stage 过滤。"""
    repo = IterationRepo(db)
    items = await repo.list_deployments(
        model_id=model_id, status=status, current_stage=current_stage,
        limit=limit, offset=offset,
    )
    total = await repo.count_deployments(
        model_id=model_id, status=status, current_stage=current_stage,
    )
    return _envelope(request, {"items": items, "total": total})


@router.get("/deployments/{deployment_id}")
async def get_deployment(
    deployment_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """获取单个部署记录详情，包含所有阶段历史。"""
    repo = IterationRepo(db)
    deployment = await repo.get_deployment(deployment_id)
    if deployment is None:
        raise NotFoundError("部署记录不存在")
    stages = await repo.get_deployment_stages(deployment_id)
    return _envelope(request, {
        "deployment": deployment,
        "stages": stages,
    })


@router.get("/deployments/{deployment_id}/stages")
async def get_deployment_stages(
    deployment_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """获取部署的所有阶段历史。"""
    repo = IterationRepo(db)
    deployment = await repo.get_deployment(deployment_id)
    if deployment is None:
        raise NotFoundError("部署记录不存在")
    stages = await repo.get_deployment_stages(deployment_id)
    return _envelope(request, {"deployment_id": deployment_id, "stages": stages})


# ═══════════════════════════════════════════════════════════════════════════════
# P4: 部署回滚接口
# ═══════════════════════════════════════════════════════════════════════════════

class RollbackRequest(BaseModel):
    reason: str = Field(default="MANUAL_ROLLBACK", min_length=1)
    rollback_target: str | None = None
    updated_by: str = "admin"


@router.post("/deployments/{deployment_id}/rollback")
async def rollback_deployment(
    deployment_id: str,
    request: Request,
    body: RollbackRequest = RollbackRequest(),
    db: AsyncSession = Depends(get_db),
):
    """触发部署回滚 — 恢复 champion，challenger 流量归 0。"""
    from ..services.iteration.deployment_safety_service import DeploymentSafetyService

    repo = IterationRepo(db)
    deployment = await repo.get_deployment(deployment_id)
    if deployment is None:
        raise NotFoundError("部署记录不存在")

    svc = DeploymentSafetyService(db)
    result = await svc.rollback(
        deployment=deployment,
        reason=body.reason,
        rollback_target=body.rollback_target,
        updated_by=body.updated_by,
    )
    await db.commit()
    return _envelope(request, result, "deployment rolled back")


# ═══════════════════════════════════════════════════════════════════════════════
# T4-GAP-06: 50 模型批量部署管控
# ═══════════════════════════════════════════════════════════════════════════════

class BatchDeploymentRequest(BaseModel):
    deployment_ids: list[str] = Field(min_length=1, max_length=50)
    updated_by: str = "admin"


class BatchAdvanceRequest(BatchDeploymentRequest):
    resume_lifecycle: bool = True


@router.post("/deployments/batch/advance")
async def batch_advance_deployments(
    request: Request,
    body: BatchAdvanceRequest,
    db: AsyncSession = Depends(get_db),
):
    """Advance up to 50 deployment lifecycles."""
    from ..services.workflow.checkpointer_manager import get_checkpointer
    from ..services.workflow.workflow_service import WorkflowService

    repo = IterationRepo(db)
    checkpointer = get_checkpointer()
    results: list[dict] = []
    succeeded = 0
    failed = 0

    for did in body.deployment_ids:
        try:
            deployment = await repo.get_deployment(did)
            if not deployment:
                results.append({"deployment_id": did, "status": "failed", "error": "not_found"})
                failed += 1
                continue

            previous_stage = deployment.get("current_stage")
            lrid = deployment.get("lifecycle_run_id")
            if not lrid:
                results.append({"deployment_id": did, "status": "failed", "error": "no_lifecycle_run"})
                failed += 1
                continue

            if body.resume_lifecycle:
                service = WorkflowService(db, checkpointer)
                await service.resume(
                    str(lrid),
                    decision="approved",
                    resume_payload={"decision": "approved", "resume_type": "BATCH_ADVANCE"},
                )

            refreshed = await repo.get_deployment(did) or deployment
            results.append({
                "deployment_id": did,
                "status": "advanced",
                "model_id": refreshed.get("model_id"),
                "previous_stage": previous_stage,
                "stage": refreshed.get("current_stage"),
            })
            succeeded += 1
        except Exception as exc:
            results.append({"deployment_id": did, "status": "failed", "error": str(exc)})
            failed += 1

    return _envelope(request, {
        "total": len(body.deployment_ids),
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
    }, f"batch advance: {succeeded}/{len(body.deployment_ids)} succeeded")


@router.post("/deployments/batch/rollback")
async def batch_rollback_deployments(
    request: Request,
    body: BatchDeploymentRequest,
    db: AsyncSession = Depends(get_db),
):
    """Rollback up to 50 deployments."""
    from ..services.iteration.deployment_safety_service import DeploymentSafetyService

    repo = IterationRepo(db)
    results: list[dict] = []
    succeeded = 0
    failed = 0

    for did in body.deployment_ids:
        try:
            deployment = await repo.get_deployment(did)
            if not deployment:
                results.append({"deployment_id": did, "status": "failed", "error": "not_found"})
                failed += 1
                continue

            svc = DeploymentSafetyService(db)
            result = await svc.rollback(
                deployment=deployment,
                reason=f"BATCH_ROLLBACK by {body.updated_by}",
                updated_by=body.updated_by,
            )
            results.append({
                "deployment_id": did,
                "status": "rolled_back",
                "model_id": deployment.get("model_id"),
                "rollback_target": result.get("rollback_target"),
            })
            succeeded += 1
        except Exception as exc:
            results.append({"deployment_id": did, "status": "failed", "error": str(exc)})
            failed += 1

    await db.commit()
    return _envelope(request, {
        "total": len(body.deployment_ids),
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
    }, f"batch rollback: {succeeded}/{len(body.deployment_ids)} succeeded")


# ═══════════════════════════════════════════════════════════════════════════════
# T4-GAP-01: 模型比对 API
# ═══════════════════════════════════════════════════════════════════════════════

class ComparisonRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=100)
    champion_version: str = Field(default="champion_v1")
    challenger_version: str = Field(default="challenger_v1")
    lifecycle_run_id: str | None = None
    qualification_run_id: str | None = None
    champion_scores: list[float] = Field(min_length=1, max_length=100000)
    challenger_scores: list[float] = Field(min_length=1, max_length=100000)
    labels: list[int] = Field(min_length=1, max_length=100000)


@router.post("/comparisons")
async def create_comparison(
    request: Request,
    body: ComparisonRequest,
    db: AsyncSession = Depends(get_db),
):
    """T4-GAP-01: champion vs challenger 10-metric comparison."""
    import numpy as np
    from ..services.iteration.model_comparison_service import ModelComparisonService

    if len(body.champion_scores) != len(body.challenger_scores) or len(body.champion_scores) != len(body.labels):
        raise ValidationAppError("INVALID_INPUT", "scores and labels must have same length")

    svc = ModelComparisonService()
    report = svc.compare(
        y_true=np.array(body.labels),
        champion_scores=np.array(body.champion_scores),
        challenger_scores=np.array(body.challenger_scores),
        model_id=body.model_id,
        champion_version=body.champion_version,
        challenger_version=body.challenger_version,
        lifecycle_run_id=body.lifecycle_run_id,
        qualification_run_id=body.qualification_run_id,
    )

    await IterationRepo(db).save_comparison_report(report)
    await db.commit()

    return _envelope(request, report.model_dump(mode="json"), "comparison completed")


@router.get("/comparisons/{comparison_id}")
async def get_comparison(
    comparison_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Get a model comparison report."""
    report = await IterationRepo(db).get_comparison_report(comparison_id)
    if report is None:
        raise NotFoundError("comparison report not found")
    return _envelope(request, report)


# ═══════════════════════════════════════════════════════════════════════════════
# T4-GAP-04: 部署健康检查 API
# ═══════════════════════════════════════════════════════════════════════════════

class HealthCheckTriggerRequest(BaseModel):
    deployment_id: str = ""
    stage: str = ""
    model_id: str = ""
    lifecycle_run_id: str | None = None
    health_metrics: dict = Field(default_factory=dict)


@router.post("/deployments/{deployment_id}/health-checks")
async def create_health_check(
    deployment_id: str,
    request: Request,
    body: HealthCheckTriggerRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Run a structured health check for a deployment stage."""
    from ..services.deployment.deployment_health_check_service import DeploymentHealthCheckService

    repo = IterationRepo(db)
    deployment = await repo.get_deployment(deployment_id)
    if deployment is None:
        raise NotFoundError("deployment not found")

    b = body or HealthCheckTriggerRequest(deployment_id=deployment_id, stage=deployment.get("current_stage", ""))
    report = await DeploymentHealthCheckService().check(
        deployment_id=deployment_id,
        stage=b.stage or deployment.get("current_stage", ""),
        health_metrics=b.health_metrics,
        lifecycle_run_id=b.lifecycle_run_id,
        model_id=b.model_id or (deployment.get("model_id") or ""),
    )

    await repo.save_deployment_stage_record({
        "deployment_id": deployment_id,
        "stage": report.stage,
        "decision": "HEALTH_CHECK",
        "status": "SUCCEEDED",
        "health_json": report.model_dump(mode="json"),
        "result_json": {"report_id": report.report_id, "passed": report.passed},
    })
    await db.commit()

    return _envelope(request, report.model_dump(mode="json"), "health check completed")


@router.get("/deployments/{deployment_id}/health-checks")
async def get_health_checks(
    deployment_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Get all health check records for a deployment."""
    repo = IterationRepo(db)
    deployment = await repo.get_deployment(deployment_id)
    if deployment is None:
        raise NotFoundError("deployment not found")
    stages = await repo.get_deployment_stages(deployment_id)
    health_checks = [s for s in stages if s.get("decision") == "HEALTH_CHECK"]
    return _envelope(request, {"deployment_id": deployment_id, "health_checks": health_checks})


@router.get("/deployments/{deployment_id}/rollback-events")
async def get_rollback_events(
    deployment_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Get rollback events for a deployment."""
    repo = IterationRepo(db)
    deployment = await repo.get_deployment(deployment_id)
    if deployment is None:
        raise NotFoundError("deployment not found")
    events = await repo.get_rollback_events(deployment_id)
    return _envelope(request, {
        "deployment_id": deployment_id,
        "model_id": deployment.get("model_id"),
        "champion_version": deployment.get("champion_version"),
        "rollback_count": len(events),
        "events": events,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# P3: 模型路由配置查询
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/routing-configs/{model_id}")
async def get_routing_config(
    model_id: str,
    request: Request,
    environment: str = "PROD",
    db: AsyncSession = Depends(get_db),
):
    """查询模型的当前路由配置（哪个版本接收多少流量）。"""
    repo = IterationRepo(db)
    state = await repo.get_model_deployment_state(model_id, environment)
    if state is None:
        return _envelope(request, {
            "model_id": model_id,
            "environment": environment,
            "active_version_code": None,
            "stable_version_code": None,
            "challenger_version_code": None,
            "challenger_traffic_ratio": 0,
            "message": "no routing config found — model has not been deployed yet",
        })
    return _envelope(request, state)
