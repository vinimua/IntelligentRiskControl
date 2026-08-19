"""任务三：根因驱动修复决策 API。

本路由只提供确定性规则、持久化和跨模块合同，不负责 Agent 或 LangGraph。
"""

from datetime import UTC, datetime
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, Query, Request

logger = structlog.get_logger(__name__)
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from packages.models.common.enums import DataTrack, ProposalStatus, ReviewDecision
from packages.models.iteration import (
    A7DecisionEnvelope,
    DecisionInput,
    ManualReviewReport,
    ManualReviewSubmission,
    QualificationInput,
    RepairCaseRecord,
)
from packages.models.callbacks.training_callback import TrainingCallback
from packages.models.iteration.training_job import TrainingJobInput
from packages.models.iteration.training_plan import TrainingPlan

from ..core.exceptions import (
    ConflictError,
    NotFoundError,
    ValidationAppError,
    request_trace_id,
)
from ..database import get_db
from ..neo4j_db import get_neo4j_driver
from ..repositories.diagnosis_repo import DiagnosisRepo
from ..repositories.iteration_repo import IterationRepo
from ..services.knowledge_service import KnowledgeService
from ..services.iteration import (
    FailureAttributionService,
    ModelTaskInterfaceService,
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
    # 最大业务轮次统一为 2（A7 定稿）
    business_round: int = Field(default=1, ge=1, le=2)
    data_snapshot_ids: list[str] = Field(min_length=1)
    label_versions: list[str] = Field(min_length=1)


class QualificationRequest(BaseModel):
    """外部资格验证请求 —— 只接收身份字段。

    所有资格指标（score_psi / train_valid_gap / 目标恢复 / 数据复现 /
    判别 / 校准 / OOT / 特征级 PSI）均由服务端从受信任的
    experiment_json（W3 验证结果 + OOT 写回）与监测漂移数据加载，
    调用方无法伪造任何门禁输入。
    """

    qualification_run_id: str
    iteration_run_id: str
    experiment_id: str
    candidate_version: str
    data_track: DataTrack = DataTrack.NATURAL


def _qualification_input_from_experiment(
    body: QualificationRequest,
    experiment_json: dict,
    feature_psi: dict[str, float],
) -> QualificationInput:
    """从受信任的 experiment_json 构建内部 QualificationInput。

    与 Graph 资格节点共用 build_qualification_input —— 单一构建入口，
    目标恢复字段完整读取、必填证据缺失拒绝评估（禁止 fail-open）。
    """
    from ..services.iteration.qualification_service import (
        build_qualification_input,
    )

    return build_qualification_input(
        qualification_run_id=body.qualification_run_id,
        iteration_run_id=body.iteration_run_id,
        experiment_id=body.experiment_id,
        candidate_version=body.candidate_version,
        experiment_json=experiment_json,
        feature_psi=feature_psi,
    )


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


class ProactiveReleaseRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=100)
    challenger_version: str = Field(min_length=1, max_length=100)
    champion_version: str | None = None
    rollback_target: str = Field(min_length=1, max_length=100)
    release_type: str = "NEW_SCENARIO"
    initial_stage: str = "SHADOW"
    health_status: str = "PASSED"
    health_metrics: dict[str, object] = Field(default_factory=dict)
    artifact_uri: str | None = None
    updated_by: str = "admin"


class BatchProactiveReleaseRequest(BaseModel):
    items: list[ProactiveReleaseRequest] = Field(min_length=1, max_length=50)


class Task4PatrolRequest(BaseModel):
    interval_seconds: int = Field(default=300, ge=10, le=86400)
    focus_model_id: str | None = None
    failure_model_id: str | None = None
    persist: bool = True
    updated_by: str = "task4_patrol"


def _envelope(request: Request, data, message: str = "success") -> dict:
    return {
        "success": True,
        "code": "OK",
        "message": message,
        "data": data,
        "trace_id": request_trace_id(request),
    }


def _evaluate_new_scenario_health(body: ProactiveReleaseRequest) -> dict:
    metrics = body.health_metrics or {}
    checks: list[dict] = []
    failures: list[str] = []

    def add_check(code: str, passed: bool, value=None, threshold=None) -> None:
        checks.append({
            "metric_code": code,
            "passed": passed,
            "value": value,
            "threshold": threshold,
        })
        if not passed:
            failures.append(code)

    health_status = body.health_status.upper()
    add_check(
        "HEALTH_STATUS",
        health_status in {"PASSED", "HEALTHY"},
        health_status,
        "PASSED|HEALTHY",
    )
    add_check("ROLLBACK_TARGET_CONFIGURED", bool(body.rollback_target), body.rollback_target, "required")

    if "artifact_loadable" in metrics:
        add_check("ARTIFACT_LOADABLE", metrics.get("artifact_loadable") is not False, metrics.get("artifact_loadable"), True)
    if "schema_consistency" in metrics:
        add_check("SCHEMA_CONSISTENCY", metrics.get("schema_consistency") is not False, metrics.get("schema_consistency"), True)
    if "inference_smoke_passed" in metrics:
        add_check("INFERENCE_SMOKE", metrics.get("inference_smoke_passed") is not False, metrics.get("inference_smoke_passed"), True)

    numeric_rules = [
        ("validation_auc", 0.70, "gte"),
        ("validation_ks", 0.20, "gte"),
        ("score_psi", 0.25, "lte"),
        ("data_quality_score", 0.80, "gte"),
        ("sample_size", 50, "gte"),
    ]
    for code, threshold, direction in numeric_rules:
        if code not in metrics:
            continue
        try:
            value = float(metrics.get(code))
        except (TypeError, ValueError):
            add_check(code.upper(), False, metrics.get(code), threshold)
            continue
        passed = value >= threshold if direction == "gte" else value <= threshold
        add_check(code.upper(), passed, value, threshold)

    return {
        "release_type": body.release_type.upper(),
        "health_status": health_status,
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "evaluated_by": "NewScenarioPredeployHealth_V1",
    }


def _jsonable(data):
    import json

    return json.loads(json.dumps(data, ensure_ascii=False, default=str))


async def _create_proactive_release(
    body: ProactiveReleaseRequest,
    db: AsyncSession,
) -> dict:
    import json

    from ..services.iteration.deployment_safety_service import (
        DeploymentSafetyService,
        STAGE_TRAFFIC_RATIO,
    )

    release_type = body.release_type.upper()
    if release_type != "NEW_SCENARIO":
        raise ValidationAppError(
            "UNSUPPORTED_RELEASE_TYPE",
            "only NEW_SCENARIO proactive release is supported without comparison",
        )

    stage = body.initial_stage.upper()
    allowed_initial_stages = {"OFFLINE_VALIDATION", "OOT_GATE", "SHADOW", "CANARY_5"}
    if stage not in allowed_initial_stages:
        raise ValidationAppError(
            "INVALID_INITIAL_STAGE",
            "NEW_SCENARIO can only start from OFFLINE_VALIDATION, OOT_GATE, SHADOW, or CANARY_5",
        )
    if stage not in STAGE_TRAFFIC_RATIO:
        raise ValidationAppError("INVALID_INITIAL_STAGE", f"unsupported stage: {stage}")

    health = _evaluate_new_scenario_health(body)
    if not health["passed"]:
        raise ValidationAppError(
            "NEW_SCENARIO_HEALTH_NOT_PASSED",
            "new scenario model health gate did not pass",
        )

    stable_version = body.champion_version or body.rollback_target
    deployment_id = str(uuid4())
    await db.execute(
        text("""
            INSERT INTO model_registry.models
                (model_id, model_name, model_type, current_champion_version,
                 stable_version, attributes_json)
            VALUES (:mid, :name, 'CREDIT_RISK', :stable, :stable, CAST(:attrs AS JSONB))
            ON CONFLICT (model_id) DO UPDATE SET
                current_champion_version = COALESCE(model_registry.models.current_champion_version, EXCLUDED.current_champion_version),
                stable_version = COALESCE(model_registry.models.stable_version, EXCLUDED.stable_version),
                attributes_json = model_registry.models.attributes_json || EXCLUDED.attributes_json,
                updated_at = NOW()
        """),
        {
            "mid": body.model_id,
            "name": body.model_id,
            "stable": stable_version,
            "attrs": json.dumps({
                "last_release_type": release_type,
                "last_proactive_release_at": datetime.now(UTC).isoformat(),
            }, ensure_ascii=False),
        },
    )
    await db.execute(
        text("""
            INSERT INTO model_registry.model_versions
                (model_id, version_code, role, status, base_version_code,
                 artifact_uri, metrics_json, created_by)
            VALUES (:mid, :ver, 'CHALLENGER', 'VALIDATED', :base, :uri, CAST(:metrics AS JSONB), :by)
            ON CONFLICT (model_id, version_code) DO UPDATE SET
                role = 'CHALLENGER',
                status = 'VALIDATED',
                base_version_code = COALESCE(EXCLUDED.base_version_code, model_registry.model_versions.base_version_code),
                artifact_uri = COALESCE(EXCLUDED.artifact_uri, model_registry.model_versions.artifact_uri),
                metrics_json = model_registry.model_versions.metrics_json || EXCLUDED.metrics_json,
                updated_at = NOW()
        """),
        {
            "mid": body.model_id,
            "ver": body.challenger_version,
            "base": stable_version,
            "uri": body.artifact_uri,
            "metrics": json.dumps({
                "release_type": release_type,
                "predeploy_health": health,
                "health_metrics": body.health_metrics,
            }, ensure_ascii=False, default=str),
            "by": body.updated_by,
        },
    )

    record = {
        "deployment_id": deployment_id,
        "lifecycle_run_id": None,
        "qualification_run_id": None,
        "model_id": body.model_id,
        "champion_version": stable_version,
        "candidate_version": body.challenger_version,
        "deployment_stage": stage,
        "deployment_decision": "ADVANCE_STAGE",
        "status": "RUNNING",
        "dispatch_mode": "PROACTIVE_NEW_SCENARIO",
        "external_task_id": None,
        "health_json": {
            "release_type": release_type,
            "predeploy_health": health,
            "artifact_uri": body.artifact_uri,
            "rollback_target": body.rollback_target,
        },
        "result_json": {
            "deployment_id": deployment_id,
            "release_type": release_type,
            "initial_stage": stage,
            "rollback_target": body.rollback_target,
        },
    }
    repo = IterationRepo(db)
    await repo.save_deployment_record(record)
    ratio = await DeploymentSafetyService(db).update_traffic_ratio(
        model_id=body.model_id,
        stage=stage,
        champion_version=stable_version,
        challenger_version=body.challenger_version,
        updated_by=body.updated_by,
    )
    routing = await repo.get_model_deployment_state(body.model_id)
    deployment = await repo.get_deployment(deployment_id)
    return _jsonable({
        "deployment_id": deployment_id,
        "release_type": release_type,
        "predeploy_health": health,
        "initial_stage": stage,
        "challenger_traffic_ratio": ratio,
        "deployment": deployment,
        "routing": routing,
    })


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


@router.post("/a7/decisions")
async def create_a7_decision(
    request: Request,
    body: A7DecisionEnvelope,
    db: AsyncSession = Depends(get_db),
):
    """严格 A7 决策入口（SIMULATED 模拟回放与 NATURAL 自然上游共用）。

    信封经 Pydantic 严格校验（CONFIRMED 根因/批准授权/规则版本四键/
    W4 隔离）后，L1 确定性选择策略并产出提案。
    """
    proposal, l1 = RepairDecisionService().propose_a7(body)
    risk = RiskAssessmentService().assess(proposal)
    repo = IterationRepo(db)
    await repo.save_proposal(proposal)
    await repo.save_risk(risk)
    return _envelope(
        request,
        {
            "proposal": proposal.model_dump(mode="json"),
            "l1_strategy_decision": l1.model_dump(mode="json"),
            "risk_assessment": risk.model_dump(mode="json"),
        },
        "strict A7 decision created",
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


class StrategyCandidateItem(BaseModel):
    """契约 §5.3 策略候选条目（KG 咨询层）。"""

    strategy_code: str
    recommends_relation_key: str
    mitigates_relation_key: str
    relation_effective_weight_snapshot: float
    historical_effectiveness: float | None = None
    strategy_rank_score: float = 0.0
    rank_score_source: str = "INITIAL_PRIOR"
    support_case_count: int = 0
    total_case_count: int = 0
    natural_case_count: int = 0
    confidence_lower_bound: float = 0.0
    training_cost_level: str = "MEDIUM"
    risk_level: str = "LOW"
    primary_training_mode: str = "FULL_RETRAIN"
    required_context: list[str] = Field(default_factory=list)
    selected: bool = False


class StrategyCandidatesResponse(BaseModel):
    iteration_run_id: str
    root_cause_code: str
    weight_version: str | None = None
    candidates: list[StrategyCandidateItem] = Field(default_factory=list)
    l1_final_strategy_code: str | None = None
    kg_consistency_status: str | None = None
    kg_repair_required: bool = False
    retrieval_degraded: bool = False


class StrategyCandidatesEnvelope(BaseModel):
    """契约 §5.3 响应包络（正式 OpenAPI schema）。"""

    success: bool
    code: str
    message: str
    data: StrategyCandidatesResponse
    trace_id: str


@router.get(
    "/runs/{iteration_run_id}/strategy-candidates",
    response_model=StrategyCandidatesEnvelope,
)
async def get_strategy_candidates(
    iteration_run_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """契约 §5.3：返回 KG 咨询候选、L1 最终策略与一致性状态。"""
    repo = IterationRepo(db)
    run = await repo.get_iteration_run(iteration_run_id)
    if run is None:
        raise NotFoundError("迭代运行不存在")
    proposal = await repo.get_proposal(run["proposal_id"])
    if proposal is None:
        raise NotFoundError("迭代决策不存在")

    # 原始根因码：KG 根因码与诊断层一致（小写），提案里是被归一化的大写码
    root_cause_code = proposal.primary_root_cause_code
    diag_run = None
    if proposal.diagnosis_run_id:
        diag_run = await DiagnosisRepo(db).get_run(proposal.diagnosis_run_id)
        if diag_run and diag_run.get("primary_root_cause_code"):
            root_cause_code = diag_run["primary_root_cause_code"]

    # 与决策时一致的结构化上下文：从持久化数据重建（decay 来自监控 B1 判定、
    # business_round 来自迭代轮次），避免前端查询与决策候选不一致
    decay_degree: str | None = None
    if diag_run and diag_run.get("monitoring_run_id"):
        from ..repositories.monitoring_repo import MonitoringRepo
        mon_run = await MonitoringRepo(db).get_run(diag_run["monitoring_run_id"])
        judgment = mon_run.get("persistence_judgment_json") if mon_run else None
        if isinstance(judgment, str):
            import json as _json
            judgment = _json.loads(judgment)
        if isinstance(judgment, dict):
            decay_degree = judgment.get("decay_degree")

    rounds = await repo.get_iteration_rounds(iteration_run_id)
    business_round = max(1, min(2, len(rounds) + 1))

    _DECAY_LEVELS = {"NONE": 0, "SHORT_TERM_7D": 1, "SUSTAINED_30D": 2, "SEVERE": 3}
    available_context_codes: list[str] = []
    if decay_degree == "SUSTAINED_30D":
        available_context_codes.append("sustained_30d")

    driver = await get_neo4j_driver()
    iteration_ctx = await KnowledgeService(driver).query_iteration_context(
        root_cause_code=root_cause_code,
        diagnosis_run_id=proposal.diagnosis_run_id,
        severity=proposal.primary_root_cause_score,
        decay_level=_DECAY_LEVELS.get(decay_degree),
        business_round=business_round,
        available_context_codes=available_context_codes,
    )

    selected_code = proposal.selected_strategy_code
    candidates = [
        StrategyCandidateItem(
            strategy_code=c.strategy_code,
            recommends_relation_key=c.recommends_relation_key,
            mitigates_relation_key=c.mitigates_relation_key,
            relation_effective_weight_snapshot=c.relation_effective_weight_snapshot,
            historical_effectiveness=c.historical_effectiveness,
            strategy_rank_score=c.strategy_rank_score,
            rank_score_source=c.rank_score_source,
            support_case_count=c.support_case_count,
            total_case_count=c.total_case_count,
            natural_case_count=c.natural_case_count,
            confidence_lower_bound=c.confidence_lower_bound,
            training_cost_level=c.training_cost_level,
            risk_level=c.risk_level,
            primary_training_mode=c.primary_training_mode,
            required_context=c.required_context,
            selected=(c.strategy_code == selected_code),
        )
        for c in iteration_ctx.strategy_candidates
    ]

    return StrategyCandidatesEnvelope(
        success=True,
        code="OK",
        message="success",
        data=StrategyCandidatesResponse(
            iteration_run_id=iteration_run_id,
            root_cause_code=root_cause_code,
            weight_version=iteration_ctx.weight_version,
            candidates=candidates,
            l1_final_strategy_code=selected_code,
            kg_consistency_status=proposal.kg_consistency_status,
            kg_repair_required=proposal.kg_repair_required,
            retrieval_degraded=iteration_ctx.retrieval_degraded,
        ),
        trace_id=request_trace_id(request),
    )


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


@router.get("/models/{model_id}/task-interface")
async def get_model_task_interface(
    model_id: str,
    request: Request,
    champion_version: str = Query("champion_v1", min_length=1),
    model_type: str | None = Query(None),
    algorithm_family: str | None = Query(None),
):
    """Return task-type metrics and risk guardrails for adaptive iteration."""

    payload = ModelTaskInterfaceService.summarize(
        model_id=model_id,
        champion_version=champion_version,
        model_type=model_type,
        algorithm_family=algorithm_family,
    ).model_dump(mode="json")
    return _envelope(request, payload)


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


class DispatchTrainingJobRequest(BaseModel):
    """人工/受控派发训练任务（外部执行链路）。"""

    model_algorithm: str = "lightgbm"


@router.post("/plans/{training_plan_id}/dispatch")
async def dispatch_training_plan(
    training_plan_id: str,
    request: Request,
    body: DispatchTrainingJobRequest,
    db: AsyncSession = Depends(get_db),
):
    """受控链路：按已批准的训练计划创建训练 Job 并真正派发 Celery Worker。

    训练窗口、特征筛选清单、客群权重、训练模式全部来自 TrainingPlan
    合同（服务端受信任数据），调用方只提供算法。
    """
    repo = IterationRepo(db)
    plan_payload = await repo.get_training_plan(training_plan_id)
    if plan_payload is None:
        raise NotFoundError("训练计划不存在")
    plan = TrainingPlan.model_validate(plan_payload)
    if plan.status.value not in {"READY", "APPROVED"}:
        raise ValidationAppError(
            "TRAINING_PLAN_NOT_READY",
            f"训练计划状态 {plan.status.value} 不允许派发",
        )
    proposal = await repo.get_proposal(plan.proposal_id)
    if proposal is None:
        raise NotFoundError("决策建议不存在")

    job_input = TrainingJobInput(
        training_job_id=str(uuid4()),
        idempotency_key=(
            f"{plan.iteration_run_id}:round-{plan.business_round}"
            f":exp-{plan.experiment_id}"
        ),
        model_id=plan.model_id,
        iteration_run_id=plan.iteration_run_id,
        training_plan_id=plan.training_plan_id,
        experiment_id=plan.experiment_id,
        business_round=plan.business_round,
        strategy_code=plan.strategy_code,
        training_window_ids=plan.windows.training_window_ids,
        validation_window_ids=plan.windows.validation_window_ids,
        oot_window_id=plan.windows.oot_window_id,
        data_snapshot_ids=plan.data_snapshot_ids,
        label_versions=plan.label_versions,
        sample_weight_policy=plan.sample_weight_policy,
        feature_schema_version=plan.feature_schema_version,
        preprocessing_version=plan.preprocessing_version,
        algorithm=body.model_algorithm or plan.algorithm,
        hyperparameters=plan.hyperparameter_space or {},
        target_metrics=plan.target_metric_codes or ["AUC", "KS"],
        qualification_rule_version=plan.qualification_rule_version,
        base_model_version=plan.frozen_champion_version,
        seed=plan.random_seed,
        training_mode=plan.training_mode,
        unstable_feature_codes=plan.unstable_feature_codes,
        selected_feature_codes=plan.selected_feature_codes,
        feature_selection_artifact_uri=plan.feature_selection_artifact_uri,
        artifact_output_uri=(
            f"s3://riskitem/challengers/{plan.model_id}"
            f"/{plan.iteration_run_id}/round-{plan.business_round}"
        ),
    )
    created, _row = await repo.create_training_job(job_input)
    await db.commit()

    from ..config import settings
    from ..services.workflow.executors import dispatch_training_job

    celery_app = None
    if settings.workflow_use_celery:
        from workers.app import app as celery_app
    job_dict = job_input.model_dump()
    job_dict["lifecycle_run_id"] = proposal.lifecycle_run_id
    dispatch_result = await dispatch_training_job(job_dict, celery_app=celery_app)

    return _envelope(
        request,
        {
            "training_job_id": job_input.training_job_id,
            "created": created,
            "dispatched": dispatch_result.get("dispatched"),
            "celery_task_id": dispatch_result.get("celery_task_id"),
            "strategy_code": plan.strategy_code,
            "training_mode": plan.training_mode,
            "training_window_ids": plan.windows.training_window_ids,
            "selected_feature_codes": plan.selected_feature_codes,
        },
        "training job created and dispatched",
    )


@internal_router.post("/experiments/{experiment_id}/oot")
async def run_experiment_oot(
    experiment_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """受控链路：对实验的冻结 Challenger 执行真实 W4 OOT 验证并回写。

    结果（oot_passed / w4_available / candidate_frozen_before_oot /
    oot_auc / oot_ks / oot_psi）合并进 experiment_json 顶层 ——
    最终资格（Gate 6）读取的正是这些字段。
    """
    import json as _json

    from sqlalchemy import text as _sql_text

    from ..services.deployment.deployment_oot_service import (
        load_frozen_challenger,
        run_oot_validation,
    )

    repo = IterationRepo(db)
    experiment = await repo.get_experiment(experiment_id)
    if experiment is None:
        raise NotFoundError("训练实验不存在")
    if experiment["technical_status"] != "SUCCEEDED":
        raise ValidationAppError(
            "EXPERIMENT_NOT_TRAINED",
            "实验技术状态必须为 SUCCEEDED 才能执行 W4 OOT",
        )
    if not experiment.get("candidate_version"):
        raise ValidationAppError(
            "CANDIDATE_MISSING",
            "实验缺少 candidate_version，无法加载冻结 Challenger",
        )

    # experiments 表无 model_id 列，从 iteration_runs 取（首轮冻结信息）
    run_record = await repo.get_iteration_run(experiment["iteration_run_id"])
    if run_record is None:
        raise NotFoundError("迭代运行不存在")
    model_id = run_record["model_id"]

    # Worker 产物路径是 challengers/{model_id}/{lifecycle_run_id}/{candidate}，
    # lifecycle_run_id 从 TrainingPlan → Proposal 获取
    plan_payload = await repo.get_training_plan(experiment["training_plan_id"])
    proposal = await repo.get_proposal(
        TrainingPlan.model_validate(plan_payload).proposal_id
    ) if plan_payload else None
    if proposal is None:
        raise NotFoundError("决策建议不存在，无法定位冻结 Challenger 产物路径")
    lifecycle_run_id = proposal.lifecycle_run_id

    frozen = load_frozen_challenger(
        model_id, lifecycle_run_id,
        experiment["candidate_version"],
    )
    if not frozen["loaded"]:
        raise ValidationAppError(
            "FROZEN_CHALLENGER_LOAD_FAILED",
            f"冻结 Challenger 加载失败: {frozen.get('load_errors', [])}",
        )
    oot_result = run_oot_validation(
        frozen["model"], frozen["feature_cols"],
        model_id=model_id,
        lifecycle_run_id=experiment["iteration_run_id"],
        candidate_version=experiment["candidate_version"],
    )

    await db.execute(
        _sql_text("""
            UPDATE iteration.experiments
            SET experiment_json = experiment_json || CAST(:payload AS JSONB),
                updated_at = NOW()
            WHERE experiment_id = :eid
        """),
        {
            "eid": experiment_id,
            "payload": _json.dumps({
                "oot_passed": oot_result["oot_passed"],
                "oot_auc": oot_result["oot_auc"],
                "oot_ks": oot_result["oot_ks"],
                "oot_psi": oot_result.get("oot_psi"),
                "candidate_frozen_before_oot": True,
                "w4_available": oot_result["w4_available"],
            }),
        },
    )
    await db.commit()

    return _envelope(
        request,
        {
            "experiment_id": experiment_id,
            "oot_passed": oot_result["oot_passed"],
            "w4_available": oot_result["w4_available"],
            "oot_auc": oot_result["oot_auc"],
            "oot_ks": oot_result["oot_ks"],
        },
        "W4 OOT validation completed and written back",
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

    # 所有资格指标服务端加载：只信任 experiment_json（W3 验证结果 +
    # OOT 写回）与监测漂移数据，调用方只能提交身份字段
    experiment_json = experiment.get("experiment_json") or {}
    feature_psi: dict[str, float] = {}
    if proposal.monitoring_run_id:
        from ..repositories.monitoring_repo import MonitoringRepo
        drift_rows = await MonitoringRepo(db).get_feature_drift_by_run(
            proposal.monitoring_run_id
        )
        for row in drift_rows:
            fname = row.get("feature_name")
            psi = row.get("psi")
            if fname and psi is not None:
                feature_psi[str(fname)] = max(
                    feature_psi.get(str(fname), 0.0), float(psi)
                )
    try:
        qualification_input = _qualification_input_from_experiment(
            body, experiment_json, feature_psi,
        )
    except ValueError as evidence_exc:
        # 证据不完整：拒绝评估，不静默降级
        raise ValidationAppError(
            "QUALIFICATION_EVIDENCE_INCOMPLETE",
            str(evidence_exc),
        ) from evidence_exc
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


@router.get("/task4/parallel-control")
async def get_task4_parallel_control(
    request: Request,
    limit: int = Query(default=50, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    """任务4并行管控总览。

    聚合最近每个模型的一条部署记录、灰度路由状态、阶段分布和回滚事件，
    用于验收“不少于 50 个模型的并行管控及分步上线”。
    """
    rows_result = await db.execute(
        text("""
            WITH latest_deployments AS (
                SELECT DISTINCT ON (d.model_id)
                    d.deployment_id,
                    d.lifecycle_run_id,
                    d.model_id,
                    d.champion_version,
                    d.candidate_version,
                    d.current_stage,
                    d.decision,
                    d.status,
                    d.created_at,
                    d.updated_at
                FROM iteration.deployment_records d
                WHERE d.model_id IS NOT NULL
                ORDER BY d.model_id, d.updated_at DESC NULLS LAST, d.created_at DESC NULLS LAST
            ),
            limited AS (
                SELECT *
                FROM latest_deployments
                ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
                LIMIT :limit
            )
            SELECT
                l.*,
                s.active_version_code,
                s.stable_version_code,
                s.challenger_version_code,
                s.challenger_traffic_ratio,
                s.state_version,
                ph.last_patrol_at,
                ph.last_patrol_status,
                ph.last_patrol_decision,
                COALESCE(rb.rollback_count, 0) AS rollback_count
            FROM limited l
            LEFT JOIN model_registry.model_deployment_state s
              ON s.model_id = l.model_id AND s.environment = 'PROD'
            LEFT JOIN LATERAL (
                SELECT
                    sr.created_at AS last_patrol_at,
                    sr.status AS last_patrol_status,
                    sr.decision AS last_patrol_decision
                FROM iteration.deployment_stage_records sr
                WHERE sr.deployment_id = l.deployment_id
                  AND sr.decision IN ('HEALTH_CHECK', 'ROLLBACK', 'HOLD', 'LIFECYCLE_ALERT')
                ORDER BY sr.created_at DESC
                LIMIT 1
            ) ph ON TRUE
            LEFT JOIN (
                SELECT d.model_id, COUNT(*) AS rollback_count
                FROM iteration.deployment_stage_records sr
                JOIN iteration.deployment_records d ON d.deployment_id = sr.deployment_id
                WHERE sr.decision = 'ROLLBACK'
                GROUP BY d.model_id
            ) rb ON rb.model_id = l.model_id
            ORDER BY l.updated_at DESC NULLS LAST, l.created_at DESC NULLS LAST
        """),
        {"limit": limit},
    )
    items = [dict(row) for row in rows_result.mappings()]

    total_models_result = await db.execute(
        text("SELECT COUNT(*) FROM model_registry.models")
    )
    total_registered_models = int(total_models_result.scalar() or 0)

    total_deployed_result = await db.execute(
        text("SELECT COUNT(DISTINCT model_id) FROM iteration.deployment_records WHERE model_id IS NOT NULL")
    )
    total_deployed_models = int(total_deployed_result.scalar() or 0)

    stage_distribution: dict[str, int] = {}
    status_distribution: dict[str, int] = {}
    rollback_models = 0
    canary_models = 0
    production_models = 0
    active_challenger_models = 0

    for item in items:
        stage = str(item.get("current_stage") or "UNKNOWN")
        status = str(item.get("status") or "UNKNOWN")
        stage_distribution[stage] = stage_distribution.get(stage, 0) + 1
        status_distribution[status] = status_distribution.get(status, 0) + 1
        if int(item.get("rollback_count") or 0) > 0 or item.get("decision") == "ROLLBACK":
            rollback_models += 1
        if stage.startswith("CANARY"):
            canary_models += 1
        if stage == "PRODUCTION":
            production_models += 1
        if (
            status not in {"PROMOTED", "ROLLED_BACK", "FAILED"}
            and item.get("challenger_version_code")
            and float(item.get("challenger_traffic_ratio") or 0) > 0
        ):
            active_challenger_models += 1

    summary = {
        "target_parallel_models": 50,
        "listed_models": len(items),
        "total_registered_models": total_registered_models,
        "total_deployed_models": total_deployed_models,
        "coverage_passed": total_deployed_models >= 50,
        "stage_distribution": stage_distribution,
        "status_distribution": status_distribution,
        "canary_models": canary_models,
        "production_models": production_models,
        "active_challenger_models": active_challenger_models,
        "rollback_models": rollback_models,
        "rollback_ready": all(
            bool(item.get("stable_version_code") or item.get("champion_version"))
            for item in items
        ) if items else False,
        "batch_action_limit": 50,
    }
    return _envelope(request, {"summary": summary, "items": items})


@router.post("/task4/patrol/run-once")
async def run_task4_patrol_once(
    request: Request,
    body: Task4PatrolRequest = Task4PatrolRequest(),
    db: AsyncSession = Depends(get_db),
):
    """Run one scheduled patrol by reusing the lifecycle Alert path.

    Patrol does not invent health metrics. For each deployed model it starts a
    lifecycle run, lets the normal monitoring/B1 Alert logic decide whether the
    model is abnormal, and only then performs deployment protection such as
    rollback to the stable version.
    """
    from ..services.iteration.deployment_safety_service import DeploymentSafetyService
    from ..services.workflow.checkpointer_manager import get_checkpointer
    from ..services.workflow.workflow_service import WorkflowService

    focus_model_id = body.focus_model_id or body.failure_model_id

    rows_result = await db.execute(
        text("""
            WITH latest_deployments AS (
                SELECT DISTINCT ON (d.model_id)
                    d.*,
                    s.active_version_code,
                    s.stable_version_code,
                    s.challenger_version_code,
                    s.challenger_traffic_ratio
                FROM iteration.deployment_records d
                LEFT JOIN model_registry.model_deployment_state s
                  ON s.model_id = d.model_id AND s.environment = 'PROD'
                WHERE d.model_id IS NOT NULL
                ORDER BY d.model_id, d.updated_at DESC NULLS LAST, d.created_at DESC NULLS LAST
            )
            SELECT *
            FROM latest_deployments
            WHERE (CAST(:focus AS text) IS NULL OR model_id = CAST(:focus AS text))
            ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
            LIMIT 50
        """),
        {"focus": focus_model_id},
    )
    deployments = [dict(row) for row in rows_result.mappings()]
    safety = DeploymentSafetyService(db)
    workflow = WorkflowService(db, get_checkpointer())

    results: list[dict] = []
    summary = {"checked": 0, "healthy": 0, "alerted": 0, "repairing": 0, "rolled_back": 0, "skipped": 0}

    for deployment in deployments:
        model_id = str(deployment.get("model_id") or "")
        status = str(deployment.get("status") or "").upper()
        stage = str(deployment.get("current_stage") or "PRODUCTION").upper()

        if status in {"FAILED", "ABORTED"}:
            summary["skipped"] += 1
            results.append({
                "deployment_id": str(deployment.get("deployment_id")),
                "model_id": model_id,
                "stage": stage,
                "patrol_status": "SKIPPED",
                "action": "SKIP_UNAVAILABLE_DEPLOYMENT",
            })
            continue

        champion_version = (
            deployment.get("active_version_code")
            or deployment.get("current_champion")
            or deployment.get("champion_version")
            or deployment.get("stable_version_code")
            or "champion_v1"
        )
        summary["checked"] += 1
        lifecycle_result = await workflow.start(
            model_id=model_id,
            champion_version=str(champion_version),
            trigger_type="SCHEDULED_TRIGGER",
        )
        state = lifecycle_result.get("state") or {}
        current_phase = lifecycle_result.get("current_phase") or state.get("current_phase")
        has_alerts = bool(state.get("has_alerts"))
        trigger_diagnosis = bool(state.get("trigger_diagnosis"))
        alert_count = int(state.get("alert_count") or 0)
        should_protect = has_alerts or trigger_diagnosis or alert_count > 0

        if should_protect:
            summary["alerted"] += 1
            rollback_target = (
                deployment.get("stable_version_code")
                or deployment.get("champion_version")
                or deployment.get("active_version_code")
            )
            active_version = deployment.get("active_version_code") or deployment.get("current_champion")
            can_rollback = bool(rollback_target and active_version and str(rollback_target) != str(active_version))
            if can_rollback:
                rollback_result = await safety.rollback(
                    deployment=deployment,
                    reason=f"TASK4_PATROL_ALERT:lifecycle={lifecycle_result.get('lifecycle_run_id')}",
                    rollback_target=str(rollback_target),
                    updated_by=body.updated_by,
                )
                summary["rolled_back"] += 1
                action = "ROLLBACK_AND_REPAIR"
                patrol_status = "ROLLED_BACK"
            else:
                rollback_result = None
                summary["repairing"] += 1
                action = "ALERT_TO_REPAIR"
                patrol_status = "ALERTED"

            await IterationRepo(db).save_deployment_stage_record({
                "deployment_id": str(deployment.get("deployment_id")),
                "stage": stage,
                "decision": "LIFECYCLE_ALERT",
                "status": patrol_status,
                "health_json": {
                    "patrol": True,
                    "source": "lifecycle_alert",
                    "lifecycle_run_id": lifecycle_result.get("lifecycle_run_id"),
                    "current_phase": current_phase,
                    "has_alerts": has_alerts,
                    "trigger_diagnosis": trigger_diagnosis,
                    "alert_count": alert_count,
                    "primary_root_cause_code": state.get("primary_root_cause_code"),
                    "recommended_action": state.get("recommended_action"),
                },
                "result_json": {
                    "action": action,
                    "rollback_result": rollback_result,
                    "checked_by": body.updated_by,
                },
            })
            results.append({
                "deployment_id": str(deployment.get("deployment_id")),
                "model_id": model_id,
                "stage": stage,
                "patrol_status": patrol_status,
                "action": action,
                "lifecycle_run_id": lifecycle_result.get("lifecycle_run_id"),
                "current_phase": current_phase,
                "has_alerts": has_alerts,
                "trigger_diagnosis": trigger_diagnosis,
                "alert_count": alert_count,
                "primary_root_cause_code": state.get("primary_root_cause_code"),
                "recommended_action": state.get("recommended_action"),
                "rollback_result": rollback_result,
            })
            continue

        await IterationRepo(db).save_deployment_stage_record({
            "deployment_id": str(deployment.get("deployment_id")),
            "stage": stage,
            "decision": "HEALTH_CHECK",
            "status": "PASSED",
            "health_json": {
                "patrol": True,
                "source": "lifecycle_monitoring",
                "lifecycle_run_id": lifecycle_result.get("lifecycle_run_id"),
                "current_phase": current_phase,
                "has_alerts": has_alerts,
                "trigger_diagnosis": trigger_diagnosis,
                "alert_count": alert_count,
            },
            "result_json": {
                "patrol": True,
                "interval_seconds": body.interval_seconds,
                "checked_by": body.updated_by,
            },
        })
        summary["healthy"] += 1
        results.append({
            "deployment_id": str(deployment.get("deployment_id")),
            "model_id": model_id,
            "stage": stage,
            "patrol_status": "PASSED",
            "action": "OBSERVE",
            "lifecycle_run_id": lifecycle_result.get("lifecycle_run_id"),
            "current_phase": current_phase,
            "has_alerts": has_alerts,
            "trigger_diagnosis": trigger_diagnosis,
            "alert_count": alert_count,
        })

    if body.persist:
        await db.commit()
    else:
        await db.rollback()

    return _envelope(
        request,
        {
            "scheduler": {
                "mode": "LIFECYCLE_ALERT_PATROL",
                "interval_seconds": body.interval_seconds,
                "persisted": body.persist,
                "checked_at": datetime.now(UTC).isoformat(),
            },
            "summary": summary,
            "results": results,
        },
        "task4 patrol completed",
    )


@router.post("/deployments/proactive-release")
async def create_proactive_release(
    request: Request,
    body: ProactiveReleaseRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create a deployment for a healthy new-scenario model.

    NEW_SCENARIO releases do not require champion-vs-challenger comparison.
    They must pass the predeploy health gate and provide a rollback target, then
    enter the normal staged rollout and task4 parallel-control surface.
    """
    result = await _create_proactive_release(body, db)
    await db.commit()
    return _envelope(request, result, "proactive release created")


@router.post("/deployments/batch/proactive-release")
async def batch_create_proactive_release(
    request: Request,
    body: BatchProactiveReleaseRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create up to 50 healthy new-scenario deployments for task4 control."""
    results: list[dict] = []
    succeeded = 0
    failed = 0

    for item in body.items:
        try:
            created = await _create_proactive_release(item, db)
            results.append({
                "model_id": item.model_id,
                "status": "created",
                "deployment_id": created.get("deployment_id"),
                "initial_stage": created.get("initial_stage"),
                "challenger_traffic_ratio": created.get("challenger_traffic_ratio"),
            })
            await db.commit()
            succeeded += 1
        except Exception as exc:
            await db.rollback()
            results.append({
                "model_id": item.model_id,
                "status": "failed",
                "error": str(exc),
            })
            failed += 1

    return _envelope(
        request,
        {
            "total": len(body.items),
            "succeeded": succeeded,
            "failed": failed,
            "results": results,
        },
        f"batch proactive release: {succeeded}/{len(body.items)} created",
    )


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


class RollbackDrillRequest(BaseModel):
    stage: str = Field(default="CANARY_20", min_length=1)
    persist: bool = False
    updated_by: str = "rollback_drill"
    health_metrics: dict[str, object] = Field(default_factory=lambda: {
        "challenger_auc": 0.62,
        "challenger_ks": 0.12,
        "score_psi": 0.36,
        "bad_rate_drift": 0.25,
        "recovery_rate": 0.18,
        "discrimination_passed": False,
        "calibration_passed": False,
        "oot_passed": False,
    })


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


@router.post("/deployments/{deployment_id}/rollback-drill")
async def rollback_drill(
    deployment_id: str,
    request: Request,
    body: RollbackDrillRequest = RollbackDrillRequest(),
    db: AsyncSession = Depends(get_db),
):
    """Run a non-destructive automatic rollback drill.

    The drill uses the same health gate, gatekeeper, and rollback service as
    DeploymentGateNode. By default it rolls the DB transaction back before
    returning, so existing deployment and routing state are not changed.
    """
    from ..services.deployment.deployment_gatekeeper_service import DeploymentGatekeeperService
    from ..services.iteration.deployment_safety_service import (
        DeploymentSafetyService,
        STAGE_TRAFFIC_RATIO,
    )

    repo = IterationRepo(db)
    source = await repo.get_deployment(deployment_id)
    if source is None:
        raise NotFoundError("deployment record not found")

    stage = body.stage.upper()
    if stage not in STAGE_TRAFFIC_RATIO:
        raise ValidationAppError(
            "INVALID_DEPLOYMENT_STAGE",
            f"unsupported deployment stage: {body.stage}",
        )

    model_id = source.get("model_id")
    if not model_id:
        raise ValidationAppError("INVALID_DEPLOYMENT", "deployment has no model_id")

    before_routing = await repo.get_model_deployment_state(model_id)
    champion_version = (
        source.get("champion_version")
        or (before_routing or {}).get("stable_version_code")
        or (before_routing or {}).get("active_version_code")
        or source.get("current_champion")
        or "champion_v1"
    )
    candidate_version = (
        source.get("candidate_version")
        or (before_routing or {}).get("challenger_version_code")
        or f"{champion_version}_rollback_drill"
    )
    lifecycle_run_id = source.get("lifecycle_run_id")
    qualification_run_id = source.get("qualification_run_id")
    drill_deployment_id = str(uuid4())
    drill_record = {
        "deployment_id": drill_deployment_id,
        "source_deployment_id": str(deployment_id),
        "lifecycle_run_id": str(lifecycle_run_id) if lifecycle_run_id else None,
        "qualification_run_id": str(qualification_run_id) if qualification_run_id else None,
        "model_id": model_id,
        "champion_version": champion_version,
        "candidate_version": candidate_version,
        "deployment_stage": stage,
        "deployment_decision": "ADVANCE_STAGE",
        "status": "RUNNING",
        "dispatch_mode": "ROLLBACK_DRILL",
        "external_task_id": None,
        "health_json": {"drill_seed": True, "source_deployment_id": deployment_id},
        "result_json": {},
    }
    await repo.save_deployment_record(drill_record)

    safety = DeploymentSafetyService(db)
    canary_ratio = await safety.update_traffic_ratio(
        model_id=model_id,
        stage=stage,
        champion_version=champion_version,
        challenger_version=candidate_version,
        updated_by=body.updated_by,
    )
    health_result = DeploymentSafetyService.check_stage_health(stage, body.health_metrics)
    gatekeeper_decision = DeploymentGatekeeperService().decide(
        stage=stage,
        health_result=health_result,
        deployment_context=None,
        challenger_qualified=True,
        current_traffic_ratio=canary_ratio,
    )

    rollback_result = None
    if gatekeeper_decision.decision == "ROLLBACK":
        drill_deployment = await repo.get_deployment(drill_deployment_id)
        rollback_result = await safety.rollback(
            deployment=drill_deployment or drill_record,
            reason="ROLLBACK_DRILL:" + ",".join(gatekeeper_decision.decision_reasons),
            rollback_target=str(champion_version),
            updated_by=body.updated_by,
        )

    after_record = await repo.get_deployment(drill_deployment_id)
    rollback_events = await repo.get_rollback_events(drill_deployment_id)
    after_routing = await repo.get_model_deployment_state(model_id)
    result = {
        "source_deployment_id": deployment_id,
        "drill_deployment_id": drill_deployment_id,
        "persisted": body.persist,
        "transaction": "committed" if body.persist else "rolled_back",
        "stage": stage,
        "simulated_canary_traffic_ratio": canary_ratio,
        "health_result": health_result,
        "gatekeeper_decision": {
            "decision": gatekeeper_decision.decision,
            "decision_reasons": gatekeeper_decision.decision_reasons,
            "gatekeeper_rule_refs": gatekeeper_decision.gatekeeper_rule_refs,
            "rollback_target": gatekeeper_decision.rollback_target,
            "selected_strategy_code": gatekeeper_decision.selected_strategy_code,
        },
        "rollback_result": rollback_result,
        "post_rollback_record": after_record,
        "post_rollback_routing": after_routing,
        "rollback_events": rollback_events,
        "before_routing": before_routing,
    }

    if body.persist:
        await db.commit()
    else:
        await db.rollback()

    return _envelope(request, result, "automatic rollback drill completed")


# ═══════════════════════════════════════════════════════════════════════════════
# T4-GAP-06: 50 模型批量部署管控
# ═══════════════════════════════════════════════════════════════════════════════

class BatchDeploymentRequest(BaseModel):
    deployment_ids: list[str] = Field(min_length=1, max_length=50)
    updated_by: str = "admin"


class AdvanceDeploymentRequest(BaseModel):
    target_stage: str | None = None
    updated_by: str = "admin"
    health_metrics: dict[str, object] = Field(default_factory=dict)


class BatchAdvanceRequest(BatchDeploymentRequest):
    resume_lifecycle: bool = True


@router.post("/deployments/{deployment_id}/advance-stage")
async def advance_deployment_stage(
    deployment_id: str,
    request: Request,
    body: AdvanceDeploymentRequest = AdvanceDeploymentRequest(),
    db: AsyncSession = Depends(get_db),
):
    """Advance one deployment through staged rollout by deployment_id.

    This is the task4 direct rollout path for proactive NEW_SCENARIO releases
    that do not have a lifecycle_run_id. It still runs the deployment health
    gate and Gatekeeper before changing traffic or promoting the challenger.
    """
    from ..services.deployment.deployment_gatekeeper_service import DeploymentGatekeeperService
    from ..services.iteration.deployment_safety_service import (
        DeploymentSafetyService,
        STAGE_TRAFFIC_RATIO,
    )

    stages = [
        "OFFLINE_VALIDATION",
        "OOT_GATE",
        "SHADOW",
        "CANARY_5",
        "CANARY_20",
        "CANARY_50",
        "PRODUCTION",
    ]
    repo = IterationRepo(db)
    deployment = await repo.get_deployment(deployment_id)
    if deployment is None:
        raise NotFoundError("deployment not found")
    if deployment.get("status") in {"PROMOTED", "ROLLED_BACK", "FAILED", "ABORTED"}:
        raise ConflictError("deployment is already terminal")

    current_stage = str(deployment.get("current_stage") or "OFFLINE_VALIDATION").upper()
    if body.target_stage:
        target_stage = body.target_stage.upper()
        if target_stage not in stages:
            raise ValidationAppError("INVALID_TARGET_STAGE", f"unsupported stage: {body.target_stage}")
        if stages.index(target_stage) <= stages.index(current_stage):
            raise ValidationAppError("INVALID_TARGET_STAGE", "target_stage must be after current_stage")
    else:
        if current_stage not in stages or current_stage == "PRODUCTION":
            raise ValidationAppError("INVALID_CURRENT_STAGE", f"cannot advance from {current_stage}")
        target_stage = stages[stages.index(current_stage) + 1]

    safety = DeploymentSafetyService(db)
    health_result = DeploymentSafetyService.check_stage_health(target_stage, body.health_metrics)
    gatekeeper_decision = DeploymentGatekeeperService().decide(
        stage=target_stage,
        health_result=health_result,
        deployment_context=None,
        challenger_qualified=True,
        current_traffic_ratio=STAGE_TRAFFIC_RATIO.get(target_stage, 0.0),
    )
    decision = gatekeeper_decision.decision
    action_result: dict | None = None

    if decision == "ROLLBACK":
        action_result = await safety.rollback(
            deployment=deployment,
            reason="ADVANCE_STAGE_GATEKEEPER:" + ",".join(gatekeeper_decision.decision_reasons),
            rollback_target=deployment.get("champion_version"),
            updated_by=body.updated_by,
        )
        await db.commit()
        refreshed = await repo.get_deployment(deployment_id)
        return _envelope(request, {
            "deployment": refreshed,
            "health_result": health_result,
            "gatekeeper_decision": gatekeeper_decision.__dict__,
            "action_result": action_result,
        }, "deployment rolled back by gatekeeper")

    if decision == "HOLD":
        record = {
            "deployment_id": deployment_id,
            "lifecycle_run_id": str(deployment.get("lifecycle_run_id")) if deployment.get("lifecycle_run_id") else None,
            "qualification_run_id": str(deployment.get("qualification_run_id")) if deployment.get("qualification_run_id") else None,
            "model_id": deployment.get("model_id"),
            "champion_version": deployment.get("champion_version"),
            "candidate_version": deployment.get("candidate_version"),
            "deployment_stage": current_stage,
            "deployment_decision": "HOLD",
            "status": "HELD",
            "dispatch_mode": "TASK4_DIRECT_ADVANCE",
            "external_task_id": None,
            "health_json": health_result,
            "result_json": {"target_stage": target_stage, "decision_reasons": gatekeeper_decision.decision_reasons},
        }
        await repo.save_deployment_record(record)
        await db.commit()
        refreshed = await repo.get_deployment(deployment_id)
        return _envelope(request, {
            "deployment": refreshed,
            "health_result": health_result,
            "gatekeeper_decision": gatekeeper_decision.__dict__,
        }, "deployment held by gatekeeper")

    if target_stage == "PRODUCTION":
        record = {
            "deployment_id": deployment_id,
            "lifecycle_run_id": str(deployment.get("lifecycle_run_id")) if deployment.get("lifecycle_run_id") else None,
            "qualification_run_id": str(deployment.get("qualification_run_id")) if deployment.get("qualification_run_id") else None,
            "model_id": deployment.get("model_id"),
            "champion_version": deployment.get("champion_version"),
            "candidate_version": deployment.get("candidate_version"),
            "deployment_stage": "PRODUCTION",
            "deployment_decision": "PROMOTE",
            "status": "PROMOTED",
            "dispatch_mode": "TASK4_DIRECT_ADVANCE",
            "external_task_id": None,
            "health_json": health_result,
            "result_json": {"from_stage": current_stage, "target_stage": target_stage},
        }
        await repo.save_deployment_record(record)
        action_result = await safety.promote_to_champion(deployment=record, updated_by=body.updated_by)
    else:
        ratio = await safety.update_traffic_ratio(
            model_id=str(deployment.get("model_id") or ""),
            stage=target_stage,
            champion_version=deployment.get("champion_version"),
            challenger_version=deployment.get("candidate_version"),
            updated_by=body.updated_by,
        )
        record = {
            "deployment_id": deployment_id,
            "lifecycle_run_id": str(deployment.get("lifecycle_run_id")) if deployment.get("lifecycle_run_id") else None,
            "qualification_run_id": str(deployment.get("qualification_run_id")) if deployment.get("qualification_run_id") else None,
            "model_id": deployment.get("model_id"),
            "champion_version": deployment.get("champion_version"),
            "candidate_version": deployment.get("candidate_version"),
            "deployment_stage": target_stage,
            "deployment_decision": "ADVANCE_STAGE",
            "status": "RUNNING",
            "dispatch_mode": "TASK4_DIRECT_ADVANCE",
            "external_task_id": None,
            "health_json": health_result,
            "result_json": {
                "from_stage": current_stage,
                "target_stage": target_stage,
                "challenger_traffic_ratio": ratio,
            },
        }
        await repo.save_deployment_record(record)
        action_result = {"target_stage": target_stage, "challenger_traffic_ratio": ratio}

    await db.commit()
    refreshed = await repo.get_deployment(deployment_id)
    routing = await repo.get_model_deployment_state(str(deployment.get("model_id") or ""))
    return _envelope(request, {
        "deployment": refreshed,
        "routing": routing,
        "health_result": health_result,
        "gatekeeper_decision": gatekeeper_decision.__dict__,
        "action_result": action_result,
    }, "deployment stage advanced")


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
