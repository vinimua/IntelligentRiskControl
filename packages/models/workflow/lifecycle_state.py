"""
LangGraph 主图 State
"""

from ..common.base import ContractModel

from ..common.enums import LifecyclePhase, TriggerType

class ModelLifecycleState(ContractModel):
    """
    主图 State — 只保存流程控制字段、结果摘要和 runId
    完整业务数据进入 PostgreSQL、MLflow 或 MinIO
    """

    schema_version: int = 2
    lifecycle_run_id: str
    model_id: str
    champion_version: str
    trigger_type: TriggerType = TriggerType.SCHEDULED_TRIGGER
    current_phase: LifecyclePhase = LifecyclePhase.CREATED
    requires_manual_review: bool = False

    # 任务一窗口配置（可选，不传则自动选择）
    baseline_window_id: str | None = None
    current_window_id: str | None = None

    # 任务一摘要
    monitoring_run_id: str | None = None
    has_alerts: bool | None = None
    alert_count: int | None = None
    max_alert_severity: str | None = None

    # 任务二摘要
    diagnosis_run_id: str | None = None
    primary_root_cause_code: str | None = None
    primary_root_cause_dimension: str | None = None
    primary_root_cause_score: float | None = None
    recommended_action: str | None = None
    need_iteration: bool | None = None

    # 诊断事件（LangGraph 开发路线 V1.0 新增）
    event_id: str | None = None

    # Agent 决策（LangGraph 开发路线 V1.0 新增）
    agent_decision_id: str | None = None
    agent_handoff_status: str | None = None
    agent_confidence: float | None = None

    # 任务三摘要
    iteration_run_id: str | None = None
    decision_proposal_id: str | None = None
    risk_assessment_id: str | None = None
    data_eligibility_assessment_id: str | None = None
    repair_plan_id: str | None = None
    calibration_plan_id: str | None = None
    threshold_plan_id: str | None = None
    manual_review_id: str | None = None
    feature_reconstruction_plan_id: str | None = None
    feature_reconstruction_status: str | None = None
    feature_reconstruction_dispatched: bool | None = None
    feature_schema_version: str | None = None
    feature_snapshot_id: str | None = None
    feature_transform_count: int | None = None
    transform_artifact_uri: str | None = None
    training_plan_id: str | None = None
    challenger_version: str | None = None
    challenger_qualified: bool | None = None
    qualification_run_id: str | None = None
    failure_report_id: str | None = None
    iteration_exit_reason: str | None = None

    # 异步训练（LangGraph 开发路线 V1.0 新增）
    training_job_id: str | None = None
    experiment_id: str | None = None
    training_callback_status: str | None = None
    training_dispatched: bool | None = None
    training_dispatch_mode: str | None = None
    training_metrics: dict | None = None
    validation_metrics: dict | None = None
    segment_metrics: dict | None = None
    business_round: int | None = None

    # 任务四摘要
    deployment_id: str | None = None
    deployment_stage: str | None = None
    deployment_decision: str | None = None

    # 异常
    last_error: dict | None = None
