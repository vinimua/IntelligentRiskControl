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
    trigger_diagnosis: bool | None = None
    decay_degree: str | None = None
    status_7d: str | None = None
    status_30d: str | None = None
    diagnosis_status: str | None = None
    persistence_judgment: dict | None = None
    # A7 §8: 触发原因记录（THRESHOLD_BREACH / SENTINEL_ANOMALY / SEVERE_PERSISTENCE）
    trigger_cause: str | None = None

    # ── A3-A6 动作执行链（严格 A7 垂直链路字段）──
    # 业务目标变更 + 授权（A6 阈值调整的前置条件）
    business_objective_changed: bool = False
    authorization_id: str | None = None
    # 动作执行上下文（executors 三计划携带的执行上下文合并源）
    action_execution_context: dict | None = None
    # A2 观察期语义（观察 7 天后重评估）
    observation_status: str | None = None
    observation_started_at: str | None = None
    observation_until: str | None = None
    # A3/A4 修复执行产物与回放资格
    repair_artifact_uri: str | None = None
    repair_artifact_checksum: str | None = None
    repair_execution_result: dict | None = None
    repair_qualified: bool | None = None
    # A3/A4 修复对象：受影响的特征清单（诊断证据驱动的修复目标）
    affected_features: list[str] | None = None
    # A5 校准 / A6 阈值执行产物
    calibration_artifact_uri: str | None = None
    threshold_artifact_uri: str | None = None
    adjustment_artifact_checksum: str | None = None
    adjustment_execution_result: dict | None = None
    # L1 策略选择结果
    selection_status: str | None = None
    primary_strategy: str | None = None
    execution_mode: str | None = None
    strategy_source: str | None = None
    # OOT 晋升治理（W4 授权读取 + 冻结身份 + 最终资格）
    allow_w4: bool | None = None
    pre_oot_status: str | None = None
    frozen_identity_checksum: str | None = None
    final_qualification_run_id: str | None = None
    final_qualification_status: str | None = None

    # 任务二摘要
    diagnosis_run_id: str | None = None
    primary_root_cause_code: str | None = None
    primary_root_cause_dimension: str | None = None
    primary_root_cause_score: float | None = None
    recommended_action: str | None = None
    need_iteration: bool | None = None
    # A7 §4/§5: L1 结构化上下文（由诊断输出持久化写入）
    impact_scope: str | None = None
    change_pattern: str | None = None
    # A7 §4: 冻结合格客群定义（segment_weighted_retrain 的权重来源）
    segment_evidence: dict | None = None
    # 人工批准：由真实 Review 记录推导（APPROVED），不能是无人写入的布尔
    manual_approval: bool = False

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
    # A7 阶段四：特征筛选（FEATURE_SELECTION 模式）
    selected_feature_codes: str | None = None
    feature_selection_report: str | None = None
    feature_selection_artifact_uri: str | None = None
    training_plan_id: str | None = None
    hyperparameter_tuning_plan_id: str | None = None
    tuning_dispatched: bool | None = None
    tuning_completed: bool | None = None
    best_hyperparameters: dict | None = None
    best_tuning_metric: float | None = None
    challenger_version: str | None = None
    challenger_qualified: bool | None = None
    qualification_run_id: str | None = None
    # A7 资格时序：W3 预资格（Gate 0-5）→ OOT → 最终资格（Gate 6 + 汇总）
    final_qualification_completed: bool = False
    failure_report_id: str | None = None
    # W3 失败归因结果（第二轮特征筛选证据的真实来源）；
    # State 只保存控制字段，失败门/特征码以逗号分隔字符串保存
    failed_gate_codes: str | None = None
    unstable_feature_codes: str | None = None
    feature_evidence_source: str | None = None
    unstable_feature_subset_confirmed: bool = False
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
    # A7 §10: W4 FINAL-OOT 完成证据（NATURAL 校准门槛）
    oot_validation_completed: bool = False
    oot_validation_run_id: str | None = None
    w4_available: bool = False
    candidate_frozen_before_oot: bool = False
    oot_passed: bool | None = None
    lifecycle_terminal: bool = False

    # 异常
    last_error: dict | None = None
