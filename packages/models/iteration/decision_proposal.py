"""根因驱动修复决策合同。"""

from pydantic import Field, model_validator

from ..common.base import ContractModel
from ..common.enums import (
    ConfidenceLevel,
    DimensionCode,
    ProposalStatus,
    RecommendedAction,
    TrainingMode,
)


class RootCauseCandidate(ContractModel):
    root_cause_code: str
    dimension: DimensionCode
    score: float = Field(ge=0.0, le=1.0)
    evidence_coverage: float = Field(ge=0.0, le=1.0)
    evidence_types: list[str] = Field(default_factory=list)
    # 严格 A7 入口（A7DecisionEnvelope）传递：候选状态 + 证据引用
    candidate_status: str = "SUSPECTED"
    evidence_refs: list[str] = Field(default_factory=list)


class MetricDegradation(ContractModel):
    metric_code: str
    baseline_value: float | None = None
    current_value: float | None = None
    healthy_lower_bound: float | None = None
    healthy_upper_bound: float | None = None
    degraded: bool = True


class DecisionInput(ContractModel):
    diagnosis_run_id: str
    lifecycle_run_id: str
    model_id: str
    champion_version: str
    # 监测运行引用：资格评估端点据此加载特征级 PSI（服务端信任源）
    monitoring_run_id: str | None = None
    root_causes: list[RootCauseCandidate] = Field(min_length=1)
    degraded_metrics: list[MetricDegradation] = Field(default_factory=list)
    business_objective_changed: bool = False
    data_repair_completed: bool = False
    pipeline_repair_completed: bool = False
    # A7 定稿 §4/§5: L1 读取结构化上下文确定策略（KG 不能替 L1 完成选择）
    # 持续性等级：NONE / SHORT_TERM_7D / SUSTAINED_30D / SEVERE
    decay_degree: str | None = None
    # 漂移影响范围：LOCAL / GLOBAL
    impact_scope: str | None = None
    # 变化模式：GRADUAL / SUDDEN
    change_pattern: str | None = None
    # 业务轮次：1 或 2（第二轮策略需 round>=2 + 证据 + 人工批准）
    business_round: int = 1
    manual_approval: bool = False
    # A7 §5: W3 失败归因证据 —— feature_selection_retrain 的 L1 选择
    # 必须由真实归因报告约束（三项缺一不可）
    failure_report_id: str | None = None
    unstable_feature_codes: list[str] = Field(default_factory=list)
    feature_evidence_source: str | None = None
    # A7 §4: 冻结合格客群定义（segment_weighted_retrain 的权重来源）
    segment_evidence: dict | None = None
    # 严格 A7 入口（A7DecisionEnvelope）上下文字段
    event_id: str | None = None
    agent_decision_id: str | None = None
    decision_source: str | None = None
    algorithm_family: str | None = None
    model_task_type: str = "BINARY_CLASSIFICATION"
    change_point: str | None = None
    ordered_evidence_window_ids: list[str] = Field(default_factory=list)
    affected_segments: list[dict] = Field(default_factory=list)
    authorization: dict | None = None
    rule_version: str = "iteration-rules-v1"


class StrategySelection(ContractModel):
    strategy_code: str
    parameters: dict = Field(default_factory=dict)
    rationale: str
    # 主训练模式：full / incremental / none —— 来自 StrategyDefinition，
    # 沿 Strategy → Candidate → Proposal → TrainingPlan 正式传递
    primary_training_mode: TrainingMode = TrainingMode.FULL_RETRAIN


class DecisionProposal(ContractModel):
    proposal_id: str
    proposal_version: int = Field(default=1, ge=1)
    parent_proposal_id: str | None = None
    diagnosis_run_id: str
    monitoring_run_id: str | None = None
    lifecycle_run_id: str
    model_id: str
    champion_version: str
    primary_root_cause_code: str
    primary_root_cause_score: float = Field(ge=0.0, le=1.0)
    top1_top2_gap: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_coverage: float = Field(ge=0.0, le=1.0)
    contributing_root_cause_codes: list[str] = Field(default_factory=list)
    action: RecommendedAction
    need_iteration: bool
    strategies: list[StrategySelection] = Field(default_factory=list)
    selected_strategy_code: str | None = None
    target_metric_codes: list[str] = Field(default_factory=list)
    proposed_window_policy: str | None = None
    expected_recovery: dict = Field(default_factory=dict)
    risk_factors: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel
    decision_reasons: list[str] = Field(default_factory=list)
    status: ProposalStatus = ProposalStatus.DRAFT
    executable: bool = False
    requires_manual_review: bool = False
    # A7 §4.2: KG 一致性状态 —— MITIGATES 缺边等不一致不单独阻断，
    # L1 仍是最终策略权威，仅标记并要求图谱修复
    kg_consistency_status: str | None = None  # 如 KG_MITIGATES_MISSING
    kg_repair_required: bool = False
    # A7 §6.3: 任务三决策输出 —— KG 咨询候选码列表 + L1 最终策略码
    kg_candidate_codes: list[str] = Field(default_factory=list)
    final_strategy_code: str | None = None
    # ── 严格 A7 入口（select_a7_strategy/propose_a7）传递的 L1 选择字段 ──
    # 自然链路（decide_with_kg）不设这些字段，plan_builder 对自然链路
    # 走既有校验；严格 A7 链路按合同赋值。
    selection_status: str | None = None
    primary_strategy: str | None = None
    execution_mode: str | None = None
    strategy_source: str | None = None
    selection_reason_codes: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    event_id: str | None = None
    agent_decision_id: str | None = None
    decision_source: str | None = None
    root_cause_status: str | None = None
    decay_degree: str | None = None
    model_task_type: str | None = None
    algorithm_family: str | None = None
    champion_artifact_checksum: str | None = None
    impact_scope: str | None = None
    change_pattern: str | None = None
    change_point: str | None = None
    ordered_evidence_window_ids: list[str] = Field(default_factory=list)
    affected_segments: list[dict] = Field(default_factory=list)
    sample_weight_required: bool = False
    sample_weight_policy: dict = Field(default_factory=dict)
    training_data_mode: str | None = None
    training_window_ids: list[str] = Field(default_factory=list)
    validation_window_ids: list[str] = Field(default_factory=list)
    oot_window_ids: list[str] = Field(default_factory=list)
    authorization_type: str | None = None
    authorization_id: str | None = None
    rule_version: str
    rule_versions: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def keep_iteration_flag_consistent(self) -> "DecisionProposal":
        expected = self.action == RecommendedAction.MODEL_ITERATION
        if self.need_iteration != expected:
            raise ValueError(
                "need_iteration must be true exactly when action is MODEL_ITERATION"
            )
        if self.executable:
            raise ValueError("DecisionProposal is advisory and must never be executable")
        return self
