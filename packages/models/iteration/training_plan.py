"""训练计划合同。"""

from datetime import datetime

from pydantic import Field, model_validator

from ..common.base import ContractModel
from ..common.enums import TrainingMode, TrainingPlanStatus


class WindowTimeRange(ContractModel):
    """窗口确定性时间范围（严格 A7 切分合同）。"""

    window_id: str
    start_at: datetime
    end_at: datetime

    @model_validator(mode="after")
    def validate_order(self) -> "WindowTimeRange":
        if self.end_at <= self.start_at:
            raise ValueError("window range end_at must be later than start_at")
        return self


class TrainingWindowSpec(ContractModel):
    baseline_window_id: str = "W1"
    training_window_ids: list[str] = Field(default_factory=lambda: ["W2"])
    validation_window_ids: list[str] = Field(default_factory=lambda: ["W3"])
    oot_window_id: str = "W4"
    oot_locked: bool = True
    # ── W3 确定性切分合同（严格 A7 plan_builder 填充；自然链路可缺省）──
    training_time_ranges: list[WindowTimeRange] = Field(default_factory=list)
    validation_time_ranges: list[WindowTimeRange] = Field(default_factory=list)
    w3_split_method: str | None = None
    w3_split_boundary: datetime | None = None
    w3_train_snapshot_id: str | None = None
    w3_validation_snapshot_id: str | None = None
    w3_train_checksum: str | None = None
    w3_validation_checksum: str | None = None

    @model_validator(mode="after")
    def forbid_oot_leakage(self) -> "TrainingWindowSpec":
        if self.oot_window_id in self.training_window_ids:
            raise ValueError("W4/OOT window must never be used for training")
        if self.oot_window_id in self.validation_window_ids:
            raise ValueError("W4/OOT window must never be used for tuning")
        if self.oot_window_id == self.baseline_window_id:
            raise ValueError("W4/OOT window must not be the baseline window")
        if set(self.training_window_ids) & set(self.validation_window_ids):
            raise ValueError("training and validation window roles must not overlap")
        # ── W3 切分一致性：仅在严格 A7 链路填充切分字段时校验 ──
        if self.training_time_ranges and self.validation_time_ranges:
            train_ranges = {item.window_id: item for item in self.training_time_ranges}
            valid_ranges = {item.window_id: item for item in self.validation_time_ranges}
            if set(train_ranges) != set(self.training_window_ids):
                raise ValueError("training time ranges must match training window ids")
            if set(valid_ranges) != set(self.validation_window_ids):
                raise ValueError("validation time ranges must match validation window ids")
            if self.w3_split_boundary is not None:
                w3_train = train_ranges.get("W3_TRAIN_SPLIT")
                w3_valid = valid_ranges.get("W3_VALIDATION_SPLIT")
                if w3_train is None or w3_valid is None:
                    raise ValueError("A7 requires deterministic W3 train and validation splits")
                if w3_train.end_at != self.w3_split_boundary:
                    raise ValueError("W3 training range must end at the frozen split boundary")
                if w3_valid.start_at != self.w3_split_boundary:
                    raise ValueError("W3 validation range must start at the frozen split boundary")
        return self


class TrainingPlan(ContractModel):
    training_plan_id: str
    proposal_id: str
    approval_id: str
    iteration_run_id: str
    experiment_id: str
    # 最大业务轮次统一为 2（A7 定稿）
    business_round: int = Field(ge=1, le=2)
    diagnosis_run_id: str
    model_id: str
    frozen_champion_version: str
    root_cause_code: str
    strategy_code: str
    strategy_parameters: dict = Field(default_factory=dict)
    # 主训练模式：full / incremental / none —— 从 StrategySelection 正式传递，
    # 不从 strategy_tier 猜测
    training_mode: TrainingMode = TrainingMode.FULL_RETRAIN
    # A7 阶段四：特征筛选合同（FEATURE_SELECTION 模式的真实执行证据）
    unstable_feature_codes: list[str] = Field(default_factory=list)
    selected_feature_codes: list[str] = Field(default_factory=list)
    feature_selection_artifact_uri: str | None = None
    target_metric_codes: list[str] = Field(default_factory=list)
    windows: TrainingWindowSpec = Field(default_factory=TrainingWindowSpec)
    data_eligibility_assessment_ids: list[str] = Field(default_factory=list)
    data_snapshot_ids: list[str] = Field(min_length=1)
    data_snapshot_checksums: dict[str, str] = Field(default_factory=dict)
    label_versions: list[str] = Field(min_length=1)
    sample_weight_policy: dict = Field(default_factory=dict)
    sample_weight_required: bool = False
    sample_weight_summary: dict = Field(default_factory=dict)
    feature_schema_version: str
    ordered_features: list[str] = Field(default_factory=list)
    ordered_features_hash: str | None = None
    preprocessing_version: str
    preprocessing_hash: str | None = None
    algorithm: str
    algorithm_family: str | None = None
    same_algorithm_family: bool = False
    champion_artifact_checksum: str | None = None
    hyperparameter_space: dict = Field(default_factory=dict)
    random_seed: int = 2026
    qualification_rule_version: str = "qualification-rules-v1"
    risk_level: str
    max_business_rounds: int = 2
    rollback_target: str
    status: TrainingPlanStatus = TrainingPlanStatus.DRAFT
    blocking_reasons: list[str] = Field(default_factory=list)
    rule_version: str
    # ── 严格 A7 链路的传递/审计字段（自然链路可缺省）──
    lifecycle_run_id: str | None = None
    event_id: str | None = None
    monitoring_run_id: str | None = None
    agent_decision_id: str | None = None
    decision_source: str | None = None
    root_cause_status: str | None = None
    decay_degree: str | None = None
    impact_scope: str | None = None
    change_pattern: str | None = None
    change_point: str | None = None
    affected_segments: list[dict] = Field(default_factory=list)
    primary_strategy: str | None = None
    execution_mode: str | None = None
    training_data_mode: str | None = None
    strategy_source: str | None = None
    kg_consistency_status: str | None = None
    kg_repair_required: bool = False
    selection_reason_codes: list[str] = Field(default_factory=list)
    authorization_type: str | None = None
    authorization_id: str | None = None
    l1_matrix_version: str | None = None
    window_rule_version: str | None = None
    threshold_status: str = "PENDING_EMPIRICAL_CALIBRATION"
