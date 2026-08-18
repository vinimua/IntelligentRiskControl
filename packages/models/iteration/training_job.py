"""Training Worker 输入输出合同。"""

from datetime import datetime

from pydantic import Field, model_validator

from ..common.base import ContractModel
from ..common.enums import TrainingMode, WorkerStatus


class TimeRange(ContractModel):
    window_id: str = ""
    start_at: datetime
    end_at: datetime

    @model_validator(mode="after")
    def end_must_follow_start(self) -> "TimeRange":
        if self.end_at <= self.start_at:
            raise ValueError("end_at must be later than start_at")
        return self


class TrainingJobInput(ContractModel):
    """训练参数；技术重试必须复用 training_job_id 和 idempotency_key。"""

    training_job_id: str
    idempotency_key: str
    model_id: str = ""
    lifecycle_run_id: str = ""
    iteration_run_id: str
    training_plan_id: str
    experiment_id: str
    # 最大业务轮次统一为 2（A7 定稿）
    business_round: int = Field(ge=1, le=2)
    strategy_code: str
    execution_mode: str = ""
    training_data_mode: str = ""
    training_window_ids: list[str] = Field(min_length=1)
    validation_window_ids: list[str] = Field(min_length=1)
    train_time_ranges: list[TimeRange] = Field(default_factory=list)
    validation_time_ranges: list[TimeRange] = Field(default_factory=list)
    oot_window_id: str = "W4"
    data_snapshot_ids: list[str] = Field(min_length=1)
    data_snapshot_checksums: dict[str, str] = Field(default_factory=dict)
    data_snapshot_uris: dict[str, str] = Field(default_factory=dict)
    label_versions: list[str] = Field(min_length=1)
    sample_weight_policy: dict = Field(default_factory=dict)
    sample_weight_required: bool = False
    affected_segments: list[dict] = Field(default_factory=list)
    change_point: datetime | None = None
    feature_schema_version: str
    ordered_features: list[str] = Field(default_factory=list)
    ordered_features_hash: str = ""
    preprocessing_version: str
    preprocessing_hash: str = ""
    algorithm: str
    algorithm_family: str = ""
    champion_artifact_checksum: str = ""
    hyperparameters: dict = Field(default_factory=dict)
    target_metrics: list[str] = Field(default_factory=list)
    qualification_rule_version: str
    base_model_version: str
    seed: int
    artifact_output_uri: str
    training_mode: TrainingMode = TrainingMode.FULL_RETRAIN
    # A7 阶段四：特征筛选合同（FEATURE_SELECTION 模式的真实执行证据）
    unstable_feature_codes: list[str] = Field(default_factory=list)
    selected_feature_codes: list[str] = Field(default_factory=list)
    feature_selection_artifact_uri: str | None = None

    @model_validator(mode="after")
    def validate_windows(self) -> "TrainingJobInput":
        selected = self.training_window_ids + self.validation_window_ids
        if self.oot_window_id in selected:
            raise ValueError("OOT window must never be used for training or tuning")
        if set(self.training_window_ids) & set(self.validation_window_ids):
            raise ValueError("training and validation window roles must not overlap")
        # 严格 A7 链路填充时间范围/校验和时才做一致性校验
        if self.train_time_ranges and {
            item.window_id for item in self.train_time_ranges
        } != set(self.training_window_ids):
            raise ValueError("train_time_ranges must match training_window_ids")
        if self.validation_time_ranges and {
            item.window_id for item in self.validation_time_ranges
        } != set(self.validation_window_ids):
            raise ValueError("validation_time_ranges must match validation_window_ids")
        if self.sample_weight_required and not self.sample_weight_policy:
            raise ValueError("weighted job requires sample_weight_policy")
        if self.data_snapshot_checksums and set(
            self.data_snapshot_checksums
        ) != set(self.data_snapshot_ids):
            raise ValueError("data snapshot ids and checksums must match")
        if self.data_snapshot_uris and set(self.data_snapshot_uris) != set(
            self.data_snapshot_ids
        ):
            raise ValueError("data snapshot ids and uris must match")
        canonical = {
            "logistic_regression": "LogisticRegression",
            "random_forest": "RandomForest",
            "xgboost": "XGBoost",
            "lightgbm": "LightGBM",
            "catboost": "CatBoost",
            "ebm": "EBM",
        }
        if self.algorithm_family and canonical.get(self.algorithm.lower()) != self.algorithm_family:
            raise ValueError("algorithm does not match algorithm_family")
        return self


class TrainingConsumptionReceipt(ContractModel):
    """训练消费回执：Worker 真实消费的证据（防"计划写了但没吃"）。"""

    consumed_training_snapshot_ids: list[str]
    consumed_validation_snapshot_ids: list[str]
    observed_train_sample_count: int = Field(ge=0)
    observed_validation_sample_count: int = Field(ge=0)
    observed_train_bad_count: int = Field(ge=0)
    observed_validation_bad_count: int = Field(ge=0)
    sample_overlap_count: int = Field(ge=0)
    actual_algorithm_family: str
    actual_execution_mode: str
    actual_base_model_checksum: str | None = None
    sample_weight_consumed: bool
    sample_weight_min: float | None = None
    sample_weight_max: float | None = None
    sample_weight_mean: float | None = None
    non_unit_weight_sample_count: int = Field(ge=0)
    affected_segment_ids_consumed: list[str] = Field(default_factory=list)
    actual_ordered_features_hash: str
    actual_preprocessing_hash: str
    actual_hyperparameters_hash: str
    w4_read_count: int = Field(ge=0, le=0)


class TrainingJobOutput(ContractModel):
    """Worker 只报告技术结果，不得宣告 Challenger 合格。"""

    training_job_id: str
    experiment_id: str
    status: WorkerStatus
    candidate_version: str | None = None
    model_artifact_uri: str | None = None
    calibrator_artifact_uri: str | None = None
    threshold_artifact_uri: str | None = None
    training_metrics: dict = Field(default_factory=dict)
    validation_metrics: dict = Field(default_factory=dict)
    segment_metrics: dict = Field(default_factory=dict)
    artifact_checksums: dict[str, str] = Field(default_factory=dict)
    environment_manifest: dict = Field(default_factory=dict)
    # 训练消费回执（Worker 产出；严格 A7 链路强制，自然链路过渡期可选）
    consumption_receipt: TrainingConsumptionReceipt | None = None
    technical_retry_count: int = 0
    error_code: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def succeeded_job_requires_artifact(self) -> "TrainingJobOutput":
        if self.status == WorkerStatus.SUCCEEDED:
            if not self.candidate_version or not self.model_artifact_uri:
                raise ValueError(
                    "successful job requires candidate_version and model_artifact_uri"
                )
        return self
