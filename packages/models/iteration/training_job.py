"""Training Worker 输入输出合同。"""

from datetime import datetime

from pydantic import Field, model_validator

from ..common.base import ContractModel
from ..common.enums import WorkerStatus


class TimeRange(ContractModel):
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
    iteration_run_id: str
    training_plan_id: str
    experiment_id: str
    business_round: int = Field(ge=1, le=3)
    strategy_code: str
    training_window_ids: list[str] = Field(min_length=1)
    validation_window_ids: list[str] = Field(min_length=1)
    train_time_ranges: list[TimeRange] = Field(default_factory=list)
    validation_time_ranges: list[TimeRange] = Field(default_factory=list)
    oot_window_id: str = "W4"
    data_snapshot_ids: list[str] = Field(min_length=1)
    label_versions: list[str] = Field(min_length=1)
    sample_weight_policy: dict = Field(default_factory=dict)
    feature_schema_version: str
    preprocessing_version: str
    algorithm: str
    hyperparameters: dict = Field(default_factory=dict)
    target_metrics: list[str] = Field(default_factory=list)
    qualification_rule_version: str
    base_model_version: str
    seed: int
    artifact_output_uri: str
    training_mode: str = "full"  # "full" (全量重训) / "incremental" (在 Champion 基础上续训)

    @model_validator(mode="after")
    def validate_windows(self) -> "TrainingJobInput":
        selected = self.training_window_ids + self.validation_window_ids
        if self.oot_window_id in selected:
            raise ValueError("OOT window must never be used for training or tuning")
        if set(self.training_window_ids) & set(self.validation_window_ids):
            raise ValueError("training and validation window roles must not overlap")
        return self


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
