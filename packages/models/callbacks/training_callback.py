"""Training Worker Callback。"""

from pydantic import Field, model_validator

from ..common.base import ContractModel
from ..common.enums import WorkerStatus


class TrainingCallback(ContractModel):
    """幂等键是 training_job_id；仅表达 Worker 技术结果。"""

    training_job_id: str
    lifecycle_run_id: str | None = None  # P1: Worker 回调时自动 resume
    idempotency_key: str
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
    # 任务三预标记字段
    data_reproducible: bool = False
    candidate_frozen_before_oot: bool = False
    # 冻结身份校验和（Worker 冻结候选时写回；晋升防换包）
    frozen_identity_checksum: str | None = None
    # 训练消费回执（Worker 真实消费证据；严格 A7 链路要求 SUCCEEDED 非空）
    consumption_receipt: dict | None = None

    @model_validator(mode="after")
    def validate_technical_result(self) -> "TrainingCallback":
        if self.status == WorkerStatus.SUCCEEDED:
            if not self.candidate_version or not self.model_artifact_uri:
                raise ValueError(
                    "successful callback requires candidate and model artifact"
                )
        return self
