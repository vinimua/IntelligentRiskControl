"""Model task interface contracts for adaptive iteration."""

from pydantic import Field

from ..common.base import ContractModel
from ..common.enums import (
    GuardrailCoverageStatus,
    MetricDirection,
    ModelTaskType,
    RiskGuardrailCode,
    TrainingMode,
)


class MetricSpec(ContractModel):
    metric_code: str
    direction: MetricDirection
    label_required: bool = True
    baseline_required: bool = False
    description: str = ""


class ModelTaskProfile(ContractModel):
    model_id: str
    champion_version: str
    model_type: str | None = None
    algorithm_family: str | None = None
    task_type: ModelTaskType
    task_type_source: str
    target_column: str | None = None
    prediction_column: str | None = None
    residual_column: str | None = None
    cluster_label_column: str | None = None
    required_metrics: list[MetricSpec] = Field(default_factory=list)
    optional_metrics: list[MetricSpec] = Field(default_factory=list)
    supported_training_modes: list[TrainingMode] = Field(default_factory=list)
    unsupported_training_modes: list[TrainingMode] = Field(default_factory=list)
    adapter_ready: bool = True
    limitations: list[str] = Field(default_factory=list)


class RiskGuardrailResult(ContractModel):
    risk_code: RiskGuardrailCode
    status: GuardrailCoverageStatus
    implemented: bool = False
    blocking: bool = False
    covered_by: list[str] = Field(default_factory=list)
    missing_controls: list[str] = Field(default_factory=list)
    recommendation: str = ""


class ModelTaskInterfaceSummary(ContractModel):
    profile: ModelTaskProfile
    risk_guardrails: list[RiskGuardrailResult] = Field(default_factory=list)
