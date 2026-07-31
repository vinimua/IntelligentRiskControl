"""Challenger 七道资格门合同。"""

from pydantic import Field

from ..common.base import ContractModel
from ..common.enums import QualificationGateCode, QualificationStatus


class MetricComparison(ContractModel):
    metric_code: str
    direction: str
    original_drop: float | None = None
    recovered_amount: float | None = None
    recovery_rate: float | None = None
    champion_value: float | None = None
    challenger_value: float | None = None
    healthy_lower_bound: float | None = None
    healthy_upper_bound: float | None = None
    bootstrap_ci_lower: float | None = None
    bootstrap_ci_upper: float | None = None


class QualificationGateResult(ContractModel):
    gate_code: QualificationGateCode
    gate_order: int = Field(ge=0, le=6)
    status: QualificationStatus
    required: bool = True
    metric_code: str | None = None
    expected: dict = Field(default_factory=dict)
    actual: dict = Field(default_factory=dict)
    bootstrap_interval: tuple[float, float] | None = None
    affected_segments: list[str] = Field(default_factory=list)
    failure_code: str | None = None
    reasons: list[str] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)


class QualificationInput(ContractModel):
    qualification_run_id: str
    iteration_run_id: str
    experiment_id: str
    candidate_version: str
    target_metrics: list[MetricComparison] = Field(default_factory=list)
    data_reproducible: bool
    discrimination_passed: bool
    calibration_passed: bool
    score_psi: float
    train_valid_gap: float
    segment_governance_passed: bool
    oot_window_id: str
    candidate_frozen_before_oot: bool
    oot_usage: str = "FINAL_QUALIFICATION"
    oot_passed: bool


class QualificationReport(ContractModel):
    qualification_run_id: str
    iteration_run_id: str
    experiment_id: str
    candidate_version: str
    status: QualificationStatus
    qualified: bool
    gate_results: list[QualificationGateResult]
    failed_gate_codes: list[QualificationGateCode] = Field(default_factory=list)
    rule_version: str
