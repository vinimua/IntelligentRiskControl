"""根因驱动修复决策合同。"""

from pydantic import Field, model_validator

from ..common.base import ContractModel
from ..common.enums import (
    ConfidenceLevel,
    DimensionCode,
    ProposalStatus,
    RecommendedAction,
)


class RootCauseCandidate(ContractModel):
    root_cause_code: str
    dimension: DimensionCode
    score: float = Field(ge=0.0, le=1.0)
    evidence_coverage: float = Field(ge=0.0, le=1.0)
    evidence_types: list[str] = Field(default_factory=list)


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
    root_causes: list[RootCauseCandidate] = Field(min_length=1)
    degraded_metrics: list[MetricDegradation] = Field(default_factory=list)
    business_objective_changed: bool = False
    data_repair_completed: bool = False
    pipeline_repair_completed: bool = False
    rule_version: str = "iteration-rules-v1"


class StrategySelection(ContractModel):
    strategy_code: str
    parameters: dict = Field(default_factory=dict)
    rationale: str


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
