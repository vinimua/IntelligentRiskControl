"""Gatekeeper、修复决策与迭代配置合同。"""

from pydantic import Field

from ..common.base import ContractModel


class GatekeeperRule(ContractModel):
    rule_code: str
    rule_version: str
    metric_code: str
    min_improvement: float | None = None
    min_recovery_rate: float | None = None
    max_score_psi: float | None = None
    max_train_valid_gap: float | None = None
    data_leakage_check: bool = True
    interpretability_required: bool = False


class RootCauseGateConfig(ContractModel):
    min_primary_score: float = Field(default=0.75, ge=0.0, le=1.0)
    min_top1_top2_gap: float = Field(default=0.15, ge=0.0, le=1.0)
    min_evidence_coverage: float = Field(default=0.70, ge=0.0, le=1.0)


class MissingRateConfig(ContractModel):
    watch: float = 0.05
    warning: float = 0.10
    critical: float = 0.20
    unavailable: float = 0.40
    label_training_block: float = 0.20


class IterationRuleConfig(ContractModel):
    max_iteration_rounds: int = Field(default=3, ge=1)
    max_technical_retries: int = Field(default=3, ge=0)
    allowed_strategy_codes: list[str] = Field(default_factory=list)
    training_window_policy: str = "W1_W3_ONLY"
    baseline_window_id: str = "W1"
    default_training_window_ids: list[str] = Field(
        default_factory=lambda: ["W2"]
    )
    default_validation_window_ids: list[str] = Field(
        default_factory=lambda: ["W3"]
    )
    oot_window_id: str = "W4"
    root_cause_gate: RootCauseGateConfig = Field(default_factory=RootCauseGateConfig)
    missing_rates: MissingRateConfig = Field(default_factory=MissingRateConfig)
    rule_version: str
    oscillation_threshold: int = 3


class StrategyDefinition(ContractModel):
    strategy_code: str
    plan_code: str
    risk_level: str
    description: str
    parameters: dict = Field(default_factory=dict)


class StrategyCatalog(ContractModel):
    rule_version: str
    strategies: dict[str, StrategyDefinition]
    root_cause_rules: dict[str, dict]


class QualificationRuleConfig(ContractModel):
    rule_version: str
    min_recovery_rate: float = 0.60
    max_score_psi: float = 0.20
    max_train_valid_gap: float = 0.03
    required_oot_window_id: str = "W4"
    require_healthy_range: bool = True
    require_same_sample_bootstrap: bool = True


class RiskRuleConfig(ContractModel):
    rule_version: str
    high_risk_strategy_codes: list[str] = Field(default_factory=list)
    medium_risk_strategy_codes: list[str] = Field(default_factory=list)
    high_risk_root_cause_codes: list[str] = Field(default_factory=list)
    high_risk_actions: list[str] = Field(default_factory=list)
    manual_review_min_level: str = "HIGH"
