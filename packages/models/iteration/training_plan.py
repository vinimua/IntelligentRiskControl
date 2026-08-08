"""训练计划合同。"""

from pydantic import Field, model_validator

from ..common.base import ContractModel
from ..common.enums import TrainingPlanStatus


class TrainingWindowSpec(ContractModel):
    baseline_window_id: str = "W1"
    training_window_ids: list[str] = Field(default_factory=lambda: ["W2"])
    validation_window_ids: list[str] = Field(default_factory=lambda: ["W3"])
    oot_window_id: str = "W4"
    oot_locked: bool = True

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
        return self


class TrainingPlan(ContractModel):
    training_plan_id: str
    proposal_id: str
    approval_id: str
    iteration_run_id: str
    experiment_id: str
    business_round: int = Field(ge=1, le=3)
    diagnosis_run_id: str
    model_id: str
    frozen_champion_version: str
    root_cause_code: str
    strategy_code: str
    strategy_parameters: dict = Field(default_factory=dict)
    target_metric_codes: list[str] = Field(default_factory=list)
    windows: TrainingWindowSpec = Field(default_factory=TrainingWindowSpec)
    data_eligibility_assessment_ids: list[str] = Field(default_factory=list)
    data_snapshot_ids: list[str] = Field(min_length=1)
    label_versions: list[str] = Field(min_length=1)
    sample_weight_policy: dict = Field(default_factory=dict)
    feature_schema_version: str
    preprocessing_version: str
    algorithm: str
    hyperparameter_space: dict = Field(default_factory=dict)
    random_seed: int = 2026
    qualification_rule_version: str = "qualification-rules-v1"
    risk_level: str
    max_business_rounds: int = 3
    rollback_target: str
    status: TrainingPlanStatus = TrainingPlanStatus.DRAFT
    blocking_reasons: list[str] = Field(default_factory=list)
    rule_version: str
