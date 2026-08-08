"""审核通过后构造不可越界的训练计划。"""

from uuid import uuid4

from packages.models.common.enums import (
    ProposalStatus,
    RecommendedAction,
    TrainingPlanStatus,
)
from packages.models.iteration.decision_proposal import DecisionProposal
from packages.models.iteration.risk_assessment import RiskAssessment
from packages.models.iteration.training_plan import TrainingPlan, TrainingWindowSpec

from .config_loader import IterationConfigBundle, load_iteration_config


class TrainingPlanBuilder:
    def __init__(self, config: IterationConfigBundle | None = None) -> None:
        self.config = config or load_iteration_config()

    def build(
        self,
        proposal: DecisionProposal,
        risk: RiskAssessment,
        *,
        approval_id: str,
        iteration_run_id: str,
        model_algorithm: str | None = None,
        feature_schema_version: str | None = None,
        preprocessing_version: str | None = None,
        business_round: int = 1,
        data_snapshot_ids: list[str] | None = None,
        label_versions: list[str] | None = None,
    ) -> TrainingPlan:
        if proposal.action != RecommendedAction.MODEL_ITERATION:
            raise ValueError("only MODEL_ITERATION can produce a TrainingPlan")
        if proposal.status != ProposalStatus.APPROVED:
            raise ValueError("DecisionProposal must be approved before plan generation")
        if not approval_id:
            raise ValueError("approval_id is required")
        if not proposal.strategies:
            raise ValueError("approved model iteration has no selected strategy")
        data_snapshot_ids = data_snapshot_ids or []
        strategy = proposal.strategies[0]
        strategy_params = strategy.parameters or {}
        resolved_label_versions = (
            label_versions
            or strategy_params.get("label_versions")
            or ["label-v1"]
        )
        if not data_snapshot_ids or not resolved_label_versions:
            raise ValueError("data snapshots and observed label versions are required")

        windows = TrainingWindowSpec(
            baseline_window_id=self.config.iteration.baseline_window_id,
            training_window_ids=(
                strategy_params.get("training_window_ids")
                or strategy_params.get("allowed_training_window_ids")
                or self.config.iteration.default_training_window_ids
            ),
            validation_window_ids=(
                strategy_params.get("validation_window_ids")
                or self.config.iteration.default_validation_window_ids
            ),
            oot_window_id=self.config.iteration.oot_window_id,
            oot_locked=True,
        )
        return TrainingPlan(
            training_plan_id=str(uuid4()),
            proposal_id=proposal.proposal_id,
            approval_id=approval_id,
            iteration_run_id=iteration_run_id,
            experiment_id=str(uuid4()),
            business_round=business_round,
            diagnosis_run_id=proposal.diagnosis_run_id,
            model_id=proposal.model_id,
            frozen_champion_version=proposal.champion_version,
            root_cause_code=proposal.primary_root_cause_code,
            strategy_code=strategy.strategy_code,
            strategy_parameters=strategy.parameters,
            target_metric_codes=proposal.target_metric_codes,
            windows=windows,
            data_snapshot_ids=data_snapshot_ids,
            label_versions=resolved_label_versions,
            sample_weight_policy=strategy_params.get("sample_weight_policy", {}),
            feature_schema_version=(
                feature_schema_version
                or strategy_params.get("feature_schema_version")
                or "feature-schema-v1"
            ),
            preprocessing_version=(
                preprocessing_version
                or strategy_params.get("preprocessing_version")
                or "preprocess-v1"
            ),
            algorithm=model_algorithm or strategy_params.get("algorithm") or "lightgbm",
            hyperparameter_space=strategy_params.get("hyperparameters", {}),
            risk_level=risk.risk_level.value,
            max_business_rounds=self.config.iteration.max_iteration_rounds,
            rollback_target=proposal.champion_version,
            status=TrainingPlanStatus.READY,
            rule_version=self.config.iteration.rule_version,
        )
