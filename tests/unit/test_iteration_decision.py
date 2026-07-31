from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from apps.modelops_api.services.iteration import (
    IterationRoundController,
    RepairDecisionService,
    RiskAssessmentService,
    TrainingPlanBuilder,
)
from packages.models.common.enums import (
    ConfidenceLevel,
    DimensionCode,
    ProposalStatus,
    RecommendedAction,
    ReviewDecision,
    RiskLevel,
)
from packages.models.iteration import (
    DataEligibilityInput,
    DecisionInput,
    ManualReviewSubmission,
    MetricDegradation,
    RootCauseCandidate,
)
from apps.modelops_api.services.iteration import DataEligibilityService
from packages.models.iteration.training_job import TrainingJobInput
from packages.models.iteration.round_control import RetryIdentity


def _root(
    code: str,
    *,
    score: float = 0.90,
    coverage: float = 0.90,
    evidence: list[str] | None = None,
) -> RootCauseCandidate:
    return RootCauseCandidate(
        root_cause_code=code,
        dimension=DimensionCode.FEATURE,
        score=score,
        evidence_coverage=coverage,
        evidence_types=evidence or [],
    )


def _decision_input(
    root: RootCauseCandidate,
    metrics: list[MetricDegradation] | None = None,
) -> DecisionInput:
    return DecisionInput(
        diagnosis_run_id="diagnosis-1",
        lifecycle_run_id="11111111-1111-1111-1111-111111111111",
        model_id="credit-model",
        champion_version="champion-v1",
        root_causes=[root],
        degraded_metrics=metrics or [],
    )


def test_feature_drift_with_complete_evidence_and_auc_loss_iterates():
    request = _decision_input(
        _root("feature_drift", evidence=["D", "I", "C", "T"]),
        [MetricDegradation(metric_code="AUC")],
    )

    proposal = RepairDecisionService().decide(request)

    assert proposal.action == RecommendedAction.MODEL_ITERATION
    assert proposal.need_iteration is True
    assert proposal.strategies[0].strategy_code == "recent_weighted_retrain"


def test_feature_drift_without_ranking_loss_only_observes():
    proposal = RepairDecisionService().decide(
        _decision_input(_root("FEATURE_DRIFT", evidence=["D", "I", "R", "T"]))
    )

    assert proposal.action == RecommendedAction.CONTINUE_OBSERVATION
    assert proposal.need_iteration is False


def test_calibration_is_checked_after_ranking_and_before_threshold():
    request = _decision_input(
        _root("PRIOR_PROBABILITY_SHIFT"),
        [MetricDegradation(metric_code="ECE")],
    )
    request = request.model_copy(update={"business_objective_changed": True})

    proposal = RepairDecisionService().decide(request)

    assert proposal.action == RecommendedAction.CALIBRATION_ADJUSTMENT
    assert proposal.need_iteration is False


def test_calibration_only_is_forbidden_when_ranking_also_degraded():
    proposal = RepairDecisionService().decide(
        _decision_input(
            _root("CALIBRATION_DRIFT"),
            [
                MetricDegradation(metric_code="AUC"),
                MetricDegradation(metric_code="ECE"),
            ],
        )
    )
    assert proposal.action == RecommendedAction.MANUAL_REVIEW
    assert proposal.need_iteration is False


def test_close_root_cause_scores_force_manual_review():
    request = _decision_input(_root("MODEL_AGING", score=0.85))
    request = request.model_copy(
        update={
            "root_causes": [
                _root("MODEL_AGING", score=0.85),
                _root("OVERFITTING", score=0.75),
            ]
        }
    )

    proposal = RepairDecisionService().decide(request)

    assert proposal.action == RecommendedAction.MANUAL_REVIEW
    assert proposal.confidence == ConfidenceLevel.LOW
    assert proposal.requires_manual_review is True


def test_data_issue_never_trains_a_model():
    proposal = RepairDecisionService().decide(
        _decision_input(
            _root("DATA_MISSING"),
            [MetricDegradation(metric_code="AUC")],
        )
    )

    assert proposal.action == RecommendedAction.DATA_REPAIR
    assert proposal.need_iteration is False
    assert proposal.strategies[0].strategy_code == "champion_replay"


def test_threshold_adjustment_is_high_risk_and_needs_review():
    request = _decision_input(_root("BUSINESS_THRESHOLD_MISMATCH"))
    request = request.model_copy(update={"business_objective_changed": True})
    proposal = RepairDecisionService().decide(request)

    risk = RiskAssessmentService().assess(proposal)

    assert proposal.action == RecommendedAction.THRESHOLD_ADJUSTMENT
    assert risk.risk_level == RiskLevel.HIGH
    assert risk.requires_manual_review is True


def test_training_plan_requires_approved_model_iteration():
    proposal = RepairDecisionService().decide(
        _decision_input(
            _root("MODEL_AGING"),
            [MetricDegradation(metric_code="AUC")],
        )
    )
    risk = RiskAssessmentService().assess(proposal)
    eligibility = DataEligibilityService().evaluate(
        DataEligibilityInput(
            window_id="W2",
            label_missing_rate=0,
            data_snapshot_id="snapshot-1",
        )
    )
    data_args = {
        "data_eligibility_assessments": [("eligibility-1", eligibility)],
        "data_snapshot_ids": ["snapshot-1"],
        "label_versions": ["labels-v1"],
    }

    with pytest.raises(ValueError, match="approved"):
        TrainingPlanBuilder().build(
            proposal,
            risk,
            approval_id="review-1",
            iteration_run_id="iteration-1",
            model_algorithm="xgboost",
            feature_schema_version="features-v1",
            preprocessing_version="preprocess-v1",
            **data_args,
        )

    approved = proposal.model_copy(update={"status": ProposalStatus.APPROVED})
    plan = TrainingPlanBuilder().build(
        approved,
        risk,
        approval_id="review-1",
        iteration_run_id="iteration-1",
        model_algorithm="xgboost",
        feature_schema_version="features-v1",
        preprocessing_version="preprocess-v1",
        **data_args,
    )
    assert plan.frozen_champion_version == "champion-v1"
    assert "W4" not in plan.windows.training_window_ids
    assert plan.windows.oot_locked is True


def test_training_job_rejects_oot_leakage():
    with pytest.raises(ValidationError, match="OOT"):
        TrainingJobInput(
            training_job_id="job-1",
            idempotency_key="job-1",
            iteration_run_id="run-1",
            training_plan_id="plan-1",
            experiment_id="experiment-1",
            business_round=1,
            strategy_code="stable_refit",
            training_window_ids=["W3", "W4"],
            validation_window_ids=["W2"],
            data_snapshot_ids=["snapshot-1"],
            label_versions=["labels-v1"],
            feature_schema_version="features-v1",
            preprocessing_version="preprocess-v1",
            algorithm="xgboost",
            qualification_rule_version="qualification-rules-v1",
            base_model_version="champion-v1",
            seed=2026,
            artifact_output_uri="s3://models/experiment-1",
        )


def test_rejected_manual_review_requires_adjustment_instructions():
    with pytest.raises(ValidationError, match="adjustment_instructions"):
        ManualReviewSubmission(
            proposal_id="proposal-1",
            reviewer_id="reviewer-1",
            decision=ReviewDecision.REJECT,
            reason="证据不足",
            reviewed_at=datetime.now(UTC),
        )


def test_technical_retry_does_not_consume_business_round_or_change_ids():
    transition = IterationRoundController().technical_retry(
        RetryIdentity(
            training_job_id="job-1",
            experiment_id="experiment-1",
            business_round=2,
            technical_retry_count=0,
        )
    )

    assert transition.allowed is True
    assert transition.next_business_round == 2
    assert transition.training_job_id == "job-1"
    assert transition.experiment_id == "experiment-1"


def test_fourth_business_round_is_forbidden():
    transition = IterationRoundController().next_business_round(3)
    assert transition.allowed is False
