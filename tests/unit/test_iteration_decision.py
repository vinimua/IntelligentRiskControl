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
    DecisionInput,
    ManualReviewSubmission,
    MetricDegradation,
    RootCauseCandidate,
)
from packages.models.iteration.iteration_context import IterationContext, StrategyCandidate
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


def _strategy_candidate(
    strategy_code: str,
    *,
    historical_effectiveness: float | None,
    support_case_count: int,
    mitigates: bool = True,
    strategy_rank_score: float | None = None,
    rank_score_source: str = "CALIBRATED_HISTORY",
) -> StrategyCandidate:
    rank = (
        strategy_rank_score
        if strategy_rank_score is not None
        else (historical_effectiveness or 0.5)
    )
    return StrategyCandidate(
        strategy_code=strategy_code,
        recommends_relation_key=f"FEATURE_DRIFT|RECOMMENDS|{strategy_code}",
        mitigates_relation_key=(
            f"{strategy_code}|MITIGATES|FEATURE_DRIFT" if mitigates else ""
        ),
        relation_effective_weight_snapshot=rank,
        historical_effectiveness=historical_effectiveness,
        strategy_rank_score=rank,
        rank_score_source=rank_score_source,
        support_case_count=support_case_count,
        total_case_count=support_case_count,
        natural_case_count=support_case_count,
        confidence_lower_bound=0.70,
        executor_code="MODEL_RETRAIN",
    )


def _iteration_context(
    candidates: list[StrategyCandidate],
    *,
    degraded: bool = False,
) -> IterationContext:
    return IterationContext(
        context_pack_id="ctx-1",
        diagnosis_run_id="diagnosis-1",
        root_cause_code="FEATURE_DRIFT",
        weight_version="KG_WEIGHT_TEST",
        strategy_candidates=candidates,
        retrieval_degraded=degraded,
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


def test_kg_top_candidate_is_selected_when_l1_guardrails_pass():
    """KG owns Strategy candidates; L1 validates executability."""
    request = _decision_input(
        _root("feature_drift", evidence=["D", "I", "C", "T"]),
        [MetricDegradation(metric_code="AUC")],
    )
    context = _iteration_context([
        # KG 第一名是 sliding（rank 0.85 > 0.70），与 L1 的选择不一致
        _strategy_candidate("sliding_window_retrain", historical_effectiveness=0.85, support_case_count=30),
        _strategy_candidate("recent_weighted_retrain", historical_effectiveness=0.70, support_case_count=25),
    ])

    proposal = RepairDecisionService().decide_with_kg(request, context)

    assert proposal.action == RecommendedAction.MODEL_ITERATION
    assert proposal.requires_manual_review is False
    assert proposal.selected_strategy_code == "sliding_window_retrain"
    assert proposal.strategies[0].strategy_code == "sliding_window_retrain"
    assert proposal.final_strategy_code == "sliding_window_retrain"
    assert proposal.kg_consistency_status == "KG_SELECTED_L1_VALIDATED"
    assert proposal.strategy_source == "KG_WITH_L1_GUARDRAILS"
    assert proposal.kg_repair_required is False  # 排名分歧不是图谱缺陷
    assert set(proposal.kg_candidate_codes) == {
        "recent_weighted_retrain", "sliding_window_retrain",
    }
    assert "KG_STRATEGY:sliding_window_retrain" in proposal.decision_reasons
    assert "KG_CONSISTENCY:KG_SELECTED_L1_VALIDATED" in proposal.decision_reasons
    assert "KG_RANK_SCORE:0.850" in proposal.decision_reasons
    assert "KG_RANK_SOURCE:CALIBRATED_HISTORY" in proposal.decision_reasons


def test_kg_candidate_rank_score_source_initial_prior():
    """零案例时 KG 候选 rank 来源是 INITIAL_PRIOR：只标注来源，
    不覆盖 L1 的自动化结论（不因此强制人工）。"""
    request = _decision_input(
        _root("feature_drift", evidence=["D", "I", "C", "T"]),
        [MetricDegradation(metric_code="AUC")],
    )
    context = _iteration_context([
        _strategy_candidate(
            "recent_weighted_retrain",
            historical_effectiveness=None,
            strategy_rank_score=0.5,
            rank_score_source="INITIAL_PRIOR",
            support_case_count=0,
        ),
    ])

    proposal = RepairDecisionService().decide_with_kg(request, context)

    assert proposal.action == RecommendedAction.MODEL_ITERATION
    assert proposal.requires_manual_review is False
    assert proposal.kg_consistency_status == "KG_SELECTED_L1_VALIDATED"
    assert "KG_RANK_SCORE:0.500" in proposal.decision_reasons
    assert "KG_RANK_SOURCE:INITIAL_PRIOR" in proposal.decision_reasons


def test_kg_low_case_count_does_not_override_l1():
    """KG 低案例数只降低 KG 可信度，不覆盖 L1 已通过的自动化结论。"""
    request = _decision_input(
        _root("feature_drift", evidence=["D", "I", "C", "T"]),
        [MetricDegradation(metric_code="AUC")],
    )
    context = _iteration_context([
        _strategy_candidate("recent_weighted_retrain", historical_effectiveness=0.85, support_case_count=9),
    ])

    proposal = RepairDecisionService().decide_with_kg(request, context)

    assert proposal.action == RecommendedAction.MODEL_ITERATION
    assert proposal.requires_manual_review is False


def test_kg_retrieval_degraded_requires_manual_review():
    """KG is the strategy source, so degraded retrieval cannot silently use YAML."""
    request = _decision_input(
        _root("feature_drift", evidence=["D", "I", "C", "T"]),
        [MetricDegradation(metric_code="AUC")],
    )
    context = _iteration_context([], degraded=True)

    proposal = RepairDecisionService().decide_with_kg(request, context)

    assert proposal.action == RecommendedAction.MANUAL_REVIEW
    assert proposal.requires_manual_review is True
    assert proposal.strategies == []
    assert proposal.kg_consistency_status == "KG_UNAVAILABLE"
    assert proposal.kg_repair_required is True
    assert "KG_UNAVAILABLE" in proposal.decision_reasons
    assert "KG_REQUIRED_FOR_STRATEGY_SELECTION" in proposal.decision_reasons


def test_kg_empty_candidates_requires_manual_review():
    request = _decision_input(
        _root("feature_drift", evidence=["D", "I", "C", "T"]),
        [MetricDegradation(metric_code="AUC")],
    )
    context = _iteration_context([])

    proposal = RepairDecisionService().decide_with_kg(request, context)

    assert proposal.action == RecommendedAction.MANUAL_REVIEW
    assert proposal.requires_manual_review is True
    assert proposal.strategies == []
    assert not any(reason.startswith("KG_STRATEGY:") for reason in proposal.decision_reasons)
    assert proposal.kg_consistency_status == "KG_NO_CANDIDATES"
    assert proposal.kg_repair_required is True


def test_l1_guardrails_skip_incremental_for_logistic_regression():
    request = _decision_input(
        _root("feature_drift", evidence=["D", "I", "C", "T"]),
        [MetricDegradation(metric_code="AUC")],
    ).model_copy(update={
        "decay_degree": "SUSTAINED_30D",
        "change_pattern": "GRADUAL",
        "algorithm_family": "LogisticRegression",
    })
    context = _iteration_context([
        _strategy_candidate(
            "incremental_retrain",
            historical_effectiveness=0.90,
            support_case_count=40,
        ),
        _strategy_candidate(
            "recent_weighted_retrain",
            historical_effectiveness=0.70,
            support_case_count=25,
        ),
    ])

    proposal = RepairDecisionService().decide_with_kg(request, context)

    assert proposal.action == RecommendedAction.MODEL_ITERATION
    assert proposal.selected_strategy_code == "recent_weighted_retrain"
    assert proposal.kg_consistency_status == "KG_TOP_BLOCKED_L1_SELECTED_NEXT"
    assert (
        "KG_CANDIDATE_BLOCKED:incremental_retrain:"
        "INCREMENTAL_UNSUPPORTED_FOR_ALGORITHM"
        in proposal.decision_reasons
    )


def test_kg_candidate_without_mitigates_relation_marks_consistency_not_blocking():
    """A7 §4.2: MITIGATES 缺边不单独阻断 —— L1 仍输出策略，
    仅标记 kg_repair_required 和 KG_MITIGATES_MISSING。"""
    request = _decision_input(
        _root("feature_drift", evidence=["D", "I", "C", "T"]),
        [MetricDegradation(metric_code="AUC")],
    )
    context = _iteration_context([
        _strategy_candidate(
            "recent_weighted_retrain",
            historical_effectiveness=0.85,
            support_case_count=25,
            mitigates=False,
        ),
    ])

    proposal = RepairDecisionService().decide_with_kg(request, context)

    assert proposal.action == RecommendedAction.MODEL_ITERATION
    assert proposal.requires_manual_review is False
    assert proposal.kg_consistency_status == "KG_MITIGATES_MISSING"
    assert proposal.kg_repair_required is True
    assert "KG_MITIGATES_MISSING" in proposal.decision_reasons
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
                _root("OVERFITTING", score=0.80),
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
    data_args = {
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


def test_training_plan_uses_strategy_parameters_for_execution_config():
    proposal = RepairDecisionService().decide(
        _decision_input(
            _root("MODEL_AGING"),
            [MetricDegradation(metric_code="AUC")],
        )
    )
    strategy = proposal.strategies[0].model_copy(
        update={
            "parameters": {
                "algorithm": "lightgbm",
                "feature_schema_version": "features-from-strategy",
                "preprocessing_version": "preprocess-from-strategy",
                "label_versions": ["labels-from-strategy"],
                "training_window_ids": ["W2"],
                "validation_window_ids": ["W3"],
                "hyperparameters": {"n_estimators": 77, "max_depth": 3},
                "sample_weight_policy": {"type": "recent"},
            }
        }
    )
    proposal = proposal.model_copy(
        update={
            "status": ProposalStatus.APPROVED,
            "strategies": [strategy],
        }
    )
    risk = RiskAssessmentService().assess(proposal)

    plan = TrainingPlanBuilder().build(
        proposal,
        risk,
        approval_id="review-1",
        iteration_run_id="iteration-1",
        data_snapshot_ids=["snapshot-1"],
    )

    assert plan.algorithm == "lightgbm"
    assert plan.feature_schema_version == "features-from-strategy"
    assert plan.preprocessing_version == "preprocess-from-strategy"
    assert plan.label_versions == ["labels-from-strategy"]
    assert plan.windows.training_window_ids == ["W2"]
    assert plan.windows.validation_window_ids == ["W3"]
    assert plan.hyperparameter_space == {"n_estimators": 77, "max_depth": 3}
    assert plan.sample_weight_policy == {"type": "recent"}


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


def test_business_round_max_two_enforced():
    """A7 §5: 最大业务轮次统一为 2，第三轮被合同拒绝。"""
    from pydantic import ValidationError

    from packages.models.iteration.training_job import TrainingJobInput
    from packages.models.iteration.training_plan import TrainingPlan

    base_job = dict(
        training_job_id="j1", idempotency_key="k1", iteration_run_id="ir1",
        training_plan_id="tp1", experiment_id="e1",
        training_window_ids=["W2"], validation_window_ids=["W3"],
        data_snapshot_ids=["s1"], label_versions=["v1"],
        feature_schema_version="f1", preprocessing_version="p1",
        algorithm="lightgbm", qualification_rule_version="q1",
        base_model_version="c1", seed=42, artifact_output_uri="u",
    )
    with pytest.raises(ValidationError):
        TrainingJobInput(**base_job, business_round=3)

    base_plan = dict(
        training_plan_id="tp1", proposal_id="p1", approval_id="a1",
        iteration_run_id="ir1", experiment_id="e1",
        diagnosis_run_id="d1", model_id="m1", frozen_champion_version="c1",
        root_cause_code="FEATURE_DRIFT", strategy_code="recent_weighted_retrain",
        data_snapshot_ids=["s1"], label_versions=["v1"],
        feature_schema_version="f1", preprocessing_version="p1",
        algorithm="lightgbm", risk_level="MEDIUM",
        rollback_target="c0", rule_version="r1",
    )
    with pytest.raises(ValidationError):
        TrainingPlan(**base_plan, business_round=3)


def test_concept_drift_scope_derives_local_segment_strategy():
    """A7 §4: CONCEPT_DRIFT + LOCAL → 细分码 CONCEPT_DRIFT_LOCAL → segment。

    segment_weighted_retrain 必须有冻结客群证据；无证据时人工复核
    （不创建必然失败的 TrainingJob）。
    """
    request = _decision_input(
        _root("CONCEPT_DRIFT", evidence=["D", "I", "C", "T"]),
        [MetricDegradation(metric_code="AUC")],
    ).model_copy(update={"impact_scope": "LOCAL"})

    proposal = RepairDecisionService().decide(request)

    assert proposal.action == RecommendedAction.MANUAL_REVIEW
    assert proposal.requires_manual_review is True
    assert "SEGMENT_EVIDENCE_INSUFFICIENT_FALLBACK" in proposal.decision_reasons


def test_concept_drift_local_with_segment_evidence_selects_segment_strategy():
    """A7 §4: CONCEPT_DRIFT_LOCAL + 冻结客群证据 → segment_weighted_retrain。"""
    request = _decision_input(
        _root("CONCEPT_DRIFT", evidence=["D", "I", "C", "T"]),
        [MetricDegradation(metric_code="AUC")],
    ).model_copy(update={
        "impact_scope": "LOCAL",
        "segment_evidence": {
            "segment_column": "city_tier",
            "affected_segments": ["3", "4"],
            "segment_boost": 3.0,
        },
    })

    proposal = RepairDecisionService().decide(request)

    assert proposal.action == RecommendedAction.MODEL_ITERATION
    assert proposal.strategies[0].strategy_code == "segment_weighted_retrain"


def test_feature_drift_sustained_local_gradual_selects_incremental():
    """A7 §4: FEATURE_DRIFT + SUSTAINED_30D + GRADUAL + LightGBM → incremental_retrain。"""
    request = _decision_input(
        _root("FEATURE_DRIFT", evidence=["D", "I", "C", "T"]),
        [MetricDegradation(metric_code="AUC")],
    ).model_copy(update={
        "decay_degree": "SUSTAINED_30D",
        "change_pattern": "GRADUAL",
        "algorithm_family": "LightGBM",
    })

    proposal = RepairDecisionService().decide(request)

    assert proposal.action == RecommendedAction.MODEL_ITERATION
    assert proposal.strategies[0].strategy_code == "incremental_retrain"
    assert any(
        "STRUCTURED_CONTEXT:SUSTAINED_30D_LOCAL_GRADUAL_INCREMENTAL" in r
        for r in proposal.decision_reasons
    )


def test_feature_drift_sustained_gradual_unsupported_family_falls_back_to_l1_rule():
    """A7 §4: 算法家族不支持增量 → 回退 L1 YAML 规则（recent_weighted_retrain），
    不得生成 Worker 必然失败的增量任务。"""
    request = _decision_input(
        _root("FEATURE_DRIFT", evidence=["D", "I", "C", "T"]),
        [MetricDegradation(metric_code="AUC")],
    ).model_copy(update={
        "decay_degree": "SUSTAINED_30D",
        "change_pattern": "GRADUAL",
        "algorithm_family": "LogisticRegression",
    })

    proposal = RepairDecisionService().decide(request)

    assert proposal.action == RecommendedAction.MODEL_ITERATION
    assert proposal.strategies[0].strategy_code == "recent_weighted_retrain"
    assert any(
        "INCREMENTAL_UNSUPPORTED_FOR_ALGORITHM_FALLBACK_L1_RULE" in r
        for r in proposal.decision_reasons
    )


def test_feature_fragility_round2_with_attribution_evidence_selects_feature_selection():
    """A7 §5: 第二轮 + 人工批准 + 完整归因证据 → feature_selection_retrain。"""
    request = _decision_input(
        _root("FEATURE_FRAGILITY", evidence=["D", "I", "C", "T"]),
        [MetricDegradation(metric_code="AUC")],
    ).model_copy(update={
        "business_round": 2,
        "manual_approval": True,
        "failure_report_id": "failure-1",
        "unstable_feature_codes": ["age", "income_level"],
        "feature_evidence_source": "QUALIFICATION_STABILITY_GATE_REASONS",
    })

    proposal = RepairDecisionService().decide(request)

    assert proposal.action == RecommendedAction.MODEL_ITERATION
    assert proposal.strategies[0].strategy_code == "feature_selection_retrain"
    assert any(
        "STRUCTURED_CONTEXT:ROUND2_ATTRIBUTION_FEATURE_SELECTION" in r
        for r in proposal.decision_reasons
    )


def test_feature_fragility_round2_without_evidence_falls_back_to_l1_rule():
    """A7 §5: 归因证据不足（无报告/无确认特征）→ 不进入特征筛选，
    回退 YAML 规则（regularized_retrain 优先）。"""
    request = _decision_input(
        _root("FEATURE_FRAGILITY", evidence=["D", "I", "C", "T"]),
        [MetricDegradation(metric_code="AUC")],
    ).model_copy(update={
        "business_round": 2,
        "manual_approval": True,
        "failure_report_id": None,  # 归因失败：没有真实报告
        "unstable_feature_codes": [],
        "feature_evidence_source": None,
    })

    proposal = RepairDecisionService().decide(request)

    assert proposal.action == RecommendedAction.MODEL_ITERATION
    assert proposal.strategies[0].strategy_code == "regularized_retrain"
    assert any(
        "FEATURE_SELECTION_EVIDENCE_INSUFFICIENT_FALLBACK_L1_RULE" in r
        for r in proposal.decision_reasons
    )


def test_short_term_7d_observes_only_even_with_ranking_loss():
    """任务一统一入口: SHORT_TERM_7D → 继续观察，不进入 A7。"""
    request = _decision_input(
        _root("FEATURE_DRIFT", evidence=["D", "I", "C", "T"]),
        [MetricDegradation(metric_code="AUC")],
    ).model_copy(update={"decay_degree": "SHORT_TERM_7D"})

    proposal = RepairDecisionService().decide(request)

    assert proposal.action == RecommendedAction.CONTINUE_OBSERVATION
    assert proposal.need_iteration is False
    assert "SHORT_TERM_7D_OBSERVE_ONLY_NOT_A7" in proposal.decision_reasons


def test_severe_forces_manual_review():
    """任务一统一入口: SEVERE → 强制人工复核，不自动训练。"""
    request = _decision_input(
        _root("FEATURE_DRIFT", evidence=["D", "I", "C", "T"]),
        [MetricDegradation(metric_code="AUC")],
    ).model_copy(update={"decay_degree": "SEVERE"})

    proposal = RepairDecisionService().decide(request)

    assert proposal.action == RecommendedAction.MANUAL_REVIEW
    assert proposal.requires_manual_review is True
    assert "SEVERE_REQUIRES_MANUAL_REVIEW" in proposal.decision_reasons


def test_mitigates_checked_on_selected_kg_candidate():
    """MITIGATES audit follows the KG candidate that passes L1 guardrails."""
    request = _decision_input(
        _root("feature_drift", evidence=["D", "I", "C", "T"]),
        [MetricDegradation(metric_code="AUC")],
    )
    context = _iteration_context([
        _strategy_candidate("sliding_window_retrain", historical_effectiveness=0.85, support_case_count=30),
        _strategy_candidate("recent_weighted_retrain", historical_effectiveness=0.70, support_case_count=25, mitigates=False),
    ])

    proposal = RepairDecisionService().decide_with_kg(request, context)

    assert proposal.selected_strategy_code == "sliding_window_retrain"
    assert proposal.kg_consistency_status == "KG_SELECTED_L1_VALIDATED"
    assert proposal.kg_repair_required is False
    assert "MITIGATES:sliding_window_retrain|MITIGATES|FEATURE_DRIFT" in proposal.decision_reasons
