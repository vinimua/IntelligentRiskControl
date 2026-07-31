"""把层次诊断结果转换为不可执行的确定性修复建议。"""

from uuid import uuid4

from packages.models.common.enums import (
    ConfidenceLevel,
    ProposalStatus,
    RecommendedAction,
)
from packages.models.iteration.decision_proposal import (
    DecisionInput,
    DecisionProposal,
    RootCauseCandidate,
    StrategySelection,
)

from .config_loader import IterationConfigBundle, load_iteration_config


RANKING_METRICS = {"AUC", "KS"}
CALIBRATION_METRICS = {"BRIER", "BRIER_SCORE", "ECE"}


class RepairDecisionService:
    def __init__(self, config: IterationConfigBundle | None = None) -> None:
        self.config = config or load_iteration_config()

    @staticmethod
    def _normalize_code(code: str) -> str:
        return code.strip().upper().replace("-", "_").replace(" ", "_")

    def _root_gate(
        self, causes: list[RootCauseCandidate]
    ) -> tuple[bool, list[str]]:
        rules = self.config.iteration.root_cause_gate
        primary = causes[0]
        reasons: list[str] = []
        if primary.score < rules.min_primary_score:
            reasons.append("PRIMARY_ROOT_CAUSE_SCORE_BELOW_THRESHOLD")
        if primary.evidence_coverage < rules.min_evidence_coverage:
            reasons.append("EVIDENCE_COVERAGE_BELOW_THRESHOLD")
        if len(causes) > 1 and primary.score - causes[1].score < rules.min_top1_top2_gap:
            reasons.append("TOP_ROOT_CAUSES_TOO_CLOSE")
        return not reasons, reasons

    @staticmethod
    def _feature_drift_evidence_ok(primary: RootCauseCandidate) -> bool:
        evidence = {code.upper() for code in primary.evidence_types}
        return {"D", "I", "T"}.issubset(evidence) and bool({"C", "R"} & evidence)

    def decide(self, request: DecisionInput) -> DecisionProposal:
        causes = sorted(request.root_causes, key=lambda item: item.score, reverse=True)
        primary = causes[0]
        primary_code = self._normalize_code(primary.root_cause_code)
        gate_passed, gate_reasons = self._root_gate(causes)

        if not gate_passed:
            return self._proposal(
                request=request,
                causes=causes,
                action=RecommendedAction.MANUAL_REVIEW,
                strategy_codes=[],
                confidence=ConfidenceLevel.LOW,
                reasons=gate_reasons,
                requires_manual_review=True,
            )

        degraded = {
            self._normalize_code(metric.metric_code)
            for metric in request.degraded_metrics
            if metric.degraded
        }
        ranking_degraded = bool(degraded & RANKING_METRICS)
        calibration_degraded = bool(degraded & CALIBRATION_METRICS)

        if not ranking_degraded and calibration_degraded:
            return self._proposal(
                request,
                causes,
                RecommendedAction.CALIBRATION_ADJUSTMENT,
                ["calibration_only"],
                ConfidenceLevel.HIGH,
                ["RANKING_STABLE_BUT_CALIBRATION_DEGRADED"],
            )

        if not ranking_degraded and not calibration_degraded:
            if request.business_objective_changed:
                return self._proposal(
                    request,
                    causes,
                    RecommendedAction.THRESHOLD_ADJUSTMENT,
                    ["threshold_only"],
                    ConfidenceLevel.HIGH,
                    ["MODEL_AND_CALIBRATION_STABLE_BUSINESS_OBJECTIVE_CHANGED"],
                    requires_manual_review=True,
                )
            if primary_code in {"FEATURE_DRIFT", "SEGMENT_DRIFT"}:
                return self._proposal(
                    request,
                    causes,
                    RecommendedAction.CONTINUE_OBSERVATION,
                    [],
                    ConfidenceLevel.HIGH,
                    ["DRIFT_WITHOUT_RANKING_PERFORMANCE_LOSS"],
                )

        root_rule = self.config.strategies.root_cause_rules.get(primary_code)
        if root_rule is None:
            return self._proposal(
                request,
                causes,
                RecommendedAction.MANUAL_REVIEW,
                [],
                ConfidenceLevel.MEDIUM,
                ["NO_DETERMINISTIC_ROOT_CAUSE_RULE"],
                requires_manual_review=True,
            )

        if primary_code == "FEATURE_DRIFT" and not self._feature_drift_evidence_ok(
            primary
        ):
            return self._proposal(
                request,
                causes,
                RecommendedAction.MANUAL_REVIEW,
                [],
                ConfidenceLevel.MEDIUM,
                ["FEATURE_DRIFT_EVIDENCE_CHAIN_INCOMPLETE"],
                requires_manual_review=True,
            )

        action = RecommendedAction(root_rule["action"])
        strategies = list(root_rule.get("strategy_codes", []))
        reasons = [f"ROOT_CAUSE_RULE_MATCHED:{primary_code}"]

        if ranking_degraded and action in {
            RecommendedAction.CALIBRATION_ADJUSTMENT,
            RecommendedAction.THRESHOLD_ADJUSTMENT,
        }:
            return self._proposal(
                request,
                causes,
                RecommendedAction.MANUAL_REVIEW,
                [],
                ConfidenceLevel.MEDIUM,
                ["RANKING_DEGRADED_REQUIRES_RANKING_ROOT_CAUSE_DIAGNOSIS"],
                requires_manual_review=True,
            )

        if action == RecommendedAction.DATA_REPAIR and request.data_repair_completed:
            reasons.append("DATA_REPAIR_ALREADY_COMPLETED_REPLAY_REQUIRED")
        if (
            action == RecommendedAction.PIPELINE_REPAIR
            and request.pipeline_repair_completed
        ):
            reasons.append("PIPELINE_REPAIR_ALREADY_COMPLETED_REPLAY_REQUIRED")

        return self._proposal(
            request,
            causes,
            action,
            strategies,
            ConfidenceLevel.HIGH,
            reasons,
        )

    def _proposal(
        self,
        request: DecisionInput,
        causes: list[RootCauseCandidate],
        action: RecommendedAction,
        strategy_codes: list[str],
        confidence: ConfidenceLevel,
        reasons: list[str],
        requires_manual_review: bool = False,
    ) -> DecisionProposal:
        selections: list[StrategySelection] = []
        for code in strategy_codes:
            definition = self.config.strategies.strategies[code]
            selections.append(
                StrategySelection(
                    strategy_code=code,
                    parameters=definition.parameters,
                    rationale=definition.description,
                )
            )
        return DecisionProposal(
            proposal_id=str(uuid4()),
            diagnosis_run_id=request.diagnosis_run_id,
            lifecycle_run_id=request.lifecycle_run_id,
            model_id=request.model_id,
            champion_version=request.champion_version,
            primary_root_cause_code=self._normalize_code(
                causes[0].root_cause_code
            ),
            primary_root_cause_score=causes[0].score,
            top1_top2_gap=(
                causes[0].score - causes[1].score if len(causes) > 1 else None
            ),
            evidence_coverage=causes[0].evidence_coverage,
            contributing_root_cause_codes=[
                self._normalize_code(item.root_cause_code) for item in causes[1:]
            ],
            action=action,
            need_iteration=action == RecommendedAction.MODEL_ITERATION,
            strategies=selections,
            selected_strategy_code=(
                selections[0].strategy_code if selections else None
            ),
            target_metric_codes=[
                metric.metric_code
                for metric in request.degraded_metrics
                if metric.degraded
            ],
            proposed_window_policy=self.config.iteration.training_window_policy,
            confidence=confidence,
            decision_reasons=reasons,
            status=(
                ProposalStatus.PENDING_REVIEW
                if requires_manual_review
                else ProposalStatus.DRAFT
            ),
            executable=False,
            requires_manual_review=requires_manual_review,
            rule_version=request.rule_version,
            rule_versions={
                "decision": request.rule_version,
                "strategy": self.config.strategies.rule_version,
            },
        )
