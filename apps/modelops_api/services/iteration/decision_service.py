"""把层次诊断结果转换为不可执行的确定性修复建议。"""

from uuid import uuid4

from packages.models.common.enums import (
    ConfidenceLevel,
    ProposalStatus,
    RecommendedAction,
    TrainingMode,
)
from packages.models.iteration.decision_proposal import (
    DecisionInput,
    DecisionProposal,
    RootCauseCandidate,
    StrategySelection,
)
from packages.models.iteration.a7_contracts import (
    A7DecisionEnvelope,
    L1StrategyDecision,
)

from .config_loader import IterationConfigBundle, load_iteration_config


RANKING_METRICS = {"AUC", "KS"}
CALIBRATION_METRICS = {"BRIER", "BRIER_SCORE", "ECE"}
INCREMENTAL_ALGORITHM_FAMILIES = {"lightgbm", "xgboost"}
SECOND_ROUND_STRATEGIES = {
    "regularized_retrain",
    "feature_reconstruction",
    "feature_selection_retrain",
}


class RepairDecisionService:
    def __init__(self, config: IterationConfigBundle | None = None) -> None:
        self.config = config or load_iteration_config()

    @staticmethod
    def _normalize_code(code: str) -> str:
        return code.strip().upper().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _normalize_algorithm_family(family: str | None) -> str:
        normalized = (
            (family or "").strip().lower().replace("-", "_").replace(" ", "_")
        )
        return {
            "logisticregression": "logistic_regression",
            "randomforest": "random_forest",
            "lightgbm": "lightgbm",
            "xgboost": "xgboost",
        }.get(normalized, normalized)

    @classmethod
    def _supports_incremental(cls, family: str | None) -> bool:
        return cls._normalize_algorithm_family(family) in INCREMENTAL_ALGORITHM_FAMILIES

    @staticmethod
    def _derive_effective_root_code(
        primary_code: str, request: DecisionInput,
    ) -> str:
        """A7 §4/§5: L1 用结构化上下文推导有效根因码。

        CONCEPT_DRIFT 总类不直接选策略，按 impact_scope/change_pattern 推导细分码；
        KG 不能替 L1 完成这个选择。
        """
        if primary_code == "CONCEPT_DRIFT":
            scope = (request.impact_scope or "").upper()
            pattern = (request.change_pattern or "").upper()
            if "LOCAL" in scope or "LOCAL" in pattern:
                return "CONCEPT_DRIFT_LOCAL"
            if "SUDDEN" in pattern:
                return "CONCEPT_DRIFT_SUDDEN"
            if "GLOBAL" in scope:
                return "CONCEPT_DRIFT_GLOBAL"
            if "GRADUAL" in pattern:
                return "CONCEPT_DRIFT_GRADUAL"
        return primary_code

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

    def decide_with_kg(
        self,
        request: DecisionInput,
        iteration_context,
    ) -> DecisionProposal:
        """Select a KG strategy candidate after L1 guardrail validation.

        KG owns RootCause -> Strategy candidate recall and ranking. L1 keeps the
        non-strategy gates from decide(request), then filters KG candidates for
        compliance, safety, business-round, evidence, and executor feasibility.
        """
        l1_shell = self.decide(request)
        if l1_shell.action != RecommendedAction.MODEL_ITERATION:
            return l1_shell

        causes = sorted(request.root_causes, key=lambda item: item.score, reverse=True)
        primary_code = self._derive_effective_root_code(
            self._normalize_code(causes[0].root_cause_code),
            request,
        )
        kg_candidates = getattr(iteration_context, "strategy_candidates", []) or []

        if iteration_context.retrieval_degraded:
            return self._proposal(
                request,
                causes,
                RecommendedAction.MANUAL_REVIEW,
                [],
                ConfidenceLevel.MEDIUM,
                self._strategy_base_reasons(l1_shell)
                + ["KG_UNAVAILABLE", "KG_REQUIRED_FOR_STRATEGY_SELECTION"],
                requires_manual_review=True,
            ).model_copy(update={
                "kg_consistency_status": "KG_UNAVAILABLE",
                "kg_repair_required": True,
                "kg_candidate_codes": [],
            })

        if not kg_candidates:
            return self._proposal(
                request,
                causes,
                RecommendedAction.MANUAL_REVIEW,
                [],
                ConfidenceLevel.MEDIUM,
                self._strategy_base_reasons(l1_shell)
                + ["KG_NO_CANDIDATES", "KG_REQUIRED_FOR_STRATEGY_SELECTION"],
                requires_manual_review=True,
            ).model_copy(update={
                "kg_consistency_status": "KG_NO_CANDIDATES",
                "kg_repair_required": True,
                "kg_candidate_codes": [],
            })

        ranked_candidates = sorted(
            kg_candidates,
            key=lambda c: (c.strategy_rank_score, c.support_case_count),
            reverse=True,
        )
        candidate_codes = sorted({c.strategy_code for c in ranked_candidates})
        selected = None
        blocked_reasons: list[str] = []
        for candidate in ranked_candidates:
            reasons = self._candidate_l1_blocking_reasons(
                primary_code, request, candidate
            )
            if reasons:
                blocked_reasons.extend(
                    f"KG_CANDIDATE_BLOCKED:{candidate.strategy_code}:{reason}"
                    for reason in reasons
                )
                continue
            selected = candidate
            break

        best = ranked_candidates[0]
        if selected is None:
            return self._proposal(
                request,
                causes,
                RecommendedAction.MANUAL_REVIEW,
                [],
                ConfidenceLevel.MEDIUM,
                self._strategy_base_reasons(l1_shell)
                + [
                    f"KG_STRATEGY:{best.strategy_code}",
                    f"KG_RANK_SCORE:{best.strategy_rank_score:.3f}",
                    f"KG_RANK_SOURCE:{best.rank_score_source}",
                    "KG_CANDIDATES_BLOCKED_BY_L1_GUARDRAILS",
                ]
                + blocked_reasons,
                requires_manual_review=True,
            ).model_copy(update={
                "kg_consistency_status": "KG_CANDIDATES_BLOCKED_BY_L1",
                "kg_repair_required": False,
                "kg_candidate_codes": candidate_codes,
            })

        reasons_extra = [
            f"KG_STRATEGY:{selected.strategy_code}",
            f"KG_RANK_SCORE:{selected.strategy_rank_score:.3f}",
            f"KG_RANK_SOURCE:{selected.rank_score_source}",
            f"KG_SUPPORT_CASES:{selected.support_case_count}",
            f"KG_RELATION:{selected.recommends_relation_key}",
            "L1_GUARDRAILS_PASSED",
        ]
        if blocked_reasons:
            reasons_extra.extend(blocked_reasons)
        if selected.mitigates_relation_key:
            reasons_extra.append(f"MITIGATES:{selected.mitigates_relation_key}")
            status = (
                "KG_SELECTED_L1_VALIDATED"
                if selected.strategy_code == best.strategy_code
                else "KG_TOP_BLOCKED_L1_SELECTED_NEXT"
            )
            repair_required = False
        else:
            reasons_extra.append("KG_MITIGATES_MISSING")
            status = "KG_MITIGATES_MISSING"
            repair_required = True
        reasons_extra.append(f"KG_CONSISTENCY:{status}")

        return l1_shell.model_copy(update={
            "strategies": [self._selection_from_kg_candidate(selected)],
            "selected_strategy_code": selected.strategy_code,
            "final_strategy_code": selected.strategy_code,
            "decision_reasons": self._strategy_base_reasons(l1_shell)
            + reasons_extra,
            "kg_consistency_status": status,
            "kg_repair_required": repair_required,
            "kg_candidate_codes": candidate_codes,
            "strategy_source": "KG_WITH_L1_GUARDRAILS",
        })

    @staticmethod
    def _strategy_base_reasons(proposal: DecisionProposal) -> list[str]:
        return [
            reason for reason in proposal.decision_reasons
            if not reason.startswith("STRUCTURED_CONTEXT:")
        ]

    def _selection_from_kg_candidate(self, candidate) -> StrategySelection:
        definition = self.config.strategies.strategies[candidate.strategy_code]
        parameters = dict(definition.parameters)
        parameters.update({
            "kg_recommends_relation_key": candidate.recommends_relation_key,
            "kg_mitigates_relation_key": candidate.mitigates_relation_key,
            "strategy_rank_score": candidate.strategy_rank_score,
            "rank_score_source": candidate.rank_score_source,
            "support_case_count": candidate.support_case_count,
            "allowed_training_window_ids": list(candidate.allowed_training_window_ids),
            "validation_window_ids": list(candidate.validation_window_ids),
            "algorithm": candidate.algorithm,
            "feature_schema_version": candidate.feature_schema_version,
            "preprocessing_version": candidate.preprocessing_version,
            "label_versions": list(candidate.label_versions),
            "hyperparameters": dict(candidate.hyperparameters),
            "sample_weight_policy": dict(candidate.sample_weight_policy),
            "strategy_tier": candidate.strategy_tier,
            "required_context": list(candidate.required_context),
        })
        return StrategySelection(
            strategy_code=candidate.strategy_code,
            parameters=parameters,
            rationale=definition.description,
            primary_training_mode=definition.primary_training_mode,
        )

    def _candidate_l1_blocking_reasons(
        self,
        primary_code: str,
        request: DecisionInput,
        candidate,
    ) -> list[str]:
        code = candidate.strategy_code
        reasons: list[str] = []
        if code not in self.config.strategies.strategies:
            return [f"STRATEGY_NOT_IN_LOCAL_EXECUTION_CATALOG:{code}"]
        if (
            self.config.iteration.allowed_strategy_codes
            and code not in self.config.iteration.allowed_strategy_codes
        ):
            reasons.append(f"STRATEGY_NOT_ALLOWED_BY_L1:{code}")

        definition = self.config.strategies.strategies[code]
        if definition.primary_training_mode == TrainingMode.NONE:
            reasons.append(f"NON_TRAINING_STRATEGY_NOT_VALID_FOR_MODEL_ITERATION:{code}")

        decay = (request.decay_degree or "").upper()
        scope = (request.impact_scope or "").upper()
        pattern = (request.change_pattern or "").upper()

        if code == "incremental_retrain":
            if primary_code != "FEATURE_DRIFT":
                reasons.append("INCREMENTAL_REQUIRES_FEATURE_DRIFT")
            if decay != "SUSTAINED_30D":
                reasons.append("INCREMENTAL_REQUIRES_SUSTAINED_30D")
            if "LOCAL" not in scope and "GRADUAL" not in pattern:
                reasons.append("INCREMENTAL_REQUIRES_LOCAL_OR_GRADUAL")
            if not self._supports_incremental(request.algorithm_family):
                reasons.append("INCREMENTAL_UNSUPPORTED_FOR_ALGORITHM")

        if code == "full_retrain":
            if decay != "SEVERE" or "GLOBAL" not in scope:
                reasons.append("FULL_RETRAIN_REQUIRES_SEVERE_GLOBAL")
            if not request.manual_approval:
                reasons.append("FULL_RETRAIN_REQUIRES_MANUAL_APPROVAL")

        if code == "segment_weighted_retrain":
            evidence = request.segment_evidence or {}
            if not (
                evidence.get("segment_column")
                and evidence.get("affected_segments")
            ):
                reasons.append("SEGMENT_EVIDENCE_INSUFFICIENT")

        if code in SECOND_ROUND_STRATEGIES and request.business_round < 2:
            reasons.append("SECOND_ROUND_STRATEGY_REQUIRES_BUSINESS_ROUND_2")
        if code == "feature_selection_retrain":
            if not request.manual_approval:
                reasons.append("FEATURE_SELECTION_REQUIRES_MANUAL_APPROVAL")
            if not (
                request.failure_report_id
                and request.unstable_feature_codes
                and request.feature_evidence_source
            ):
                reasons.append("FEATURE_SELECTION_EVIDENCE_INSUFFICIENT")
        if code == "feature_reconstruction":
            if not request.manual_approval:
                reasons.append("FEATURE_RECONSTRUCTION_REQUIRES_MANUAL_APPROVAL")
            if not request.failure_report_id:
                reasons.append("FEATURE_RECONSTRUCTION_REQUIRES_FAILURE_ATTRIBUTION")
        if code == "regularized_retrain" and not request.failure_report_id:
            reasons.append("REGULARIZED_RETRAIN_REQUIRES_FAILURE_ATTRIBUTION")

        return reasons

    def _annotate_kg(
        self,
        l1: DecisionProposal,
        kg_consistency_status: str,
        *,
        repair_required: bool,
        reasons_extra: list[str],
        kg_candidate_codes: list[str] | None = None,
    ) -> DecisionProposal:
        """L1 提案保持不变（策略/动作/人工复核均以 L1 为准），仅附加 KG 一致性注解。"""
        return l1.model_copy(update={
            "decision_reasons": list(l1.decision_reasons) + reasons_extra,
            "kg_consistency_status": kg_consistency_status,
            "kg_repair_required": repair_required,
            "kg_candidate_codes": list(kg_candidate_codes or []),
        })

    def decide(self, request: DecisionInput) -> DecisionProposal:
        causes = sorted(request.root_causes, key=lambda item: item.score, reverse=True)
        primary = causes[0]
        primary_code = self._normalize_code(primary.root_cause_code)
        # A7 §4/§5: L1 读取结构化上下文推导有效根因码（KG 不能替 L1 选择）
        primary_code = self._derive_effective_root_code(primary_code, request)
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

        # 任务一统一入口规则：SHORT_TERM_7D → 继续观察；SEVERE → 人工复核
        decay = (request.decay_degree or "").upper()
        if (
            primary_code in {"FEATURE_DRIFT", "SEGMENT_DRIFT"}
            and decay == "SHORT_TERM_7D"
        ):
            return self._proposal(
                request,
                causes,
                RecommendedAction.CONTINUE_OBSERVATION,
                [],
                ConfidenceLevel.HIGH,
                ["SHORT_TERM_7D_OBSERVE_ONLY_NOT_A7"],
            )
        if decay == "SEVERE" and not request.manual_approval:
            return self._proposal(
                request,
                causes,
                RecommendedAction.MANUAL_REVIEW,
                [],
                ConfidenceLevel.MEDIUM,
                ["SEVERE_REQUIRES_MANUAL_REVIEW"],
                requires_manual_review=True,
            )

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

        # A7 §4: FEATURE_DRIFT + SUSTAINED_30D + LOCAL/GRADUAL → incremental_retrain。
        # 定稿条件含"算法支持增量训练"：仅 LightGBM/XGBoost 家族可选增量，
        # 其余家族回退 YAML 规则（recent_weighted_retrain），不得生成
        # Worker 必然失败的增量任务。
        if (
            primary_code == "FEATURE_DRIFT"
            and decay == "SUSTAINED_30D"
            and (
                "LOCAL" in (request.impact_scope or "").upper()
                or "GRADUAL" in (request.change_pattern or "").upper()
            )
        ):
            if self._supports_incremental(request.algorithm_family):
                strategies = ["incremental_retrain"]
                reasons.append(
                    "STRUCTURED_CONTEXT:SUSTAINED_30D_LOCAL_GRADUAL_INCREMENTAL"
                )
            else:
                # 算法家族不支持增量 → 保持 L1 YAML 默认（recent_weighted_retrain）
                reasons.append(
                    "STRUCTURED_CONTEXT:INCREMENTAL_UNSUPPORTED_FOR_ALGORITHM_"
                    "FALLBACK_L1_RULE"
                )
        # A7 §5: 第二轮 + 特征脆弱 + 人工批准 → feature_selection_retrain。
        # L1 是最终权威，必须由真实归因证据约束：三项缺一不可，
        # 证据不足时回退 YAML 规则（regularized_retrain / feature_reconstruction），
        # 不能凭轮次和批准单独进入特征筛选。
        if (
            primary_code == "FEATURE_FRAGILITY"
            and request.business_round >= 2
            and request.manual_approval
        ):
            if (
                request.failure_report_id
                and request.unstable_feature_codes
                and request.feature_evidence_source
            ):
                strategies = ["feature_selection_retrain"]
                reasons.append(
                    "STRUCTURED_CONTEXT:ROUND2_ATTRIBUTION_FEATURE_SELECTION"
                )
            else:
                reasons.append(
                    "FEATURE_SELECTION_EVIDENCE_INSUFFICIENT_FALLBACK_L1_RULE"
                )

        # A7 §4: segment_weighted_retrain 必须有完整的冻结客群证据，
        # 否则不创建必然失败的 TrainingJob —— 回退其他策略或人工复核
        if "segment_weighted_retrain" in strategies:
            evidence = request.segment_evidence or {}
            segment_complete = bool(
                evidence.get("segment_column")
                and evidence.get("affected_segments")
            )
            if not segment_complete:
                reasons.append("SEGMENT_EVIDENCE_INSUFFICIENT_FALLBACK")
                strategies = [
                    s for s in strategies if s != "segment_weighted_retrain"
                ]
                if not strategies:
                    return self._proposal(
                        request,
                        causes,
                        RecommendedAction.MANUAL_REVIEW,
                        [],
                        ConfidenceLevel.MEDIUM,
                        reasons,
                        requires_manual_review=True,
                    )

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
            parameters = dict(definition.parameters)
            # A7 §4: segment_weighted_retrain 的冻结客群定义来自诊断证据，
            # 经 Proposal → TrainingPlan.sample_weight_policy → Worker 消费
            if (
                code == "segment_weighted_retrain"
                and request.segment_evidence
            ):
                parameters["sample_weight_policy"] = request.segment_evidence
            selections.append(
                StrategySelection(
                    strategy_code=code,
                    parameters=parameters,
                    rationale=definition.description,
                    primary_training_mode=definition.primary_training_mode,
                )
            )
        return DecisionProposal(
            proposal_id=str(uuid4()),
            diagnosis_run_id=request.diagnosis_run_id,
            # 资格评估端点据此加载特征级 PSI（服务端信任源）
            monitoring_run_id=request.monitoring_run_id,
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
            # A7 §6.3: final_strategy_code = l1_strategy_code（L1 是最终权威）
            final_strategy_code=(
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

    # ═══════════════════════════════════════════════════════════════
    #  严格 A7 入口（A7DecisionEnvelope → L1StrategyDecision → Proposal）
    #  A7 定稿 §2-§4：SHORT_TERM_7D 不自动训练；SEVERE 强制人工复核；
    #  FEATURE_DRIFT + SUSTAINED_30D + LOCAL/GRADUAL → incremental_retrain；
    #  full_retrain 只出现在 SEVERE+GLOBAL 且人工批准语境（本入口不自动选择）。
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _kg_strategy_codes(envelope: "A7DecisionEnvelope") -> list[str]:
        """从 A7 决策信封抽取 KG 策略候选码（去重、去空）。"""
        codes: list[str] = []
        for item in envelope.kg_strategy_candidates:
            code = str(item.get("strategy_code") or "").strip()
            if code and code not in codes:
                codes.append(code)
        return codes

    @staticmethod
    def _blocked_l1(
        envelope: "A7DecisionEnvelope",
        reasons: list[str],
    ) -> "L1StrategyDecision":
        """构造阻断态 L1 决策（不自动执行，转人工复核）。"""
        kg_codes = RepairDecisionService._kg_strategy_codes(envelope)
        return L1StrategyDecision(
            selection_status="BLOCKED",
            same_algorithm_family=True,
            algorithm_family=envelope.algorithm_family,
            kg_strategy_candidates=envelope.kg_strategy_candidates,
            strategy_source="L1_FALLBACK" if not kg_codes else "L1_OVERRIDE",
            kg_consistency_status=(
                "KG_STRATEGY_MISSING" if not kg_codes else "KG_STRATEGY_MISMATCH"
            ),
            kg_repair_required=not kg_codes,
            selection_rule_version=envelope.rule_versions["l1_matrix_version"],
            blocking_reasons=reasons,
        )

    @staticmethod
    def _segment_gate_reasons(envelope: "A7DecisionEnvelope") -> list[str]:
        """分段漂移的样本门槛校验（样本/坏样本/占比/退化证据/主分段）。"""
        reasons: list[str] = []
        primary_seen = False
        for segment in envelope.affected_segments:
            primary_seen = primary_seen or segment.primary
            if segment.current_sample_count < 200:
                reasons.append(f"SEGMENT_SAMPLE_INSUFFICIENT:{segment.segment_id}")
            if segment.current_bad_count < 50:
                reasons.append(f"SEGMENT_BAD_SAMPLE_INSUFFICIENT:{segment.segment_id}")
            if segment.current_share < 0.05:
                reasons.append(f"SEGMENT_SHARE_INSUFFICIENT:{segment.segment_id}")
            if not segment.performance_affected:
                reasons.append(f"SEGMENT_PERFORMANCE_NOT_AFFECTED:{segment.segment_id}")
        if envelope.affected_segments and not primary_seen:
            reasons.append("PRIMARY_SEGMENT_MISSING")
        return reasons

    def _envelope_candidate_blocking_reasons(
        self,
        envelope: "A7DecisionEnvelope",
        strategy: str,
    ) -> list[str]:
        reasons: list[str] = []
        if strategy not in self.config.strategies.strategies:
            return [f"STRATEGY_NOT_IN_LOCAL_EXECUTION_CATALOG:{strategy}"]

        business_round = int(envelope.trigger_context.get("business_round", 1) or 1)
        if strategy == "incremental_retrain":
            if envelope.primary_root_cause.root_cause_code != "FEATURE_DRIFT":
                reasons.append("INCREMENTAL_REQUIRES_FEATURE_DRIFT")
            if envelope.decay_degree != "SUSTAINED_30D":
                reasons.append("INCREMENTAL_REQUIRES_SUSTAINED_30D")
            if (
                envelope.impact_scope != "LOCAL"
                and envelope.change_pattern != "GRADUAL"
            ):
                reasons.append("INCREMENTAL_REQUIRES_LOCAL_OR_GRADUAL")
            if not self._supports_incremental(envelope.algorithm_family):
                reasons.append("INCREMENTAL_UNSUPPORTED_FOR_ALGORITHM")

        if strategy == "full_retrain":
            if envelope.decay_degree != "SEVERE" or envelope.impact_scope != "GLOBAL":
                reasons.append("FULL_RETRAIN_REQUIRES_SEVERE_GLOBAL")
            if (
                envelope.authorization.authorization_type != "MANUAL_REVIEW"
                or not envelope.authorization.approved
            ):
                reasons.append("FULL_RETRAIN_REQUIRES_MANUAL_APPROVAL")

        if strategy == "segment_weighted_retrain":
            segment_reasons = self._segment_gate_reasons(envelope)
            if segment_reasons:
                reasons.extend(segment_reasons)

        if strategy in SECOND_ROUND_STRATEGIES and business_round < 2:
            reasons.append("SECOND_ROUND_STRATEGY_REQUIRES_BUSINESS_ROUND_2")
        if strategy in {"feature_selection_retrain", "feature_reconstruction"}:
            if (
                envelope.authorization.authorization_type != "MANUAL_REVIEW"
                or not envelope.authorization.approved
            ):
                reasons.append(f"{strategy.upper()}_REQUIRES_MANUAL_APPROVAL")

        return reasons

    @staticmethod
    def _envelope_strategy_fields(
        envelope: "A7DecisionEnvelope",
        strategy: str,
    ) -> tuple[str, str, list[str], bool, dict]:
        training_data_mode = "WINDOW_BASED"
        training_windows = ["W2", "W3_TRAIN_SPLIT"]
        sample_weight_required = False
        sample_weight_policy: dict = {}

        execution_mode = "FRESH_REFIT_ON_RECENT_WEIGHTED_WINDOW"
        if strategy == "incremental_retrain":
            execution_mode = "NATIVE_CONTINUATION"
        elif strategy == "sliding_window_retrain":
            execution_mode = "FRESH_REFIT_ON_SLIDING_WINDOW"
        elif strategy == "segment_weighted_retrain":
            execution_mode = "SEGMENT_WEIGHTED_FRESH_REFIT"
            sample_weight_required = True
            sample_weight_policy = {
                "type": "SEGMENT_SHARE_RATIO",
                "min_weight": 0.5,
                "max_weight": 3.0,
                "normalize_mean_to_one": True,
                "multiple_match_policy": "MAX",
                "affected_segments": [
                    segment.model_dump(mode="json")
                    for segment in envelope.affected_segments
                ],
                "threshold_status": "PENDING_EMPIRICAL_CALIBRATION",
            }
        elif strategy == "full_retrain":
            execution_mode = "FULL_RETRAIN_ON_ELIGIBLE_HISTORY"
        elif strategy == "feature_selection_retrain":
            execution_mode = "FEATURE_SELECTION_THEN_REFIT"
        elif strategy == "feature_reconstruction":
            execution_mode = "FEATURE_RECONSTRUCTION_THEN_REFIT"
        elif strategy == "regularized_retrain":
            execution_mode = "HYPERPARAMETER_TUNING_THEN_REFIT"

        if envelope.change_pattern == "SUDDEN":
            training_windows = ["W3_TRAIN_SPLIT"]
            training_data_mode = "POST_CHANGE_ONLY"

        return (
            execution_mode,
            training_data_mode,
            training_windows,
            sample_weight_required,
            sample_weight_policy,
        )

    def select_a7_strategy(
        self,
        envelope: "A7DecisionEnvelope",
        *,
        available_algorithm_families: set[str] | None = None,
    ) -> "L1StrategyDecision":
        """L1 确定性策略选择（严格 A7 信封入口；KG 仅作参考建议）。

        与 decide()/decide_with_kg() 共享同一 L1 哲学：
        - SHORT_TERM_7D → 不自动进入 A7（定稿任务一统一入口规则）
        - SEVERE → 强制人工复核（定稿 §4），不自动训练
        - 算法家族无适配器 → 阻断
        - 分段漂移必须过分段样本门槛
        """
        available = available_algorithm_families or {
            "LogisticRegression",
            "RandomForest",
            "LightGBM",
        }
        if envelope.algorithm_family not in available:
            return self._blocked_l1(
                envelope,
                [f"ALGORITHM_ADAPTER_NOT_IMPLEMENTED:{envelope.algorithm_family}"],
            )

        # 根因门槛（与自然链路同源）：主根因置信度不足 → 阻断
        min_score = self.config.iteration.root_cause_gate.min_primary_score
        if envelope.primary_root_cause.confidence < min_score:
            return self._blocked_l1(
                envelope,
                ["PRIMARY_ROOT_CAUSE_SCORE_BELOW_THRESHOLD"],
            )

        # A7 定稿任务一统一入口规则：SHORT_TERM_7D → 观察，不自动训练
        if envelope.decay_degree == "SHORT_TERM_7D":
            return self._blocked_l1(
                envelope,
                ["SHORT_TERM_7D_OBSERVE_ONLY_NOT_A7"],
            )
        # A7 定稿 §4：SEVERE → 完成诊断后强制人工复核，不自动训练
        if envelope.decay_degree == "SEVERE" and not envelope.authorization.approved:
            return self._blocked_l1(
                envelope,
                ["SEVERE_REQUIRES_MANUAL_REVIEW"],
            )

        cause = envelope.primary_root_cause.root_cause_code
        if cause in {"SEGMENT_DRIFT", "CONCEPT_DRIFT"} and (
            cause == "SEGMENT_DRIFT" or envelope.impact_scope == "LOCAL"
        ):
            segment_reasons = self._segment_gate_reasons(envelope)
            if segment_reasons:
                return self._blocked_l1(envelope, segment_reasons)

        kg_codes = self._kg_strategy_codes(envelope)
        if not kg_codes:
            return self._blocked_l1(envelope, ["KG_STRATEGY_MISSING"])

        selected_strategy: str | None = None
        blocked_reasons: list[str] = []
        for candidate_code in kg_codes:
            reasons = self._envelope_candidate_blocking_reasons(
                envelope, candidate_code
            )
            if reasons:
                blocked_reasons.extend(
                    f"KG_CANDIDATE_BLOCKED:{candidate_code}:{reason}"
                    for reason in reasons
                )
                continue
            selected_strategy = candidate_code
            break

        if selected_strategy is None:
            return self._blocked_l1(
                envelope,
                ["KG_CANDIDATES_BLOCKED_BY_L1_GUARDRAILS"] + blocked_reasons,
            )

        (
            execution_mode,
            training_data_mode,
            training_windows,
            sample_weight_required,
            sample_weight_policy,
        ) = self._envelope_strategy_fields(envelope, selected_strategy)
        consistency = (
            "KG_SELECTED_L1_VALIDATED"
            if selected_strategy == kg_codes[0]
            else "KG_TOP_BLOCKED_L1_SELECTED_NEXT"
        )
        return L1StrategyDecision(
            selection_status="SELECTED",
            primary_strategy=selected_strategy,
            execution_mode=execution_mode,
            training_data_mode=training_data_mode,
            training_window_ids=training_windows,
            validation_window_ids=["W3_VALIDATION_SPLIT"],
            oot_window_ids=["W4"],
            same_algorithm_family=True,
            algorithm_family=envelope.algorithm_family,
            sample_weight_required=sample_weight_required,
            sample_weight_policy=sample_weight_policy,
            feature_reconstruction_required=(
                selected_strategy == "feature_reconstruction"
            ),
            feature_schema_change_allowed=False,
            kg_strategy_candidates=envelope.kg_strategy_candidates,
            strategy_source="KG_WITH_L1_GUARDRAILS",
            kg_consistency_status=consistency,
            kg_repair_required=False,
            selection_reason_codes=[
                f"KG_STRATEGY:{selected_strategy}",
                "L1_GUARDRAILS_PASSED",
            ] + blocked_reasons,
            selection_rule_version=envelope.rule_versions["l1_matrix_version"],
        )

        training_windows: list[str]
        validation_windows = ["W3_VALIDATION_SPLIT"]
        strategy: str
        execution_mode: str
        training_data_mode = "WINDOW_BASED"
        sample_weight_required = False
        sample_weight_policy: dict = {}

        if cause == "FEATURE_DRIFT":
            # A7 定稿 §4：SUSTAINED_30D + LOCAL/GRADUAL → incremental_retrain
            # （仅增量支持的算法族；其余族回退滑动窗口重训）
            scope = (envelope.impact_scope or "").upper()
            pattern = (envelope.change_pattern or "").upper()
            if (
                envelope.decay_degree == "SUSTAINED_30D"
                and ("LOCAL" in scope or "GRADUAL" in pattern)
                and envelope.algorithm_family == "LightGBM"
            ):
                strategy = "incremental_retrain"
                training_windows = ["W2", "W3_TRAIN_SPLIT"]
                execution_mode = "NATIVE_CONTINUATION"
            else:
                strategy = "sliding_window_retrain"
                training_windows = ["W2", "W3_TRAIN_SPLIT"]
                execution_mode = "FRESH_REFIT_ON_SLIDING_WINDOW"
        elif cause == "SEGMENT_DRIFT":
            strategy = "segment_weighted_retrain"
            training_windows = ["W2", "W3_TRAIN_SPLIT"]
            execution_mode = "SEGMENT_WEIGHTED_FRESH_REFIT"
            sample_weight_required = True
            sample_weight_policy = {
                "type": "SEGMENT_SHARE_RATIO",
                "min_weight": 0.5,
                "max_weight": 3.0,
                "normalize_mean_to_one": True,
                "multiple_match_policy": "MAX",
                "affected_segments": [
                    segment.model_dump(mode="json")
                    for segment in envelope.affected_segments
                ],
                "threshold_status": "PENDING_EMPIRICAL_CALIBRATION",
            }
        elif cause == "CONCEPT_DRIFT":
            # 定稿 §11.2：CONCEPT_DRIFT 总类不直接建策略边；信封已强制
            # 声明 impact_scope/change_pattern，此处按其细分语义映射，
            # 绝不自动选择 full_retrain（full_retrain 必须人工批准）。
            is_local = envelope.impact_scope == "LOCAL"
            is_sudden = envelope.change_pattern == "SUDDEN"
            if is_local:
                strategy = "segment_weighted_retrain"
                execution_mode = "SEGMENT_WEIGHTED_FRESH_REFIT"
                sample_weight_required = True
                sample_weight_policy = {
                    "type": "SEGMENT_SHARE_RATIO",
                    "min_weight": 0.5,
                    "max_weight": 3.0,
                    "normalize_mean_to_one": True,
                    "multiple_match_policy": "MAX",
                    "affected_segments": [
                        segment.model_dump(mode="json")
                        for segment in envelope.affected_segments
                    ],
                    "threshold_status": "PENDING_EMPIRICAL_CALIBRATION",
                }
            else:
                strategy = "sliding_window_retrain"
                execution_mode = "FRESH_REFIT_ON_SLIDING_WINDOW"
            if is_sudden:
                training_windows = ["W3_TRAIN_SPLIT"]
                training_data_mode = "POST_CHANGE_ONLY"
            else:
                training_windows = ["W2", "W3_TRAIN_SPLIT"]
        else:  # pragma: no cover - A7DecisionEnvelope rejects this first
            return self._blocked_l1(envelope, ["L1_RULE_NOT_FOUND"])

        # L1 与 KG 的一致性比对（KG 仅咨询，L1 恒为最终权威）
        kg_codes = self._kg_strategy_codes(envelope)
        if not kg_codes:
            source = "L1_FALLBACK"
            consistency = "KG_STRATEGY_MISSING"
            kg_repair = True
        elif strategy in kg_codes:
            source = "KG_AND_L1"
            consistency = "CONSISTENT"
            kg_repair = False
        else:
            source = "L1_OVERRIDE"
            consistency = "KG_STRATEGY_MISMATCH"
            kg_repair = True

        return L1StrategyDecision(
            selection_status="SELECTED",
            primary_strategy=strategy,
            execution_mode=execution_mode,
            training_data_mode=training_data_mode,
            training_window_ids=training_windows,
            validation_window_ids=validation_windows,
            oot_window_ids=["W4"],
            same_algorithm_family=True,
            algorithm_family=envelope.algorithm_family,
            sample_weight_required=sample_weight_required,
            sample_weight_policy=sample_weight_policy,
            feature_reconstruction_required=False,
            feature_schema_change_allowed=False,
            kg_strategy_candidates=envelope.kg_strategy_candidates,
            strategy_source=source,
            kg_consistency_status=consistency,
            kg_repair_required=kg_repair,
            selection_reason_codes=[
                f"{cause}_{envelope.decay_degree}",
                (
                    f"{envelope.impact_scope}_{envelope.change_pattern}"
                    if cause == "CONCEPT_DRIFT"
                    else cause
                ),
            ],
            selection_rule_version=envelope.rule_versions["l1_matrix_version"],
        )

    # L1StrategyDecision 三态 → 本地 DecisionProposal 六档一致性命名映射
    _KG_CONSISTENCY_MAP = {
        "KG_SELECTED_L1_VALIDATED": "KG_SELECTED_L1_VALIDATED",
        "KG_TOP_BLOCKED_L1_SELECTED_NEXT": "KG_TOP_BLOCKED_L1_SELECTED_NEXT",
        "KG_CANDIDATES_BLOCKED_BY_L1": "KG_CANDIDATES_BLOCKED_BY_L1",
        "CONSISTENT": "KG_AGREES",
        "KG_STRATEGY_MISMATCH": "KG_L1_CANDIDATE_MISSING",
        "KG_STRATEGY_MISSING": "KG_NO_CANDIDATES",
    }

    def propose_a7(
        self,
        envelope: "A7DecisionEnvelope",
        *,
        available_algorithm_families: set[str] | None = None,
    ) -> tuple[DecisionProposal, "L1StrategyDecision"]:
        """基于严格信封与 L1 结果生成 A7 提案。

        SELECTED → MODEL_ITERATION 提案（信封授权已批准，status=APPROVED）；
        BLOCKED → MANUAL_REVIEW 提案（PENDING_REVIEW，requires_manual_review）。
        """
        l1 = self.select_a7_strategy(
            envelope,
            available_algorithm_families=available_algorithm_families,
        )
        selected = l1.selection_status == "SELECTED"
        selections = (
            [
                StrategySelection(
                    strategy_code=l1.primary_strategy or "",
                    parameters={
                        "execution_mode": l1.execution_mode,
                        "training_data_mode": l1.training_data_mode,
                        "training_window_ids": l1.training_window_ids,
                        "validation_window_ids": l1.validation_window_ids,
                        "oot_window_ids": l1.oot_window_ids,
                        "algorithm_family": l1.algorithm_family,
                        "sample_weight_required": l1.sample_weight_required,
                        "sample_weight_policy": l1.sample_weight_policy,
                    },
                    # §7 传递链：训练模式来自 StrategyDefinition 正式字段
                    primary_training_mode=(
                        self.config.strategies.strategies[
                            l1.primary_strategy
                        ].primary_training_mode
                        if selected and l1.primary_strategy in self.config.strategies.strategies
                        else TrainingMode.FULL_RETRAIN
                    ),
                    rationale=";".join(l1.selection_reason_codes),
                )
            ]
            if selected
            else []
        )
        proposal = DecisionProposal(
            proposal_id=str(uuid4()),
            diagnosis_run_id=envelope.diagnosis_run_id,
            monitoring_run_id=envelope.monitoring_run_id,
            lifecycle_run_id=envelope.lifecycle_run_id,
            model_id=envelope.model_id,
            champion_version=envelope.champion_version,
            primary_root_cause_code=envelope.primary_root_cause.root_cause_code,
            primary_root_cause_score=envelope.primary_root_cause.confidence,
            # 严格信封已强制 CONFIRMED 根因 + evidence_refs 非空 + 授权批准，
            # 证据链完整性由合同层保证
            evidence_coverage=1.0,
            action=(
                RecommendedAction.MODEL_ITERATION
                if selected
                else RecommendedAction.MANUAL_REVIEW
            ),
            need_iteration=selected,
            strategies=selections,
            selected_strategy_code=l1.primary_strategy,
            target_metric_codes=[
                str(code)
                for code in envelope.trigger_context.get("trigger_metric_codes", [])
            ],
            proposed_window_policy=self.config.iteration.training_window_policy,
            confidence=(ConfidenceLevel.HIGH if selected else ConfidenceLevel.LOW),
            decision_reasons=(
                l1.selection_reason_codes if selected else l1.blocking_reasons
            ),
            status=(
                ProposalStatus.APPROVED if selected else ProposalStatus.PENDING_REVIEW
            ),
            executable=False,
            requires_manual_review=not selected,
            rule_version=envelope.rule_versions["agent_rule_version"],
            rule_versions=envelope.rule_versions,
            selection_status=l1.selection_status,
            primary_strategy=l1.primary_strategy,
            execution_mode=l1.execution_mode,
            strategy_source=l1.strategy_source,
            kg_consistency_status=self._KG_CONSISTENCY_MAP.get(
                l1.kg_consistency_status, l1.kg_consistency_status
            ),
            kg_repair_required=l1.kg_repair_required,
            kg_candidate_codes=self._kg_strategy_codes(envelope),
            final_strategy_code=l1.primary_strategy,
            selection_reason_codes=l1.selection_reason_codes,
            blocking_reasons=l1.blocking_reasons,
            event_id=envelope.event_id,
            agent_decision_id=envelope.agent_decision_id,
            decision_source=envelope.decision_source,
            root_cause_status=envelope.primary_root_cause.candidate_status,
            decay_degree=envelope.decay_degree,
            model_task_type=envelope.model_task_type,
            algorithm_family=envelope.algorithm_family,
            champion_artifact_checksum=envelope.champion_artifact_checksum,
            impact_scope=envelope.impact_scope,
            change_pattern=envelope.change_pattern,
            change_point=envelope.change_point,
            ordered_evidence_window_ids=envelope.ordered_evidence_window_ids,
            affected_segments=[
                item.model_dump(mode="json") for item in envelope.affected_segments
            ],
            sample_weight_required=l1.sample_weight_required,
            sample_weight_policy=l1.sample_weight_policy,
            training_data_mode=l1.training_data_mode,
            training_window_ids=l1.training_window_ids,
            validation_window_ids=l1.validation_window_ids,
            oot_window_ids=l1.oot_window_ids,
            authorization_type=envelope.authorization.authorization_type,
            authorization_id=envelope.authorization.authorization_id,
        )
        return proposal, l1
