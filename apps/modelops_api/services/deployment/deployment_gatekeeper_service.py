"""Deployment Gatekeeper service.

The Gatekeeper is the final deployment decision maker. KG provides risk and
strategy evidence, while health checks provide hard metric gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from packages.models.deployment.deployment_context import DeploymentContext


@dataclass
class GatekeeperDecision:
    """Final deployment decision produced by the Gatekeeper."""

    decision: str
    selected_strategy_code: str | None = None
    decision_reasons: list[str] = field(default_factory=list)
    gatekeeper_rule_refs: list[str] = field(default_factory=list)
    requires_manual_review: bool = False
    rollback_target: str | None = None
    next_stage: str | None = None


class DeploymentGatekeeperService:
    """Combine health gates, KG risk context, and release policy."""

    KG_ACTION_MAP: dict[str, str] = {
        "ROLLBACK": "ROLLBACK",
        "ROLLBACK_TO_STABLE": "ROLLBACK",
        "PAUSE_CANARY": "HOLD",
        "REDUCE_TRAFFIC": "HOLD",
        "HOLD": "HOLD",
        "MANUAL_REVIEW": "HOLD",
        "ADVANCE_STAGE": "ADVANCE_STAGE",
    }
    CRITICAL_STAGES: set[str] = {"CANARY_20", "CANARY_50", "PRODUCTION"}

    def decide(
        self,
        stage: str,
        health_result: dict,
        deployment_context: DeploymentContext | None = None,
        *,
        challenger_qualified: bool = True,
        current_traffic_ratio: float = 0.0,
    ) -> GatekeeperDecision:
        reasons: list[str] = []
        rule_refs: list[str] = ["DEPLOYMENT_GATEKEEPER_V1"]

        if not challenger_qualified:
            return GatekeeperDecision(
                decision="ABORT_DEPLOYMENT",
                decision_reasons=["challenger_not_qualified"],
                gatekeeper_rule_refs=rule_refs,
            )

        health_passed = health_result.get("passed", True)
        rollback_recommended = health_result.get("rollback_recommended", False)
        failures = health_result.get("failures", [])
        is_critical = stage in self.CRITICAL_STAGES

        if rollback_recommended:
            reasons.append("health_check_recommends_rollback")
            reasons.extend(health_result.get("rollback_reasons", []))
            return GatekeeperDecision(
                decision="ROLLBACK",
                decision_reasons=reasons,
                gatekeeper_rule_refs=rule_refs,
            )

        if not health_passed and is_critical:
            reasons.append(f"health_failed_in_critical_stage:{stage}")
            reasons.extend(failures)
            return GatekeeperDecision(
                decision="ROLLBACK",
                decision_reasons=reasons,
                gatekeeper_rule_refs=rule_refs,
            )

        if not health_passed:
            reasons.append(f"health_failed_in_non_critical_stage:{stage}")
            reasons.extend(failures)
            return GatekeeperDecision(
                decision="HOLD",
                decision_reasons=reasons,
                gatekeeper_rule_refs=rule_refs,
            )

        ctx = deployment_context
        if ctx is not None:
            if ctx.retrieval_degraded and is_critical:
                return GatekeeperDecision(
                    decision="HOLD",
                    decision_reasons=["kg_degraded_in_critical_stage"],
                    gatekeeper_rule_refs=rule_refs + ctx.gatekeeper_rule_refs,
                    requires_manual_review=True,
                )

            if ctx.retrieval_degraded:
                reasons.append("kg_degraded_but_non_critical")

            if not ctx.retrieval_degraded and ctx.deployment_risks:
                best_strategy = self._pick_best_strategy(ctx, stage)
                if best_strategy:
                    action = self.KG_ACTION_MAP.get(str(best_strategy.action_type).upper(), "")
                    rule_refs.append(f"KG_STRATEGY:{best_strategy.strategy_code}")

                    if (
                        action == "ROLLBACK"
                        and best_strategy.effective_weight_snapshot >= 0.6
                        and best_strategy.support_case_count >= 3
                    ):
                        return GatekeeperDecision(
                            decision="ROLLBACK",
                            selected_strategy_code=best_strategy.strategy_code,
                            decision_reasons=[
                                f"kg_high_confidence_rollback:{best_strategy.strategy_code}",
                                f"kg_weight:{best_strategy.effective_weight_snapshot:.3f}",
                            ],
                            gatekeeper_rule_refs=rule_refs,
                        )

                    if action == "HOLD":
                        return GatekeeperDecision(
                            decision="HOLD",
                            selected_strategy_code=best_strategy.strategy_code,
                            decision_reasons=[f"kg_strategy_suggests_hold:{best_strategy.strategy_code}"],
                            gatekeeper_rule_refs=rule_refs,
                        )

                    if best_strategy.effective_weight_snapshot < 0.4:
                        reasons.append(f"kg_low_confidence_strategy_ignored:{best_strategy.strategy_code}")

        if stage == "PRODUCTION" and health_passed:
            reasons.append("production_stage_all_clear")
            return GatekeeperDecision(
                decision="PROMOTE",
                decision_reasons=reasons,
                gatekeeper_rule_refs=rule_refs,
            )

        reasons.append("health_passed_advancing")
        return GatekeeperDecision(
            decision="ADVANCE_STAGE",
            decision_reasons=reasons,
            gatekeeper_rule_refs=rule_refs,
        )

    @staticmethod
    def _pick_best_strategy(ctx: DeploymentContext, stage: str):
        best = None
        best_weight = -1.0
        for risk in ctx.deployment_risks:
            for strategy in risk.strategy_candidates:
                allowed_stages = [str(item) for item in getattr(strategy, "allowed_stages", [])]
                if allowed_stages and stage not in allowed_stages:
                    continue
                weight = strategy.effective_weight_snapshot
                if weight > best_weight:
                    best_weight = weight
                    best = strategy
        return best
