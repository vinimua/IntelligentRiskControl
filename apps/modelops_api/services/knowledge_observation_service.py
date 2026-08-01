"""KnowledgeObservationService — 生命周期结束后自动写入 KG 观测。

职责：
- EventCloseNode → 调用 record_lifecycle_observations(state)
- 根据任务二/三/四的业务结果，写入 SUPPORT/AGAINST/NEUTRAL 观测
"""
from __future__ import annotations

import structlog

from packages.models.knowledge.kg_entity import KgRelationObservation

logger = structlog.get_logger(__name__)


class KnowledgeObservationService:
    """从 lifecycle State 提取观测并写入 kg_relation_observations。"""

    @staticmethod
    def build_observations(state: dict) -> list[KgRelationObservation]:
        """根据 State 构建观测列表。"""
        lifecycle_run_id = state.get("lifecycle_run_id", "")
        observations: list[KgRelationObservation] = []

        # ── 任务二：诊断结果 → INDICATES 关系 ──
        diagnosis_run_id = state.get("diagnosis_run_id")
        primary_code = state.get("primary_root_cause_code")
        primary_score = state.get("primary_root_cause_score")

        if diagnosis_run_id and primary_code and primary_score:
            # 诊断置信度映射到 evidence_score
            score = min(1.0, max(0.0, (primary_score or 0) / 1.0))
            direction = "SUPPORT" if (primary_score or 0) >= 0.75 else "NEUTRAL"

            # 构造关系 key：Alert → RootCause
            alert_code = _infer_alert_code(state) or "AUC_DROP"
            relation_key = f"{alert_code}|INDICATES|{primary_code}"

            observations.append(
                KgRelationObservation(
                    relation_key=relation_key,
                    source_domain="DIAGNOSIS",
                    source_record_id=f"diag:{diagnosis_run_id}",
                    lifecycle_run_id=lifecycle_run_id,
                    direction=direction,
                    evidence_score=score,
                    quality_weight=1.0,
                    weighted_strength=score,
                    data_track="NATURAL",
                    evidence_detail={
                        "primary_root_cause_code": primary_code,
                        "primary_root_cause_score": primary_score,
                        "recommended_action": state.get("recommended_action"),
                    },
                )
            )

        # ── 任务三：策略选择 → RECOMMENDS 关系 ──
        proposal_id = state.get("decision_proposal_id")
        selected_strategy = state.get("selected_strategy_code")
        decision_reasons = state.get("decision_reasons") or []

        if proposal_id and selected_strategy and primary_code:
            # 如果 KG 推荐被采纳 → SUPPORT；降级到 YAML → NEUTRAL；低置信度 → AGAINST
            kg_used = any(r.startswith("KG_STRATEGY:") for r in decision_reasons)
            low_cases = any(r.startswith("SUPPORT_CASES:") for r in decision_reasons)
            requires_review = state.get("requires_manual_review", False)

            if kg_used and not requires_review:
                direction = "SUPPORT"
                score = 0.85
            elif kg_used and requires_review:
                direction = "NEUTRAL"
                score = 0.50
            else:
                direction = "AGAINST"
                score = 0.20

            relation_key = f"{primary_code}|RECOMMENDS|{selected_strategy}"

            observations.append(
                KgRelationObservation(
                    relation_key=relation_key,
                    source_domain="ITERATION",
                    source_record_id=f"proposal:{proposal_id}",
                    lifecycle_run_id=lifecycle_run_id,
                    direction=direction,
                    evidence_score=score,
                    quality_weight=1.0,
                    weighted_strength=score,
                    data_track="NATURAL",
                    evidence_detail={
                        "selected_strategy_code": selected_strategy,
                        "kg_used": kg_used,
                        "decision_reasons": decision_reasons[:5],
                    },
                )
            )

            # MITIGATES 反边：策略 → 根因
            mitigates_key = f"{selected_strategy}|MITIGATES|{primary_code}"
            observations.append(
                KgRelationObservation(
                    relation_key=mitigates_key,
                    source_domain="ITERATION",
                    source_record_id=f"mitigates:{proposal_id}",
                    lifecycle_run_id=lifecycle_run_id,
                    direction=direction,  # 同向
                    evidence_score=score,
                    quality_weight=0.8,  # 反边权重略低
                    weighted_strength=score * 0.8,
                    data_track="NATURAL",
                    evidence_detail={
                        "selected_strategy_code": selected_strategy,
                        "primary_root_cause_code": primary_code,
                    },
                )
            )

        # ── 任务四：部署结果 → RECOMMENDS/MITIGATES 强化 ──
        deployment_id = state.get("deployment_id")
        challenger_qualified = state.get("challenger_qualified")
        deployment_decision = state.get("deployment_decision")

        if deployment_id and primary_code and selected_strategy:
            if deployment_decision == "PROMOTE":
                direction = "SUPPORT"
                score = 0.95
            elif deployment_decision == "ROLLBACK":
                direction = "AGAINST"
                score = 0.15
            else:
                direction = "NEUTRAL"
                score = 0.50

            # 策略→根因 的 MITIGATES 关系由部署结果强化
            mitigates_key = f"{selected_strategy}|MITIGATES|{primary_code}"
            observations.append(
                KgRelationObservation(
                    relation_key=mitigates_key,
                    source_domain="DEPLOY",
                    source_record_id=f"deploy:{deployment_id}",
                    lifecycle_run_id=lifecycle_run_id,
                    direction=direction,
                    evidence_score=score,
                    quality_weight=1.0,
                    weighted_strength=score,
                    data_track="NATURAL",
                    evidence_detail={
                        "deployment_decision": deployment_decision,
                        "challenger_qualified": challenger_qualified,
                    },
                )
            )

        logger.info(
            "observations_built",
            lifecycle_run_id=lifecycle_run_id,
            count=len(observations),
        )
        return observations


def _infer_alert_code(state: dict) -> str | None:
    """从 State 推断触发诊断的告警代码。"""
    severity = state.get("max_alert_severity", "")
    primary = state.get("primary_root_cause_code", "")

    if severity in ("HIGH", "CRITICAL"):
        return "HIGH_FEATURE_PSI"
    if primary and "DRIFT" in str(primary).upper():
        return "HIGH_FEATURE_PSI"
    return "AUC_DROP"
