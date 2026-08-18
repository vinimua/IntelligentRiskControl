"""KnowledgeObservationService — 生命周期结束后自动写入 KG 观测。

职责：
- EventCloseNode → 调用 record_lifecycle_observations(state)
- 观测轨道纪律（A7 §5）：
  * 诊断置信度 → AUDIT 轨道（不是真实执行结果，不参与校准）
  * 策略被选择 → 不写任何观测（L1 与 KG 不一致也不是 AGAINST）
  * 只有部署结果（W4 FINAL-OOT 完成、生命周期结果冻结）→ NATURAL 轨道
"""
from __future__ import annotations

import structlog

from packages.models.common.enums import DataTrack
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

            # 构造关系 key：Alert → RootCause（逐条告警写入 INDICATES 观测）
            alert_codes = _infer_alert_codes(state)
            if not alert_codes:
                # 无真实告警码 → 跳过 INDICATES 观测，不污染 AUC_DROP 权重
                alert_codes = []

            for alert_code in alert_codes:
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
                        # 诊断置信度不是真实执行结果 → 纯审计轨道，不参与校准
                        data_track=DataTrack.AUDIT,
                        evidence_detail={
                            "primary_root_cause_code": primary_code,
                            "primary_root_cause_score": primary_score,
                            "recommended_action": state.get("recommended_action"),
                        },
                    ),
                )

        # ── 任务三：策略选择不产生校准观测 ──
        # A7 §5：策略被选择不是真实执行结果，L1 与 KG 不一致也不是 AGAINST。
        # NATURAL 校准证据只能来自：策略真实执行 + W3 完成 + W4 完成 +
        # 生命周期结果冻结（见任务四部署块）。

        # ── 任务四：部署结果 → 配对 RECOMMENDS + MITIGATES 观测（NATURAL）──
        # A7 §10: 真正的 W4 门槛 = W4 FINAL-OOT 完成证据 + 生命周期终态。
        # qualification_run_id 只证明 W3 阶段完成，不能证明 W4 已执行；
        # 必须检查 oot_validation_completed / w4_available /
        # candidate_frozen_before_oot / lifecycle_terminal。
        # Canary 阶段的 ROLLBACK 同样是终态失败案例，必须纳入校准。
        deployment_id = state.get("deployment_id")
        challenger_qualified = state.get("challenger_qualified")
        deployment_decision = state.get("deployment_decision")
        selected_strategy = state.get("selected_strategy_code")
        lifecycle_frozen = (
            bool(challenger_qualified)
            and bool(state.get("qualification_run_id"))
            and bool(state.get("oot_validation_completed"))
            and bool(state.get("w4_available"))
            and bool(state.get("candidate_frozen_before_oot"))
            and bool(state.get("lifecycle_terminal"))
            and deployment_decision in {"PROMOTE", "ROLLBACK"}
        )

        if lifecycle_frozen and deployment_id and primary_code and selected_strategy:
            if deployment_decision == "PROMOTE":
                direction = "SUPPORT"
                score = 0.95
            else:  # ROLLBACK（含 Canary 阶段回滚）
                direction = "AGAINST"
                score = 0.15

            evidence_detail = {
                "deployment_decision": deployment_decision,
                "challenger_qualified": challenger_qualified,
                "deployment_stage": state.get("deployment_stage"),
            }

            # 策略排序主要读取 RECOMMENDS.historical_effectiveness，
            # 终态结果必须同时产生配对的 RECOMMENDS 观测
            recommends_key = f"{primary_code}|RECOMMENDS|{selected_strategy}"
            observations.append(
                KgRelationObservation(
                    relation_key=recommends_key,
                    source_domain="DEPLOY",
                    source_record_id=f"deploy-rec:{deployment_id}",
                    lifecycle_run_id=lifecycle_run_id,
                    direction=direction,
                    evidence_score=score,
                    quality_weight=1.0,
                    weighted_strength=score,
                    data_track="NATURAL",
                    evidence_detail=evidence_detail,
                )
            )
            # 反向 MITIGATES 关联真实修复结果
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
                    evidence_detail=evidence_detail,
                )
            )

        logger.info(
            "observations_built",
            lifecycle_run_id=lifecycle_run_id,
            count=len(observations),
        )
        return observations


def _infer_alert_codes(state: dict) -> list[str]:
    """从 State 读取触发诊断的告警代码列表。

    严禁根据 severity 猜测告警类型（HIGH/CRITICAL → HIGH_FEATURE_PSI）。
    没有真实告警码就返回空列表，不回退假值。
    若一次诊断由多个告警触发，返回全部 alarm_code 逐条写入 INDICATES 观测。
    """
    alert_codes = state.get("alert_codes") or state.get("source_alert_codes")
    if alert_codes and isinstance(alert_codes, list) and len(alert_codes) > 0:
        return [str(c) for c in alert_codes]
    return []
