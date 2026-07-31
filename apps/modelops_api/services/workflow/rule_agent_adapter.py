"""RuleAgentAdapter — 第一阶段规则代理。

实现 LangGraph 开发路线 V1.0 §7.2 的规则：
- 如果 handoff 中 recommended_action 非空，则输出同一 action
- 如果 primary_root_cause_score < 0.75，强制 MANUAL_REVIEW
- 如果 event_status != WAITING_AGENT_DECISION，拒绝推进
- Agent 输出只作为建议
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from apps.modelops_api.repositories.diagnosis_repo import DiagnosisRepo
from packages.models.workflow.agent_decision import AgentDecisionInput, AgentDecisionOutput

logger = structlog.get_logger(__name__)

# 低置信度阈值 — root_cause_score 低于此值时强制手工复核
LOW_CONFIDENCE_THRESHOLD = 0.75


class RuleAgentAdapter:
    """基于规则的 Agent 代理。

    第一阶段不调用真实 LLM Agent，只用确定性规则生成建议。
    后续可替换为真实 Agent 调用，输入/输出合同不变。
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def decide(self, agent_input: AgentDecisionInput) -> AgentDecisionOutput:
        """执行规则决策。"""

        score = agent_input.primary_root_cause_score
        recommended = agent_input.recommended_action or "MANUAL_REVIEW"

        # 规则 1: 低置信度 → 强制人工复核
        requires_manual_review = False
        rationale = ""
        final_action = recommended

        if score is None or score < LOW_CONFIDENCE_THRESHOLD:
            requires_manual_review = True
            rationale = (
                f"根因置信度 ({score or 'N/A'}) 低于阈值 {LOW_CONFIDENCE_THRESHOLD}，"
                f"强制进入人工复核"
            )
            final_action = "MANUAL_REVIEW"
            confidence = score if score is not None else 0.0
        else:
            confidence = score
            rationale = f"根因置信度 {score} 达到阈值，采纳推荐动作 {recommended}"

        # 规则 2: 是否允许自动决策
        allow_auto_decision = not requires_manual_review and confidence >= 0.85

        # 规则 3: 验证事件状态（调用方应已确保状态正确）
        event_id = agent_input.event_id
        if event_id:
            event = await DiagnosisRepo(self.session).get_event(event_id)
            if event and event.get("status") != "WAITING_AGENT_DECISION":
                requires_manual_review = True
                allow_auto_decision = False
                rationale += (
                    f" | 事件状态 {event.get('status')} 不是 WAITING_AGENT_DECISION"
                )

        decision_id = str(uuid.uuid4())

        logger.info(
            "rule_agent_decided",
            agent_decision_id=decision_id,
            event_id=event_id,
            recommended_action=final_action,
            confidence=confidence,
            requires_manual_review=requires_manual_review,
        )

        return AgentDecisionOutput(
            agent_decision_id=decision_id,
            event_id=event_id or "",
            diagnosis_run_id=agent_input.diagnosis_run_id,
            recommended_action=final_action,
            confidence=confidence,
            rationale=rationale,
            selected_root_cause_code=agent_input.primary_root_cause_code,
            allow_auto_decision=allow_auto_decision,
            requires_manual_review=requires_manual_review,
            forbidden_strategies=(
                ["W4_TRAINING", "W4_CALIBRATION"]
                if final_action == "MODEL_ITERATION"
                else []
            ),
            required_adjustments=(
                ["AUDIT_TRAIL", "SEGMENT_GOVERNANCE"]
                if final_action == "MODEL_ITERATION"
                else []
            ),
        )
