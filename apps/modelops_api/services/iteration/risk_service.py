"""硬规则优先的修复决策风险评估。"""

from uuid import uuid4

from packages.models.common.enums import RecommendedAction, RiskLevel
from packages.models.iteration.decision_proposal import DecisionProposal
from packages.models.iteration.risk_assessment import RiskAssessment

from .config_loader import IterationConfigBundle, load_iteration_config


class RiskAssessmentService:
    def __init__(self, config: IterationConfigBundle | None = None) -> None:
        self.config = config or load_iteration_config()

    def assess(self, proposal: DecisionProposal) -> RiskAssessment:
        rules = self.config.risk
        strategy_codes = {item.strategy_code for item in proposal.strategies}
        hard_rules: list[str] = []
        reasons: list[str] = []
        risk_level = RiskLevel.LOW
        risk_score = 20

        if proposal.action.value in rules.high_risk_actions:
            hard_rules.append("HIGH_RISK_ACTION")
            reasons.append(f"高风险动作：{proposal.action.value}")
        if proposal.primary_root_cause_code in rules.high_risk_root_cause_codes:
            hard_rules.append("HIGH_RISK_ROOT_CAUSE")
            reasons.append(f"高风险根因：{proposal.primary_root_cause_code}")
        if strategy_codes & set(rules.high_risk_strategy_codes):
            hard_rules.append("HIGH_RISK_STRATEGY")
            reasons.append("策略涉及特征重构、全量重训或正式阈值变更")
        if proposal.action == RecommendedAction.MANUAL_REVIEW:
            hard_rules.append("DECISION_RULE_REQUIRES_REVIEW")
            reasons.append("确定性规则无法安全自动决策")

        if hard_rules:
            risk_level = RiskLevel.HIGH
            risk_score = 80
        elif strategy_codes & set(rules.medium_risk_strategy_codes):
            risk_level = RiskLevel.MEDIUM
            risk_score = 50
            reasons.append("策略改变样本权重、正则化或校准器")

        requires_review = (
            risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}
            or proposal.requires_manual_review
        )
        return RiskAssessment(
            assessment_id=str(uuid4()),
            proposal_id=proposal.proposal_id,
            risk_level=risk_level,
            risk_score=risk_score,
            hard_rule_codes=hard_rules,
            reasons=reasons,
            requires_manual_review=requires_review,
            rule_version=rules.rule_version,
        )
