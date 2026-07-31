"""决策风险评估合同。"""

from pydantic import Field

from ..common.base import ContractModel
from ..common.enums import RiskLevel


class RiskAssessment(ContractModel):
    assessment_id: str
    proposal_id: str
    risk_level: RiskLevel
    risk_score: int = Field(ge=0, le=100)
    hard_rule_codes: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    requires_manual_review: bool
    rule_version: str
