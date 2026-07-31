"""人工复核合同。"""

from datetime import datetime

from pydantic import Field, model_validator

from ..common.base import ContractModel
from ..common.enums import ReviewDecision


class ManualReviewSubmission(ContractModel):
    proposal_id: str
    reviewer_id: str
    decision: ReviewDecision
    reason: str = Field(min_length=1)
    rejection_reason_codes: list[str] = Field(default_factory=list)
    adjustment_instructions: list[str] = Field(default_factory=list)
    forbidden_adjustments: list[str] = Field(default_factory=list)
    expected_evidence: list[str] = Field(default_factory=list)
    reviewed_at: datetime

    @model_validator(mode="after")
    def rejection_requires_adjustment(self) -> "ManualReviewSubmission":
        if (
            self.decision == ReviewDecision.REJECT
            and not self.adjustment_instructions
        ):
            raise ValueError(
                "rejected review must include adjustment_instructions for the Agent"
            )
        return self


class ManualReviewReport(ContractModel):
    review_id: str
    proposal_id: str
    reviewer_id: str
    decision: ReviewDecision
    reason: str
    rejection_reason_codes: list[str] = Field(default_factory=list)
    adjustment_instructions: list[str] = Field(default_factory=list)
    forbidden_adjustments: list[str] = Field(default_factory=list)
    expected_evidence: list[str] = Field(default_factory=list)
    parent_review_id: str | None = None
    reviewed_at: datetime
