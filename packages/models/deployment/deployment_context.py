"""Deployment KG context contracts.

KnowledgeService returns deployment risks and strategy candidates. The final
PROMOTE/HOLD/ROLLBACK decision is still made by the Gatekeeper.
"""

from pydantic import Field

from ..common.base import ContractModel


class DeploymentStrategyCandidate(ContractModel):
    """A strategy candidate recommended by the deployment KG."""

    strategy_code: str
    relation_key: str
    effective_weight_snapshot: float
    confidence_lower_bound_snapshot: float
    historical_success_rate: float | None = None
    support_case_count: int = 0
    natural_case_count: int = 0
    action_type: str = ""
    parameters: dict = Field(default_factory=dict)
    allowed_stages: list[str] = Field(default_factory=list)
    policy_refs: list[str] = Field(default_factory=list)
    mitigates_relation_key: str | None = None


class DeploymentRisk(ContractModel):
    """A deployment risk node inferred from one or more deployment alerts."""

    risk_code: str
    risk_name: str | None = None
    relation_key: str
    effective_weight_snapshot: float
    confidence_lower_bound_snapshot: float
    severity: str | None = None
    alert_codes: list[str] = Field(default_factory=list)
    evidence_detail: dict = Field(default_factory=dict)
    strategy_candidates: list[DeploymentStrategyCandidate] = Field(default_factory=list)


class DeploymentContext(ContractModel):
    """Deployment KG retrieval result used by the Gatekeeper."""

    context_pack_id: str
    model_id: str = ""
    stage: str = ""
    deployment_alerts: list = Field(default_factory=list)
    deployment_risks: list[DeploymentRisk] = Field(default_factory=list)
    gatekeeper_rule_refs: list[str] = Field(default_factory=list)
    retrieval_degraded: bool = False
    degradation_reason: str | None = None
