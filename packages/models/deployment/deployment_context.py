"""
Deployment Context — KnowledgeService 返回部署风险和策略建议
来源：知识图谱接口 V1.1 §6.1
"""

from pydantic import Field
from ..common.base import ContractModel

class DeploymentRisk(ContractModel):
    """部署风险 — 图谱风险节点"""

    risk_code: str
    relation_key: str  # DeploymentAlert|INDICATES|DeploymentRisk
    effective_weight_snapshot: float
    confidence_lower_bound_snapshot: float
    strategy_candidates: list[dict] = Field(default_factory=list)

class DeploymentContext(ContractModel):
    """
    部署上下文 — 图谱返回风险和处置候选
    最终 PROMOTE / HOLD / ROLLBACK 等决策由 Gatekeeper 产生
    """

    context_pack_id: str
    deployment_risks: list[DeploymentRisk] = Field(default_factory=list)
    gatekeeper_rule_refs: list[str] = Field(default_factory=list)
    retrieval_degraded: bool = False
