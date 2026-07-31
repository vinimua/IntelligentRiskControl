"""
DeploymentNode 输出
"""

from ..common.base import ContractModel

from ..common.enums import DeploymentDecision

class DeploymentStateOutput(ContractModel):
    """任务四节点的 LangGraph State 摘要"""

    deployment_id: str
    stage: str
    decision: DeploymentDecision
