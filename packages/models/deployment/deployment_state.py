"""
DeploymentNode 输出
来源：技术开发文档 V1.4.2 §4.1, 接口总汇 V1.1 §8.2
"""

from ..common.base import ContractModel

from ..common.enums import DeploymentDecision

class DeploymentStateOutput(ContractModel):
    """任务四节点的 LangGraph State 摘要"""

    deployment_id: str
    stage: str
    decision: DeploymentDecision
