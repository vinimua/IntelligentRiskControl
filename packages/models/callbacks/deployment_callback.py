"""
Deployment Worker Callback
来源：技术开发文档 V1.4.2 §13.2, 接口总汇 V1.1 §3.3
"""

from ..common.base import ContractModel

from ..common.enums import WorkerStatus

class DeploymentCallback(ContractModel):
    """
    部署动作完成后恢复 DeploymentNode
    幂等键：deployment_action_id
    """

    deployment_action_id: str
    status: WorkerStatus
    action_type: str
    error_code: str | None = None
    error_message: str | None = None
