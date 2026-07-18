"""
部署告警字段
来源：技术开发文档 V1.4.2 §9.4, 接口总汇 V1.1 §8.4
"""

from ..common.base import ContractModel

from ..common.enums import Severity

class DeploymentAlert(ContractModel):
    """部署阶段异常 — DeploymentAlert 查询入口"""

    alert_code: str
    metric_code: str
    champion_value: float | None = None
    challenger_value: float | None = None
    value: float | None = None
    threshold: float | None = None
    severity: Severity = Severity.WARNING
    stage: str  # offline_validation / oot / shadow / canary_5 / canary_20 / canary_50 / production
