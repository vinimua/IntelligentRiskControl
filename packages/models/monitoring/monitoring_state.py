"""
MonitoringNode State 输出
来源：技术开发文档 V1.4.2 §6.4, 接口总汇 V1.1 §5.3
"""

from ..common.base import ContractModel

from ..common.enums import Severity

class MonitoringStateOutput(ContractModel):
    """任务一节点的 LangGraph State 摘要 — 完整指标和 Alert 明细写业务表"""

    monitoring_run_id: str
    has_alerts: bool
    alert_count: int = 0
    max_alert_severity: Severity | None = None
