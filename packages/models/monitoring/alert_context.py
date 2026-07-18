"""
Alert Context — 任务一的核心输出对象
同时也是 DiagnosisService 调用 KnowledgeService 的输入上下文
来源：技术开发文档 V1.4.2 §6.3, 接口总汇 V1.1 §5.2
"""

from pydantic import Field
from ..common.base import ContractModel

from datetime import datetime

from ..common.enums import (
    AvailabilityStatus,
    DataTrack,
    ObjectType,
    RuleType,
    Severity,
)

class AlertDetail(ContractModel):
    """逐项告警明细 — 对应 monitoring.monitoring_alerts.alert_id"""

    alert_id: str
    alert_code: str
    severity: Severity
    object_type: ObjectType
    object_code: str
    metric_code: str
    metric_version: str
    unit: str | None = None
    baseline_value: float | None = None
    current_value: float | None = None
    delta: float | None = None
    threshold: float | None = None
    rule_type: RuleType | None = None
    threshold_rule_id: str | None = None
    threshold_rule_version: str | None = None
    availability_status: AvailabilityStatus
    metric_detail: dict | None = None
    created_at: datetime | None = None

class AlertContext(ContractModel):
    """
    任务一正式输出对象
    顶层使用 monitoring_run_id 标识本次监控与告警集合；
    逐项告警使用 alert_details[].alert_id
    """

    schema_version: str
    trace_id: str
    monitoring_run_id: str
    model_id: str
    model_version: str
    monitor_window_id: str
    baseline_id: str
    data_track: DataTrack
    scenario_id: str | None = None
    anomaly_probability: float | None = None
    top_signals: list[str] = Field(default_factory=list)
    alert_details: list[AlertDetail] = Field(default_factory=list)
