"""任务一正式监控合约模型 — 完整版（基于交接包 V1.4.2 监控边界）

本模块包含 WP02-WP08 监控链路的完整输出合约：
DetectionWindowPolicy → MonitoringMetricRecord → AlertContext → MonitoringRunEnvelope

与 alert_context.py 的关系：
- alert_context.py: 简化版 AlertDetail/AlertContext，供现有 API 路由使用
- 本模块: 完整版，供 MonitoringService 完整模式（run_detailed）使用
  两者不冲突，同时存在。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from ..common.base import ContractModel
from ..common.enums import AvailabilityStatus, DataTrack, ObjectType, RuleType, Severity

# ── 窗口策略 ──

WindowRole = Literal[
    "FIXED_REFERENCE",
    "HISTORICAL_NORMAL",
    "TRANSITION_OBSERVATION",
    "CURRENT_MONITORING",
]


class DetectionWindowPolicy(ContractModel):
    horizons_days: list[int]
    step_days: int = Field(gt=0)
    require_full_window: bool = True

    @model_validator(mode="after")
    def _validate_horizons(self) -> "DetectionWindowPolicy":
        if not self.horizons_days or any(v <= 0 for v in self.horizons_days):
            raise ValueError("horizons_days must contain positive integers")
        if self.horizons_days != sorted(set(self.horizons_days)):
            raise ValueError("horizons_days must be sorted and unique")
        return self


class ResolvedMonitoringWindow(ContractModel):
    window_id: str
    role: WindowRole
    start_date: date
    end_date: date
    locked: bool = False
    resolution_mode: Literal["FIXED_RANGE", "ENDING_AT_EVALUATION", "PRECEDING_WINDOW"]

    @model_validator(mode="after")
    def _validate_range(self) -> "ResolvedMonitoringWindow":
        if self.start_date >= self.end_date:
            raise ValueError(f"{self.window_id} start_date must be before end_date")
        return self


class ResolvedMonitoringWindowSet(ContractModel):
    policy_id: str
    policy_version: str
    timezone: str
    interval_boundary: Literal["LEFT_CLOSED_RIGHT_OPEN"] = "LEFT_CLOSED_RIGHT_OPEN"
    evaluation_date: date
    detection: DetectionWindowPolicy
    windows: dict[str, ResolvedMonitoringWindow]

    @model_validator(mode="after")
    def _validate_semantics(self) -> "ResolvedMonitoringWindowSet":
        expected: dict[str, WindowRole] = {
            "W0": "FIXED_REFERENCE",
            "W1": "HISTORICAL_NORMAL",
            "W2": "TRANSITION_OBSERVATION",
            "W3": "CURRENT_MONITORING",
        }
        missing = set(expected) - set(self.windows)
        if missing:
            raise ValueError(f"Missing required windows: {sorted(missing)}")
        for window_id, role in expected.items():
            w = self.windows[window_id]
            if w.window_id != window_id or w.role != role:
                raise ValueError(f"{window_id} must use role {role}")
        ordered = [self.windows[k] for k in ("W0", "W1", "W2", "W3")]
        for prev, cur in zip(ordered, ordered[1:]):
            if prev.end_date > cur.start_date:
                raise ValueError(f"{prev.window_id} and {cur.window_id} must not overlap")
        max_horizon = max(self.detection.horizons_days)
        for window_id in ("W1", "W3"):
            dur = (self.windows[window_id].end_date - self.windows[window_id].start_date).days
            if dur < max_horizon:
                raise ValueError(f"{window_id} must cover at least {max_horizon} days")
        return self


# ── 阈值规则 ──


class MetricAlertRule(ContractModel):
    metric_code: str
    window_days: int = Field(gt=0)
    alert_code: str
    threshold: float
    warning_threshold: float | None = None
    rule_type: RuleType
    direction: Literal["HIGHER_BETTER", "LOWER_BETTER", "DEVIATION_BAD"]
    threshold_rule_id: str
    threshold_rule_version: str
    severity: Severity


class MonitoringWindowSemantics(ContractModel):
    reference_window_id: str
    historical_window_id: str
    current_window_id: str
    historical_aggregation_7d: str
    historical_aggregation_30d: str


# ── 指标记录 ──


class MonitoringMetricRecord(ContractModel):
    """单条监控指标记录（含 reference/historical/current 三窗对比）。"""

    schema_version: str = "1.0"
    trace_id: str
    lifecycle_run_id: str
    monitoring_run_id: str
    metric_id: str
    model_id: str
    model_version: str
    baseline_id: str
    data_track: DataTrack
    scenario_id: str | None = None
    monitor_window_id: str
    metric_code: str
    metric_version: str
    object_type: ObjectType
    object_code: str
    unit: str | None = None
    window_days: int = Field(gt=0)
    reference_window_id: str
    historical_window_id: str
    current_window_id: str
    historical_value: float | None
    current_value: float | None
    delta: float | None
    availability_status: AvailabilityStatus
    metric_detail: dict[str, Any] = Field(default_factory=dict)
    calculated_at: datetime


# ── 证据记录 ──


class MonitoringEvidenceRecord(ContractModel):
    """单条监控证据（D/R/C/T/I 证据类型的基础层）。"""

    schema_version: str = "1.0"
    evidence_id: str
    trace_id: str
    lifecycle_run_id: str
    monitoring_run_id: str
    model_id: str
    model_version: str
    baseline_id: str
    monitor_window_id: str
    evidence_domain: Literal[
        "DATA_QUALITY", "DISTRIBUTION", "MODEL_PERFORMANCE", "DETECTOR", "SENTINEL", "DATA_INTEGRITY"
    ]
    evidence_role: Literal["SYMPTOM", "CONTEXT", "DECISION"]
    metric_code: str
    object_type: ObjectType
    object_code: str
    window_days: int | None = Field(default=None, gt=0)
    baseline_value: float | None = None
    current_value: float | None = None
    delta: float | None = None
    threshold: float | None = None
    evidence_status: Literal["NORMAL", "WARNING", "ABNORMAL", "UNAVAILABLE"]
    availability_status: AvailabilityStatus
    root_cause_status: Literal["PENDING_DIAGNOSIS"] = "PENDING_DIAGNOSIS"
    related_evidence_ids: list[str] = Field(default_factory=list)
    evidence_detail: dict[str, Any] = Field(default_factory=dict)
    observed_at: datetime


# ── 告警明细（增强版，含 episode 信息） ──


class AlertDetailExtended(ContractModel):
    """告警明细 — 增强版，包含 episode 持续区间追踪。"""

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
    threshold: float
    rule_type: RuleType
    threshold_rule_id: str
    threshold_rule_version: str
    availability_status: AvailabilityStatus
    metric_detail: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


# ── Alert Context（增强版） ──


class AlertContextExtended(ContractModel):
    """任务一正式输出对象 — 增强版，含 anomaly_probability + top_signals。"""

    schema_version: str = "1.2"
    trace_id: str
    monitoring_run_id: str
    model_id: str
    model_version: str
    monitor_window_id: str
    baseline_id: str
    data_track: DataTrack
    scenario_id: str | None = None
    anomaly_probability: float | None = Field(default=None, ge=0, le=1)
    top_signals: list[str] = Field(default_factory=list)
    alert_details: list[AlertDetailExtended] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_scenario_id_for_scenario_track(self) -> "AlertContextExtended":
        if self.data_track == DataTrack.SCENARIO and not self.scenario_id:
            raise ValueError("scenario_id is required for SCENARIO Alert Context")
        return self


# ── 节点状态 ──


class MonitoringNodeState(ContractModel):
    monitoring_run_id: str
    has_alerts: bool
    alert_count: int = Field(ge=0)
    max_alert_severity: Literal["NONE", "INFO", "WARNING", "HIGH", "CRITICAL"]


# ── 监控运行信封（任务一 → 任务二的交接边界） ──


class MonitoringRunEnvelope(ContractModel):
    """任务一完成后的完整监控运行输出。

    是任务二（WP09-WP12 层次化渐进诊断）的正式输入。
    """

    lifecycle_run_id: str
    monitoring_status: Literal["NORMAL", "EARLY_WARNING", "ALERT"]
    alert_context: AlertContextExtended
    state: MonitoringNodeState
