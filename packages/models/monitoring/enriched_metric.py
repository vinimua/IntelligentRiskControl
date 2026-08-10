"""富化指标响应模型 — 监控判定台前端接口契约。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EnrichedMetric(BaseModel):
    """富化后的单个指标，包含阈值、分类、解释等判定台所需全部字段。"""

    metric_code: str
    display_name: str
    category: str  # performance | drift | quality | stability
    baseline_value: float | None = None
    current_value: float | None = None
    delta: float | None = None
    direction: str | None = None  # higher_better | lower_better | deviation_bad
    availability_status: str = "AVAILABLE"
    rule_enabled: bool = False
    warning_threshold: float | None = None
    critical_threshold: float | None = None
    triggered: bool = False
    severity: str | None = None
    threshold_usage_ratio: float | None = None
    status_reason: str = ""
    metric_detail: dict | None = None

    model_config = {"extra": "allow"}


class CategoryBreakdown(BaseModel):
    """单个类别统计。"""

    total: int = 0
    normal: int = 0
    warning: int = 0
    critical: int = 0
    unmonitored: int = 0
    unavailable: int = 0

    model_config = {"extra": "allow"}


class ClosestThreshold(BaseModel):
    """最接近预警阈值的指标条目。"""

    metric_code: str
    display_name: str
    usage_ratio: float | None = None

    model_config = {"extra": "allow"}


class UnmonitoredMetric(BaseModel):
    """未接入告警规则的指标。"""

    metric_code: str
    display_name: str

    model_config = {"extra": "allow"}


class CoverageSummary(BaseModel):
    """规则覆盖摘要。"""

    total_metrics: int = 0
    calculated: int = 0
    available: int = 0
    rules_enabled: int = 0
    triggered: int = 0
    category_breakdown: dict[str, CategoryBreakdown] = Field(default_factory=dict)
    closest_thresholds: list[ClosestThreshold] = Field(default_factory=list)
    label_maturity: dict = Field(default_factory=dict)
    unmonitored_metrics: list[UnmonitoredMetric] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class EnrichedMetricsResponse(BaseModel):
    """GET /api/monitoring/runs/{id}/enriched-metrics 的响应 data。"""

    metrics: list[EnrichedMetric] = Field(default_factory=list)
    summary: CoverageSummary | None = None

    model_config = {"extra": "allow"}


class FeatureDriftItem(BaseModel):
    """特征漂移表格行。"""

    feature_name: str
    psi_7d: float | None = None
    psi_30d: float | None = None
    max_psi: float | None = None
    threshold: float = 0.10
    status: str = "normal"  # normal | warning | critical
    model_importance: str | None = None  # 高 | 中 | 低
    trend: str = "stable"  # up | down | stable

    model_config = {"extra": "allow"}


class DataQualityField(BaseModel):
    """单个字段的数据质量信息。"""

    field_name: str
    baseline_missing_rate: float | None = None
    current_missing_rate: float | None = None
    missing_delta: float | None = None
    outlier_rate: float | None = None
    outlier_delta: float | None = None
    dq_flag: str = "OK"  # OK | WARN | ALERT

    model_config = {"extra": "allow"}


class SchemaChange(BaseModel):
    """Schema 变更条目。"""

    change_type: str  # added | removed | type_changed
    column_name: str
    detail: str = ""

    model_config = {"extra": "allow"}


class DataQualityResponse(BaseModel):
    """GET /api/monitoring/runs/{id}/data-quality 的响应 data。"""

    overall_missing_rate: float | None = None
    overall_outlier_rate: float | None = None
    dq_score: float | None = None
    fields: list[DataQualityField] = Field(default_factory=list)
    schema_changes: list[SchemaChange] = Field(default_factory=list)

    model_config = {"extra": "allow"}
