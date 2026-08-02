"""T4-GAP-01: 新老模型比对合同。

ModelComparisonReport — champion vs challenger 完整指标矩阵。
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


class MetricPair(BaseModel):
    """一对指标值（champion vs challenger）。"""
    metric_code: str
    champion_value: float | None = None
    challenger_value: float | None = None
    delta: float | None = None
    delta_pct: float | None = None
    direction: str = ""  # "higher_is_better" | "lower_is_better" | "neutral"
    passed: bool | None = None


class ModelComparisonReport(BaseModel):
    """champion vs challenger 完整对比报告。"""

    comparison_id: str = Field(default_factory=lambda: str(uuid4()))
    model_id: str = ""
    champion_version: str = ""
    challenger_version: str = ""
    lifecycle_run_id: str | None = None
    qualification_run_id: str | None = None

    # 核心指标
    metrics: list[MetricPair] = Field(default_factory=list)
    passed: bool = False
    summary: str = ""

    # 分群指标
    segment_metrics: dict[str, list[MetricPair]] = Field(default_factory=dict)

    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
