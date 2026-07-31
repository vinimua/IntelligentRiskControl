"""指标计算器注册表 — 可插拔的 Metric Calculator Registry"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from packages.models.common.enums import AvailabilityStatus


@dataclass
class MetricResult:
    """单个指标的计算结果。"""

    metric_code: str
    baseline_value: float | None = None
    current_value: float | None = None
    delta: float | None = None
    availability_status: AvailabilityStatus = AvailabilityStatus.AVAILABLE
    metric_detail: dict = field(default_factory=dict)


# 类型别名：每个指标计算器接受 baseline 和 current 数据集，返回 MetricResult
MetricCalculator = Callable[[list[dict], list[dict]], MetricResult]

# 全局注册表
METRIC_CALCULATORS: dict[str, MetricCalculator] = {}


def register(metric_code: str):
    """装饰器：将指标计算器注册到全局注册表。"""

    def decorator(fn: MetricCalculator) -> MetricCalculator:
        METRIC_CALCULATORS[metric_code] = fn
        return fn

    return decorator
