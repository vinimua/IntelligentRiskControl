"""趋势特征 — 滑动窗口最小二乘斜率。

本模块从 risk_inquiry_agent/src/monitoring/trend_features.py 移植。

trailing_slope 对最近 N 个时序值做线性拟合，返回斜率。
斜率 > 0 且很大 → "指标在快速恶化"
斜率 ≈ 0     → "指标稳定"
斜率 < 0     → "指标在改善"
"""

from __future__ import annotations

import numpy as np


def trailing_slope(values: list[float | None], count: int = 5) -> float | None:
    """计算最近 count 个有效值的线性回归斜率。

    Args:
        values: 时序值列表（允许 None）。
        count: 取最近的多少个值做拟合。

    Returns:
        斜率（float），如果有效值 < 2 则返回 None。
    """
    usable = [float(v) for v in values[-count:] if v is not None and np.isfinite(v)]
    if len(usable) < 2:
        return None
    return float(np.polyfit(np.arange(len(usable), dtype=float), np.asarray(usable), 1)[0])
