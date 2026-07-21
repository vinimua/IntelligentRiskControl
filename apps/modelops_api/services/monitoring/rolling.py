"""日历时间滚动窗口迭代器 — 基于交接包 rolling_window.py。

按固定步长在时间序列数据上生成滑动窗口，
用于 7 天/30 天多窗口持续监控。
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def iter_rolling_windows(
    frame: pd.DataFrame,
    window_days: int = 7,
    step_days: int = 1,
    require_full_window: bool = False,
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]]:
    """按日历时间滑动，生成 (窗口开始, 窗口结束, 子集DataFrame) 迭代器。

    Args:
        frame: 含 apply_time 列的时间序列数据。
        window_days: 每个窗口的天数（常见值：7 或 30）。
        step_days: 滑动步长（天数）。
        require_full_window: True 时只返回完整的窗口（末尾不完整窗口丢弃）。

    Yields:
        (start_timestamp, end_timestamp, window_dataframe)
    """
    if frame.empty:
        return

    ordered = frame.sort_values("apply_time").copy()
    ordered["apply_time"] = pd.to_datetime(ordered["apply_time"])

    start = ordered["apply_time"].min().normalize()
    final = ordered["apply_time"].max()
    coverage_end = final.normalize() + pd.Timedelta(days=1)

    while start <= final:
        end = start + pd.Timedelta(days=window_days)
        if require_full_window and end > coverage_end:
            break

        subset = ordered[
            (ordered["apply_time"] >= start) & (ordered["apply_time"] < end)
        ].copy()

        if not subset.empty:
            yield start, end, subset

        start += pd.Timedelta(days=step_days)
