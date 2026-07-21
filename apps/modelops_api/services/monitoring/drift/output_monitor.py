"""预测输出分布监控 — 基于交接包 output_monitor.py。

监控模型打分分布的漂移：均值、标准差、范围、PSI。
"""

from __future__ import annotations

import pandas as pd

from .algorithms import psi_from_edges


def output_metrics(
    scores: pd.Series,
    reference: pd.Series,
    frozen_edges: list[float],
) -> dict[str, float]:
    """计算预测输出的完整分布指标。

    Args:
        scores: 当前窗口的模型预测分。
        reference: W0 参照窗口的模型预测分。
        frozen_edges: W0 分数分箱边界（冻结）。

    Returns:
        dict with keys: prediction_mean, prediction_std, prediction_min,
        prediction_max, prediction_psi
    """
    numeric = pd.to_numeric(scores, errors="coerce").dropna()
    return {
        "prediction_mean": float(numeric.mean()),
        "prediction_std": float(numeric.std(ddof=0)),
        "prediction_min": float(numeric.min()),
        "prediction_max": float(numeric.max()),
        "prediction_psi": psi_from_edges(reference, numeric, frozen_edges),
    }
