"""检测器批量运行编排 — 按 (模型, 数据轨道, 场景实例) 分组，按时间顺序推进。

本模块从 risk_inquiry_agent/src/monitoring/detectors/runner.py 移植。

每个检测信号（如 auc、max_feature_psi_7d 等）创建 4 个检测器实例：
    ADWIN + PageHinkley + KSWIN + RobustZ

总输出：N 个信号 × 4 个检测器 = 4N 列检测器特征。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .algorithms import ADWINDetector, KSWINDetector, PageHinkleyDetector, RobustZDetector

# ── 默认检测器配置（与交接包 configs/detectors.yaml 一致） ──

DEFAULT_DETECTOR_CONFIG = {
    "adwin": {"delta": 0.002, "grace_period": 5},
    "page_hinkley": {"min_instances": 5, "delta": 0.005, "threshold": 25.0, "alpha": 0.999},
    "kswin": {"alpha": 0.005, "window_size": 20, "stat_size": 8, "seed": 2026},
    "robust_z": {"window_size": 10, "warmup": 5, "z_threshold": 3.5},
}

# ── 默认监测信号列表（与交接包 configs/detectors.yaml 一致） ──

DEFAULT_SIGNALS = [
    "auc",
    "ks",
    "prediction_mean",
    "prediction_psi_7d",
    "prediction_psi_30d",
    "max_feature_psi_7d",
    "max_feature_psi_30d",
    "missing_rate_max_delta",
    "outlier_rate_max_delta",
]


def _build_detectors(config: dict | None = None) -> list[object]:
    """根据配置构建四个检测器的实例列表。"""
    cfg = {**DEFAULT_DETECTOR_CONFIG, **(config or {})}
    return [
        ADWINDetector(**{k: v for k, v in cfg.get("adwin", {}).items() if k != "clock"}),
        PageHinkleyDetector(**cfg.get("page_hinkley", {})),
        KSWINDetector(**cfg.get("kswin", {})),
        RobustZDetector(**cfg.get("robust_z", {})),
    ]


def run_detectors(
    features: pd.DataFrame,
    signals: list[str] | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """按时间顺序批量运行所有检测器。

    分组策略：每个 (model_id, data_track, scenario_instance_id) 组合
    拥有一组独立的、有状态的检测器实例。不同组合之间互不干扰。

    Args:
        features: 监测特征向量 DataFrame。必须包含：
            model_id, data_track, scenario_instance_id,
            window_end（或 monitor_window_id，用于排序），
            以及 signals 参数指定的所有列。
        signals: 要监测的信号列名列表。默认 9 个信号。
        config: 检测器参数配置。默认与交接包 detectors.yaml 一致。

    Returns:
        DataFrame，每行 = 一个窗口 × 一个信号 × 一个检测器。
        列：model_id, model_version, monitor_window_id, scenario_id,
             scenario_instance_id, data_track, detector_name, signal_name,
             signal_value, alarm_flag, detector_score, warmup_status, created_at
    """
    signal_list = signals or DEFAULT_SIGNALS
    cfg = config or {}

    output: list[dict] = []
    group_columns = ["model_id", "data_track", "scenario_instance_id"]
    # 如果 DataFrame 中没有 scenario_instance_id 列，只用 model_id + data_track 分组
    available_groups = [c for c in group_columns if c in features.columns]

    for _, group in features.groupby(available_groups, dropna=False):
        # 每个分组创建独立的检测器组
        states = {sig: _build_detectors(cfg) for sig in signal_list}

        # 按时间排序
        sort_col = "window_end" if "window_end" in group else "monitor_window_id"
        ordered = group.sort_values(sort_col)

        for _, row in ordered.iterrows():
            ts = pd.to_datetime(
                row.get("window_end", datetime.now(timezone.utc))
            ).to_pydatetime()

            for signal_name, detector_list in states.items():
                if signal_name not in row:
                    continue
                value = row.get(signal_name)

                for detector in detector_list:
                    if pd.isna(value):
                        result = {
                            "alarm_flag": False,
                            "detector_score": 0.0,
                            "warmup_status": "MISSING",
                        }
                    else:
                        detector.update(float(value), ts)
                        cur = detector.get_result()
                        result = {
                            "alarm_flag": cur.alarm_flag,
                            "detector_score": cur.detector_score,
                            "warmup_status": cur.warmup_status,
                        }

                    record = {
                        "model_id": str(row.get("model_id", "")),
                        "model_version": str(row.get("model_version", "")),
                        "monitor_window_id": str(row.get("monitor_window_id", "")),
                        "scenario_id": row.get("scenario_id"),
                        "scenario_instance_id": row.get("scenario_instance_id"),
                        "data_track": str(row.get("data_track", "NATURAL")),
                        "detector_name": detector.name,
                        "signal_name": signal_name,
                        "signal_value": None if pd.isna(value) else float(value),
                        "created_at": datetime.now(timezone.utc),
                        **result,
                    }
                    output.append(record)

    return pd.DataFrame(output)
