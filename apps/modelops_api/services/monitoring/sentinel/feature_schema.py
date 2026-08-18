"""Sentinel 特征契约 — 训练和推理必须共用此列表。

新增字段需更新 SENTINEL_FEATURE_SCHEMA_VERSION 并重新训练模型。
"""

from __future__ import annotations

import hashlib
import json

SENTINEL_FEATURE_SCHEMA_VERSION = "sentinel-features-v1"

SENTINEL_FEATURES: list[str] = [
    # ── 模型性能 ──
    "auc",
    "ks",
    "pr_auc",
    "brier",
    "ece",
    "bad_recall",
    "bad_rate",
    "bad_rate_delta",
    "performance_drop_max",
    "prediction_mean",
    "prediction_std",
    # ── 多窗口漂移（7D + 30D）──
    "prediction_psi_7d",
    "prediction_psi_30d",
    "max_feature_psi_7d",
    "max_feature_psi_30d",
    "max_feature_psi",
    "max_feature_js",
    "min_ks_q_value",
    "max_feature_ks_statistic",
    "max_segment_share_delta",
    # ── 数据质量 ──
    "missing_rate_max_delta",
    "outlier_rate_max_delta",
    "dq_score_min",
    "range_violation_rate_max",
    "unknown_category_rate_max",
    "default_value_rate_max_delta",
    # ── 检测器信号 ──
    "adwin_alarm_count",
    "ph_alarm_count",
    "kswin_alarm_count",
    "robust_z_alarm_count",
    "detector_vote_ratio",
]


def compute_schema_hash() -> str:
    """计算当前特征契约的 SHA256 哈希。"""
    return hashlib.sha256(
        json.dumps(SENTINEL_FEATURES, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
