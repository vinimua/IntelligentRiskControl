"""Sentinel 推理 — LightGBM 异常概率 + Top-5 贡献信号。

本模块从 risk_inquiry_agent 的以下文件移植：
- sentinel_inference.py: infer_sentinel() + Top-5 信号提取
- train_sentinel.py: SentinelBundle + calibrated_probability() + choose_threshold()

生产环境只需推理。Sentinel 模型的训练（train_sentinel）
由离线 Celery 任务异步执行，不在本模块中实现。

使用方式：
    bundle = SentinelBundle(model, features, medians, threshold, ...)
    results = infer_sentinel(bundle, feature_frame)
    # results 包含 anomaly_probability 和 top_signals
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import uuid

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# SentinelBundle — 已训练的 Sentinel 模型包
# ═══════════════════════════════════════════════════════════════


@dataclass
class SentinelBundle:
    """已训练的 LightGBM Sentinel 模型包。

    训练时由 train_sentinel() 生成，序列化为 joblib 文件。
    生产环境从对象存储（MinIO）加载后直接用于推理。
    """

    model: object               # LightGBM Booster
    features: list[str]         # 特征列名（按顺序）
    medians: dict[str, float]   # 特征中位数（用于填充缺失值）
    threshold: float            # 告警阈值（FPR ≤ 3% 约束下选出）
    sentinel_version: str       # 模型版本号
    calibrator: object | None = None    # Platt 校准器（LogisticRegression）
    calibration_method: str = "NONE"    # 校准方法："PLATT" | "NONE"


# ═══════════════════════════════════════════════════════════════
# 概率校准
# ═══════════════════════════════════════════════════════════════


def calibrated_probability(bundle: SentinelBundle, frame: pd.DataFrame) -> np.ndarray:
    """计算校准后的异常概率。

    LightGBM 直接输出的概率可能不等于真实频率。
    Platt 校准（LogisticRegression 在 raw_score 上拟合）修正系统性偏差。
    """
    raw = bundle.model.booster_.predict(frame, raw_score=True)
    if bundle.calibrator is None:
        return bundle.model.predict_proba(frame)[:, 1]
    return bundle.calibrator.predict_proba(np.asarray(raw).reshape(-1, 1))[:, 1]


# ═══════════════════════════════════════════════════════════════
# Sentinel 推理
# ═══════════════════════════════════════════════════════════════


def infer_sentinel(
    bundle: SentinelBundle,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """对每个监测窗口计算异常概率 + Top-5 贡献信号。

    Args:
        bundle: 已加载的 Sentinel 模型包。
        frame: 监测特征向量 DataFrame。必须包含 bundle.features 中的所有列。

    Returns:
        DataFrame，每行对应一个监测窗口。列：
        trace_id, model_id, model_version, baseline_id, baseline_version,
        monitor_window_id, scenario_id, scenario_instance_id, data_track,
        anomaly_probability, alert_threshold, sentinel_version,
        top_signals, top_signal_details, created_at
    """
    # 特征工程：替换 inf、填充缺失
    features = frame[bundle.features].replace([np.inf, -np.inf], np.nan).fillna(bundle.medians)

    # Sentinel 打分
    probability = calibrated_probability(bundle, features)

    # LightGBM pred_contrib 提取每个特征对最终预测的贡献值
    contributions = bundle.model.booster_.predict(features, pred_contrib=True)[:, :-1]

    rows = []
    for pos, (_, source) in enumerate(frame.iterrows()):
        # 取贡献绝对值最大的 5 个信号
        order = np.argsort(np.abs(contributions[pos]))[::-1][:5]
        details = [
            {
                "signal": bundle.features[i],
                "contribution": float(contributions[pos, i]),
                "value": float(features.iloc[pos, i]),
            }
            for i in order
        ]

        trace_id = str(source.get("trace_id") or uuid.uuid4())
        rows.append({
            "trace_id": trace_id,
            "model_id": str(source["model_id"]),
            "model_version": str(source["model_version"]),
            "baseline_id": str(source.get("baseline_id", "")),
            "baseline_version": str(source.get("baseline_version", "")),
            "monitor_window_id": str(source["monitor_window_id"]),
            "scenario_id": source.get("scenario_id"),
            "scenario_instance_id": source.get("scenario_instance_id"),
            "data_track": str(source.get("data_track", "NATURAL")),
            "anomaly_probability": float(probability[pos]),
            "alert_threshold": bundle.threshold,
            "sentinel_version": bundle.sentinel_version,
            "top_signals": [d["signal"] for d in details],
            "top_signal_details": details,
            "created_at": datetime.now(timezone.utc),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# 阈值选择（供离线训练使用，此处保留供参考）
# ═══════════════════════════════════════════════════════════════


def _rates(
    y: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, object]:
    """计算给定阈值下的全部分类指标。"""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    pred = (probability >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) > 1 else None,
        "pr_auc": float(average_precision_score(y, probability)) if len(np.unique(y)) > 1 else None,
        "fpr": float(fp / max(1, fp + tn)),
        "confusion_matrix": cm.tolist(),
        "threshold": float(threshold),
    }


def choose_threshold(
    y: np.ndarray,
    probability: np.ndarray,
    fpr_constraint: float = 0.03,
) -> tuple[float, bool]:
    """选择最优告警阈值：先约束 FPR ≤ fpr_constraint，再最大化 Recall。

    Args:
        y: 真实标签（0=NORMAL, 1=ANOMALY）。
        probability: Sentinel 输出的原始概率。
        fpr_constraint: 最大允许假阳性率。

    Returns:
        (threshold, constraint_met): 最优阈值 + 是否满足 FPR 约束。
    """
    candidates = np.unique(np.concatenate([np.linspace(0, 1, 201), probability]))
    evaluated = [(float(t), _rates(y, probability, float(t))) for t in candidates]

    feasible = [item for item in evaluated if float(item[1]["fpr"]) <= fpr_constraint]
    if feasible:
        best = max(
            feasible,
            key=lambda item: (
                float(item[1]["recall"]),
                float(item[1]["precision"]),
                float(item[1]["f1"]),
                item[0],
            ),
        )
        return best[0], True

    # 如果所有阈值都无法满足 FPR 约束，选 FPR 最接近的
    best = min(
        evaluated,
        key=lambda item: (float(item[1]["fpr"]), -float(item[1]["recall"])),
    )
    return best[0], False
