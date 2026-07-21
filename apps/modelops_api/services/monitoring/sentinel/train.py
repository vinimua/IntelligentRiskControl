"""Sentinel 离线训练管道 — 基于交接包 train_sentinel.py + sentinel_dataset.py。

Celery 离线任务入口。完整流程：
  1. group_split() — 按 scenario_instance_id 分层分组防泄漏
  2. train_sentinel() — LightGBM 训练 + Platt 校准 + FPR约束阈值选择

训练完成后将 SentinelBundle 序列化为 joblib 存入 MinIO。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .inference import SentinelBundle, calibrated_probability

# ── 非特征列（不进入 LightGBM） ──

_NON_FEATURES = {
    "trace_id", "model_id", "model_version", "baseline_id", "baseline_version",
    "monitor_window_id", "window_start", "window_end", "window_days",
    "data_track", "scenario_id", "scenario_family", "scenario_instance_id",
    "sentinel_group_id", "scenario_acceptance_status", "anomaly_scope",
    "label_source", "label_rule_version", "scenario_category", "drift_type",
    "business_driver", "generation_method", "anomaly_label",
    "prediction_psi", "max_feature_psi",
}

# ═══════════════════════════════════════════════════════════════
# 分组拆分（防泄漏）
# ═══════════════════════════════════════════════════════════════


def sentinel_feature_columns(frame: pd.DataFrame) -> list[str]:
    """从 DataFrame 中提取数值型 Sentinel 特征列。"""
    return [
        c for c in frame.columns
        if c not in _NON_FEATURES and pd.api.types.is_numeric_dtype(frame[c])
    ]


def group_split(
    frame: pd.DataFrame,
    group_field: str = "scenario_instance_id",
    random_seed: int = 2026,
) -> dict[str, pd.DataFrame]:
    """按 sentinel_group_id 分层分组拆分为 train/validation/test。

    关键约束：同一个 scenario_instance_id 的所有窗口只能进入一个集合，
    防止同一异常实例的不同时间窗口跨集合泄漏。

    分层策略：label | scenario_family | event_mode，保证每层在三个集合中均衡分布。

    比例：3:1:1（每 5 个实例 → 3 train / 1 val / 1 test）

    Returns:
        {"train": DataFrame, "validation": DataFrame, "test": DataFrame}
    """
    labeled = frame[frame["anomaly_label"].notna()].copy()
    if group_field not in labeled:
        raise ValueError(f"Sentinel group field missing: {group_field}")

    # 聚合：每个 group 取 max label + first scenario_id
    aggregate: dict[str, Any] = {"anomaly_label": "max"}
    for col in ("scenario_id", "label_source"):
        if col in labeled:
            aggregate[col] = "first"
    summary = labeled.groupby(group_field, dropna=False).agg(aggregate).reset_index()

    if summary["anomaly_label"].nunique() < 2:
        raise ValueError("Sentinel labeled groups require both normal and anomalous classes")

    # 构建分层键
    if "scenario_id" in summary:
        scenario = summary["scenario_id"].fillna("").astype(str)
        source = (
            summary.get("label_source", pd.Series("NATURAL", index=summary.index))
            .fillna("NATURAL")
            .astype(str)
        )
        groups_as_text = summary[group_field].astype(str)
        event_mode = groups_as_text.map(
            lambda v: (
                "ACUTE_7D" if "_ACUTE_7D_" in v
                else "PERSISTENT_30D" if "_PERSISTENT_30D_" in v
                else "ROLLING"
            )
        )
        summary["_split_stratum"] = (
            summary["anomaly_label"].astype(str) + "|"
            + scenario.where(scenario.ne(""), source) + "|"
            + event_mode
        )
    else:
        summary["_split_stratum"] = summary["anomaly_label"].astype(str)

    # 每层随机分配
    rng = np.random.default_rng(random_seed)
    assignments: dict[object, str] = {}

    for _, class_groups in summary.groupby("_split_stratum"):
        values = class_groups[group_field].to_numpy(copy=True)
        rng.shuffle(values)
        for i, group in enumerate(values):
            assignments[group] = (
                "test" if i % 5 == 0 else "validation" if i % 5 == 1 else "train"
            )

    splits = {
        name: labeled[labeled[group_field].map(assignments) == name].copy()
        for name in ("train", "validation", "test")
    }

    # 泄漏检查
    train_set = set(splits["train"][group_field])
    val_set = set(splits["validation"][group_field])
    test_set = set(splits["test"][group_field])
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise AssertionError("Sentinel group split leakage detected")

    return splits


# ═══════════════════════════════════════════════════════════════
# 指标计算
# ═══════════════════════════════════════════════════════════════


def _rates(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, object]:
    """计算给定阈值下的全部分类指标。"""
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
    """先约束 FPR ≤ fpr_constraint，再最大化 Recall。"""
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

    best = min(
        evaluated,
        key=lambda item: (float(item[1]["fpr"]), -float(item[1]["recall"])),
    )
    return best[0], False


# ═══════════════════════════════════════════════════════════════
# Sentinel 训练入口
# ═══════════════════════════════════════════════════════════════


def train_sentinel(
    dataset: pd.DataFrame,
    random_seed: int = 2026,
    fpr_constraint: float = 0.03,
    calibration_method: str = "PLATT",
    min_recall: float = 0.90,
    min_pr_auc: float = 0.90,
    lightgbm_params: dict[str, Any] | None = None,
    group_field: str = "sentinel_group_id",
    sentinel_version: str = "sentinel_lgbm_v1",
    artifact_dir: str | Path = ".",
) -> tuple[SentinelBundle, dict[str, object], dict[str, pd.DataFrame]]:
    """训练 LightGBM Sentinel 模型。

    Args:
        dataset: 标注好的训练数据集（含 anomaly_label 和特征列）。
        random_seed: 随机种子。
        fpr_constraint: 最大允许假阳性率（默认 0.03）。
        calibration_method: 校准方法 "PLATT" 或 "NONE"。
        min_recall: 最低召回率目标。
        min_pr_auc: 最低 PR-AUC 目标。
        lightgbm_params: LightGBM 超参（默认使用交接包配置）。
        group_field: 分组字段名。
        sentinel_version: 模型版本号。
        artifact_dir: 模型保存路径。

    Returns:
        (SentinelBundle, metrics_dict, splits_dict)
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError("LightGBM is required for Sentinel training") from exc

    # 默认超参（与交接包 sentinel.yaml 一致）
    if lightgbm_params is None:
        lightgbm_params = {
            "class_weight": "balanced",
            "n_estimators": 120,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "max_depth": 5,
            "min_child_samples": 2,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "n_jobs": 1,
        }

    artifact_dir = Path(artifact_dir)

    # ① 分组拆分
    splits = group_split(dataset, group_field=group_field, random_seed=random_seed)
    features = sentinel_feature_columns(dataset)
    train_df = splits["train"]

    if train_df.empty or train_df["anomaly_label"].nunique() < 2:
        raise ValueError("Sentinel training split requires both normal and anomalous groups")

    # ② 特征预处理
    medians = train_df[features].median(numeric_only=True).fillna(0.0).to_dict()

    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        return df[features].replace([np.inf, -np.inf], np.nan).fillna(medians)

    # ③ LightGBM 训练
    model = LGBMClassifier(
        random_state=random_seed,
        verbosity=-1,
        **lightgbm_params,
    )
    model.fit(_prepare(train_df), train_df["anomaly_label"].astype(int))

    # ④ Platt 校准
    validation = splits["validation"] if not splits["validation"].empty else train_df
    calibrator = None
    if calibration_method.upper() == "PLATT" and validation["anomaly_label"].nunique() == 2:
        val_raw = model.booster_.predict(_prepare(validation), raw_score=True)
        calibrator = LogisticRegression(random_state=random_seed, solver="lbfgs")
        calibrator.fit(
            np.asarray(val_raw).reshape(-1, 1),
            validation["anomaly_label"].astype(int),
        )

    # ⑤ 构建 SentinelBundle
    bundle = SentinelBundle(
        model=model,
        features=features,
        medians={str(k): float(v) for k, v in medians.items()},
        threshold=0.5,  # 临时值，下面会重选
        sentinel_version=sentinel_version,
        calibrator=calibrator,
        calibration_method=calibration_method if calibrator is not None else "NONE",
    )

    # ⑥ 阈值选择
    val_prob = calibrated_probability(bundle, _prepare(validation))
    threshold, constraint_met = choose_threshold(
        validation["anomaly_label"].astype(int).to_numpy(),
        val_prob,
        fpr_constraint,
    )
    bundle.threshold = threshold

    # ⑦ 评估
    metrics: dict[str, object] = {
        "constraint_met": constraint_met,
        "calibration_method": bundle.calibration_method,
        "split_counts": {name: len(df) for name, df in splits.items()},
        "split_label_counts": {
            name: {str(int(k)): int(v) for k, v in df["anomaly_label"].value_counts().items()}
            for name, df in splits.items()
        },
    }

    for name, df in splits.items():
        if df.empty:
            metrics[name] = None
        else:
            prob = calibrated_probability(bundle, _prepare(df))
            metrics[name] = _rates(
                df["anomaly_label"].astype(int).to_numpy(), prob, threshold
            )

    test_metrics = metrics.get("test") or {}
    metrics["recall_target_met"] = bool(
        float(test_metrics.get("recall", 0.0)) >= min_recall
    )
    metrics["pr_auc_target_met"] = bool(
        float(test_metrics.get("pr_auc") or 0.0) >= min_pr_auc
    )

    # ⑧ 保存模型
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_dir / "sentinel.joblib")

    return bundle, metrics, splits
