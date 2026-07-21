"""真实窗口数据加载器。

从 assets/data/windows/ 读取 W0-W4 Parquet 文件。
同时支持加载 Champion V1 模型生成预测分（y_pred_proba）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# 窗口数据根目录（相对于项目根目录）
_WINDOWS_ROOT = Path(__file__).resolve().parents[4] / "assets" / "data" / "windows"
_CHAMPION_ROOT = Path(__file__).resolve().parents[4] / "assets" / "champion_models"

# 窗口 ID 列表
WINDOW_IDS = ["W0", "W1", "W2", "W3", "W4"]


def load_window(window_id: str) -> pd.DataFrame:
    """加载单个窗口的真实 Parquet 数据。

    Args:
        window_id: "W0" | "W1" | "W2" | "W3" | "W4"

    Returns:
        DataFrame（含 sample_id / apply_time / 特征列 / is_bad）。
    """
    path = _WINDOWS_ROOT / window_id / "data.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Window data not found: {path}")
    return pd.read_parquet(path)


def load_all_windows() -> dict[str, pd.DataFrame]:
    """加载全部 5 个窗口的数据。"""
    return {wid: load_window(wid) for wid in WINDOW_IDS}


def load_champion_model(model_id: str = "credit_model_001"):
    """加载一个 Champion V1 模型。

    Returns:
        (model, calibrator, feature_names): sklearn Pipeline + IsotonicCalibrator + 特征名列表。
    """
    import joblib

    bundle = _CHAMPION_ROOT / model_id / "champion_v1"
    if not bundle.is_dir():
        raise FileNotFoundError(f"Champion bundle not found: {bundle}")

    model = joblib.load(bundle / "model.joblib")
    calibrator = joblib.load(bundle / "calibrator.joblib")

    schema = json.loads((bundle / "feature_schema.json").read_text(encoding="utf-8"))
    feature_names = schema["ordered_features"]

    return model, calibrator, feature_names


def predict_on_window(
    window_df: pd.DataFrame,
    model_id: str = "credit_model_001",
) -> pd.DataFrame:
    """用 Champion V1 模型对窗口数据做预测，添加 y_pred_proba 列。

    Args:
        window_df: 窗口数据 DataFrame（需含特征列）。
        model_id: Champion 模型 ID。

    Returns:
        原 DataFrame 加上 risk_score 和 y_pred_proba 列。
    """
    model, calibrator, feature_names = load_champion_model(model_id)

    # Champion 模型需要的时间特征工程
    df = window_df.copy()
    ts = pd.to_datetime(df["apply_time"])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    weekday = ts.dt.weekday
    df["apply_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["apply_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["apply_weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    df["apply_weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    df["apply_is_weekend"] = (weekday >= 5).astype(float)
    df["apply_is_night"] = ((ts.dt.hour < 6) | (ts.dt.hour >= 22)).astype(float)

    # 准备特征
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(
            f"Window data missing required features for {model_id}: {missing}"
        )

    X = df[feature_names].copy()
    # 处理缺失值和无穷值
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True).fillna(0))

    raw_proba = model.predict_proba(X)[:, 1]
    calibrated = calibrator.predict(raw_proba)

    df["risk_score"] = raw_proba       # 原始概率 → 排序指标（AUC/KS/PR_AUC）
    df["y_pred_proba"] = calibrated    # 校准概率 → 校准指标（Brier/ECE/SCORE_PSI）
    df["y_true"] = df["is_bad"]  # 映射到计算器期望的列名
    return df


def load_window_with_predictions(
    window_id: str,
    model_id: str = "credit_model_001",
) -> pd.DataFrame:
    """加载窗口数据并附加模型预测分。

    一步完成：读 Parquet → 模型预测 → 返回含 y_pred_proba 的 DataFrame。
    """
    df = load_window(window_id)
    return predict_on_window(df, model_id)
