"""真实窗口数据加载器。

从 assets/data/windows/ 读取 W0-W4 Parquet 文件。
同时支持加载 Champion V1 模型生成预测分（y_pred_proba）。
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# 窗口数据根目录（相对于项目根目录）
_WINDOWS_ROOT = Path(__file__).resolve().parents[4] / "assets" / "data" / "windows"
_CHAMPION_ROOT = Path(__file__).resolve().parents[4] / "assets" / "champion_models"
_MANIFEST_PATH = (
    Path(__file__).resolve().parents[4]
    / "assets" / "data" / "contracts" / "window_manifest.csv"
)

# 窗口 ID 列表
WINDOW_IDS = ["W0", "W1", "W2", "W3", "W4"]


class WindowContractError(ValueError):
    """物理窗口与正式全量数据契约不一致。"""


def validate_window_labels(window_id: str, frame: pd.DataFrame) -> pd.Series:
    """校验监督标签必须为真实观测到的二元 is_bad。"""
    if "is_bad" not in frame.columns:
        raise WindowContractError(f"{window_id} 缺少字段: ['is_bad']")
    if frame["is_bad"].isna().any():
        raise WindowContractError(
            f"{window_id} is_bad contains missing labels; labels must never be imputed"
        )
    labels = pd.to_numeric(frame["is_bad"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise WindowContractError(f"{window_id} is_bad must contain only 0 or 1")
    return labels


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_row(window_id: str) -> pd.Series:
    manifest = pd.read_csv(_MANIFEST_PATH, dtype={"window_id": str})
    matched = manifest.loc[manifest["window_id"] == window_id]
    if len(matched) != 1:
        raise WindowContractError(f"清单必须且只能包含一条 {window_id}")
    return matched.iloc[0]


def validate_window_contract(
    window_id: str, frame: pd.DataFrame, *, path: Path
) -> None:
    """拒绝抽样、截断、过期或日期错位的所谓正式窗口。"""
    row = _manifest_row(window_id)
    expected_rows = int(row["row_count"])
    if len(frame) != expected_rows:
        raise WindowContractError(
            f"{window_id} 行数不一致: manifest={expected_rows}, actual={len(frame)}"
        )
    if (
        str(row.get("sampling_mode", "")).upper() != "FULL_POPULATION"
        or float(row.get("population_fraction", 0.0)) != 1.0
    ):
        raise WindowContractError(f"{window_id} 不是全量正式窗口")
    required = {"sample_id", "apply_time", "is_bad"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise WindowContractError(f"{window_id} 缺少字段: {missing}")
    if frame["sample_id"].isna().any() or not frame["sample_id"].is_unique:
        raise WindowContractError(f"{window_id} sample_id 必须非空且唯一")
    labels = validate_window_labels(window_id, frame)
    apply_time = pd.to_datetime(frame["apply_time"], errors="raise")
    start, end = pd.Timestamp(row["start_date"]), pd.Timestamp(row["end_date"])
    if bool(((apply_time < start) | (apply_time >= end)).any()):
        raise WindowContractError(f"{window_id} 存在窗口日期范围外的样本")
    if _sha256(path) != str(row.get("data_checksum", "")).lower():
        raise WindowContractError(f"{window_id} Parquet 哈希与清单不一致")
    if int(labels.sum()) != int(row["bad_count"]):
        raise WindowContractError(f"{window_id} 坏样本数与清单不一致")


def load_window(window_id: str, *, validate_contract: bool = True) -> pd.DataFrame:
    """加载单个窗口的真实 Parquet 数据。

    Args:
        window_id: "W0" | "W1" | "W2" | "W3" | "W4"

    Returns:
        DataFrame（含 sample_id / apply_time / 特征列 / is_bad）。
    """
    path = _WINDOWS_ROOT / window_id / "data.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Window data not found: {path}")
    frame = pd.read_parquet(path)
    if validate_contract:
        validate_window_contract(window_id, frame, path=path)
    return frame


def load_all_windows() -> dict[str, pd.DataFrame]:
    """加载全部 5 个窗口的数据。"""
    return {wid: load_window(wid) for wid in WINDOW_IDS}


def resolve_monitoring_baseline_window(
    model_id: str,
    champion_version: str = "champion_v1",
) -> str:
    """Return the model-specific monitoring baseline window.

    Formal full-feature champions record their healthy confirmation window in
    ``training_manifest.json``. That window is a better monitoring reference
    than the in-sample fitting window. Models without an explicit declaration
    keep the legacy W0 baseline.
    """
    bundle = _CHAMPION_ROOT / model_id / champion_version
    manifest_path = bundle / "training_manifest.json"
    if not manifest_path.is_file():
        return "W0"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline_window = str(
        manifest.get("monitoring_reference_window")
        or manifest.get("healthy_confirmation_boundary")
        or "W0"
    ).strip()
    return baseline_window if baseline_window in WINDOW_IDS else "W0"


def resolve_monitoring_window_ids(
    model_id: str,
    champion_version: str = "champion_v1",
    *,
    current_window_id: str = "W3",
) -> tuple[str, list[str]]:
    """Resolve baseline and post-baseline windows for monitoring.

    W4 is intentionally excluded here because task-three monitoring and
    iteration must not consume final OOT labels.
    """
    baseline_window_id = resolve_monitoring_baseline_window(
        model_id, champion_version
    )
    allowed_windows = [wid for wid in WINDOW_IDS if wid != "W4"]
    if baseline_window_id not in allowed_windows:
        baseline_window_id = "W0"
    baseline_index = allowed_windows.index(baseline_window_id)
    current_index = allowed_windows.index(current_window_id)
    if current_index <= baseline_index:
        raise WindowContractError(
            f"{model_id} monitoring current window {current_window_id} must be after "
            f"baseline {baseline_window_id}"
        )
    return baseline_window_id, allowed_windows[baseline_index + 1: current_index + 1]


def add_apply_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add Champion V1 time features derived from ``apply_time``."""
    if "apply_time" not in frame.columns:
        return frame
    df = frame.copy()
    ts = pd.to_datetime(df["apply_time"])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    weekday = ts.dt.weekday
    df["apply_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["apply_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["apply_weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    df["apply_weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    df["apply_is_weekend"] = (weekday >= 5).astype(float)
    df["apply_is_night"] = ((ts.dt.hour < 6) | (ts.dt.hour >= 22)).astype(float)
    return df


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
    df = add_apply_time_features(window_df)
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
