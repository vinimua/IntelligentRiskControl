"""受控数据实验：向 W3 注入持续漂移 + 坏样本率上升（可恢复）。

目的：让自然生命周期链路的 B1 判定从 SHORT_TERM_7D 升级到
SUSTAINED_30D（中等强度 HIGH 告警，避免 CRITICAL 双窗口触发 SEVERE），
从而自然进入 A7 训练段。

用法：
  python scripts/experiment_sustained_drift.py            # 注入
  python scripts/experiment_sustained_drift.py --restore  # 恢复原始数据
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WINDOWS_ROOT = PROJECT_ROOT / "assets" / "data" / "windows"
MANIFEST_PATH = PROJECT_ROOT / "assets" / "data" / "contracts" / "window_manifest.csv"

TARGET_WINDOW = "W3"
SEED = 20260815


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inject(window_id: str = TARGET_WINDOW) -> None:
    parquet_path = WINDOWS_ROOT / window_id / "data.parquet"
    bak_path = parquet_path.with_suffix(parquet_path.suffix + ".exp_bak")
    manifest_bak = MANIFEST_PATH.with_suffix(".csv.exp_bak")

    if not parquet_path.is_file():
        raise FileNotFoundError(f"Window data not found: {parquet_path}")
    if not bak_path.is_file():
        shutil.copy2(parquet_path, bak_path)
        shutil.copy2(MANIFEST_PATH, manifest_bak)
        print(f"备份完成: {bak_path.name} / {manifest_bak.name}")

    df = pd.read_parquet(parquet_path)
    rng = np.random.default_rng(SEED)
    n = len(df)

    # ── v5-final 平滑漂移（无整数位移 → 无新离群；无标签翻转）──
    # FEATURE_PSI 标定在 WARNING/HIGH 区（0.10-0.25），30D 窗口告警点 ≥3
    if "income_level" in df.columns:
        df["income_level"] = df["income_level"] * 0.80
    if "consumption_level" in df.columns:
        df["consumption_level"] = df["consumption_level"] * 1.15
    if "credit_utilization" in df.columns:
        df["credit_utilization"] = df["credit_utilization"] * 0.90

    bad_count = int(df["is_bad"].sum())
    bad_rate = df["is_bad"].mean()
    print(
        f"注入完成: rows={n} bad_count={bad_count} bad_rate={bad_rate:.4f}"
    )

    # ── 回写 Parquet + 更新清单（校验和与坏样本数）──
    df.to_parquet(parquet_path, index=False)

    manifest = pd.read_csv(MANIFEST_PATH, dtype={"window_id": str})
    mask = manifest["window_id"] == window_id
    if mask.sum() != 1:
        raise RuntimeError("清单必须且只能包含一条目标窗口记录")
    manifest.loc[mask, "data_checksum"] = _sha256(parquet_path)
    manifest.loc[mask, "bad_count"] = bad_count
    manifest.loc[mask, "bad_rate"] = round(bad_rate, 6)
    manifest.to_csv(MANIFEST_PATH, index=False)
    print("清单已更新: data_checksum + bad_count + bad_rate")


def restore(window_id: str = TARGET_WINDOW) -> None:
    parquet_path = WINDOWS_ROOT / window_id / "data.parquet"
    bak_path = parquet_path.with_suffix(parquet_path.suffix + ".exp_bak")
    manifest_bak = MANIFEST_PATH.with_suffix(".csv.exp_bak")

    if not bak_path.is_file():
        print("未找到实验备份，无需恢复")
        return
    shutil.copy2(bak_path, parquet_path)
    shutil.copy2(manifest_bak, MANIFEST_PATH)
    bak_path.unlink()
    manifest_bak.unlink()
    print(f"已恢复原始数据: {window_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    if args.restore:
        restore()
    else:
        inject()
    sys.exit(0)
