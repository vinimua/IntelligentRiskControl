"""v9 受控数据实验：对 credit_model_053 的 Top 重要性特征做重排序噪声漂移。

目的：让自然生命周期链路（监控 → 诊断 → KG 决策 → 自动审批 → 训练）在
真实性能退化下全自动跑通：
- FEATURE_PSI 落在 WARNING 区（0.10-0.25，避开 CRITICAL 防 SEVERE）
- AUC/KS 退化落在 WARNING 区（0.02-0.05，避开 CRITICAL）
- OUTLIER_RATE 增量 < 0.03（不触发 OUTLIER_RATE_SPIKE → 诊断不受数据质量污染）
- 无标签翻转（坏样本率不变）

原理：LightGBM 只依赖特征顺序 → 纯缩放不改变预测；加性/乘性噪声会打乱
排序 → 模型真实退化，且"修复特征分布即恢复性能"（counterfactual SUPPORT）。

用法：
  python scripts/experiment_v9_drift.py --calibrate [--sigma 0.6 --fraction 0.6]
  python scripts/experiment_v9_drift.py --inject --sigma 0.6 --fraction 0.6
  python scripts/experiment_v9_drift.py --restore
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

WINDOWS_ROOT = PROJECT_ROOT / "assets" / "data" / "windows"
MANIFEST_PATH = PROJECT_ROOT / "assets" / "data" / "contracts" / "window_manifest.csv"

TARGET_WINDOWS = ["W2", "W3"]
# 053 的 manifest 声明 healthy_confirmation_boundary=W1 → 监控基线取 W1
# （与 window_loader.resolve_monitoring_baseline_window 口径一致）
BASELINE_WINDOW = "W1"
MODEL_ID = "credit_model_053"
SEED = 20260817
# 双通道扰动设计：
# - SCALE_FEATURES：保序缩放 → 只产生 FEATURE_PSI（分布漂移证据），不改变预测
#   （树模型只看顺序，缩放不退化）→ 监控检测到真实漂移；
# - NOISE_FEATURES：轻量加性噪声（σ_noise × std）→ 只打乱排序，不改变分布
#   → 产生真实的 AUC/KS WARNING 退化（counterfactual/permutation 证据指向该特征）。
SCALE_FEATURES = {"income_level": 0.80, "consumption_level": 0.80}
# 部分样本整数位移：让高重要性特征产生 PSI>0.1 漂移证据（同时轻微重排序）
SHIFT_FEATURES = {"login_fail_count": (1, 0.30), "reg_to_apply_days": (3, 0.40)}
NOISE_FEATURES = {"login_fail_count": 0.5, "max_overdue_days": 0.5, "reg_to_apply_days": 0.5}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """等宽分桶 PSI（与监控层口径一致）。"""
    lo = min(float(np.nanmin(reference)), float(np.nanmin(current)))
    hi = max(float(np.nanmax(reference)), float(np.nanmax(current)))
    edges = np.linspace(lo, hi, bins + 1)
    p = np.histogram(reference, bins=edges)[0] / len(reference)
    q = np.histogram(current, bins=edges)[0] / len(current)
    p = np.clip(p, 1e-6, None)
    q = np.clip(q, 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p)))


def _score(df: pd.DataFrame) -> pd.Series:
    """用 053 Champion（model + calibrator）对窗口打分，返回 y_pred_proba。"""
    from apps.modelops_api.services.monitoring.window_loader import (
        add_apply_time_features,
        load_champion_model,
    )

    model, calibrator, feature_names = load_champion_model(MODEL_ID)
    prepared = add_apply_time_features(df.copy())
    raw = model.predict_proba(prepared[feature_names])[:, 1]
    # champion 管道：概率 → 校准 → 阈值（监控打分链）
    if hasattr(calibrator, "predict"):
        calibrated = calibrator.predict(raw)
    else:
        calibrated = calibrator.transform(raw)
    return pd.Series(calibrated, index=df.index)


def _inject(
    df: pd.DataFrame,
    sigma: float,
    fraction: float,
    seed: int,
    window_factor: float = 1.0,
) -> pd.DataFrame:
    """双通道扰动：保序缩放（PSI 证据）+ 加性噪声（排序退化）。

    sigma 参数只作用于噪声通道（×std 尺度）；缩放系数由 SCALE_FEATURES 常量
    决定（保序 → 不改变预测，只改变分布 → FEATURE_PSI 上升）。
    window_factor：窗口级漂移加速系数（W3 > W2 → 漂移随时间加剧，
    最新 7D 窗口退化最强 —— 监控汇总指标取最新 7D，保证必发）。
    """
    out = df.copy()
    for feature, factor in SCALE_FEATURES.items():
        if feature in out.columns:
            out[feature] = out[feature] * factor
    rng = np.random.default_rng(seed)
    for feature, (offset, shift_fraction) in SHIFT_FEATURES.items():
        if feature in out.columns:
            shift_mask = rng.random(len(out)) < shift_fraction
            out.loc[shift_mask, feature] = out.loc[shift_mask, feature] + offset
    # 时间渐进噪声：每日噪声强度 = base × window_factor × (1 + boost × day_progress)
    # day_progress = (月中第几天) / 31 —— 漂移加速的真实模式
    day_progress = np.clip(
        pd.to_datetime(out["apply_time"]).dt.day.to_numpy() / 31.0, 0.0, 1.0
    )
    mask = rng.random(len(out)) < fraction
    for feature, sigma_rel in NOISE_FEATURES.items():
        if feature not in out.columns:
            continue
        std = float(out[feature].std())
        if std <= 0:
            continue
        sigma_by_row = sigma_rel * window_factor * (1.0 + sigma * day_progress)
        noise = rng.normal(0.0, 1.0, int(mask.sum()))
        out.loc[mask, feature] = (
            out.loc[mask, feature] + noise * (sigma_by_row[mask] * std)
        ).clip(lower=0)
    return out


# 窗口级漂移加速系数：W3 比 W2 更强（漂移随时间加剧）
WINDOW_FACTORS = {"W2": 1.0, "W3": 1.35}


def _outlier_rate_delta(ref: pd.DataFrame, cur: pd.DataFrame) -> float:
    """3×MAD 离群率增量（离群判定用基线分布统计量）。"""
    delta = 0.0
    for feature in sorted(set(SCALE_FEATURES) | set(NOISE_FEATURES)):
        med = ref[feature].median()
        mad = (ref[feature] - med).abs().median()
        if mad == 0:
            continue
        fence = 3.0 * mad
        ref_out = ((ref[feature] - med).abs() > fence).mean()
        cur_out = ((cur[feature] - med).abs() > fence).mean()
        delta = max(delta, float(cur_out - ref_out))
    return delta


def calibrate(sigma: float, fraction: float, natural: bool = False) -> None:
    """不落盘：内存注入 → 用监控层同源函数打分/漂移 → 输出校准指标。

    PSI 用监控层冻结基线分箱边界（continuous_drift + binning_rules_json），
    OUTLIER_RATE 用监控层 calc_outlier_rate 全特征口径 —— 与真实运行完全一致。
    natural=True 时不注入，只报告基线 vs 各当前窗口的自然指标（健康校验）。
    """
    from sklearn.metrics import roc_auc_score

    from apps.modelops_api.services.monitoring.baseline import (
        build_monitoring_baseline,
    )
    from apps.modelops_api.services.monitoring.drift.algorithms import (
        continuous_drift,
    )
    from apps.modelops_api.services.monitoring.metric_calculators import (
        calc_outlier_rate,
    )

    base = pd.read_parquet(WINDOWS_ROOT / BASELINE_WINDOW / "data.parquet")

    def _ks(y, s):
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y, s)
        return float(np.max(tpr - fpr))

    # 监控层同源：基线冻结分箱 + 全特征 OUTLIER_RATE
    feature_names = [
        col for col in base.columns
        if col not in ("sample_id", "apply_time", "is_bad", "y_true",
                       "risk_score", "y_pred_proba")
    ]
    baseline = build_monitoring_baseline(
        base, MODEL_ID, "champion_v1",
        feature_names=feature_names,
        score_column="y_pred_proba", label_column="y_true",
    )
    s_base = _score(base)
    auc_base = roc_auc_score(base["is_bad"], s_base)
    ks_base = _ks(base["is_bad"], s_base)
    print(f"MODEL={MODEL_ID} BASELINE={BASELINE_WINDOW} "
          f"sigma={sigma} fraction={fraction} natural={natural}")
    print(f"  baseline {BASELINE_WINDOW}: AUC={auc_base:.4f} KS={ks_base:.4f} "
          f"rows={len(base)}")

    for window_id in TARGET_WINDOWS:
        cur_orig = pd.read_parquet(WINDOWS_ROOT / window_id / "data.parquet")
        cur = (
            cur_orig
            if natural
            else _inject(
                cur_orig, sigma, fraction, SEED,
                window_factor=WINDOW_FACTORS.get(window_id, 1.0),
            )
        )

        s_cur = _score(cur)
        auc_cur = roc_auc_score(cur["is_bad"], s_cur)
        ks_cur = _ks(cur["is_bad"], s_cur)
        auc_delta = auc_base - auc_cur
        ks_delta = ks_base - ks_cur

        def _zone(v, lo, hi):
            return ("WARNING" if lo <= v < hi else ("CRITICAL" if v >= hi else "ok"))

        psi_vals: dict[str, float] = {}
        for fname in feature_names:
            rule = baseline.binning_rules_json.get(fname) or {}
            edges = rule.get("edges")
            if not edges or fname not in cur.columns:
                continue
            res = continuous_drift(base[fname], cur[fname], edges)
            psi_vals[fname] = float(res["psi"] or 0.0)
        max_psi_feature = max(psi_vals, key=psi_vals.get)
        outlier = calc_outlier_rate(
            base.to_dict("records"), cur.to_dict("records"),
        )

        print(f"  {window_id}: AUC={auc_cur:.4f} KS={ks_cur:.4f} "
              f"AUC_delta={auc_delta:.4f} {_zone(auc_delta, 0.02, 0.05)} "
              f"KS_delta={ks_delta:.4f} {_zone(ks_delta, 0.02, 0.05)}")
        print(f"  {window_id}: FEATURE_PSI max={psi_vals[max_psi_feature]:.4f} "
              f"(feature={max_psi_feature}) "
              f"{_zone(psi_vals[max_psi_feature], 0.10, 0.25)}")
        for feature in sorted(set(SCALE_FEATURES) | set(NOISE_FEATURES) | set(SHIFT_FEATURES)):
            pv = psi_vals.get(feature)
            if pv is not None:
                print(f"  {window_id}: PSI  {feature}: {pv:.4f} "
                      f"{_zone(pv, 0.10, 0.25)}")
        print(f"  {window_id}: OUTLIER_RATE delta={outlier.current_value:.4f} "
              f"({'WARNING' if outlier.current_value and abs(outlier.current_value) >= 0.03 else 'ok'})")


def inject(sigma: float, fraction: float) -> None:
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"window_id": str})
    for window_id in TARGET_WINDOWS:
        parquet_path = WINDOWS_ROOT / window_id / "data.parquet"
        bak_path = parquet_path.with_suffix(parquet_path.suffix + ".v9_bak")
        manifest_bak = MANIFEST_PATH.with_suffix(".csv.v9_bak")
        if bak_path.is_file():
            raise RuntimeError(
                f"{bak_path.name} 已存在 —— 数据已注入过，先 --restore 再注入"
            )
        shutil.copy2(parquet_path, bak_path)
        shutil.copy2(MANIFEST_PATH, manifest_bak)
        print(f"备份完成: {bak_path.name} / {manifest_bak.name}")

        df = pd.read_parquet(parquet_path)
        # 每个窗口独立随机流（seed + 窗口序号），漂移模式一致但样本扰动独立
        df = _inject(
            df, sigma, fraction, SEED + args.seed_offset + int(window_id[1:]),
            window_factor=WINDOW_FACTORS.get(window_id, 1.0),
        )
        bad_count = int(df["is_bad"].sum())
        df.to_parquet(parquet_path, index=False)

        mask = manifest["window_id"] == window_id
        if mask.sum() != 1:
            raise RuntimeError(f"清单必须且只能包含一条窗口 {window_id} 记录")
        manifest.loc[mask, "data_checksum"] = _sha256(parquet_path)
        manifest.loc[mask, "bad_count"] = bad_count
        manifest.loc[mask, "bad_rate"] = round(float(df["is_bad"].mean()), 6)
        print(f"注入完成: {window_id} sigma={sigma} fraction={fraction} "
              f"rows={len(df)} bad_count={bad_count}")
    manifest.to_csv(MANIFEST_PATH, index=False)


def restore() -> None:
    for window_id in TARGET_WINDOWS:
        parquet_path = WINDOWS_ROOT / window_id / "data.parquet"
        bak_path = parquet_path.with_suffix(parquet_path.suffix + ".v9_bak")
        if not bak_path.is_file():
            continue
        shutil.copy2(bak_path, parquet_path)
        bak_path.unlink()
        print(f"已恢复原始数据: {window_id}")
    manifest_bak = MANIFEST_PATH.with_suffix(".csv.v9_bak")
    if manifest_bak.is_file():
        shutil.copy2(manifest_bak, MANIFEST_PATH)
        manifest_bak.unlink()
        print("已恢复原始清单")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--inject", action="store_true")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--natural", action="store_true",
                        help="不注入，只报告自然指标（健康校验）")
    parser.add_argument("--sigma", type=float, default=0.6)
    parser.add_argument("--fraction", type=float, default=0.6)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    if args.restore:
        restore()
    elif args.inject:
        inject(args.sigma, args.fraction)
    else:
        calibrate(args.sigma, args.fraction, natural=args.natural)
    sys.exit(0)
