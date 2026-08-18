"""Sentinel 训练数据生成流水线。

串联：场景注入 → Champion 预测 → 监测计算 → 场景验收 → 特征构建 → 快照保存。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .feature_builder import build_monitor_feature_vector, select_canonical_sentinel_rows
from .feature_schema import SENTINEL_FEATURE_SCHEMA_VERSION, compute_schema_hash

PROJECT_ROOT = Path(__file__).resolve().parents[5]
ASSETS_ROOT = PROJECT_ROOT / "assets"
CONFIG_ROOT = ASSETS_ROOT / "configs"


def stable_bucket(value: object, count: int) -> int:
    digest = hashlib.sha256(str(value).encode()).hexdigest()
    return int(digest[:8], 16) % count


def assign_source_cohorts(frame: pd.DataFrame, cohort_count: int = 5) -> pd.DataFrame:
    """按 sample_id 稳定哈希分配 source_cohort_id。"""
    result = frame.copy()
    if "sample_id" in result.columns:
        result["source_cohort_id"] = result["sample_id"].apply(
            lambda v: f"cohort_{stable_bucket(v, cohort_count):02d}"
        )
    else:
        result["source_cohort_id"] = "cohort_00"
    return result


def build_sentinel_training_dataset(
    model_id: str = "credit_model_001",
    champion_version: str = "champion_v1",
    base_window_ids: list[str] | None = None,
    scenario_ids: list[str] | None = None,
    intensities: list[float] | None = None,
    random_seed: int = 2026,
) -> tuple[pd.DataFrame, dict]:
    """生成 Sentinel 训练数据集。

    Returns:
        (training_df, summary): training_df 含 anomaly_label，summary 含统计信息。
    """
    rng = np.random.default_rng(random_seed)
    base_window_ids = base_window_ids or ["W1", "W2", "W3"]
    intensities = intensities or [0.2, 0.4, 0.6]

    # ── 加载场景配置 ──
    with open(CONFIG_ROOT / "scenarios.yaml", encoding="utf-8") as f:
        scenarios_cfg = yaml.safe_load(f)

    all_scenarios = scenarios_cfg.get("scenarios", [])
    scenario_map: dict[str, dict] = {s["scenario_name"]: s for s in all_scenarios}
    if scenario_ids:
        scenario_map = {k: v for k, v in scenario_map.items() if k in scenario_ids}
    # acceptance 阈值在配置文件顶层
    acceptance_thresholds = scenarios_cfg.get("acceptance", {})

    # ── 加载窗口数据 ──
    from apps.modelops_api.services.monitoring.window_loader import load_window_with_predictions

    window_frames: dict[str, pd.DataFrame] = {}
    for wid in base_window_ids:
        window_frames[wid] = load_window_with_predictions(wid, model_id=model_id)
        window_frames[wid] = assign_source_cohorts(window_frames[wid])

    # ── 构建基线 ──
    w0_df = load_window_with_predictions("W0", model_id=model_id)
    from apps.modelops_api.services.monitoring.baseline import MonitoringBaseline, build_monitoring_baseline

    feature_names = [
        c for c in w0_df.columns
        if c not in ("sample_id", "apply_time", "is_bad", "y_true", "risk_score", "y_pred_proba",
                     "apply_hour_sin", "apply_hour_cos", "apply_weekday_sin", "apply_weekday_cos",
                     "apply_is_weekend", "apply_is_night")
    ]
    baseline = build_monitoring_baseline(
        w0_data=w0_df, model_id=model_id, model_version=champion_version,
        feature_names=feature_names,
    )

    # ── 遍历场景生成 ──
    from apps.modelops_api.services.monitoring.drift.algorithms import (
        compute_performance_metrics, continuous_drift, feature_quality,
    )
    from apps.modelops_api.services.monitoring.drift.output_monitor import output_metrics
    from apps.modelops_api.services.monitoring.scenarios.injectors import ScenarioFactory
    from apps.modelops_api.services.monitoring.scenario_acceptance import evaluate_scenario_acceptance

    all_perf: list[dict] = []
    all_qual: list[dict] = []
    all_drift: list[dict] = []
    all_detector: list[dict] = []

    stats = {
        "total_instances": 0,
        "accepted_normal": 0,
        "accepted_anomaly": 0,
        "uncertain": 0,
        "errors": 0,
    }

    for scenario_id, scenario_cfg in scenario_map.items():
        for base_window_id in base_window_ids:
            window_df = window_frames[base_window_id]
            # 取一个不重叠的 cohort 子集
            cohort_ids = sorted(window_df["source_cohort_id"].unique())
            if not cohort_ids:
                continue

            for cohort_id in cohort_ids:
                cohort_df = window_df[window_df["source_cohort_id"] == cohort_id].copy()
                if len(cohort_df) < 100:
                    continue

                for intensity in intensities:
                    seed = rng.integers(0, 10_000_000)
                    scenario_instance_id = f"{scenario_id}_{base_window_id}_{cohort_id}_i{intensity}_s{seed}"

                    try:
                        # 注入异常
                        # 使用 cohort 的日期范围作为事件窗口
                        cohort_dates = pd.to_datetime(cohort_df["apply_time"])
                        inj_cfg = {
                            "scenario_name": scenario_id,
                            "intensity": intensity,
                            "affected_features": scenario_cfg.get("affected_features", []),
                            "event_start_date": str(cohort_dates.min().date()),
                            "event_end_date": str(cohort_dates.max().date()),
                        }
                        injected_result = ScenarioFactory.inject(
                            cohort_df.copy(), inj_cfg, seed,
                        )
                        injected_df = injected_result.dataframe

                        # 对照组（未修改副本）
                        control_df = cohort_df.copy()

                        # 分别执行监测
                        scenario_perf, scenario_qual, scenario_drift = _run_monitoring(
                            injected_df, w0_df, baseline, feature_names,
                        )
                        control_perf, control_qual, control_drift = _run_monitoring(
                            control_df, w0_df, baseline, feature_names,
                        )

                        # 场景验收
                        acceptance = evaluate_scenario_acceptance(
                            scenario_perf,
                            scenario_qual,
                            scenario_drift,
                            control_performance=control_perf,
                            control_quality=control_qual,
                            control_drift=control_drift,
                            baseline_performance=baseline.performance_reference_json,
                            scenario_name=scenario_id,
                            anomaly_scope=scenario_cfg.get("anomaly_scope", "FEATURE"),
                            thresholds=acceptance_thresholds,
                            scenario_category=scenario_cfg.get("scenario_category", "DRIFT"),
                            drift_type=scenario_cfg.get("drift_type", "FEATURE_DRIFT"),
                        )

                        status = acceptance["scenario_acceptance_status"]
                        label = acceptance["anomaly_label"]

                        if status == "UNCERTAIN":
                            stats["uncertain"] += 1
                            continue

                        stats["accepted_anomaly" if label == 1 else "accepted_normal"] += 1
                        stats["total_instances"] += 1

                        # 标签写入所有行（异常场景实例）
                        for frame in (scenario_perf, scenario_qual, scenario_drift):
                            frame["scenario_instance_id"] = scenario_instance_id
                            frame["source_cohort_id"] = cohort_id
                            frame["scenario_acceptance_status"] = status
                            frame["anomaly_label"] = int(label)

                        all_perf.append(scenario_perf)
                        all_qual.append(scenario_qual)
                        all_drift.append(scenario_drift)

                        # 对照组作为 NORMAL 实例（label=0）
                        control_instance_id = f"control_{base_window_id}_{cohort_id}_i{intensity}_s{seed}"
                        for frame in (control_perf, control_qual, control_drift):
                            frame["scenario_instance_id"] = control_instance_id
                            frame["source_cohort_id"] = cohort_id
                            frame["scenario_acceptance_status"] = "ACCEPTED_NORMAL"
                            frame["anomaly_label"] = 0

                        all_perf.append(control_perf)
                        all_qual.append(control_qual)
                        all_drift.append(control_drift)
                        stats["accepted_normal"] += 1
                        stats["total_instances"] += 1

                    except Exception:
                        stats["errors"] += 1

    if not all_perf:
        raise ValueError("No valid training instances generated")

    # ── 合并 + 构建特征 ──
    perf_df = pd.concat(all_perf, ignore_index=True)
    qual_df = pd.concat(all_qual, ignore_index=True) if all_qual else pd.DataFrame()
    drift_df = pd.concat(all_drift, ignore_index=True) if all_drift else pd.DataFrame()
    detector_df = pd.DataFrame()

    feature_df = build_monitor_feature_vector(perf_df, qual_df, drift_df, detector_df)
    feature_df = select_canonical_sentinel_rows(feature_df)

    # 只保留 ACCEPTED
    training_df = feature_df[feature_df["anomaly_label"].isin([0, 1])].copy()
    training_df["anomaly_label"] = training_df["anomaly_label"].astype(int)

    stats["row_count"] = len(training_df)
    stats["normal_count"] = int((training_df["anomaly_label"] == 0).sum())
    stats["anomaly_count"] = int((training_df["anomaly_label"] == 1).sum())

    return training_df, stats


def _run_monitoring(
    df: pd.DataFrame,
    w0_df: pd.DataFrame,
    baseline,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """对单个场景窗口执行一次完整监测计算。"""
    from apps.modelops_api.services.monitoring.drift.algorithms import (
        compute_performance_metrics, continuous_drift, categorical_drift, feature_quality,
    )
    from apps.modelops_api.services.monitoring.drift.output_monitor import output_metrics
    from apps.modelops_api.services.monitoring.rolling import iter_rolling_windows

    all_data = df.sort_values("apply_time")
    reference_scores = w0_df["y_pred_proba"]

    perf_rows, qual_rows, drift_rows = [], [], []

    # 训练数据生成使用 step_days=7（非重叠窗口），大幅减少计算量
    # 每实例产生 ~5 个窗口，足够特征聚合；正式监控推理仍用 step_days=1
    for start, end, window in iter_rolling_windows(all_data, window_days=7, step_days=7):
        window_id = f"7D_{start:%Y%m%d}_{end:%Y%m%d}"
        sample_count = len(window)
        bad_count = int(window["is_bad"].sum()) if "is_bad" in window.columns else 0

        label_ready = (
            "is_bad" in window.columns and "y_pred_proba" in window.columns
            and sample_count >= 50 and bad_count >= 1
        )
        if label_ready:
            perf = compute_performance_metrics(window["is_bad"], window["y_pred_proba"])
        else:
            perf = {k: None for k in ("auc", "ks", "pr_auc", "brier", "ece", "bad_recall")}

        out = output_metrics(window.get("y_pred_proba", pd.Series()), reference_scores, baseline.score_edges)

        common = {
            "monitor_window_id": window_id,
            "window_start": start, "window_end": end,
            "window_days": 7,
            "sample_count": sample_count, "bad_count": bad_count,
            "data_track": "SCENARIO",
            "model_id": "credit_model_001",
            "model_version": "champion_v1",
            "baseline_id": baseline.baseline_id,
            "baseline_version": baseline.baseline_version,
        }
        perf_rows.append({**common, **perf, **out})

        p_positions: list[int] = []
        p_values: list[float | None] = []
        for fname in feature_names:
            rule = baseline.binning_rules_json.get(fname)
            if rule is None or fname not in window.columns or fname not in w0_df.columns:
                continue
            if fname in getattr(baseline, "feature_profiles", {}):
                quality = feature_quality(
                    window[fname], pd.Series(baseline.feature_profiles[fname]), rule["feature_type"],
                )
                qual_rows.append({**common, "feature_name": fname, **quality})
            if rule["feature_type"] == "continuous":
                drift = continuous_drift(w0_df[fname], window[fname], rule["edges"])
                row = {"feature_name": fname, **drift, "ks_q_value": None,
                       "category_share_change": None, "unknown_category_rate": 0.0}
                if row.get("ks_p_value") is not None:
                    p_positions.append(len(drift_rows))
                    p_values.append(row["ks_p_value"])
            else:
                drift = categorical_drift(w0_df[fname], window[fname], rule["categories"])
                row = {"feature_name": fname, **drift, "ks_q_value": None,
                       "wasserstein_distance": None, "ks_statistic": None}
            drift_rows.append({**common, **row})

        from apps.modelops_api.services.monitoring.drift.algorithms import benjamini_hochberg
        for pos, q_val in zip(p_positions, benjamini_hochberg(p_values)):
            drift_rows[pos]["ks_q_value"] = q_val

    return (
        pd.DataFrame(perf_rows),
        pd.DataFrame(qual_rows),
        pd.DataFrame(drift_rows),
    )


def save_training_snapshot(
    training_df: pd.DataFrame,
    stats: dict,
    model_id: str,
) -> str:
    """保存训练数据集为本地的 Parquet 快照，返回存储路径。"""
    output_dir = ASSETS_ROOT / "sentinel_training" / model_id
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_id = str(uuid.uuid4())
    parquet_path = output_dir / f"{snapshot_id}.parquet"
    training_df.to_parquet(parquet_path, index=False)

    # manifest
    manifest = {
        "dataset_id": snapshot_id,
        "model_id": model_id,
        "schema_version": SENTINEL_FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": compute_schema_hash(),
        "row_count": len(training_df),
        "scenario_count": stats.get("total_instances", 0),
        "accepted_normal_count": stats.get("accepted_normal", 0),
        "accepted_anomaly_count": stats.get("accepted_anomaly", 0),
        **stats,
    }
    manifest_path = output_dir / f"{snapshot_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return str(parquet_path)
