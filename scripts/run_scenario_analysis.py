"""场景注入 + 监控 + 因果分析 — 全链路演示。

对每个模型：
1. clean_control → W3 上跑 _monitor_one 得到基准
2. 注入 3 种场景 → 跑 _monitor_one
3. 对比差异 → 判断因果关系 → 生成 KG 观测

运行：
    python scripts/run_scenario_analysis.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from apps.modelops_api.services.monitoring.baseline import build_monitoring_baseline
from apps.modelops_api.services.knowledge_observation_mapper import (
    enrich_scenario_observation,
    validate_mapped_observation,
)
from apps.modelops_api.services.monitoring.pipeline_core import _monitor_one
from apps.modelops_api.services.monitoring.scenarios.injectors import ScenarioFactory
from apps.modelops_api.services.monitoring.window_loader import (
    load_window_with_predictions,
)

# ── 配置 ──
MODELS_TO_TEST = [
    "credit_model_001",  # LogisticRegression
    "credit_model_017",  # XGBoost
    "credit_model_045",  # EBM
]

SCENARIOS = [
    {
        "scenario_name": "covariate_drift",
        "intensity": 0.3,
        "affected_features": ["loan_amount_request"],
    },
    {
        "scenario_name": "missing_rate_anomaly",
        "intensity": 0.3,
        "affected_features": ["income_level", "social_score"],
    },
    {
        "scenario_name": "concept_drift",
        "intensity": 0.3,
        "affected_features": [],
    },
]

CATEGORICAL = {
    "device_type": [0, 1], "education_level": [1, 2, 3, 4, 5],
    "marital_status": [0, 1], "gender": [0, 1],
    "city_tier": [1, 2, 3, 4], "repayment_period": [6, 12, 24, 36],
}


def _f(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    return float(v)


def _observation(**kwargs):
    result = enrich_scenario_observation(kwargs)
    validate_mapped_observation(result)
    return result


def main():
    print("=" * 70)
    print("  场景注入 + 因果分析全链路")
    print(f"  {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    w0_df = load_window_with_predictions("W0", "credit_model_001")
    w3_base = load_window_with_predictions("W3", "credit_model_001")

    all_observations = []

    for model_id in MODELS_TO_TEST:
        print(f"\n{'─' * 60}")
        print(f"  模型: {model_id}")
        print(f"{'─' * 60}")

        # 加载该模型的 W0 + W3 预测
        w0_df = load_window_with_predictions("W0", model_id)
        w3_df = load_window_with_predictions("W3", model_id)

        # 构建基线
        feature_names = [c for c in w0_df.columns
                         if c not in ("sample_id", "apply_time", "is_bad", "y_true",
                                      "risk_score", "y_pred_proba",
                                      "apply_hour_sin", "apply_hour_cos",
                                      "apply_weekday_sin", "apply_weekday_cos",
                                      "apply_is_weekend", "apply_is_night")]
        reference = w0_df.drop(columns=["risk_score"]) if "risk_score" in w0_df.columns else w0_df
        reference_scores = pd.Series(w0_df["y_pred_proba"])

        baseline = build_monitoring_baseline(
            w0_df, model_id=model_id, model_version="champion_v1",
            feature_names=feature_names, categorical_features=CATEGORICAL,
        )
        baseline_profile = pd.read_parquet(Path(baseline.feature_profile_uri))

        # ── 1. Clean Control ──
        print(f"\n  [clean_control] 基准运行...")
        control_result = ScenarioFactory.inject(
            w3_df,
            scenario_config={
                "scenario_name": "clean_control",
                "intensity": 0.0,
                "affected_features": [],
                "event_start_date": str(w3_df["apply_time"].min())[:10],
                "event_end_date": str(w3_df["apply_time"].max())[:10],
            },
            random_seed=42,
        )
        ctrl_source = control_result.dataframe.drop(columns=["risk_score"]) if "risk_score" in control_result.dataframe.columns else control_result.dataframe
        ctrl_preds = control_result.dataframe[["sample_id", "risk_score"]].copy()
        ctrl_perf, _, _ = _monitor_one(
            ctrl_source, ctrl_preds, "SCEN_CTRL",
            baseline, baseline, reference, reference_scores, baseline_profile,
            "SCENARIO", f"scenario-{model_id}-ctrl",
            min_samples=50, min_bad=1,
        )

        ctrl_metrics = {
            k: _f(ctrl_perf.get(k))
            for k in ("auc", "ks", "pr_auc", "brier", "ece", "bad_recall", "bad_rate")
        }
        print(f"     AUC={ctrl_metrics['auc']:.4f}  KS={ctrl_metrics['ks']:.4f}  "
              f"Brier={ctrl_metrics['brier']:.4f}  bad_rate={ctrl_metrics['bad_rate']:.4f}")

        # ── 2. 逐个场景注入 ──
        for sc in SCENARIOS:
            name = sc["scenario_name"]
            affected = sc.get("affected_features", [])
            intensity = sc["intensity"]

            print(f"\n  [{name}] affected={affected} intensity={intensity}")

            try:
                inc_result = ScenarioFactory.inject(
                    w3_df,
                    scenario_config={
                        "scenario_name": name,
                        "intensity": intensity,
                        "affected_features": affected,
                        "event_start_date": str(w3_df["apply_time"].min())[:10],
                        "event_end_date": str(w3_df["apply_time"].max())[:10],
                    },
                    random_seed=42,
                )
            except ValueError as e:
                print(f"     SKIP: {e}")
                continue

            inc_source = inc_result.dataframe.drop(columns=["risk_score"]) if "risk_score" in inc_result.dataframe.columns else inc_result.dataframe
            inc_preds = inc_result.dataframe[["sample_id", "risk_score"]].copy()
            inc_perf, _, inc_drift = _monitor_one(
                inc_source, inc_preds, f"SCEN_{name}",
                baseline, baseline, reference, reference_scores, baseline_profile,
                "SCENARIO", f"scenario-{model_id}-{name}",
                min_samples=50, min_bad=1,
            )

            inc_metrics = {
                k: _f(inc_perf.get(k))
                for k in ("auc", "ks", "pr_auc", "brier", "ece", "bad_recall", "bad_rate")
            }

            # ── 3. 对比分析 ──
            auc_drop = ctrl_metrics["auc"] - inc_metrics["auc"] if ctrl_metrics["auc"] and inc_metrics["auc"] else 0
            ks_drop = ctrl_metrics["ks"] - inc_metrics["ks"] if ctrl_metrics["ks"] and inc_metrics["ks"] else 0
            brier_rise = inc_metrics["brier"] - ctrl_metrics["brier"] if inc_metrics["brier"] and ctrl_metrics["brier"] else 0

            print(f"     AUC={inc_metrics['auc']:.4f} (Δ={auc_drop:+.4f})  "
                  f"KS={inc_metrics['ks']:.4f} (Δ={ks_drop:+.4f})  "
                  f"Brier={inc_metrics['brier']:.4f} (Δ={brier_rise:+.4f})")

            # ── 4. 特征漂移检查 ──
            high_psi_features = []
            if inc_drift:
                for d in inc_drift:
                    psi = _f(d.get("psi"))
                    if psi and psi > 0.1:
                        high_psi_features.append({
                            "feature": d.get("feature_name"),
                            "psi": psi,
                        })
            if high_psi_features:
                print(f"     High PSI features: {[f['feature'] for f in high_psi_features[:5]]}")

            # ── 5. 生成 KG 观测 ──
            observations = []

            # 如果 AUC 下降 > 0.02 → 信号
            if auc_drop > 0.02:
                for fp in high_psi_features:
                    ev_score = min(fp["psi"] / 0.5, 1.0)  # 归一化到 0-1
                    observations.append(_observation(
                        source_entity=fp["feature"],
                        source_type="Feature",
                        relation="MAY_CAUSE",
                        target_entity="AUC_DROP",
                        target_type="Alert",
                        direction="SUPPORT",
                        evidence_score=round(ev_score, 4),
                        data_track="SCENARIO",
                        scenario=name,
                        model_id=model_id,
                        auc_drop=round(auc_drop, 4),
                        feature_psi=round(fp["psi"], 4),
                    ))

            # KS 下降
            if ks_drop > 0.05:
                for fp in high_psi_features:
                    observations.append(_observation(
                        source_entity=fp["feature"],
                        source_type="Feature",
                        relation="MAY_CAUSE",
                        target_entity="KS_DROP",
                        target_type="Alert",
                        direction="SUPPORT",
                        evidence_score=round(min(fp["psi"] / 0.5, 1.0), 4),
                        data_track="SCENARIO",
                        scenario=name,
                        model_id=model_id,
                        ks_drop=round(ks_drop, 4),
                        feature_psi=round(fp["psi"], 4),
                    ))

            # 如果 AUC 没降 → AGAINST 证据
            if abs(auc_drop) <= 0.02:
                for fp in high_psi_features:
                    observations.append(_observation(
                        source_entity=fp["feature"],
                        source_type="Feature",
                        relation="MAY_CAUSE",
                        target_entity="AUC_DROP",
                        target_type="Alert",
                        direction="AGAINST",
                        evidence_score=0.1,
                        data_track="SCENARIO",
                        scenario=name,
                        model_id=model_id,
                        auc_drop=round(auc_drop, 4),
                        feature_psi=round(fp["psi"], 4),
                    ))

            all_observations.extend(observations)
            if observations:
                print(f"     → {len(observations)} KG observations generated")

    # ── 汇总 ──
    print(f"\n{'=' * 70}")
    print(f"  总计生成 {len(all_observations)} 条 KG 观测")
    print(f"{'=' * 70}")

    if all_observations:
        # 按 relation + direction 统计
        from collections import Counter
        summary = Counter()
        for o in all_observations:
            key = f"{o['source_entity']} → {o['target_entity']} [{o['direction']}]"
            summary[key] += 1
        print("\n  Top 因果信号:")
        for k, v in summary.most_common(10):
            print(f"    {k}: {v} 条")

    # 保存结果
    output_path = PROJECT_ROOT / "tmp" / "scenario_observations.json"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(
        json.dumps(all_observations, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n  结果已保存: {output_path}")


if __name__ == "__main__":
    main()
