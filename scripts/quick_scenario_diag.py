"""一次性脚本：场景注入 → 监控 → 诊断 → 验证 Dashboard 诊断面板。"""
import asyncio, sys, uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from apps.modelops_api.services.monitoring.window_loader import load_window_with_predictions
from apps.modelops_api.services.monitoring.baseline import build_monitoring_baseline
from apps.modelops_api.services.monitoring.pipeline_core import _monitor_one
from apps.modelops_api.services.monitoring.scenarios.injectors import ScenarioFactory
from apps.modelops_api.database import async_session
from apps.modelops_api.repositories.monitoring_repo import MonitoringRepo
from apps.modelops_api.neo4j_db import get_neo4j_driver
from apps.modelops_api.services.knowledge_service import KnowledgeService


async def main():
    model_id = "credit_model_001"
    champion_version = "champion_v1"

    # 1. Load data
    w0_df = load_window_with_predictions("W0", model_id)
    w3_df = load_window_with_predictions("W3", model_id)

    feature_names = [c for c in w0_df.columns
                     if c not in ("sample_id", "apply_time", "is_bad", "y_true",
                                  "risk_score", "y_pred_proba",
                                  "apply_hour_sin", "apply_hour_cos",
                                  "apply_weekday_sin", "apply_weekday_cos",
                                  "apply_is_weekend", "apply_is_night")]
    categorical = {
        "device_type": [0, 1], "education_level": [1, 2, 3, 4, 5],
        "marital_status": [0, 1], "gender": [0, 1],
        "city_tier": [1, 2, 3, 4], "repayment_period": [6, 12, 24, 36],
    }

    reference = w0_df.drop(columns=["risk_score"]) if "risk_score" in w0_df.columns else w0_df
    reference_scores = pd.Series(w0_df["y_pred_proba"])

    async with async_session() as session:
        driver = await get_neo4j_driver()
        knowledge = KnowledgeService(driver)

        from apps.modelops_api.services.monitoring.monitoring_service import MonitoringService
        service = MonitoringService(session, knowledge)

        baseline = service.build_baseline(
            w0_data=w0_df, model_id=model_id, model_version=champion_version,
            feature_names=feature_names, categorical_features=categorical,
        )
        baseline_profile = pd.read_parquet(Path(baseline.feature_profile_uri))

        # 2. Inject covariate_drift
        factory = ScenarioFactory(random_seed=42)
        injector = factory.create("covariate_drift", intensity=0.3,
                                  affected_features=["loan_amount_request"])

        w3_source = w3_df.drop(columns=["risk_score"]) if "risk_score" in w3_df.columns else w3_df
        w3_predictions = w3_df[["sample_id", "risk_score"]].copy()

        injected_source = injector.inject(w3_source)
        print(f"[1] Injected covariate_drift on loan_amount_request")
        print(f"    Original mean: {w3_source['loan_amount_request'].mean():.4f}")
        print(f"    Injected mean: {injected_source['loan_amount_request'].mean():.4f}")

        # 3. Run _monitor_one
        w_perf, w_qual, w_drift = _monitor_one(
            source=injected_source, predictions=w3_predictions,
            monitor_window_id="W3",
            context=baseline, baseline=baseline,
            reference=reference, reference_scores=reference_scores,
            baseline_profile=baseline_profile,
            data_track="SCENARIO", trace_id=str(uuid.uuid4()),
            min_samples=50, min_bad=1,
        )

        # 4. Create monitoring run
        run = await service.repo.create_run(
            model_id=model_id, champion_version=champion_version,
            baseline_window_id="W0", current_window_id="W3",
            data_track="SCENARIO", trace_id=str(uuid.uuid4()),
        )
        monitoring_run_id = run["monitoring_run_id"]
        print(f"[2] Created monitoring run: {monitoring_run_id}")

        # 5. Persist metrics
        alert_count = 0
        for m in w_perf:
            triggered = False
            if m.metric_code in ("AUC", "KS", "PR_AUC") and m.delta is not None and m.delta < -0.02:
                triggered = True

            await service.repo.insert_metric(
                monitoring_run_id=monitoring_run_id, metric_code=m.metric_code,
                object_code=model_id, baseline_value=m.baseline_value,
                current_value=m.current_value, delta=m.delta,
                triggered=triggered,
                metric_detail={"window_id": "W3", "score_type": "calibrated", "category": "core"},
            )

            if triggered:
                await service.repo.insert_alert(
                    monitoring_run_id=monitoring_run_id, metric_id=None,
                    alert_code=f"{m.metric_code}_DROP", severity="HIGH",
                    object_code=model_id, metric_code=m.metric_code,
                    baseline_value=m.baseline_value, current_value=m.current_value,
                    delta=m.delta,
                )
                alert_count += 1
                print(f"    Alert: {m.metric_code}_DROP delta={m.delta:.4f}")

        # Persist drift metrics and check for HIGH_FEATURE_PSI
        for m in w_drift:
            md = m.metric_detail or {}
            await service.repo.insert_metric(
                monitoring_run_id=monitoring_run_id, metric_code=m.metric_code,
                object_code=md.get("feature_name", model_id),
                baseline_value=m.baseline_value, current_value=m.current_value,
                delta=m.delta,
                triggered=m.current_value is not None and m.current_value > 0.25,
                metric_detail={"window_id": "W3", "score_type": "raw", "category": "drift",
                               "feature_name": md.get("feature_name", ""),
                               "psi": md.get("psi"), "max_psi": m.current_value},
            )

        # High PSI alert
        psi_metric = next((m for m in w_drift if m.metric_code == "FEATURE_PSI"), None)
        if psi_metric and psi_metric.current_value and psi_metric.current_value > 0.25:
            await service.repo.insert_alert(
                monitoring_run_id=monitoring_run_id, metric_id=None,
                alert_code="HIGH_FEATURE_PSI", severity="HIGH",
                object_code="FEATURE", metric_code="FEATURE_PSI",
                baseline_value=psi_metric.baseline_value,
                current_value=psi_metric.current_value, delta=psi_metric.delta,
            )
            alert_count += 1
            print(f"    Alert: HIGH_FEATURE_PSI psi={psi_metric.current_value:.4f}")

        # Persist drift rows
        drift_rows = []
        for m in w_drift:
            md = m.metric_detail or {}
            drift_rows.append({
                "window_id": md.get("window_id", "W3"),
                "feature_name": md.get("feature_name", ""),
                "feature_type": md.get("feature_type", "continuous"),
                "psi": md.get("psi"),
                "js_divergence": md.get("js_divergence"),
                "wasserstein_distance": md.get("wasserstein_distance"),
                "ks_statistic": md.get("ks_statistic"),
                "ks_p_value": md.get("ks_p_value"),
                "ks_q_value": md.get("ks_q_value"),
                "data_track": "SCENARIO",
            })
        if drift_rows:
            await service.repo.batch_insert_feature_drift(monitoring_run_id, drift_rows)
        print(f"[3] Persisted {len(drift_rows)} drift rows, {alert_count} alerts")

        await service.repo.complete_run(
            monitoring_run_id=monitoring_run_id, overall_status="COMPLETED",
            alert_count=alert_count, max_alert_severity="HIGH",
        )
        await session.commit()
        print(f"[4] Run completed. MONITORING_RUN_ID={monitoring_run_id}")

    # 6. Trigger diagnosis via API
    import urllib.request, json as jmod
    data = jmod.dumps({"monitoring_run_id": monitoring_run_id}).encode()
    req = urllib.request.Request("http://localhost:8000/api/diagnosis/trigger",
                                 data=data, headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        result = jmod.loads(resp.read())
        print(f"[5] Diagnosis triggered: {jmod.dumps(result, indent=2, ensure_ascii=False)[:500]}")
    except Exception as e:
        print(f"[5] Diagnosis trigger failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
