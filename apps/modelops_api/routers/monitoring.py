"""监控 API 路由 — 任务一：模型监控"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from pydantic import BaseModel, Field

from ..core.exceptions import NotFoundError, request_trace_id
from ..database import get_db
from ..neo4j_db import get_neo4j_driver
from ..repositories.monitoring_repo import MonitoringRepo
from ..services.knowledge_service import KnowledgeService
from ..services.monitoring.monitoring_service import MonitoringService
from ..services.monitoring.window_loader import load_window_with_predictions, load_window
from ..services.monitoring.metrics_registry import MetricResult
from packages.models.common.enums import AvailabilityStatus, DataTrack, ObjectType, RuleType, Severity
from packages.models.monitoring.alert_context import AlertContext, AlertDetail

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


class RunMonitoringRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=100)
    champion_version: str = Field(min_length=1, max_length=100)
    baseline_window_id: str = ""
    current_window_id: str = ""
    data_track: str = "NATURAL"


class RunDetailedMonitoringRequest(BaseModel):
    """V2 完整模式监控请求（含分箱规则和特征列表）。"""

    model_id: str = Field(min_length=1, max_length=100)
    champion_version: str = Field(min_length=1, max_length=100)
    baseline_window_id: str = ""
    current_window_id: str = ""
    data_track: str = "NATURAL"
    binning_rules: dict | None = None
    feature_names: list[str] | None = None
    min_samples: int = 2000
    min_bad: int = 50


class RunParallelCycleRequest(BaseModel):
    """多模型并行监控周期请求。"""

    model_ids: list[str] = Field(min_length=1, max_length=50)
    champion_versions: dict[str, str] = Field(default_factory=dict)
    max_concurrency: int = Field(default=30, ge=1, le=50)


def _envelope(request: Request, data, message: str = "success") -> dict:
    return {
        "success": True,
        "code": "OK",
        "message": message,
        "data": data,
        "trace_id": request_trace_id(request),
    }


@router.get("/runs")
async def list_runs(
    request: Request,
    model_id: str | None = Query(None, description="按模型筛选"),
    limit: int = Query(50, description="返回数量上限"),
    db: AsyncSession = Depends(get_db),
):
    """列出最近的监控运行。"""
    repo = MonitoringRepo(db)
    runs = await repo.list_runs(model_id=model_id, limit=limit)
    return _envelope(request, {"items": runs})


@router.get("/runs/{monitoring_run_id}")
async def get_run(
    monitoring_run_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """查看一次监控运行的详情。"""
    repo = MonitoringRepo(db)
    run = await repo.get_run(monitoring_run_id)
    if not run:
        raise NotFoundError(f"监控运行 {monitoring_run_id} 不存在")
    return _envelope(request, run)


@router.get("/runs/{monitoring_run_id}/metrics")
async def get_metrics(
    monitoring_run_id: str,
    request: Request,
    category: str | None = Query(None, description="按 category 过滤: core|distribution|drift|quality|aggregate|meta"),
    db: AsyncSession = Depends(get_db),
):
    """查看一次监控运行的全部指标。支持 ?category=drift 按分类过滤。"""
    repo = MonitoringRepo(db)
    run = await repo.get_run(monitoring_run_id)
    if not run:
        raise NotFoundError(f"监控运行 {monitoring_run_id} 不存在")
    all_metrics = await repo.get_metrics(monitoring_run_id)

    if category:
        filtered = [m for m in all_metrics
                    if (m.get("metric_detail") or {}).get("category") == category]
        return _envelope(request, {"items": filtered})

    return _envelope(request, {"items": all_metrics})


@router.get("/runs/{monitoring_run_id}/alerts")
async def get_alerts(
    monitoring_run_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """查看一次监控运行的全部告警。"""
    repo = MonitoringRepo(db)
    run = await repo.get_run(monitoring_run_id)
    if not run:
        raise NotFoundError(f"监控运行 {monitoring_run_id} 不存在")
    alerts = await repo.get_alerts(monitoring_run_id)
    return _envelope(request, {"items": alerts})


# ═══════════════════════════════════════════════════════════════
# 指标持久化辅助函数
# ═══════════════════════════════════════════════════════════════

RANKING_METRICS = [("AUC", "auc"), ("KS", "ks"), ("PR_AUC", "pr_auc"), ("BAD_RECALL", "bad_recall")]
CALIBRATION_METRICS = [("BRIER", "brier"), ("ECE", "ece")]


def _safe_float(val) -> float | None:
    """安全转 float，处理 None / NaN / inf。"""
    if val is None:
        return None
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (ValueError, TypeError):
        return None


async def _persist_window_metrics(
    service: MonitoringService,
    monitoring_run_id: str,
    window_id: str,
    w_df: pd.DataFrame,
    w0_df: pd.DataFrame,
    w_perf: dict,
    w_qual: list[dict],
    w_drift: list[dict],
    baseline,
    knowledge,
) -> tuple[list[MetricResult], list[AlertDetail]]:
    """持久化单个窗口的全部指标（performance + quality + drift）。

    Returns:
        (all_metrics, triggered_alerts)
    """

    from ..services.monitoring.drift.algorithms import compute_performance_metrics
    from ..services.monitoring.drift.output_monitor import output_metrics

    sample_n = w_perf.get("sample_count", len(w_df))
    bad_n = w_perf.get("bad_count", int(w_df["is_bad"].sum()))
    detail = {"window_id": window_id, "sample_count": sample_n, "bad_count": bad_n}

    all_metrics: list[MetricResult] = []
    triggered_alerts: list[AlertDetail] = []

    # ═══════════════════════════════════════════════════════════
    # 1. 模型级性能指标（CORE）— 10 个
    # ═══════════════════════════════════════════════════════════

    # 1a. 排序能力（raw proba = risk_score）
    for code, key in RANKING_METRICS:
        cur_val = _safe_float(w_perf.get(key))
        raw_base = baseline.raw_performance_reference_json.get(key) if baseline.raw_performance_reference_json else None
        base_val = _safe_float(raw_base if raw_base is not None else baseline.performance_reference_json.get(key))
        delta_val = (cur_val - base_val) if cur_val is not None and base_val is not None else None
        mr = MetricResult(metric_code=code, current_value=cur_val, baseline_value=base_val,
                          delta=delta_val,
                          metric_detail={**detail, "category": "core", "score_type": "raw", "computed": True})
        all_metrics.append(mr)
        await service._persist_metric(monitoring_run_id, mr)

    # 1b. 校准指标（calibrated proba = y_pred_proba）
    cal_perf = compute_performance_metrics(w_df["is_bad"], w_df["y_pred_proba"])
    for code, key in CALIBRATION_METRICS:
        cur_val = _safe_float(cal_perf.get(key))
        base_val = _safe_float(baseline.performance_reference_json.get(key))
        delta_val = (cur_val - base_val) if cur_val is not None and base_val is not None else None
        mr = MetricResult(metric_code=code, current_value=cur_val, baseline_value=base_val,
                          delta=delta_val,
                          metric_detail={**detail, "category": "core", "score_type": "calibrated", "computed": True})
        all_metrics.append(mr)
        await service._persist_metric(monitoring_run_id, mr)

    # 1c. 预测分布（calibrated proba）— 4 个
    cal_out = output_metrics(w_df["y_pred_proba"], w0_df["y_pred_proba"], baseline.score_edges)
    for code, key in [("PREDICTION_STD", "prediction_std"),
                       ("PREDICTION_MIN", "prediction_min"),
                       ("PREDICTION_MAX", "prediction_max"),
                       ("SCORE_PSI", "prediction_psi")]:
        cur_val = _safe_float(cal_out.get(key))
        mr = MetricResult(metric_code=code, current_value=cur_val,
                          metric_detail={**detail, "category": "distribution", "score_type": "calibrated",
                                         "computed": True})
        all_metrics.append(mr)
        await service._persist_metric(monitoring_run_id, mr)

    pm = _safe_float(w_df["y_pred_proba"].mean())
    base_pm = _safe_float(w0_df["y_pred_proba"].mean())
    mr = MetricResult(metric_code="PREDICTION_MEAN", current_value=pm, baseline_value=base_pm,
                      delta=(pm - base_pm) if pm is not None and base_pm is not None else None,
                      metric_detail={**detail, "category": "distribution", "score_type": "calibrated",
                                     "computed": True})
    all_metrics.append(mr)
    await service._persist_metric(monitoring_run_id, mr)

    # 1d. 标签统计 — 4 个
    cur_br = _safe_float(w_perf.get("bad_rate"))
    base_br = _safe_float(baseline.performance_reference_json.get("bad_rate"))
    mr = MetricResult(metric_code="BAD_RATE", current_value=cur_br, baseline_value=base_br,
                      delta=(cur_br - base_br) if cur_br is not None and base_br is not None else None,
                      metric_detail={**detail, "category": "core", "score_type": "label", "computed": True})
    all_metrics.append(mr)
    await service._persist_metric(monitoring_run_id, mr)

    br_delta = _safe_float(w_perf.get("bad_rate_delta"))
    mr = MetricResult(metric_code="BAD_RATE_DELTA", current_value=br_delta,
                      metric_detail={**detail, "category": "core", "score_type": "label", "computed": True})
    all_metrics.append(mr)
    await service._persist_metric(monitoring_run_id, mr)

    perf_drop = _safe_float(w_perf.get("performance_drop_max"))
    mr = MetricResult(metric_code="PERFORMANCE_DROP_MAX", current_value=perf_drop,
                      metric_detail={**detail, "category": "core", "score_type": "derived", "computed": True})
    all_metrics.append(mr)
    await service._persist_metric(monitoring_run_id, mr)

    status_val = w_perf.get("status", "READY")
    mr = MetricResult(metric_code="MONITOR_STATUS", current_value=None,
                      metric_detail={**detail, "category": "core", "computed": False,
                                     "status": status_val, "reason": w_perf.get("reason")})
    all_metrics.append(mr)
    await service._persist_metric(monitoring_run_id, mr)

    # 1e. 元数据字段
    mr = MetricResult(metric_code="SAMPLE_SIZE",
                      current_value=float(len(w_df)),
                      metric_detail={"window_id": window_id, "category": "meta", "computed": False,
                                     "bad_count": int(w_df["is_bad"].sum())})
    all_metrics.append(mr)
    await service._persist_metric(monitoring_run_id, mr)

    w0_cols = set(w0_df.columns)
    w_cols = set(w_df.columns)
    missing = w0_cols - w_cols
    new_cols = w_cols - w0_cols
    schema_ok = len(missing) == 0 and len(new_cols) == 0
    mr = MetricResult(metric_code="SCHEMA_CONSISTENCY",
                      current_value=1.0 if schema_ok else 0.0,
                      metric_detail={"window_id": window_id, "category": "meta", "computed": False,
                                     "missing": list(missing), "new": list(new_cols)})
    all_metrics.append(mr)
    await service._persist_metric(monitoring_run_id, mr)

    # ═══════════════════════════════════════════════════════════
    # 2. 特征质量（×34 特征）— 每个 8 个指标
    # ═══════════════════════════════════════════════════════════
    for q in w_qual:
        fname = q.get("feature_name", "?")
        fd = {"window_id": window_id, "category": "quality", "feature_name": fname, "computed": True}
        for key in ("missing_rate", "missing_rate_delta", "outlier_rate", "outlier_rate_delta",
                     "dq_score", "dq_flag", "default_value_rate", "range_violation_rate",
                     "unknown_category_rate"):
            val = _safe_float(q.get(key)) if key != "dq_flag" else q.get(key)
            if val is not None or key == "dq_flag":
                mr = MetricResult(
                    metric_code=f"Q_{key.upper()}", current_value=val if not isinstance(val, str) else None,
                    metric_detail={**fd, "value_str": val if isinstance(val, str) else None})
                all_metrics.append(mr)
                await service._persist_metric(monitoring_run_id, mr)

    # ═══════════════════════════════════════════════════════════
    # 3. 特征漂移（×34 特征）— 每个 8 个指标
    # ═══════════════════════════════════════════════════════════
    for d in w_drift:
        fname = d.get("feature_name", "?")
        ftype = d.get("feature_type", "continuous")
        fd = {"window_id": window_id, "category": "drift", "feature_name": fname,
              "feature_type": ftype, "computed": True}
        for key in ("psi", "js_divergence", "wasserstein_distance", "ks_statistic",
                     "ks_p_value", "ks_q_value", "category_share_change", "unknown_category_rate"):
            val = _safe_float(d.get(key))
            if val is not None:
                mr = MetricResult(metric_code=f"D_{key.upper()}", current_value=val,
                                  metric_detail=fd)
                all_metrics.append(mr)
                await service._persist_metric(monitoring_run_id, mr)

    # ═══════════════════════════════════════════════════════════
    # 4. FEATURE_PSI（聚合）
    # ═══════════════════════════════════════════════════════════
    if len(w_drift) > 0:
        drift_df_w = pd.DataFrame(w_drift)
        psi_vals = drift_df_w["psi"].dropna()
        mean_psi = _safe_float(psi_vals.mean()) if len(psi_vals) > 0 else None
        max_psi = _safe_float(psi_vals.max()) if len(psi_vals) > 0 else None
        mr = MetricResult(metric_code="FEATURE_PSI", current_value=mean_psi,
                          metric_detail={"window_id": window_id, "category": "aggregate",
                                         "max_psi": max_psi, "score_type": "feature",
                                         "computed": True, "n_features": len(psi_vals)})
        all_metrics.append(mr)
        metric_id = await service._persist_metric(monitoring_run_id, mr)

        rule = service.rules.get("FEATURE_PSI")
        if rule and max_psi and max_psi > rule.warning_threshold:
            triggered, sev = rule.evaluate(max_psi, max_psi)
            if triggered and sev:
                alert_type = await knowledge.resolve_alert("FEATURE_PSI", sev)
                code = alert_type.alert_code if alert_type else "FEATURE_PSI_HIGH"
                detail_alert = AlertDetail(
                    alert_id=str(uuid.uuid4()), alert_code=code, severity=sev,
                    object_type=ObjectType.FEATURE, object_code="ALL",
                    metric_code="FEATURE_PSI", metric_version="V2",
                    current_value=max_psi, baseline_value=mean_psi,
                    delta=max_psi, threshold=rule.critical_threshold,
                    rule_type=RuleType.SHIFT_THRESHOLD,
                    threshold_rule_id=rule.rule_id, threshold_rule_version=rule.rule_version,
                    availability_status=AvailabilityStatus.AVAILABLE,
                    metric_detail={"max_psi": max_psi, "mean_psi": mean_psi, "window_id": window_id},
                    created_at=datetime.now(timezone.utc),
                )
                triggered_alerts.append(detail_alert)
                await service._persist_alert(monitoring_run_id, metric_id, mr, detail_alert)

    return all_metrics, triggered_alerts


# ═══════════════════════════════════════════════════════════════
# 核心端点
# ═══════════════════════════════════════════════════════════════

@router.post("/runs")
async def trigger_run(
    request: Request,
    body: RunMonitoringRequest,
    db: AsyncSession = Depends(get_db),
):
    """触发一次完整监控管道 — 每个 W1/W2/W3 窗口独立运行 _monitor_one。

    W0 做基线，W1/W2/W3 各自作为独立监控窗口。
    指标按窗口分别持久化，Dashboard 按窗口分组展示。
    """
    from pathlib import Path
    from ..services.monitoring.pipeline_core import _monitor_one

    driver = await get_neo4j_driver()
    knowledge = KnowledgeService(driver)
    service = MonitoringService(db, knowledge)

    w0_df = load_window_with_predictions("W0", body.model_id)
    w1_df = load_window_with_predictions("W1", body.model_id)
    w2_df = load_window_with_predictions("W2", body.model_id)
    w3_df = load_window_with_predictions("W3", body.model_id)

    categorical = {
        "device_type": [0, 1], "education_level": [1, 2, 3, 4, 5],
        "marital_status": [0, 1], "gender": [0, 1],
        "city_tier": [1, 2, 3, 4], "repayment_period": [6, 12, 24, 36],
    }

    # ── W0 基线 ──
    feature_names = [c for c in w0_df.columns
                     if c not in ("sample_id", "apply_time", "is_bad", "y_true",
                                  "risk_score", "y_pred_proba",
                                  "apply_hour_sin", "apply_hour_cos",
                                  "apply_weekday_sin", "apply_weekday_cos",
                                  "apply_is_weekend", "apply_is_night")]
    reference = w0_df.drop(columns=["risk_score"]) if "risk_score" in w0_df.columns else w0_df

    baseline = service.build_baseline(
        w0_data=w0_df, model_id=body.model_id, model_version=body.champion_version,
        feature_names=feature_names, categorical_features=categorical,
    )

    reference_scores = pd.Series(w0_df["y_pred_proba"])
    baseline_profile = pd.read_parquet(Path(baseline.feature_profile_uri))

    # ── 创建监控运行 ──
    trace_id = request_trace_id(request)
    run = await service.repo.create_run(
        model_id=body.model_id, champion_version=body.champion_version,
        baseline_window_id="W0", current_window_id="W3",
        data_track="NATURAL", trace_id=trace_id,
    )
    monitoring_run_id = run["monitoring_run_id"]

    # ── 每个窗口独立运行 _monitor_one ──
    WINDOWS = [("W1", w1_df), ("W2", w2_df), ("W3", w3_df)]

    all_metrics: list[MetricResult] = []
    all_alerts: list[AlertDetail] = []

    for window_id, w_df in WINDOWS:
        w_source = w_df.drop(columns=["risk_score"]) if "risk_score" in w_df.columns else w_df
        w_predictions = w_df[["sample_id", "risk_score"]].copy()

        w_perf, w_qual, w_drift = _monitor_one(
            source=w_source, predictions=w_predictions,
            monitor_window_id=window_id,
            context=baseline, baseline=baseline,
            reference=reference, reference_scores=reference_scores,
            baseline_profile=baseline_profile,
            data_track="NATURAL", trace_id=trace_id,
            min_samples=50, min_bad=1,
        )

        metrics, alerts = await _persist_window_metrics(
            service, monitoring_run_id, window_id, w_df, w0_df,
            w_perf, w_qual, w_drift, baseline, knowledge,
        )
        all_metrics.extend(metrics)
        all_alerts.extend(alerts)

    # ── 完成运行 ──
    has_alerts = len(all_alerts) > 0
    max_sev = None
    if all_alerts:
        sev_order = {"CRITICAL": 4, "HIGH": 3, "WARNING": 2, "INFO": 1}
        max_sev = max(all_alerts, key=lambda a: sev_order.get(a.severity.value, 0)).severity

    alert_context = AlertContext(
        schema_version="V2-WP08",
        trace_id=trace_id, monitoring_run_id=monitoring_run_id,
        model_id=body.model_id, model_version=body.champion_version,
        monitor_window_id="W1_W2_W3", baseline_id=baseline.baseline_id,
        data_track=DataTrack.NATURAL,
        alert_details=all_alerts,
    )
    await service.repo.complete_run(
        monitoring_run_id=monitoring_run_id, overall_status="COMPLETED",
        alert_count=len(all_alerts),
        max_alert_severity=max_sev.value if max_sev else None,
        alert_context_json=alert_context.model_dump(),
    )
    await db.commit()

    window_counts = {}
    for m in all_metrics:
        wid = m.metric_detail.get("window_id", "?")
        window_counts[wid] = window_counts.get(wid, 0) + 1

    return _envelope(
        request,
        {
            "monitoring_run_id": monitoring_run_id,
            "total_metrics": len(all_metrics),
            "metrics_per_window": window_counts,
            "has_alerts": has_alerts,
            "alert_count": len(all_alerts),
            "max_alert_severity": max_sev.value if max_sev else None,
        },
        message=f"3-window pipeline completed ({len(all_metrics)} metrics)",
    )


@router.post("/runs/detailed")
async def trigger_detailed_run(
    request: Request,
    body: RunDetailedMonitoringRequest,
    db: AsyncSession = Depends(get_db),
):
    """触发一次完整模式监控运行（V2）。

    使用交接包全套算法：PSI/JS/KS/Wasserstein + BH 校正 + 4 个检测器 + 趋势斜率。
    需要提供分箱规则和特征列表才能启用漂移检测。
    """
    driver = await get_neo4j_driver()
    knowledge = KnowledgeService(driver)
    service = MonitoringService(db, knowledge)

    baseline_df = load_window_with_predictions("W0")
    current_df = load_window_with_predictions("W3")
    reference_df = load_window_with_predictions("W0")
    baseline_data = baseline_df.to_dict(orient="records")
    current_data = current_df.to_dict(orient="records")
    reference_data = reference_df.to_dict(orient="records")

    result = await service.run_detailed(
        model_id=body.model_id,
        champion_version=body.champion_version,
        baseline_data=baseline_data,
        current_data=current_data,
        reference_data=reference_data,
        binning_rules=body.binning_rules,
        feature_names=body.feature_names,
        baseline_window_id=body.baseline_window_id,
        current_window_id=body.current_window_id,
        data_track=body.data_track,
        trace_id=request_trace_id(request),
        min_samples=body.min_samples,
        min_bad=body.min_bad,
    )

    return _envelope(
        request,
        {
            "monitoring_run_id": result.monitoring_run_id,
            "has_alerts": result.has_alerts,
            "alert_count": result.alert_count,
            "max_alert_severity": result.max_alert_severity.value if result.max_alert_severity else None,
        },
        message="detailed monitoring run completed",
    )


@router.post("/parallel-cycle")
async def trigger_parallel_cycle(
    request: Request,
    body: RunParallelCycleRequest,
    db: AsyncSession = Depends(get_db),
):
    """触发多模型并行监控周期（V2）。

    使用 asyncio 并发执行，最多 max_concurrency 个模型同时监控。
    """
    driver = await get_neo4j_driver()
    knowledge = KnowledgeService(driver)
    service = MonitoringService(db, knowledge)

    result = await service.run_parallel_cycle(
        model_ids=body.model_ids,
        champion_versions=body.champion_versions,
        max_concurrency=body.max_concurrency,
    )

    return _envelope(request, result, message="parallel monitoring cycle completed")
