"""监控 API 路由 — 任务一：模型监控"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from pydantic import BaseModel, Field

from ..core.exceptions import NotFoundError, request_trace_id
from ..database import get_db
from ..neo4j_db import get_neo4j_driver
from ..repositories.monitoring_repo import MonitoringRepo
from ..services.knowledge_service import KnowledgeService
from ..services.monitoring.monitoring_service import MonitoringService
from ..services.monitoring.window_loader import load_window_with_predictions

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
# 富化指标 & 判定台端点（Phase 1-2）
# ═══════════════════════════════════════════════════════════════

from ..services.monitoring.metric_enricher import (
    enrich_metric,
    build_coverage_summary,
)


@router.get("/runs/{monitoring_run_id}/enriched-metrics")
async def get_enriched_metrics(
    monitoring_run_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """获取富化后的全部指标 + 覆盖摘要（监控判定台核心端点）。

    与 /metrics 的区别：
    - 每个指标附带 category、display_name、direction、warning_threshold、
      critical_threshold、threshold_usage_ratio、status_reason
    - 额外返回 CoverageSummary（规则覆盖、类别统计、最接近阈值）
    """
    repo = MonitoringRepo(db)
    run = await repo.get_run(monitoring_run_id)
    if not run:
        raise NotFoundError(f"监控运行 {monitoring_run_id} 不存在")
    all_metrics = await repo.get_metrics(monitoring_run_id)

    enriched_all = [enrich_metric(m) for m in all_metrics]
    summary = build_coverage_summary(enriched_all, run)

    # 指标卡片区只展示 17 个核心指标（去重取第一个）
    from ..services.monitoring.metric_enricher import METRIC_CATEGORY_MAP
    canonical_codes = set(METRIC_CATEGORY_MAP.keys())
    seen_codes: set[str] = set()
    enriched_canonical: list[dict] = []
    for m in enriched_all:
        code = m["metric_code"]
        if code in canonical_codes and code not in seen_codes:
            seen_codes.add(code)
            enriched_canonical.append(m)

    # B1 持续性判定 — 从 monitoring_runs 读取
    raw_persistence = run.get("persistence_judgment_json")
    if raw_persistence is None:
        persistence = None
    elif isinstance(raw_persistence, dict):
        persistence = raw_persistence
    else:
        persistence = _parse_detail(raw_persistence)
    diagnosis_status = run.get("diagnosis_status")

    return _envelope(request, {
        "metrics": enriched_canonical,
        "summary": summary,
        "persistence": persistence,
        "diagnosis_status": diagnosis_status,
        "_v": "V2-persistence",
    })


def _psi_status(psi: float) -> str:
    if psi >= 0.25:
        return "critical"
    if psi >= 0.10:
        return "warning"
    return "normal"


def _psi_trend(psi_7d: float | None, psi_30d: float | None) -> str:
    if psi_7d is not None and psi_30d is not None and psi_30d > 0:
        if psi_7d > psi_30d * 1.1:
            return "up"
        if psi_7d < psi_30d * 0.9:
            return "down"
    return "stable"


def _parse_detail(detail) -> dict:
    """安全解析 metric_detail，处理 str/bytes/dict 等类型。"""
    if detail is None:
        return {}
    if isinstance(detail, dict):
        return detail
    if isinstance(detail, (str, bytes)):
        import json
        try:
            return json.loads(detail)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


@router.get("/runs/{monitoring_run_id}/feature-drift")
async def get_feature_drift(
    monitoring_run_id: str,
    request: Request,
    sort_by: str | None = Query(None, description="排序字段: psi|importance|status"),
    sort_order: str | None = Query("desc", description="排序方向: asc|desc"),
    db: AsyncSession = Depends(get_db),
):
    """获取特征漂移表格数据。

    支持按 PSI、模型重要性、状态排序。
    7D/30D 从 monitoring_feature_drift 表读取不同时间窗口的数据。
    """
    repo = MonitoringRepo(db)
    run = await repo.get_run(monitoring_run_id)
    if not run:
        raise NotFoundError(f"监控运行 {monitoring_run_id} 不存在")

    drift_rows = await repo.get_feature_drift_by_run(monitoring_run_id)

    # 结构化为前端格式
    items: list[dict] = []

    if drift_rows:
        # ── 优先从 monitoring_feature_drift 表读取 ──
        for row in drift_rows:
            psi = _safe_float(row.get("psi"))
            psi_7d = _safe_float(row.get("psi_7d"))
            psi_30d = _safe_float(row.get("psi_30d"))
            max_psi = psi or psi_30d or psi_7d or 0
            threshold = 0.10
            status = _psi_status(max_psi)
            trend = _psi_trend(psi_7d, psi_30d)
            items.append({
                "feature_name": row.get("feature_name", "-"),
                "psi_7d": psi_7d, "psi_30d": psi_30d, "max_psi": max_psi,
                "threshold": threshold, "status": status,
                "model_importance": row.get("model_importance"),
                "trend": trend,
                "js_divergence": _safe_float(row.get("js_divergence")),
                "wasserstein_distance": _safe_float(row.get("wasserstein_distance")),
                "ks_statistic": _safe_float(row.get("ks_statistic")),
                "missing_rate": _safe_float(row.get("missing_rate")),
                "missing_rate_delta": _safe_float(row.get("missing_rate_delta")),
                "outlier_rate": _safe_float(row.get("outlier_rate")),
                "dq_score": _safe_float(row.get("dq_score")),
                "dq_flag": row.get("dq_flag"),
            })
    else:
        # ── 回退：从 FEATURE_PSI 指标的 metric_detail.per_column_psi 提取 ──
        all_metrics = await repo.get_metrics(monitoring_run_id)
        for m in all_metrics:
            if m.get("metric_code") == "FEATURE_PSI":
                detail = _parse_detail(m.get("metric_detail"))
                per_col = detail.get("per_column_psi") or {}
                for feat_name, psi_val in per_col.items():
                    psi = _safe_float(psi_val) or 0
                    items.append({
                        "feature_name": feat_name,
                        "psi_7d": None, "psi_30d": None, "max_psi": psi,
                        "threshold": 0.10, "status": _psi_status(psi),
                        "model_importance": None, "trend": "stable",
                        "js_divergence": None, "wasserstein_distance": None,
                        "ks_statistic": None, "missing_rate": None,
                        "missing_rate_delta": None, "outlier_rate": None,
                        "dq_score": None, "dq_flag": None,
                    })
                break

    # 排序
    if sort_by == "psi":
        items.sort(key=lambda x: x["max_psi"] or 0, reverse=sort_order != "asc")
    elif sort_by == "importance":
        imp_order = {"高": 3, "中": 2, "低": 1}
        items.sort(key=lambda x: imp_order.get(x["model_importance"] or "", 0), reverse=sort_order != "asc")
    elif sort_by == "status":
        sev_order = {"critical": 3, "warning": 2, "normal": 1}
        items.sort(key=lambda x: sev_order.get(x["status"], 0), reverse=sort_order != "asc")
    else:
        # 默认按 max_psi 降序
        items.sort(key=lambda x: x["max_psi"] or 0, reverse=True)

    return _envelope(request, {"items": items})


@router.get("/runs/{monitoring_run_id}/data-quality")
async def get_data_quality(
    monitoring_run_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """获取数据质量下钻数据（字段级）。

    包含：整体缺失率/异常值率/DQ分、逐字段明细、Schema 变更。
    """
    repo = MonitoringRepo(db)
    run = await repo.get_run(monitoring_run_id)
    if not run:
        raise NotFoundError(f"监控运行 {monitoring_run_id} 不存在")

    drift_rows = await repo.get_feature_drift_by_run(monitoring_run_id)

    fields: list[dict] = []
    overall_missing = 0.0
    overall_outlier = 0.0
    dq_scores: list[float] = []

    if drift_rows:
        # ── 优先从 monitoring_feature_drift 表读取 ──
        for row in drift_rows:
            field_missing = _safe_float(row.get("missing_rate")) or 0
            field_missing_delta = _safe_float(row.get("missing_rate_delta"))
            field_outlier = _safe_float(row.get("outlier_rate")) or 0
            field_outlier_delta = _safe_float(row.get("outlier_rate_delta"))
            field_dq = _safe_float(row.get("dq_score"))
            fields.append({
                "field_name": row.get("feature_name", "-"),
                "baseline_missing_rate": round(field_missing - (field_missing_delta or 0), 6),
                "current_missing_rate": round(field_missing, 6),
                "missing_delta": field_missing_delta,
                "outlier_rate": round(field_outlier, 6),
                "outlier_delta": field_outlier_delta,
                "dq_flag": row.get("dq_flag") or "OK",
            })
            overall_missing = max(overall_missing, abs(field_missing_delta or 0))
            overall_outlier = max(overall_outlier, abs(field_outlier_delta or 0))
            if field_dq is not None:
                dq_scores.append(field_dq)
    else:
        # ── 回退：从 MISSING_RATE / OUTLIER_RATE / DATA_QUALITY_SCORE 指标的 metric_detail 提取 ──
        all_metrics = await repo.get_metrics(monitoring_run_id)
        for m in all_metrics:
            code = m.get("metric_code")
            detail = _parse_detail(m.get("metric_detail"))

            if code == "MISSING_RATE":
                per_col = detail.get("per_column") or {}
                for feat_name, missing_val in per_col.items():
                    mv = _safe_float(missing_val) or 0
                    existing = next((f for f in fields if f["field_name"] == feat_name), None)
                    if existing:
                        existing["current_missing_rate"] = round(mv, 6)
                        existing["missing_delta"] = round(mv, 6)  # 无法区分基线，假设基线为0
                    else:
                        fields.append({
                            "field_name": feat_name,
                            "baseline_missing_rate": None,
                            "current_missing_rate": round(mv, 6),
                            "missing_delta": round(mv, 6),
                            "outlier_rate": None,
                            "outlier_delta": None,
                            "dq_flag": "WARN" if mv > 0.05 else "OK",
                        })
                    overall_missing = max(overall_missing, abs(mv))

            elif code == "OUTLIER_RATE":
                max_delta = _safe_float(detail.get("max_delta")) or 0
                overall_outlier = max(overall_outlier, abs(max_delta))

            elif code == "DATA_QUALITY_SCORE":
                dq_val = _safe_float(m.get("current_value"))
                if dq_val is not None:
                    dq_scores.append(dq_val)
                dq_flag = detail.get("dq_flag") or "OK"
                # 将 DQ flag 应用到所有字段
                for f in fields:
                    if f["dq_flag"] == "OK" and dq_flag != "OK":
                        f["dq_flag"] = dq_flag

    avg_dq = round(sum(dq_scores) / len(dq_scores), 4) if dq_scores else None

    # Schema 一致性 — 从 SCHEMA_CONSISTENCY 指标读
    all_metrics = await repo.get_metrics(monitoring_run_id)
    schema_changes: list[dict] = []
    for m in all_metrics:
        if m.get("metric_code") == "SCHEMA_CONSISTENCY":
            detail = m.get("metric_detail") or {}
            if isinstance(detail, str):
                import json
                try:
                    detail = json.loads(detail)
                except (json.JSONDecodeError, TypeError):
                    detail = {}
            for col in detail.get("missing_columns", []):
                schema_changes.append({"change_type": "removed", "column_name": str(col), "detail": "基线存在但当前缺失"})
            for col in detail.get("new_columns", []):
                schema_changes.append({"change_type": "added", "column_name": str(col), "detail": "当前新增但基线不存在"})
            for col in detail.get("type_changes", []):
                schema_changes.append({
                    "change_type": "type_changed",
                    "column_name": str(col.get("column", col)),
                    "detail": f"{col.get('from', '?')} → {col.get('to', '?')}" if isinstance(col, dict) else str(col),
                })
            break

    return _envelope(request, {
        "overall_missing_rate": round(overall_missing, 6),
        "overall_outlier_rate": round(overall_outlier, 6),
        "dq_score": avg_dq,
        "fields": fields,
        "schema_changes": schema_changes,
    })


# ═══════════════════════════════════════════════════════════════
# ── 辅助函数 ──


def _safe_float(val) -> float | None:
    """安全转 float，处理 None / NaN / inf。"""
    if val is None:
        return None
    try:
        import math
        f = float(val)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


# ═══════════════════════════════════════════════════════════════
# 核心端点
# ═══════════════════════════════════════════════════════════════

@router.post("/runs")
async def trigger_run(
    request: Request,
    body: RunMonitoringRequest,
    db: AsyncSession = Depends(get_db),
):
    """触发一次完整监控管道 — 调用 MonitoringService.run_full_pipeline()。

    管道：W0 基线 → W1+W2+W3 滚动窗口 → per-feature 漂移/质量 → 检测器 → Sentinel。
    产出：monitoring_metrics + monitoring_feature_drift + 诊断时间线证据。
    """
    driver = await get_neo4j_driver()
    knowledge = KnowledgeService(driver)
    service = MonitoringService(db, knowledge)

    w0_df = load_window_with_predictions("W0", body.model_id)
    w1_df = load_window_with_predictions("W1", body.model_id)
    w2_df = load_window_with_predictions("W2", body.model_id)
    w3_df = load_window_with_predictions("W3", body.model_id)

    result = await service.run_full_pipeline(
        model_id=body.model_id,
        champion_version=body.champion_version,
        w0_df=w0_df, w1_df=w1_df, w2_df=w2_df, w3_df=w3_df,
        trace_id=request_trace_id(request),
    )

    return _envelope(
        request,
        {
            "monitoring_run_id": result.monitoring_run_id,
            "has_alerts": result.has_alerts,
            "alert_count": result.alert_count,
            "max_alert_severity": result.max_alert_severity.value if result.max_alert_severity else None,
            "total_metrics": len(result.metrics),
        },
        message="full pipeline completed",
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
