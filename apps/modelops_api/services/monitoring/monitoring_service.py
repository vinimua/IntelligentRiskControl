"""MonitoringService — 任务一：模型监控 Pipeline

基于确定性指标计算、阈值规则和知识图谱查询。
输入：baseline + current 数据集快照
输出：AlertContext + MonitoringStateOutput

V2 增强（2026-07-20）：
- 新增 run_detailed() 完整模式：PSI + 漂移 + 检测器 + Sentinel 推理
- 新增 run_parallel_cycle() 多模型并发
- 新增 build_baseline() W0 基线构建
- 新增 run_rolling() 多窗口持续监控
- 保留 run() 快速模式不变
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from packages.models.common.enums import (
    AvailabilityStatus,
    DataTrack,
    MetricDirection,
    ObjectType,
    RuleType,
    Severity,
)
from packages.models.monitoring.alert_context import AlertContext, AlertDetail

from ...repositories.monitoring_repo import MonitoringRepo
from ..dataset_access_policy import DatasetAccessPolicy
from ..knowledge_service import KnowledgeService
from .metrics_registry import METRIC_CALCULATORS, MetricResult
from .threshold_rules import DEFAULT_THRESHOLD_RULES

# 导入 metric_calculators 触发 @register 装饰器，填充 METRIC_CALCULATORS
from . import metric_calculators  # noqa: F401

# V2 算法模块（第一类 P0-P1）
from .baseline import (
    MonitoringBaseline,
    build_monitoring_baseline,
    compute_monitoring_performance_metrics,
)
from .drift.algorithms import (
    benjamini_hochberg,
    categorical_drift,
    compute_performance_metrics,
    continuous_drift,
    feature_quality,
    psi_from_edges,
)
from .drift.output_monitor import output_metrics
from .detectors.runner import run_detectors
from .rolling import iter_rolling_windows, iter_end_aligned_windows
from .sentinel.alert import AlertEvent, alert_summary, build_alerts
from .sentinel.feature_builder import (
    build_monitor_feature_vector,
    select_canonical_sentinel_rows,
)
from .sentinel.inference import SentinelBundle, infer_sentinel
from .trend_features import trailing_slope

logger = structlog.get_logger(__name__)


# ── 7D/30D 单窗口处理辅助函数 ──

def _process_single_window(
    start, end, window, baseline, reference_scores,
    perf_rows: list, qual_rows: list, drift_rows: list,
    *, model_id: str, champion_version: str, trace_id: str,
) -> None:
    """处理单个滚动窗口：性能 + 漂移 + 质量 → 追加到 perf_rows/qual_rows/drift_rows。"""
    w_days = (end - start).days
    window_id = f"{w_days}D_{start:%Y%m%d}_{end:%Y%m%d}"
    sample_count = len(window)
    bad_count = int(window["is_bad"].sum()) if "is_bad" in window.columns else None

    label_ready = (
        "is_bad" in window.columns and "y_pred_proba" in window.columns
        and sample_count >= 50 and bad_count is not None and bad_count >= 1
    )
    if label_ready:
        perf = compute_monitoring_performance_metrics(
            window["is_bad"],
            window["y_pred_proba"],
            window["risk_score"] if "risk_score" in window.columns else None,
        )
    else:
        perf = {k: None for k in ("auc", "ks", "pr_auc", "brier", "ece", "bad_recall")}

    out = output_metrics(window["y_pred_proba"], reference_scores, baseline.score_edges)

    common = {
        "trace_id": trace_id, "model_id": model_id, "model_version": champion_version,
        "baseline_id": baseline.baseline_id, "baseline_version": baseline.baseline_version,
        "monitor_window_id": window_id,
        "window_start": start, "window_end": end,
        "window_days": w_days,
        "sample_count": sample_count, "bad_count": bad_count,
        "data_track": "NATURAL",
        "scenario_id": float("nan"), "scenario_instance_id": float("nan"),
    }
    perf_rows.append({**common, **perf, **out})

    p_positions: list[int] = []
    p_values: list[float | None] = []

    w0_df = baseline._w0_df if hasattr(baseline, '_w0_df') else None

    for fname in baseline.feature_names:
        rule = baseline.binning_rules_json.get(fname)
        if rule is None or fname not in window.columns:
            continue

        if fname in baseline.feature_profiles:
            quality = feature_quality(
                window[fname], pd.Series(baseline.feature_profiles[fname]),
                rule["feature_type"],
            )
            qual_rows.append({**common, "feature_name": fname, **quality})

        w0_col = w0_df[fname] if w0_df is not None and fname in w0_df.columns else None
        if w0_col is not None:
            if rule["feature_type"] == "categorical":
                drift = categorical_drift(w0_col, window[fname], rule["categories"])
                row = {"feature_name": fname, "feature_type": "categorical", **drift,
                       "wasserstein_distance": None, "ks_statistic": None,
                       "ks_p_value": None, "ks_q_value": None}
            else:
                drift = continuous_drift(w0_col, window[fname], rule["edges"])
                row = {"feature_name": fname, "feature_type": "continuous", **drift,
                       "category_share_change": None, "unknown_category_rate": 0.0,
                       "ks_q_value": None}
                if row.get("ks_p_value") is not None:
                    p_positions.append(len(drift_rows))
                    p_values.append(row["ks_p_value"])
            drift_rows.append({**common, **row})

    for pos, q_val in zip(p_positions, benjamini_hochberg(p_values)):
        drift_rows[pos]["ks_q_value"] = q_val


# MetricDirection → RuleType 映射
_DIRECTION_TO_RULE: dict[MetricDirection, RuleType] = {
    MetricDirection.HIGHER_BETTER: RuleType.DROP_THRESHOLD,
    MetricDirection.LOWER_BETTER: RuleType.LOWER_THRESHOLD,
    MetricDirection.DEVIATION_BAD: RuleType.SHIFT_THRESHOLD,
}


@dataclass
class MonitoringRunResult:
    """MonitoringService.run() 的完整返回。"""

    monitoring_run_id: str
    has_alerts: bool
    alert_count: int = 0
    max_alert_severity: Severity | None = None
    alerts: list[AlertDetail] = field(default_factory=list)
    metrics: list[dict] = field(default_factory=list)


class MonitoringService:
    """模型监控服务 — 指标计算 → 阈值比对 → 告警生成。"""

    def __init__(self, session: AsyncSession, knowledge_service: KnowledgeService):
        self.session = session
        self.repo = MonitoringRepo(session)
        self.policy = DatasetAccessPolicy(session)
        self.knowledge = knowledge_service
        self.calculators = METRIC_CALCULATORS
        self.rules = DEFAULT_THRESHOLD_RULES

    async def run(
        self,
        model_id: str,
        champion_version: str,
        baseline_data: list[dict],
        current_data: list[dict],
        baseline_window_id: str = "",
        current_window_id: str = "",
        data_track: str = "NATURAL",
        trace_id: str | None = None,
    ) -> MonitoringRunResult:
        """执行一次完整的监控运行。

        baseline_data / current_data 可以直接传入 dict 列表（测试用），
        如果为空则从 MinIO 快照读取（生产路径）。
        """
        # ① 创建 monitoring_run 记录
        run = await self.repo.create_run(
            model_id=model_id,
            champion_version=champion_version,
            baseline_window_id=baseline_window_id or "W0",
            current_window_id=current_window_id or "W3",
            data_track=data_track,
            trace_id=trace_id,
        )
        monitoring_run_id = run["monitoring_run_id"]
        logger.info("monitoring_run_started", monitoring_run_id=monitoring_run_id, model_id=model_id)

        # ② 检查标签成熟度（完整性校验，不只是布尔值）
        availability_issues: dict[str, AvailabilityStatus] = {}
        for label, win_id in [("baseline", baseline_window_id), ("current", current_window_id)]:
            if not win_id:
                continue
            window = {"window_id": win_id, "allows_monitoring_label": True}
            if not self.policy.can_read_labels(window, "TASK_1"):
                availability_issues[label] = AvailabilityStatus.LABEL_NOT_MATURE
            else:
                # 检查标签成熟时间
                try:
                    self.policy.validate_label_maturity(None, "TASK_1", datetime.now(timezone.utc))
                except Exception:
                    availability_issues[label] = AvailabilityStatus.LABEL_NOT_MATURE

        # ③ 遍历指标计算器
        all_metrics: list[MetricResult] = []
        triggered_alerts: list[AlertDetail] = []
        # 需要标签的指标（后续可从 registry 声明派生）
        _LABEL_DEPENDENT_METRICS = {"AUC", "KS"}

        for metric_code, calc_fn in self.calculators.items():
            # 如果标签不成熟，跳过需要标签的指标
            if availability_issues and metric_code in _LABEL_DEPENDENT_METRICS:
                mr = MetricResult(
                    metric_code=metric_code,
                    availability_status=AvailabilityStatus.LABEL_NOT_MATURE,
                )
                all_metrics.append(mr)
                await self._persist_metric(monitoring_run_id, mr)
                continue

            # 计算指标
            try:
                mr = calc_fn(baseline_data, current_data)
            except Exception:
                logger.warning("metric_calculation_failed", metric_code=metric_code, exc_info=True)
                mr = MetricResult(
                    metric_code=metric_code,
                    availability_status=AvailabilityStatus.CALCULATION_FAILED,
                )
            all_metrics.append(mr)

            # ④ 持久化指标（无论是否触发告警）
            metric_id = await self._persist_metric(monitoring_run_id, mr)

            # ⑤ 应用阈值规则
            rule = self.rules.get(metric_code)
            if rule and mr.availability_status == AvailabilityStatus.AVAILABLE:
                triggered, severity = rule.evaluate(mr.delta, mr.current_value)
                if triggered and severity:
                    # ⑥ 查询知识图谱获取 alert_code
                    alert_type = await self.knowledge.resolve_alert(metric_code, severity)

                    alert_code = alert_type.alert_code if alert_type else f"ANOMALY_{metric_code}"
                    alert_detail = AlertDetail(
                        alert_id=str(uuid.uuid4()),
                        alert_code=alert_code,
                        severity=severity,
                        object_type=ObjectType.MODEL,
                        object_code=model_id,
                        metric_code=metric_code,
                        metric_version="V1",
                        baseline_value=mr.baseline_value,
                        current_value=mr.current_value,
                        delta=mr.delta,
                        threshold=rule.critical_threshold,
                        rule_type=_DIRECTION_TO_RULE.get(rule.direction, RuleType.SHIFT_THRESHOLD),
                        threshold_rule_id=rule.rule_id,
                        threshold_rule_version=rule.rule_version,
                        availability_status=mr.availability_status,
                        metric_detail=mr.metric_detail,
                        created_at=datetime.now(timezone.utc),
                    )
                    triggered_alerts.append(alert_detail)

                    # 持久化告警（关联 metric_id）
                    await self._persist_alert(monitoring_run_id, metric_id, mr, alert_detail)

        # ⑦ 生成 AlertContext
        has_alerts = len(triggered_alerts) > 0
        max_sev = None
        if triggered_alerts:
            sev_order = {"CRITICAL": 4, "HIGH": 3, "WARNING": 2, "INFO": 1}
            max_sev = max(triggered_alerts, key=lambda a: sev_order.get(a.severity.value, 0)).severity

        # 安全解析 DataTrack，非法输入回退到 NATURAL
        try:
            track = DataTrack(data_track)
        except ValueError:
            logger.warning("invalid_data_track", data_track=data_track)
            track = DataTrack.NATURAL

        alert_context = AlertContext(
            schema_version="V1",
            trace_id=trace_id or "",
            monitoring_run_id=monitoring_run_id,
            model_id=model_id,
            model_version=champion_version,
            monitor_window_id=current_window_id or "W3",
            baseline_id=baseline_window_id or "W0",
            data_track=track,
            alert_details=triggered_alerts,
        )

        # ⑦ B1 持续性判定
        from .persistence_judgment import PersistenceJudgmentService
        judgment_svc = PersistenceJudgmentService(self.session)
        judgment = await judgment_svc.judge(monitoring_run_id)
        diagnosis_status = "PENDING" if judgment.trigger_diagnosis else "SKIPPED"
        await self.repo.update_persistence_judgment(
            monitoring_run_id, judgment.__dict__, diagnosis_status,
        )

        # ⑧ 完成 run
        await self.repo.complete_run(
            monitoring_run_id=monitoring_run_id,
            overall_status="COMPLETED",
            alert_count=len(triggered_alerts),
            max_alert_severity=max_sev.value if max_sev else None,
            alert_context_json=alert_context.model_dump(),
        )

        await self.session.commit()
        logger.info(
            "monitoring_run_completed",
            monitoring_run_id=monitoring_run_id,
            alert_count=len(triggered_alerts),
        )

        return MonitoringRunResult(
            monitoring_run_id=monitoring_run_id,
            has_alerts=has_alerts,
            alert_count=len(triggered_alerts),
            max_alert_severity=max_sev,
            alerts=triggered_alerts,
            metrics=[_metric_to_dict(m) for m in all_metrics],
        )

    async def _emit_alert(
        self, monitoring_run_id: str, metric_id: str, mr: MetricResult,
        rule, triggered_alerts: list, *,
        object_type: ObjectType = ObjectType.MODEL, object_code: str = "",
    ) -> None:
        """统一告警生成：阈值评估 → KG查alert_code → 持久化 → 更新triggered。severity严格使用规则档位。"""
        if rule is None or mr.availability_status != AvailabilityStatus.AVAILABLE:
            return
        triggered, sev = rule.evaluate(mr.delta, mr.current_value)
        if not (triggered and sev):
            return
        alert_type = await self.knowledge.resolve_alert(mr.metric_code, sev)
        metric_detail = dict(mr.metric_detail or {})
        value = mr.delta if mr.delta is not None else mr.current_value
        blocking_threshold = getattr(rule, "blocking_threshold", None)
        if blocking_threshold is not None and value is not None:
            try:
                if abs(float(value)) >= float(blocking_threshold):
                    metric_detail["blocking"] = True
                    metric_detail["blocking_threshold"] = float(blocking_threshold)
                    metric_detail["blocking_reason"] = f"{mr.metric_code}_GE_BLOCKING_THRESHOLD"
            except (TypeError, ValueError):
                pass
        metric_detail["alert_source"] = (
            "KG" if (alert_type and getattr(alert_type, "from_neo4j", False)) else "FALLBACK"
        )
        detail = AlertDetail(
            alert_id=str(uuid.uuid4()),
            alert_code=alert_type.alert_code if alert_type else f"ANOMALY_{mr.metric_code}",
            severity=sev,
            object_type=object_type, object_code=object_code,
            metric_code=mr.metric_code, metric_version="V2",
            baseline_value=mr.baseline_value, current_value=mr.current_value,
            delta=mr.delta, threshold=rule.critical_threshold,
            rule_type=_DIRECTION_TO_RULE.get(rule.direction, RuleType.SHIFT_THRESHOLD),
            threshold_rule_id=rule.rule_id, threshold_rule_version=rule.rule_version,
            availability_status=mr.availability_status,
            metric_detail=metric_detail,
            created_at=datetime.now(timezone.utc),
        )
        triggered_alerts.append(detail)
        await self._persist_alert(monitoring_run_id, metric_id, mr, detail)
        await self.repo.update_metric_triggered(metric_id, True)

    # ── 完整 WP02-WP08 管道（与交接包 pipeline.py 一致） ──

    async def run_full_pipeline(
        self,
        model_id: str,
        champion_version: str,
        w0_df: pd.DataFrame,
        w1_df: pd.DataFrame,
        w2_df: pd.DataFrame,
        w3_df: pd.DataFrame,
        trace_id: str | None = None,
        window_days: int = 7,
        categorical_features: dict[str, list[str]] | None = None,
    ) -> MonitoringRunResult:
        """完整 WP02-WP08 监控管道 — 与交接包 pipeline.py 行为一致。

        管道：
          WP02: W0 基线构建（分箱规则 + 性能基准 + 特征概要）
          WP03: 已由调用方完成预测（w0_df 等已含 y_pred_proba）
          WP04: 滚动窗口性能监控
          WP05: 特征漂移 + 数据质量 + BH 校正
          WP07: 4 个流式检测器（跨窗口有状态）
          WP08: Sentinel 特征向量 + 趋势斜率

        持久化：17 个指标 + 告警 + 检测器信号 → PostgreSQL
        """
        trace_id = trace_id or str(uuid.uuid4())

        # ── ① 创建 run 记录 ──
        run = await self.repo.create_run(
            model_id=model_id, champion_version=champion_version,
            baseline_window_id="W0", current_window_id="W3",
            data_track="NATURAL", trace_id=trace_id,
        )
        monitoring_run_id = run["monitoring_run_id"]
        logger.info("full_pipeline_started", monitoring_run_id=monitoring_run_id, model_id=model_id)

        # ── WP02: W0 基线 ──
        feature_names = [
            c for c in w0_df.columns
            if c not in ("sample_id", "apply_time", "is_bad", "y_true",
                         "risk_score", "y_pred_proba",
                         "apply_hour_sin", "apply_hour_cos",
                         "apply_weekday_sin", "apply_weekday_cos",
                         "apply_is_weekend", "apply_is_night")
        ]
        baseline = self.build_baseline(
            w0_data=w0_df, model_id=model_id, model_version=champion_version,
            feature_names=feature_names, categorical_features=categorical_features,
        )
        baseline._w0_df = w0_df

        # ── WP04-WP05: 滚动窗口 + 漂移 ──
        all_data = pd.concat([w1_df, w2_df, w3_df], ignore_index=True).sort_values("apply_time")
        reference_scores = w0_df["y_pred_proba"]

        perf_rows: list[dict] = []
        qual_rows: list[dict] = []
        drift_rows: list[dict] = []

        for start, end, window in iter_rolling_windows(all_data, window_days=window_days, step_days=1):
            window_id = f"{window_days}D_{start:%Y%m%d}_{end:%Y%m%d}"
            sample_count = len(window)
            bad_count = int(window["is_bad"].sum()) if "is_bad" in window.columns else None

            # 性能指标
            label_ready = (
                "is_bad" in window.columns and "y_pred_proba" in window.columns
                and sample_count >= 50 and bad_count is not None and bad_count >= 1
            )
            if label_ready:
                perf = compute_monitoring_performance_metrics(
                    window["is_bad"],
                    window["y_pred_proba"],
                    window["risk_score"] if "risk_score" in window.columns else None,
                )
            else:
                perf = {k: None for k in ("auc", "ks", "pr_auc", "brier", "ece", "bad_recall")}

            # 预测输出
            out = output_metrics(window["y_pred_proba"], reference_scores, baseline.score_edges)

            common = {
                "trace_id": trace_id, "model_id": model_id, "model_version": champion_version,
                "baseline_id": baseline.baseline_id, "baseline_version": baseline.baseline_version,
                "monitor_window_id": window_id,
                "window_start": start, "window_end": end,
                "window_days": window_days,
                "sample_count": sample_count, "bad_count": bad_count,
                "data_track": "NATURAL",
                "scenario_id": float("nan"),
                "scenario_instance_id": float("nan"),
            }
            perf_rows.append({**common, **perf, **out})

            # 特征漂移 + 质量
            p_positions: list[int] = []
            p_values: list[float | None] = []
            for fname in baseline.feature_names:
                rule = baseline.binning_rules_json.get(fname)
                if rule is None or fname not in window.columns or fname not in w0_df.columns:
                    continue

                # 质量
                if fname in baseline.feature_profiles:
                    quality = feature_quality(
                        window[fname], pd.Series(baseline.feature_profiles[fname]),
                        rule["feature_type"],
                    )
                    qual_rows.append({**common, "feature_name": fname, **quality})

                # 漂移
                if rule["feature_type"] == "categorical":
                    drift = categorical_drift(w0_df[fname], window[fname], rule["categories"])
                    row = {"feature_name": fname, "feature_type": "categorical", **drift,
                           "wasserstein_distance": None, "ks_statistic": None,
                           "ks_p_value": None, "ks_q_value": None}
                else:
                    drift = continuous_drift(w0_df[fname], window[fname], rule["edges"])
                    row = {"feature_name": fname, "feature_type": "continuous", **drift,
                           "category_share_change": None, "unknown_category_rate": 0.0,
                           "ks_q_value": None}
                    if row.get("ks_p_value") is not None:
                        p_positions.append(len(drift_rows))
                        p_values.append(row["ks_p_value"])
                drift_rows.append({**common, **row})

            for pos, q_val in zip(p_positions, benjamini_hochberg(p_values)):
                drift_rows[pos]["ks_q_value"] = q_val

        # ── 30D 评估点（从最新往回，每 7 天一个，窗口 30 天）──
        for start, end, window in iter_end_aligned_windows(all_data, window_days=30, step_days=7):
            _process_single_window(
                start, end, window, baseline, reference_scores,
                perf_rows, qual_rows, drift_rows,
                model_id=model_id, champion_version=champion_version,
                trace_id=trace_id,
            )

        perf_df = pd.DataFrame(perf_rows)
        qual_df = pd.DataFrame(qual_rows)
        drift_df = pd.DataFrame(drift_rows)
        triggered_alerts: list[AlertDetail] = []

        w0_schema_cols = set(w0_df.columns) - {"y_pred_proba", "risk_score"}
        w3_schema_cols = set(w3_df.columns) - {"y_pred_proba", "risk_score"}
        schema_inconsistent_value = 1.0 if (w0_schema_cols - w3_schema_cols or w3_schema_cols - w0_schema_cols) else 0.0

        async def _persist_event_metric(
            metric_code: str,
            current_value: float | None,
            baseline_value: float | None,
            delta_value: float | None,
            detail: dict,
            availability: AvailabilityStatus = AvailabilityStatus.AVAILABLE,
            object_type: ObjectType = ObjectType.MODEL,
            object_code: str | None = None,
        ) -> str:
            mr = MetricResult(
                metric_code=metric_code,
                baseline_value=baseline_value,
                current_value=current_value,
                delta=delta_value,
                availability_status=availability,
                metric_detail=detail,
            )
            result = await self.repo.insert_metric(
                monitoring_run_id=monitoring_run_id,
                metric_code=metric_code,
                metric_version="V2_EVENT_TIME",
                object_type=object_type.value,
                object_code=object_code or model_id,
                baseline_value=baseline_value,
                current_value=current_value,
                delta=delta_value,
                availability_status=availability.value,
                metric_detail=detail,
            )
            # 不在这里触发告警 — per-window V2_EVENT_TIME 是诊断证据，不是告警源
            # 告警统一由外层 _persist_summary_metric + _emit_alert 处理
            return result["metric_id"]

        # Diagnosis V2 contract: persist the complete model-specific timeline.
        # A diagnosis at event time T may only read four historical,
        # non-overlapping windows ending at T; latest-window summaries are not
        # sufficient evidence for correlation or temporal precedence.
        for row in perf_rows:
            window_detail = {
                "window_id": row["monitor_window_id"],
                "window_start": row["window_start"],
                "window_end": row["window_end"],
                "window_days": row["window_days"],
                "sample_count": row["sample_count"],
                "bad_count": row["bad_count"],
                "category": "diagnosis_timeline",
            }
            for code in ("AUC", "KS", "PR_AUC", "BRIER", "ECE", "BAD_RECALL"):
                key = code.lower()
                current = row.get(key)
                baseline_value = baseline.performance_reference_json.get(key)
                current_value = (
                    float(current) if current is not None and pd.notna(current) else None
                )
                delta_value = (
                    current_value - baseline_value
                    if current_value is not None and baseline_value is not None
                    else None
                )
                await _persist_event_metric(
                    code,
                    current_value,
                    baseline_value,
                    delta_value,
                    window_detail,
                    (
                        AvailabilityStatus.AVAILABLE
                        if current_value is not None
                        else AvailabilityStatus.SAMPLE_TOO_SMALL
                    ),
                )

            score_psi = row.get("prediction_psi")
            if score_psi is not None and pd.notna(score_psi):
                await _persist_event_metric(
                    "SCORE_PSI", float(score_psi), None, None,
                    {**window_detail, "category": "distribution"},
                )

            await _persist_event_metric(
                "SAMPLE_SIZE", float(row["sample_count"]), None, None,
                {**window_detail, "category": "guardrail"},
            )
            await _persist_event_metric(
                "SCHEMA_CONSISTENCY", schema_inconsistent_value, None, None,
                {**window_detail, "category": "guardrail"},
            )

            wid = row["monitor_window_id"]
            window_drift = (
                drift_df[drift_df["monitor_window_id"] == wid]
                if not drift_df.empty and "monitor_window_id" in drift_df.columns
                else pd.DataFrame()
            )
            if not window_drift.empty and "psi" in window_drift.columns:
                psi_vals = window_drift["psi"].dropna()
                if len(psi_vals) > 0:
                    max_psi = float(psi_vals.max())
                    mean_psi = float(psi_vals.mean())
                    await _persist_event_metric(
                        "FEATURE_PSI",
                        max_psi,
                        mean_psi,
                        None,
                        {
                            **window_detail,
                            "category": "drift",
                            "max_psi": max_psi,
                            "mean_psi": mean_psi,
                            "n_features": int(len(psi_vals)),
                        },
                        object_type=ObjectType.FEATURE,
                        object_code="ALL",
                    )

            window_qual = (
                qual_df[qual_df["monitor_window_id"] == wid]
                if not qual_df.empty and "monitor_window_id" in qual_df.columns
                else pd.DataFrame()
            )
            if not window_qual.empty and "missing_rate" in window_qual.columns:
                miss_vals = window_qual["missing_rate"].dropna()
                if len(miss_vals) > 0:
                    await _persist_event_metric(
                        "MISSING_RATE",
                        float(miss_vals.max()),
                        None,
                        None,
                        {**window_detail, "category": "quality", "source": "qual_df_window_max"},
                    )
            if not window_qual.empty and "outlier_rate" in window_qual.columns:
                out_vals = window_qual["outlier_rate"].dropna()
                if len(out_vals) > 0:
                    await _persist_event_metric(
                        "OUTLIER_RATE",
                        float(out_vals.max()),
                        None,
                        None,
                        {**window_detail, "category": "quality", "source": "qual_df_window_max"},
                    )

        await self.repo.batch_insert_feature_drift(
            monitoring_run_id,
            drift_df.to_dict(orient="records"),
            qual_df.to_dict(orient="records"),
        )

        # ── WP07: 检测器 ──
        features_input = perf_df[["monitor_window_id", "window_end"]].copy()
        features_input["model_id"] = model_id
        features_input["model_version"] = champion_version
        features_input["data_track"] = "NATURAL"
        features_input["scenario_id"] = float("nan")
        features_input["scenario_instance_id"] = float("nan")
        for col in ("auc", "ks", "prediction_mean"):
            features_input[col] = features_input["monitor_window_id"].map(
                perf_df.set_index("monitor_window_id")[col]
            ) if col in perf_df.columns else None

        if not drift_df.empty:
            max_psi = drift_df.groupby("monitor_window_id")["psi"].max()
            features_input["max_feature_psi_7d"] = features_input["monitor_window_id"].map(max_psi)
        if not qual_df.empty:
            max_miss = qual_df.groupby("monitor_window_id")["missing_rate_delta"].apply(lambda x: x.abs().max())
            max_out = qual_df.groupby("monitor_window_id")["outlier_rate_delta"].apply(lambda x: x.abs().max())
            features_input["missing_rate_max_delta"] = features_input["monitor_window_id"].map(max_miss)
            features_input["outlier_rate_max_delta"] = features_input["monitor_window_id"].map(max_out)

        for col in ("auc", "ks", "prediction_mean", "max_feature_psi_7d",
                    "prediction_psi_7d", "prediction_psi_30d", "max_feature_psi_30d",
                    "missing_rate_max_delta", "outlier_rate_max_delta"):
            if col not in features_input.columns:
                features_input[col] = None

        detector_df = run_detectors(features_input)

        # ── WP08: Sentinel 特征向量 ──
        feature_df = build_monitor_feature_vector(perf_df, qual_df, drift_df, detector_df)
        feature_df = select_canonical_sentinel_rows(feature_df)

        # ── 趋势斜率 ──
        if len(perf_df) >= 5:
            auc_vals = [r["auc"] for r in perf_rows if r.get("auc") is not None]
            auc_slope = trailing_slope(auc_vals)
        else:
            auc_slope = None

        # ── 汇总为 API 指标格式 ──
        all_metrics: list[MetricResult] = []
        metric_ids: dict[int, str] = {}

        async def _persist_summary_metric(mr: MetricResult, *, triggered: bool = False) -> str:
            all_metrics.append(mr)
            metric_id = await self._persist_metric(monitoring_run_id, mr, triggered=triggered)
            metric_ids[id(mr)] = metric_id
            return metric_id

        latest = perf_df.iloc[-1] if len(perf_df) > 0 else None

        # 6 个性能指标（从最新窗口取）
        for code in ("AUC", "KS", "PR_AUC", "BRIER", "ECE", "BAD_RECALL"):
            key = code.lower()
            cur_val = float(latest[key]) if latest is not None and key in latest.index and pd.notna(latest[key]) else None
            base_val = baseline.performance_reference_json.get(key)
            delta_val = (cur_val - base_val) if cur_val is not None and base_val is not None else None
            mr = MetricResult(metric_code=code, current_value=cur_val, baseline_value=base_val,
                              delta=delta_val,
                              availability_status=AvailabilityStatus.AVAILABLE if cur_val is not None else AvailabilityStatus.SAMPLE_TOO_SMALL)
            await _persist_summary_metric(mr)

        # BAD_RATE
        cur_br = float(drift_df["bad_count"].iloc[-1] / max(1, drift_df["sample_count"].iloc[-1])) if not drift_df.empty else None
        base_br = baseline.performance_reference_json.get("bad_rate")
        mr = MetricResult(metric_code="BAD_RATE", current_value=cur_br, baseline_value=base_br,
                          delta=(cur_br - base_br) if cur_br is not None and base_br is not None else None)
        await _persist_summary_metric(mr)

        # 漂移汇总
        if not drift_df.empty:
            psi_vals = drift_df["psi"].dropna()
            mean_psi = float(psi_vals.mean())
            max_psi = float(psi_vals.max())

            mr = MetricResult(metric_code="FEATURE_PSI", current_value=max_psi,
                              baseline_value=mean_psi, delta=None,
                              metric_detail={"max_psi": max_psi, "mean_psi": mean_psi,
                                             "window_count": len(perf_df)})
            metric_id = await _persist_summary_metric(mr, triggered=False)
            await self._emit_alert(monitoring_run_id, metric_id, mr,
                                   self.rules.get("FEATURE_PSI"), triggered_alerts,
                                   object_type=ObjectType.FEATURE, object_code="ALL")

            for code, col in [("MAX_FEATURE_PSI_7D", "max_feature_psi_7d"),
                              ("MAX_FEATURE_PSI_30D", "max_feature_psi_30d")]:
                val = float(max_psi)
                mr2 = MetricResult(metric_code=code, current_value=val)
                await _persist_summary_metric(mr2)

        # SCORE_PSI
        if latest is not None and "prediction_psi" in latest.index:
            score_psi_val = float(latest["prediction_psi"]) if pd.notna(latest["prediction_psi"]) else None
            mr = MetricResult(metric_code="SCORE_PSI", current_value=score_psi_val)
            await _persist_summary_metric(mr)

        # 数据质量
        if not qual_df.empty:
            dq_scores = qual_df["dq_score"].dropna()
            mean_dq = float(dq_scores.mean()) if len(dq_scores) > 0 else None
            mr = MetricResult(metric_code="DATA_QUALITY_SCORE", current_value=mean_dq)
            await _persist_summary_metric(mr)

        # 预测均值
        if latest is not None and "prediction_mean" in latest.index:
            base_pm = float(reference_scores.mean()) if len(reference_scores) > 0 else None
            cur_pm = float(latest["prediction_mean"]) if pd.notna(latest["prediction_mean"]) else None
            mr = MetricResult(metric_code="PREDICTION_MEAN", current_value=cur_pm,
                              baseline_value=base_pm,
                              delta=(cur_pm - base_pm) if cur_pm is not None and base_pm is not None else None)
            await _persist_summary_metric(mr)

        # 元数据指标（SAMPLE_SIZE / SCHEMA / MISSING_RATE / OUTLIER_RATE 从真实数据计算 + 告警）
        mr = MetricResult(metric_code="SAMPLE_SIZE", current_value=float(len(w3_df)))
        mid = await _persist_summary_metric(mr, triggered=False)
        await self._emit_alert(monitoring_run_id, mid, mr, self.rules.get("SAMPLE_SIZE"), triggered_alerts)

        mr = MetricResult(metric_code="SCHEMA_CONSISTENCY", current_value=schema_inconsistent_value)
        mid = await _persist_summary_metric(mr, triggered=False)
        await self._emit_alert(monitoring_run_id, mid, mr, self.rules.get("SCHEMA_CONSISTENCY"), triggered_alerts)

        max_missing = 0.0
        if not qual_df.empty and "missing_rate" in qual_df.columns:
            max_missing = float(qual_df["missing_rate"].max())
        mr = MetricResult(metric_code="MISSING_RATE", current_value=max_missing,
                          metric_detail={"source": "qual_df_max", "window_count": len(perf_df)})
        mid = await _persist_summary_metric(mr, triggered=False)
        await self._emit_alert(monitoring_run_id, mid, mr, self.rules.get("MISSING_RATE"), triggered_alerts)

        max_outlier = 0.0
        if not qual_df.empty and "outlier_rate" in qual_df.columns:
            max_outlier = float(qual_df["outlier_rate"].max())
        mr = MetricResult(metric_code="OUTLIER_RATE", current_value=max_outlier,
                          metric_detail={"source": "qual_df_max", "window_count": len(perf_df)})
        mid = await _persist_summary_metric(mr, triggered=False)
        await self._emit_alert(monitoring_run_id, mid, mr, self.rules.get("OUTLIER_RATE"), triggered_alerts)

        # AUC 趋势斜率
        if auc_slope is not None:
            await _persist_summary_metric(MetricResult(metric_code="AUC_TREND", current_value=auc_slope))

        # 检测器汇总
        if not detector_df.empty:
            alarm_count = int(detector_df["alarm_flag"].sum())
            mr = MetricResult(metric_code="DETECTOR_ALARMS", current_value=float(alarm_count),
                              metric_detail={"detector_df_rows": len(detector_df)})
            await _persist_summary_metric(mr)

        # ── 统一阈值评估：遍历未单独处理告警的指标，套用规则 + _emit_alert ──
        _already_alerted = {"FEATURE_PSI", "MISSING_RATE", "OUTLIER_RATE",
                            "SCHEMA_CONSISTENCY", "SAMPLE_SIZE", "DETECTOR_ALARMS",
                            "AUC_TREND", "MAX_FEATURE_PSI_7D", "MAX_FEATURE_PSI_30D"}
        for mr in all_metrics:
            if mr.metric_code in _already_alerted:
                continue
            rule = self.rules.get(mr.metric_code)
            if rule and mr.availability_status == AvailabilityStatus.AVAILABLE:
                mid = metric_ids.get(id(mr))
                if mid is None:
                    mid = await self._persist_metric(monitoring_run_id, mr, triggered=False)
                    metric_ids[id(mr)] = mid
                await self._emit_alert(monitoring_run_id, mid, mr, rule, triggered_alerts,
                                       object_code=model_id)

        # ── 告警组装 ──
        has_alerts = len(triggered_alerts) > 0
        max_sev = None
        if triggered_alerts:
            sev_order = {"CRITICAL": 4, "HIGH": 3, "WARNING": 2, "INFO": 1}
            max_sev = max(triggered_alerts, key=lambda a: sev_order.get(a.severity.value, 0)).severity

        alert_context = AlertContext(
            schema_version="V2-WP08",
            trace_id=trace_id, monitoring_run_id=monitoring_run_id,
            model_id=model_id, model_version=champion_version,
            monitor_window_id="W3", baseline_id=baseline.baseline_id,
            data_track=DataTrack.NATURAL,
            anomaly_probability=None,  # 需要已训练的 Sentinel 模型
            top_signals=[],
            alert_details=triggered_alerts,
        )

        # ── B1 持续性判定 ──
        from .persistence_judgment import PersistenceJudgmentService
        judgment_svc = PersistenceJudgmentService(self.session)
        judgment = await judgment_svc.judge(monitoring_run_id)
        diagnosis_status = "PENDING" if judgment.trigger_diagnosis else "SKIPPED"
        await self.repo.update_persistence_judgment(
            monitoring_run_id, judgment.__dict__, diagnosis_status,
        )
        logger.info("persistence_judgment_complete",
                     monitoring_run_id=monitoring_run_id,
                     trigger_diagnosis=judgment.trigger_diagnosis,
                     decay_degree=judgment.decay_degree)

        await self.repo.complete_run(
            monitoring_run_id=monitoring_run_id, overall_status="COMPLETED",
            alert_count=len(triggered_alerts),
            max_alert_severity=max_sev.value if max_sev else None,
            alert_context_json=alert_context.model_dump(),
        )
        await self.session.commit()
        logger.info("full_pipeline_completed", monitoring_run_id=monitoring_run_id,
                     alert_count=len(triggered_alerts),
                     window_count=len(perf_df),
                     drift_records=len(drift_df),
                     detector_alarms=int(detector_df["alarm_flag"].sum()) if not detector_df.empty else 0)

        return MonitoringRunResult(
            monitoring_run_id=monitoring_run_id, has_alerts=has_alerts,
            alert_count=len(triggered_alerts), max_alert_severity=max_sev,
            alerts=triggered_alerts,
            metrics=[_metric_to_dict(m) for m in all_metrics],
        )

    # ── V2 增强方法 ──

    def build_baseline(
        self,
        w0_data: pd.DataFrame,
        model_id: str,
        model_version: str,
        feature_names: list[str] | None = None,
        categorical_features: dict[str, list[str]] | None = None,
        baseline_version: str = "V1",
    ) -> MonitoringBaseline:
        """在 W0 数据上构建监控基线包。

        基线包包含分箱规则、特征概要、性能基准和分数分箱边界。
        只需构建一次，之后冻结不变。所有后续 PSI/漂移计算
        都以这个基线为参照。

        Args:
            w0_data: W0 窗口完整数据（含特征列 + y_true + y_pred_proba）。
            model_id: 模型标识。
            model_version: 模型版本。
            feature_names: 特征列名列表（默认自动检测 feature_ 前缀）。
            categorical_features: 类别特征映射。
            baseline_version: 基线版本号。

        Returns:
            MonitoringBaseline 对象。
        """
        return build_monitoring_baseline(
            w0_data=w0_data,
            model_id=model_id,
            model_version=model_version,
            baseline_id=f"BASELINE_{model_id}_{model_version}_{baseline_version}",
            baseline_version=baseline_version,
            feature_names=feature_names,
            categorical_features=categorical_features,
        )

    async def run_rolling(
        self,
        model_id: str,
        champion_version: str,
        data: pd.DataFrame,
        reference_data: pd.DataFrame,
        baseline: MonitoringBaseline,
        window_days: int = 7,
        step_days: int = 1,
        trace_id: str | None = None,
        min_samples: int = 2000,
        min_bad: int = 50,
    ) -> dict:
        """多窗口滚动监控 — 按时间滑动，逐窗口推进检测器状态。

        在每个窗口上计算完整的性能 + 质量 + 漂移指标，
        检测器跨窗口保持状态，最后构建 Sentinel 特征向量。

        Args:
            model_id: 模型标识。
            champion_version: Champion 版本号。
            data: 完整时序数据（含 apply_time 列）。
            reference_data: W0 参照数据。
            baseline: 监控基线包。
            window_days: 每个窗口的天数（7 或 30）。
            step_days: 滑动步长。
            trace_id: 追踪 ID。
            min_samples: 性能指标最小样本门槛。
            min_bad: 性能指标最小坏样本门槛。

        Returns:
            {
                "window_count": int,
                "performance_df": DataFrame,
                "quality_df": DataFrame,
                "drift_df": DataFrame,
                "detector_df": DataFrame,
                "feature_df": DataFrame,
                "alert_summary": dict,
            }
        """
        trace_id = trace_id or str(uuid.uuid4())

        reference_scores = (
            pd.to_numeric(reference_data["y_pred_proba"], errors="coerce").dropna()
            if "y_pred_proba" in reference_data
            else pd.Series(dtype=float)
        )

        perf_rows: list[dict] = []
        qual_rows: list[dict] = []
        drift_rows: list[dict] = []
        window_meta: list[dict] = []

        # 逐窗口迭代
        for start, end, window_data in iter_rolling_windows(
            data, window_days=window_days, step_days=step_days, require_full_window=False
        ):
            window_id = f"ROLLING_{window_days}D_{start:%Y%m%d}_{end:%Y%m%d}"
            sample_count = len(window_data)
            bad_count = (
                int(window_data["y_true"].sum()) if "y_true" in window_data else None
            )

            # 性能指标
            label_ready = (
                "y_true" in window_data
                and sample_count >= min_samples
                and bad_count is not None
                and bad_count >= min_bad
            )
            if label_ready:
                perf = compute_monitoring_performance_metrics(
                    window_data["y_true"],
                    window_data["y_pred_proba"],
                    window_data["risk_score"] if "risk_score" in window_data.columns else None,
                )
            else:
                perf = {
                    k: None
                    for k in ("auc", "ks", "pr_auc", "brier", "ece", "bad_recall")
                }

            # 预测输出
            if "y_pred_proba" in window_data and not reference_scores.empty:
                out = output_metrics(
                    window_data["y_pred_proba"],
                    reference_scores,
                    baseline.score_edges,
                )
            else:
                out = {}

            common = {
                "trace_id": trace_id,
                "model_id": model_id,
                "model_version": champion_version,
                "baseline_id": baseline.baseline_id,
                "baseline_version": baseline.baseline_version,
                "monitor_window_id": window_id,
                "window_start": start,
                "window_end": end,
                "window_days": window_days,
                "sample_count": sample_count,
                "bad_count": bad_count,
                "data_track": "NATURAL",
                "scenario_id": float("nan"),
                "scenario_instance_id": float("nan"),
            }

            perf_rows.append({**common, **perf, **out})

            # 特征漂移 + 数据质量
            p_positions: list[int] = []
            p_values: list[float | None] = []
            for fname in baseline.feature_names:
                rule = baseline.binning_rules_json.get(fname)
                if rule is None or fname not in window_data.columns:
                    continue

                # 质量
                if fname in baseline.feature_profiles:
                    quality = feature_quality(
                        window_data[fname],
                        pd.Series(baseline.feature_profiles[fname]),
                        rule["feature_type"],
                    )
                    quality["feature_name"] = fname
                    qual_rows.append({**common, **quality})

                # 漂移
                if fname not in reference_data.columns:
                    continue
                if rule["feature_type"] == "categorical":
                    drift = categorical_drift(
                        reference_data[fname],
                        window_data[fname],
                        rule["categories"],
                    )
                    row = {
                        "feature_name": fname, "feature_type": "categorical",
                        **drift, "wasserstein_distance": None,
                        "ks_statistic": None, "ks_p_value": None, "ks_q_value": None,
                    }
                else:
                    drift = continuous_drift(
                        reference_data[fname],
                        window_data[fname],
                        rule["edges"],
                    )
                    row = {
                        "feature_name": fname, "feature_type": "continuous",
                        **drift, "category_share_change": None,
                        "unknown_category_rate": 0.0, "ks_q_value": None,
                    }
                    if row.get("ks_p_value") is not None:
                        p_positions.append(len(drift_rows))
                        p_values.append(row["ks_p_value"])
                drift_rows.append({**common, **row})

            # BH 校正
            for pos, q_val in zip(p_positions, benjamini_hochberg(p_values)):
                drift_rows[pos]["ks_q_value"] = q_val

            window_meta.append(common)

        # 构建 DataFrames
        perf_df = pd.DataFrame(perf_rows)
        qual_df = pd.DataFrame(qual_rows)
        drift_df = pd.DataFrame(drift_rows)

        # 运行检测器（跨窗口有状态）
        features_input = perf_df.copy()
        if not drift_df.empty:
            psi_series = drift_df.groupby("monitor_window_id")["psi"].max()
            features_input["max_feature_psi_7d"] = features_input["monitor_window_id"].map(psi_series)
        features_input["prediction_mean"] = features_input.get("prediction_mean", None)
        if "auc" not in features_input:
            features_input["auc"] = None
        if "ks" not in features_input:
            features_input["ks"] = None

        detector_df = run_detectors(features_input)

        # 构建 Sentinel 特征向量
        feature_df = build_monitor_feature_vector(perf_df, qual_df, drift_df, detector_df)
        feature_df = select_canonical_sentinel_rows(feature_df)

        return {
            "window_count": len(perf_rows),
            "performance_df": perf_df,
            "quality_df": qual_df,
            "drift_df": drift_df,
            "detector_df": detector_df,
            "feature_df": feature_df,
        }

    def infer_sentinel_on_features(
        self,
        sentinel_bundle: SentinelBundle,
        feature_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """在 Sentinel 特征矩阵上执行推理。

        Args:
            sentinel_bundle: 已加载的 Sentinel 模型包。
            feature_df: build_monitor_feature_vector() 的输出。

        Returns:
            infer_sentinel() 的推理结果 DataFrame。
        """
        return infer_sentinel(sentinel_bundle, feature_df)

    def build_sentinel_alerts(self, sentinel_results: pd.DataFrame) -> list[AlertEvent]:
        """从 Sentinel 推理结果构建告警事件。

        Args:
            sentinel_results: infer_sentinel() 的输出。

        Returns:
            AlertEvent 列表。
        """
        return build_alerts(sentinel_results)

    def get_alert_summary(self, events: list[AlertEvent]) -> dict:
        """汇总告警事件统计。"""
        return alert_summary(events)

    async def run_detailed(
        self,
        model_id: str,
        champion_version: str,
        baseline_data: list[dict],
        current_data: list[dict],
        reference_data: list[dict] | None = None,
        binning_rules: dict[str, dict] | None = None,
        feature_names: list[str] | None = None,
        baseline_window_id: str = "",
        current_window_id: str = "",
        data_track: str = "NATURAL",
        trace_id: str | None = None,
        min_samples: int = 2000,
        min_bad: int = 50,
    ) -> MonitoringRunResult:
        """完整模式监控运行（V2）— 使用交接包全套算法。

        包含：性能指标 + 数据质量 + 特征漂移（PSI/JS/KS/Wasserstein）
        + BH 多重检验校正 + 4 个检测器 + 趋势斜率。

        Args:
            model_id: 模型标识。
            champion_version: Champion 版本号。
            baseline_data: 基线数据集（用于对比）。
            current_data: 当前窗口数据集。
            reference_data: W0 参照数据集（如有，用于冻结分箱 PSI）。
            binning_rules: 从基线包加载的分箱规则。
            feature_names: 特征列名列表。
            baseline_window_id: 基线窗口 ID。
            current_window_id: 当前窗口 ID。
            data_track: 数据轨道（NATURAL 或 SCENARIO）。
            trace_id: 追踪 ID。
            min_samples: 性能指标最小样本数门槛（默认 2000）。
            min_bad: 性能指标最小坏样本数门槛（默认 50）。
        """
        trace_id = trace_id or str(uuid.uuid4())

        # ① 创建 monitoring_run 记录
        run = await self.repo.create_run(
            model_id=model_id,
            champion_version=champion_version,
            baseline_window_id=baseline_window_id or "W0",
            current_window_id=current_window_id or "W3",
            data_track=data_track,
            trace_id=trace_id,
        )
        monitoring_run_id = run["monitoring_run_id"]
        logger.info("monitoring_run_detailed_started", monitoring_run_id=monitoring_run_id, model_id=model_id)

        # ② 标签成熟度检查
        availability_issues: dict[str, AvailabilityStatus] = {}
        for label, win_id in [("baseline", baseline_window_id), ("current", current_window_id)]:
            if not win_id:
                continue
            window = {"window_id": win_id, "allows_monitoring_label": True}
            if not self.policy.can_read_labels(window, "TASK_1"):
                availability_issues[label] = AvailabilityStatus.LABEL_NOT_MATURE

        # ③ 转换为 DataFrame
        df_current = pd.DataFrame(current_data)
        df_baseline = pd.DataFrame(baseline_data) if baseline_data else None
        df_reference = pd.DataFrame(reference_data) if reference_data else df_baseline

        sample_count = len(df_current)
        bad_count = int(df_current["y_true"].sum()) if "y_true" in df_current else None

        # 检查性能指标的标签门槛
        label_ready = True
        if "y_true" not in df_current:
            label_ready = False
        elif sample_count < min_samples or bad_count is None or bad_count < min_bad:
            label_ready = False

        # ④ 性能指标
        perf: dict[str, float | None] = {}
        if label_ready and "y_true" in df_current and "y_pred_proba" in df_current:
            perf = compute_monitoring_performance_metrics(
                df_current["y_true"],
                df_current["y_pred_proba"],
                df_current["risk_score"] if "risk_score" in df_current.columns else None,
            )
        else:
            perf = {k: None for k in ("auc", "ks", "pr_auc", "brier", "ece", "bad_recall")}

        # ⑤ 特征漂移 + 数据质量（如果有分箱规则和参考数据）
        quality_rows: list[dict] = []
        drift_rows: list[dict] = []
        if binning_rules and feature_names and df_reference is not None:
            p_positions: list[int] = []
            p_values: list[float | None] = []

            for feature in feature_names:
                rule = binning_rules.get(feature)
                if rule is None or feature not in df_current.columns or feature not in df_reference.columns:
                    continue

                feat_type = rule.get("feature_type", "continuous")

                # 数据质量
                if "baseline_profile" in rule:
                    quality = feature_quality(
                        df_current[feature],
                        pd.Series(rule["baseline_profile"]),
                        feat_type,
                    )
                    quality["feature_name"] = feature
                    quality_rows.append(quality)

                # 漂移
                if feat_type == "categorical":
                    drift = categorical_drift(
                        df_reference[feature], df_current[feature], rule.get("categories", [])
                    )
                    row = {
                        "feature_name": feature, "feature_type": "categorical",
                        **drift, "wasserstein_distance": None,
                        "ks_statistic": None, "ks_p_value": None, "ks_q_value": None,
                    }
                else:
                    drift = continuous_drift(
                        df_reference[feature], df_current[feature], rule.get("edges", [])
                    )
                    row = {
                        "feature_name": feature, "feature_type": "continuous",
                        **drift, "category_share_change": None,
                        "unknown_category_rate": 0.0, "ks_q_value": None,
                    }
                    if row.get("ks_p_value") is not None:
                        p_positions.append(len(drift_rows))
                        p_values.append(row["ks_p_value"])

                drift_rows.append(row)

            # BH 校正
            for pos, q_val in zip(p_positions, benjamini_hochberg(p_values)):
                drift_rows[pos]["ks_q_value"] = q_val

        # ⑥ 汇总指标并持久化
        all_metrics: list[MetricResult] = []
        triggered_alerts: list[AlertDetail] = []

        # 性能指标
        for metric_code in ("AUC", "KS"):
            if metric_code.lower() in perf:
                val = perf[metric_code.lower()]
                mr = MetricResult(
                    metric_code=metric_code,
                    current_value=val,
                    availability_status=(
                        AvailabilityStatus.AVAILABLE
                        if val is not None
                        else AvailabilityStatus.LABEL_NOT_MATURE
                    ),
                )
                all_metrics.append(mr)
                await self._persist_metric(monitoring_run_id, mr)

        # 漂移汇总
        if drift_rows:
            psi_values = [r["psi"] for r in drift_rows if r.get("psi") is not None]
            if psi_values:
                mean_psi = float(np.mean(psi_values))
                max_psi = float(np.max(psi_values))
                mr = MetricResult(
                    metric_code="FEATURE_PSI",
                    current_value=mean_psi,
                    metric_detail={"max_psi": max_psi},
                )
                all_metrics.append(mr)
                metric_id = await self._persist_metric(monitoring_run_id, mr)

                # 阈值比对
                rule = self.rules.get("FEATURE_PSI")
                if rule:
                    triggered, severity = rule.evaluate(max_psi, max_psi)
                    if triggered and severity:
                        alert_type = await self.knowledge.resolve_alert("FEATURE_PSI", severity)
                        alert_code = alert_type.alert_code if alert_type else "ANOMALY_FEATURE_PSI"
                        detail = AlertDetail(
                            alert_id=str(uuid.uuid4()),
                            alert_code=alert_code,
                            severity=severity,
                            object_type=ObjectType.FEATURE,
                            object_code="ALL",
                            metric_code="FEATURE_PSI",
                            metric_version="V2",
                            baseline_value=0.0,
                            current_value=max_psi,
                            delta=max_psi,
                            threshold=rule.critical_threshold,
                            rule_type=RuleType.SHIFT_THRESHOLD,
                            threshold_rule_id=rule.rule_id,
                            threshold_rule_version=rule.rule_version,
                            availability_status=AvailabilityStatus.AVAILABLE,
                            metric_detail={"max_psi": max_psi, "mean_psi": mean_psi},
                            created_at=datetime.now(timezone.utc),
                        )
                        triggered_alerts.append(detail)
                        await self._persist_alert(monitoring_run_id, metric_id, mr, detail)

        # 数据质量分数
        if quality_rows:
            dq_scores = [r["dq_score"] for r in quality_rows]
            mean_dq = float(np.mean(dq_scores))
            mr = MetricResult(
                metric_code="DATA_QUALITY_SCORE",
                current_value=mean_dq,
                metric_detail={"worst_flag": min((r["dq_flag"] for r in quality_rows), key=lambda f: {"OK": 2, "WARN": 1, "ALERT": 0}.get(f, 0))},
            )
            all_metrics.append(mr)
            await self._persist_metric(monitoring_run_id, mr)

        # ⑦ 运行检测器
        feature_df = pd.DataFrame({
            "model_id": [model_id] * max(1, len(drift_rows)),
            "model_version": [champion_version] * max(1, len(drift_rows)),
            "data_track": [data_track] * max(1, len(drift_rows)),
            "monitor_window_id": [current_window_id or "W3"] * max(1, len(drift_rows)),
            "auc": [perf.get("auc")],
            "ks": [perf.get("ks")],
            "prediction_mean": [float(df_current["y_pred_proba"].mean()) if "y_pred_proba" in df_current else None],
            "max_feature_psi_7d": [float(max([r["psi"] for r in drift_rows if r.get("psi") is not None])) if drift_rows else None],
            "missing_rate_max_delta": [float(max([abs(r.get("missing_rate_delta", 0.0)) for r in quality_rows])) if quality_rows else None],
            "outlier_rate_max_delta": [float(max([abs(r.get("outlier_rate_delta", 0.0)) for r in quality_rows])) if quality_rows else None],
        })
        detector_df = run_detectors(feature_df)
        if not detector_df.empty:
            logger.info(
                "detector_signals",
                monitoring_run_id=monitoring_run_id,
                alarm_count=int(detector_df["alarm_flag"].sum()),
            )

        # ⑧ 趋势斜率
        if drift_rows:
            psi_series = [r["psi"] for r in drift_rows if r.get("psi") is not None]
            slope = trailing_slope(psi_series)
            if slope is not None:
                mr = MetricResult(
                    metric_code="PSI_TREND_SLOPE",
                    current_value=slope,
                    metric_detail={"window_count": len(psi_series)},
                )
                all_metrics.append(mr)
                await self._persist_metric(monitoring_run_id, mr)

        # ⑨ 生成 AlertContext + 完成 run
        has_alerts = len(triggered_alerts) > 0
        max_sev = None
        if triggered_alerts:
            sev_order = {"CRITICAL": 4, "HIGH": 3, "WARNING": 2, "INFO": 1}
            max_sev = max(triggered_alerts, key=lambda a: sev_order.get(a.severity.value, 0)).severity

        try:
            track = DataTrack(data_track)
        except ValueError:
            track = DataTrack.NATURAL

        alert_context = AlertContext(
            schema_version="V2",
            trace_id=trace_id,
            monitoring_run_id=monitoring_run_id,
            model_id=model_id,
            model_version=champion_version,
            monitor_window_id=current_window_id or "W3",
            baseline_id=baseline_window_id or "W0",
            data_track=track,
            alert_details=triggered_alerts,
            # V2 新增字段（AlertContext 已支持）
            anomaly_probability=None,  # Sentinel 推理需要已训练的 Sentinel 模型
            top_signals=[],
        )

        await self.repo.complete_run(
            monitoring_run_id=monitoring_run_id,
            overall_status="COMPLETED",
            alert_count=len(triggered_alerts),
            max_alert_severity=max_sev.value if max_sev else None,
            alert_context_json=alert_context.model_dump(),
        )

        await self.session.commit()
        logger.info(
            "monitoring_run_detailed_completed",
            monitoring_run_id=monitoring_run_id,
            alert_count=len(triggered_alerts),
            drift_feature_count=len(drift_rows),
            quality_feature_count=len(quality_rows),
        )

        return MonitoringRunResult(
            monitoring_run_id=monitoring_run_id,
            has_alerts=has_alerts,
            alert_count=len(triggered_alerts),
            max_alert_severity=max_sev,
            alerts=triggered_alerts,
            metrics=[_metric_to_dict(m) for m in all_metrics],
        )

    async def run_parallel_cycle(
        self,
        model_ids: list[str],
        champion_versions: dict[str, str],
        data_provider: Any = None,
        max_concurrency: int = 30,
    ) -> dict:
        """多模型并发监控周期（V2）— 使用 asyncio 替代 ThreadPoolExecutor。

        Args:
            model_ids: 要监控的模型 ID 列表。
            champion_versions: {model_id: version} 映射。
            data_provider: 数据提供者（含 load_window 等方法），如为 None 则使用合成数据。
            max_concurrency: 最大并发数。

        Returns:
            {
                "status": "PASS" | "FAIL",
                "success_count": int,
                "failure_count": int,
                "results": list[dict],
                "failures": list[dict],
            }
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def monitor_one(model_id: str) -> dict:
            async with semaphore:
                try:
                    version = champion_versions.get(model_id, "champion_v1")
                    from .window_loader import load_window_with_predictions

                    baseline_df = load_window_with_predictions("W0", model_id)
                    current_df = load_window_with_predictions("W3", model_id)
                    baseline_data = baseline_df.to_dict(orient="records")
                    current_data = current_df.to_dict(orient="records")

                    result = await self.run_detailed(
                        model_id=model_id,
                        champion_version=version,
                        baseline_data=baseline_data,
                        current_data=current_data,
                        baseline_window_id="W0",
                        current_window_id="W3",
                    )
                    return {
                        "model_id": model_id,
                        "status": "PASS",
                        "monitoring_run_id": result.monitoring_run_id,
                        "alert_count": result.alert_count,
                    }
                except Exception as exc:
                    return {
                        "model_id": model_id,
                        "status": "FAIL",
                        "error": f"{type(exc).__name__}: {exc}",
                    }

        tasks = [monitor_one(mid) for mid in model_ids]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        successes = [r for r in results if r.get("status") == "PASS"]
        failures = [r for r in results if r.get("status") != "PASS"]

        return {
            "status": "PASS" if len(successes) == len(model_ids) else "FAIL",
            "success_count": len(successes),
            "failure_count": len(failures),
            "results": successes,
            "failures": failures,
        }

    # ── 私有方法 ──

    async def _persist_metric(self, run_id: str, mr: MetricResult, *, triggered: bool | None = None) -> str:
        result = await self.repo.insert_metric(
            monitoring_run_id=run_id,
            metric_code=mr.metric_code,
            metric_version="V1",
            object_type="MODEL",
            baseline_value=mr.baseline_value,
            current_value=mr.current_value,
            delta=mr.delta,
            triggered=triggered if triggered is not None else (mr.availability_status == AvailabilityStatus.AVAILABLE),
            availability_status=mr.availability_status.value,
            metric_detail=mr.metric_detail,
        )
        return result["metric_id"]

    async def _persist_alert(self, run_id: str, metric_id: str, mr: MetricResult, alert: AlertDetail) -> None:
        await self.repo.insert_alert(
            monitoring_run_id=run_id,
            metric_id=metric_id,
            alert_code=alert.alert_code,
            severity=alert.severity.value,
            object_type=alert.object_type.value,
            object_code=alert.object_code,
            metric_code=mr.metric_code,
            metric_version="V1",
            baseline_value=mr.baseline_value,
            current_value=mr.current_value,
            delta=mr.delta,
            threshold=alert.threshold,
            rule_type=alert.rule_type,
            threshold_rule_id=alert.threshold_rule_id,
            threshold_rule_version=alert.threshold_rule_version,
            availability_status=mr.availability_status.value,
            alert_detail=alert.model_dump(),
        )


def _metric_to_dict(mr: MetricResult) -> dict:
    return {
        "metric_code": mr.metric_code,
        "baseline_value": mr.baseline_value,
        "current_value": mr.current_value,
        "delta": mr.delta,
        "availability_status": mr.availability_status.value,
    }
