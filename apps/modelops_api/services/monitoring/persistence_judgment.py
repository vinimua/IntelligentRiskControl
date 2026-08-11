"""B1 持续性判定服务 — 监控指标 → 衰减程度评估。

输入 monitoring_run_id，读取窗口指标/告警，产出：
- trigger_diagnosis: 是否触发诊断
- decay_degree: SHORT_TERM_7D | SUSTAINED_30D | SEVERE | NONE
- persistence_evidence: 各指标持续计数明细
- dimension_alert_summary: 四维度告警汇总
- recovery_status: 恢复观察 / 恢复确认
- requires_manual_review: 是否强制人工

规格依据：doc/任务一（未完成其他上下游通路，单任务一）.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)

# ── 核心 / 非核心指标划分 ──
CORE_METRICS = {"AUC", "KS", "SCORE_PSI", "FEATURE_PSI", "MISSING_RATE", "OUTLIER_RATE"}
GUARDRAIL_METRICS = {"SCHEMA_CONSISTENCY", "SAMPLE_SIZE"}

# ── window_id 前缀 → 窗口天数 ──


def _window_days_from_id(window_id: str) -> int:
    if window_id.startswith("30D_"):
        return 30
    if window_id.startswith("7D_"):
        return 7
    # fallback: parse from prefix
    try:
        return int(window_id.split("D_")[0])
    except (ValueError, IndexError):
        return 7


def _is_core(metric_code: str) -> bool:
    return metric_code in CORE_METRICS


def _severity_rank(sev: str | None) -> int:
    """CRITICAL=4, HIGH=3, WARNING=2, INFO=1."""
    order = {"CRITICAL": 4, "HIGH": 3, "WARNING": 2, "INFO": 1}
    return order.get(str(sev).upper() if sev else "", 0)


def _is_severe_level(sev: str | None) -> bool:
    return str(sev).upper() in ("CRITICAL", "HIGH")


def _safe_float_for_judgment(val) -> float | None:
    """安全转 float，处理 None / NaN / inf / Decimal。"""
    if val is None:
        return None
    try:
        import math
        f = float(val)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _is_warning_level(sev: str | None) -> bool:
    return str(sev).upper() == "WARNING"


@dataclass
class PersistenceJudgment:
    """B1 持续性判定结果。"""

    trigger_diagnosis: bool = False
    decay_degree: str = "NONE"  # SHORT_TERM_7D | SUSTAINED_30D | SEVERE | NONE
    requires_manual_review: bool = False

    # 窗口维度状态
    status_7d: str = "NORMAL"       # NORMAL | OBSERVING | TRIGGERED
    status_30d: str = "NORMAL"      # NORMAL | OBSERVING | TRIGGERED

    # 证据明细
    persistence_evidence: list[dict] = field(default_factory=list)
    dimension_alert_summary: dict = field(default_factory=dict)
    recovery_status: str = "NONE"

    # 元数据
    judgment_at: str = ""
    judged_by: str = "PersistenceJudgmentService_V1"


class PersistenceJudgmentService:
    """B1 持续性判定 — 从 monitoring 数据判定衰减程度和诊断触发。"""

    def __init__(self, session: AsyncSession):
        self.session = session

    # ── 入口 ──

    async def judge(self, monitoring_run_id: str) -> PersistenceJudgment:
        """执行完整持续性判定。"""
        # ① 加载窗口告警
        window_alerts = await self._load_window_alerts(monitoring_run_id)

        # ② 加载窗口指标
        window_metrics = await self._load_window_metrics(monitoring_run_id)

        # ③ 按 7D/30D + 指标分组计数
        counts_7d = self._count_consecutive(window_alerts, window_days=7)
        counts_30d = self._count_cumulative(window_alerts, window_days=30)

        # ④ 计算 7D / 30D 触发状态
        status_7d = self._compute_window_status(counts_7d, window_days=7)
        status_30d = self._compute_window_status(counts_30d, window_days=30)

        # ⑤ 组合判定
        decay_degree, trigger_diag = self._combine_status(status_7d, status_30d)

        # ⑥ SEVERE 判定
        is_severe, severe_reason = self._check_severe(
            counts_7d, counts_30d, window_alerts, window_metrics,
        )

        if is_severe:
            decay_degree = "SEVERE"
            trigger_diag = True
            # SEVERE 升级时同步更新窗口状态，避免前端显示"严重衰减"但 7D/30D 仍为"正常"
            status_7d = "TRIGGERED"
            status_30d = "TRIGGERED"

        # ⑦ 汇总证据
        evidence = self._build_evidence(counts_7d, counts_30d, window_alerts)
        dimension_summary = self._build_dimension_summary(window_alerts)

        return PersistenceJudgment(
            trigger_diagnosis=trigger_diag,
            decay_degree=decay_degree,
            requires_manual_review=is_severe or (decay_degree == "SEVERE"),
            status_7d=status_7d,
            status_30d=status_30d,
            persistence_evidence=evidence,
            dimension_alert_summary=dimension_summary,
            recovery_status=self._recovery_status(status_7d, status_30d),
            judgment_at=datetime.now(timezone.utc).isoformat(),
        )

    # ── 数据加载 ──

    async def _load_window_alerts(self, monitoring_run_id: str) -> list[dict]:
        """加载所有 V2_EVENT_TIME 窗口指标，逐条套用阈值规则，返回窗口级告警。

        B1 持续性判定的证据层不应依赖 monitoring_alerts（告警产品层），
        而应直接从窗口指标重新评估阈值，避免被告警展示策略（去重/合并/suppress）影响。
        """
        rows = await self.session.execute(
            text("""
                SELECT metric_code, current_value, baseline_value, delta,
                       availability_status,
                       metric_detail->>'window_id' AS window_id,
                       (metric_detail->>'window_days')::int AS window_days,
                       metric_detail->>'window_start' AS window_start
                FROM monitoring.monitoring_metrics
                WHERE monitoring_run_id = :run_id
                  AND metric_version = 'V2_EVENT_TIME'
                  AND metric_detail->>'window_id' IS NOT NULL
                ORDER BY metric_detail->>'window_start' ASC
            """),
            {"run_id": monitoring_run_id},
        )
        metrics = [dict(r._mapping) for r in rows]

        # 复用 DEFAULT_THRESHOLD_RULES，保证与汇总告警同一套阈值
        from .threshold_rules import DEFAULT_THRESHOLD_RULES as RULES

        alerts: list[dict] = []
        for m in metrics:
            code = m.get("metric_code")
            if not code:
                continue
            rule = RULES.get(code)
            if rule is None:
                continue
            if m.get("availability_status") != "AVAILABLE":
                continue

            delta_val = m.get("delta")
            cur_val = m.get("current_value")
            triggered, severity = rule.evaluate(
                _safe_float_for_judgment(delta_val),
                _safe_float_for_judgment(cur_val),
            )
            if triggered and severity:
                alerts.append({
                    "metric_code": code,
                    "severity": severity.value if hasattr(severity, "value") else str(severity),
                    "window_id": m.get("window_id"),
                    "window_days": m.get("window_days"),
                    "current_value": cur_val,
                    "delta": delta_val,
                })

        return alerts

    async def _load_window_metrics(self, monitoring_run_id: str) -> list[dict]:
        """加载所有窗口指标（含未触发告警的）。"""
        rows = await self.session.execute(
            text("""
                SELECT metric_code, current_value, delta, availability_status,
                       metric_detail->>'window_id' AS window_id,
                       (metric_detail->>'window_days')::int AS window_days,
                       metric_detail->>'window_start' AS window_start
                FROM monitoring.monitoring_metrics
                WHERE monitoring_run_id = :run_id
                  AND metric_detail->>'window_id' IS NOT NULL
                ORDER BY metric_detail->>'window_start' ASC
            """),
            {"run_id": monitoring_run_id},
        )
        return [dict(r._mapping) for r in rows]

    # ── 计数逻辑 ──

    def _count_consecutive(self, alerts: list[dict], window_days: int) -> dict[str, dict]:
        """7D 连续窗口计数：按指标 + 告警级别，统计最近连续异常窗口数。

        Returns: {metric_code: {"WARNING": N, "HIGH": N, "CRITICAL": N, "max_consecutive": N}}
        """
        # 按 window_id 排序（按时间）
        target_alerts = [a for a in alerts if a.get("window_days") == window_days]
        if not target_alerts:
            return {}

        # 按 window_id 分组，取每个窗口最严重告警
        window_max_sev: dict[str, dict[str, int]] = {}  # window_id -> {metric: max_sev_rank}
        for a in target_alerts:
            wid = a["window_id"]
            mc = a["metric_code"]
            sev_rank = _severity_rank(a.get("severity"))
            if wid not in window_max_sev:
                window_max_sev[wid] = {}
            window_max_sev[wid][mc] = max(window_max_sev[wid].get(mc, 0), sev_rank)

        # 获取按时间排序的唯一窗口
        sorted_windows = sorted(window_max_sev.keys())

        # 对每个指标，统计最近连续异常窗口
        result: dict[str, dict] = {}
        all_metrics = set()
        for wd in window_max_sev.values():
            all_metrics.update(wd.keys())

        for mc in all_metrics:
            consecutive = 0
            max_consecutive = 0
            crit_high_consecutive = 0
            max_crit_high_consecutive = 0
            warning_consecutive = 0
            max_warning_consecutive = 0
            warning_count = 0
            high_count = 0
            critical_count = 0

            for wid in sorted_windows:
                sev = window_max_sev[wid].get(mc, 0)
                if sev >= 2:  # WARNING+
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                    if sev >= 3:  # HIGH/CRITICAL
                        crit_high_consecutive += 1
                        max_crit_high_consecutive = max(max_crit_high_consecutive, crit_high_consecutive)
                    else:  # WARNING only
                        warning_consecutive += 1
                        max_warning_consecutive = max(max_warning_consecutive, warning_consecutive)
                        crit_high_consecutive = 0

                    if sev == 2:
                        warning_count += 1
                    elif sev == 3:
                        high_count += 1
                    elif sev >= 4:
                        critical_count += 1
                else:
                    consecutive = 0
                    crit_high_consecutive = 0
                    warning_consecutive = 0

            result[mc] = {
                "consecutive_windows": consecutive,
                "max_consecutive": max_consecutive,
                "max_consecutive_crit_high": max_crit_high_consecutive,
                "max_consecutive_warning": max_warning_consecutive,
                "WARNING": warning_count,
                "HIGH": high_count,
                "CRITICAL": critical_count,
            }

        return result

    def _count_cumulative(self, alerts: list[dict], window_days: int) -> dict[str, dict]:
        """30D 累计评估点计数：按指标 + 告警级别，统计累计异常评估点数。

        Returns: 与 _count_consecutive 同结构，consecutive_windows = 累计总数。
        """
        target_alerts = [a for a in alerts if a.get("window_days") == window_days]
        if not target_alerts:
            return {}

        window_max_sev: dict[str, dict[str, int]] = {}
        for a in target_alerts:
            wid = a["window_id"]
            mc = a["metric_code"]
            sev_rank = _severity_rank(a.get("severity"))
            if wid not in window_max_sev:
                window_max_sev[wid] = {}
            window_max_sev[wid][mc] = max(window_max_sev[wid].get(mc, 0), sev_rank)

        all_metrics = set()
        for wd in window_max_sev.values():
            all_metrics.update(wd.keys())

        result: dict[str, dict] = {}
        for mc in all_metrics:
            warning_count = 0
            high_count = 0
            critical_count = 0
            for wid in window_max_sev:
                sev = window_max_sev[wid].get(mc, 0)
                if sev == 2:
                    warning_count += 1
                elif sev == 3:
                    high_count += 1
                elif sev >= 4:
                    critical_count += 1

            total = warning_count + high_count + critical_count
            result[mc] = {
                "consecutive_windows": total,  # 累计总数
                "max_consecutive": total,
                "WARNING": warning_count,
                "HIGH": high_count,
                "CRITICAL": critical_count,
            }

        return result

    # ── 窗口状态判定 ──

    def _compute_window_status(
        self, counts: dict[str, dict], window_days: int,
    ) -> str:
        """根据各指标计数判定该窗口维度的整体状态。

        Returns: NORMAL | OBSERVING | TRIGGERED
        """
        if not counts:
            return "NORMAL"

        triggered = False
        observing = False

        for mc, cnt in counts.items():
            core = _is_core(mc)

            if window_days == 7:
                # 7D：连续窗口计数，用 max_consecutive
                crit_high_consecutive = cnt.get("max_consecutive_crit_high", 0)
                warning_consecutive = cnt.get("max_consecutive_warning", 0)

                if crit_high_consecutive > 0:
                    threshold = 2  # core 和 non-core 对严重级别都是 2
                    if crit_high_consecutive >= threshold:
                        triggered = True
                        break
                if warning_consecutive > 0:
                    threshold = 3 if core else 5
                    if warning_consecutive >= threshold:
                        triggered = True
                        break

                # 观察中：有连续异常但未达阈值
                if cnt.get("max_consecutive", 0) > 0:
                    observing = True
            else:
                # 30D：累计评估点计数
                if cnt["CRITICAL"] > 0 or cnt["HIGH"] > 0:
                    threshold = 2 if core else 3
                    if cnt["CRITICAL"] + cnt["HIGH"] >= threshold:
                        triggered = True
                        break
                if cnt["WARNING"] > 0:
                    threshold = 3 if core else 4
                    if cnt["WARNING"] >= threshold:
                        triggered = True
                        break

                if cnt["consecutive_windows"] > 0:
                    observing = True

        if triggered:
            return "TRIGGERED"
        if observing:
            return "OBSERVING"
        return "NORMAL"

    # ── 组合判定矩阵（§3）──

    def _combine_status(self, status_7d: str, status_30d: str) -> tuple[str, bool]:
        """7D × 30D 组合判定 → (decay_degree, trigger_diagnosis)。"""
        matrix = {
            ("NORMAL", "NORMAL"): ("NONE", False),
            ("NORMAL", "OBSERVING"): ("NONE", False),
            ("NORMAL", "TRIGGERED"): ("SUSTAINED_30D", True),
            ("OBSERVING", "NORMAL"): ("NONE", False),
            ("OBSERVING", "OBSERVING"): ("NONE", False),
            ("OBSERVING", "TRIGGERED"): ("SUSTAINED_30D", True),
            ("TRIGGERED", "NORMAL"): ("SHORT_TERM_7D", True),
            ("TRIGGERED", "OBSERVING"): ("SHORT_TERM_7D", True),
            ("TRIGGERED", "TRIGGERED"): ("SUSTAINED_30D", True),
        }
        result = matrix.get((status_7d, status_30d), ("NONE", False))
        return result

    # ── SEVERE 判定（§4）──

    def _check_severe(
        self,
        counts_7d: dict[str, dict],
        counts_30d: dict[str, dict],
        alerts: list[dict],
        metrics: list[dict],
    ) -> tuple[bool, str]:
        """检查 SEVERE 条件。

        1. 核心指标 CRITICAL + 短长同现
        2. 护栏失败：SCHEMA_CONSISTENCY 不一致 / MISSING_RATE >= 0.40 / SAMPLE_SIZE < 50
        3. 恢复率大幅下降：核心指标 Δ >= 0.05（AUC/KS）连续 >= 2 窗口
        """
        reasons = []

        # 1. 核心 CRITICAL + 短长同现
        for mc in CORE_METRICS:
            cnt_7 = counts_7d.get(mc, {})
            cnt_30 = counts_30d.get(mc, {})
            has_7d_crit = cnt_7.get("CRITICAL", 0) > 0
            has_30d_crit = cnt_30.get("CRITICAL", 0) > 0
            if has_7d_crit and has_30d_crit:
                reasons.append(f"core_critical_both_windows:{mc}")

        # 2. 护栏失败
        for a in alerts:
            if a.get("metric_code") == "SCHEMA_CONSISTENCY":
                reasons.append("guardrail_schema_inconsistent")
            if a.get("metric_code") == "MISSING_RATE":
                try:
                    val = float(a.get("current_value", 0))
                except (ValueError, TypeError):
                    val = 0.0
                if val >= 0.40:
                    reasons.append("guardrail_missing_rate_blocking")

        for m in metrics:
            if m.get("metric_code") == "SAMPLE_SIZE":
                try:
                    val = float(m.get("current_value", 0))
                except (ValueError, TypeError):
                    val = 200
                if val < 50:
                    reasons.append("guardrail_sample_size_critical")

        # 3. 核心指标恢复率大幅下降（AUC/KS Δ >= 0.05 连续 >= 2 窗口）
        for mc in ("AUC", "KS"):
            cnt = counts_7d.get(mc, {})
            if cnt.get("consecutive_windows", 0) >= 2:
                # 检查 delta
                relevant = [a for a in alerts if a.get("metric_code") == mc]
                large_drops = 0
                for a in relevant:
                    try:
                        d = abs(float(a.get("delta", 0) or 0))
                    except (ValueError, TypeError):
                        d = 0.0
                    if d >= 0.05:
                        large_drops += 1
                if large_drops >= 2:
                    reasons.append(f"severe_drop_{mc}_delta>=0.05_x{large_drops}")

        is_severe = len(reasons) > 0
        return is_severe, ";".join(reasons) if reasons else ""

    # ── 证据构建 ──

    def _build_evidence(
        self, counts_7d: dict, counts_30d: dict, alerts: list[dict],
    ) -> list[dict]:
        """汇总持续性证据。"""
        evidence = []
        all_metrics = set(list(counts_7d.keys()) + list(counts_30d.keys()))
        for mc in sorted(all_metrics):
            cnt_7d = counts_7d.get(mc, {})
            cnt_30d = counts_30d.get(mc, {})
            severity_counts = {
                "WARNING": int(cnt_7d.get("WARNING", 0) or 0) + int(cnt_30d.get("WARNING", 0) or 0),
                "HIGH": int(cnt_7d.get("HIGH", 0) or 0) + int(cnt_30d.get("HIGH", 0) or 0),
                "CRITICAL": int(cnt_7d.get("CRITICAL", 0) or 0) + int(cnt_30d.get("CRITICAL", 0) or 0),
            }
            max_severity = (
                "CRITICAL" if severity_counts["CRITICAL"] > 0
                else "HIGH" if severity_counts["HIGH"] > 0
                else "WARNING" if severity_counts["WARNING"] > 0
                else None
            )
            entry = {
                "metric_code": mc,
                "is_core": _is_core(mc),
                "count_7d": cnt_7d,
                "count_30d": cnt_30d,
                "window_count_7d": len(set(a["window_id"] for a in alerts
                                            if a.get("metric_code") == mc
                                            and a.get("window_days") == 7)),
                "window_count_30d": len(set(a["window_id"] for a in alerts
                                             if a.get("metric_code") == mc
                                             and a.get("window_days") == 30)),
                "consecutive_count": cnt_7d.get("max_consecutive", 0),
                "max_severity": max_severity,
            }
            evidence.append(entry)
        return evidence

    def _build_dimension_summary(self, alerts: list[dict]) -> dict:
        """按四维度汇总告警。"""
        dim_map = {
            "MODEL": {"AUC", "KS", "SCORE_PSI", "PR_AUC", "BRIER", "ECE", "PREDICTION_MEAN"},
            "DATA": {"MISSING_RATE", "OUTLIER_RATE", "SCHEMA_CONSISTENCY", "SAMPLE_SIZE"},
            "FEATURE": {"FEATURE_PSI", "MAX_FEATURE_PSI_7D", "MAX_FEATURE_PSI_30D"},
            "BUSINESS": {"BAD_RATE"},
        }
        summary: dict[str, dict] = {}
        for dim, metrics in dim_map.items():
            dim_alerts = [a for a in alerts if a.get("metric_code") in metrics]
            warning = sum(1 for a in dim_alerts if str(a.get("severity")).upper() == "WARNING")
            critical = sum(
                1 for a in dim_alerts
                if str(a.get("severity")).upper() in {"HIGH", "CRITICAL"}
            )
            summary[dim] = {
                "total": len(dim_alerts),
                "warning": warning,
                "critical": critical,
                "alert_count": len(dim_alerts),
                "max_severity": max(
                    (_severity_rank(a.get("severity")) for a in dim_alerts), default=0
                ),
                "triggered_metrics": list(set(a.get("metric_code") for a in dim_alerts)),
            }
        return summary

    def _recovery_status(self, status_7d: str, status_30d: str) -> str:
        # TODO: 实现 recovery 状态机（7D 连续 3 正常 → RECOVERY_OBSERVING；30D 连续 2 正常 → RECOVERY_CONFIRMED）
        return "NONE"
