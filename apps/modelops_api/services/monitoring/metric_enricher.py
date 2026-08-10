"""指标富化 — 将 DB 原始指标行合并阈值规则、分类、中文名等信息，供前端"监控判定台"使用。"""

from __future__ import annotations

from .threshold_rules import DEFAULT_THRESHOLD_RULES
from packages.models.common.enums import MetricDirection

# ── 指标分类 ──
# 将 17 个指标映射到 4 个业务组
METRIC_CATEGORY_MAP: dict[str, str] = {
    # 模型性能 (label-dependent)
    "AUC": "performance",
    "KS": "performance",
    "PR_AUC": "performance",
    "BAD_RECALL": "performance",
    "BRIER": "performance",
    "ECE": "performance",
    # 分布漂移
    "FEATURE_PSI": "drift",
    "SCORE_PSI": "drift",
    "PREDICTION_MEAN": "drift",
    "MAX_FEATURE_PSI_7D": "drift",
    "MAX_FEATURE_PSI_30D": "drift",
    # 数据质量
    "MISSING_RATE": "quality",
    "OUTLIER_RATE": "quality",
    "DATA_QUALITY_SCORE": "quality",
    # 数据稳定性
    "BAD_RATE": "stability",
    "SAMPLE_SIZE": "stability",
    "SCHEMA_CONSISTENCY": "stability",
}

# ── 中文展示名 ──
METRIC_DISPLAY_NAMES: dict[str, str] = {
    "AUC": "AUC 模型区分能力",
    "KS": "KS 区分度",
    "PR_AUC": "PR-AUC 精确召回曲线下面积",
    "BAD_RECALL": "Bad Recall 坏样本召回率",
    "BRIER": "Brier 校准误差",
    "ECE": "ECE 期望校准误差",
    "FEATURE_PSI": "Feature PSI 特征分布漂移",
    "SCORE_PSI": "Score PSI 分数分布漂移",
    "PREDICTION_MEAN": "Prediction Mean 预测均值",
    "MAX_FEATURE_PSI_7D": "Max Feature PSI 7D 近7天最大特征PSI",
    "MAX_FEATURE_PSI_30D": "Max Feature PSI 30D 近30天最大特征PSI",
    "MISSING_RATE": "Missing Rate 缺失率变化",
    "OUTLIER_RATE": "Outlier Rate 异常值率变化",
    "DATA_QUALITY_SCORE": "Data Quality Score 数据质量综合分",
    "BAD_RATE": "Bad Rate 坏样本率",
    "SAMPLE_SIZE": "Sample Size 样本量",
    "SCHEMA_CONSISTENCY": "Schema Consistency Schema一致性",
}

# ── 类别中文名 ──
CATEGORY_DISPLAY_NAMES: dict[str, str] = {
    "performance": "模型性能",
    "drift": "分布漂移",
    "quality": "数据质量",
    "stability": "数据稳定性",
}


def compute_threshold_usage_ratio(
    delta: float | None,
    current_value: float | None,
    direction: MetricDirection,
    warning_threshold: float,
) -> float | None:
    """计算当前值距离 Warning 阈值的百分比 (0-1+)。

    返回值含义：
    - 0.0 ~ 1.0: 正常范围内，比例越高越接近阈值
    - >= 1.0: 已超过 Warning 阈值
    - None: 无法计算（无阈值或无可用值）
    """
    if warning_threshold is None or warning_threshold == 0:
        return None

    if direction == MetricDirection.HIGHER_BETTER:
        # delta < 0 下降时才有风险；用 abs(delta) 比 warning_threshold
        if delta is not None and delta < 0:
            return min(abs(delta) / warning_threshold, 2.0)
        if delta is None and current_value is not None:
            # 无 baseline 时无法判断 delta，标记为 0（无法评估）
            return 0.0
        return 0.0  # delta >= 0 表示在改善

    elif direction == MetricDirection.LOWER_BETTER:
        # SAMPLE_SIZE: 当前值低于阈值触发；ratio = 阈值/当前值（值越小越危险）
        if current_value is not None and current_value > 0:
            if current_value <= warning_threshold:
                return 1.0 + (warning_threshold - current_value) / max(warning_threshold, 1)
            return warning_threshold / current_value
        return None

    elif direction == MetricDirection.DEVIATION_BAD:
        # abs(delta) 或 abs(current_value) 比 warning_threshold
        value = abs(delta) if delta is not None else (abs(current_value) if current_value is not None else None)
        if value is not None and warning_threshold > 0:
            return min(value / warning_threshold, 2.0)
        return None

    return None


def build_status_reason(
    metric_code: str,
    availability_status: str,
    rule_enabled: bool,
    triggered: bool,
    severity: str | None,
    usage_ratio: float | None,
    direction: MetricDirection | None,
    warning_threshold: float | None,
    critical_threshold: float | None,
) -> str:
    """生成"为什么正常/为什么告警"的中文解释。"""
    display = METRIC_DISPLAY_NAMES.get(metric_code, metric_code)

    if availability_status != "AVAILABLE":
        status_messages = {
            "LABEL_NOT_MATURE": "标签未成熟，暂时无法计算",
            "DATA_NOT_AVAILABLE": "数据不可用",
            "SAMPLE_TOO_SMALL": "样本量过小，无法可靠计算",
            "NOT_APPLICABLE": "该指标不适用于当前场景",
            "CALCULATION_FAILED": "计算失败",
        }
        return status_messages.get(availability_status, f"不可用（{availability_status}）")

    if not rule_enabled:
        return "已计算，但未配置告警规则，不参与告警判定"

    if triggered and severity:
        sev_labels = {"WARNING": "预警", "HIGH": "高风险", "CRITICAL": "严重"}
        sev_label = sev_labels.get(severity, severity)
        return f"已触发{sev_label}告警"

    if usage_ratio is None:
        return "无法评估（缺少基线或当前值）"

    if direction == MetricDirection.HIGHER_BETTER:
        pct = usage_ratio * 100
        return f"当前只达到预警阈值的 {pct:.1f}%（指标未下降或下降幅度小）"
    elif direction == MetricDirection.LOWER_BETTER:
        pct = usage_ratio * 100
        if usage_ratio < 1.0:
            return f"当前只达到预警阈值的 {pct:.1f}%（当前值仍高于预警阈值）"
        else:
            return f"已使用预警阈值的 {pct:.1f}%"
    elif direction == MetricDirection.DEVIATION_BAD:
        pct = usage_ratio * 100
        return f"当前只达到预警阈值的 {pct:.1f}%（偏差在可控范围内）"

    return "所有受监控指标均未达到告警阈值"


def enrich_metric(db_row: dict) -> dict:
    """将 DB 原始指标行富化为前端 EnrichedMetric 结构。"""
    metric_code = db_row.get("metric_code") or "UNKNOWN"
    category = METRIC_CATEGORY_MAP.get(metric_code, "stability")
    display_name = METRIC_DISPLAY_NAMES.get(metric_code, metric_code)

    baseline_value = _safe_float(db_row.get("baseline_value"))
    current_value = _safe_float(db_row.get("current_value"))
    delta = _safe_float(db_row.get("delta"))
    availability_status = db_row.get("availability_status") or "AVAILABLE"
    triggered = bool(db_row.get("triggered"))

    # 从阈值规则获取信息
    rule = DEFAULT_THRESHOLD_RULES.get(metric_code)
    rule_enabled = rule is not None
    direction = rule.direction if rule else None
    direction_str = direction.value if direction else None
    warning_threshold = rule.warning_threshold if rule else None
    critical_threshold = rule.critical_threshold if rule else None

    # 判定 severity：从 alerts 表关联或从 triggered + metric_detail
    severity = None
    if triggered:
        # 尝试从 DB 行的 alert 信息获取 severity
        severity = db_row.get("severity")
        if not severity and rule and availability_status == "AVAILABLE":
            # 用规则重新评估一次得到 severity
            _triggered, _sev = rule.evaluate(delta, current_value)
            severity = _sev.value if _sev else None

    # 计算阈值使用比
    usage_ratio = None
    if rule and availability_status == "AVAILABLE":
        usage_ratio = compute_threshold_usage_ratio(delta, current_value, rule.direction, warning_threshold)

    # 生成状态解释
    status_reason = build_status_reason(
        metric_code=metric_code,
        availability_status=availability_status,
        rule_enabled=rule_enabled,
        triggered=triggered,
        severity=severity,
        usage_ratio=usage_ratio,
        direction=direction,
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
    )

    # metric_detail 合并趋势数据（如果存在）
    metric_detail = db_row.get("metric_detail") or {}
    if isinstance(metric_detail, str):
        import json
        try:
            metric_detail = json.loads(metric_detail)
        except (json.JSONDecodeError, TypeError):
            metric_detail = {}

    return {
        "metric_code": metric_code,
        "display_name": display_name,
        "category": category,
        "baseline_value": baseline_value,
        "current_value": current_value,
        "delta": delta,
        "direction": direction_str,
        "availability_status": availability_status,
        "rule_enabled": rule_enabled,
        "warning_threshold": warning_threshold,
        "critical_threshold": critical_threshold,
        "triggered": triggered,
        "severity": severity,
        "threshold_usage_ratio": round(usage_ratio, 6) if usage_ratio is not None else None,
        "status_reason": status_reason,
        "metric_detail": metric_detail if metric_detail else None,
    }


def build_coverage_summary(enriched_metrics: list[dict], run: dict | None = None) -> dict:
    """从富化指标列表生成 CoverageSummary。

    只统计 17 个核心指标（METRIC_CATEGORY_MAP 中的 key），
    诊断时间线和 per-feature 指标不参与规则覆盖计数。
    """
    # 只保留核心指标，并按 metric_code 去重（取第一个，通常是最新窗口的）
    canonical_codes = set(METRIC_CATEGORY_MAP.keys())
    seen: set[str] = set()
    canonical_metrics: list[dict] = []
    for m in enriched_metrics:
        code = m["metric_code"]
        if code in canonical_codes and code not in seen:
            seen.add(code)
            canonical_metrics.append(m)

    total = len(canonical_codes)
    calculated = sum(1 for m in canonical_metrics if m["availability_status"] == "AVAILABLE")
    available = sum(1 for m in canonical_metrics if m["availability_status"] != "CALCULATION_FAILED"
                    and m["availability_status"] != "DATA_NOT_AVAILABLE")
    rules_enabled = sum(1 for m in canonical_metrics if m["rule_enabled"])
    triggered = sum(1 for m in canonical_metrics if m["triggered"])

    # 按类别统计
    category_breakdown: dict[str, dict] = {}
    for cat in ["performance", "drift", "quality", "stability"]:
        cat_codes = [c for c, cat_ in METRIC_CATEGORY_MAP.items() if cat_ == cat]
        cat_metrics = [m for m in canonical_metrics if m["category"] == cat]
        cat_total = len(cat_codes)
        cat_normal = sum(1 for m in cat_metrics if not m["triggered"] and m["rule_enabled"]
                         and m["availability_status"] == "AVAILABLE")
        cat_warning = sum(1 for m in cat_metrics if m["triggered"] and m["severity"] == "WARNING")
        cat_critical = sum(1 for m in cat_metrics if m["triggered"] and m["severity"] in ("HIGH", "CRITICAL"))
        cat_unmonitored = sum(1 for m in cat_metrics if not m["rule_enabled"]
                              and m["availability_status"] == "AVAILABLE")
        cat_unavailable = sum(1 for m in cat_metrics if m["availability_status"] != "AVAILABLE")
        category_breakdown[cat] = {
            "total": cat_total,
            "normal": cat_normal,
            "warning": cat_warning,
            "critical": cat_critical,
            "unmonitored": cat_unmonitored,
            "unavailable": cat_unavailable,
        }

    # 最接近阈值的前 3 个指标（只看核心指标中未触发的）
    with_ratio = [m for m in canonical_metrics if m["threshold_usage_ratio"] is not None
                  and m["rule_enabled"] and not m["triggered"]]
    with_ratio.sort(key=lambda m: m["threshold_usage_ratio"] or 0, reverse=True)
    closest_thresholds = [
        {
            "metric_code": m["metric_code"],
            "display_name": m["display_name"],
            "usage_ratio": m["threshold_usage_ratio"],
        }
        for m in with_ratio[:3]
    ]

    # 标签成熟度
    label_maturity: dict = {}
    if run:
        alert_context = run.get("alert_context_json") or {}
        if isinstance(alert_context, str):
            import json
            try:
                alert_context = json.loads(alert_context)
            except (json.JSONDecodeError, TypeError):
                alert_context = {}
        label_maturity = {
            "mature": True,
        }

    # 未接入规则的指标列表（只看核心指标）
    unmonitored_metrics = [
        {"metric_code": m["metric_code"], "display_name": m["display_name"]}
        for m in canonical_metrics
        if not m["rule_enabled"] and m["availability_status"] == "AVAILABLE"
    ]

    return {
        "total_metrics": total,
        "calculated": calculated,
        "available": available,
        "rules_enabled": rules_enabled,
        "triggered": triggered,
        "category_breakdown": category_breakdown,
        "closest_thresholds": closest_thresholds,
        "label_maturity": label_maturity,
        "unmonitored_metrics": unmonitored_metrics,
    }


def _safe_float(val) -> float | None:
    """安全转 float，处理 None / NaN / inf。"""
    if val is None:
        return None
    try:
        f = float(val)
        return f if __import__("math").isfinite(f) else None
    except (ValueError, TypeError):
        return None
