"""阈值规则配置 — V1 默认规则，后续可从 monitoring.threshold_configs 表加载。"""

from __future__ import annotations

from dataclasses import dataclass

from packages.models.common.enums import MetricDirection, Severity


@dataclass
class ThresholdRule:
    """单个指标阈值规则。

    direction 语义：
    - DEVIATION_BAD: abs(delta) 或 abs(current_value) 超过阈值触发
    - LOWER_BETTER: 值低于阈值触发（如 SAMPLE_SIZE）
    - HIGHER_BETTER: delta < 0（下降）超过阈值触发，或 current_value 低于 absolute_minimum 时触发

    multi_tier 模式（MISSING_RATE 专用）：
    - high_threshold > critical_threshold 时启用三档：WARNING / HIGH / CRITICAL
    - blocking_threshold 设置后，>= 该值触发 CRITICAL + blocking 标记
    """

    metric_code: str
    direction: MetricDirection
    warning_threshold: float
    critical_threshold: float
    absolute_minimum: float | None = None
    high_threshold: float | None = None        # multi-tier: HIGH 级别阈值
    blocking_threshold: float | None = None    # multi-tier: 阻断阈值（≥触发 CRITICAL+blocking）
    rule_id: str = ""
    rule_version: str = "V1"

    def __post_init__(self):
        if not self.rule_id:
            self.rule_id = f"THRESHOLD_{self.metric_code}_V1"

    def evaluate(self, delta: float | None, current_value: float | None) -> tuple[bool, Severity | None]:
        """评估是否触发告警。

        Returns:
            (triggered, severity) — triggered=True 时 severity 为告警级别。
        """
        value = delta if delta is not None else current_value
        if value is None:
            return False, None

        abs_value = abs(value)

        if self.direction == MetricDirection.DEVIATION_BAD:
            # multi_tier 路径（MISSING_RATE 四档）
            if self.high_threshold is not None and self.high_threshold > self.critical_threshold:
                blocking = self.blocking_threshold
                if blocking is not None and abs_value >= blocking:
                    return True, Severity.CRITICAL
                if abs_value >= self.high_threshold:
                    return True, Severity.CRITICAL
                if abs_value >= self.critical_threshold:
                    return True, Severity.HIGH
                if abs_value >= self.warning_threshold:
                    return True, Severity.WARNING
                return False, None
            # 标准两档路径
            if abs_value > 0 and abs_value >= self.critical_threshold:
                return True, Severity.CRITICAL
            if abs_value > 0 and abs_value >= self.warning_threshold:
                return True, Severity.WARNING
            return False, None

        elif self.direction == MetricDirection.LOWER_BETTER:
            if value <= self.critical_threshold:
                return True, Severity.CRITICAL
            if value <= self.warning_threshold:
                return True, Severity.WARNING
            return False, None

        elif self.direction == MetricDirection.HIGHER_BETTER:
            # 方式1：有 delta 时，下降超过阈值触发
            if delta is not None and delta < 0:
                drop = abs(delta)
                if drop >= self.critical_threshold:
                    return True, Severity.CRITICAL
                if drop >= self.warning_threshold:
                    return True, Severity.WARNING
            # 方式2：无 delta 时，用 absolute_minimum 做绝对阈值兜底
            if delta is None and self.absolute_minimum is not None and current_value is not None:
                if current_value <= self.absolute_minimum:
                    return True, Severity.CRITICAL
            return False, None

        return False, None


# ── V1 默认规则 ──

DEFAULT_THRESHOLD_RULES: dict[str, ThresholdRule] = {
    "AUC": ThresholdRule(
        metric_code="AUC",
        direction=MetricDirection.HIGHER_BETTER,
        warning_threshold=0.02,
        critical_threshold=0.05,
        absolute_minimum=0.55,  # AUC < 0.55 无 baseline 时也告警
    ),
    "KS": ThresholdRule(
        metric_code="KS",
        direction=MetricDirection.HIGHER_BETTER,
        warning_threshold=0.02,
        critical_threshold=0.05,
        absolute_minimum=0.15,  # KS < 0.15 无 baseline 时也告警
    ),
    "FEATURE_PSI": ThresholdRule(
        metric_code="FEATURE_PSI",
        direction=MetricDirection.DEVIATION_BAD,
        warning_threshold=0.10,
        critical_threshold=0.25,
    ),
    "SCORE_PSI": ThresholdRule(
        metric_code="SCORE_PSI",
        direction=MetricDirection.DEVIATION_BAD,
        warning_threshold=0.10,
        critical_threshold=0.25,
    ),
    "MISSING_RATE": ThresholdRule(
        metric_code="MISSING_RATE",
        direction=MetricDirection.DEVIATION_BAD,
        warning_threshold=0.05,       # ≥0.05 WARNING（观察）
        critical_threshold=0.10,      # ≥0.10 HIGH（诊断确认可填）
        high_threshold=0.20,          # ≥0.20 CRITICAL（关键阻断·非关键可填）
        blocking_threshold=0.40,      # ≥0.40 CRITICAL + 阻断（强制人工）
    ),
    "SCHEMA_CONSISTENCY": ThresholdRule(
        metric_code="SCHEMA_CONSISTENCY",
        direction=MetricDirection.DEVIATION_BAD,
        warning_threshold=0.0,
        critical_threshold=0.0,
    ),
    "SAMPLE_SIZE": ThresholdRule(
        metric_code="SAMPLE_SIZE",
        direction=MetricDirection.LOWER_BETTER,
        warning_threshold=200,
        critical_threshold=50,
    ),
    # ── V1.1 补充规则：覆盖之前缺失的 10 个指标 ──
    "PR_AUC": ThresholdRule(
        metric_code="PR_AUC",
        direction=MetricDirection.HIGHER_BETTER,
        warning_threshold=0.03,
        critical_threshold=0.08,
        absolute_minimum=0.30,
    ),
    "BRIER": ThresholdRule(
        metric_code="BRIER",
        direction=MetricDirection.DEVIATION_BAD,
        warning_threshold=0.003,
        critical_threshold=0.006,
    ),
    "ECE": ThresholdRule(
        metric_code="ECE",
        direction=MetricDirection.DEVIATION_BAD,
        warning_threshold=0.003,
        critical_threshold=0.006,
    ),
    "BAD_RECALL": ThresholdRule(
        metric_code="BAD_RECALL",
        direction=MetricDirection.HIGHER_BETTER,
        warning_threshold=0.03,
        critical_threshold=0.08,
        absolute_minimum=0.40,
    ),
    "BAD_RATE": ThresholdRule(
        metric_code="BAD_RATE",
        direction=MetricDirection.DEVIATION_BAD,
        warning_threshold=0.01,
        critical_threshold=0.03,
    ),
    "OUTLIER_RATE": ThresholdRule(
        metric_code="OUTLIER_RATE",
        direction=MetricDirection.DEVIATION_BAD,
        # 3×MAD 尾部事件是正常现象：1-3% 的离群率增量是诊断证据，
        # 不应频繁打出 CRITICAL 污染诊断与根因排序
        warning_threshold=0.03,
        critical_threshold=0.06,
    ),
    "PREDICTION_MEAN": ThresholdRule(
        metric_code="PREDICTION_MEAN",
        direction=MetricDirection.DEVIATION_BAD,
        warning_threshold=0.02,
        critical_threshold=0.05,
    ),
}
