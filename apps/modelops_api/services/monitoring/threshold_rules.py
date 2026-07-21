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
    """

    metric_code: str
    direction: MetricDirection
    warning_threshold: float
    critical_threshold: float
    absolute_minimum: float | None = None  # HIGHER_BETTER 无 baseline 时的绝对兜底阈值
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
        warning_threshold=0.10,
        critical_threshold=0.30,
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
}
