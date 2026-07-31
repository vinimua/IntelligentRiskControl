"""Challenger 七道资格门的确定性判定。"""

from packages.models.common.enums import (
    QualificationGateCode,
    QualificationStatus,
)
from packages.models.iteration.qualification import (
    MetricComparison,
    QualificationGateResult,
    QualificationInput,
    QualificationReport,
)

from .config_loader import IterationConfigBundle, load_iteration_config


class QualificationService:
    def __init__(self, config: IterationConfigBundle | None = None) -> None:
        self.config = config or load_iteration_config()

    @staticmethod
    def _metric_recovery_rate(metric: MetricComparison) -> float | None:
        if metric.recovery_rate is not None:
            return metric.recovery_rate
        if metric.original_drop in (None, 0) or metric.recovered_amount is None:
            return None
        return abs(metric.recovered_amount) / abs(metric.original_drop)

    @staticmethod
    def _in_healthy_range(metric: MetricComparison) -> bool:
        value = metric.challenger_value
        if value is None:
            return False
        if (
            metric.healthy_lower_bound is not None
            and value < metric.healthy_lower_bound
        ):
            return False
        if (
            metric.healthy_upper_bound is not None
            and value > metric.healthy_upper_bound
        ):
            return False
        return (
            metric.healthy_lower_bound is not None
            or metric.healthy_upper_bound is not None
        )

    @staticmethod
    def _bootstrap_supports_improvement(metric: MetricComparison) -> bool:
        if metric.bootstrap_ci_lower is None or metric.bootstrap_ci_upper is None:
            return False
        direction = metric.direction.upper()
        if direction == "HIGHER_BETTER":
            return metric.bootstrap_ci_lower > 0
        if direction == "LOWER_BETTER":
            return metric.bootstrap_ci_upper < 0
        return not (
            metric.bootstrap_ci_lower <= 0 <= metric.bootstrap_ci_upper
        )

    def _target_gate(
        self, metrics: list[MetricComparison]
    ) -> QualificationGateResult:
        rule = self.config.qualification
        reasons: list[str] = []
        details: dict[str, dict] = {}
        if not metrics:
            reasons.append("NO_TARGET_METRICS")
        for metric in metrics:
            recovery_rate = self._metric_recovery_rate(metric)
            healthy = self._in_healthy_range(metric)
            bootstrap = self._bootstrap_supports_improvement(metric)
            details[metric.metric_code] = {
                "recovery_rate": recovery_rate,
                "healthy_range_reached": healthy,
                "bootstrap_improvement_supported": bootstrap,
            }
            if recovery_rate is None or recovery_rate < rule.min_recovery_rate:
                reasons.append(f"RECOVERY_RATE_FAILED:{metric.metric_code}")
            if rule.require_healthy_range and not healthy:
                reasons.append(f"HEALTHY_RANGE_FAILED:{metric.metric_code}")
            if rule.require_same_sample_bootstrap and not bootstrap:
                reasons.append(f"BOOTSTRAP_FAILED:{metric.metric_code}")
        return self._gate(
            QualificationGateCode.TARGET_RECOVERY,
            passed=not reasons,
            reasons=reasons,
            metrics=details,
        )

    @staticmethod
    def _gate(
        code: QualificationGateCode,
        *,
        passed: bool,
        reasons: list[str] | None = None,
        metrics: dict | None = None,
    ) -> QualificationGateResult:
        return QualificationGateResult(
            gate_code=code,
            gate_order=list(QualificationGateCode).index(code),
            status=(
                QualificationStatus.PASSED
                if passed
                else QualificationStatus.FAILED
            ),
            reasons=reasons or [],
            metrics=metrics or {},
        )

    def evaluate(self, request: QualificationInput) -> QualificationReport:
        rule = self.config.qualification
        gates = [
            self._gate(
                QualificationGateCode.DATA_REPRODUCIBILITY,
                passed=request.data_reproducible,
                reasons=[] if request.data_reproducible else ["DATA_NOT_REPRODUCIBLE"],
            ),
            self._target_gate(request.target_metrics),
            self._gate(
                QualificationGateCode.DISCRIMINATION,
                passed=request.discrimination_passed,
                reasons=[] if request.discrimination_passed else ["DISCRIMINATION_REGRESSION"],
            ),
            self._gate(
                QualificationGateCode.CALIBRATION,
                passed=request.calibration_passed,
                reasons=[] if request.calibration_passed else ["CALIBRATION_REGRESSION"],
            ),
            self._gate(
                QualificationGateCode.STABILITY,
                passed=(
                    request.score_psi <= rule.max_score_psi
                    and request.train_valid_gap <= rule.max_train_valid_gap
                ),
                reasons=[
                    reason
                    for condition, reason in (
                        (
                            request.score_psi > rule.max_score_psi,
                            "SCORE_PSI_EXCEEDED",
                        ),
                        (
                            request.train_valid_gap > rule.max_train_valid_gap,
                            "TRAIN_VALID_GAP_EXCEEDED",
                        ),
                    )
                    if condition
                ],
                metrics={
                    "score_psi": request.score_psi,
                    "train_valid_gap": request.train_valid_gap,
                },
            ),
            self._gate(
                QualificationGateCode.SEGMENT_GOVERNANCE,
                passed=request.segment_governance_passed,
                reasons=(
                    []
                    if request.segment_governance_passed
                    else ["SEGMENT_OR_GOVERNANCE_FAILED"]
                ),
            ),
            self._gate(
                QualificationGateCode.OOT,
                passed=(
                    request.oot_window_id == rule.required_oot_window_id
                    and request.candidate_frozen_before_oot
                    and request.oot_usage == "FINAL_QUALIFICATION"
                    and request.oot_passed
                ),
                reasons=[
                    reason
                    for condition, reason in (
                        (
                            request.oot_window_id != rule.required_oot_window_id,
                            "INVALID_OOT_WINDOW",
                        ),
                        (
                            not request.candidate_frozen_before_oot,
                            "CANDIDATE_NOT_FROZEN_BEFORE_OOT",
                        ),
                        (
                            request.oot_usage != "FINAL_QUALIFICATION",
                            "OOT_USAGE_FORBIDDEN",
                        ),
                        (not request.oot_passed, "OOT_PERFORMANCE_FAILED"),
                    )
                    if condition
                ],
                metrics={"oot_window_id": request.oot_window_id},
            ),
        ]
        failed = [
            gate.gate_code
            for gate in gates
            if gate.required and gate.status != QualificationStatus.PASSED
        ]
        qualified = not failed
        return QualificationReport(
            qualification_run_id=request.qualification_run_id,
            iteration_run_id=request.iteration_run_id,
            experiment_id=request.experiment_id,
            candidate_version=request.candidate_version,
            status=(
                QualificationStatus.PASSED
                if qualified
                else QualificationStatus.FAILED
            ),
            qualified=qualified,
            gate_results=gates,
            failed_gate_codes=failed,
            rule_version=rule.rule_version,
        )
