from apps.modelops_api.services.iteration import (
    FailureAttributionService,
    QualificationService,
)
from packages.models.common.enums import FailureCode, QualificationGateCode
from packages.models.iteration import MetricComparison, QualificationInput


def _qualified_input(**updates) -> QualificationInput:
    payload = {
        "qualification_run_id": "22222222-2222-2222-2222-222222222222",
        "iteration_run_id": "33333333-3333-3333-3333-333333333333",
        "experiment_id": "44444444-4444-4444-4444-444444444444",
        "candidate_version": "challenger-v1",
        "target_metrics": [
            MetricComparison(
                metric_code="AUC",
                direction="HIGHER_BETTER",
                original_drop=0.10,
                recovered_amount=0.095,
                challenger_value=0.76,
                healthy_lower_bound=0.75,
                bootstrap_ci_lower=0.01,
                bootstrap_ci_upper=0.03,
            ),
            # 严格 A7：AUC + KS 双指标必填（REQUIRED_TARGET_METRICS_MISSING 防单指标蒙混）
            MetricComparison(
                metric_code="KS",
                direction="HIGHER_BETTER",
                original_drop=0.06,
                recovered_amount=0.057,
                challenger_value=0.36,
                healthy_lower_bound=0.15,
                bootstrap_ci_lower=0.02,
                bootstrap_ci_upper=0.06,
            ),
        ],
        "data_reproducible": True,
        "discrimination_passed": True,
        "calibration_passed": True,
        "score_psi": 0.10,
        "train_valid_gap": 0.02,
        "segment_governance_passed": True,
        "oot_window_id": "W4",
        "candidate_frozen_before_oot": True,
        # 冻结身份三要素：晋升防换包
        "frozen_identity_checksum": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "oot_passed": True,
    }
    payload.update(updates)
    return QualificationInput(**payload)


def test_all_seven_gates_must_pass():
    report = QualificationService().evaluate(_qualified_input())

    assert report.qualified is True
    assert len(report.gate_results) == 7
    assert report.failed_gate_codes == []


def test_recovery_below_60_percent_fails_target_gate():
    metric = MetricComparison(
        metric_code="AUC",
        direction="HIGHER_BETTER",
        original_drop=0.10,
        recovered_amount=0.05,
        challenger_value=0.76,
        healthy_lower_bound=0.75,
        bootstrap_ci_lower=0.01,
        bootstrap_ci_upper=0.03,
    )
    report = QualificationService().evaluate(
        _qualified_input(target_metrics=[metric])
    )

    assert report.qualified is False
    assert QualificationGateCode.TARGET_RECOVERY in report.failed_gate_codes


def test_stability_thresholds_are_inclusive():
    report = QualificationService().evaluate(
        _qualified_input(score_psi=0.20, train_valid_gap=0.03)
    )
    assert report.qualified is True


def test_only_w4_can_satisfy_final_oot_gate():
    report = QualificationService().evaluate(
        _qualified_input(oot_window_id="W3")
    )
    assert report.qualified is False
    assert QualificationGateCode.OOT in report.failed_gate_codes


def test_unfrozen_candidate_cannot_use_w4():
    report = QualificationService().evaluate(
        _qualified_input(candidate_frozen_before_oot=False)
    )
    assert report.qualified is False
    assert QualificationGateCode.OOT in report.failed_gate_codes


def test_failure_report_is_generated_for_failed_candidate():
    report = QualificationService().evaluate(
        _qualified_input(calibration_passed=False)
    )
    failure = FailureAttributionService().from_qualification(
        "proposal-1", report
    )

    assert failure is not None
    assert failure.failure_code == FailureCode.CALIBRATION_FAILED
    assert failure.retryable is True
