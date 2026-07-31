import pytest

from apps.modelops_api.services.iteration import DataEligibilityService
from packages.models.common.enums import DataUsabilityStatus, MissingRateBand
from packages.models.iteration import DataEligibilityInput, FeatureMissingStat


@pytest.mark.parametrize(
    ("rate", "expected"),
    [
        (0.0499, MissingRateBand.NORMAL),
        (0.05, MissingRateBand.WATCH),
        (0.10, MissingRateBand.WARNING),
        (0.20, MissingRateBand.CRITICAL),
        (0.40, MissingRateBand.UNAVAILABLE),
    ],
)
def test_feature_missing_rate_boundaries(rate, expected):
    service = DataEligibilityService()
    assert service.classify_feature_missing_rate(rate) == expected


def test_label_missing_at_20_percent_blocks_supervised_training():
    result = DataEligibilityService().evaluate(
        DataEligibilityInput(window_id="W3", label_missing_rate=0.20)
    )

    assert result.status == DataUsabilityStatus.UNSUPERVISED_ONLY
    assert result.supervised_training_allowed is False
    assert result.label_imputation_forbidden is True
    assert "LABEL_MISSING_RATE_REACHED_20_PERCENT" in result.blocking_reasons


def test_unlabelled_rows_below_threshold_are_excluded_not_imputed():
    result = DataEligibilityService().evaluate(
        DataEligibilityInput(window_id="W3", label_missing_rate=0.05)
    )

    assert result.supervised_training_allowed is True
    assert result.excluded_unlabelled_rows is True
    assert result.label_imputation_forbidden is True


def test_any_is_bad_imputation_request_is_rejected():
    result = DataEligibilityService().evaluate(
        DataEligibilityInput(
            window_id="W3",
            label_missing_rate=0.01,
            label_imputation_requested=True,
        )
    )

    assert result.supervised_training_allowed is False
    assert "LABEL_IMPUTATION_FORBIDDEN" in result.blocking_reasons


def test_critical_feature_at_20_percent_blocks_training():
    result = DataEligibilityService().evaluate(
        DataEligibilityInput(
            window_id="W3",
            label_missing_rate=0,
            feature_missing_stats=[
                FeatureMissingStat(
                    feature_code="income", missing_rate=0.20, is_critical=True
                )
            ],
        )
    )

    assert result.status == DataUsabilityStatus.BLOCKED
    assert result.feature_results[0].training_blocked is True


def test_w4_is_never_a_training_window():
    result = DataEligibilityService().evaluate(
        DataEligibilityInput(window_id="W4", label_missing_rate=0)
    )

    assert result.supervised_training_allowed is False
    assert "OOT_WINDOW_FORBIDDEN_FOR_TRAINING" in result.blocking_reasons
