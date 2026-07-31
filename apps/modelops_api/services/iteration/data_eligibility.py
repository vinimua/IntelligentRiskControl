"""训练数据与标签的确定性门禁。"""

from packages.models.common.enums import (
    DataUsabilityStatus,
    MissingRateBand,
)
from packages.models.iteration.data_eligibility import (
    DataEligibilityInput,
    DataEligibilityResult,
    FeatureMissingStat,
)

from .config_loader import IterationConfigBundle, load_iteration_config


class DataEligibilityService:
    def __init__(self, config: IterationConfigBundle | None = None) -> None:
        self.config = config or load_iteration_config()

    def classify_feature_missing_rate(self, missing_rate: float) -> MissingRateBand:
        rules = self.config.iteration.missing_rates
        if missing_rate < rules.watch:
            return MissingRateBand.NORMAL
        if missing_rate < rules.warning:
            return MissingRateBand.WATCH
        if missing_rate < rules.critical:
            return MissingRateBand.WARNING
        if missing_rate < rules.unavailable:
            return MissingRateBand.CRITICAL
        return MissingRateBand.UNAVAILABLE

    def evaluate(self, request: DataEligibilityInput) -> DataEligibilityResult:
        rules = self.config.iteration.missing_rates
        blocking_reasons: list[str] = []
        warnings: list[str] = []
        feature_results: list[FeatureMissingStat] = []

        if (
            request.requested_for_supervised_training
            and request.window_id == self.config.iteration.oot_window_id
        ):
            blocking_reasons.append("OOT_WINDOW_FORBIDDEN_FOR_TRAINING")

        if request.label_imputation_requested:
            blocking_reasons.append("LABEL_IMPUTATION_FORBIDDEN")
        if request.requested_for_supervised_training and not request.label_mature:
            blocking_reasons.append("LABEL_NOT_MATURE")
        if request.label_missing_rate >= rules.label_training_block:
            blocking_reasons.append("LABEL_MISSING_RATE_REACHED_20_PERCENT")
        elif request.label_missing_rate > 0:
            warnings.append("UNLABELLED_ROWS_EXCLUDED_FROM_SUPERVISED_USE")

        for stat in request.feature_missing_stats:
            band = self.classify_feature_missing_rate(stat.missing_rate)
            training_blocked = band == MissingRateBand.UNAVAILABLE or (
                stat.is_critical and band == MissingRateBand.CRITICAL
            )
            feature_results.append(
                stat.model_copy(
                    update={"band": band, "training_blocked": training_blocked}
                )
            )
            if training_blocked:
                blocking_reasons.append(
                    f"FEATURE_MISSING_BLOCK:{stat.feature_code}:{band.value}"
                )
            elif band in {
                MissingRateBand.WATCH,
                MissingRateBand.WARNING,
                MissingRateBand.CRITICAL,
            }:
                warnings.append(
                    f"FEATURE_MISSING_WARNING:{stat.feature_code}:{band.value}"
                )

        supervised_allowed = not blocking_reasons
        if not request.requested_for_supervised_training:
            status = (
                DataUsabilityStatus.USABLE_WITH_WARNING
                if warnings or blocking_reasons
                else DataUsabilityStatus.USABLE
            )
            supervised_allowed = False
        elif {
            "LABEL_MISSING_RATE_REACHED_20_PERCENT",
            "LABEL_NOT_MATURE",
        } & set(blocking_reasons):
            status = DataUsabilityStatus.UNSUPERVISED_ONLY
        elif blocking_reasons:
            status = DataUsabilityStatus.BLOCKED
        elif warnings:
            status = DataUsabilityStatus.USABLE_WITH_WARNING
        else:
            status = DataUsabilityStatus.USABLE

        return DataEligibilityResult(
            window_id=request.window_id,
            status=status,
            supervised_training_allowed=supervised_allowed,
            data_track=request.data_track,
            data_snapshot_id=request.data_snapshot_id,
            data_checksum=request.data_checksum,
            feature_results=feature_results,
            blocking_reasons=blocking_reasons,
            warnings=warnings,
            rule_version=self.config.iteration.rule_version,
        )
