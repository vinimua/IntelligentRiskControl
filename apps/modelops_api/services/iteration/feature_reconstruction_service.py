"""Feature reconstruction service.

The service converts diagnosis evidence into a deterministic feature
engineering plan before the training plan is built.
"""

from __future__ import annotations

from packages.models.iteration.feature_reconstruction import (
    FeatureOperation,
    FeatureReconstructionPlan,
    FeatureTransformItem,
)


class FeatureReconstructionService:
    """Decide which features should be dropped, transformed, or expanded."""

    DRIFT_DROP_THRESHOLD: float = 0.25
    MISSING_DROP_THRESHOLD: float = 0.40
    SKEW_LOG_THRESHOLD: float = 1.5

    def build_plan(
        self,
        *,
        model_id: str = "",
        lifecycle_run_id: str | None = None,
        diagnosis_run_id: str | None = None,
        current_schema_version: str = "v1",
        drift_features: list[dict] | None = None,
        high_missing_features: list[dict] | None = None,
        current_feature_names: list[str] | None = None,
        feature_importance: dict[str, float] | None = None,
        skewness: dict[str, float] | None = None,
    ) -> FeatureReconstructionPlan:
        drift_features = drift_features or []
        high_missing_features = high_missing_features or []
        current_feature_names = current_feature_names or []
        feature_importance = feature_importance or {}
        skewness = skewness or {}

        transforms: list[FeatureTransformItem] = []
        drift_names = [f.get("feature_name", "") for f in drift_features]
        missing_names = [f.get("feature_name", "") for f in high_missing_features]

        for feat in drift_features:
            name = feat.get("feature_name", "")
            psi = feat.get("psi_value", feat.get("current_value", 0))
            if isinstance(psi, (int, float)) and psi > self.DRIFT_DROP_THRESHOLD:
                transforms.append(
                    FeatureTransformItem(
                        operation=FeatureOperation.DROP,
                        source_feature=name,
                        reason=f"PSI={psi:.3f} > {self.DRIFT_DROP_THRESHOLD}",
                    )
                )

        for feat in high_missing_features:
            name = feat.get("feature_name", "")
            miss_rate = feat.get("missing_rate", feat.get("current_value", 0))
            already_dropped = any(
                t.operation == FeatureOperation.DROP and t.source_feature == name
                for t in transforms
            )
            if isinstance(miss_rate, (int, float)) and miss_rate > self.MISSING_DROP_THRESHOLD and not already_dropped:
                transforms.append(
                    FeatureTransformItem(
                        operation=FeatureOperation.DROP,
                        source_feature=name,
                        reason=f"missing_rate={miss_rate:.3f} > {self.MISSING_DROP_THRESHOLD}",
                    )
                )

        dropped_features = {
            t.source_feature for t in transforms if t.operation == FeatureOperation.DROP
        }
        for name in current_feature_names:
            sk = skewness.get(name, 0)
            if abs(sk) > self.SKEW_LOG_THRESHOLD and name not in dropped_features:
                transforms.append(
                    FeatureTransformItem(
                        operation=FeatureOperation.LOG_TRANSFORM,
                        source_feature=name,
                        target_feature=f"{name}_log",
                        reason=f"skewness={sk:.2f} > {self.SKEW_LOG_THRESHOLD}",
                        parameters={"offset": 1.0},
                    )
                )

        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:4]
        if len(top_features) >= 2:
            for i in range(min(2, len(top_features))):
                for j in range(i + 1, min(4, len(top_features))):
                    f1, imp1 = top_features[i]
                    f2, imp2 = top_features[j]
                    if f1 not in drift_names and f2 not in drift_names:
                        transforms.append(
                            FeatureTransformItem(
                                operation=FeatureOperation.INTERACTION,
                                source_feature=f1,
                                target_feature=f"{f1}_x_{f2}",
                                reason=f"top features interaction (imp={imp1:.3f}x{imp2:.3f})",
                                parameters={"features": [f1, f2], "type": "multiply"},
                            )
                        )

        drop_count = sum(1 for t in transforms if t.operation == FeatureOperation.DROP)
        add_count = sum(1 for t in transforms if t.operation != FeatureOperation.DROP)
        before = len(current_feature_names)
        after = before - drop_count + add_count

        base_version = current_schema_version.lstrip("v")
        try:
            next_ver = int(base_version) + 1
        except ValueError:
            next_ver = 2
        target_version = f"v{next_ver}" if transforms else current_schema_version

        return FeatureReconstructionPlan(
            lifecycle_run_id=lifecycle_run_id,
            diagnosis_run_id=diagnosis_run_id,
            model_id=model_id,
            current_schema_version=current_schema_version,
            target_schema_version=target_version,
            transforms=transforms,
            drift_features=drift_names,
            high_missing_features=missing_names,
            expected_feature_count_before=before,
            expected_feature_count_after=after,
        )
