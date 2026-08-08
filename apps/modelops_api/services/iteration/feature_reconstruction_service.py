"""Feature reconstruction service.

The service converts diagnosis evidence into a deterministic feature
engineering plan before the training plan is built.

Rules vary by algorithm family:
- Tree models (lightgbm, xgboost, random_forest): native missing handling,
  monotonic-transform tolerance, implicit interactions
- Linear models (logistic_regression): MUST handle missing, MUST scale,
  MUST explicitly construct interactions
"""

from __future__ import annotations

from packages.models.iteration.feature_reconstruction import (
    FeatureOperation,
    FeatureReconstructionPlan,
    FeatureTransformItem,
)


# ── 算法族分类 ──

TREE_ALGORITHMS = {"lightgbm", "xgboost", "random_forest", "catboost"}
LINEAR_ALGORITHMS = {"logistic_regression", "linear_svc", "ridge"}

# ── 核心特征：不允许 DROP，只能修改（LOG_TRANSFORM / INTERACTION / IMPUTE / STANDARDIZE）──
CORE_FEATURES: set[str] = {
    "credit_query_times", "multi_loan_count", "overdue_history",
    "credit_utilization", "max_overdue_days", "social_score",
    "telecom_score", "ecomm_risk_score", "judicial_risk_score",
    "blacklist_hit", "login_fail_count", "device_risk_score",
    "ip_change_freq", "gps_anomaly", "emulator_flag",
    "income_level", "job_stability", "debt_income_ratio",
    "loan_amount_request", "repayment_period",
}


def _is_tree(algorithm: str) -> bool:
    return algorithm.lower() in TREE_ALGORITHMS


def _is_linear(algorithm: str) -> bool:
    return algorithm.lower() in LINEAR_ALGORITHMS


class FeatureReconstructionService:
    """Decide which features should be dropped, transformed, or expanded.

    Thresholds differ by algorithm family.
    """

    # ── 共享阈值 ──
    DRIFT_DROP_THRESHOLD: float = 0.25       # PSI > 0.25 → DROP（所有算法）

    # ── 树模型阈值 ──
    TREE_MISSING_DROP: float = 0.40          # missing > 40% → DROP
    TREE_SKEW_LOG: float = 1.5               # |skew| > 1.5 → LOG_TRANSFORM

    # ── 线性模型阈值（更严格）──
    LINEAR_MISSING_DROP: float = 0.20        # missing > 20% → DROP
    LINEAR_SKEW_LOG: float = 1.0             # |skew| > 1.0 → LOG_TRANSFORM
    LINEAR_MISSING_IMPUTE: float = 0.10      # 10-20% missing → IMPUTE（不删，但必须填）

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
        algorithm: str = "lightgbm",
    ) -> FeatureReconstructionPlan:
        drift_features = drift_features or []
        high_missing_features = high_missing_features or []
        current_feature_names = current_feature_names or []
        feature_importance = feature_importance or {}
        skewness = skewness or {}

        transforms: list[FeatureTransformItem] = []
        drift_names = [f.get("feature_name", "") for f in drift_features]
        missing_names = [f.get("feature_name", "") for f in high_missing_features]
        is_tree = _is_tree(algorithm)
        is_linear = _is_linear(algorithm)

        # ═══════════════════════════════════════════
        # Rule 1: PSI drift — DROP（核心特征不允许删除，改做 LOG_TRANSFORM）
        # ═══════════════════════════════════════════
        for feat in drift_features:
            name = feat.get("feature_name", "")
            psi = feat.get("psi_value", feat.get("current_value", 0))
            if not isinstance(psi, (int, float)) or psi <= self.DRIFT_DROP_THRESHOLD:
                continue
            if name in CORE_FEATURES:
                transforms.append(
                    FeatureTransformItem(
                        operation=FeatureOperation.LOG_TRANSFORM,
                        source_feature=name,
                        target_feature=f"{name}_log",
                        reason=f"PSI={psi:.3f} > {self.DRIFT_DROP_THRESHOLD}（核心特征，不删除，对数变换稳定）",
                        parameters={"offset": 1.0},
                    )
                )
            else:
                transforms.append(
                    FeatureTransformItem(
                        operation=FeatureOperation.DROP,
                        source_feature=name,
                        reason=f"PSI={psi:.3f} > {self.DRIFT_DROP_THRESHOLD}",
                    )
                )

        # ═══════════════════════════════════════════
        # Rule 2: Missing rate — DROP / IMPUTE
        # ═══════════════════════════════════════════
        missing_drop_threshold = (
            self.LINEAR_MISSING_DROP if is_linear else self.TREE_MISSING_DROP
        )
        for feat in high_missing_features:
            name = feat.get("feature_name", "")
            miss_rate = feat.get("missing_rate", feat.get("current_value", 0))
            already_dropped = any(
                t.operation == FeatureOperation.DROP and t.source_feature == name
                for t in transforms
            )
            if already_dropped:
                continue
            if not isinstance(miss_rate, (int, float)):
                continue

            if miss_rate > missing_drop_threshold:
                if name in CORE_FEATURES:
                    # 核心特征不允许删除 → 中位数插补
                    transforms.append(
                        FeatureTransformItem(
                            operation=FeatureOperation.IMPUTE,
                            source_feature=name,
                            reason=f"missing_rate={miss_rate:.3f} > {missing_drop_threshold}（核心特征，不删除，中位数插补）",
                            parameters={"strategy": "median"},
                        )
                    )
                else:
                    transforms.append(
                        FeatureTransformItem(
                            operation=FeatureOperation.DROP,
                            source_feature=name,
                            reason=f"missing_rate={miss_rate:.3f} > {missing_drop_threshold} (algorithm={algorithm})",
                        )
                    )
            elif is_linear and miss_rate > self.LINEAR_MISSING_IMPUTE:
                # 线性模型 10-20% 缺失 → 不能删（信息损失太大），不能不管（线性不能处理 NaN）
                # → 标记为需中位数插补
                transforms.append(
                    FeatureTransformItem(
                        operation=FeatureOperation.IMPUTE,
                        source_feature=name,
                        reason=f"missing_rate={miss_rate:.3f} > {self.LINEAR_MISSING_IMPUTE} → median impute (linear model)",
                        parameters={"strategy": "median"},
                    )
                )

        # ═══════════════════════════════════════════
        # Rule 3: Skewness — LOG_TRANSFORM
        # ═══════════════════════════════════════════
        skew_threshold = self.LINEAR_SKEW_LOG if is_linear else self.TREE_SKEW_LOG
        dropped_features = {
            t.source_feature for t in transforms if t.operation == FeatureOperation.DROP
        }
        for name in current_feature_names:
            sk = skewness.get(name, 0)
            if abs(sk) > skew_threshold and name not in dropped_features:
                transforms.append(
                    FeatureTransformItem(
                        operation=FeatureOperation.LOG_TRANSFORM,
                        source_feature=name,
                        target_feature=f"{name}_log",
                        reason=(
                            f"skewness={sk:.2f} > {skew_threshold} (algorithm={algorithm})"
                        ),
                        parameters={"offset": 1.0},
                    )
                )

        # ═══════════════════════════════════════════
        # Rule 4: Interaction — 线性模型必须做，树模型可选
        # ═══════════════════════════════════════════
        if is_linear or is_tree:
            top_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:4]
            if len(top_features) >= 2:
                max_interactions = min(3, len(top_features)) if is_linear else min(2, len(top_features) // 2)
                count = 0
                for i in range(min(2 if is_tree else 3, len(top_features))):
                    for j in range(i + 1, min(4, len(top_features))):
                        if count >= max_interactions:
                            break
                        f1, imp1 = top_features[i]
                        f2, imp2 = top_features[j]
                        if f1 not in drift_names and f2 not in drift_names:
                            transforms.append(
                                FeatureTransformItem(
                                    operation=FeatureOperation.INTERACTION,
                                    source_feature=f1,
                                    target_feature=f"{f1}_x_{f2}",
                                    reason=(
                                        f"top features interaction (imp={imp1:.3f}x{imp2:.3f}, algo={algorithm})"
                                    ),
                                    parameters={"features": [f1, f2], "type": "multiply"},
                                )
                            )
                            count += 1

        # ═══════════════════════════════════════════
        # Rule 5 (线性专属): StandardScaler
        # ═══════════════════════════════════════════
        if is_linear:
            transforms.append(
                FeatureTransformItem(
                    operation=FeatureOperation.STANDARDIZE,
                    source_feature="*",  # 全量标准化
                    reason=f"linear model ({algorithm}) requires feature scaling",
                    parameters={"with_mean": True, "with_std": True},
                )
            )

        # ── 统计 ──
        drop_count = sum(1 for t in transforms if t.operation == FeatureOperation.DROP)
        add_count = sum(1 for t in transforms if t.operation not in {
            FeatureOperation.DROP, FeatureOperation.STANDARDIZE, FeatureOperation.IMPUTE
        })
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
