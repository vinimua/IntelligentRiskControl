"""Task-type interfaces for adaptive model iteration.

The current engine is strongest for binary credit-risk classification.  This
service exposes the task profile and metric contract that regression and
clustering adapters can plug into without reusing AUC/KS by accident.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from packages.models.common.enums import (
    GuardrailCoverageStatus,
    MetricDirection,
    ModelTaskType,
    RiskGuardrailCode,
    TrainingMode,
)
from packages.models.iteration.model_task_interface import (
    MetricSpec,
    ModelTaskInterfaceSummary,
    ModelTaskProfile,
    RiskGuardrailResult,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]


_CLASSIFICATION_REQUIRED = [
    MetricSpec(metric_code="AUC", direction=MetricDirection.HIGHER_BETTER, description="Ranking discrimination"),
    MetricSpec(metric_code="KS", direction=MetricDirection.HIGHER_BETTER, description="Bad/good separation"),
]
_CLASSIFICATION_OPTIONAL = [
    MetricSpec(metric_code="PR_AUC", direction=MetricDirection.HIGHER_BETTER, description="Imbalanced ranking quality"),
    MetricSpec(metric_code="BAD_RECALL", direction=MetricDirection.HIGHER_BETTER, description="Bad-customer recall"),
    MetricSpec(metric_code="BRIER", direction=MetricDirection.LOWER_BETTER, description="Probability calibration"),
    MetricSpec(metric_code="ECE", direction=MetricDirection.LOWER_BETTER, description="Expected calibration error"),
    MetricSpec(metric_code="SCORE_PSI", direction=MetricDirection.LOWER_BETTER, baseline_required=True, description="Score distribution drift"),
    MetricSpec(metric_code="FEATURE_PSI", direction=MetricDirection.LOWER_BETTER, baseline_required=True, description="Feature distribution drift"),
]

_REGRESSION_REQUIRED = [
    MetricSpec(metric_code="RMSE", direction=MetricDirection.LOWER_BETTER, description="Root mean squared error"),
    MetricSpec(metric_code="MAE", direction=MetricDirection.LOWER_BETTER, description="Mean absolute error"),
    MetricSpec(metric_code="R2", direction=MetricDirection.HIGHER_BETTER, description="Explained variance"),
    MetricSpec(metric_code="RESIDUAL_PSI", direction=MetricDirection.LOWER_BETTER, baseline_required=True, description="Residual distribution drift"),
]
_REGRESSION_OPTIONAL = [
    MetricSpec(metric_code="RESIDUAL_MEAN_SHIFT", direction=MetricDirection.DEVIATION_BAD, baseline_required=True),
    MetricSpec(metric_code="RESIDUAL_STD_SHIFT", direction=MetricDirection.DEVIATION_BAD, baseline_required=True),
    MetricSpec(metric_code="PREDICTION_PSI", direction=MetricDirection.LOWER_BETTER, baseline_required=True),
    MetricSpec(metric_code="FEATURE_PSI", direction=MetricDirection.LOWER_BETTER, baseline_required=True),
    MetricSpec(metric_code="SEGMENT_RMSE", direction=MetricDirection.LOWER_BETTER),
]

_CLUSTERING_REQUIRED = [
    MetricSpec(metric_code="SILHOUETTE", direction=MetricDirection.HIGHER_BETTER, label_required=False, description="Cluster separation"),
    MetricSpec(metric_code="CLUSTER_STABILITY", direction=MetricDirection.HIGHER_BETTER, label_required=False, baseline_required=True, description="Assignment stability"),
    MetricSpec(metric_code="CLUSTER_PSI", direction=MetricDirection.LOWER_BETTER, label_required=False, baseline_required=True, description="Cluster population drift"),
]
_CLUSTERING_OPTIONAL = [
    MetricSpec(metric_code="DAVIES_BOULDIN", direction=MetricDirection.LOWER_BETTER, label_required=False),
    MetricSpec(metric_code="CALINSKI_HARABASZ", direction=MetricDirection.HIGHER_BETTER, label_required=False),
    MetricSpec(metric_code="NOISE_RATE", direction=MetricDirection.DEVIATION_BAD, label_required=False, baseline_required=True),
    MetricSpec(metric_code="CLUSTER_SIZE_PSI", direction=MetricDirection.LOWER_BETTER, label_required=False, baseline_required=True),
]


class ModelTaskInterfaceService:
    @staticmethod
    def _normalize(value: str | None) -> str:
        return (value or "").strip().lower().replace("-", "_").replace(" ", "_")

    @classmethod
    def infer_task_type(
        cls,
        *,
        model_type: str | None = None,
        algorithm_family: str | None = None,
        manifest: dict[str, Any] | None = None,
    ) -> tuple[ModelTaskType, str]:
        manifest = manifest or {}
        declared = cls._normalize(
            manifest.get("task_type")
            or manifest.get("model_task_type")
            or model_type
        )
        algorithm = cls._normalize(algorithm_family or manifest.get("algorithm_family"))

        if any(token in declared for token in ("regression", "regressor", "scorecard_regression")):
            return ModelTaskType.REGRESSION, "declared_model_type"
        if any(token in declared for token in ("cluster", "clustering", "unsupervised")):
            return ModelTaskType.CLUSTERING, "declared_model_type"
        if any(token in declared for token in ("classification", "classifier", "credit_risk", "binary")):
            return ModelTaskType.CLASSIFICATION, "declared_model_type"

        if any(token in algorithm for token in ("regressor", "regression")):
            return ModelTaskType.REGRESSION, "algorithm_family"
        if any(token in algorithm for token in ("kmeans", "dbscan", "cluster")):
            return ModelTaskType.CLUSTERING, "algorithm_family"
        return ModelTaskType.CLASSIFICATION, "default_classification"

    @staticmethod
    def _manifest(model_id: str, champion_version: str) -> dict[str, Any] | None:
        path = (
            PROJECT_ROOT
            / "assets"
            / "champion_models"
            / model_id
            / champion_version
            / "training_manifest.json"
        )
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _metrics(task_type: ModelTaskType) -> tuple[list[MetricSpec], list[MetricSpec]]:
        if task_type == ModelTaskType.REGRESSION:
            return list(_REGRESSION_REQUIRED), list(_REGRESSION_OPTIONAL)
        if task_type == ModelTaskType.CLUSTERING:
            return list(_CLUSTERING_REQUIRED), list(_CLUSTERING_OPTIONAL)
        return list(_CLASSIFICATION_REQUIRED), list(_CLASSIFICATION_OPTIONAL)

    @classmethod
    def build_profile(
        cls,
        *,
        model_id: str,
        champion_version: str = "champion_v1",
        model_type: str | None = None,
        algorithm_family: str | None = None,
    ) -> ModelTaskProfile:
        manifest = cls._manifest(model_id, champion_version) or {}
        algorithm = algorithm_family or manifest.get("algorithm_family")
        task_type, source = cls.infer_task_type(
            model_type=model_type,
            algorithm_family=algorithm,
            manifest=manifest,
        )
        required, optional = cls._metrics(task_type)

        supported = [
            TrainingMode.FULL_RETRAIN,
            TrainingMode.PARAMETER_TUNING,
            TrainingMode.FEATURE_RECONSTRUCTION,
            TrainingMode.FEATURE_SELECTION,
        ]
        unsupported: list[TrainingMode] = []
        adapter_ready = True
        limitations: list[str] = []

        normalized_algorithm = cls._normalize(algorithm)
        if task_type == ModelTaskType.CLASSIFICATION:
            if normalized_algorithm == "lightgbm":
                supported.append(TrainingMode.INCREMENTAL_TRAIN)
            else:
                unsupported.append(TrainingMode.INCREMENTAL_TRAIN)
                limitations.append("Incremental training adapter is currently implemented only for LightGBM.")
        else:
            adapter_ready = False
            unsupported.append(TrainingMode.INCREMENTAL_TRAIN)
            limitations.append(
                f"{task_type.value} metrics are exposed as an interface; "
                "training and qualification adapters still need implementation."
            )

        if task_type == ModelTaskType.CLUSTERING:
            limitations.append("Clustering has no supervised label target; qualification must use unsupervised stability gates.")

        return ModelTaskProfile(
            model_id=model_id,
            champion_version=champion_version,
            model_type=model_type or manifest.get("model_type"),
            algorithm_family=algorithm,
            task_type=task_type,
            task_type_source=source,
            target_column=None if task_type == ModelTaskType.CLUSTERING else str(manifest.get("target_column") or "is_bad"),
            prediction_column=str(manifest.get("prediction_column") or ("cluster_id" if task_type == ModelTaskType.CLUSTERING else "y_pred_proba")),
            residual_column="residual" if task_type == ModelTaskType.REGRESSION else None,
            cluster_label_column="cluster_id" if task_type == ModelTaskType.CLUSTERING else None,
            required_metrics=required,
            optional_metrics=optional,
            supported_training_modes=supported,
            unsupported_training_modes=unsupported,
            adapter_ready=adapter_ready,
            limitations=limitations,
        )

    @staticmethod
    def risk_guardrails(profile: ModelTaskProfile) -> list[RiskGuardrailResult]:
        type_gap = [] if profile.task_type == ModelTaskType.CLASSIFICATION else [
            f"{profile.task_type.value} qualification gates are not wired to worker outputs yet."
        ]
        return [
            RiskGuardrailResult(
                risk_code=RiskGuardrailCode.OVERFITTING,
                status=GuardrailCoverageStatus.PARTIAL,
                implemented=True,
                blocking=True,
                covered_by=["train_valid_gap", "healthy_range", "same_sample_bootstrap", "OOT gate"],
                missing_controls=["train/validation/OOT unified overfit report", "feature contribution concentration gate", *type_gap],
                recommendation="Add task-type-specific generalization gates before publishing challengers.",
            ),
            RiskGuardrailResult(
                risk_code=RiskGuardrailCode.DATA_LEAKAGE,
                status=GuardrailCoverageStatus.IMPLEMENTED,
                implemented=True,
                blocking=True,
                covered_by=["W4/OOT access policy", "TrainingPlan OOT validator", "TrainingJob OOT validator", "frozen candidate identity", "snapshot checksum"],
                recommendation="Keep W4/OOT read access restricted to final qualification and deployment gates.",
            ),
            RiskGuardrailResult(
                risk_code=RiskGuardrailCode.CONCEPT_DRIFT_MISDIAGNOSIS,
                status=GuardrailCoverageStatus.PARTIAL,
                implemented=True,
                blocking=True,
                covered_by=["root cause confidence gate", "evidence coverage gate", "temporal precedence", "counterfactual/permutation validators"],
                missing_controls=["regression residual drift validator", "clustering cluster-drift validator", "active Sentinel prerequisite"],
                recommendation="Require task-specific drift validators before allowing automatic strategy selection.",
            ),
            RiskGuardrailResult(
                risk_code=RiskGuardrailCode.ITERATION_OSCILLATION,
                status=GuardrailCoverageStatus.PARTIAL,
                implemented=True,
                blocking=True,
                covered_by=["max_iteration_rounds", "active lifecycle dedupe", "cooldown", "idempotency key"],
                missing_controls=["oscillation_threshold consumer", "strategy flip-flop detector"],
                recommendation="Block or require review when recent runs alternate strategies or repeatedly fail the same gate.",
            ),
            RiskGuardrailResult(
                risk_code=RiskGuardrailCode.EXPLAINABILITY_LOSS,
                status=GuardrailCoverageStatus.PARTIAL,
                implemented=True,
                blocking=False,
                covered_by=["feature importance", "unstable feature attribution", "feature selection report"],
                missing_controls=["SHAP stability", "top feature concentration", "explanation drift gate"],
                recommendation="Make explanation stability a qualification gate for medium/high-risk strategies.",
            ),
            RiskGuardrailResult(
                risk_code=RiskGuardrailCode.SAMPLE_SELECTION_BIAS,
                status=GuardrailCoverageStatus.PARTIAL,
                implemented=True,
                blocking=True,
                covered_by=["sample size guardrail", "bad-count guardrail", "segment governance", "segment evidence"],
                missing_controls=["train-vs-online representativeness", "propensity/coverage bias check"],
                recommendation="Compare training snapshots against monitoring population before dispatching training.",
            ),
            RiskGuardrailResult(
                risk_code=RiskGuardrailCode.CUMULATIVE_DEGRADATION,
                status=GuardrailCoverageStatus.PARTIAL,
                implemented=True,
                blocking=True,
                covered_by=["pre-OOT qualification", "final OOT gate", "deployment health checks", "rollback recommendation"],
                missing_controls=["multi-version degradation trend gate", "cumulative champion-vs-challenger lineage score"],
                recommendation="Persist cross-version health deltas and block promotion on accumulated negative drift.",
            ),
        ]

    @classmethod
    def summarize(
        cls,
        *,
        model_id: str,
        champion_version: str = "champion_v1",
        model_type: str | None = None,
        algorithm_family: str | None = None,
    ) -> ModelTaskInterfaceSummary:
        profile = cls.build_profile(
            model_id=model_id,
            champion_version=champion_version,
            model_type=model_type,
            algorithm_family=algorithm_family,
        )
        return ModelTaskInterfaceSummary(
            profile=profile,
            risk_guardrails=cls.risk_guardrails(profile),
        )
