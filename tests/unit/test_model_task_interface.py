from packages.models.common.enums import (
    GuardrailCoverageStatus,
    ModelTaskType,
    RiskGuardrailCode,
    TrainingMode,
)

from apps.modelops_api.services.iteration.model_task_interface_service import (
    ModelTaskInterfaceService,
)


def _metric_codes(metrics):
    return {item.metric_code for item in metrics}


def test_credit_risk_defaults_to_classification_interface():
    summary = ModelTaskInterfaceService.summarize(
        model_id="credit_model_001",
        model_type="CREDIT_RISK",
        algorithm_family="LightGBM",
    )

    assert summary.profile.task_type == ModelTaskType.CLASSIFICATION
    assert {"AUC", "KS"}.issubset(_metric_codes(summary.profile.required_metrics))
    assert TrainingMode.INCREMENTAL_TRAIN in summary.profile.supported_training_modes
    assert summary.profile.adapter_ready is True


def test_regression_interface_exposes_residual_metrics_and_not_ready_adapter():
    summary = ModelTaskInterfaceService.summarize(
        model_id="loss_amount_model",
        model_type="REGRESSION",
        algorithm_family="LightGBMRegressor",
    )

    assert summary.profile.task_type == ModelTaskType.REGRESSION
    assert {"RMSE", "MAE", "R2", "RESIDUAL_PSI"}.issubset(
        _metric_codes(summary.profile.required_metrics)
    )
    assert summary.profile.residual_column == "residual"
    assert summary.profile.adapter_ready is False
    assert TrainingMode.INCREMENTAL_TRAIN in summary.profile.unsupported_training_modes


def test_clustering_interface_exposes_unsupervised_metrics():
    summary = ModelTaskInterfaceService.summarize(
        model_id="customer_cluster_model",
        model_type="CLUSTERING",
        algorithm_family="KMeans",
    )

    assert summary.profile.task_type == ModelTaskType.CLUSTERING
    assert {"SILHOUETTE", "CLUSTER_STABILITY", "CLUSTER_PSI"}.issubset(
        _metric_codes(summary.profile.required_metrics)
    )
    assert all(not item.label_required for item in summary.profile.required_metrics)
    assert summary.profile.target_column is None
    assert summary.profile.cluster_label_column == "cluster_id"


def test_risk_guardrail_interface_lists_all_seven_risks():
    summary = ModelTaskInterfaceService.summarize(
        model_id="credit_model_001",
        model_type="CREDIT_RISK",
        algorithm_family="RandomForest",
    )

    risks = {item.risk_code: item for item in summary.risk_guardrails}
    assert set(risks) == set(RiskGuardrailCode)
    assert risks[RiskGuardrailCode.DATA_LEAKAGE].status == GuardrailCoverageStatus.IMPLEMENTED
    assert risks[RiskGuardrailCode.OVERFITTING].status == GuardrailCoverageStatus.PARTIAL
    assert TrainingMode.INCREMENTAL_TRAIN in summary.profile.unsupported_training_modes
