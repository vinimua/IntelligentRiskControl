import pytest
import numpy as np

from apps.modelops_api.repositories.iteration_repo import IterationRepo
from apps.modelops_api.services.deployment.deployment_health_check_service import (
    DeploymentHealthCheckService,
)
from apps.modelops_api.services.iteration.model_comparison_service import (
    ModelComparisonService,
)


def test_model_comparison_missing_metric_fails_report():
    report = ModelComparisonService().compare(
        y_true=np.array([1, 1, 1, 1]),
        champion_scores=np.array([0.1, 0.2, 0.3, 0.4]),
        challenger_scores=np.array([0.2, 0.3, 0.4, 0.5]),
    )

    assert report.passed is False
    assert len(report.metrics) == 10
    assert any(metric.passed is False for metric in report.metrics)


@pytest.mark.asyncio
async def test_deployment_health_missing_required_metrics_fails():
    report = await DeploymentHealthCheckService().check(
        deployment_id="deployment-001",
        stage="OFFLINE_VALIDATION",
        health_metrics={},
    )

    assert report.passed is False
    failed_metrics = {check.metric for check in report.checks if not check.passed}
    assert {"challenger_auc", "challenger_ks", "score_psi"} <= failed_metrics


class _FakeResult:
    def __init__(self, row):
        self.row = row

    def mappings(self):
        return self

    def first(self):
        return self.row


class _FakeSession:
    def __init__(self, row):
        self.row = row

    async def execute(self, *args, **kwargs):
        return _FakeResult(self.row)


@pytest.mark.asyncio
async def test_get_comparison_report_returns_report_payload():
    session = _FakeSession({
        "plan_id": "cmp-001",
        "plan_type": "MODEL_COMPARISON",
        "request_json": {
            "comparison_id": "cmp-001",
            "model_id": "credit_model_001",
            "metrics": [],
            "passed": False,
        },
    })

    report = await IterationRepo(session).get_comparison_report("cmp-001")

    assert report == {
        "comparison_id": "cmp-001",
        "model_id": "credit_model_001",
        "metrics": [],
        "passed": False,
    }
