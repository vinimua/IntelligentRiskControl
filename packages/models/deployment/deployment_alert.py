"""Deployment alert contracts.

DeploymentAlert is the structured input for deployment KG lookup:
DeploymentAlert -> DeploymentRisk -> DeploymentStrategy.
"""

from ..common.base import ContractModel
from ..common.enums import Severity


class DeploymentAlert(ContractModel):
    """A deployment-stage abnormal signal produced from health metrics."""

    alert_code: str
    metric_code: str
    champion_value: float | None = None
    challenger_value: float | None = None
    value: float | None = None
    threshold: float | None = None
    severity: Severity = Severity.WARNING
    stage: str

    lifecycle_run_id: str | None = None
    deployment_id: str | None = None
    direction: str | None = None
    evidence_detail: dict | None = None
