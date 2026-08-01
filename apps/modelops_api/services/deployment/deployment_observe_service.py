"""Deployment observation service.

This module converts stage health metrics into structured DeploymentAlert
objects. The KG layer then uses those alerts as lookup entries.
"""

from __future__ import annotations

from packages.models.common.enums import Severity
from packages.models.deployment.deployment_alert import DeploymentAlert

_METRIC_ALERT_MAP: dict[str, dict] = {
    "challenger_auc": {
        "alert_code": "CHALLENGER_AUC_REGRESSION",
        "direction": "BELOW_THRESHOLD",
        "rule_key": "min_auc",
        "aliases": ["AUC"],
    },
    "challenger_ks": {
        "alert_code": "CHALLENGER_KS_REGRESSION",
        "direction": "BELOW_THRESHOLD",
        "rule_key": "min_ks",
        "aliases": ["KS"],
    },
    "score_psi": {
        "alert_code": "HIGH_DEPLOYMENT_SCORE_PSI",
        "direction": "ABOVE_THRESHOLD",
        "rule_key": "max_score_psi",
        "aliases": [],
    },
    "bad_rate_drift": {
        "alert_code": "BAD_RATE_DRIFT_HIGH",
        "direction": "ABOVE_THRESHOLD",
        "rule_key": "max_bad_rate_drift",
        "aliases": [],
    },
    "train_valid_gap": {
        "alert_code": "TRAIN_VALID_GAP_LARGE",
        "direction": "ABOVE_THRESHOLD",
        "rule_key": "max_train_valid_gap",
        "aliases": [],
    },
    "recovery_rate": {
        "alert_code": "RECOVERY_RATE_LOW",
        "direction": "BELOW_THRESHOLD",
        "rule_key": "min_recovery_rate",
        "aliases": [],
    },
}


def _metric_value(health_metrics: dict, metric_code: str, aliases: list[str]) -> float | None:
    value = health_metrics.get(metric_code)
    if value is not None:
        return value
    for alias in aliases:
        value = health_metrics.get(alias)
        if value is not None:
            return value
    return None


def _is_failed(value: float | None, threshold: float | None, direction: str) -> bool:
    if value is None or threshold is None:
        return False
    if direction == "ABOVE_THRESHOLD":
        return value > threshold
    return value < threshold


def _failure_reasons_for(metric_code: str, failures: list[str], rollback_reasons: list[str]) -> list[str]:
    normalized_metric = metric_code.lower().replace("_", "")
    matched = []
    for reason in failures + rollback_reasons:
        normalized_reason = str(reason).lower().replace("_", "").replace(" ", "")
        if normalized_metric in normalized_reason:
            matched.append(str(reason))
    return matched


def build_deployment_alerts(
    stage: str,
    health_metrics: dict,
    health_result: dict,
    *,
    lifecycle_run_id: str | None = None,
    deployment_id: str | None = None,
) -> list[DeploymentAlert]:
    """Build DeploymentAlert objects from health metrics and stage result."""

    alerts: list[DeploymentAlert] = []
    if not health_metrics:
        return alerts

    failures = [str(item) for item in health_result.get("failures", [])]
    rollback_reasons = [str(item) for item in health_result.get("rollback_reasons", [])]

    try:
        from apps.modelops_api.services.iteration.deployment_safety_service import STAGE_HEALTH_RULES
        stage_rules = STAGE_HEALTH_RULES.get(stage, {})
    except Exception:
        stage_rules = {}

    for metric_code, config in _METRIC_ALERT_MAP.items():
        threshold = stage_rules.get(config["rule_key"])
        value = _metric_value(health_metrics, metric_code, config["aliases"])
        failed_by_rule = _is_failed(value, threshold, config["direction"])
        matched_reasons = _failure_reasons_for(metric_code, failures, rollback_reasons)

        if not failed_by_rule and not matched_reasons:
            continue

        severity = Severity.HIGH if failed_by_rule or matched_reasons else Severity.WARNING
        alerts.append(
            DeploymentAlert(
                alert_code=config["alert_code"],
                metric_code=metric_code,
                champion_value=health_metrics.get("champion_auc") or health_metrics.get("baseline_auc"),
                challenger_value=value if metric_code in {"challenger_auc", "challenger_ks"} else None,
                value=value,
                threshold=threshold,
                severity=severity,
                stage=stage,
                lifecycle_run_id=lifecycle_run_id,
                deployment_id=deployment_id,
                direction=config["direction"],
                evidence_detail={
                    "failure_reasons": matched_reasons,
                    "health_passed": health_result.get("passed", False),
                    "rollback_recommended": health_result.get("rollback_recommended", False),
                    "stage": stage,
                },
            )
        )

    if health_metrics.get("oot_passed") is False or any("oot" in f.lower() for f in failures):
        alerts.append(
            DeploymentAlert(
                alert_code="OOT_DEPLOYMENT_RISK",
                metric_code="oot_passed",
                value=0,
                threshold=1,
                severity=Severity.HIGH,
                stage=stage,
                lifecycle_run_id=lifecycle_run_id,
                deployment_id=deployment_id,
                direction="BELOW_THRESHOLD",
                evidence_detail={"failure_reasons": failures, "stage": stage},
            )
        )

    for metric_code, alert_code in [
        ("discrimination_passed", "DISCRIMINATION_GATE_FAILED"),
        ("calibration_passed", "CALIBRATION_GATE_FAILED"),
    ]:
        if health_metrics.get(metric_code) is False:
            alerts.append(
                DeploymentAlert(
                    alert_code=alert_code,
                    metric_code=metric_code,
                    value=0,
                    threshold=1,
                    severity=Severity.HIGH,
                    stage=stage,
                    lifecycle_run_id=lifecycle_run_id,
                    deployment_id=deployment_id,
                    direction="BELOW_THRESHOLD",
                    evidence_detail={"failure_reasons": failures, "stage": stage},
                )
            )

    return alerts
