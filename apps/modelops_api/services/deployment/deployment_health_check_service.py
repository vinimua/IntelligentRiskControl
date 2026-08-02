"""T4-GAP-04: 部署健康检查服务。

每个灰度阶段生成 DeploymentHealthReport，包含：
- 阶段内 challenger AUC / KS
- 线上预测分布 PSI
- 业务拒绝率异常
- 分群指标
- 健康检查详情记录到 deployment_stage_records.health_metrics
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from packages.models.iteration.deployment_health import DeploymentHealthReport

logger = structlog.get_logger(__name__)

# 每阶段健康阈值（已有，从 DeploymentSafetyService 复用）
from ..iteration.deployment_safety_service import STAGE_HEALTH_RULES, STAGE_TRAFFIC_RATIO


class DeploymentHealthCheckService:
    """为每个部署阶段生成结构化健康报告。"""

    async def check(
        self,
        deployment_id: str,
        stage: str,
        health_metrics: dict,
        *,
        lifecycle_run_id: str | None = None,
        model_id: str = "",
    ) -> DeploymentHealthReport:
        """生成完整健康报告。"""
        rules = STAGE_HEALTH_RULES.get(stage, {})
        traffic_ratio = STAGE_TRAFFIC_RATIO.get(stage, 0.0)

        checks: list[dict] = []
        all_passed = True

        # ── 1. Challenger AUC ──
        auc = _first_present(health_metrics, "challenger_auc", "AUC")
        min_auc = rules.get("min_auc")
        auc_passed = True
        if min_auc is not None and auc is None:
            auc_passed = False
            all_passed = False
        elif auc is not None and min_auc is not None:
            auc_passed = auc >= min_auc
            if not auc_passed:
                all_passed = False
        checks.append({
            "metric": "challenger_auc", "value": auc, "threshold": min_auc,
            "direction": ">=", "passed": auc_passed,
            "detail": f"AUC={auc:.4f} vs threshold={min_auc}" if auc is not None else "missing_required_metric",
        })

        # ── 2. KS ──
        ks = _first_present(health_metrics, "challenger_ks", "KS")
        min_ks = rules.get("min_ks")
        ks_passed = True
        if min_ks is not None and ks is None:
            ks_passed = False
            all_passed = False
        elif ks is not None and min_ks is not None:
            ks_passed = ks >= min_ks
            if not ks_passed:
                all_passed = False
        checks.append({
            "metric": "challenger_ks", "value": ks, "threshold": min_ks,
            "direction": ">=", "passed": ks_passed,
        })

        # ── 3. Score PSI ──
        psi = health_metrics.get("score_psi")
        max_psi = rules.get("max_score_psi")
        psi_passed = True
        if max_psi is not None and psi is None:
            psi_passed = False
            all_passed = False
        elif psi is not None and max_psi is not None:
            psi_passed = psi <= max_psi
            if not psi_passed:
                all_passed = False
        checks.append({
            "metric": "score_psi", "value": psi, "threshold": max_psi,
            "direction": "<=", "passed": psi_passed,
        })

        # ── 4. Bad rate drift ──
        bad_drift = health_metrics.get("bad_rate_drift")
        max_drift = rules.get("max_bad_rate_drift")
        drift_passed = True
        if max_drift is not None and bad_drift is None:
            drift_passed = False
            all_passed = False
        elif bad_drift is not None and max_drift is not None:
            drift_passed = bad_drift <= max_drift
            if not drift_passed:
                all_passed = False
        checks.append({
            "metric": "bad_rate_drift", "value": bad_drift, "threshold": max_drift,
            "direction": "<=", "passed": drift_passed,
        })

        # ── 5. Recovery rate ──
        rec = health_metrics.get("recovery_rate")
        min_rec = rules.get("min_recovery_rate")
        rec_passed = True
        if min_rec is not None and rec is None:
            rec_passed = False
            all_passed = False
        elif rec is not None and min_rec is not None:
            rec_passed = rec >= min_rec
            if not rec_passed:
                all_passed = False
        checks.append({
            "metric": "recovery_rate", "value": rec, "threshold": min_rec,
            "direction": ">=", "passed": rec_passed,
        })

        # ── 6. Train/valid gap ──
        gap = health_metrics.get("train_valid_gap")
        max_gap = rules.get("max_train_valid_gap")
        gap_passed = True
        if max_gap is not None and gap is None:
            gap_passed = False
            all_passed = False
        elif gap is not None and max_gap is not None:
            gap_passed = gap <= max_gap
            if not gap_passed:
                all_passed = False
        checks.append({
            "metric": "train_valid_gap", "value": gap, "threshold": max_gap,
            "direction": "<=", "passed": gap_passed,
        })

        # ── 7. Gate flags ──
        for gate in ["discrimination_passed", "calibration_passed", "oot_passed", "segment_governance_passed"]:
            v = health_metrics.get(gate)
            if v is False:
                checks.append({"metric": gate, "value": False, "threshold": True, "direction": "==", "passed": False})
                all_passed = False

        # ── 8. Online metrics (simulated) ──
        online_metrics = health_metrics.get("online_metrics", {})
        rejection_rate = online_metrics.get("rejection_rate")
        rejection_limit = 0.15
        if rejection_rate is not None and rejection_rate > rejection_limit:
            checks.append({
                "metric": "rejection_rate", "value": rejection_rate, "threshold": rejection_limit,
                "direction": "<=", "passed": False,
            })
            all_passed = False

        rollback_recommended = (
            not auc_passed or
            (bad_drift is not None and max_drift is not None and bad_drift > max_drift * 2) or
            (psi is not None and max_psi is not None and psi > max_psi * 1.5)
        )

        now = datetime.now(timezone.utc)
        return DeploymentHealthReport(
            deployment_id=deployment_id,
            stage=stage,
            lifecycle_run_id=lifecycle_run_id,
            model_id=model_id,
            traffic_ratio=traffic_ratio,
            passed=all_passed,
            checks=checks,
            rollback_recommended=rollback_recommended,
            rollback_reasons=(
                ["severe_health_failure"] if rollback_recommended else []
            ),
            created_at=now.isoformat(),
        )


def _first_present(values: dict, *keys: str):
    for key in keys:
        if key in values:
            return values[key]
    return None
