"""P2: 部署安全服务 — 健康检查 + 自动/手动回滚。

职责：
- 检查每个阶段的健康指标是否通过
- 不通过 → 自动 HOLD 或 ROLLBACK
- 回滚 → 更新 model_deployment_state 恢复 champion
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import structlog

logger = structlog.get_logger(__name__)

# 阶段 → challenger 流量比例映射
STAGE_TRAFFIC_RATIO: dict[str, float] = {
    "OFFLINE_VALIDATION": 0.0,
    "OOT_GATE": 0.0,
    "SHADOW": 0.0,
    "CANARY_5": 0.05,
    "CANARY_20": 0.20,
    "CANARY_50": 0.50,
    "PRODUCTION": 1.0,
}

# 每阶段的健康检查阈值
STAGE_HEALTH_RULES: dict[str, dict] = {
    "OFFLINE_VALIDATION": {
        "min_auc": 0.70,
        "min_ks": 0.20,
        "max_score_psi": 0.25,
    },
    "OOT_GATE": {
        "min_auc": 0.70,
        "min_ks": 0.25,
        "max_score_psi": 0.25,
        "max_train_valid_gap": 0.05,
    },
    "SHADOW": {
        "min_auc": 0.72,
        "max_score_psi": 0.15,
    },
    "CANARY_5": {
        "min_auc": 0.72,
        "max_bad_rate_drift": 0.10,
    },
    "CANARY_20": {
        "min_auc": 0.73,
        "max_bad_rate_drift": 0.08,
    },
    "CANARY_50": {
        "min_auc": 0.73,
        "max_bad_rate_drift": 0.05,
        "min_recovery_rate": 0.5,
    },
    "PRODUCTION": {
        "min_auc": 0.74,
        "max_bad_rate_drift": 0.03,
        "min_recovery_rate": 0.6,
    },
}


class DeploymentSafetyService:
    """部署安全门禁 — 检查健康指标并决定推进/暂停/回滚。"""

    def __init__(self, session):
        self.session = session

    @staticmethod
    def check_stage_health(
        stage: str,
        health_metrics: dict,
    ) -> dict:
        """检查当前阶段的健康指标。

        Returns:
            {passed: bool, failures: list[str], warnings: list[str]}
        """
        rules = STAGE_HEALTH_RULES.get(stage, {})
        failures: list[str] = []
        warnings: list[str] = []
        rollback_reasons: list[str] = []

        if not health_metrics:
            return {
                "passed": True,
                "failures": [],
                "warnings": ["no_health_metrics_provided"],
                "rollback_recommended": False,
                "rollback_reasons": [],
            }

        # AUC 检查
        auc = health_metrics.get("challenger_auc") or health_metrics.get("AUC")
        min_auc = rules.get("min_auc")
        if auc is not None and min_auc is not None and auc < min_auc:
            failures.append(f"AUC {auc:.4f} < {min_auc}")
            if auc < min_auc - 0.03:
                rollback_reasons.append(f"severe_auc_drop:{auc:.4f}<{min_auc - 0.03:.4f}")

        # KS 检查
        ks = health_metrics.get("challenger_ks") or health_metrics.get("KS")
        min_ks = rules.get("min_ks")
        if ks is not None and min_ks is not None and ks < min_ks:
            failures.append(f"KS {ks:.4f} < {min_ks}")
            if ks < min_ks * 0.75:
                rollback_reasons.append(f"severe_ks_drop:{ks:.4f}<{min_ks * 0.75:.4f}")

        # PSI 检查
        psi = health_metrics.get("score_psi")
        max_psi = rules.get("max_score_psi")
        if psi is not None and max_psi is not None and psi > max_psi:
            failures.append(f"Score PSI {psi:.4f} > {max_psi}")
            if psi > max_psi * 2:
                rollback_reasons.append(f"severe_score_psi:{psi:.4f}>{max_psi * 2:.4f}")

        # Bad rate drift
        bad_rate_drift = health_metrics.get("bad_rate_drift")
        max_drift = rules.get("max_bad_rate_drift")
        if bad_rate_drift is not None and max_drift is not None and bad_rate_drift > max_drift:
            failures.append(f"Bad rate drift {bad_rate_drift:.4f} > {max_drift}")
            if bad_rate_drift > max_drift * 2:
                rollback_reasons.append(
                    f"severe_bad_rate_drift:{bad_rate_drift:.4f}>{max_drift * 2:.4f}"
                )

        # Recovery rate
        recovery_rate = health_metrics.get("recovery_rate")
        min_recovery = rules.get("min_recovery_rate")
        if recovery_rate is not None and min_recovery is not None and recovery_rate < min_recovery:
            failures.append(f"Recovery rate {recovery_rate:.4f} < {min_recovery}")
            if recovery_rate < min_recovery * 0.5:
                rollback_reasons.append(
                    f"severe_recovery_loss:{recovery_rate:.4f}<{min_recovery * 0.5:.4f}"
                )

        # Train/valid gap
        gap = health_metrics.get("train_valid_gap")
        max_gap = rules.get("max_train_valid_gap")
        if gap is not None and max_gap is not None and gap > max_gap:
            failures.append(f"Train/valid gap {gap:.4f} > {max_gap}")
            if gap > max_gap * 2:
                rollback_reasons.append(f"severe_train_valid_gap:{gap:.4f}>{max_gap * 2:.4f}")

        # Discrimination
        if health_metrics.get("discrimination_passed") is False:
            failures.append("Discrimination gate not passed")
            rollback_reasons.append("discrimination_gate_failed")
        if health_metrics.get("calibration_passed") is False:
            failures.append("Calibration gate not passed")
            rollback_reasons.append("calibration_gate_failed")
        if health_metrics.get("oot_passed") is False:
            failures.append("OOT gate not passed")
            rollback_reasons.append("oot_gate_failed")
        if health_metrics.get("segment_governance_passed") is False:
            warnings.append("Segment governance not passed")

        passed = len(failures) == 0
        return {
            "passed": passed,
            "failures": failures,
            "warnings": warnings,
            "stage": stage,
            "rollback_recommended": bool(rollback_reasons),
            "rollback_reasons": rollback_reasons,
        }

    async def rollback(
        self,
        deployment: dict,
        reason: str = "HEALTH_CHECK_FAILED",
        rollback_target: str | None = None,
        updated_by: str = "system",
    ) -> dict:
        """执行回滚操作。

        1. 更新 deployment_records 为 ROLLED_BACK
        2. 写入 ROLLBACK stage_record
        3. 恢复 model_deployment_state: champion 回到 stable，challenger 流量归零
        """
        from sqlalchemy import text

        now = datetime.now(timezone.utc)
        deployment_id = str(deployment["deployment_id"])
        model_id = deployment.get("model_id")
        champion = deployment.get("champion_version", "")
        current_stage = deployment.get("current_stage", "")
        target = rollback_target or champion

        # 1. 更新 deployment_records
        await self.session.execute(
            text("""
                UPDATE iteration.deployment_records
                SET status = 'ROLLED_BACK',
                    decision = 'ROLLBACK',
                    current_stage = :stage,
                    completed_at = :now,
                    updated_at = :now,
                    record_json = record_json || CAST(:detail AS JSONB)
                WHERE deployment_id = :did
            """),
            {
                "did": deployment_id,
                "stage": current_stage,
                "now": now,
                "detail": json.dumps({
                    "rollback_reason": reason,
                    "rollback_target": target,
                    "rolled_back_by": updated_by,
                    "rolled_back_at": now.isoformat(),
                }),
            },
        )

        # 2. 写入 stage_record
        await self.session.execute(
            text("""
                INSERT INTO iteration.deployment_stage_records
                    (deployment_id, stage, decision, status, health_json, result_json)
                VALUES (:did, :stage, 'ROLLBACK', 'ROLLED_BACK',
                        CAST(:health AS JSONB), CAST(:result AS JSONB))
            """),
            {
                "did": deployment_id,
                "stage": current_stage,
                "health": json.dumps({"rollback_reason": reason}),
                "result": json.dumps({
                    "deployment_id": deployment_id,
                    "rollback_target": target,
                    "rolled_back_by": updated_by,
                    "rolled_back_at": now.isoformat(),
                }),
            },
        )

        # 3. 恢复 model_deployment_state
        if model_id:
            await self.session.execute(
                text("""
                    INSERT INTO model_registry.model_deployment_state
                        (model_id, environment, active_version_code, stable_version_code,
                         challenger_version_code, challenger_traffic_ratio, state_version, updated_by)
                    VALUES (:mid, 'PROD', :stable, :stable, NULL, 0, 1, :by)
                    ON CONFLICT (model_id, environment) DO UPDATE SET
                        active_version_code      = EXCLUDED.stable_version_code,
                        challenger_version_code  = NULL,
                        challenger_traffic_ratio = 0,
                        state_version            = model_registry.model_deployment_state.state_version + 1,
                        updated_by               = EXCLUDED.updated_by,
                        updated_at               = NOW()
                """),
                {
                    "mid": model_id,
                    "stable": target,
                    "by": updated_by,
                },
            )
            await self.session.execute(
                text("""
                    UPDATE model_registry.models
                    SET current_champion_version = :target,
                        stable_version = COALESCE(stable_version, :target),
                        updated_at = NOW()
                    WHERE model_id = :mid
                """),
                {"mid": model_id, "target": target},
            )

        logger.info(
            "deployment_rolled_back",
            deployment_id=deployment_id,
            model_id=model_id,
            reason=reason,
            rollback_target=target,
        )

        return {
            "deployment_id": deployment_id,
            "status": "ROLLED_BACK",
            "reason": reason,
            "rollback_target": target,
            "rolled_back_at": now.isoformat(),
        }

    async def promote_to_champion(
        self,
        deployment: dict,
        updated_by: str = "system",
    ) -> dict:
        """将 challenger 提升为 champion（PRODUCTION 阶段通过后调用）。

        1. 更新 model_deployment_state: challenger → active，旧 champion → stable
        2. 更新 deployment_records 为 PROMOTED
        """
        from sqlalchemy import text

        now = datetime.now(timezone.utc)
        deployment_id = str(deployment["deployment_id"])
        model_id = deployment.get("model_id")
        champion = deployment.get("champion_version", "")
        challenger = deployment.get("candidate_version", "")

        # 1. 更新 model_deployment_state
        if model_id:
            await self.session.execute(
                text("""
                    INSERT INTO model_registry.model_deployment_state
                        (model_id, environment, active_version_code, stable_version_code,
                         challenger_version_code, challenger_traffic_ratio, state_version, updated_by)
                    VALUES (:mid, 'PROD', :challenger, :champion, NULL, 0, 1, :by)
                    ON CONFLICT (model_id, environment) DO UPDATE SET
                        active_version_code      = :challenger,
                        stable_version_code      = :champion,
                        challenger_version_code  = NULL,
                        challenger_traffic_ratio = 0,
                        state_version            = model_registry.model_deployment_state.state_version + 1,
                        updated_by               = :by,
                        updated_at               = NOW()
                """),
                {"mid": model_id, "champion": champion, "challenger": challenger, "by": updated_by},
            )
            await self.session.execute(
                text("""
                    UPDATE model_registry.models
                    SET current_champion_version = :challenger,
                        stable_version = :champion,
                        updated_at = NOW()
                    WHERE model_id = :mid
                """),
                {"mid": model_id, "champion": champion, "challenger": challenger},
            )

        # 2. 更新 deployment_records
        await self.session.execute(
            text("""
                UPDATE iteration.deployment_records
                SET status = 'PROMOTED',
                    decision = 'PROMOTE',
                    completed_at = :now,
                    updated_at = :now,
                    record_json = record_json || CAST(:detail AS JSONB)
                WHERE deployment_id = :did
            """),
            {
                "did": deployment_id,
                "now": now,
                "detail": json.dumps({
                    "promoted_champion": challenger,
                    "previous_champion": champion,
                    "promoted_by": updated_by,
                    "promoted_at": now.isoformat(),
                }),
            },
        )

        logger.info(
            "challenger_promoted_to_champion",
            deployment_id=deployment_id,
            model_id=model_id,
            new_champion=challenger,
            old_champion=champion,
        )

        return {
            "deployment_id": deployment_id,
            "status": "PROMOTED",
            "new_champion": challenger,
            "previous_champion": champion,
            "promoted_at": now.isoformat(),
        }

    async def update_traffic_ratio(
        self,
        model_id: str,
        stage: str,
        champion_version: str | None = None,
        challenger_version: str | None = None,
        updated_by: str = "system",
    ) -> float:
        """根据当前阶段更新 model_deployment_state 的 traffic_ratio。"""
        ratio = STAGE_TRAFFIC_RATIO.get(stage, 0.0)

        from sqlalchemy import text
        await self.session.execute(
            text("""
                INSERT INTO model_registry.model_deployment_state
                    (model_id, environment, active_version_code, stable_version_code,
                     challenger_version_code, challenger_traffic_ratio, state_version, updated_by)
                VALUES (:mid, 'PROD', :active, :stable, :challenger, :ratio, 1, :by)
                ON CONFLICT (model_id, environment) DO UPDATE SET
                    active_version_code      = COALESCE(:active, model_registry.model_deployment_state.active_version_code),
                    stable_version_code      = COALESCE(:stable, model_registry.model_deployment_state.stable_version_code),
                    challenger_version_code  = COALESCE(:challenger, model_registry.model_deployment_state.challenger_version_code),
                    challenger_traffic_ratio = :ratio,
                    state_version            = model_registry.model_deployment_state.state_version + 1,
                    updated_by               = :by,
                    updated_at               = NOW()
            """),
            {
                "mid": model_id,
                "active": champion_version,
                "stable": champion_version,
                "challenger": challenger_version,
                "ratio": ratio,
                "by": updated_by,
            },
        )

        logger.info(
            "traffic_ratio_updated",
            model_id=model_id,
            stage=stage,
            ratio=ratio,
        )
        return ratio
