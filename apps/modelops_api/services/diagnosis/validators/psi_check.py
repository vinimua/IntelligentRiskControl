"""psi_check 验证器 — D 类型证据：特征分布漂移是否导致了指标异常。

根因感知（2026-08-15 治理收窄）：
  特征 PSI 漂移本身只直接支持"漂移类"假设。对非漂移类根因
  （business_policy_change / data_pipeline_issue / data_quality_issue /
  PRIOR_PROBABILITY_SHIFT），漂移数据不构成支持证据也不构成反证，
  返回 applicable=False 的中性证据，防止 blanket SUPPORT 污染排序。

  population_shift（客群结构迁移）只有分类特征的份额变化才是判别证据：
  纯数值特征漂移与 covariate drift 不可区分 → NEUTRAL；
  存在分类特征显著 PSI → SUPPORT。
"""

from __future__ import annotations

import uuid

from packages.models.diagnosis.evidence import EvidenceItem
from packages.models.common.enums import (
    AvailabilityStatus,
    ConfidenceLevel,
    EvidenceDirection,
    EvidenceType,
)

# ── 漂移类根因：PSI 漂移直接支持这些假设 ──
_DRIFT_CONSISTENT_ROOT_CAUSES = {
    "feature_drift",
    "feature_failure",
    "model_aging",
    "CONCEPT_DRIFT",
    "FRAUD_PATTERN_SHIFT",
}


def _neutral_not_applicable(message: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=str(uuid.uuid4()),
        evidence_type=EvidenceType.D,
        method_code="psi_check",
        executor_version="V2",
        normalized_score=None,
        direction=EvidenceDirection.NEUTRAL,
        applicable=False,
        availability_status=AvailabilityStatus.NOT_APPLICABLE,
        confidence_level=ConfidenceLevel.LOW,
        evidence_detail_json={"message": message},
    )


async def psi_check(
    drift_rows: list[dict],
    alert_metric_code: str,
    root_cause_code: str | None = None,
    **_kwargs,
) -> EvidenceItem:
    """检查高 PSI 特征是否与告警指标相关。

    Args:
        drift_rows: 逐特征漂移数据（从 monitoring_feature_drift 读取）
        alert_metric_code: 告警的指标代码（如 AUC_DROP, KS_DROP）
        root_cause_code: 正在验证的候选根因码（根因感知）

    Returns:
        EvidenceItem with direction and normalized score.
    """
    # ── 0. 根因感知：非漂移类假设不适用本证据 ──
    if root_cause_code and root_cause_code not in _DRIFT_CONSISTENT_ROOT_CAUSES:
        if root_cause_code == "population_shift":
            categorical_psi = [
                d for d in drift_rows
                if d.get("psi") is not None and d["psi"] > 0.1
                and str(d.get("feature_type") or "").lower() in {
                    "categorical", "category",
                }
            ]
            if not categorical_psi:
                return _neutral_not_applicable(
                    "纯数值特征漂移与 covariate drift 不可区分，"
                    "不能作为客群结构迁移的判别证据"
                )
            # 有分类份额漂移 → 走正常 SUPPORT 路径
        else:
            return _neutral_not_applicable(
                f"特征 PSI 漂移不直接支持根因 {root_cause_code} 的假设，"
                "本证据不参与该候选评分"
            )

    max_psi = max(
        (d["psi"] for d in drift_rows if d.get("psi") is not None), default=0
    )
    high_psi = [d for d in drift_rows if d.get("psi") is not None and d["psi"] > 0.1]

    if not high_psi:
        return EvidenceItem(
            evidence_id=str(uuid.uuid4()),
            evidence_type=EvidenceType.D,
            method_code="psi_check",
            executor_version="V2",
            normalized_score=0.1,
            direction=EvidenceDirection.AGAINST,
            applicable=True,
            confidence_level=ConfidenceLevel.HIGH,
            evidence_detail_json={
                "max_psi": max_psi,
                "high_psi_count": 0,
                "threshold": 0.1,
                "message": f"No features with PSI > 0.1 found (max={max_psi:.4f})",
            },
        )

    top_feature = max(high_psi, key=lambda d: d["psi"])
    score = min(top_feature["psi"] / 0.25, 1.0)

    return EvidenceItem(
        evidence_id=str(uuid.uuid4()),
        evidence_type=EvidenceType.D,
        method_code="psi_check",
        executor_version="V2",
        normalized_score=round(score, 4),
        direction=EvidenceDirection.SUPPORT,
        applicable=True,
        confidence_level=ConfidenceLevel.HIGH if score > 0.8 else ConfidenceLevel.MEDIUM,
        evidence_detail_json={
            "max_psi": max_psi,
            "high_psi_count": len(high_psi),
            "top_feature": top_feature["feature_name"],
            "top_psi": top_feature["psi"],
            "top_features": [d["feature_name"] for d in high_psi[:5]],
            "alert_metric": alert_metric_code,
        },
    )
