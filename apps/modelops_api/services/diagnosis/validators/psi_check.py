"""psi_check 验证器 — D 类型证据：特征分布漂移是否导致了指标异常。"""

from __future__ import annotations

import uuid

from packages.models.diagnosis.evidence import EvidenceItem
from packages.models.common.enums import (
    AvailabilityStatus,
    ConfidenceLevel,
    EvidenceDirection,
    EvidenceType,
)


async def psi_check(
    drift_rows: list[dict],
    alert_metric_code: str,
    **_kwargs,
) -> EvidenceItem:
    """检查高 PSI 特征是否与告警指标相关。

    Args:
        drift_rows: 逐特征漂移数据（从 monitoring_feature_drift 读取）
        alert_metric_code: 告警的指标代码（如 AUC_DROP, KS_DROP）

    Returns:
        EvidenceItem with direction and normalized score.
    """
    max_psi = max(
        (d["psi"] for d in drift_rows if d.get("psi") is not None), default=0
    )
    high_psi = [d for d in drift_rows if d.get("psi") is not None and d["psi"] > 0.1]

    if not high_psi:
        return EvidenceItem(
            evidence_id=str(uuid.uuid4()),
            evidence_type=EvidenceType.D,
            method_code="psi_check",
            executor_version="V1",
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
        executor_version="V1",
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
