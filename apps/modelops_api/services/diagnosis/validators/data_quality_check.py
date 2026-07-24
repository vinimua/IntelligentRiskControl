"""data_quality_check 验证器 — D 类型证据：数据质量恶化是否导致了指标异常。"""

from __future__ import annotations

import uuid

from packages.models.diagnosis.evidence import EvidenceItem
from packages.models.common.enums import (
    AvailabilityStatus,
    ConfidenceLevel,
    EvidenceDirection,
    EvidenceType,
)


async def data_quality_check(
    drift_rows: list[dict],
    alert_metric_code: str,
    **_kwargs,
) -> EvidenceItem:
    """检查数据质量劣化信号。

    Args:
        drift_rows: 逐特征漂移数据（含 quality 字段）
        alert_metric_code: 告警指标代码

    Returns:
        EvidenceItem with direction and score.
    """
    alert_features = [
        d for d in drift_rows
        if d.get("dq_flag") in ("ALERT", "WARN")
    ]
    max_missing_rate = max(
        (d.get("missing_rate", 0) or 0 for d in drift_rows if d.get("missing_rate") is not None),
        default=0,
    )

    if not alert_features and max_missing_rate < 0.05:
        return EvidenceItem(
            evidence_id=str(uuid.uuid4()),
            evidence_type=EvidenceType.D,
            method_code="missing_outlier_range_check",
            executor_version="V1",
            normalized_score=0.05,
            direction=EvidenceDirection.AGAINST,
            applicable=True,
            confidence_level=ConfidenceLevel.HIGH,
            evidence_detail_json={
                "dq_alert_count": 0,
                "max_missing_rate": max_missing_rate,
                "message": "No data quality alerts found — all features OK",
            },
        )

    score = min(len(alert_features) * 0.2, 1.0)
    return EvidenceItem(
        evidence_id=str(uuid.uuid4()),
        evidence_type=EvidenceType.D,
        method_code="missing_outlier_range_check",
        executor_version="V1",
        normalized_score=round(score, 4),
        direction=EvidenceDirection.SUPPORT,
        applicable=True,
        confidence_level=ConfidenceLevel.HIGH if score > 0.6 else ConfidenceLevel.MEDIUM,
        evidence_detail_json={
            "dq_alert_count": len(alert_features),
            "alert_features": [d["feature_name"] for d in alert_features[:5]],
            "max_missing_rate": max_missing_rate,
        },
    )
