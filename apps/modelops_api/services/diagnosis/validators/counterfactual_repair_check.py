"""counterfactual_repair_check 验证器 — R 类型证据：修复漂移特征后性能是否可恢复。

核心逻辑：
  如果漂移严重的特征恰好是模型依赖的高重要性特征，那么修复这些特征
  （重新对齐分布）理论上能将模型性能拉回基线 → 支持"需要迭代模型"的结论。

输入:
  - drift_rows: 逐特征漂移数据（含 psi 等指标）
  - feature_importance: dict[feature_name, importance_score]
  - alert_metric_code: 告警指标代码（如 AUC_DROP, KS_DROP）

输出: EvidenceItem with direction and normalized_score.

公式:
  repair_potential = Σ(psi_i × importance_i) / Σ(importance_i) 对所有漂移特征
  normalized_score = clamp(repair_potential / 0.5, 0, 1)

解释:
  - repair_potential > 0.3 → SUPPORT：修复高重要性且严重漂移的特征，模型大概率恢复
  - repair_potential < 0.1 → AGAINST：漂移的特征不重要，修复没用，得找其他原因
  - 中间 → NEUTRAL
"""

from __future__ import annotations

import uuid

from packages.models.diagnosis.evidence import EvidenceItem
from packages.models.common.enums import (
    AvailabilityStatus,
    ConfidenceLevel,
    DataTrack,
    EvidenceDirection,
    EvidenceType,
)


async def counterfactual_repair_check(
    drift_rows: list[dict],
    alert_metric_code: str,
    feature_importance: dict[str, float] | None = None,
    **_kwargs,
) -> EvidenceItem:
    """R 类型验证器：反事实修复潜力评估。

    Args:
        drift_rows: 逐特征漂移数据
        alert_metric_code: 告警指标代码
        feature_importance: 特征名 → 重要性分数 映射（如缺失则返回 NOT_APPLICABLE）

    Returns:
        EvidenceItem with R-type evidence.
    """

    # ── 前置检查：特征重要性数据是否可用 ──
    if not feature_importance:
        return EvidenceItem(
            evidence_id=str(uuid.uuid4()),
            evidence_type=EvidenceType.R,
            method_code="counterfactual_repair_check",
            executor_version="V1",
            normalized_score=0.0,
            direction=EvidenceDirection.NEUTRAL,
            applicable=False,
            availability_status=AvailabilityStatus.DATA_NOT_AVAILABLE,
            confidence_level=ConfidenceLevel.LOW,
            evidence_detail_json={
                "message": "特征重要性数据不可用，无法评估修复潜力",
                "has_feature_importance": False,
            },
        )

    # ── 1. 找到漂移特征（PSI > 0.1）──
    drifted = [
        d for d in drift_rows
        if d.get("psi") is not None and d["psi"] > 0.1
    ]

    if not drifted:
        return EvidenceItem(
            evidence_id=str(uuid.uuid4()),
            evidence_type=EvidenceType.R,
            method_code="counterfactual_repair_check",
            executor_version="V1",
            normalized_score=0.05,
            direction=EvidenceDirection.AGAINST,
            applicable=True,
            confidence_level=ConfidenceLevel.HIGH,
            evidence_detail_json={
                "message": "无特征发生显著漂移，修复无对象可操作",
                "drifted_count": 0,
                "repair_potential": 0.0,
                "threshold": 0.1,
            },
        )

    # ── 2. 计算修复潜力 ──
    # repaired_impact = Σ(psi_i × importance_i) only for drifted features
    # total_importance = Σ(importance_i) for ALL features (including non-drifted)
    # repair_potential = repaired_impact / total_importance

    repaired_impact = 0.0
    total_importance = 0.0
    matched_drifted: list[dict] = []

    for d in drifted:
        fn = d["feature_name"]
        imp = feature_importance.get(fn, 0.0)
        psi = d["psi"]

        if imp > 0:
            repaired_impact += psi * imp
            matched_drifted.append({
                "feature_name": fn,
                "psi": round(psi, 4),
                "importance": round(imp, 4),
                "contribution": round(psi * imp, 6),
            })

    total_importance = sum(feature_importance.values())

    if total_importance == 0:
        return EvidenceItem(
            evidence_id=str(uuid.uuid4()),
            evidence_type=EvidenceType.R,
            method_code="counterfactual_repair_check",
            executor_version="V1",
            normalized_score=0.5,
            direction=EvidenceDirection.NEUTRAL,
            applicable=True,
            confidence_level=ConfidenceLevel.LOW,
            evidence_detail_json={
                "message": "特征重要性总和为 0，无法计算修复潜力",
                "repair_potential": 0.0,
            },
        )

    repair_potential = repaired_impact / total_importance

    # ── 3. 归一化并判定方向 ──
    # repair_potential > 0.3 → SUPPORT
    # repair_potential < 0.1 → AGAINST
    # 中间 → NEUTRAL

    normalized = min(repair_potential / 0.5, 1.0)  # 0.5 是归一化分母

    if repair_potential > 0.3:
        direction = EvidenceDirection.SUPPORT
        confidence = ConfidenceLevel.HIGH
        message = (
            f"修复潜力 {repair_potential:.2%}：{len(matched_drifted)} 个漂移特征"
            f"占模型总重要性的 {repaired_impact/total_importance:.1%}，"
            f"修复后性能大概率恢复"
        )
    elif repair_potential < 0.1:
        direction = EvidenceDirection.AGAINST
        confidence = ConfidenceLevel.HIGH
        message = (
            f"修复潜力仅 {repair_potential:.2%}：漂移特征对模型不重要，"
            f"仅靠修复特征无法恢复性能，需排查其他维度"
        )
    else:
        direction = EvidenceDirection.NEUTRAL
        confidence = ConfidenceLevel.MEDIUM
        message = (
            f"修复潜力 {repair_potential:.2%}：有一定修复价值但不够显著，"
            f"建议结合其他证据综合判断"
        )

    # ── 4. 按贡献度排序，取 Top-5 ──
    matched_drifted.sort(key=lambda x: x["contribution"], reverse=True)

    return EvidenceItem(
        evidence_id=str(uuid.uuid4()),
        evidence_type=EvidenceType.R,
        method_code="counterfactual_repair_check",
        executor_version="V1",
        normalized_score=round(normalized, 4),
        direction=direction,
        applicable=True,
        confidence_level=confidence,
        evidence_detail_json={
            "message": message,
            "repair_potential": round(repair_potential, 4),
            "repaired_impact": round(repaired_impact, 6),
            "total_importance": round(total_importance, 6),
            "drifted_count": len(drifted),
            "matched_count": len(matched_drifted),
            "importance_source": "feature_importance.csv",
            "top_contributors": matched_drifted[:5],
            "alert_metric": alert_metric_code,
        },
    )
