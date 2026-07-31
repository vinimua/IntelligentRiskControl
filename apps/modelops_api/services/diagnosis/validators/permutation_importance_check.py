"""permutation_importance_check 验证器 — I 类型证据：漂移特征是否对模型预测有实质影响。

核心逻辑：
  特征漂移本身不一定是坏消息——如果漂移的是低重要性特征，
  对模型影响微乎其微。本验证器交叉引用"漂移"与"重要性"：
  - 如果漂移的特征恰好是高重要性特征（Top-25%）→ SUPPORT
  - 如果漂移的特征都是低重要性特征（Bottom-50%）→ AGAINST
  - 混合 → NEUTRAL

公式:
  weighted_psi_importance = Σ(psi_i × importance_i) for drifted features
  total_psi_importance = Σ(psi_j × importance_j) for ALL features
  importance_ratio = weighted_psi_importance / total_psi_importance

  importance_ratio > 0.5 → 漂移集中在高重要性特征 → SUPPORT
  importance_ratio < 0.1 → 漂移集中在低重要性特征 → AGAINST

输入:
  - drift_rows: 逐特征漂移数据
  - feature_importance: dict[feature_name, importance_score]
  - alert_metric_code: 告警指标代码

输出: EvidenceItem with direction and normalized_score.
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


async def permutation_importance_check(
    drift_rows: list[dict],
    alert_metric_code: str,
    feature_importance: dict[str, float] | None = None,
    **_kwargs,
) -> EvidenceItem:
    """I 类型验证器：漂移特征的重要性评估。

    如果特征重要性数据不可得，尝试从 KG 查询 Feature 节点属性作为降级方案。

    Args:
        drift_rows: 逐特征漂移数据
        alert_metric_code: 告警指标代码
        feature_importance: 特征名 → 重要性分数 映射

    Returns:
        EvidenceItem with I-type evidence.
    """

    # ── 前置检查 ──
    if not feature_importance:
        return EvidenceItem(
            evidence_id=str(uuid.uuid4()),
            evidence_type=EvidenceType.I,
            method_code="permutation_importance_check",
            executor_version="V1",
            normalized_score=0.0,
            direction=EvidenceDirection.NEUTRAL,
            applicable=False,
            availability_status=AvailabilityStatus.DATA_NOT_AVAILABLE,
            confidence_level=ConfidenceLevel.LOW,
            evidence_detail_json={
                "message": "特征重要性数据不可用，无法评估漂移特征的模型依赖度",
                "has_feature_importance": False,
            },
        )

    # ── 1. 构建全量特征的 PSI × Importance 加权分布 ──
    # 先将 feature_importance 映射到 drift 数据
    feature_data: dict[str, dict] = {}
    for d in drift_rows:
        fn = d.get("feature_name", "")
        psi = d.get("psi") or 0.0
        imp = feature_importance.get(fn, 0.0)
        feature_data[fn] = {
            "psi": psi,
            "importance": imp,
            "weighted_psi": psi * imp,
        }

    if not feature_data:
        return EvidenceItem(
            evidence_id=str(uuid.uuid4()),
            evidence_type=EvidenceType.I,
            method_code="permutation_importance_check",
            executor_version="V1",
            normalized_score=0.5,
            direction=EvidenceDirection.NEUTRAL,
            applicable=True,
            confidence_level=ConfidenceLevel.LOW,
            evidence_detail_json={"message": "无有效特征数据"},
        )

    # ── 2. 找到漂移特征（PSI > 0.1）──
    drifted_features = {
        fn: fd for fn, fd in feature_data.items() if fd["psi"] > 0.1
    }

    if not drifted_features:
        return EvidenceItem(
            evidence_id=str(uuid.uuid4()),
            evidence_type=EvidenceType.I,
            method_code="permutation_importance_check",
            executor_version="V1",
            normalized_score=0.05,
            direction=EvidenceDirection.AGAINST,
            applicable=True,
            confidence_level=ConfidenceLevel.HIGH,
            evidence_detail_json={
                "message": "无特征发生显著漂移，重要性评估无对象",
                "drifted_count": 0,
                "threshold": 0.1,
            },
        )

    # ── 3. 计算加权重要性比 ──
    total_weighted = sum(fd["weighted_psi"] for fd in feature_data.values())
    drifted_weighted = sum(fd["weighted_psi"] for fd in drifted_features.values())

    if total_weighted == 0:
        return EvidenceItem(
            evidence_id=str(uuid.uuid4()),
            evidence_type=EvidenceType.I,
            method_code="permutation_importance_check",
            executor_version="V1",
            normalized_score=0.5,
            direction=EvidenceDirection.NEUTRAL,
            applicable=True,
            confidence_level=ConfidenceLevel.LOW,
            evidence_detail_json={
                "message": "所有特征重要性为 0，无法评估",
                "total_weighted_psi": 0.0,
            },
        )

    importance_ratio = drifted_weighted / total_weighted

    # ── 4. 补充统计：漂移特征的重要性排名分布 ──
    # 按重要性排序所有特征
    all_ranked = sorted(
        feature_data.items(), key=lambda x: x[1]["importance"], reverse=True
    )
    total_count = len(all_ranked)
    top_quartile_threshold = max(1, total_count // 4)

    drifted_in_top_quartile = 0
    drifted_ranks: list[dict] = []
    for rank_idx, (fn, fd) in enumerate(all_ranked):
        if fn in drifted_features:
            rank_info = {
                "feature_name": fn,
                "importance_rank": rank_idx + 1,
                "importance": round(fd["importance"], 4),
                "psi": round(fd["psi"], 4),
                "in_top_quartile": rank_idx < top_quartile_threshold,
            }
            drifted_ranks.append(rank_info)
            if rank_idx < top_quartile_threshold:
                drifted_in_top_quartile += 1

    # ── 5. 判定 ──
    if importance_ratio > 0.5:
        direction = EvidenceDirection.SUPPORT
        confidence = ConfidenceLevel.HIGH
        normalized = min(importance_ratio / 0.7, 1.0)
        message = (
            f"漂移特征的重要性加权占比 {importance_ratio:.1%}："
            f"{len(drifted_features)} 个漂移特征中 {drifted_in_top_quartile} 个"
            f"位于 Top-25% 重要性。漂移确实影响模型预测。"
        )
    elif importance_ratio > 0.2:
        direction = EvidenceDirection.SUPPORT
        confidence = ConfidenceLevel.MEDIUM
        normalized = importance_ratio / 0.7
        message = (
            f"漂移特征的重要性加权占比 {importance_ratio:.1%}："
            f"有一定影响但不够突出，建议结合其他证据判断。"
        )
    elif importance_ratio < 0.05:
        direction = EvidenceDirection.AGAINST
        confidence = ConfidenceLevel.HIGH
        normalized = 0.05
        message = (
            f"漂移特征的重要性加权占比仅 {importance_ratio:.2%}："
            f"漂移集中在不影响模型预测的低重要性特征上，"
            f"特征漂移不太可能是性能退化的主因。"
        )
    else:
        direction = EvidenceDirection.NEUTRAL
        confidence = ConfidenceLevel.MEDIUM
        normalized = 0.40
        message = (
            f"漂移特征的重要性加权占比 {importance_ratio:.1%}："
            f"影响中等，需多维度证据交叉验证。"
        )

    return EvidenceItem(
        evidence_id=str(uuid.uuid4()),
        evidence_type=EvidenceType.I,
        method_code="permutation_importance_check",
        executor_version="V1",
        normalized_score=round(normalized, 4),
        direction=direction,
        applicable=True,
        confidence_level=confidence,
        evidence_detail_json={
            "message": message,
            "importance_ratio": round(importance_ratio, 4),
            "drifted_weighted_psi": round(drifted_weighted, 6),
            "total_weighted_psi": round(total_weighted, 6),
            "drifted_count": len(drifted_features),
            "total_count": total_count,
            "drifted_in_top_quartile": drifted_in_top_quartile,
            "top_quartile_threshold": top_quartile_threshold,
            "drifted_ranks": drifted_ranks[:10],  # Top-10 漂移特征排名
            "alert_metric": alert_metric_code,
        },
    )
