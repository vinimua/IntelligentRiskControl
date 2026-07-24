"""drift_group_regression 验证器 — C 类型证据：特征集体漂移与指标退化的关联强度。

核心逻辑：
  如果多个窗口的漂移总量与 AUC/KS 退化呈正相关（漂移越大退化越大），
  则支持"特征漂移是导致性能下降的原因"这一假设。

  采用 Spearman 秩相关系数：比较各窗口的 aggregate_psi 排名 vs auc_delta 排名。
  秩相关 > 0.7 → 强正相关 → SUPPORT
  秩相关 < -0.3 → 负相关 → AGAINST

输入:
  - drift_rows: 当前窗口的逐特征漂移数据
  - multi_window_drift: dict[str, list[dict]] — 各窗口的完整漂移数据
  - metrics: list[dict] — 该次运行的指标数据（含 baseline_value, current_value, delta, metric_code）
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


async def drift_group_regression(
    drift_rows: list[dict],
    alert_metric_code: str,
    multi_window_drift: dict[str, list[dict]] | None = None,
    metrics: list[dict] | None = None,
    **_kwargs,
) -> EvidenceItem:
    """C 类型验证器：漂移-退化关联分析。

    Args:
        drift_rows: 当前窗口漂移数据（保留兼容性，本验证器实际用 multi_window_drift）
        alert_metric_code: 告警指标代码
        multi_window_drift: 各窗口 → 漂移行列表
        metrics: 指标数据列表

    Returns:
        EvidenceItem with C-type evidence.
    """

    # ── 前置检查：多窗口数据是否可用 ──
    if not multi_window_drift or len(multi_window_drift) < 2:
        return EvidenceItem(
            evidence_id=str(uuid.uuid4()),
            evidence_type=EvidenceType.C,
            method_code="drift_group_regression",
            executor_version="V1",
            normalized_score=0.5,
            direction=EvidenceDirection.NEUTRAL,
            applicable=True,
            availability_status=AvailabilityStatus.DATA_NOT_AVAILABLE
            if not multi_window_drift
            else AvailabilityStatus.SAMPLE_TOO_SMALL,
            confidence_level=ConfidenceLevel.LOW,
            evidence_detail_json={
                "message": (
                    "多窗口漂移数据不足（需要至少 2 个窗口），无法执行关联分析"
                ),
                "window_count": len(multi_window_drift) if multi_window_drift else 0,
            },
        )

    # ── 1. 按窗口聚合漂移指数 ──
    # aggregate_psi = mean(psi) across all features in that window
    window_psi: dict[str, float] = {}
    for wid, rows in multi_window_drift.items():
        psi_values = [r["psi"] for r in rows if r.get("psi") is not None]
        if psi_values:
            window_psi[wid] = sum(psi_values) / len(psi_values)

    if len(window_psi) < 2:
        return EvidenceItem(
            evidence_id=str(uuid.uuid4()),
            evidence_type=EvidenceType.C,
            method_code="drift_group_regression",
            executor_version="V1",
            normalized_score=0.5,
            direction=EvidenceDirection.NEUTRAL,
            applicable=True,
            confidence_level=ConfidenceLevel.LOW,
            evidence_detail_json={
                "message": "计算后的有效窗口数不足 2",
                "valid_window_count": len(window_psi),
            },
        )

    # ── 2. 从 metrics 中提取每个窗口的 AUC/KS delta ──
    # 找 AUC 或 KS（优先匹配 alert_metric_code）
    target_code = alert_metric_code.replace("_DROP", "").replace("_DELTA", "")
    # e.g. HIGH_FEATURE_PSI → 无法直接对应，回退到 AUC + KS

    window_delta: dict[str, float] = {}
    if metrics:
        for m in metrics:
            mc = m.get("metric_code", "")
            obj = m.get("object_code", "")
            wid = m.get("window_id", m.get("current_window_id", ""))
            delta = m.get("delta")
            if wid and delta is not None:
                if mc in ("AUC", "PR_AUC", "KS", "BRIER", "BAD_RATE") or mc == target_code:
                    # 一个窗口可能有多个指标，保留绝对值最大的 delta 作为退化指数
                    abs_delta = abs(float(delta))
                    if wid not in window_delta or abs_delta > abs(window_delta[wid]):
                        window_delta[wid] = float(delta)

    # ── 3. 找 window_psi 和 window_delta 的交集 ──
    common_windows = sorted(set(window_psi.keys()) & set(window_delta.keys()))
    if len(common_windows) < 2:
        return EvidenceItem(
            evidence_id=str(uuid.uuid4()),
            evidence_type=EvidenceType.C,
            method_code="drift_group_regression",
            executor_version="V1",
            normalized_score=0.5,
            direction=EvidenceDirection.NEUTRAL,
            applicable=True,
            confidence_level=ConfidenceLevel.MEDIUM,
            evidence_detail_json={
                "message": (
                    f"漂移数据与指标数据的窗口交集不足（{len(common_windows)} 个），"
                    f"无法计算关联"
                ),
                "psi_windows": list(window_psi.keys()),
                "delta_windows": list(window_delta.keys()),
                "common_windows": common_windows,
            },
        )

    # ── 4. 计算 Spearman 秩相关系数 ──
    # 对于 < 4 个窗口的样本，用简化版：方向一致性检查
    psi_vals = [window_psi[w] for w in common_windows]
    delta_vals = [window_delta[w] for w in common_windows]

    spearman = _spearman_rank_correlation(psi_vals, delta_vals)

    # ── 5. 判定 ──
    if spearman is None:
        return EvidenceItem(
            evidence_id=str(uuid.uuid4()),
            evidence_type=EvidenceType.C,
            method_code="drift_group_regression",
            executor_version="V1",
            normalized_score=0.5,
            direction=EvidenceDirection.NEUTRAL,
            applicable=True,
            confidence_level=ConfidenceLevel.LOW,
            evidence_detail_json={
                "message": "窗口数据方差为 0（所有窗口 PSI 或 delta 相同），无法计算秩相关",
                "window_count": len(common_windows),
            },
        )

    abs_corr = abs(spearman)
    normalized = min(abs_corr / 0.7, 1.0)  # 归一化到 0–1

    if spearman > 0.7:
        direction = EvidenceDirection.SUPPORT
        confidence = ConfidenceLevel.HIGH
        message = (
            f"Spearman ρ = {spearman:.3f}：特征集体漂移与指标退化呈强正相关，"
            f"漂移越大 → 退化越严重，支持漂移为根因"
        )
    elif spearman > 0.3:
        direction = EvidenceDirection.SUPPORT
        confidence = ConfidenceLevel.MEDIUM
        message = (
            f"Spearman ρ = {spearman:.3f}：存在中等正相关，"
            f"漂移与退化方向基本一致"
        )
    elif spearman < -0.3:
        direction = EvidenceDirection.AGAINST
        confidence = ConfidenceLevel.MEDIUM
        message = (
            f"Spearman ρ = {spearman:.3f}：负相关，漂移越大反而指标越好，"
            f"漂移可能不是退化原因"
        )
    else:
        direction = EvidenceDirection.NEUTRAL
        confidence = ConfidenceLevel.LOW
        message = (
            f"Spearman ρ = {spearman:.3f}：相关性弱，无法确定漂移与退化的关联"
        )

    return EvidenceItem(
        evidence_id=str(uuid.uuid4()),
        evidence_type=EvidenceType.C,
        method_code="drift_group_regression",
        executor_version="V1",
        normalized_score=round(normalized, 4),
        direction=direction,
        applicable=True,
        confidence_level=confidence,
        evidence_detail_json={
            "message": message,
            "spearman_rho": round(spearman, 4),
            "window_count": len(common_windows),
            "common_windows": common_windows,
            "per_window_psi": {w: round(window_psi[w], 4) for w in common_windows},
            "per_window_delta": {w: round(window_delta.get(w, 0), 4) for w in common_windows},
            "psi_values": [round(v, 4) for v in psi_vals],
            "delta_values": [round(v, 4) for v in delta_vals],
            "alert_metric": alert_metric_code,
        },
    )


def _spearman_rank_correlation(
    x: list[float], y: list[float]
) -> float | None:
    """计算 Spearman 秩相关系数。

    将两个序列转换为排名，计算排名的 Pearson 相关系数。
    若任一序列方差为 0，返回 None。
    """
    n = len(x)
    if n < 2:
        return None

    # 计算排名（平均值处理平局）
    def _rank(vals: list[float]) -> list[float]:
        sorted_pairs = sorted(enumerate(vals), key=lambda p: p[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and sorted_pairs[j][1] == sorted_pairs[i][1]:
                j += 1
            avg_rank = (i + j - 1) / 2.0 + 1  # 1-indexed average
            for k in range(i, j):
                ranks[sorted_pairs[k][0]] = avg_rank
            i = j
        return ranks

    rx = _rank(x)
    ry = _rank(y)

    # Pearson on ranks
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    std_x = sum((rx[i] - mean_rx) ** 2 for i in range(n)) ** 0.5
    std_y = sum((ry[i] - mean_ry) ** 2 for i in range(n)) ** 0.5

    if std_x == 0 or std_y == 0:
        return None

    return cov / (std_x * std_y)
