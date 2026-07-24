"""temporal_precedence_check 验证器 — T 类型证据：漂移是否先于性能退化发生。

核心逻辑（因果推断的时序条件）：
  原因必须发生在结果之前。如果特征漂移在早期窗口（W1）已经出现，
  而指标退化在后期窗口（W3）才显现，则满足时序优先 → SUPPORT。
  如果退化先于漂移 → AGAINST（违反因果方向）。
  如果同步出现 → NEUTRAL（无法确认方向）。

实现策略:
  1. 将窗口按时间排序（W1 < W3 < W6）
  2. 计算每个窗口的 PSI rank（漂移程度排名）和 AUC delta rank（退化程度排名）
  3. 比较漂移领先窗口 vs 退化领先窗口：
     - 若漂移最高峰所在的窗口早于退化最严重的窗口 → SUPPORT
     - 若漂移最高峰所在窗口晚于退化最严重窗口 → AGAINST
     - 若在同一窗口 → NEUTRAL

输入:
  - drift_rows: 当前窗口漂移数据（保留兼容性）
  - multi_window_drift: dict[str, list[dict]]
  - metrics: list[dict]
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

# ── 窗口排序映射 ──
_WINDOW_ORDER = {"W1": 0, "W2": 1, "W3": 2, "W4": 3, "W5": 4, "W6": 5}


def _window_sort_key(window_id: str) -> int:
    """按 W1 < W3 < W6 排序，未知窗口排最后。"""
    return _WINDOW_ORDER.get(window_id.upper(), 99)


async def temporal_precedence_check(
    drift_rows: list[dict],
    alert_metric_code: str,
    multi_window_drift: dict[str, list[dict]] | None = None,
    metrics: list[dict] | None = None,
    **_kwargs,
) -> EvidenceItem:
    """T 类型验证器：时序优先检查。

    Args:
        drift_rows: 当前窗口漂移数据
        alert_metric_code: 告警指标代码
        multi_window_drift: 各窗口 → 漂移行列表
        metrics: 指标数据列表

    Returns:
        EvidenceItem with T-type evidence.
    """

    # ── 前置检查 ──
    if not multi_window_drift or len(multi_window_drift) < 2:
        return EvidenceItem(
            evidence_id=str(uuid.uuid4()),
            evidence_type=EvidenceType.T,
            method_code="temporal_precedence_check",
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
                    "多窗口漂移数据不足（需要至少 2 个窗口），无法判断时序关系"
                ),
                "window_count": len(multi_window_drift) if multi_window_drift else 0,
            },
        )

    # ── 1. 按窗口聚合 PSI（取均值 + 最大值）──
    # 使用 max PSI 作为"漂移严重程度"指标（更能代表突变）
    window_psi_max: dict[str, float] = {}
    window_psi_mean: dict[str, float] = {}
    for wid, rows in multi_window_drift.items():
        psi_values = [r["psi"] for r in rows if r.get("psi") is not None]
        if psi_values:
            window_psi_max[wid] = max(psi_values)
            window_psi_mean[wid] = sum(psi_values) / len(psi_values)

    # ── 2. 从 metrics 中提取每个窗口的性能退化 ──
    # 优先匹配 alert_metric_code 相关的指标
    target_codes = _resolve_metric_codes(alert_metric_code)
    window_degradation: dict[str, float] = {}
    if metrics:
        for m in metrics:
            mc = m.get("metric_code", "")
            wid = m.get("window_id", m.get("current_window_id", ""))
            delta = m.get("delta")
            if wid and delta is not None and mc in target_codes:
                # 退化 = 负的 delta（delta < 0 表示性能下降）
                degradation = -float(delta)
                if wid not in window_degradation or degradation > window_degradation[wid]:
                    window_degradation[wid] = degradation

    # ── 3. 找交集窗口 ──
    common_windows = sorted(
        set(window_psi_max.keys()) & set(window_degradation.keys()),
        key=_window_sort_key,
    )
    if len(common_windows) < 2:
        return EvidenceItem(
            evidence_id=str(uuid.uuid4()),
            evidence_type=EvidenceType.T,
            method_code="temporal_precedence_check",
            executor_version="V1",
            normalized_score=0.5,
            direction=EvidenceDirection.NEUTRAL,
            applicable=True,
            confidence_level=ConfidenceLevel.LOW,
            evidence_detail_json={
                "message": (
                    f"漂移与指标数据的窗口交集不足（{len(common_windows)} 个），"
                    f"无法判断时序"
                ),
                "psi_windows": sorted(window_psi_max.keys(), key=_window_sort_key),
                "degradation_windows": sorted(window_degradation.keys(), key=_window_sort_key),
            },
        )

    # ── 4. 找"漂移峰值窗口"和"退化峰值窗口"──
    peak_psi_window = max(common_windows, key=lambda w: window_psi_max[w])
    peak_degradation_window = max(common_windows, key=lambda w: window_degradation[w])

    psi_peak_order = _window_sort_key(peak_psi_window)
    deg_peak_order = _window_sort_key(peak_degradation_window)

    # ── 5. 同时检查：早期窗口的 PSI 是否已经偏高 ──
    # 补充逻辑：即使峰值在同一窗口，如果最早窗口已有显著漂移（PSI > 0.15）
    # 而指标当时正常，也说明漂移领先
    earliest_window = common_windows[0]
    early_psi = window_psi_max.get(earliest_window, 0)
    early_degradation = window_degradation.get(earliest_window, 0)
    drift_was_early = early_psi > 0.15 and early_degradation < 0.01

    # ── 6. 判定 ──
    if psi_peak_order < deg_peak_order or drift_was_early:
        # 漂移峰值出现在退化峰值之前 → 满足时序
        direction = EvidenceDirection.SUPPORT
        confidence = ConfidenceLevel.HIGH if psi_peak_order < deg_peak_order else ConfidenceLevel.MEDIUM
        gap = deg_peak_order - psi_peak_order if psi_peak_order < deg_peak_order else "early_drift_detected"
        normalized = 0.85 if psi_peak_order < deg_peak_order else 0.60
        message = (
            f"漂移峰值窗口 {peak_psi_window}（PSI_max={window_psi_max[peak_psi_window]:.3f}）"
            f"早于退化峰值窗口 {peak_degradation_window}"
            f"（degradation={window_degradation[peak_degradation_window]:.4f}），"
            f"满足因果时序条件"
        )
    elif psi_peak_order > deg_peak_order:
        # 退化先于漂移 → 违反因果方向
        direction = EvidenceDirection.AGAINST
        confidence = ConfidenceLevel.HIGH
        gap = psi_peak_order - deg_peak_order
        normalized = 0.15
        message = (
            f"退化峰值窗口 {peak_degradation_window} 早于漂移峰值窗口 {peak_psi_window}，"
            f"违反因果时序（结果先于原因），漂移不是退化的原因"
        )
    else:
        # 同一窗口
        direction = EvidenceDirection.NEUTRAL
        confidence = ConfidenceLevel.MEDIUM
        gap = 0
        normalized = 0.45
        message = (
            f"漂移峰值和退化峰值同窗口 {peak_psi_window}，"
            f"无法从时序上判断因果方向，需要其他证据配合"
        )

    return EvidenceItem(
        evidence_id=str(uuid.uuid4()),
        evidence_type=EvidenceType.T,
        method_code="temporal_precedence_check",
        executor_version="V1",
        normalized_score=round(normalized, 4),
        direction=direction,
        applicable=True,
        confidence_level=confidence,
        evidence_detail_json={
            "message": message,
            "peak_psi_window": peak_psi_window,
            "peak_psi_value": round(window_psi_max[peak_psi_window], 4),
            "peak_degradation_window": peak_degradation_window,
            "peak_degradation_value": round(window_degradation[peak_degradation_window], 4),
            "psi_peak_order": psi_peak_order,
            "deg_peak_order": deg_peak_order,
            "window_gap": gap,
            "early_psi": round(early_psi, 4),
            "early_degradation": round(early_degradation, 4),
            "drift_was_early_flag": drift_was_early,
            "common_windows": common_windows,
            "per_window_psi_max": {w: round(window_psi_max[w], 4) for w in common_windows},
            "per_window_degradation": {w: round(window_degradation[w], 4) for w in common_windows},
            "alert_metric": alert_metric_code,
        },
    )


def _resolve_metric_codes(alert_code: str) -> list[str]:
    """从告警代码推断相关的指标代码。"""
    mapping = {
        "AUC_DROP": ["AUC", "PR_AUC"],
        "AUC_DROP_P50": ["AUC", "PR_AUC"],
        "KS_DROP": ["KS"],
        "KS_DROP_P50": ["KS"],
        "PR_AUC_DROP_P50": ["PR_AUC", "AUC"],
        "HIGH_FEATURE_PSI": ["AUC", "KS", "PR_AUC", "BRIER"],
        "SCORE_PSI_DELTA": ["SCORE_PSI", "AUC", "KS"],
        "BAD_RATE_DELTA": ["BAD_RATE", "AUC"],
        "BH_DELTA": ["BH", "AUC"],
        "HIGH_MISSING_RATE": ["AUC", "KS", "PR_AUC"],
        "HIGH_OUTLIER_RATE": ["AUC", "KS"],
        "LABEL_SHIFT": ["BAD_RATE", "AUC"],
    }
    return mapping.get(alert_code, ["AUC", "KS", "PR_AUC", "BAD_RATE"])
