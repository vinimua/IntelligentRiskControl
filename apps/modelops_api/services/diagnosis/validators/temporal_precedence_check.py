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

from datetime import date, datetime
import re
import uuid

from packages.models.diagnosis.evidence import EvidenceItem
from packages.models.common.enums import (
    AvailabilityStatus,
    ConfidenceLevel,
    EvidenceDirection,
    EvidenceType,
)
from .metric_binding import degradation_from_delta, resolve_alert_metric_code

# ── 窗口排序映射 ──
_WINDOW_ORDER = {"W1": 0, "W2": 1, "W3": 2, "W4": 3, "W5": 4, "W6": 5}
_DATED_WINDOW_PATTERN = re.compile(r"(?:^|_)(\d{8})_(\d{8})$")


def _parse_window_interval(window_id: str) -> tuple[date, date] | None:
    """Parse window IDs such as ``7D_20251225_20251231``."""

    matched = _DATED_WINDOW_PATTERN.search(window_id or "")
    if not matched:
        return None
    try:
        start = datetime.strptime(matched.group(1), "%Y%m%d").date()
        end = datetime.strptime(matched.group(2), "%Y%m%d").date()
    except ValueError:
        return None
    if start > end:
        return None
    return start, end


def _order_temporal_windows(
    window_ids: list[str],
) -> tuple[list[str] | None, str | None]:
    """Return a defensible chronological order or an inapplicability reason."""

    if all(w.upper() in _WINDOW_ORDER for w in window_ids):
        return sorted(window_ids, key=lambda w: _WINDOW_ORDER[w.upper()]), None

    intervals = {w: _parse_window_interval(w) for w in window_ids}
    if not all(intervals.values()):
        return None, "窗口编号无法解析为连续时间区间"

    ordered = sorted(
        window_ids,
        key=lambda w: (intervals[w][0], intervals[w][1]),  # type: ignore[index]
    )
    for index, left_id in enumerate(ordered):
        left = intervals[left_id]
        assert left is not None
        for right_id in ordered[index + 1:]:
            right = intervals[right_id]
            assert right is not None
            if left[0] <= right[1] and right[0] <= left[1]:
                return (
                    None,
                    f"窗口 {left_id} 与 {right_id} 时间重叠，"
                    "不能用于判断原因和结果的先后顺序",
                )
    return ordered, None


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
            executor_version="V2",
            normalized_score=None,
            direction=EvidenceDirection.NEUTRAL,
            applicable=False,
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
    # 严格匹配触发告警对应的唯一指标，不允许 AUC/KS 混用。
    target_code = resolve_alert_metric_code(alert_metric_code)
    if target_code is None:
        return EvidenceItem(
            evidence_id=str(uuid.uuid4()),
            evidence_type=EvidenceType.T,
            method_code="temporal_precedence_check",
            executor_version="V2",
            normalized_score=None,
            direction=EvidenceDirection.NEUTRAL,
            applicable=False,
            availability_status=AvailabilityStatus.NOT_APPLICABLE,
            confidence_level=ConfidenceLevel.LOW,
            evidence_detail_json={
                "message": (
                    f"告警 {alert_metric_code} 没有可唯一绑定的性能指标，"
                    "不能执行时序验证"
                ),
                "alert_metric": alert_metric_code,
            },
        )

    window_degradation: dict[str, float] = {}
    window_delta: dict[str, float] = {}
    if metrics:
        for m in metrics:
            mc = str(m.get("metric_code", "")).upper()
            wid = m.get("window_id", m.get("current_window_id", ""))
            delta = m.get("delta")
            if wid and delta is not None and mc == target_code:
                raw_delta = float(delta)
                degradation = degradation_from_delta(mc, raw_delta)
                if degradation is not None:
                    window_delta[wid] = raw_delta
                    window_degradation[wid] = degradation

    # ── 3. 找交集窗口 ──
    common_windows = list(set(window_psi_max.keys()) & set(window_degradation.keys()))
    if len(common_windows) < 2:
        return EvidenceItem(
            evidence_id=str(uuid.uuid4()),
            evidence_type=EvidenceType.T,
            method_code="temporal_precedence_check",
            executor_version="V2",
            normalized_score=None,
            direction=EvidenceDirection.NEUTRAL,
            applicable=False,
            availability_status=AvailabilityStatus.SAMPLE_TOO_SMALL,
            confidence_level=ConfidenceLevel.LOW,
            evidence_detail_json={
                "message": (
                    f"漂移与指标数据的窗口交集不足（{len(common_windows)} 个），"
                    f"无法判断时序"
                ),
                "psi_windows": sorted(window_psi_max.keys()),
                "degradation_windows": sorted(window_degradation.keys()),
                "target_metric_code": target_code,
            },
        )

    common_windows, invalid_reason = _order_temporal_windows(common_windows)
    if common_windows is None:
        return EvidenceItem(
            evidence_id=str(uuid.uuid4()),
            evidence_type=EvidenceType.T,
            method_code="temporal_precedence_check",
            executor_version="V2",
            normalized_score=None,
            direction=EvidenceDirection.NEUTRAL,
            applicable=False,
            availability_status=AvailabilityStatus.NOT_APPLICABLE,
            confidence_level=ConfidenceLevel.LOW,
            evidence_detail_json={
                "message": (
                    f"{invalid_reason}；本次时序证据不参与根因评分。"
                    "需要不同监测日期的连续、非重叠窗口"
                ),
                "windows": sorted(
                    set(window_psi_max.keys()) & set(window_degradation.keys())
                ),
                "target_metric_code": target_code,
                "alert_metric": alert_metric_code,
                "per_window_delta": {
                    w: round(window_delta[w], 4) for w in window_delta
                },
            },
        )

    # ── 4. 找"漂移峰值窗口"和"退化峰值窗口"──
    peak_psi_window = max(common_windows, key=lambda w: window_psi_max[w])
    peak_degradation_window = max(common_windows, key=lambda w: window_degradation[w])

    psi_peak_order = common_windows.index(peak_psi_window)
    deg_peak_order = common_windows.index(peak_degradation_window)

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
        executor_version="V2",
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
            "per_window_delta": {w: round(window_delta[w], 4) for w in common_windows},
            "target_metric_code": target_code,
            "alert_metric": alert_metric_code,
        },
    )
