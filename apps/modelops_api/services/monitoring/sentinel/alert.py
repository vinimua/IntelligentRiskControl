"""确定性告警规则引擎 — 基于交接包 alert_engine.py。

从 Sentinel 推理结果构建完整的告警事件：
- 按 anomaly_probability 超出阈值的幅度分级严重度
- 生成 AlertEvent 记录（含 alert_id / trigger_type / diagnosis_required）
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd


# ═══════════════════════════════════════════════════════════════
# 告警事件
# ═══════════════════════════════════════════════════════════════


@dataclass
class AlertEvent:
    """单次 Sentinel 告警事件。"""

    trace_id: str
    alert_id: str | None
    model_id: str
    model_version: str
    baseline_id: str
    baseline_version: str
    monitor_window_id: str
    status: str  # "NORMAL" | "ALERT"
    severity: str  # "NONE" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    anomaly_probability: float
    alert_threshold: float
    sentinel_version: str
    top_signals: list[str] = field(default_factory=list)
    top_signal_details: list[dict[str, Any]] = field(default_factory=list)
    diagnosis_required: bool = False
    trigger_type: str = "SENTINEL_FUSION"
    trigger_source: str = "WP08_SENTINEL"
    created_at: str = ""


# ═══════════════════════════════════════════════════════════════
# 严重度分级
# ═══════════════════════════════════════════════════════════════


def _severity(probability: float, threshold: float) -> str:
    """按 anomaly_probability 超出阈值的幅度分级。

    超出幅度 = (probability - threshold) / (1 - threshold)

    margin ≥ 80%  → CRITICAL
    margin ≥ 50%  → HIGH
    margin ≥ 20%  → MEDIUM
    超出但 < 20%  → LOW
    未超出        → NONE
    """
    if probability < threshold:
        return "NONE"
    margin = (probability - threshold) / max(1e-9, 1.0 - threshold)
    if margin >= 0.8:
        return "CRITICAL"
    if margin >= 0.5:
        return "HIGH"
    if margin >= 0.2:
        return "MEDIUM"
    return "LOW"


# ═══════════════════════════════════════════════════════════════
# 告警构建
# ═══════════════════════════════════════════════════════════════


def build_alerts(results: pd.DataFrame) -> list[AlertEvent]:
    """从 Sentinel 推理结果构建告警事件列表。

    Args:
        results: infer_sentinel() 的输出 DataFrame。
            必须包含 anomaly_probability, alert_threshold, trace_id,
            monitor_window_id, top_signals, top_signal_details 等列。

    Returns:
        AlertEvent 列表，按时间顺序排列。
        - status="ALERT" + severity=CRITICAL/HIGH/MEDIUM/LOW → 异常窗口
        - status="NORMAL" + severity=NONE → 正常窗口
        - alert_id 只在 status="ALERT" 时生成
        - diagnosis_required 只在 status="ALERT" 时为 True
    """
    events: list[AlertEvent] = []

    for _, row in results.iterrows():
        prob = float(row["anomaly_probability"])
        threshold = float(row["alert_threshold"])
        is_alert = prob >= threshold

        alert_id = None
        if is_alert:
            alert_id = (
                "ALT_"
                + hashlib.sha256(
                    f"{row['trace_id']}|{row['monitor_window_id']}".encode()
                ).hexdigest()[:16]
            )

        events.append(AlertEvent(
            trace_id=str(row.get("trace_id", "")),
            alert_id=alert_id,
            model_id=str(row.get("model_id", "")),
            model_version=str(row.get("model_version", "")),
            baseline_id=str(row.get("baseline_id", "")),
            baseline_version=str(row.get("baseline_version", "")),
            monitor_window_id=str(row.get("monitor_window_id", "")),
            status="ALERT" if is_alert else "NORMAL",
            severity=_severity(prob, threshold),
            anomaly_probability=prob,
            alert_threshold=threshold,
            sentinel_version=str(row.get("sentinel_version", "")),
            top_signals=list(row.get("top_signals", [])),
            top_signal_details=list(row.get("top_signal_details", [])),
            diagnosis_required=is_alert,
            trigger_type="SENTINEL_FUSION",
            trigger_source="WP08_SENTINEL",
            created_at=datetime.now(timezone.utc).isoformat(),
        ))

    return events


# ═══════════════════════════════════════════════════════════════
# 综合告警摘要
# ═══════════════════════════════════════════════════════════════


def alert_summary(events: list[AlertEvent]) -> dict:
    """汇总告警事件的统计信息。

    Returns:
        {
            "total_windows": int,
            "normal_count": int,
            "alert_count": int,
            "severity_counts": {"NONE": n, "LOW": n, ...},
            "highest_severity": str,
            "alert_model_ids": list[str],
        }
    """
    sevs = [e.severity for e in events]
    sev_order = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

    severity_counts = {s: sevs.count(s) for s in sev_order}
    alert_events = [e for e in events if e.status == "ALERT"]

    return {
        "total_windows": len(events),
        "normal_count": len(events) - len(alert_events),
        "alert_count": len(alert_events),
        "severity_counts": severity_counts,
        "highest_severity": max(sevs, key=lambda s: sev_order.get(s, 0)),
        "alert_model_ids": sorted(set(e.model_id for e in alert_events)),
    }
