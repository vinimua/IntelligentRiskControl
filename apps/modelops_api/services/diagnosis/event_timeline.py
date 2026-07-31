"""Chronological diagnosis-event and evidence-window helpers."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any


def alert_event_time(alert: dict[str, Any]) -> datetime:
    """Return business event time, never the database insertion timestamp first."""

    detail = alert.get("alert_detail") or alert.get("metric_detail") or {}
    raw = detail.get("window_end") or detail.get("event_time")
    if raw:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=None)
    created_at = alert.get("created_at")
    if isinstance(created_at, datetime):
        return created_at.replace(tzinfo=None)
    if created_at:
        return datetime.fromisoformat(str(created_at).replace("Z", "+00:00")).replace(
            tzinfo=None
        )
    raise ValueError("alert has neither window_end nor created_at")


def alerts_at_first_event_time(alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select all metrics raised at the earliest unprocessed business time."""

    if not alerts:
        return []
    first_time = min(alert_event_time(alert) for alert in alerts)
    return sorted(
        [alert for alert in alerts if alert_event_time(alert) == first_time],
        key=lambda alert: (str(alert.get("metric_code", "")), str(alert.get("alert_id", ""))),
    )


def four_non_overlapping_windows(
    event_end: date | datetime,
    window_days: int = 7,
) -> list[dict[str, Any]]:
    """Build W-3..W0 using only information available at alert time.

    Window IDs retain the project's existing convention: start is ``end-window_days``.
    Adjacent selected windows therefore move back by ``window_days + 1`` calendar days
    so their closed intervals do not overlap.
    """

    end = event_end.date() if isinstance(event_end, datetime) else event_end
    span = timedelta(days=window_days)
    step = timedelta(days=window_days + 1)
    result: list[dict[str, Any]] = []
    for relative_index in (-3, -2, -1, 0):
        selected_end = end - step * abs(relative_index)
        selected_start = selected_end - span
        result.append(
            {
                "relative_index": relative_index,
                "role": "W0" if relative_index == 0 else f"W{relative_index}",
                "window_id": (
                    f"{window_days}D_{selected_start:%Y%m%d}_{selected_end:%Y%m%d}"
                ),
                "window_start": selected_start,
                "window_end": selected_end,
            }
        )
    return result
