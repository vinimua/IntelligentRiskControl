from datetime import datetime

from apps.modelops_api.services.diagnosis.event_timeline import (
    alerts_at_first_event_time,
    four_non_overlapping_windows,
)


def test_first_event_uses_business_time_and_keeps_simultaneous_metrics():
    alerts = [
        {
            "alert_id": "dec",
            "metric_code": "KS",
            "alert_detail": {"window_end": "2025-12-28T00:00:00"},
        },
        {
            "alert_id": "nov-auc",
            "metric_code": "AUC",
            "alert_detail": {"window_end": "2025-11-20T00:00:00"},
        },
        {
            "alert_id": "nov-ks",
            "metric_code": "KS",
            "alert_detail": {"window_end": "2025-11-20T00:00:00"},
        },
    ]

    selected = alerts_at_first_event_time(alerts)

    assert [item["alert_id"] for item in selected] == ["nov-auc", "nov-ks"]


def test_four_windows_are_ordered_non_overlapping_and_have_no_future_data():
    windows = four_non_overlapping_windows(datetime(2025, 11, 20), 7)

    assert [window["role"] for window in windows] == ["W-3", "W-2", "W-1", "W0"]
    assert windows[-1]["window_end"].isoformat() == "2025-11-20"
    assert all(
        left["window_end"] < right["window_start"]
        for left, right in zip(windows, windows[1:])
    )
