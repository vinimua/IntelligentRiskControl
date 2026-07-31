from apps.modelops_api.routers.dashboard import _dashboard_window_key


def test_dashboard_maps_formal_horizons_to_display_slots() -> None:
    assert _dashboard_window_key({"window_days": 7, "window_id": "7D_x"}) == "W2"
    assert _dashboard_window_key({"window_days": 30, "window_id": "30D_x"}) == "W3"
    assert _dashboard_window_key({"window_id": "W1"}) == "W1"
