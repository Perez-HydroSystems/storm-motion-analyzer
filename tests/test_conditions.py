"""Conditions of a tracked storm (stormcatalog_analyzer.conditions)."""
import numpy as np

from stormcatalog_analyzer.conditions import condition_flags, hourly_conditions, summarize_conditions, track_conditions

LAT = 35 + np.arange(300) / 30.0
LON = -100 + np.arange(400) / 30.0


def _blob(cy, cx, r=12, amp=5.0, shape=(300, 400)):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return np.where((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r, amp, 0.0)


def test_compact_storm_has_no_condition():
    rain = np.array([_blob(150, 100 + 6 * t) for t in range(10)])
    c = track_conditions(rain, rain > 0, LAT, LON)
    assert c["track_condition"] == "none" and not any(c[k] for k in c if k.startswith("flag_"))
    assert c["touch_hours"] == 0 and c["far_hours"] == 0 and c["pieces_median"] == 1 and c["big_share_median"] == 1.0
    assert c["other_pull_max_km"] == 0.0 and abs(c["step_max_km"] - 6 / 30 * 111.32 * np.cos(np.radians(LAT[150]))) < 3


def test_two_distant_pieces():
    rain = np.array([_blob(150, 100 + 3 * t) + _blob(150, 260 + 3 * t, amp=3.0) for t in range(8)])
    c = track_conditions(rain, rain > 0, LAT, LON)
    assert c["flag_two_pieces"] and c["track_condition"] == "two_pieces" and c["far_hours"] == 8
    assert 440 < c["second_km_max"] < 520                       # 160 cells of 1/30 degree at 35-45 N
    assert not c["flag_window"] and not c["flag_scattered"]      # the larger piece holds 62% of the rain


def test_absorbed_rain_area_moves_the_centroid():
    rain = np.array([_blob(150, 100) + _blob(150, 160, r=20, amp=6.0) for _ in range(6)])
    mask = np.array([_blob(150, 100) > 0] * 3 + [rain[0] > 0] * 3)   # the second area joins the core at hour 3
    h = hourly_conditions(rain, mask, LAT, LON)
    assert h.other_pull_km.iloc[3] > 50 and h.other_share.iloc[3] > 0.5 and np.nan_to_num(h.other_pull_km.iloc[[1, 2, 4, 5]]).max() == 0
    c = track_conditions(rain, mask, LAT, LON)
    assert c["flag_absorbed"] and c["other_pull_max_km"] > 50


def test_window_edge():
    rain = np.array([_blob(150, 360 + 8 * t, r=30) for t in range(8)])       # the storm leaves through the east edge
    h = hourly_conditions(rain, rain > 0, LAT, LON)
    s = summarize_conditions(h, dict(W=True, E=True, S=True, N=True))
    assert s["touch_hours"] >= 5 and s["touch_margin_hours"] == s["touch_hours"] and s["span_hours"] == 0
    assert condition_flags(s)["flag_window"]
    data_edge = summarize_conditions(h, dict(W=True, E=False, S=True, N=True))   # the east edge is the data extent
    assert data_edge["touch_margin_hours"] == 0 and data_edge["touch_hours"] == s["touch_hours"]
    band = np.zeros((4, 300, 400))
    band[:, 140:160, :] = 2.0                                               # a band across the whole window
    assert track_conditions(band, band > 0, LAT, LON)["span_hours"] == 4
