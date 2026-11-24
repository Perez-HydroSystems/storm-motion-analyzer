"""Stall, turn and meander (motion.linger): each rule on a constructed track."""
import numpy as np
import pytest
from pyproj import Geod

from stormcatalog_analyzer.motion.linger import classify_track, geometric_class, track_geometry

GEOD = Geod(ellps="WGS84")
AREA = 50_000.0                                   # km2: equivalent radius 126 km


def track(headings_deg, step_km, lon0=-95.0, lat0=40.0):
    """Positions after successive steps with the given compass headings (deg) and lengths (km)."""
    lon, lat = [lon0], [lat0]
    for h, s in zip(headings_deg, np.broadcast_to(step_km, len(headings_deg))):
        x, y, _ = GEOD.fwd(lon[-1], lat[-1], h, s * 1000.0)
        lon.append(x)
        lat.append(y)
    return np.array(lon), np.array(lat)


def test_coherent_storm_keeps_its_label():
    lon, lat = track([60] * 11, 50)
    out = classify_track(lon, lat, AREA, coherent=True)
    assert out["linger_class"] == "coherent" and out["turn_half_deg"] < 2
    assert out["net_req"] == pytest.approx(550 / np.sqrt(AREA / np.pi), rel=0.01)


def test_stall_stays_within_one_core_radius():
    rng = np.random.default_rng(0)
    lon, lat = track(rng.uniform(0, 360, 11), 8)              # jitter of a storm that stays in place
    out = classify_track(lon, lat, AREA, coherent=False)
    assert out["net_req"] < 1 and out["linger_class"] == "stall"


def test_slow_storm_is_a_stall_even_if_it_leaves_the_core():
    lon, lat = track([90] * 11, 5)                            # 55 km in 11 h = 1.4 m/s
    out = classify_track(lon, lat, 2_000.0, coherent=False)   # small core: net displacement > one radius
    assert out["net_req"] > 1 and out["linger_class"] == "stall"


def test_sharp_turn():
    lon, lat = track([90] * 6 + [350] * 5, 50)                # east, then north-northwest: a 100-degree turn
    out = classify_track(lon, lat, AREA, coherent=False)
    assert out["turn_half_deg"] == pytest.approx(100, abs=3) and out["linger_class"] == "turn"


def test_reversal_is_a_turn_not_a_stall():
    lon, lat = track([90] * 6 + [270] * 5, 60)                # out and back: ends near the start
    out = classify_track(lon, lat, AREA, coherent=False)
    assert out["net_req"] < 1 and out["turn_half_deg"] > 150 and out["linger_class"] == "turn"


def test_steady_rotation_is_caught_by_the_thirds():
    lon, lat = track(np.linspace(0, 200, 11), 50)             # heading rotates steadily clockwise
    turn_half, rot, _, _ = track_geometry(lon, lat)
    assert abs(rot) >= 90
    assert classify_track(lon, lat, AREA, coherent=False)["linger_class"] == "turn"


def test_meander_travels_without_a_clean_turn():
    lon, lat = track([40, 140] * 5 + [40], 45)                # zigzag about an eastward course
    out = classify_track(lon, lat, AREA, coherent=False)
    assert out["net_req"] > 1 and out["turn_half_deg"] < 90 and out["rot_thirds_deg"] == 0
    assert out["linger_class"] == "meander"


def test_parameters_and_given_net_values():
    lon, lat = track([90] * 6 + [20] * 5, 50)                 # a 70-degree turn
    assert geometric_class(lon, lat, AREA)["geometric_class"] == "meander"
    assert geometric_class(lon, lat, AREA, A=60.0)["geometric_class"] == "turn"
    net = GEOD.inv(lon[0], lat[0], lon[-1], lat[-1])[2] / 1e3
    a = geometric_class(lon, lat, AREA)
    b = geometric_class(lon, lat, AREA, net_length_km=net, net_speed_ms=net * 1e3 / (11 * 3600))
    assert a == b
