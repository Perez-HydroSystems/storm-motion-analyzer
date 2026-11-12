"""Unit checks of the V6 analysis window and the directional-coherence test."""
import numpy as np
import pytest

from stormcatalog_analyzer.motion.direction import coherence_test, select_analysis_window
from stormcatalog_analyzer.stats.circular import weighted_rayleigh_test


def track(bearings_deg, step_deg=0.08, lon0=-94.0, lat0=42.0):
    """lon/lat track with one hourly step per heading (math convention: 0 = east, CCW)."""
    lon, lat = [lon0], [lat0]
    for b in np.radians(bearings_deg):
        lon.append(lon[-1] + step_deg * np.cos(b) / np.cos(np.radians(lat[-1])))
        lat.append(lat[-1] + step_deg * np.sin(b))
    return np.array(lon), np.array(lat)


# ---------------------------------------------------------------- analysis window
def test_short_track_kept_in_full():
    assert select_analysis_window(np.ones(9), max_positions=12) == (0, 9)


def test_long_track_cut_to_wettest_12_end_exclusive():
    v = np.zeros(20); v[5:17] = 1.0; v[16] = 5.0
    start, end = select_analysis_window(v, max_positions=12)
    assert end - start == 12
    assert (start, end) == (5, 17)
    assert v[start:end].sum() == pytest.approx(v.max() + 11.0)


def test_ties_take_first_window():
    assert select_analysis_window(np.ones(15), max_positions=12) == (0, 12)


# ---------------------------------------------------------------- coherence test
def test_straight_track_is_coherent():
    lon, lat = track([10.0] * 11)
    out = coherence_test(lon, lat, seed=1)
    assert out["tested"] and out["coherent"]
    assert out["r_weighted"] == pytest.approx(1.0, abs=1e-6)
    assert out["n_segments"] == 11


def test_u_turn_is_not_coherent():
    lon, lat = track([0.0] * 6 + [180.0] * 5)
    out = coherence_test(lon, lat, seed=1)
    assert out["tested"] and not out["coherent"]


def test_too_few_segments_not_tested():
    lon, lat = track([0.0, 0.0])
    out = coherence_test(lon, lat, seed=1, min_segments=3)
    assert not out["tested"] and not out["coherent"]


def test_stationary_positions_are_dropped():
    lon, lat = track([30.0] * 6)
    lon = np.insert(lon, 3, lon[3]); lat = np.insert(lat, 3, lat[3])   # repeated centroid -> zero-length segment
    assert coherence_test(lon, lat, seed=1)["n_segments"] == 6


def test_seed_reproducible():
    lon, lat = track([0, 40, -30, 70, 10, -50, 20, 60])
    a = coherence_test(lon, lat, seed=[20261103, 7], refine_n_sim=0)
    b = coherence_test(lon, lat, seed=[20261103, 7], refine_n_sim=0)
    assert a["p_weighted"] == b["p_weighted"]


def test_weighted_rayleigh_uniform_null_calibrated():
    """Under the null (uniform directions), rejections at alpha=0.05 stay near 5%."""
    rng = np.random.default_rng(0)
    w = rng.uniform(5e3, 4e4, size=11)
    rej = [weighted_rayleigh_test(rng.uniform(0, 2 * np.pi, 11), w, n_sim=2000, seed=i)[1] < 0.05
           for i in range(400)]
    assert 0.025 < np.mean(rej) < 0.08
