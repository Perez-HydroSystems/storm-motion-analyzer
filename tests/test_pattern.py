"""Rain-pattern velocity (motion.pattern): sub-cell accuracy of the storm-masked correlation, immunity to rain outside
the storm, the storm-level velocity and the propagation, and the option of the enlarged-window analysis."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

from stormcatalog_analyzer import enlarged as E
from stormcatalog_analyzer.motion.pattern import masked_shift, storm_pattern

ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT / "testing_data" / "StormCatalog"
REF = Path(__file__).with_name("fixtures") / "pattern_reference_testing_data.csv"
needs_catalog = pytest.mark.skipif(not CAT.is_dir(), reason="testing_data/StormCatalog not available")
LAT = 40 + np.arange(200) / 30.0
LON = -95 + np.arange(220) / 30.0
KM_Y = 111.32 / 30.0                                   # km per cell (north)
KM_X = KM_Y * np.cos(np.radians(LAT.mean()))           # km per cell (east), at the centre of the grid


def blob_field(shape=(120, 130), seed=0, sigma=4):
    rng = np.random.default_rng(seed)
    f = ndimage.gaussian_filter(rng.random(shape), sigma)
    f = (f - f.min()) / (f.max() - f.min()) * 20
    yy, xx = np.indices(shape)
    env = np.exp(-(((yy - shape[0] / 2) / 25) ** 2 + ((xx - shape[1] / 2) / 30) ** 2))
    return np.where(env > 0.2, f * env, 0.0)


def centroids(rain, masks):
    lon2d, lat2d = np.meshgrid(LON, LAT)
    w = [np.where(m, r, 0.0) for r, m in zip(rain, masks)]
    return (np.array([(lon2d * x).sum() / x.sum() for x in w]), np.array([(lat2d * x).sum() / x.sum() for x in w]))


@pytest.mark.parametrize("shift", [(3.4, -5.7), (-6.2, 2.5), (0.0, 7.3)])
def test_subcell_shift(shift):
    a = blob_field()
    b = ndimage.shift(a, shift, order=3, mode="constant")
    dr, dc, c = masked_shift(a, b, a > 0.5, b > 0.5)
    assert c > 0.9
    assert dr == pytest.approx(shift[0], abs=0.25) and dc == pytest.approx(shift[1], abs=0.25)


def test_ignores_rain_outside_the_storm():
    a = blob_field(shape=(200, 200), seed=1)
    a[:, 140:] = 0
    other = np.roll(blob_field(shape=(200, 200), seed=2), 70, axis=1) * (np.indices((200, 200))[1] > 150)
    b = ndimage.shift(a, (4, 3), order=3)
    other_b = ndimage.shift(other, (-6, -9), order=3)          # a second storm moving the other way
    dr, dc, c = masked_shift(a + other, b + other_b, a > 0.5, b > 0.5)
    assert dr == pytest.approx(4, abs=0.4) and dc == pytest.approx(3, abs=0.4)


def test_translating_storm_has_no_propagation():
    """A storm that only translates: the pattern velocity equals the velocity of the rain mass."""
    base = np.zeros((200, 220))
    base[20:140, 20:150] = blob_field(seed=3)
    step = (4.0, 6.0)                                           # cells per hour (north, east)
    rain = np.array([ndimage.shift(base, (step[0] * t, step[1] * t), order=3, mode="constant").clip(0)
                     for t in range(8)])
    masks = [r > 0.5 for r in rain]
    hours = np.arange(8)
    clon, clat = centroids(rain, masks)
    out, steps = storm_pattern(rain, LON, LAT, masks, hours, clon, clat, seed=[1, 2])
    u, v = step[1] * KM_X / 3.6, step[0] * KM_Y / 3.6
    assert len(steps) == 7 and out["pattern_accepted"] == 7
    assert out["pattern_u"] == pytest.approx(u, rel=0.05) and out["pattern_v"] == pytest.approx(v, rel=0.05)
    assert out["pattern_direction_deg"] == pytest.approx(np.degrees(np.arctan2(v, u)), abs=2)
    assert out["pattern_av"] < 0.01 and out["pattern_coherent"]
    assert out["propagation_speed"] < 0.5
    assert out["centroid_vs_pattern_deg"] < 3
    assert out["pattern_robust"]


def test_propagation_is_mass_velocity_minus_pattern_velocity():
    """The same rain fields with a rain-mass centroid that does not move: the propagation cancels the pattern."""
    base = np.zeros((200, 220))
    base[20:140, 20:150] = blob_field(seed=3)
    rain = np.array([ndimage.shift(base, (4.0 * t, 6.0 * t), order=3, mode="constant").clip(0) for t in range(8)])
    masks = [r > 0.5 for r in rain]
    still = np.full(8, LON[110]), np.full(8, LAT[100])
    out, _ = storm_pattern(rain, LON, LAT, masks, np.arange(8), *still, seed=[1, 2], cross_check=False)
    assert out["centroid_net_u"] == 0 and out["centroid_net_v"] == 0
    assert out["propagation_u"] == pytest.approx(-out["pattern_u"]) and out["propagation_v"] == pytest.approx(-out["pattern_v"])
    assert out["propagation_along"] == pytest.approx(-out["pattern_speed"]) and abs(out["propagation_cross"]) < 1e-9
    assert out["propagation_speed"] == pytest.approx(out["pattern_speed"])
    assert "pattern_robust" not in out                          # no cross-check requested


def test_too_few_accepted_steps_give_no_velocity():
    rng = np.random.default_rng(5)
    rain = rng.random((6, 200, 220)) * 5                        # no pattern persists from hour to hour
    masks = [np.zeros((200, 220), bool) for _ in range(6)]
    for m in masks:
        m[60:140, 60:160] = True
    clon, clat = centroids(rain, masks)
    out, steps = storm_pattern(rain, LON, LAT, masks, np.arange(6), clon, clat, seed=[1, 2])
    assert out["pattern_steps"] == 5 and out["pattern_accepted"] < 3
    assert "pattern_u" not in out and (steps["reason"] != "").sum() >= 3


def test_robust_preset_needs_no_registration():
    rain = np.array([ndimage.shift(blob_field(), (2.0 * t, 3.0 * t), order=1, mode="constant") for t in range(4)])
    from stormcatalog_analyzer.tracking import robust
    cfg = E.EnlargedConfig(tracker="robust:partition_keepsplit")
    a = E.track_field(rain, cfg, LAT[:120], LON[:130])
    b = robust.enlarged_tracker("partition_keepsplit")(rain, LAT[:120], LON[:130], cfg)
    np.testing.assert_array_equal(a, b)


@needs_catalog
def test_enlarged_options_on_testing_catalog():
    """The options add columns and leave every other value unchanged; the values are those of the reference
    (computed with this package and checked against the storm table of the storm-motion study)."""
    ref = pd.read_csv(REF).set_index("storm_event")
    files = [CAT / f for f in ref.index]
    base = E.EnlargedConfig(margin_deg=0)
    full = E.EnlargedConfig(margin_deg=0, pattern=True, linger_class=True)
    rows = []
    for f in files:
        ev = E.read_catalog_event(f, load_rain=True)
        fld = E.field_from_catalog(ev)
        r0, _ = E.analyze_event(ev, fld, base, tile="B7test", with_tile=False, return_footprints=False)
        r1, _ = E.analyze_event(ev, fld, full, tile="B7test", with_tile=False, return_footprints=False)
        assert not any(k.startswith(("pattern", "propagation", "linger")) for k in r0)
        for k, v in r0.items():
            if isinstance(v, float) and np.isnan(v):
                assert np.isnan(r1[k])
            else:
                assert r1[k] == v, k
        rows.append(r1)
    d = pd.DataFrame(rows).set_index("storm_event").loc[ref.index]
    assert list(d["linger_class"]) == list(ref["linger_class"])
    for c in ("pattern_accepted", "pattern_steps"):
        np.testing.assert_array_equal(d[c].values, ref[c].values)
    np.testing.assert_array_equal(d["pattern_coherent"].astype(bool).values, ref["pattern_coherent"].values)
    np.testing.assert_array_equal(d["pattern_robust"].astype(bool).values, ref["pattern_robust"].values)
    for c in ("pattern_u", "pattern_v", "pattern_av", "pattern_p", "propagation_u", "propagation_v",
              "turn_half_deg", "net_req"):
        np.testing.assert_allclose(d[c].astype(float), ref[c], rtol=1e-6, atol=1e-8, err_msg=c)
