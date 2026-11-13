"""Enlarged-window tracking (stormcatalog_analyzer.enlarged) and the vectorised overlap tracker."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

from stormcatalog_analyzer.enlarged import (CatalogEvent, EnlargedConfig, Field, analyze_event, assign_system_ids,
                                            field_from_catalog, read_catalog_event, storm_number)
from stormcatalog_analyzer.tracking.identification import storm_area_identification
from stormcatalog_analyzer.tracking.tracking import track, track_fast

ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT / "testing_data" / "StormCatalog"
REF = Path(__file__).with_name("fixtures") / "v6_reference_testing_data_B7test.csv"
needs_catalog = pytest.mark.skipif(not CAT.is_dir(), reason="testing_data/StormCatalog not available")


# ------------------------------------------------------------------ vectorised tracker == V6 tracker
@pytest.mark.parametrize("dry_spell", [0, 1, 2])
@pytest.mark.parametrize("ratio", [0.2, 0.5])
def test_track_fast_identical_on_random_fields(dry_spell, ratio):
    rng = np.random.default_rng(dry_spell * 10 + int(ratio * 10))
    for _ in range(8):
        z = ndimage.gaussian_filter(rng.standard_normal((10, 70, 70)), (0.8, 3, 3))
        ident = storm_area_identification(np.clip(10 * z, 0, None), 1, 0.5)
        np.testing.assert_array_equal(track(ident, ratio, dry_spell), track_fast(ident, ratio, dry_spell))


@needs_catalog
def test_track_fast_identical_on_catalog():
    for f in sorted(CAT.glob("*.nc"), key=storm_number)[:6]:
        ev = read_catalog_event(f, load_rain=True)
        ident = storm_area_identification(np.nan_to_num(ev.rain), 2, 0.5)
        np.testing.assert_array_equal(track(ident, 0.2, 0), track_fast(ident, 0.2, 0))


# ------------------------------------------------------------------ catalog event: circle placement
B7_REPRO = Path("/Users/gjperezm/Databases/RainyDay_Storm_Catalogs_CONUS/Run_CONUS/Results_repro_2002_2022/B7/"
                "StormCatalog")


def _check_circle(files, max_cells, rel):
    from scipy.signal import correlate
    for f in files:
        ev = read_catalog_event(f, load_rain=True)
        tot = np.nansum(ev.rain, axis=0)
        w = ev.circle_on_tile()
        assert (tot * w).sum() / ev.circle.sum() == pytest.approx(ev.basin_rain_mm, rel=rel)
        c = correlate(tot, ev.circle, mode="valid")
        y, x = np.unravel_index(np.argmax(c), c.shape)
        assert abs(y - ev.yloc) <= max_cells and abs(x - ev.xloc) <= max_cells
        lon, lat = ev.circle_center
        assert ev.lon[0] < lon < ev.lon[-1] and ev.lat[0] < lat < ev.lat[-1]


@needs_catalog
def test_circle_location_testing_catalog():
    """xlocation/ylocation = lower-left cell of the circle mask's bounding box (latitude ascending). The bundled
    catalog was written by an older RainyDay from slightly different rain, so placements agree within one cell."""
    _check_circle(sorted(CAT.glob("*.nc"), key=storm_number)[::5], max_cells=1, rel=5e-3)


@pytest.mark.skipif(not B7_REPRO.is_dir(), reason="B7 reproduction catalog (RainyDay_fast 42d51ae) not available")
def test_circle_location_rainyday_fast_catalog():
    """With the production RainyDay (42d51ae) the placed circle is exactly the maximising placement and the mean
    24-h depth over it is basinrainfall."""
    _check_circle(sorted(B7_REPRO.glob("*.nc"), key=storm_number)[::20], max_cells=0, rel=1e-5)


# ------------------------------------------------------------------ margin 0: the tile pipeline is V6
@needs_catalog
def test_tile_descriptors_reproduce_v6_reference():
    ref = pd.read_csv(REF).set_index("storm_event")
    cfg = EnlargedConfig(margin_deg=0)
    rows = []
    for f in sorted(CAT.glob("*.nc"), key=storm_number):
        ev = read_catalog_event(f, load_rain=True)
        rows.append(analyze_event(ev, field_from_catalog(ev), cfg, tile="B7test", return_footprints=False)[0])
    df = pd.DataFrame(rows).set_index("storm_event").loc[ref.index]
    assert df["tile_retained"].all()
    np.testing.assert_array_equal(df["tile_n_positions"].values, ref["n_steps"].values)
    np.testing.assert_array_equal(df["tile_coherent"].values, ref["coherent"].values)
    np.testing.assert_allclose(df["tile_angular_variance"], ref["angular_variance_weighted"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(df["tile_path_speed_ms"], ref["storm_velocity"], rtol=1e-9)
    np.testing.assert_allclose(df["tile_p_weighted"], ref["p_weighted"], rtol=1e-9)
    d = (df["tile_direction_deg"].values - ref["storm_direction"].values + 180) % 360 - 180
    assert np.nanmax(np.abs(d)) < 1e-9
    # circle-selected storm on the tile is mostly the V6 storm, and its rain never leaves the tile
    assert (df["tile_fp_covered_pct"] > 50).mean() > 0.8
    assert np.nanmax(df["rain_outside_tile_pct"]) == 0.0


# ------------------------------------------------------------------ synthetic clipped storm
def _synthetic(n=360, nt=24, speed_cells=12.0, heading_deg=20.0, t0=6, npos=14, seed=0):
    """A rain blob (plus a weaker distractor) moving on a straight line across a large canvas."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    rain = np.zeros((nt, n, n))
    h = np.radians(heading_deg)
    start = np.array([n / 2 - 0.5 * npos * speed_cells * np.sin(h), n / 2 - 0.5 * npos * speed_cells * np.cos(h)])
    truth = []
    for i in range(npos):
        cy, cx = start + i * speed_cells * np.array([np.sin(h), np.cos(h)])
        g = 20 * np.exp(-(((yy - cy) / 18) ** 2 + ((xx - cx) / 30) ** 2))
        rain[t0 + i] += g * (1 + 0.1 * ndimage.gaussian_filter(rng.standard_normal((n, n)), 4))
        truth.append((cy, cx))
    rain[:, 20:40, 20:40] += 2.0     # stationary distractor far from the path
    lat = 35 + np.arange(n) / 30.0
    lon = -100 + np.arange(n) / 30.0
    times = np.datetime64("2010-06-01T01:00") + np.arange(nt) * np.timedelta64(1, "h")
    return Field(np.clip(rain, 0, None), lat, lon, times), np.array(truth)


def test_enlarged_window_recovers_clipped_track():
    fld, truth = _synthetic()
    n = len(fld.lat)
    # 150 x 150 tile covering only the first part of the path; circle (21 x 21) where the path enters the tile
    r0, c0 = int(truth[0][0]) - 60, int(truth[0][1]) - 40
    lat_t, lon_t = fld.lat[r0:r0 + 150], fld.lon[c0:c0 + 150]
    yy, xx = np.mgrid[0:21, 0:21]
    circle = ((yy - 10) ** 2 + (xx - 10) ** 2 <= 100).astype(float)
    tot = fld.rain[:, r0:r0 + 150, c0:c0 + 150].sum(0)
    from scipy.signal import correlate
    c = correlate(tot, circle, mode="valid")
    yl, xl = np.unravel_index(np.argmax(c), c.shape)
    ev = CatalogEvent("SYN.nc_storm_1_20100601.nc", 1, 1, fld.times, lat_t, lon_t, int(xl), int(yl), circle)
    errs = {}
    for name, f in (("tile", fld.crop(*ev.tile_bounds)), ("big", fld)):
        row, _ = analyze_event(ev, f, EnlargedConfig(margin_deg=0), tile="SYN", with_tile=False)
        assert row["retained"]
        lon, lat = np.array(row["big_track_lon"]), np.array(row["big_track_lat"])
        hrs = [int((np.datetime64(t) - fld.times[0]) / np.timedelta64(1, "h")) - 6 for t in row["big_track_time"]]
        ty, tx = truth[hrs, 0], truth[hrs, 1]
        err_km = np.hypot((lat - fld.lat[0] - ty / 30.0) * 111, (lon - fld.lon[0] - tx / 30.0) * 111 * np.cos(np.radians(lat)))
        errs[name] = (np.mean(err_km), row["big_direction_deg"], row["big_track_positions"], row)
    assert errs["big"][0] < 5.0                       # sub-cell-ish accuracy on the full canvas
    assert errs["big"][0] < errs["tile"][0] / 3       # the tile-clipped track is biased
    lat_mid = np.radians(fld.lat.mean())                # 20 deg in grid cells -> geographic direction
    expected = np.degrees(np.arctan2(np.sin(np.radians(20.0)), np.cos(np.radians(20.0)) * np.cos(lat_mid)))
    assert abs((errs["big"][1] - expected + 180) % 360 - 180) < 2.0
    assert errs["big"][2] > errs["tile"][2]           # the tile loses hours when the storm leaves it
    assert errs["tile"][3]["edge_touch_hours"] > 0 and errs["big"][3]["edge_touch_hours"] == 0
    assert errs["big"][3]["rain_outside_tile_pct"] > 10


# ------------------------------------------------------------------ duplicates
def test_assign_system_ids():
    T = pd.date_range("2010-06-01 01:00", periods=12, freq="h")
    rows = []
    for t in T:
        rows.append(("A1", "a", t, np.arange(0, 20)))            # system 1 seen from tile A1
        rows.append(("A2", "b", t, np.arange(5, 25)))            # same system from A2 (75% of cells shared)
        rows.append(("A3", "c", t, np.arange(100, 120)))         # same hours, elsewhere -> separate
    for t in T[:4]:
        rows.append(("A4", "d", t, np.arange(0, 20)))            # same place but only 4 of its 4 hours ...
    for t in T[:10]:
        rows.append(("A1", "e", t, np.arange(0, 20)))            # same tile as "a": never linked to it directly
    fp = pd.DataFrame(rows, columns=["tile", "storm_event", "time", "cells"])
    ids, pairs = assign_system_ids(fp)
    s = ids.set_index("storm_event")["system_id"]
    assert s["a"] == s["b"]
    assert s["c"] != s["a"]
    assert s["d"] == s["a"]          # 4 shared hours = 100% of the shorter event, full footprint overlap
    # "e" (tile A1) links to "b" (A2) and "d" (A4), hence joins the system transitively
    assert s["e"] == s["a"]
    assert not ((pairs["tile_a"] == pairs["tile_b"]).any())
    assert ids["system_id"].nunique() == 2
