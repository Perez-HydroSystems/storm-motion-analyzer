"""Robust linking options (tracking.robust): the defaults reproduce V6, and each option does what it claims on
small constructed cases (merge with a distractor, split, the V6 t = 1 label collision, pattern shift, cores)."""
from pathlib import Path

import numpy as np
import pytest

from stormcatalog_analyzer.tracking.identification import storm_area_identification
from stormcatalog_analyzer.tracking.robust import (PRESETS, LinkingConfig, core_identification, pattern_shift,
                                                   preset, track_field, track_robust)
from stormcatalog_analyzer.tracking.tracking import track

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "testing_data" / "StormCatalog"
LAT = 40 + np.arange(80) / 30.0
LON = -95 + np.arange(80) / 30.0


def blob(shape, r, c, rad, peak=10.0):
    yy, xx = np.indices(shape)
    d2 = (yy - r) ** 2 + (xx - c) ** 2
    return np.where(d2 <= rad ** 2, peak * np.exp(-d2 / (2 * (rad / 2) ** 2)) + 0.6, 0.0)


def centroid(rain, mask):
    w = np.where(mask, rain, 0.0)
    yy, xx = np.indices(rain.shape)
    return (yy * w).sum() / w.sum(), (xx * w).sum() / w.sum()


# --------------------------------------------------------------------------------------- default = V6
@pytest.mark.skipif(not CATALOG.is_dir(), reason="testing_data not available")
@pytest.mark.parametrize("dry", [0, 1])
def test_default_reproduces_v6_track(dry):
    import xarray as xr
    for f in sorted(CATALOG.glob("*.nc"))[::7]:
        with xr.open_dataset(f) as ds:
            rain = np.nan_to_num(ds["rain"].values)
            lat, lon = ds["latitude"].values, ds["longitude"].values
        ident = storm_area_identification(rain, 2, 0.5)
        ref = track(ident, ratio_threshold=0.2, dry_spell_time=dry)
        got = track_robust(ident, rain, lat, lon, LinkingConfig(dry_spell_time=dry))
        np.testing.assert_array_equal(ref, got)


def test_default_reproduces_v6_on_constructed_cases():
    for case in (merge_case(), split_case()):
        ident = storm_area_identification(case, 2, 0.5)
        np.testing.assert_array_equal(track(ident, 0.2, 0), track_robust(ident, case, LAT, LON))


def test_presets_valid_and_v6_flag():
    assert LinkingConfig().is_v6()
    for name in PRESETS:
        cfg = preset(name)
        assert cfg.is_v6() == (name == "v6")


# --------------------------------------------------------------------------------------- merge
def merge_case(nt=8):
    """Target (moving east 3 cells/h) and a somewhat smaller distractor (> 25% of the target's rain) coming from
    the north that merges into it from t = 3 on (the objects then touch and form one object)."""
    rain = np.zeros((nt, 80, 80))
    for t in range(nt):
        rain[t] += blob((80, 80), 40, 15 + 3 * t, 9, 12.0)
        rain[t] += blob((80, 80), 40 + max(28 - 6 * t, 12), 40 + max(0, 12 - 3 * t), 8, 9.0)
    return rain


def target_steps(rain, labels, start=(40, 15)):
    """Hourly centroid steps (cells) of the track holding the target at t = 0."""
    k = labels[0][start]
    pos = [centroid(rain[t], labels[t] == k) for t in range(rain.shape[0]) if (labels[t] == k).any()]
    return np.hypot(*np.diff(np.array(pos), axis=0).T)


def test_partition_keeps_target_position_through_merge():
    rain = merge_case()
    ident = storm_area_identification(rain, 2, 0.5)
    v6 = track_robust(ident, rain, LAT, LON)
    part, info = track_robust(ident, rain, LAT, LON, preset("partition"), return_info=True)
    s_v6, s_part = target_steps(rain, v6), target_steps(rain, part)
    # V6: the merge pulls the target's centroid towards the distractor (a step well above the 3-cell motion)
    assert s_v6.max() > 4.5
    # partition: the target keeps its own rain mass, steps stay close to its true motion
    assert info["partitions"], "the merged object should have been partitioned"
    assert s_part.max() < 4.0
    assert len(s_part) == len(s_v6)


def test_merge_volume_keeps_largest_parent():
    rain = merge_case()
    ident = storm_area_identification(rain, 2, 0.5)
    lab = track_robust(ident, rain, LAT, LON, preset("merge_volume"))
    k = lab[0][40, 15]
    assert (lab[-1] == k).any()                       # the target's id survives the merge


# --------------------------------------------------------------------------------------- split
def split_case(nt=8):
    """One object (large core + small lobe) splits at t = 3: the lobe drifts away to the north."""
    rain = np.zeros((nt, 80, 80))
    for t in range(nt):
        rain[t] += blob((80, 80), 40, 20 + 2 * t, 9, 12.0)
        rain[t] += blob((80, 80), 40 + 8 + (0 if t < 3 else 6 * (t - 2)), 20 + 2 * t, 5, 5.0)
    return rain


def test_split_largest_gives_new_id_to_smaller_child():
    rain = split_case()
    ident = storm_area_identification(rain, 2, 0.5)
    v6 = track_robust(ident, rain, LAT, LON)
    lg = track_robust(ident, rain, LAT, LON, preset("split_largest"))
    t = rain.shape[0] - 1
    k6, kl = v6[0][40, 20], lg[0][40, 20]
    # V6: both children carry the parent's id at the last hour
    assert len(np.unique(ident[t][v6[t] == k6])) == 2
    # largest: only the main core keeps it
    assert len(np.unique(ident[t][lg[t] == kl])) == 1
    assert lg[t][40, 20 + 2 * t] == kl


# --------------------------------------------------------------------------------------- V6 t = 1 bookkeeping
def test_fix_labels_removes_t1_collision():
    ident = np.zeros((3, 20, 20), int)
    ident[0, 2:5, 2:5] = 1
    ident[0, 10:13, 10:13] = 2          # storm 2 continues ...
    ident[1, 10:13, 10:13] = 1          # ... as raw label 1 at t = 1
    ident[1, 2:5, 15:18] = 2            # a new storm at t = 1 (V6 numbers it max(ident[1]) + 1 = 3)
    ident[1, 15:18, 2:5] = 3            # another new storm -> 4? no: 3 + 1, 3 + 2 ... collides with nothing here
    ident[2] = ident[1]
    v6, info = track_robust(ident.astype(float), None, None, None, return_info=True)
    np.testing.assert_array_equal(v6, track(ident.astype(float), 0.2, 0))
    # build a colliding case: the continuing storm inherits id 3 from t = 0
    ident2 = np.zeros((2, 20, 20), int)
    ident2[0, 2:5, 2:5] = 1
    ident2[0, 6:8, 6:8] = 2
    ident2[0, 10:13, 10:13] = 3          # continues
    ident2[1, 10:13, 10:13] = 1          # raw label 1 at t = 1, linked to id 3
    ident2[1, 2:5, 15:18] = 2            # new -> V6 gives max(ident2[1]) + 1 = 3: collision with the continuing storm
    v6b, info_b = track_robust(ident2.astype(float), None, None, None, return_info=True)
    np.testing.assert_array_equal(v6b, track(ident2.astype(float), 0.2, 0))
    assert v6b[1][11, 11] == v6b[1][3, 16]           # V6 fuses the two objects into one id
    assert info_b["collisions"]
    fixed = track_robust(ident2.astype(float), None, None, None, LinkingConfig(fix_labels=True))
    assert fixed[1][11, 11] != fixed[1][3, 16]
    assert fixed[1][11, 11] == fixed[0][11, 11]


# --------------------------------------------------------------------------------------- prediction and gates
def test_pattern_shift_recovers_displacement():
    rng = np.random.default_rng(0)
    from scipy import ndimage
    f = ndimage.gaussian_filter(rng.standard_normal((90, 90)), 3)
    r0 = np.clip(f * 20, 0, None)
    r1 = np.roll(np.roll(r0, 4, axis=0), -6, axis=1)
    mask = np.zeros_like(r0, bool)
    mask[30:60, 30:60] = r0[30:60, 30:60] > 0
    dr, dc, corr = pattern_shift(r0, r1, mask, 15)
    assert abs(dr - 4) < 0.6 and abs(dc + 6) < 0.6 and corr > 0.9


def test_prediction_links_fast_small_storm():
    """A storm (diameter 14, 16 cells/h) never overlaps itself: V6 starts a new id every hour, advection links it."""
    rain = np.zeros((5, 80, 80))
    for t in range(5):
        rain[t] = blob((80, 80), 40, 8 + 16 * t, 7, 10.0)
    ident = storm_area_identification(rain, 2, 0.5)
    assert len(np.unique(track_robust(ident, rain, LAT, LON)[ident > 0])) == 5
    # the pattern velocity is measured from the rain field itself, so it links from the first hour on
    lab = track_robust(ident, rain, LAT, LON, preset("predict_pattern"))
    assert len(np.unique(lab[ident > 0])) == 1
    # the centroid velocity needs a linked history: it cannot start a track that never overlaps itself
    lab = track_robust(ident, rain, LAT, LON, preset("predict_centroid"))
    assert len(np.unique(lab[ident > 0])) == 5


def test_centroid_prediction_keeps_link_of_accelerating_storm():
    """A small storm that overlaps itself at first (3 cells/h) and then speeds up to 7 cells/h: the centroid
    velocity learnt from the first hours keeps the link where the plain overlap is lost."""
    rain = np.zeros((7, 80, 80))
    x = 10.0
    for t in range(7):
        rain[t] = blob((80, 80), 40, x, 4, 10.0)
        x += 3 if t < 3 else 7
    ident = storm_area_identification(rain, 2, 0.5)
    n_v6 = len(np.unique(track_robust(ident, rain, LAT, LON)[ident > 0]))
    n_c = len(np.unique(track_robust(ident, rain, LAT, LON, preset("predict_centroid"))[ident > 0]))
    assert n_c < n_v6


def test_speed_gate_rejects_far_link():
    rain = merge_case()
    ident = storm_area_identification(rain, 2, 0.5)
    _, info = track_robust(ident, rain, LAT, LON, LinkingConfig(max_speed_ms=1.0), return_info=True)
    assert info["gated"] > 0


def test_hungarian_is_one_to_one():
    rain = split_case()
    ident = storm_area_identification(rain, 2, 0.5)
    lab = track_robust(ident, rain, LAT, LON, LinkingConfig(assignment="hungarian"))
    for t in range(rain.shape[0]):
        for k in np.unique(lab[t][lab[t] > 0]):
            assert len(np.unique(ident[t][lab[t] == k])) == 1


# --------------------------------------------------------------------------------------- core identification
def test_core_identification_separates_bridged_cores():
    rain = np.zeros((1, 60, 60))
    rain[0] += blob((60, 60), 30, 15, 6, 10.0) + blob((60, 60), 30, 45, 6, 10.0)
    rain[0, 28:33, 15:46] = np.maximum(rain[0, 28:33, 15:46], 0.8)     # light-rain bridge
    v6 = storm_area_identification(rain, 2, 0.5)
    cores = core_identification(rain, 3.0, 0.5, 2)
    assert len(np.unique(v6[v6 > 0])) == 1
    assert len(np.unique(cores[cores > 0])) == 2
    assert ((cores > 0) <= (rain >= 0.5)).all()


def test_track_field_runs_all_presets():
    rain = merge_case()
    for name in PRESETS:
        lab = track_field(rain, LAT, LON, name)
        assert lab.shape == rain.shape and lab.min() == 0
    lab = track_field(rain, LAT, LON, "partition_pattern", core_threshold=3.0)
    assert lab.max() >= 1


# --------------------------------------------------------------------------------------- package integration
@pytest.mark.skipif(not CATALOG.is_dir(), reason="testing_data not available")
def test_storm_tracking_event_options():
    import contextlib
    import io

    import xarray as xr
    from stormcatalog_analyzer.tracking.dataset import storm_tracking_event
    f = sorted(CATALOG.glob("*.nc"))[-1]
    out = {}
    with xr.open_dataset(f) as ds, contextlib.redirect_stdout(io.StringIO()):
        for key, kw in {"v6": {}, "v6_explicit": {"linking": "v6"}, "robust": {"linking": "partition_pattern"},
                        "core": {"core_threshold": 2.0}}.items():
            d = {}
            storm_tracking_event(ds, d, morph_radius=2, high_threshold=0.5, ratio_threshold=0.2, dry_spell_time=0,
                                 **kw)
            out[key] = d["track_array"]
    np.testing.assert_array_equal(out["v6"], out["v6_explicit"])
    assert out["robust"].shape == out["v6"].shape and out["robust"].max() >= 1
    assert out["core"].max() >= 1


def test_config_accepts_linking():
    from stormcatalog_analyzer.config import TrackingConfig
    assert TrackingConfig().linking is None and TrackingConfig().core_threshold_mmhr is None
