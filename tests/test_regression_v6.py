"""Regression against the V6 paper code on the bundled Iowa catalog (testing_data, 51 events).

The fixture ``fixtures/v6_reference_testing_data_B7test.csv`` was produced by the paper's tracking
notebook (Perez-HydroSystems/Storm_Motion_CONUS @ e00e55b, ``Notebooks/00_storm_tracking_tile.ipynb``,
run with papermill on ``testing_data/StormCatalog`` with ``domain_name='B7test'``). The package must
reproduce it: same retained events and analysed windows, identical coherence classification, and
floating-point agreement for the motion descriptors.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stormcatalog_analyzer.config import load_config
from stormcatalog_analyzer.pipeline import run

ROOT = Path(__file__).resolve().parents[1]
REF = Path(__file__).with_name("fixtures") / "v6_reference_testing_data_B7test.csv"
DOMAIN = "B7test"   # coherence seeds are built from "<domain>/<event>", as in the V6 notebook

EXACT = {  # V6 column -> package column, integer or boolean
    "n_steps": "n_positions", "track_positions": "track_positions", "truncated": "truncated",
    "window_start": "window_start", "n_segments": "n_segments", "n_sim_used": "n_sim_used",
    "tested": "tested", "coherent": "coherent",
}
CLOSE = {  # V6 column -> package column, floating point
    "storm_velocity": "mean_speed_ms", "angular_variance_weighted": "angular_variance",
    "area": "storm_area_km2", "r_weighted": "r_weighted", "p_weighted": "p_weighted",
    "r_unweighted": "r_unweighted", "p_unweighted": "p_unweighted",
    "mean_intensity_ellipse": "mean_intensity_mmh", "ellipse_area_km2": "mean_ellipse_area_km2",
}


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    if not (ROOT / "testing_data" / "StormCatalog").is_dir():
        pytest.skip("testing_data/StormCatalog not available")
    cfg = load_config(ROOT / "configs" / "testing_data.json")
    cfg.io.output_dir = str(tmp_path_factory.mktemp("out"))
    cfg.io.domain_name = DOMAIN
    run(cfg, verbose=False)
    pkg = pd.read_csv(Path(cfg.io.output_dir) / DOMAIN / f"storm_properties_{DOMAIN}.csv")
    ref = pd.read_csv(REF)
    return ref.set_index("storm_event").sort_index(), pkg.set_index("storm_event").sort_index()


def test_same_retained_events(tables):
    ref, pkg = tables
    assert list(ref.index) == list(pkg.index)
    assert len(ref) == 51


@pytest.mark.parametrize("v6_col,pkg_col", EXACT.items())
def test_exact_columns(tables, v6_col, pkg_col):
    ref, pkg = tables
    np.testing.assert_array_equal(ref[v6_col].astype(float).values, pkg[pkg_col].astype(float).values)


@pytest.mark.parametrize("v6_col,pkg_col", CLOSE.items())
def test_float_columns(tables, v6_col, pkg_col):
    ref, pkg = tables
    np.testing.assert_allclose(pkg[pkg_col].values, ref[v6_col].values, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_direction(tables):
    ref, pkg = tables
    d = (pkg["mean_direction_deg"].values - ref["storm_direction"].values + 180) % 360 - 180
    assert np.nanmax(np.abs(d)) < 1e-9
