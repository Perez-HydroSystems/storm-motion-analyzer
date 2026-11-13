"""Envelope-drift checks on synthetic fields with a known translation."""
import numpy as np
import pytest

from stormcatalog_analyzer.motion.drift import event_drift, axis_unit_vector, percentile_of


def gaussian_storm(speed_ms, bearing_deg, hours=180, sigma_km=60.0):
    """Hourly Gaussian rain blob on a 0.05-deg grid, translating at a constant velocity."""
    lat = np.arange(38.0, 46.0, 0.05)
    lon = np.arange(-99.0, -89.0, 0.05)
    lo2, la2 = np.meshgrid(lon, lat)
    kx = 111.32 * np.cos(np.deg2rad(42.0))                  # km per degree of longitude near 42 N
    u = axis_unit_vector(bearing_deg) * speed_ms * 3.6      # km/h (east, north)
    t = np.arange(hours) - hours / 2
    rain = np.empty((hours, lat.size, lon.size))
    for i, ti in enumerate(t):
        cx, cy = -94.0 + u[0] * ti / kx, 42.0 + u[1] * ti / 111.32
        rain[i] = np.exp(-(((lo2 - cx) * kx) ** 2 + ((la2 - cy) * 111.32) ** 2) / (2 * sigma_km ** 2))
    return rain, lon, lat


@pytest.mark.parametrize("speed,bearing", [(0.5, 156.0), (1.0, 90.0), (2.0, 300.0)])
def test_recovers_imposed_drift(speed, bearing):
    rain, lon, lat = gaussian_storm(speed, bearing, hours=96 if speed >= 2 else 180)
    d = event_drift(rain, lon, lat, window_hr=48, axis_azimuth_deg=bearing)
    assert d["drift_speed_ms"] == pytest.approx(speed, rel=0.03)
    assert abs((d["drift_bearing_deg"] - bearing + 180) % 360 - 180) < 2.0
    assert d["drift_axis_ms"] == pytest.approx(speed, rel=0.03)
    assert abs(d["drift_cross_ms"]) < 0.03 * speed
    assert d["drift_r2"] > 0.99
    assert d["w48_drift_speed_ms"] == pytest.approx(speed, rel=0.05)


def test_stationary_storm_has_no_drift():
    rain, lon, lat = gaussian_storm(0.0, 0.0)
    d = event_drift(rain, lon, lat)
    assert d["drift_speed_ms"] < 1e-6


def test_mask_restricts_centroid():
    # Only the western half of the grid: an eastward-moving blob that leaves the mask drifts less
    rain, lon, lat = gaussian_storm(1.0, 90.0)
    mask = np.broadcast_to(lon[None, :] < -94.0, (lat.size, lon.size))
    full = event_drift(rain, lon, lat)["drift_speed_ms"]
    masked = event_drift(rain, lon, lat, mask=mask)["drift_speed_ms"]
    assert masked < full


def test_percentile_of():
    assert percentile_of(0.5, [0.1, 0.2, 0.6, 0.9]) == 0.5
    assert np.isnan(percentile_of(np.nan, [1, 2]))
