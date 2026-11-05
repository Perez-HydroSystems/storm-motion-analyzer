"""Envelope drift of a storm's rain field.

Storm *cell* motion (``motion.direction``, ``pipeline.summarize_motion``) follows the tracked
storm object over its most-intense window and is typically fast (several m/s). The **envelope
drift** is the slow, sustained displacement of the whole rain field over the event (or over its
wettest multi-day window): where the rain falls on day 1 versus day 5. It is the quantity a
design storm "moving down the basin" changes, and it is what flood response is sensitive to,
because a rain envelope that drifts with the flood wave concentrates the runoff in time.

Method (rain-weighted centroid trend):

1. Sum the hourly field into ``block_hr`` blocks (default 6 h).
2. For each block, the rain-weighted centroid (x, y) in a projected CRS (default EPSG:5070,
   km), with cell weights = rain x relative cell area (cos(latitude) on an equal-angle grid).
3. Keep the blocks holding at least ``min_step_frac`` of the event total (default 2%).
4. Weighted (by block rain) least-squares trend of x(t) and y(t): the drift velocity
   (vx, vy) in m/s, its speed and compass bearing, the weighted R^2 of the track, and,
   optionally, the component along a reference axis (e.g., a river basin's downstream axis).

The same fit is repeated on the wettest ``window_hr`` hours (default 72 h), which isolates
the drift during the storm's main rain. A 6-h block and 2% threshold follow Perez et al.
(Paper 2 of the Red Rock PMF study); results are reported for the whole event (no prefix)
and the wettest window (prefix ``w{window_hr}_``).

Directions: ``drift_bearing_deg`` is a compass bearing of the direction of motion (0 = N,
clockwise); ``drift_direction_deg`` follows the package convention (0 = E, counterclockwise).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from pyproj import Transformer

_KEYS = ["drift_n_steps", "drift_vx_ms", "drift_vy_ms", "drift_speed_ms", "drift_bearing_deg",
         "drift_direction_deg", "drift_r2", "drift_axis_ms", "drift_cross_ms"]


def axis_unit_vector(axis_azimuth_deg: float) -> np.ndarray:
    """(east, north) unit vector of a compass azimuth (0 = N, clockwise)."""
    a = np.deg2rad(axis_azimuth_deg)
    return np.array([np.sin(a), np.cos(a)])


def project_grid(lon, lat, crs: str = "EPSG:5070"):
    """Cell-center coordinates (km) in ``crs`` and relative cell areas for a regular lon/lat grid.

    ``lon`` and ``lat`` are 1-D coordinate vectors; returns flattened x, y (km) and area weights
    (cos(latitude), i.e. relative areas of equal-angle cells), in row-major (lat, lon) order.
    """
    lo2, la2 = np.meshgrid(np.asarray(lon, float), np.asarray(lat, float))
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = tr.transform(lo2, la2)
    return x.ravel() / 1e3, y.ravel() / 1e3, np.cos(np.deg2rad(la2)).ravel()


def block_centroids(rain2d: np.ndarray, x: np.ndarray, y: np.ndarray, area: np.ndarray,
                    block_hr: int = 6, min_step_frac: float = 0.02):
    """Block totals, their rain-weighted centroids and volumes.

    ``rain2d`` is (time, cells) hourly rain; ``x``, ``y`` (km) and ``area`` are per cell.
    Returns (block mid-times in h, cx, cy, volume) for the blocks holding at least
    ``min_step_frac`` of the total. A trailing incomplete block is dropped.
    """
    rain2d = np.nan_to_num(np.asarray(rain2d, float), nan=0.0)
    rain2d[rain2d < 0] = 0.0
    nb = rain2d.shape[0] // block_hr
    if nb == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    blk = rain2d[: nb * block_hr].reshape(nb, block_hr, -1).sum(axis=1) * area[None, :]
    vol = blk.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = (blk * x[None, :]).sum(axis=1) / vol
        cy = (blk * y[None, :]).sum(axis=1) / vol
    tmid = (np.arange(nb) + 0.5) * block_hr
    ok = (vol > 0) & (vol >= min_step_frac * vol.sum())
    return tmid[ok], cx[ok], cy[ok], vol[ok]


def drift_fit(tmid_h, cx, cy, w, axis_azimuth_deg: Optional[float] = None) -> dict:
    """Weighted linear trend of a centroid track (km, h) -> drift velocity and diagnostics."""
    out = {k: np.nan for k in _KEYS}
    out["drift_n_steps"] = int(len(w))
    w = np.asarray(w, float)
    if len(w) < 2 or w.sum() <= 0:
        return out
    W = w / w.sum()
    tm = (W * tmid_h).sum()
    st = (W * (tmid_h - tm) ** 2).sum()
    if st <= 0:
        return out
    ss_res = ss_tot = 0.0
    slope = {}
    for name, c in (("x", cx), ("y", cy)):
        cm = (W * c).sum()
        b = (W * (tmid_h - tm) * (c - cm)).sum() / st
        slope[name] = b
        ss_res += (W * (c - cm - b * (tmid_h - tm)) ** 2).sum()
        ss_tot += (W * (c - cm) ** 2).sum()
    vx, vy = slope["x"] / 3.6, slope["y"] / 3.6               # km/h -> m/s
    speed = float(np.hypot(vx, vy))
    bearing = float(np.degrees(np.arctan2(vx, vy)) % 360) if speed > 0 else np.nan
    out.update({"drift_vx_ms": float(vx), "drift_vy_ms": float(vy), "drift_speed_ms": speed,
                "drift_bearing_deg": bearing,
                "drift_direction_deg": float((90.0 - bearing) % 360) if speed > 0 else np.nan,
                "drift_r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan})
    if axis_azimuth_deg is not None:
        u = axis_unit_vector(axis_azimuth_deg)
        out["drift_axis_ms"] = float(vx * u[0] + vy * u[1])
        out["drift_cross_ms"] = float(-vx * u[1] + vy * u[0])
    return out


def wettest_window_start(rain2d: np.ndarray, area: np.ndarray, window_hr: int) -> int:
    """Start index of the wettest ``window_hr`` consecutive hours of the area-weighted total."""
    vol_h = (np.nan_to_num(rain2d) * area[None, :]).sum(axis=1)
    if len(vol_h) <= window_hr:
        return 0
    c = np.concatenate([[0.0], np.cumsum(vol_h)])
    return int(np.argmax(c[window_hr:] - c[:-window_hr]))


def event_drift(rain: np.ndarray, lon, lat, block_hr: int = 6, min_step_frac: float = 0.02,
                window_hr: Optional[int] = 72, axis_azimuth_deg: Optional[float] = None,
                crs: str = "EPSG:5070", mask: Optional[np.ndarray] = None) -> dict:
    """Envelope drift of one event.

    ``rain`` is hourly (time, lat, lon); ``lon``/``lat`` are 1-D coordinates. ``mask`` (lat, lon),
    optional, restricts the centroid to a region (e.g., a watershed). Returns the whole-event
    drift (keys in ``_KEYS``) and, if ``window_hr`` is set and shorter than the event, the drift of
    the wettest ``window_hr`` hours with prefix ``w{window_hr}_`` plus its start index.
    """
    rain = np.asarray(rain, float)
    x, y, area = project_grid(lon, lat, crs)
    r2d = rain.reshape(rain.shape[0], -1)
    if mask is not None:
        keep = np.asarray(mask, bool).ravel()
        r2d, x, y, area = r2d[:, keep], x[keep], y[keep], area[keep]
    out = drift_fit(*block_centroids(r2d, x, y, area, block_hr, min_step_frac), axis_azimuth_deg)
    out["n_hours"] = int(rain.shape[0])
    if window_hr:
        w0 = wettest_window_start(r2d, area, window_hr)
        ow = drift_fit(*block_centroids(r2d[w0:w0 + window_hr], x, y, area, block_hr, min_step_frac),
                       axis_azimuth_deg)
        out.update({f"w{window_hr}_{k}": v for k, v in ow.items()})
        out[f"w{window_hr}_start_h"] = w0
    return out


def percentile_of(value: float, reference) -> float:
    """Fraction of ``reference`` values strictly below ``value`` (an empirical percentile, 0-1)."""
    ref = np.asarray(reference, float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0 or not np.isfinite(value):
        return np.nan
    return float((ref < value).mean())
