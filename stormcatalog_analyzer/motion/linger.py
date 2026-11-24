"""Travel or linger: how a storm that is not directionally coherent moved (stall, turn, or meander).

The coherence test (``motion.direction.coherence_test``) separates the storms that kept a preferred direction of
motion (coherent: they travel) from those that did not (non-coherent: they linger). Non-coherent storms are of three
kinds, which this module separates from the analysed track alone (hourly rain-weighted centroids, 5 to 12 positions)
and the mean area of the storm core:

    stall     the centroid ends within one core radius of where it started, or the net speed is below 2 m/s: the
              storm stays over roughly the same area
    turn      the heading changes by 90 degrees or more between two parts of the track, and every part moves
    meander   the other non-coherent storms: they travel more than one core radius, without a clean turn

Per storm (n positions, n - 1 hourly steps):
    r_eq        equivalent radius of the mean core area, sqrt(area / pi)                                    (km)
    net_req     net displacement / r_eq
    turn_half   |heading change| between the first and the second half of the track (deg, 0-180). The halves split
                the positions at m = n // 2 (positions 0..m and m..n-1); the heading of a half is the geodesic
                azimuth of its own net displacement.
    rot_thirds  signed heading change over the three thirds of the track (positions 0, round((n-1)/3),
                round(2(n-1)/3), n-1): turn(1->2) + turn(2->3), kept only when both turns have the same sign
                (a consistent rotation), else 0 (deg).
    seg_ratio   displacement of each half (or third) divided by r_eq x (its share of the steps): how fast the
                segment moved, in core radii per track. The jitter of a storm that stays in place gives small
                values; a segment "moves" when its ratio is at least K.

Rules, applied in this order (default parameters A = 90 deg, K = 1, C = 1, V = 2 m/s):
    turn      (turn_half >= A and both halves move) or (|rot_thirds| >= A and all three thirds move)
    stall     not turn, and (net_req < C or net speed < V)
    meander   otherwise
A coherent storm is labelled "coherent" by ``classify_track``; ``geometric_class`` applies the rules to any track.
"""
from __future__ import annotations

import numpy as np
from pyproj import Geod

_GEOD = Geod(ellps="WGS84")
PARAMS = {"A": 90.0, "K": 1.0, "C": 1.0, "V": 2.0}
CLASSES = ("coherent", "stall", "turn", "meander")


def _wrap(d):
    return (np.asarray(d, float) + 180) % 360 - 180


def track_geometry(lon, lat) -> tuple:
    """(turn_half, rot_thirds, smallest half displacement, smallest third displacement) of a track; the last two
    are in km per track-equivalent (the displacement of the segment divided by its share of the steps)."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    n = lon.size
    s = n - 1
    m = n // 2
    i3 = [0, int(round(s / 3)), int(round(2 * s / 3)), s]
    a, _, dist = _GEOD.inv(lon[[0, m]], lat[[0, m]], lon[[m, s]], lat[[m, s]])
    turn_half = abs(float(_wrap(a[1] - a[0])))
    half_ratio = dist / 1e3 / (np.array([m, s - m]) / s)
    a3, _, d3 = _GEOD.inv(lon[i3[:-1]], lat[i3[:-1]], lon[i3[1:]], lat[i3[1:]])
    t = _wrap(np.diff(a3))
    rot = float(t.sum()) if np.sign(t[0]) == np.sign(t[1]) else 0.0
    third_ratio = d3 / 1e3 / (np.diff(i3) / s)
    return turn_half, rot, float(half_ratio.min()), float(third_ratio.min())


def geometric_class(lon, lat, core_area_km2: float, net_length_km: float = None, net_speed_ms: float = None,
                    A: float = PARAMS["A"], K: float = PARAMS["K"], C: float = PARAMS["C"],
                    V: float = PARAMS["V"]) -> dict:
    """Class of a track by its geometry alone ("turn", "stall" or "meander"), with the quantities of the rules.

    lon, lat        hourly centroid positions of the analysed track (at least 4)
    core_area_km2   mean area of the storm core over the track
    net_length_km,  net displacement and net speed of the track; computed from the positions (hourly steps) when
    net_speed_ms    not given
    """
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    if net_length_km is None:
        net_length_km = float(_GEOD.inv(lon[0], lat[0], lon[-1], lat[-1])[2]) / 1e3
    if net_speed_ms is None:
        net_speed_ms = net_length_km * 1e3 / ((lon.size - 1) * 3600.0)
    turn_half, rot, half, third = track_geometry(lon, lat)
    r_eq = float(np.sqrt(core_area_km2 / np.pi))
    net_req = net_length_km / r_eq
    seg_half, seg_third = half / r_eq, third / r_eq
    turn = bool((turn_half >= A and seg_half >= K) or (abs(rot) >= A and seg_third >= K))
    stall = bool(not turn and (net_req < C or net_speed_ms < V))
    return {"geometric_class": "turn" if turn else "stall" if stall else "meander", "r_eq_km": r_eq,
            "net_req": float(net_req), "turn_half_deg": turn_half, "rot_thirds_deg": rot,
            "seg_ratio_half": float(seg_half), "seg_ratio_third": float(seg_third)}


def classify_track(lon, lat, core_area_km2: float, coherent: bool, net_length_km: float = None,
                   net_speed_ms: float = None, **params) -> dict:
    """``geometric_class`` plus ``linger_class``: "coherent" for a coherent storm, else the geometric class."""
    out = geometric_class(lon, lat, core_area_km2, net_length_km, net_speed_ms, **params)
    out["linger_class"] = "coherent" if coherent else out["geometric_class"]
    return out
