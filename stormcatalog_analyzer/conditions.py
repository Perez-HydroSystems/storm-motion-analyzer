"""Conditions of a tracked storm under which its rain-mass track may not follow the motion of the rain system.

The track of a storm is the hourly rain-weighted centroid of its cores. Four conditions make that centroid move for
reasons other than the motion of the rain:

    window      the core reaches the edge of the tracking window: it spans the window (two opposite edges in the same
                hour), holds a large share of its rain near the edge, or loses most of its area while it touches the
                edge (the storm leaves the window)
    two_pieces  the core is two pieces far apart, each with a relevant share of the rain: the centroid lies between
                them and follows the balance of rain between the pieces
    absorbed    rain that already existed outside the core in the previous hour joins the core and moves the centroid
    scattered   no piece of the core holds most of the rain

``hourly_conditions`` measures every analysed hour, ``summarize_conditions`` reduces the hours to one record per
storm, and ``condition_flags`` applies the thresholds (``PARAMS``). The conditions are descriptive: a flagged storm
is not removed, and a storm without flags is not guaranteed to be tracked correctly.

Distances use 111.32 km per degree of latitude and 111.32 km x cos(latitude) per degree of longitude at the mean
latitude of the track.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import ndimage

KM_DEG = 111.32
RAIN_THRESHOLD = 0.5    # mm/h: rain that "already existed" in the previous hour
NEAR_KM = 50.0          # "near the edge of the window"
PIECE_MIN = 0.10        # a second piece counts when it holds at least this share of the core rain
_S8 = np.ones((3, 3), bool)

PARAMS = dict(span_hours=1,        # window: hours touching two opposite edges
              near_share=0.20,     # window: share of the core rain within NEAR_KM of the edge, in some hour
              exit_drop=0.50,      # window: share of the core area lost while it touches the edge,
              exit_hours=3,        # ... touching it for at least this many hours,
              exit_near=0.05,      # ... with at least this share of its rain near the edge in some hour
              far_min_km=150.0,    # two pieces: farther apart than this distance and than the equivalent radius,
              far_hours=2,         # ... in at least this many hours
              other_pull_km=50.0,  # absorbed: rain taken from another area moves the centroid by this much
              big_share=0.60)      # scattered: median share of the core rain in its largest piece below this


def _centroid(w, lat, lon):
    s = w.sum()
    if s <= 0:
        return np.nan, np.nan
    return float((w.sum(axis=0) * lon).sum() / s), float((w.sum(axis=1) * lat).sum() / s)


def _km(a, b, kx):
    return float(np.hypot((a[0] - b[0]) * kx, (a[1] - b[1]) * KM_DEG))


def hourly_conditions(rain, mask, lat, lon) -> pd.DataFrame:
    """Measures of every analysed hour.

    rain   (nt, ny, nx) mm/h, no NaN, the analysed hours in order
    mask   (nt, ny, nx) bool, the core of the storm at each hour
    lat, lon   grid coordinates of the tracking window (degrees)
    """
    rain, mask = np.asarray(rain, float), np.asarray(mask, bool)
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    nt, ny, nx = mask.shape
    dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    area = np.repeat((dlat * KM_DEG * dlon * KM_DEG * np.cos(np.radians(lat)))[:, None], nx, axis=1)
    cen = [_centroid(np.where(mask[k], rain[k], 0.0), lat, lon) for k in range(nt)]
    kx = KM_DEG * np.cos(np.radians(float(np.nanmean([c[1] for c in cen]))))
    nr, nc = int(np.ceil(NEAR_KM / (dlat * KM_DEG))), int(np.ceil(NEAR_KM / (dlon * kx)))
    near = np.zeros((ny, nx), bool)
    near[:nr] = near[-nr:] = True
    near[:, :nc] = near[:, -nc:] = True
    rows = []
    for k in range(nt):
        m = mask[k]
        r = np.where(m, rain[k], 0.0)
        v = float(r.sum())
        ys, xs = np.nonzero(m.any(axis=1))[0], np.nonzero(m.any(axis=0))[0]
        lab, n = ndimage.label(m, structure=_S8)
        big, second, dist2 = (1.0 if n == 1 else 0.0), 0.0, np.nan
        if n > 1:
            vol = np.atleast_1d(ndimage.sum(rain[k], lab, range(1, n + 1)))
            o = np.argsort(vol)[::-1]
            big, second = float(vol[o[0]] / max(v, 1e-12)), float(vol[o[1]] / max(v, 1e-12))
            c2 = [_centroid(np.where(lab == o[i] + 1, rain[k], 0.0), lat, lon) for i in (0, 1)]
            dist2 = _km(c2[0], c2[1], kx)
        row = dict(hour=k, lon=cen[k][0], lat=cen[k][1], area_km2=float(area[m].sum()), volume=float((r * area).sum()),
                   touch_W=bool(m[:, 0].any()), touch_E=bool(m[:, -1].any()), touch_S=bool(m[0].any()), touch_N=bool(m[-1].any()),
                   near_share=float(r[near].sum() / max(v, 1e-12)),
                   width_frac=float((xs.max() - xs.min() + 1) / nx) if xs.size else 0.0,
                   height_frac=float((ys.max() - ys.min() + 1) / ny) if ys.size else 0.0,
                   pieces=int(n), big_share=big, second_share=second, second_km=dist2,
                   step_km=np.nan, keep_km=np.nan, gain_pull_km=np.nan, gain_share=np.nan, other_share=np.nan,
                   other_pull_km=np.nan, overlap=True)
        if k > 0:
            p = mask[k - 1]
            keep, gain = m & p, m & ~p
            row["step_km"] = _km(cen[k], cen[k - 1], kx)
            if keep.any() and rain[k][keep].sum() > 0 and rain[k - 1][keep].sum() > 0:
                ck = _centroid(np.where(keep, rain[k], 0.0), lat, lon)        # the rain that stays in the core
                cp = _centroid(np.where(keep, rain[k - 1], 0.0), lat, lon)
                other = gain & (rain[k - 1] >= RAIN_THRESHOLD)                 # gained rain that already existed
                co = _centroid(np.where(keep | other, rain[k], 0.0), lat, lon)
                row.update(keep_km=_km(ck, cp, kx), gain_pull_km=_km(cen[k], ck, kx),
                           gain_share=float(rain[k][gain].sum() / max(v, 1e-12)),
                           other_share=float(rain[k][other].sum() / max(v, 1e-12)), other_pull_km=_km(co, ck, kx))
            else:
                row["overlap"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_conditions(h: pd.DataFrame, margin_edge: Optional[dict] = None) -> dict:
    """One record per storm from its hourly table. ``margin_edge``: for "W", "E", "S", "N", True when that edge of
    the window is the margin around the tile (a larger window would move it) and False when it is the extent of the
    data; all True when not given."""
    margin_edge = margin_edge or dict(W=True, E=True, S=True, N=True)
    t = {e: h[f"touch_{e}"].values for e in "WESN"}
    any_touch = t["W"] | t["E"] | t["S"] | t["N"]
    span = (t["W"] & t["E"]) | (t["S"] & t["N"])
    movable = np.zeros(len(h), bool)
    for e in "WESN":
        if margin_edge[e]:
            movable |= t[e]
    a = h.area_km2.values
    exit_drop = float(np.max(np.where(any_touch, 1 - a / np.maximum(np.maximum.accumulate(a), 1e-9), 0.0)))
    r_eq = float(np.sqrt(np.mean(a) / np.pi))
    second = h.second_share >= PIECE_MIN
    return dict(touch_hours=int(any_touch.sum()), touch_margin_hours=int(movable.sum()), span_hours=int(span.sum()),
                near_max=float(h.near_share.max()), exit_drop=exit_drop,
                width_max=float(h.width_frac.max()), height_max=float(h.height_frac.max()),
                pieces_median=float(h.pieces.median()), big_share_median=float(h.big_share.median()),
                second_km_max=float(np.nanmax(np.where(second, h.second_km, np.nan))) if second.any() else 0.0,
                far_hours=int((second & (h.second_km >= max(PARAMS["far_min_km"], r_eq))).sum()),
                gain_pull_max_km=float(h.gain_pull_km.max()), other_pull_max_km=float(h.other_pull_km.max()),
                other_share_max=float(h.other_share.max()), step_max_km=float(h.step_km.max()),
                no_overlap_steps=int((~h.overlap).sum()))


def condition_flags(s: dict, params: Optional[dict] = None) -> dict:
    """Flags (not exclusive) of one storm from its summary, and the first condition that applies, in the order
    window, two_pieces, absorbed, scattered ("none" when no flag is set)."""
    p = {**PARAMS, **(params or {})}
    window = bool(s["span_hours"] >= p["span_hours"] or s["near_max"] >= p["near_share"]
                  or (s["exit_drop"] >= p["exit_drop"] and s["touch_hours"] >= p["exit_hours"] and s["near_max"] >= p["exit_near"]))
    two = bool(s["far_hours"] >= p["far_hours"])
    absorbed = bool(np.nan_to_num(s["other_pull_max_km"]) >= p["other_pull_km"])
    scattered = bool(s["big_share_median"] < p["big_share"])
    first = "window" if window else "two_pieces" if two else "absorbed" if absorbed else "scattered" if scattered else "none"
    return dict(flag_window=window, flag_two_pieces=two, flag_absorbed=absorbed, flag_scattered=scattered, track_condition=first)


def track_conditions(rain, mask, lat, lon, margin_edge: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """Summary and flags of one storm (``hourly_conditions`` -> ``summarize_conditions`` -> ``condition_flags``)."""
    s = summarize_conditions(hourly_conditions(rain, mask, lat, lon), margin_edge)
    return {**s, **condition_flags(s, params)}
