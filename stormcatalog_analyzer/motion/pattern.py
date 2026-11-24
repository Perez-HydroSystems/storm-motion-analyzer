"""Rain-pattern velocity of a storm, by storm-masked area correlation with quality control.

A storm has two motions. The rain mass is located each hour by the rain-weighted centroid of the storm footprint
(``motion.direction``, ``enlarged``); it moves when the rain features move and also when rain forms on one side of
the storm and decays on the other. The rain pattern is the displacement of the rain features themselves from one
hour to the next. This module measures the second motion and the difference between the two (the propagation).

Method (a storm-restricted form of the area-correlation / TREC method of radar nowcasting; Rinehart & Garvey 1978,
Li et al. 1995): for each pair of consecutive hours, the displacement of the rain pattern inside the storm footprint
is the shift that maximizes the masked normalized cross-correlation (Padfield 2012) between log(1 + rain) at t and
t+1, using the footprint at t and at t+1, each dilated by a few cells so that features moving in are not cut off.
The peak is refined to sub-cell precision with a separable parabolic fit.

Quality control per hourly step (rejected steps are left out, never filled in):
    corr     peak correlation below ``min_corr``
    speed    implied speed above ``vmax`` (m/s)
    reverse  step pointing more than ``reverse_deg`` away from the median direction of the other accepted steps
             while being faster than ``reverse_speed_ratio`` x their median speed (a jump, typically at a merge)

Storm-level outputs (``storm_pattern``): the rain-weighted vector-mean pattern velocity (u, v, speed, direction
counter-clockwise from east), the angular variance and the coherence test of the accepted steps, the propagation
velocity (net velocity of the rain-mass centroid minus the pattern velocity) with its components along and across
the pattern motion, and two optical-flow cross-checks (Farneback dense flow and pyramidal Lucas-Kanade) with their
angular disagreement. ``pattern_robust`` is True when both cross-checks agree with the pattern direction within
``ROBUST_MAX_DEG``.

Velocities are in m/s on a regular latitude/longitude grid (latitude ascending with the row index), converted with
111.32 km per degree of latitude and 111.32 km x cos(latitude) per degree of longitude at the mean latitude of the
storm.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import ndimage

from .direction import coherence_test

QC = dict(min_corr=0.3, vmax=40.0, reverse_deg=135.0, reverse_speed_ratio=1.5, dilate=3, overlap_ratio=0.3)
KM_DEG = 111.32
RMAX = 40.0             # mm/h mapped to 255 when rain is converted to 8-bit images for the optical-flow checks
ROBUST_MAX_DEG = 20.0   # largest disagreement of the two optical-flow directions for ``pattern_robust``
MIN_ACCEPTED = 3        # accepted steps needed for a storm-level pattern velocity


def _parabola(c_m, c_0, c_p):
    den = c_m - 2 * c_0 + c_p
    return 0.0 if den >= 0 or not np.isfinite(den) else float(0.5 * (c_m - c_p) / den)


def masked_shift(r0, r1, m0, m1, max_disp_px=45, dilate=3, overlap_ratio=0.3):
    """Displacement (drow, dcol) in cells of the rain pattern from ``r0`` to ``r1`` inside the masks, and the peak
    correlation. Returns (nan, nan, nan) if either mask is empty."""
    from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked
    if not m0.any() or not m1.any():
        return np.nan, np.nan, np.nan
    st = ndimage.generate_binary_structure(2, 2)
    m0 = ndimage.binary_dilation(m0, st, iterations=dilate)
    m1 = ndimage.binary_dilation(m1, st, iterations=dilate)
    rr, cc = np.nonzero(m0 | m1)
    a, b = max(rr.min() - max_disp_px, 0), min(rr.max() + max_disp_px + 1, r0.shape[0])
    c, d = max(cc.min() - max_disp_px, 0), min(cc.max() + max_disp_px + 1, r0.shape[1])
    R0, R1 = np.log1p(r0[a:b, c:d]), np.log1p(r1[a:b, c:d])
    M0, M1 = m0[a:b, c:d], m1[a:b, c:d]
    xc = cross_correlate_masked(R1, R0, M1, M0, mode="full", axes=(0, 1), overlap_ratio=overlap_ratio)
    xc = np.where(np.isfinite(xc), xc, -np.inf)
    off = np.array(R0.shape) - 1
    ii, jj = np.indices(xc.shape)                              # restrict the search to |shift| <= max_disp_px
    xc[(np.abs(ii - off[0]) > max_disp_px) | (np.abs(jj - off[1]) > max_disp_px)] = -np.inf
    pk = np.unravel_index(np.argmax(xc), xc.shape)
    peak = float(xc[pk])
    if not np.isfinite(peak):
        return np.nan, np.nan, np.nan
    sub = []
    for ax in (0, 1):
        lo, hi = list(pk), list(pk)
        lo[ax] -= 1
        hi[ax] += 1
        if lo[ax] < 0 or hi[ax] >= xc.shape[ax] or not (np.isfinite(xc[tuple(lo)]) and np.isfinite(xc[tuple(hi)])):
            sub.append(0.0)
        else:
            sub.append(_parabola(xc[tuple(lo)], peak, xc[tuple(hi)]))
    return pk[0] - off[0] + sub[0], pk[1] - off[1] + sub[1], peak


# ------------------------------------------------------------------------------- optical-flow cross-checks
def _img(r):
    return np.clip(np.log1p(r) / np.log1p(RMAX) * 255, 0, 255).astype(np.uint8)


def flow_farneback(r0, r1, mask, winsize=15, levels=3):
    """Rain-weighted mean displacement (dx, dy) in cells inside ``mask``: dense optical flow (Farneback 2003)."""
    import cv2
    f = cv2.calcOpticalFlowFarneback(_img(r0), _img(r1), None, 0.5, levels, winsize, 3, 5, 1.2, 0)
    w = np.where(mask, r0, 0.0)
    if w.sum() <= 0:
        return np.nan, np.nan
    return float((f[..., 0] * w).sum() / w.sum()), float((f[..., 1] * w).sum() / w.sum())


def flow_lucas_kanade(r0, r1, mask, max_corners=300):
    """Rain-weighted mean displacement (dx, dy) in cells of the corners tracked inside ``mask`` (pyramidal
    Lucas-Kanade on Shi-Tomasi corners). Falls back to the dense flow when fewer than 3 corners are tracked."""
    import cv2
    i0, i1 = _img(r0), _img(r1)
    pts = cv2.goodFeaturesToTrack(i0, maxCorners=max_corners, qualityLevel=0.01, minDistance=3,
                                  mask=mask.astype(np.uint8) * 255)
    if pts is None:
        return flow_farneback(r0, r1, mask)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(i0, i1, pts, None, winSize=(21, 21), maxLevel=3)
    ok = st.ravel() == 1
    if ok.sum() < 3:
        return flow_farneback(r0, r1, mask)
    p0, p1 = pts[ok, 0], p1[ok, 0]
    w = r0[np.clip(p0[:, 1].round().astype(int), 0, r0.shape[0] - 1),
           np.clip(p0[:, 0].round().astype(int), 0, r0.shape[1] - 1)]
    w = np.where(w > 0, w, 1e-6)
    d = p1 - p0
    return float((d[:, 0] * w).sum() / w.sum()), float((d[:, 1] * w).sum() / w.sum())


_FLOWS = {"farneback": flow_farneback, "lk": flow_lucas_kanade}


def _flow_steps(rain, masks, hours, kind):
    """Optical-flow displacement (drow, dcol) in cells of every hourly step."""
    fn = _FLOWS[kind]
    out = []
    for i in range(len(hours) - 1):
        dx, dy = fn(rain[hours[i]], rain[hours[i + 1]], masks[i])
        out.append((dy, dx))
    return np.array(out, float)


def _angle(a_deg, b_deg):
    return float(abs((a_deg - b_deg + 180) % 360 - 180))


# ------------------------------------------------------------------------------------------------- storm
def storm_pattern(rain, lon, lat, masks, hours, cen_lon, cen_lat, seed=None, qc: Optional[dict] = None,
                  cross_check: bool = True, n_sim: int = 10_000):
    """Rain-pattern velocity of one storm.

    rain       (nt, ny, nx) mm/h on a regular lat/lon grid (lat ascending with the row index), no NaN
    lon, lat   grid coordinates (degrees)
    masks      list of (ny, nx) bool footprints of the storm, one per analysed hour (same order as ``hours``)
    hours      analysed time indices into ``rain`` (consecutive)
    cen_lon,   rain-mass centroid of the storm at the same hours (rain-weighted mean position of the footprint);
    cen_lat    used for the propagation velocity
    seed       seed of the Monte Carlo coherence test of the pattern steps (None: not reproducible)
    qc         overrides of ``QC``
    Returns (summary dict, per-step DataFrame). The summary holds the pattern velocity only when at least
    ``MIN_ACCEPTED`` hourly steps pass the quality control.
    """
    q = {**QC, **(qc or {})}
    cen_lon, cen_lat = np.asarray(cen_lon, float), np.asarray(cen_lat, float)
    dlon, dlat = float(np.mean(np.diff(lon))), float(np.mean(np.diff(lat)))
    lat_c = float(np.mean(cen_lat))
    km_x, km_y = dlon * KM_DEG * np.cos(np.radians(lat_c)), dlat * KM_DEG
    max_px = int(np.ceil(q["vmax"] * 3.6 / min(abs(km_x), abs(km_y)))) + 2
    rows = []
    for i in range(len(hours) - 1):
        t0, t1 = hours[i], hours[i + 1]
        dr, dc, corr = masked_shift(rain[t0], rain[t1], masks[i], masks[i + 1], max_disp_px=max_px,
                                    dilate=q["dilate"], overlap_ratio=q["overlap_ratio"])
        u, v = dc * km_x / 3.6, dr * km_y / 3.6                 # m/s (east, north)
        w = float(rain[t0][masks[i]].sum())
        reason = ""
        if not np.isfinite(corr):
            reason = "empty"
        elif corr < q["min_corr"]:
            reason = "corr"
        elif np.hypot(u, v) > q["vmax"]:
            reason = "speed"
        rows.append(dict(step=i, t=t0, u=u, v=v, speed=float(np.hypot(u, v)), corr=corr, weight=w, reason=reason))
    s = pd.DataFrame(rows, columns=["step", "t", "u", "v", "speed", "corr", "weight", "reason"])
    ok = s["reason"] == ""
    for i in s.index[ok]:                                       # reversal / jump against the other accepted steps
        others = s[ok & (s.index != i)]
        if len(others) < 3:
            continue
        md = np.degrees(np.arctan2(np.median(others.v), np.median(others.u)))
        di = np.degrees(np.arctan2(s.at[i, "v"], s.at[i, "u"]))
        if abs((di - md + 180) % 360 - 180) > q["reverse_deg"] \
                and s.at[i, "speed"] > q["reverse_speed_ratio"] * others.speed.median():
            s.at[i, "reason"] = "reverse"
    s["accepted"] = s["reason"] == ""
    a = s[s.accepted]
    out = {"pattern_steps": len(s), "pattern_accepted": int(len(a)),
           "pattern_rejected_corr": int((s.reason == "corr").sum()),
           "pattern_rejected_speed": int((s.reason == "speed").sum()),
           "pattern_rejected_reverse": int((s.reason == "reverse").sum()),
           "pattern_median_corr": float(s["corr"].median()) if len(s) else np.nan}
    hrs = (len(hours) - 1) * 3600.0                             # net rain-mass velocity over the same hours
    cu = (cen_lon[-1] - cen_lon[0]) * KM_DEG * np.cos(np.radians(lat_c)) * 1000 / hrs
    cv = (cen_lat[-1] - cen_lat[0]) * KM_DEG * 1000 / hrs
    out.update(centroid_net_u=cu, centroid_net_v=cv)
    if len(a) >= MIN_ACCEPTED:
        w = a.weight.values / a.weight.sum()
        pu, pv = float((a.u * w).sum()), float((a.v * w).sum())
        L = a.speed.values
        av = 1 - np.hypot((a.u).sum(), (a.v).sum()) / L.sum() if L.sum() > 0 else np.nan
        # coherence test on the track rebuilt from the accepted steps (equal 1-h steps)
        x = np.r_[0, np.cumsum(a.u.values)] * 3.6 / (KM_DEG * np.cos(np.radians(lat_c)))
        y = np.r_[0, np.cumsum(a.v.values)] * 3.6 / KM_DEG
        coh = coherence_test(cen_lon[0] + x, cen_lat[0] + y, n_sim=n_sim, seed=seed)
        qu, qv = cu - pu, cv - pv
        sp = np.hypot(pu, pv)
        pdir = float(np.degrees(np.arctan2(pv, pu)))
        out.update(pattern_u=pu, pattern_v=pv, pattern_speed=float(sp), pattern_direction_deg=pdir,
                   pattern_av=float(av), pattern_coherent=bool(coh["coherent"]), pattern_p=coh["p_weighted"],
                   propagation_u=qu, propagation_v=qv, propagation_speed=float(np.hypot(qu, qv)),
                   propagation_along=float((qu * pu + qv * pv) / sp) if sp > 0 else np.nan,
                   propagation_cross=float((-qu * pv + qv * pu) / sp) if sp > 0 else np.nan,
                   centroid_vs_pattern_deg=_angle(np.degrees(np.arctan2(cv, cu)), pdir))
        if cross_check:
            for kind in ("farneback", "lk"):
                f = _flow_steps(rain, masks, hours, kind)[a.step.values]          # the same accepted steps
                fu = float(np.nansum(f[:, 1] * w)) * km_x / 3.6
                fv = float(np.nansum(f[:, 0] * w)) * km_y / 3.6
                fdir = float(np.degrees(np.arctan2(fv, fu)))
                out[f"{kind}_direction_deg"] = fdir
                out[f"{kind}_speed"] = float(np.hypot(fu, fv))
                out[f"{kind}_vs_masked_deg"] = _angle(fdir, pdir)
            out["pattern_robust"] = bool(max(out["farneback_vs_masked_deg"], out["lk_vs_masked_deg"])
                                         <= ROBUST_MAX_DEG)
    return out, s
