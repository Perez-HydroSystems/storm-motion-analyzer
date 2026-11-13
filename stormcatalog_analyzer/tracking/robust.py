"""Robust linking for the V6 overlap tracker: options that reduce identity hand-offs ("jumps").

``tracking.track`` (V6) links each storm object at hour t to the storm at t-1 with the largest overlap score
A/S_t + A/S_(t-1) (> ``ratio_threshold``). Every child of a split inherits the parent's id and a merged object
takes the id of its best-scoring parent, so a track id can change composition from one hour to the next and its
rain-weighted centroid then leaps (a "jump": an hourly step > 3x the track's median step and > 30 km).

``track_robust`` re-implements that linking with options; with the default ``LinkingConfig`` it returns exactly
the output of ``tracking.track`` (same ids, including its label bookkeeping). The options are:

predict          "none" (V6) | "centroid" | "pattern": the parent footprint is advected to hour t before the
                 overlap is measured (TITAN/TREC-style). "centroid" uses the median hourly centroid step over the
                 last ``history`` steps; "pattern" uses the displacement of the rain pattern inside the parent
                 footprint (masked normalized cross-correlation of log(1 + rain) between t-1 and t), falling back to
                 the centroid velocity for small objects or weak correlation.
max_speed_ms     distance gate: reject a link if the child's centroid lies farther than this speed x elapsed time
                 from the parent's predicted centroid (None = no gate).
mass_ratio       mass-continuity rule (as the "gated" baseline): reject a link if the rain volume changes by more
                 than this factor AND the centroid step is an outlier (> ``mass_step_factor`` x the parent's median
                 step and > ``min_step_km``); needs >= 2 previous steps (None = off).
split            "v6" (every child inherits the id) | "largest" (only the child carrying most of the parent's
                 advected footprint rain keeps the id; the others start new storms or take another parent).
merge            "v6" (best overlap score) | "volume" (parent contributing the largest rain volume to the child) |
                 "partition" (if several parents each have >= ``partition_min_share`` of their advected footprint
                 inside the child, the child is divided among them by a watershed of the rain field seeded with the
                 advected parent footprints; every parent keeps its own part, so the track follows the parent's
                 rain mass instead of the composite object; otherwise as "volume").
assignment       "greedy" (per child, as V6, then the split/merge rules) | "hungarian" (one-to-one optimal
                 assignment on cost = (1 - score/2) + w_dist * d/d_ref + w_vol * |ln(V_t / V_t-1)| / ln 10).
fix_labels       False (V6) | True: new storms at t = 1 are numbered after every id already used (V6 numbers them
                 from the largest raw label of hour 1, which can collide with ids inherited from hour 0 and fuse two
                 unrelated objects into one id).

``core_identification`` is an optional identification step: cores at a high threshold (V6 morphology), each grown
by a watershed of the rain field into the >= ``footprint_threshold`` area, so that positions follow persistent cores
and a light-rain bridge cannot fuse two cores into one object.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import relabel_sequential, watershed

from .identification import storm_area_identification

KM_DEG = 111.32


@dataclass
class LinkingConfig:
    ratio_threshold: float = 0.2
    dry_spell_time: int = 0
    predict: str = "none"                 # none | centroid | pattern
    history: int = 3                      # hourly steps used for the centroid velocity
    max_speed_ms: Optional[float] = None  # distance gate on the predicted position
    mass_ratio: Optional[float] = None    # mass-continuity rule
    mass_step_factor: float = 2.0
    min_step_km: float = 30.0
    split: str = "v6"                     # v6 | largest
    merge: str = "v6"                     # v6 | volume | partition
    partition_min_share: float = 0.3
    assignment: str = "greedy"            # greedy | hungarian
    w_dist: float = 1.0
    w_vol: float = 0.5
    dist_ref_km: float = 100.0
    fix_labels: bool = False
    pattern_min_pixels: int = 30
    pattern_min_corr: float = 0.3
    pattern_vmax_ms: float = 40.0

    def is_v6(self) -> bool:
        return (self.predict == "none" and self.max_speed_ms is None and self.mass_ratio is None
                and self.split == "v6" and self.merge == "v6" and self.assignment == "greedy" and not self.fix_labels)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_any(cls, x) -> "LinkingConfig":
        if x is None:
            return cls()
        if isinstance(x, cls):
            return x
        if isinstance(x, str):
            return cls(**PRESETS[x])
        return cls(**dict(x))


# named configurations evaluated in docs/robust_tracking.md (storm_tracking_method)
PRESETS = {
    "v6": {},
    "fixlabels": {"fix_labels": True},
    "predict_centroid": {"fix_labels": True, "predict": "centroid"},
    "predict_pattern": {"fix_labels": True, "predict": "pattern"},
    "split_largest": {"fix_labels": True, "split": "largest"},
    "merge_volume": {"fix_labels": True, "split": "largest", "merge": "volume"},
    "partition": {"fix_labels": True, "split": "largest", "merge": "partition"},
    "partition_pattern": {"fix_labels": True, "split": "largest", "merge": "partition", "predict": "pattern"},
    "hungarian_pattern": {"fix_labels": True, "assignment": "hungarian", "predict": "pattern", "max_speed_ms": 40.0},
    "gated_pattern": {"fix_labels": True, "predict": "pattern", "max_speed_ms": 40.0, "mass_ratio": 2.0},
    "robust": {"fix_labels": True, "predict": "pattern", "split": "largest", "merge": "partition",
               "max_speed_ms": 40.0, "mass_ratio": 2.0},
}


def preset(name: str, **kw) -> LinkingConfig:
    return LinkingConfig(**{**PRESETS[name], **kw})


# ------------------------------------------------------------------------------------------------ identification
def core_identification(rain: np.ndarray, core_threshold: float, footprint_threshold: float = 0.5,
                        morph_radius: int = 2) -> np.ndarray:
    """Cores identified as in V6 at ``core_threshold`` (morphology radius ``morph_radius``), each grown by a
    watershed of -rain into the contiguous area >= ``footprint_threshold``. Footprint rain not reached by any core
    is dropped. Returns (nt, ny, nx) object labels (unique within each hour)."""
    rain = np.asarray(rain, float)
    cores = storm_area_identification(rain, morph_radius, core_threshold).astype(np.int64)
    out = np.zeros(rain.shape, np.int64)
    for t in range(rain.shape[0]):
        if cores[t].max() == 0:
            continue
        out[t] = watershed(-rain[t], markers=cores[t], mask=rain[t] >= footprint_threshold, connectivity=2)
    return out


# ------------------------------------------------------------------------------------------------ helpers
def _cell_km(lat, lon, shape):
    """Cell size (km) along rows and columns at the domain's mean latitude (unit cells if lat/lon missing)."""
    if lat is None or lon is None:
        return 1.0, 1.0, 0.0
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    dlat = float(np.abs(np.diff(lat)).mean()) if lat.size > 1 else 1.0
    dlon = float(np.abs(np.diff(lon)).mean()) if lon.size > 1 else 1.0
    return dlat * KM_DEG, dlon * KM_DEG * np.cos(np.radians(lat.mean())), 1.0


def _stats(lab_flat, rain_flat, rows_flat, cols_flat, n):
    """Pixel count, rain volume and rain-weighted (row, col) centroid of labels 0..n-1."""
    cnt = np.bincount(lab_flat, minlength=n)
    v = np.bincount(lab_flat, weights=rain_flat, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.bincount(lab_flat, weights=rain_flat * rows_flat, minlength=n) / v
        c = np.bincount(lab_flat, weights=rain_flat * cols_flat, minlength=n) / v
    # objects with no rain (should not happen above a positive threshold): geometric centre
    bad = (cnt > 0) & ~(v > 0)
    if bad.any():
        rg = np.bincount(lab_flat, weights=rows_flat, minlength=n) / np.maximum(cnt, 1)
        cg = np.bincount(lab_flat, weights=cols_flat, minlength=n) / np.maximum(cnt, 1)
        r[bad], c[bad] = rg[bad], cg[bad]
    return cnt, v, r, c


def pattern_shift(r0, r1, mask0, max_disp: int, min_corr: float = 0.3):
    """Displacement (drow, dcol) in cells of the rain pattern inside ``mask0`` from ``r0`` to ``r1`` (masked
    normalized cross-correlation of log(1 + rain), search radius ``max_disp`` cells, parabolic sub-cell peak).
    Returns (drow, dcol, corr); NaN shift if the peak correlation is below ``min_corr``."""
    from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked
    rr, cc = np.nonzero(mask0)
    if rr.size == 0:
        return np.nan, np.nan, np.nan
    a, b = max(rr.min() - max_disp, 0), min(rr.max() + max_disp + 1, r0.shape[0])
    c, d = max(cc.min() - max_disp, 0), min(cc.max() + max_disp + 1, r0.shape[1])
    R0, R1 = np.log1p(r0[a:b, c:d]), np.log1p(r1[a:b, c:d])
    M0 = mask0[a:b, c:d]
    M1 = np.ones_like(M0)
    xc = cross_correlate_masked(R1, R0, M1, M0, mode="full", axes=(0, 1), overlap_ratio=0.3)
    xc = np.where(np.isfinite(xc), xc, -np.inf)
    off = np.array(R0.shape) - 1
    ii, jj = np.indices(xc.shape)
    xc[(np.abs(ii - off[0]) > max_disp) | (np.abs(jj - off[1]) > max_disp)] = -np.inf
    pk = np.unravel_index(np.argmax(xc), xc.shape)
    peak = float(xc[pk])
    if not np.isfinite(peak) or peak < min_corr:
        return np.nan, np.nan, peak
    sub = []
    for ax in (0, 1):
        lo, hi = list(pk), list(pk)
        lo[ax] -= 1
        hi[ax] += 1
        if lo[ax] < 0 or hi[ax] >= xc.shape[ax] or not (np.isfinite(xc[tuple(lo)]) and np.isfinite(xc[tuple(hi)])):
            sub.append(0.0)
        else:
            den = xc[tuple(lo)] - 2 * peak + xc[tuple(hi)]
            sub.append(0.0 if den >= 0 else float(0.5 * (xc[tuple(lo)] - xc[tuple(hi)]) / den))
    return pk[0] - off[0] + sub[0], pk[1] - off[1] + sub[1], peak


# ------------------------------------------------------------------------------------------------ tracker
def track_robust(ident: np.ndarray, rain: Optional[np.ndarray] = None, lat=None, lon=None,
                 cfg: Optional[LinkingConfig] = None, return_info: bool = False):
    """Link per-hour storm objects (``ident``: (nt, ny, nx) labels, unique within each hour) into tracks.

    ``rain`` (mm/h, same shape) weights centroids, volumes and the pattern velocity; ``lat``/``lon`` (1-D,
    ascending or descending) convert cells to km for the gates. With the default ``cfg`` the result equals
    ``tracking.track(ident, cfg.ratio_threshold, cfg.dry_spell_time)``.

    Returns the (nt, ny, nx) int64 track ids (sequential, 0 = none); with ``return_info`` also a dict with the
    per-hour link events (ids before the final sequential relabelling): ``merges`` [(t, id given to the child, n
    parents)], ``splits`` [(t, parent id, n children)], ``partitions`` [(t, n parts)], ``collisions`` [(t, id)] (a
    new storm numbered with an id already linked at t: the V6 t = 1 bookkeeping),
    ``gated`` (links rejected by the gates) and ``pattern_steps`` / ``pattern_fallbacks``.
    """
    cfg = LinkingConfig.from_any(cfg)
    g_all = np.asarray(ident).astype(np.int64)
    nt, ny, nx = g_all.shape
    rain = np.ones(g_all.shape) if rain is None else np.nan_to_num(np.asarray(rain, float), nan=0.0)
    km_r, km_c, has_geo = _cell_km(lat, lon, (ny, nx))
    rows_flat = np.repeat(np.arange(ny, dtype=float), nx)
    cols_flat = np.tile(np.arange(nx, dtype=float), ny)
    max_disp = int(np.ceil(cfg.pattern_vmax_ms * 3.6 / max(min(km_r, km_c), 1e-6))) + 1 if has_geo else 15
    result = g_all.copy()
    info = {"merges": [], "splits": [], "partitions": [], "collisions": [], "gated": 0, "pattern_steps": 0, "pattern_fallbacks": 0}
    hist: dict = {}            # track id -> list of (t, row, col, volume)

    def km(dr, dc):
        return np.hypot(np.asarray(dr) * km_r, np.asarray(dc) * km_c)

    def record(t):
        lab = result[t].ravel()
        n = int(lab.max()) + 1
        cnt, v, r, c = _stats(lab, rain[t].ravel(), rows_flat, cols_flat, n)
        for k in np.flatnonzero(cnt):
            if k:
                hist.setdefault(int(k), []).append((t, r[k], c[k], v[k]))

    def centroid_velocity(k, t_prev):
        h = [x for x in hist.get(k, []) if x[0] <= t_prev][-(cfg.history + 1):]
        if len(h) < 2:
            return 0.0, 0.0
        h = np.array(h)
        dt = np.diff(h[:, 0])
        ok = dt > 0
        if not ok.any():
            return 0.0, 0.0
        vr = np.diff(h[:, 1])[ok] / dt[ok]
        vc = np.diff(h[:, 2])[ok] / dt[ok]
        return float(np.median(vr)), float(np.median(vc))

    def median_step_km(k):
        h = hist.get(k, [])
        if len(h) < 3:
            return None
        h = np.array(h)
        return float(np.median(km(np.diff(h[:, 1]), np.diff(h[:, 2]))))

    record(0)
    for t in range(1, nt):
        g = g_all[t]
        cur = np.unique(g)
        cur = cur[cur != 0]
        if cfg.fix_labels:
            max_label = int(result[:t].max())
        else:
            max_label = int(result[1].max()) if t == 1 else int(result[:t].max())
        if cur.size == 0:
            continue
        gf = g.ravel()
        ng = int(gf.max()) + 1
        S_c, V_c, R_c, C_c = _stats(gf, rain[t].ravel(), rows_flat, cols_flat, ng)
        mapping: dict = {}                       # child label -> track id
        partition: dict = {}                     # child label -> {track id: predicted footprint mask (flat idx)}
        unmatched = [int(c) for c in cur]
        steps = range(1, cfg.dry_spell_time + 2) if t >= cfg.dry_spell_time + 1 else range(1, 2)
        for back in steps:
            if not unmatched:
                break
            prev = result[t - back]
            pf = prev.ravel()
            K = int(pf.max()) + 1
            if K <= 1:
                continue
            S_p, V_p, R_p, C_p = _stats(pf, rain[t - back].ravel(), rows_flat, cols_flat, K)
            pids = np.flatnonzero(S_p)
            pids = pids[pids > 0]
            # ---- advected parent footprints and their overlaps with the hour-t objects
            cand = []                            # (child, parent, score, A, contrib_prev, child_rain_in_pred, d_pred)
            pred_idx = {}
            if cfg.predict == "none":
                m = (gf > 0) & (pf > 0)
                if m.any():
                    code, cnt = np.unique(gf[m] * K + pf[m], return_counts=True)
                    cc_, pp_ = code // K, code % K
                    w_prev = rain[t - back].ravel()[m]
                    w_cur = rain[t].ravel()[m]
                    inv = np.searchsorted(code, gf[m] * K + pf[m])
                    vp = np.bincount(inv, weights=w_prev, minlength=code.size)
                    vc = np.bincount(inv, weights=w_cur, minlength=code.size)
                    for c, p, a, x, y in zip(cc_, pp_, cnt, vp, vc):
                        cand.append([int(c), int(p), a / S_c[c] + a / S_p[p], int(a), float(x), float(y), 0.0])
                shifts = {int(p): (0.0, 0.0) for p in pids}
            else:
                vidx = ndimage.value_indices(prev, ignore_value=0)
                shifts = {}
                for p in pids:
                    p = int(p)
                    rr, cc = vidx[p]
                    dr = dc = np.nan
                    if cfg.predict == "pattern" and rr.size >= cfg.pattern_min_pixels:
                        mask0 = prev == p
                        dr, dc, _ = pattern_shift(rain[t - back], rain[t], mask0, max_disp * back,
                                                  cfg.pattern_min_corr)
                        info["pattern_steps"] += 1
                        if not np.isfinite(dr):
                            info["pattern_fallbacks"] += 1
                    if not np.isfinite(dr):
                        vr, vc_ = centroid_velocity(p, t - back)
                        dr, dc = vr * back, vc_ * back
                    shifts[p] = (dr, dc)
                    r2 = rr + int(round(dr))
                    c2 = cc + int(round(dc))
                    ok = (r2 >= 0) & (r2 < ny) & (c2 >= 0) & (c2 < nx)
                    flat = r2[ok] * nx + c2[ok]
                    pred_idx[p] = flat
                    lab = gf[flat]
                    hit = lab > 0
                    if not hit.any():
                        continue
                    labs, inv = np.unique(lab[hit], return_inverse=True)
                    a = np.bincount(inv)
                    vp = np.bincount(inv, weights=rain[t - back][rr[ok][hit], cc[ok][hit]])
                    vc = np.bincount(inv, weights=rain[t].ravel()[flat[hit]])
                    for c, aa, x, y in zip(labs, a, vp, vc):
                        cand.append([int(c), p, aa / S_c[c] + aa / S_p[p], int(aa), float(x), float(y), 0.0])
            # ---- thresholds and gates
            keep = []
            for row in cand:
                c, p, s = row[0], row[1], row[2]
                if c not in unmatched or s <= cfg.ratio_threshold:
                    continue
                dr, dc = shifts.get(p, (0.0, 0.0))
                d_pred = float(km(R_c[c] - (R_p[p] + dr), C_c[c] - (C_p[p] + dc)))
                row[6] = d_pred
                # a parent that will claim its own part of a merged child (partition) is not gated on the
                # whole child's centroid or volume: its position will be that of its part
                if cfg.merge == "partition" and row[3] / max(S_p[p], 1) >= cfg.partition_min_share:
                    keep.append(row)
                    continue
                if cfg.max_speed_ms is not None and d_pred > cfg.max_speed_ms * 3.6 * back:
                    info["gated"] += 1
                    continue
                if cfg.mass_ratio is not None:
                    med = median_step_km(p)
                    if med is not None:
                        vr = V_c[c] / V_p[p] if V_p[p] > 0 else np.inf
                        d_step = float(km(R_c[c] - R_p[p], C_c[c] - C_p[p]))
                        if max(vr, 1 / vr if vr > 0 else np.inf) > cfg.mass_ratio and \
                                d_step > max(cfg.mass_step_factor * med, cfg.min_step_km):
                            info["gated"] += 1
                            continue
                keep.append(row)
            if not keep:
                continue
            # ---- link events (for diagnostics)
            kids, pars = {}, {}
            for row in keep:
                kids.setdefault(row[1], []).append(row)
                pars.setdefault(row[0], []).append(row)
            for p, rs in kids.items():
                if len(rs) > 1:
                    info["splits"].append((t, p, len(rs)))
            merged = {c: len(rs) for c, rs in pars.items() if len(rs) > 1}
            # ---- assignment
            if cfg.assignment == "hungarian":
                cs = sorted(pars)
                ps = sorted(kids)
                ci, pi = {c: i for i, c in enumerate(cs)}, {p: j for j, p in enumerate(ps)}
                cost = np.full((len(cs), len(ps)), 1e6)
                for c, p, s, a, x, y, d in keep:
                    vr = V_c[c] / V_p[p] if V_p[p] > 0 and V_c[c] > 0 else 10.0
                    cost[ci[c], pi[p]] = ((1 - s / 2) + cfg.w_dist * min(d / cfg.dist_ref_km, 1.0)
                                          + cfg.w_vol * min(abs(np.log(vr)) / np.log(10), 1.0))
                ri, cj = linear_sum_assignment(cost)
                for i, j in zip(ri, cj):
                    if cost[i, j] < 1e6:
                        mapping[cs[i]] = ps[j]
            else:
                if cfg.split == "largest":
                    for p, rs in kids.items():
                        if len(rs) > 1:
                            best = max(rs, key=lambda r: (r[5], r[3], -r[0]))
                            for r in rs:
                                if r is not best:
                                    pars[r[0]].remove(r)
                for c in sorted(pars):
                    rs = pars[c]
                    if not rs:
                        continue
                    if cfg.merge == "v6":
                        best = max(rs, key=lambda r: (r[2], -r[1]))
                    else:
                        best = max(rs, key=lambda r: (r[4], r[2], -r[1]))
                    if cfg.merge == "partition" and len(rs) > 1:
                        claim = [r for r in rs if r[3] / max(S_p[r[1]], 1) >= cfg.partition_min_share]
                        if len(claim) > 1:
                            partition[c] = {r[1]: (pred_idx.get(r[1]), r[1]) for r in claim}
                    mapping[c] = int(best[1])
            for c in list(mapping):
                if c in unmatched:
                    unmatched.remove(c)
            for c, n in merged.items():
                info["merges"].append((t, mapping.get(c, -1), n))
            # partitions need the parent footprints (unshifted when predict == "none")
            for c, parts in partition.items():
                for p in list(parts):
                    if parts[p][0] is None:
                        rr, cc = np.nonzero(prev == p)
                        parts[p] = (rr * nx + cc, p)
        # ---- new storms, ascending raw label
        linked_ids = set(mapping.values())
        for c in cur:
            c = int(c)
            if c not in mapping:
                max_label += 1
                mapping[c] = max_label
                if max_label in linked_ids:          # V6 numbering at t = 1 (fix_labels=False)
                    info["collisions"].append((t, max_label))
        lut = np.zeros(ng, dtype=np.int64)
        for c, v in mapping.items():
            lut[c] = v
        out_t = lut[g]
        # ---- partition merged objects among their parents
        for c, parts in partition.items():
            obj = g == c
            markers = np.zeros(g.shape, np.int64)
            owners = np.zeros(g.size, np.int64)
            seen = np.zeros(g.size, np.int64)
            for p, (flat, _) in parts.items():
                seen[flat] += 1
                owners[flat] = p
            mk = (owners.reshape(g.shape) > 0) & (seen.reshape(g.shape) == 1) & obj
            markers[mk] = owners.reshape(g.shape)[mk]
            present = np.unique(markers[markers > 0])
            if present.size < 2:
                continue
            rr, cc = np.nonzero(obj)
            a, b, c0, d = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
            ws = watershed(-rain[t][a:b, c0:d], markers=markers[a:b, c0:d], mask=obj[a:b, c0:d], connectivity=2)
            sub = out_t[a:b, c0:d]
            sub[obj[a:b, c0:d]] = ws[obj[a:b, c0:d]]
            info["partitions"].append((t, int(present.size)))
        result[t] = out_t
        record(t)
    seq, fwd, _ = relabel_sequential(result)
    seq = seq.astype(np.int64)
    if return_info:
        for key in ("merges", "splits", "collisions"):
            info[key] = [(e[0], int(fwd[e[1]]) if e[1] > 0 else -1, *e[2:]) for e in info[key]]
        return seq, info
    return seq


def track_field(rain: np.ndarray, lat=None, lon=None, linking=None, threshold: float = 0.5, morph_radius: int = 2,
                core_threshold: Optional[float] = None, return_info: bool = False):
    """Identification (V6, or cores grown into the footprint with ``core_threshold``) + robust linking."""
    rain = np.nan_to_num(np.asarray(rain, float), nan=0.0)
    if core_threshold is None:
        ident = storm_area_identification(rain, morph_radius, threshold)
    else:
        ident = core_identification(rain, core_threshold, threshold, morph_radius)
    return track_robust(ident, rain, lat, lon, LinkingConfig.from_any(linking), return_info=return_info)


def enlarged_tracker(linking=None, core_threshold: Optional[float] = None):
    """Tracker for ``stormcatalog_analyzer.enlarged.register_tracker``: ``fn(rain0, lat, lon, cfg) -> ids``."""
    lk = LinkingConfig.from_any(linking)

    def fn(rain0, lat, lon, cfg):
        return track_field(rain0, lat, lon, lk, threshold=cfg.rainfall_threshold_mmhr,
                           morph_radius=cfg.morph_radius_cells, core_threshold=core_threshold)
    return fn
