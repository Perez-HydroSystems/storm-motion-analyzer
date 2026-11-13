"""Enlarged-window tracking: screen by tile, track on the tile plus a margin.

RainyDay screens storms tile by tile: an event is a 24-h window (``cattime``) and the position of the screening
circle (the control area, e.g. 10,000 km^2) where the window's rain over the circle is largest. The V6 tracker then
follows the storm only inside the 5-degree tile, so a large share of most storms' rain falls outside the tracked
domain and their tracks are clipped at the tile edge. This module tracks the same event on the tile enlarged by a
margin, reading the hourly rain field directly (e.g. AORC), and selects the event's storm by the screening circle:

    event    the RainyDay window (24 h) and the circle placed at the event location (``xlocation``/``ylocation``:
             column/row of the lower-left cell of the circle mask's bounding box in the tile grid, latitude
             ascending; the mask is the catalog's ``gridmask`` trimmed to its bounding box, fractional weights)
    field    hourly rain on [tile + margin], NaN = no data (outside the AORC coverage); the valid-data fraction is
             recorded and NaN is treated as dry for the tracking
    tracker  the V6 identification and overlap tracking (``tracking.identification``/``tracking.tracking``),
             V6 settings (threshold 0.5 mm/h, morphological radius 2, overlap 0.2, no dry spell)
    storm    the track with the largest rain volume inside the circle over the 24-h window (its contiguous run of
             hours with the most in-circle volume), trimmed like V6 to the first/last hour whose area exceeds
             ``min_area_fraction`` of the TILE area (an absolute area, so the rule does not change with the margin);
             retained if at least ``min_duration_hr`` hours exceed that area
    window   the V6 analysed window: at most ``window_hr`` consecutive positions with the largest summed hourly
             in-ellipse mean rain (``select_analysis_window``), at least ``min_positions``
    motion   the V6 descriptors (``pipeline.compute_storm_trajectory``: geodesic mean bearing and angular variance,
             path speed, length-weighted Rayleigh coherence test with the seed key ``"<tile>/<event file>"``) plus
             the net (end-to-end) speed

Flags recorded per event: share of the storm's rain outside the tile, valid-data fraction of the domain and of the
storm's neighbourhood, hours in which the footprint touches the field border or no-data cells, jumps (hourly steps
> ``jump_factor`` x the median step and > ``jump_min_km``), the share of the circle's rain carried by the selected
storm, and the comparison with the V6 result on the tile alone (same field, V6 rule): distance between the tracks
and the share of the tile storm's rain covered by the enlarged footprint.

Cross-tile duplicates (``assign_system_ids``): the same system appears in the catalogs of several tiles. Two events
of different tiles are the same system when their storm tracks share >= 50% of the hours of the shorter one and,
over the shared hours, their footprints (coarse cells, ``footprint_res_deg``) overlap by >= 50% of the smaller one;
connected events get one ``system_id``.
"""
from __future__ import annotations

import contextlib
import io as _io
import re
import warnings
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from pyproj import Geod, Transformer
from scipy import ndimage

from .config import CoherenceConfig
from .motion.direction import bearing_to_math
from .pipeline import compute_storm_trajectory
from .tracking.dataset import storm_tracking_features
from .tracking.identification import storm_area_identification
from .tracking.tracking import track_fast as overlap_track

_GEOD = Geod(ellps="WGS84")
_TO_PRJ = Transformer.from_crs(4326, 2163)        # same transform (and axis order) as tracking.catalog
EPS = 1e-3
AORC_PATTERN = "AORC.{day}.preciptemp.nc"          # file D holds 01Z of D .. 00Z of D+1
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA",
           9: "SON", 10: "SON", 11: "SON"}


@dataclass
class EnlargedConfig:
    margin_deg: float = 5.0                 # degrees added on every side of the tile (clipped to the data extent)
    rainfall_threshold_mmhr: float = 0.5    # V6 tracker settings
    morph_radius_cells: int = 2
    overlap_ratio_threshold: float = 0.2
    dry_spell_hr: int = 0
    min_area_fraction: float = 0.05         # trimming area = this fraction of the TILE area (km^2)
    min_duration_hr: int = 5                # hours above the trimming area needed to retain the storm
    window_hr: int = 12                     # analysed window: at most this many positions ...
    min_positions: int = 5                  # ... and at least this many
    selection: str = "circle"               # "circle" (default) | "v6" (longest duration, as the tile pipeline)
    jump_factor: float = 3.0                # jump: step > jump_factor x median step and > jump_min_km
    jump_min_km: float = 30.0
    tracker: str = "v6"                     # key of TRACKERS: "v6" (V6 overlap tracking) | "gated" | registered
    jump_rule: str = "none"                 # "none" (flag jumps only) | "split" (longest jump-free segment)
    max_link_speed_ms: float = 40.0         # "gated": reject links implying a centroid speed above this
    mass_ratio: float = 2.0                 # "gated": reject links where the rain volume changes by > this factor
    mass_step_factor: float = 2.0           # ... and the centroid moves > this x the track's median step
    neighbourhood_cells: int = 15           # dilation of the footprint for the storm's valid-data fraction
    footprint_res_deg: float = 0.25         # coarse grid of the footprints used for duplicate detection
    coherence: CoherenceConfig = field(default_factory=CoherenceConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def variant(cls, name: str, **kw) -> "EnlargedConfig":
        """Jump-handling variants: v0 = V6 linking, jumps flagged; v1 = V6 linking, analyse the longest jump-free
        segment; v2 = gated linking (speed gate + mass continuity)."""
        base = {"v0": {}, "v1": {"jump_rule": "split"}, "v2": {"tracker": "gated"}}[name]
        return cls(**{**base, **kw})


# ------------------------------------------------------------------------------------------------ inputs
@dataclass
class CatalogEvent:
    """One RainyDay catalog event: its 24-h window and screening-circle placement on the tile grid."""
    name: str                     # storm file name, e.g. B7.nc_storm_400_20080612.nc
    storm_id: int                 # RainyDay storm number (1 = smallest of the catalog, n = largest)
    rank: int                     # 1 = largest event of the catalog
    times: np.ndarray             # (24,) datetime64, hour-ending stamps of the window
    lat: np.ndarray               # tile grid (ascending), cell centres
    lon: np.ndarray
    xloc: int                     # column of the circle mask's lower-left bounding-box cell
    yloc: int                     # row (latitude ascending)
    circle: np.ndarray            # (h, w) trimmed mask weights (fraction of each cell inside the circle)
    basin_rain_mm: float = np.nan  # RainyDay basinrainfall: mean 24-h depth over the circle (mm)
    rain: Optional[np.ndarray] = None   # (24, ny, nx) catalog rain on the tile (NaN = no data), if loaded

    @property
    def tile_bounds(self) -> tuple:
        """(west, south, east, north) cell edges of the tile grid."""
        dlon, dlat = float(np.mean(np.diff(self.lon))), float(np.mean(np.diff(self.lat)))
        return (self.lon[0] - dlon / 2, self.lat[0] - dlat / 2, self.lon[-1] + dlon / 2, self.lat[-1] + dlat / 2)

    @property
    def circle_center(self) -> tuple:
        """(lon, lat) of the weighted centre of the circle mask placed at the event location."""
        h, w = self.circle.shape
        wsum = self.circle.sum()
        cy = float((np.arange(h)[:, None] * self.circle).sum() / wsum)
        cx = float((np.arange(w)[None, :] * self.circle).sum() / wsum)
        dlon, dlat = float(np.mean(np.diff(self.lon))), float(np.mean(np.diff(self.lat)))
        return self.lon[0] + (self.xloc + cx) * dlon, self.lat[0] + (self.yloc + cy) * dlat

    def circle_on_tile(self) -> np.ndarray:
        """(ny, nx) circle weights on the tile grid."""
        out = np.zeros((len(self.lat), len(self.lon)))
        h, w = self.circle.shape
        out[self.yloc:self.yloc + h, self.xloc:self.xloc + w] = self.circle[:len(self.lat) - self.yloc,
                                                                            :len(self.lon) - self.xloc]
        return out


def storm_number(path) -> int:
    m = re.search(r"_storm_(\d+)", Path(path).name)
    return int(m.group(1)) if m else -1


def read_catalog_event(path, load_rain: bool = False, var_name: str = "rain") -> CatalogEvent:
    """Read one RainyDay storm file (any catalog written by RainyDay: every file carries the whole catalog's
    ``cattime``, ``xlocation``, ``ylocation``, ``basinrainfall`` and the circle ``gridmask``)."""
    import xarray as xr
    path = Path(path)
    with xr.open_dataset(path) as ds:
        times = np.asarray(ds["time"].values)
        cattime = np.asarray(ds["cattime"].values)
        idx = np.flatnonzero((cattime[:, 0] == times[0]) & (cattime[:, -1] == times[-1]))
        if idx.size != 1:
            raise ValueError(f"{path.name}: window not found once in cattime ({idx.size} matches)")
        i = int(idx[0])
        gm = np.nan_to_num(np.asarray(ds["gridmask"].values, float))
        rows, cols = np.flatnonzero(gm.sum(1) > 0), np.flatnonzero(gm.sum(0) > 0)
        circle = gm[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        lat = np.asarray(ds["latitude"].values, float)
        lon = np.asarray(ds["longitude"].values, float)
        rain = None
        if load_rain:
            rain = np.asarray(ds[var_name].values, float)
            rain = np.where(np.isfinite(rain) & (rain >= 0), rain, np.nan)
        n = cattime.shape[0]
        sid = storm_number(path)
        ev = CatalogEvent(name=path.name, storm_id=sid, rank=n - i, times=times, lat=lat, lon=lon,
                          xloc=int(ds["xlocation"].values[i]), yloc=int(ds["ylocation"].values[i]), circle=circle,
                          basin_rain_mm=float(ds["basinrainfall"].values[i]), rain=rain)
    if lat[0] > lat[-1]:
        raise ValueError(f"{path.name}: latitude must be ascending")
    return ev


@dataclass
class Field:
    """Hourly rain (mm/h) on a regular grid; NaN = no data."""
    rain: np.ndarray       # (nt, ny, nx)
    lat: np.ndarray        # (ny,) ascending
    lon: np.ndarray        # (nx,)
    times: np.ndarray      # (nt,)
    missing_hours: int = 0  # hours with no file / all no-data

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.rain)

    def crop(self, west, south, east, north) -> "Field":
        r = (self.lat >= south - EPS) & (self.lat <= north + EPS)
        c = (self.lon >= west - EPS) & (self.lon <= east + EPS)
        return Field(self.rain[:, r][:, :, c], self.lat[r], self.lon[c], self.times, self.missing_hours)


def read_aorc_field(aorc_dir, times, west, south, east, north, pattern: str = AORC_PATTERN) -> Field:
    """Hourly RAINRATE for ``times`` on [west, east] x [south, north] (clipped to the data extent) from daily AORC
    files (file D holds 01Z of D .. 00Z of D+1). Negative values are no data (NaN). A missing file gives NaN hours."""
    import xarray as xr
    t = pd.to_datetime(np.asarray(times))
    day = [(x - pd.Timedelta(hours=1)).strftime("%Y%m%d") for x in t]
    out, lat, lon, miss = [None] * len(t), None, None, 0
    for d in dict.fromkeys(day):
        f = Path(aorc_dir) / pattern.format(day=d)
        sel_i = [i for i, x in enumerate(day) if x == d]
        if not f.exists():
            miss += len(sel_i)
            continue
        with xr.open_dataset(f) as ds:
            da = ds["RAINRATE"].sel(latitude=slice(south - EPS, north + EPS), longitude=slice(west - EPS, east + EPS))
            ft = pd.to_datetime(da["time"].values)
            pos = {x: k for k, x in enumerate(ft)}
            have = [i for i in sel_i if t[i] in pos]
            if have:
                vals = np.asarray(da.isel(time=[pos[t[i]] for i in have]).values, float)
                for j, i in enumerate(have):
                    out[i] = vals[j]
            miss += len(sel_i) - len(have)
            lat, lon = da["latitude"].values.astype(float), da["longitude"].values.astype(float)
    if lat is None:
        raise FileNotFoundError(f"no AORC file for {t[0]} .. {t[-1]} in {aorc_dir}")
    shape = (len(lat), len(lon))
    rain = np.stack([o if o is not None else np.full(shape, np.nan) for o in out])
    rain = np.where(np.isfinite(rain) & (rain >= 0), rain, np.nan)
    miss += int(sum(1 for k in range(len(t)) if out[k] is not None and not np.isfinite(out[k]).any()))
    return Field(rain, lat, lon, np.asarray(times), miss)


def field_from_catalog(ev: CatalogEvent) -> Field:
    """The catalog's own tile field (margin 0, no extra reading)."""
    if ev.rain is None:
        raise ValueError("event loaded without rain (read_catalog_event(load_rain=True))")
    return Field(ev.rain, ev.lat, ev.lon, ev.times)


# ------------------------------------------------------------------------------------------------ tracking
def pixel_area_km2(lat, lon) -> np.ndarray:
    """Cell area (km^2) as tracking.dataset.compute_pixel_area: cos(lat) * 111^2 * dlat * dlon."""
    dlat = np.abs(np.diff(lat)).mean()
    dlon = np.abs(np.diff(lon)).mean()
    return np.cos(np.radians(lat))[:, None] * 111 * 111 * dlat * dlon * np.ones((1, len(lon)))


def _track_v6(rain0, lat, lon, cfg: EnlargedConfig) -> np.ndarray:
    """V6 identification + overlap tracking (identical output to the V6 pipeline)."""
    ident = storm_area_identification(rain0, cfg.morph_radius_cells, cfg.rainfall_threshold_mmhr)
    return np.asarray(overlap_track(ident, ratio_threshold=cfg.overlap_ratio_threshold,
                                    dry_spell_time=cfg.dry_spell_hr), dtype=np.int64)


def _object_stats(lab, rain, lat, lon, n):
    """Rain volume (sum of rain, cell units) and rain-weighted centroid (lon, lat) of labels 0..n-1."""
    w = rain.ravel()
    lf = lab.ravel()
    v = np.bincount(lf, weights=w, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.bincount(lf, weights=w * np.broadcast_to(lon[None, :], lab.shape).ravel(), minlength=n) / v
        y = np.bincount(lf, weights=w * np.broadcast_to(lat[:, None], lab.shape).ravel(), minlength=n) / v
    return v, x, y


def _km(x0, y0, x1, y1):
    """Equirectangular distance (km), adequate for hourly steps."""
    return np.hypot((x1 - x0) * 111.32 * np.cos(np.radians(0.5 * (y0 + y1))), (y1 - y0) * 111.32)


def track_gated(rain0, lat, lon, cfg: EnlargedConfig) -> np.ndarray:
    """V6 identification and overlap linking (as ``tracking.track_fast``) with two extra conditions on every link
    of a current object to a previous storm (the best-overlap candidate that passes is used; none -> new storm):
      speed gate      centroid distance / elapsed time <= ``max_link_speed_ms``
      mass continuity not (rain volume ratio > ``mass_ratio`` and centroid step > ``mass_step_factor`` x the
                      storm's median previous step [at least ``jump_min_km``]; needs >= 2 previous steps)
    Merges that move the centroid abruptly therefore start a new storm instead of continuing an identity."""
    from skimage.segmentation import relabel_sequential
    grown = storm_area_identification(rain0, cfg.morph_radius_cells, cfg.rainfall_threshold_mmhr)
    g_int = np.asarray(grown).astype(np.int64)
    result = g_int.copy()
    nt = g_int.shape[0]
    hist: dict = {}                                   # label -> list of (t, lon, lat)
    n0 = int(result[0].max()) + 1
    _, x0, y0 = _object_stats(result[0], rain0[0], lat, lon, n0)
    for k in np.unique(result[0]):
        if k:
            hist[int(k)] = [(0, x0[k], y0[k])]
    dry = cfg.dry_spell_hr
    for t in range(1, nt):
        g = g_int[t]
        cur = np.unique(g)
        cur = cur[cur != 0]
        max_label = int(result[1].max()) if t == 1 else int(result[:t].max())
        if cur.size == 0:
            continue
        gf = g.ravel()
        ng = int(gf.max()) + 1
        curr_size = np.bincount(gf, minlength=ng)
        cv, cx, cy = _object_stats(g, rain0[t], lat, lon, ng)
        mapping, unmatched = {}, set(int(c) for c in cur)
        steps = range(1, dry + 2) if t >= dry + 1 else range(1, 2)
        for back in steps:
            if not unmatched:
                break
            prev = result[t - back]
            pf = prev.ravel()
            K = int(pf.max()) + 1
            prev_size = np.bincount(pf, minlength=K)
            pv, px, py = _object_stats(prev, rain0[t - back], lat, lon, K)
            m = (gf > 0) & (pf > 0)
            if not m.any():
                continue
            code, cnt = np.unique(gf[m] * K + pf[m], return_counts=True)
            cc, pp = code // K, code % K
            ratio = cnt / curr_size[cc] + cnt / prev_size[pp]
            order = np.lexsort((pp, -ratio, cc))
            for c, p, r in zip(cc[order], pp[order], ratio[order]):
                c, p = int(c), int(p)
                if c not in unmatched or r <= cfg.overlap_ratio_threshold:
                    continue
                d = float(_km(px[p], py[p], cx[c], cy[c]))
                if d / (back * 3.6) > cfg.max_link_speed_ms:
                    continue
                h = hist.get(p, [])
                if len(h) >= 3:
                    hs = np.array(h)
                    med = float(np.median(_km(hs[:-1, 1], hs[:-1, 2], hs[1:, 1], hs[1:, 2])))
                    vr = cv[c] / pv[p] if pv[p] > 0 else np.inf
                    if max(vr, 1 / vr if vr > 0 else np.inf) > cfg.mass_ratio and \
                            d > max(cfg.mass_step_factor * med, cfg.jump_min_km):
                        continue
                mapping[c] = p
                unmatched.discard(c)
        for c in cur:
            c = int(c)
            if c not in mapping:
                max_label += 1
                mapping[c] = max_label
        lut = np.zeros(ng, dtype=np.int64)
        for c, v in mapping.items():
            lut[c] = v
        result[t] = lut[g]
        nr = int(result[t].max()) + 1
        _, rx, ry = _object_stats(result[t], rain0[t], lat, lon, nr)
        for k in np.unique(result[t]):
            if k:
                hist.setdefault(int(k), []).append((t, rx[k], ry[k]))
    return np.asarray(relabel_sequential(result)[0], dtype=np.int64)


TRACKERS = {"v6": _track_v6, "gated": track_gated}


def register_tracker(name: str, fn) -> None:
    """Plug in another tracker: ``fn(rain0, lat, lon, cfg) -> (nt, ny, nx) int track ids`` (0 = none)."""
    TRACKERS[name] = fn


def track_field(rain0: np.ndarray, cfg: EnlargedConfig, lat=None, lon=None) -> np.ndarray:
    """Track a (nt, ny, nx) field (no NaN) with the configured tracker; returns int track ids (0 = none)."""
    return TRACKERS[cfg.tracker](rain0, lat, lon, cfg)


def jump_segments(lon, lat, cfg: EnlargedConfig) -> list:
    """Split consecutive positions at jumps (step > jump_factor x median step and > jump_min_km); returns
    (start, end) index pairs, end exclusive."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    if lon.size < 3:
        return [(0, lon.size)]
    _, _, st = _GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    st = np.asarray(st) / 1000.0
    med = float(np.median(st))
    j = np.flatnonzero((st > cfg.jump_factor * med) & (st > cfg.jump_min_km))
    cuts = np.r_[0, j + 1, lon.size]
    return [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:])]


def track_table(labels: np.ndarray, rain0: np.ndarray, area: np.ndarray,
                weight: Optional[np.ndarray] = None, lat=None, lon=None) -> pd.DataFrame:
    """Per (track, hour): pixel count, area (km^2), rain volume (mm km^2) and, with ``weight`` (ny, nx), the
    weighted volume (e.g. inside the screening circle)."""
    rows = []
    for t in range(labels.shape[0]):
        lab = labels[t].ravel()
        if not lab.any():
            continue
        n = int(lab.max()) + 1
        vol = rain0[t].ravel() * area.ravel()
        cnt = np.bincount(lab, minlength=n)
        a = np.bincount(lab, weights=area.ravel(), minlength=n)
        v = np.bincount(lab, weights=vol, minlength=n)
        wv = np.bincount(lab, weights=vol * weight.ravel(), minlength=n) if weight is not None else np.zeros(n)
        ids = np.flatnonzero(cnt)
        ids = ids[ids > 0]
        d = {"track_id": ids, "t": t, "pixels": cnt[ids], "area_km2": a[ids], "volume": v[ids],
             "circle_volume": wv[ids]}
        if lat is not None:
            _, x, y = _object_stats(labels[t], rain0[t], lat, lon, n)
            d.update(lon=x[ids], lat=y[ids])
        rows.append(pd.DataFrame(d))
    if not rows:
        return pd.DataFrame(columns=["track_id", "t", "pixels", "area_km2", "volume", "circle_volume"])
    return pd.concat(rows, ignore_index=True)


def _runs(ts: np.ndarray) -> list:
    ts = np.sort(np.asarray(ts, int))
    brk = np.r_[0, np.flatnonzero(np.diff(ts) != 1) + 1, len(ts)]
    return [ts[a:b] for a, b in zip(brk[:-1], brk[1:]) if b > a]


def select_circle_storm(tab: pd.DataFrame) -> Optional[tuple]:
    """Track id and hours (contiguous run) with the largest rain volume inside the screening circle."""
    if tab.empty or tab["circle_volume"].max() <= 0:
        return None
    tot = tab.groupby("track_id")["circle_volume"].sum()
    k = int(tot.idxmax())
    g = tab[tab["track_id"] == k].set_index("t")
    run = max(_runs(g.index.values), key=lambda r: g.loc[r, "circle_volume"].sum())
    return k, run


def select_longest_storm(tab: pd.DataFrame, min_pixels: float) -> Optional[tuple]:
    """V6 rule (tracking.single_event_selection): the track with the most hours above ``min_pixels`` pixels
    (ties: lowest id), all its hours."""
    if tab.empty:
        return None
    dur = tab.assign(big=tab["pixels"] > min_pixels).groupby("track_id")["big"].sum()
    k = int(dur.index[np.argmax(dur.values)])
    return k, np.sort(tab.loc[tab["track_id"] == k, "t"].values.astype(int))


def trim_hours(tab_k: pd.DataFrame, hours: np.ndarray, min_area_km2: float) -> tuple:
    """V6 trimming (dataset.continuos_storm): from the first to the last hour whose area exceeds ``min_area_km2``.
    Returns (trimmed hours, number of hours above the area)."""
    a = tab_k.set_index("t").reindex(hours)["area_km2"].fillna(0).values
    big = np.flatnonzero(a > min_area_km2)
    if big.size == 0:
        return hours[:0], 0
    return hours[big[0]:big[-1] + 1], int(big.size)


def projected_grid(lat, lon) -> tuple:
    """EPSG:2163 x, y (m) of every cell, as tracking.catalog.compute_projected_coordinate_system (vectorised)."""
    lon2d, lat2d = np.meshgrid(lon, lat)
    x, y = _TO_PRJ.transform(lat2d, lon2d)
    return np.asarray(x), np.asarray(y)


def v6_descriptors(rain0, labels, k, hours, lat, lon, times, seed_key, cfg: EnlargedConfig, prj=None) -> Optional[dict]:
    """V6 descriptors of track ``k`` over ``hours`` (contiguous): the package's own V6 functions
    (storm_tracking_features + compute_storm_trajectory) on the storm-masked field."""
    import xarray as xr
    a, b = int(hours[0]), int(hours[-1]) + 1
    lon2d, lat2d = np.meshgrid(lon, lat)
    lonp, latp = prj if prj is not None else projected_grid(lat, lon)
    area = pixel_area_km2(lat, lon)
    sel = np.where(labels[a:b] == k, rain0[a:b], 0.0)
    storm = {"selected_storm": sel, "lon_array": lon2d, "lat_array": lat2d, "lon_prj_array": lonp,
             "lat_prj_array": latp, "selected_storm_time_steps": xr.DataArray(np.asarray(times)[a:b]),
             "selected_storm_area": (np.where(labels[a:b] == k, area, 0.0)).sum(axis=(1, 2))}
    with contextlib.redirect_stdout(_io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        storm_tracking_features(storm, ellipse_fit="moments")
        tr = compute_storm_trajectory(storm, cfg.window_hr, min_positions=cfg.min_positions,
                                      coherence=cfg.coherence, seed_key=seed_key)
    if tr is None:
        return None
    m = tr["_metrics"]
    s, e = tr["start_time_index"], tr["end_time_index"]
    wl = np.asarray(storm["storm_lon_cent"][s:e], float)
    wa = np.asarray(storm["storm_lat_cent"][s:e], float)
    coh = m["coherence"] or {}
    _, _, net = _GEOD.inv(wl[0], wa[0], wl[-1], wa[-1])
    _, _, steps = _GEOD.inv(wl[:-1], wa[:-1], wl[1:], wa[1:])
    steps = np.atleast_1d(steps) / 1000.0
    med = float(np.median(steps)) if steps.size else np.nan
    win_hours = np.arange(a + s, a + e)
    return {
        "n_positions": int(e - s), "track_positions": int(b - a), "window_start": int(s),
        "start_time": pd.Timestamp(m["start_time"]), "end_time": pd.Timestamp(m["end_time"]),
        "direction_deg": float(bearing_to_math(m["bearing"])),
        "endpoint_direction_deg": float(bearing_to_math(m["bearing_ep"])),
        "angular_variance": float(m["bearing_var"]),
        "path_speed_ms": float(m["v"]),
        "net_speed_ms": float(net / ((e - s - 1) * 3600)) if e - s > 1 else np.nan,
        "path_length_km": float(m["trajectory_length_km"]), "net_length_km": float(net / 1000.0),
        "coherent": bool(coh.get("coherent", False)), "tested": bool(coh.get("tested", False)),
        "p_weighted": coh.get("p_weighted", np.nan), "r_weighted": coh.get("r_weighted", np.nan),
        "n_segments": coh.get("n_segments", 0), "n_sim_used": coh.get("n_sim_used", 0),
        "mean_intensity_mmh": float(m["intensity"]), "peak_intensity_mmh": float(m["peak_intensity"]),
        # V6 'mean_intensity': temporal mean of the hourly area-weighted mean rate of the core over the window
        "avg_intensity_mmh": float(np.mean(np.asarray(storm["storm_record"]["avg_prcp(mm/h)"])[s:e])),
        "storm_area_km2": float(m["area"]), "ellipse_area_km2": float(m["ellipse_area_km2"]),
        "major_axis_km": float(m["major_km"]), "minor_axis_km": float(m["minor_km"]),
        "ellipse_angle_deg": float(m["ellipse_angle"]),
        "max_step_km": float(steps.max()) if steps.size else np.nan,
        "n_jumps": int(np.sum((steps > cfg.jump_factor * med) & (steps > cfg.jump_min_km))) if steps.size else 0,
        "track_lon": wl.tolist(), "track_lat": wa.tolist(),
        "track_time": [pd.Timestamp(x) for x in np.asarray(times)[win_hours]],
        "_win_hours": win_hours,
    }


def v6_on_field(fld: Field, seed_key: str, cfg: EnlargedConfig, labels=None) -> tuple:
    """The tile pipeline (V6) on a field: longest-duration track above ``min_area_fraction`` of the domain pixels,
    trimmed by the domain area, V6 window and descriptors. Returns (descriptors or None, labels, k, trimmed hours)."""
    rain0 = np.nan_to_num(fld.rain, nan=0.0)
    if labels is None:
        labels = _track_v6(rain0, fld.lat, fld.lon, cfg)       # the tile baseline is always the V6 pipeline
    area = pixel_area_km2(fld.lat, fld.lon)
    tab = track_table(labels, rain0, area)
    pick = select_longest_storm(tab, labels.shape[1] * labels.shape[2] * cfg.min_area_fraction)
    if pick is None:
        return None, labels, None, None
    k, hours = pick
    n_big = int((tab[tab["track_id"] == k]["pixels"] > labels.shape[1] * labels.shape[2] * cfg.min_area_fraction).sum())
    if n_big < cfg.min_duration_hr:
        return None, labels, k, None
    hours, _ = trim_hours(tab[tab["track_id"] == k], hours, cfg.min_area_fraction * area.sum())
    if len(hours) == 0:
        return None, labels, k, None
    d = v6_descriptors(rain0, labels, k, hours, fld.lat, fld.lon, fld.times, seed_key, cfg)
    return d, labels, k, hours


# ------------------------------------------------------------------------------------------------ one event
def _footprint_cells(mask: np.ndarray, lat, lon, res: float) -> np.ndarray:
    """Coarse global cell ids (res-degree grid) touched by a boolean (ny, nx) mask."""
    r, c = np.nonzero(mask)
    iy = np.floor((lat[r] + 90.0) / res).astype(np.int64)
    ix = np.floor((lon[c] + 180.0) / res).astype(np.int64)
    return np.unique(iy * int(round(360 / res)) + ix)


def analyze_event(ev: CatalogEvent, fld: Field, cfg: EnlargedConfig, tile: str = "", with_tile: bool = True,
                  return_footprints: bool = True, labels: Optional[np.ndarray] = None,
                  tile_result: Optional[tuple] = None, keep_private: bool = False) -> tuple:
    """Enlarged-window analysis of one catalog event on ``fld`` (the tile plus margin, containing the tile).

    Returns (row dict, footprints DataFrame [time, cells] of the storm's trimmed hours, or None). The row holds the
    event identification, the enlarged-window descriptors (``big_*``), the tile-only V6 descriptors (``tile_*``,
    the same field cropped to the tile, V6 rule), and the flags; see docs of the main storm table."""
    seed_key = f"{tile}/{ev.name}"
    w, s, e, n = ev.tile_bounds
    clon, clat = ev.circle_center
    t0 = pd.Timestamp(ev.times[0])
    m = re.search(r"_(\d{8})\.nc$", ev.name)
    edate = pd.Timestamp(m.group(1)) if m else t0
    row = {"tile": tile, "storm_event": ev.name, "storm_id": ev.storm_id, "rank": ev.rank,
           "event_date": edate, "window_start_time": t0, "window_end_time": pd.Timestamp(ev.times[-1]),
           "season": SEASONS[edate.month], "month": edate.month, "year": edate.year,
           "circle_lon": clon, "circle_lat": clat, "xlocation": ev.xloc, "ylocation": ev.yloc,
           "basin_rain_mm": ev.basin_rain_mm, "margin_deg": cfg.margin_deg,
           "domain_ny": len(fld.lat), "domain_nx": len(fld.lon), "missing_hours": fld.missing_hours}
    rain0 = np.nan_to_num(fld.rain, nan=0.0)
    valid = fld.valid
    row["valid_frac_domain"] = float(valid.mean())
    # tile position in the field grid
    r0 = int(np.argmin(np.abs(fld.lat - ev.lat[0])))
    c0 = int(np.argmin(np.abs(fld.lon - ev.lon[0])))
    if abs(fld.lat[r0] - ev.lat[0]) > 0.25 * abs(ev.lat[1] - ev.lat[0]) or r0 + len(ev.lat) > len(fld.lat) \
            or c0 + len(ev.lon) > len(fld.lon):
        raise ValueError("field does not contain the tile grid")
    inside = np.zeros(rain0.shape[1:], bool)
    inside[r0:r0 + len(ev.lat), c0:c0 + len(ev.lon)] = True
    circ = np.zeros(rain0.shape[1:])
    circ[r0:r0 + len(ev.lat), c0:c0 + len(ev.lon)] = ev.circle_on_tile()
    area = pixel_area_km2(fld.lat, fld.lon)
    tile_area = float(area[inside].sum())

    if labels is None:                     # labels of the same field and tracker may be passed in (variant reuse)
        labels = track_field(rain0, cfg, fld.lat, fld.lon)
    row["tracker"], row["jump_rule"] = cfg.tracker, cfg.jump_rule
    tab = track_table(labels, rain0, area, circ, fld.lat, fld.lon)
    circle_total = float(sum((rain0[t] * area * circ).sum() for t in range(rain0.shape[0])))
    row["circle_volume_total"] = circle_total
    if cfg.selection == "circle":
        pick = select_circle_storm(tab)
    elif cfg.selection == "v6":
        pick = select_longest_storm(tab, inside.sum() * cfg.min_area_fraction)
    else:
        raise ValueError(cfg.selection)
    fps = None
    row["retained"] = False
    if pick is not None:
        k, hours = pick
        tk = tab[tab["track_id"] == k]
        row["circle_share_pct"] = 100 * float(tk[tk["t"].isin(hours)]["circle_volume"].sum()) / max(circle_total, 1e-9)
        row["track_hours_raw"] = int(len(hours))
        hours, n_big = trim_hours(tk, hours, cfg.min_area_fraction * tile_area)
        row["track_hours"] = int(len(hours))
        row["hours_above_area"] = n_big
        if len(hours) >= 2:
            pk = tk.set_index("t").loc[hours]
            segs = jump_segments(pk["lon"].values, pk["lat"].values, cfg)
            row["track_n_jumps"] = len(segs) - 1
            if cfg.jump_rule == "split" and len(segs) > 1:
                vol = pk["volume"].values
                a, b = max(segs, key=lambda ab: (ab[1] - ab[0], vol[ab[0]:ab[1]].sum()))
                hours = hours[a:b]
                row["track_hours_split"] = int(len(hours))
                if len(hours) < cfg.min_positions:
                    n_big = 0                                      # drop: no jump-free segment of >= 5 hours
            elif cfg.jump_rule not in ("none", "split"):
                raise ValueError(cfg.jump_rule)
        if n_big >= cfg.min_duration_hr and len(hours):
            d = v6_descriptors(rain0, labels, k, hours, fld.lat, fld.lon, fld.times, seed_key, cfg)
            if d is not None:
                wh = d.pop("_win_hours")
                row["retained"] = True
                row.update({f"big_{c}": v for c, v in d.items()})
                row.update(_flags(rain0, valid, labels, k, wh, inside, area, cfg))
                row["volume_track_mm_km2"] = float(tk[tk["t"].isin(hours)]["volume"].sum())
                if return_footprints:
                    fps = pd.DataFrame({"time": [pd.Timestamp(fld.times[t]) for t in hours],
                                        "cells": [_footprint_cells(labels[t] == k, fld.lat, fld.lon,
                                                                   cfg.footprint_res_deg) for t in hours],
                                        "window": [bool(t in set(wh.tolist())) for t in hours]})
                    fps.insert(0, "storm_event", ev.name)
                    fps.insert(0, "tile", tile)
                big_k, big_hours = k, wh
    if with_tile or tile_result is not None:
        if tile_result is None:
            sub = fld.crop(*ev.tile_bounds)
            if sub.rain.shape[1:] != (len(ev.lat), len(ev.lon)):
                raise ValueError("tile crop does not match the tile grid")
            dt, lab_t, kt, ht = v6_on_field(sub, seed_key, cfg)
            twh = dt.pop("_win_hours") if dt is not None else None
            tile_result = (dt, lab_t, kt, twh, sub)
        dt, lab_t, kt, twh, sub = tile_result
        row["_tile"] = tile_result
        row["tile_retained"] = dt is not None
        if dt is not None:
            row.update({f"tile_{c}": v for c, v in dt.items()})
            if row["retained"]:
                row.update(_compare_tile(row, labels, big_k, big_hours, lab_t, kt, twh, rain0, r0, c0, sub, area))
    if keep_private:                       # track ids and tile result, for reuse by analyze_catalog_file_variants
        row["_labels"] = labels
    else:
        row.pop("_tile", None)
    return row, fps


def _flags(rain0, valid, labels, k, wh, inside, area, cfg) -> dict:
    tot = out = a_in = a_all = 0.0
    edge_h = nod_h = 0
    st = ndimage.generate_binary_structure(2, 2)
    neigh = np.zeros(labels.shape[1:], bool)
    nodata = ~valid.all(axis=0) if valid.ndim == 3 else ~valid
    for t in wh:
        m = labels[t] == k
        wv = np.where(m, rain0[t], 0.0) * area
        tot += wv.sum(); out += wv[~inside].sum()
        a_in += area[m & inside].sum(); a_all += area[m].sum()
        if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
            edge_h += 1
        if (ndimage.binary_dilation(m, st) & ~valid[t]).any():
            nod_h += 1
        neigh |= m
    neigh = ndimage.binary_dilation(neigh, st, iterations=cfg.neighbourhood_cells)
    vf = float(np.mean([valid[t][neigh].mean() for t in wh])) if neigh.any() else np.nan
    nwin = len(wh)
    return {"rain_outside_tile_pct": 100 * out / max(tot, 1e-9), "volume_window_mm_km2": float(tot),
            "footprint_area_km2": a_all / nwin, "footprint_area_in_tile_km2": a_in / nwin,
            "valid_frac_storm": vf, "edge_touch_hours": edge_h, "nodata_touch_hours": nod_h}


def _compare_tile(row, labels, k, big_hours, lab_t, kt, twh, rain0, r0, c0, sub, area) -> dict:
    """Same storm on the tile? Mean/max distance between the enlarged and the tile tracks over shared hours, and
    the share of the tile storm's rain (its window hours) inside the enlarged storm's footprint."""
    bt = pd.Series(range(len(row["big_track_time"])), index=row["big_track_time"])
    tt = pd.Series(range(len(row["tile_track_time"])), index=row["tile_track_time"])
    shared = bt.index.intersection(tt.index)
    out = {"tile_shared_hours": int(len(shared))}
    if len(shared):
        bl = np.array(row["big_track_lon"])[bt[shared].values]; ba = np.array(row["big_track_lat"])[bt[shared].values]
        tl = np.array(row["tile_track_lon"])[tt[shared].values]; ta = np.array(row["tile_track_lat"])[tt[shared].values]
        _, _, d = _GEOD.inv(bl, ba, tl, ta)
        out["tile_match_km"] = float(np.mean(d) / 1000.0)
        out["tile_match_max_km"] = float(np.max(d) / 1000.0)
    ny, nx = lab_t.shape[1:]
    num = den = 0.0
    for t in twh:
        mt = lab_t[t] == kt
        w = np.where(mt, rain0[t, r0:r0 + ny, c0:c0 + nx], 0.0) * area[r0:r0 + ny, c0:c0 + nx]
        den += w.sum()
        num += w[labels[t, r0:r0 + ny, c0:c0 + nx] == k].sum()
    out["tile_fp_covered_pct"] = 100 * num / den if den > 0 else np.nan
    return out


# ------------------------------------------------------------------------------------------------ duplicates
def assign_system_ids(fp: pd.DataFrame, time_overlap: float = 0.5, footprint_overlap: float = 0.5,
                      key=("tile", "storm_event")) -> tuple:
    """Cross-tile duplicate systems.

    ``fp`` has one row per (event, hour) with columns ``tile``, ``storm_event``, ``time`` and ``cells`` (coarse cell
    ids of the storm footprint at that hour). Events A and B of DIFFERENT tiles are linked when
        shared hours / min(hours A, hours B) >= ``time_overlap``   and
        sum over shared hours |F_A & F_B| / min(sum |F_A|, sum |F_B|) (shared hours) >= ``footprint_overlap``.
    Linked events (transitively) get one system id. Returns (DataFrame tile, storm_event, system_id, system_size)
    and the DataFrame of linked pairs with their overlaps."""
    k0, k1 = key
    keys = fp[[k0, k1]].drop_duplicates().reset_index(drop=True)
    kid = {tuple(r): i for i, r in enumerate(keys.itertuples(index=False))}
    hours = fp.groupby([k0, k1]).size()
    nh = np.array([hours[tuple(r)] for r in keys.itertuples(index=False)])
    acc: dict = {}
    for _, g in fp.groupby("time"):
        if g[k0].nunique() < 2:
            continue
        items = [(kid[(a, b)], a, set(np.asarray(c).tolist())) for a, b, c in zip(g[k0], g[k1], g["cells"])]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (ia, ta, ca), (ib, tb, cb) = items[i], items[j]
                if ta == tb:
                    continue
                p = (min(ia, ib), max(ia, ib))
                s = acc.setdefault(p, [0, 0, 0, 0])
                s[0] += 1
                s[1] += len(ca & cb)
                na, nb = (len(ca), len(cb)) if ia < ib else (len(cb), len(ca))
                s[2] += na; s[3] += nb
    parent = list(range(len(keys)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    pairs = []
    for (a, b), (sh, inter, sa, sb) in acc.items():
        to = sh / min(nh[a], nh[b])
        fo = inter / max(min(sa, sb), 1)
        if to >= time_overlap and fo >= footprint_overlap:
            parent[find(a)] = find(b)
            pairs.append((*keys.iloc[a], *keys.iloc[b], sh, to, fo))
    root = np.array([find(i) for i in range(len(keys))])
    # system id: stable, the smallest key index of the group
    first = {}
    for i, r in enumerate(root):
        first.setdefault(r, i)
    sid = np.array([first[r] for r in root])
    out = keys.assign(system_id=sid)
    out["system_size"] = out.groupby("system_id")["system_id"].transform("size")
    pairs = pd.DataFrame(pairs, columns=[f"{k0}_a", f"{k1}_a", f"{k0}_b", f"{k1}_b", "shared_hours",
                                         "time_overlap", "footprint_overlap"])
    return out, pairs


# ------------------------------------------------------------------------------------------------ catalog driver
def tile_bounds_with_margin(ev: CatalogEvent, margin: float) -> tuple:
    w, s, e, n = ev.tile_bounds
    return w - margin, s - margin, e + margin, n + margin


def analyze_catalog_file(path, aorc_dir, cfg: EnlargedConfig, tile: str = "", with_tile: bool = True) -> tuple:
    """Read one storm file, the AORC field on tile + margin, and analyse it (``analyze_event``)."""
    ev = read_catalog_event(path)
    fld = read_aorc_field(aorc_dir, ev.times, *tile_bounds_with_margin(ev, cfg.margin_deg))
    row, fps = analyze_event(ev, fld, cfg, tile=tile, with_tile=with_tile)
    return row, fps


def analyze_catalog_file_variants(path, aorc_dir, cfgs: dict, tile: str = "") -> dict:
    """Several configurations (same margin) of one event from a single AORC read; the tracking is reused between
    configurations with the same tracker, and the tile-only V6 columns are computed once (first configuration).
    Returns {name: (row, footprints)}."""
    ev = read_catalog_event(path)
    margins = {c.margin_deg for c in cfgs.values()}
    if len(margins) != 1:
        raise ValueError("variants must share the margin")
    fld = read_aorc_field(aorc_dir, ev.times, *tile_bounds_with_margin(ev, margins.pop()))
    out, cache, tres = {}, {}, None
    for name, cfg in cfgs.items():
        row, fps = analyze_event(ev, fld, cfg, tile=tile, labels=cache.get(cfg.tracker), tile_result=tres,
                                 keep_private=True)
        cache[cfg.tracker] = row.pop("_labels")
        tres = row.pop("_tile", None)      # the tile-only V6 result is computed once and shared
        row["variant"] = name
        out[name] = (row, fps)
    return out


def _job(args):
    path, aorc_dir, cfg, tile, with_tile = args
    try:
        return analyze_catalog_file(path, aorc_dir, cfg, tile, with_tile)
    except Exception as exc:                                           # noqa: BLE001
        return {"tile": tile, "storm_event": Path(path).name, "error": f"{type(exc).__name__}: {exc}"}, None


def run_catalog(catalog_dir, aorc_dir, cfg: EnlargedConfig, tile: str, workers: int = 1,
                with_tile: bool = True, files: Optional[Sequence] = None) -> tuple:
    """Analyse every storm file of a catalog; returns (events DataFrame, footprints DataFrame)."""
    files = list(files) if files is not None else sorted(Path(catalog_dir).glob("*.nc"), key=storm_number)
    jobs = [(str(f), str(aorc_dir), cfg, tile, with_tile) for f in files]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_job, jobs, chunksize=2))
    else:
        res = [_job(j) for j in jobs]
    rows = pd.DataFrame([r for r, _ in res])
    fps = [f for _, f in res if f is not None]
    fps = pd.concat(fps, ignore_index=True) if fps else pd.DataFrame(columns=["tile", "storm_event", "time", "cells"])
    return rows, fps


def seed_for(tile: str, event: str, base: int = CoherenceConfig().seed) -> list:
    """Coherence-test seed used for ``"<tile>/<event>"`` (as the V6 pipeline)."""
    return [base, zlib.crc32(f"{tile}/{event}".encode())]
