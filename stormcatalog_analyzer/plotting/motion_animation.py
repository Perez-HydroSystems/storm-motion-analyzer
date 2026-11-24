"""Animated GIFs of storm motion: the hourly rain, the tracked storm core and the track of its rain mass.

``animate_storm_motion``   one panel per storm, all at the same map scale, so that the distances the storms cover can
                           be compared (for example a storm that travels next to storms that stall, turn or meander)
``animate_two_motions``    one storm with its two motions: the track of the rain mass and the track that the rain
                           pattern alone would give, with the velocities of both and the propagation

A storm is a dict with
    rain        (nt, ny, nx) hourly rain rate (mm/h) on a regular latitude/longitude grid, the analysed hours
    lon, lat    grid coordinates (degrees)
    mask        (nt, ny, nx) bool, the tracked storm core at each hour (optional)
    track_lon,  hourly position of the rain-mass centroid (nt)
    track_lat
    label       title of the panel; ``sublabel`` an optional short description next to it
    times       time of each hour (optional; shown as a clock in the panel)
and, for ``animate_two_motions``,
    step_u,     hourly rain-pattern velocity (m/s, east and north) of each step, as given by
    step_v      ``motion.pattern.storm_pattern`` (nt - 1 values, or nt with a last value that is not used)
    step_ok     bool, the steps accepted by the quality control (optional; all accepted if missing)
    pattern_u,  storm-level pattern velocity (m/s); the mean of the accepted steps if missing
    pattern_v

Maps are drawn in kilometres from the centre of each track (111.32 km per degree of latitude, 111.32 km x
cos(latitude) per degree of longitude), which keeps one scale for every panel.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import patheffects as pe
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from pyproj import Geod

KM_DEG = 111.32
_GEOD = Geod(ellps="WGS84")

# roles (light surface): text in ink, rain as one blue ramp, the two motions as two hues
SURFACE, INK, INK2, RULE, LAND = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d2", "#b9b8b1"
MASS, PATTERN = "#eb6834", "#1baf7a"
RAIN_LEVELS = [0.5, 1, 2, 4, 8, 16, 32]
RAIN_COLORS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_HALO = [pe.Stroke(linewidth=4.2, foreground=SURFACE), pe.Normal()]
_TEXT_HALO = [pe.Stroke(linewidth=2.6, foreground=SURFACE), pe.Normal()]


def _rain_style():
    cmap = ListedColormap(RAIN_COLORS)
    cmap.set_under((0, 0, 0, 0))
    return cmap, BoundaryNorm(RAIN_LEVELS + [1e9], cmap.N)


def _prepare(storm: dict) -> dict:
    """Coordinates in km from the centre of the track, and the path and net lengths up to each hour."""
    s = dict(storm)
    tl, ta = np.asarray(s["track_lon"], float), np.asarray(s["track_lat"], float)
    s["lon0"], s["lat0"] = 0.5 * (tl.min() + tl.max()), 0.5 * (ta.min() + ta.max())
    s["kx"], s["ky"] = KM_DEG * np.cos(np.radians(s["lat0"])), KM_DEG
    s["x"], s["y"] = (np.asarray(s["lon"], float) - s["lon0"]) * s["kx"], (np.asarray(s["lat"], float) - s["lat0"]) * s["ky"]
    s["tx"], s["ty"] = (tl - s["lon0"]) * s["kx"], (ta - s["lat0"]) * s["ky"]
    step = np.atleast_1d(_GEOD.inv(tl[:-1], ta[:-1], tl[1:], ta[1:])[2]) / 1e3
    s["path_km"] = np.r_[0.0, np.cumsum(step)]
    s["net_km"] = np.array([_GEOD.inv(tl[0], ta[0], tl[k], ta[k])[2] / 1e3 for k in range(tl.size)])
    s["n"] = tl.size
    return s


def _borders(s: dict, half: tuple) -> list:
    """State and country borders inside the panel, as (x, y) arrays in km. Empty if the map data are missing."""
    try:
        import cartopy.feature as cfeature
        from shapely.geometry import box
        w, h = half[0] / s["kx"] * 1.05, half[1] / s["ky"] * 1.05
        win = box(s["lon0"] - w, s["lat0"] - h, s["lon0"] + w, s["lat0"] + h)
        out = []
        feat = cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lakes", "50m")
        for g in feat.intersecting_geometries((s["lon0"] - w, s["lon0"] + w, s["lat0"] - h, s["lat0"] + h)):
            part = g.boundary.intersection(win)
            for line in getattr(part, "geoms", [part]):
                if line.is_empty or not hasattr(line, "xy"):
                    continue
                lx, ly = np.asarray(line.xy[0]), np.asarray(line.xy[1])
                out.append(((lx - s["lon0"]) * s["kx"], (ly - s["lat0"]) * s["ky"]))
        return out
    except Exception:                                                   # noqa: BLE001  (no map data: no borders)
        return []


def _panel(ax, s: dict, k: int, final: bool, half: tuple, scale_km: float, pattern: bool = False, pat=None):
    """Draw hour ``k`` of one storm: rain, storm core, borders and the track up to that hour."""
    cmap, norm = _rain_style()
    ax.clear()
    ax.set_facecolor(SURFACE)
    for bx, by in s["borders"]:
        ax.plot(bx, by, color=LAND, lw=0.7, zorder=1)
    rain = np.asarray(s["rain"][k], float)
    ax.pcolormesh(s["x"], s["y"], np.where(rain >= RAIN_LEVELS[0], rain, np.nan), cmap=cmap, norm=norm,
                  shading="nearest", zorder=2, rasterized=True)
    if s.get("mask") is not None and np.any(s["mask"][k]):
        ax.contour(s["x"], s["y"], np.asarray(s["mask"][k], float), levels=[0.5], colors=[INK2], linewidths=0.9,
                   zorder=3)
    if pattern and pat is not None:                                     # the track the rain pattern alone would give
        ax.plot(pat[0][:k + 1], pat[1][:k + 1], color=PATTERN, lw=2.2, zorder=5, path_effects=_HALO,
                solid_capstyle="round")
        ax.scatter(pat[0][k], pat[1][k], s=62, color=PATTERN, edgecolor=SURFACE, linewidth=1.6, zorder=7)
    ax.plot(s["tx"][:k + 1], s["ty"][:k + 1], color=MASS, lw=2.2, zorder=6, path_effects=_HALO,
            solid_capstyle="round")
    if k > 0:
        ax.scatter(s["tx"][1:k], s["ty"][1:k], s=13, color=MASS, edgecolor=SURFACE, linewidth=0.7, zorder=7)
    if final and s["n"] > 1 and not pattern:                            # net displacement, over the track
        ax.add_patch(FancyArrowPatch((s["tx"][0], s["ty"][0]), (s["tx"][-1], s["ty"][-1]), arrowstyle="-|>",
                                     mutation_scale=12, lw=1.4, ls=(0, (4, 3)), color=INK, zorder=7.5,
                                     shrinkA=6, shrinkB=8))
    ax.scatter(s["tx"][0], s["ty"][0], s=46, facecolor=SURFACE, edgecolor=MASS, linewidth=1.8, zorder=8)
    if k > 0:
        ax.scatter(s["tx"][k], s["ty"][k], s=62, color=MASS, edgecolor=SURFACE, linewidth=1.6, zorder=9)
    ax.set_xlim(-half[0], half[0])
    ax.set_ylim(-half[1], half[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(RULE)
        sp.set_linewidth(1.0)
    ax.text(0.0, 1.025, s.get("label", ""), transform=ax.transAxes, ha="left", va="bottom", fontsize=13,
            fontweight="bold", color=INK, zorder=12)
    if s.get("sublabel"):
        ax.annotate(s["sublabel"], xy=(0.0, 1.025), xycoords="axes fraction",
                    xytext=(len(s.get("label", "")) * 9.6 + 12, 1), textcoords="offset points", ha="left",
                    va="bottom", fontsize=10, color=INK2, zorder=12)
    ax.add_patch(matplotlib.patches.Rectangle((0, 0), 1, 0.092, transform=ax.transAxes, facecolor=SURFACE,
                                              edgecolor="none", alpha=0.90, zorder=11))
    clock = ""
    if s.get("times") is not None:
        import pandas as pd
        ts = pd.Timestamp(str(s["times"][k]))
        clock = f"{ts.day} {ts:%b %Y %H:%M} UTC     "
    ax.text(0.02, 0.046, f"{clock}Path {s['path_km'][k]:.0f} km     Net {s['net_km'][k]:.0f} km",
            transform=ax.transAxes, ha="left", va="center", fontsize=9.5, color=INK, zorder=12)
    x1, x0 = 0.98, 0.98 - scale_km / (2 * half[0])                       # scale bar, in the band
    ax.plot([x0, x1], [0.030, 0.030], transform=ax.transAxes, color=INK, lw=2.0, solid_capstyle="butt", zorder=12)
    ax.text(0.5 * (x0 + x1), 0.040, f"{scale_km:.0f} km", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, color=INK, zorder=12)


def _colorbar(fig, rect):
    """Discrete rain-rate key."""
    cax = fig.add_axes(rect)
    n = len(RAIN_COLORS)
    for i, c in enumerate(RAIN_COLORS):
        cax.add_patch(matplotlib.patches.Rectangle((i, 0), 1, 1, facecolor=c, edgecolor=SURFACE, linewidth=1.0))
    cax.set_xlim(0, n)
    cax.set_ylim(0, 1)
    cax.set_yticks([])
    cax.set_xticks(range(n))
    cax.set_xticklabels([f"{v:g}" for v in RAIN_LEVELS], fontsize=8.5, color=INK2)
    cax.tick_params(length=0, pad=3)
    for sp in cax.spines.values():
        sp.set_visible(False)
    cax.set_title("Rain rate (mm/h)", fontsize=9, color=INK2, loc="left", pad=4)
    return cax


def _save(frames: list, path, fps: float, hold_frames: int) -> Path:
    """Write the frames (RGB arrays) as a looping GIF; the last frame is held for ``hold_frames`` frame times."""
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f).quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
            for f in frames]
    ms = int(round(1000 / fps))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], loop=0, optimize=True,
                 duration=[ms] * (len(imgs) - 1) + [ms * max(hold_frames, 1)], disposal=1)
    return path


def _grab(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def animate_storm_motion(storms, save_path, ncols: int = 2, half_width_km: float = 600.0,
                         half_height_km: float = 450.0, title: str = None, subtitle: str = None,
                         scale_km: float = 200.0, fps: float = 2.0, hold_frames: int = 7, dpi: int = 100,
                         panel_width_in: float = 5.1, borders: bool = True) -> Path:
    """GIF with one panel per storm at a common map scale: rain, storm core and the track of the rain mass, hour by
    hour; the last frame adds the net displacement. Returns the path of the GIF.

    storms          list of storm dicts (see the module docstring); storms with fewer hours stop at their last hour
    half_width_km,  half size of every panel (km), centred on the track of its storm
    half_height_km
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    S = [_prepare(s) for s in storms]
    half = (float(half_width_km), float(half_height_km))
    for s in S:
        s["borders"] = _borders(s, half) if borders else []
    nrows = int(np.ceil(len(S) / ncols))
    pw, ph = panel_width_in, panel_width_in * half[1] / half[0]
    top, bottom, gap, vgap, side = (1.22 if title else 0.42), 0.78, 0.14, 0.42, 0.15
    W, H = side * 2 + ncols * pw + (ncols - 1) * gap, top + bottom + nrows * ph + (nrows - 1) * vgap
    fig = Figure(figsize=(W, H), dpi=dpi, facecolor=SURFACE)
    FigureCanvasAgg(fig)
    axes = []
    for i in range(len(S)):
        r, c = divmod(i, ncols)
        axes.append(fig.add_axes([(side + c * (pw + gap)) / W, (bottom + (nrows - 1 - r) * (ph + vgap)) / H,
                                  pw / W, ph / H]))
    if title:
        fig.text(side / W, 1 - 0.36 / H, title, fontsize=16, fontweight="bold", color=INK, ha="left", va="center")
        if subtitle:
            fig.text(side / W, 1 - 0.70 / H, subtitle, fontsize=10.5, color=INK2, ha="left", va="center")
    _colorbar(fig, [(W - side - 2.5) / W, 0.27 / H, 2.5 / W, 0.17 / H])
    handles = [Line2D([], [], color=MASS, lw=2.2, marker="o", ms=6, mec=SURFACE, label="Track of the rain mass"),
               Line2D([], [], color=INK2, lw=0.9, label="Storm core"),
               Line2D([], [], color=INK, lw=1.5, ls=(0, (4, 3)), label="Net displacement")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(side / W * 0.6, 0.20 / H), ncols=3, frameon=False,
               fontsize=9.5, labelcolor=INK, handlelength=2.6, columnspacing=1.8)
    n = max(s["n"] for s in S)
    frames = []
    for k in range(n):
        for ax, s in zip(axes, S):
            _panel(ax, s, min(k, s["n"] - 1), final=(k == n - 1), half=half, scale_km=scale_km)
        frames.append(_grab(fig))
    return _save(frames, save_path, fps, hold_frames)


def _pattern_track(s: dict) -> tuple:
    """Positions (km) of a point that starts at the first centroid and moves with the accepted pattern steps."""
    n = s["n"]
    u, v = np.asarray(s["step_u"], float)[:n - 1], np.asarray(s["step_v"], float)[:n - 1]
    ok = np.asarray(s["step_ok"], bool)[:n - 1] if s.get("step_ok") is not None else np.isfinite(u)
    du, dv = np.where(ok, u, 0.0) * 3.6, np.where(ok, v, 0.0) * 3.6
    return s["tx"][0] + np.r_[0.0, np.cumsum(du)], s["ty"][0] + np.r_[0.0, np.cumsum(dv)]


def animate_two_motions(storm, save_path, half_width_km: float = 500.0, half_height_km: float = 400.0,
                        title: str = None, subtitle: str = None, scale_km: float = 200.0, fps: float = 2.0,
                        hold_frames: int = 8, dpi: int = 100, map_width_in: float = 6.2,
                        borders: bool = True) -> Path:
    """GIF of the two motions of one storm. Left: the rain, the track of the rain mass, and the track that the rain
    pattern alone would give (the accepted hourly pattern displacements added from the first position). Right: the
    hourly velocities of both, and in the last frame the two storm velocities and the propagation between them
    (rain mass minus rain pattern). Returns the path of the GIF."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    s = _prepare(storm)
    half = (float(half_width_km), float(half_height_km))
    s["borders"] = _borders(s, half) if borders else []
    n = s["n"]
    pat = _pattern_track(s)
    u, v = np.asarray(s["step_u"], float)[:n - 1], np.asarray(s["step_v"], float)[:n - 1]
    ok = np.asarray(s["step_ok"], bool)[:n - 1] if s.get("step_ok") is not None else np.isfinite(u)
    mu, mv = np.diff(s["tx"]) / 3.6, np.diff(s["ty"]) / 3.6            # hourly rain-mass velocity (m/s)
    cu, cv = (s["tx"][-1] - s["tx"][0]) / (3.6 * (n - 1)), (s["ty"][-1] - s["ty"][0]) / (3.6 * (n - 1))
    pu = float(s["pattern_u"]) if s.get("pattern_u") is not None else float(np.mean(u[ok]))
    pv = float(s["pattern_v"]) if s.get("pattern_v") is not None else float(np.mean(v[ok]))
    hourly = np.r_[np.hypot(u[ok], v[ok]), np.hypot(mu, mv)]          # the scale is not set by one fast hourly step
    lim = 1.15 * max(np.percentile(hourly, 90), np.hypot(pu, pv), np.hypot(cu, cv))
    ring = 5.0 if lim <= 22 else 10.0

    mw, mh = map_width_in, map_width_in * half[1] / half[0]
    vw = mh                                                             # square velocity panel
    top, bottom, gap, side = (1.22 if title else 0.42), 0.78, 0.35, 0.15
    W, H = side * 2 + mw + gap + vw, top + bottom + mh
    fig = Figure(figsize=(W, H), dpi=dpi, facecolor=SURFACE)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([side / W, bottom / H, mw / W, mh / H])
    av = fig.add_axes([(side + mw + gap) / W, bottom / H, vw / W, mh / H])
    if title:
        fig.text(side / W, 1 - 0.36 / H, title, fontsize=16, fontweight="bold", color=INK, ha="left", va="center")
        if subtitle:
            fig.text(side / W, 1 - 0.70 / H, subtitle, fontsize=10.5, color=INK2, ha="left", va="center")
    _colorbar(fig, [(W - side - 2.5) / W, 0.27 / H, 2.5 / W, 0.17 / H])
    handles = [Line2D([], [], color=MASS, lw=2.2, marker="o", ms=6, mec=SURFACE, label="Rain mass"),
               Line2D([], [], color=PATTERN, lw=2.2, marker="o", ms=6, mec=SURFACE, label="Rain pattern"),
               Line2D([], [], color=INK2, lw=0.9, label="Storm core")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(side / W * 0.6, 0.20 / H), ncols=3, frameon=False,
               fontsize=9.5, labelcolor=INK, handlelength=2.6, columnspacing=1.8)

    def arrow(a, x, y, color, lw, alpha=1.0, ls="-", z=5, x0=0.0, y0=0.0, scale=11):
        a.add_patch(FancyArrowPatch((x0, y0), (x, y), arrowstyle="-|>", mutation_scale=scale, lw=lw, color=color,
                                    alpha=alpha, ls=ls, zorder=z, shrinkA=0, shrinkB=0, clip_on=True))

    def velocity_panel(k, final):
        av.clear()
        av.set_facecolor(SURFACE)
        t = np.linspace(0, 2 * np.pi, 181)
        for r in np.arange(ring, lim + 1e-9, ring):
            av.plot(r * np.cos(t), r * np.sin(t), color=RULE, lw=0.8, zorder=1)
            av.text(r * np.cos(np.radians(-45)), r * np.sin(np.radians(-45)), f"{r:g}", fontsize=8, color=INK2,
                    ha="center", va="center", zorder=2,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor=SURFACE, edgecolor="none"))
        av.plot([-lim, lim], [0, 0], color=RULE, lw=0.8, zorder=1)
        av.plot([0, 0], [-lim, lim], color=RULE, lw=0.8, zorder=1)
        for txt, xy, ha, va in (("N", (0, lim * 0.97), "center", "top"), ("E", (lim * 0.97, 0), "right", "center"),
                                ("S", (0, -lim * 0.97), "center", "bottom"), ("W", (-lim * 0.97, 0), "left", "center")):
            av.text(*xy, txt, fontsize=9, color=INK2, ha=ha, va=va, zorder=2,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none"))
        for i in range(k):                                              # the hourly steps up to this hour
            a = 0.30 if (final or i < k - 1) else 0.95
            if ok[i]:
                arrow(av, u[i], v[i], PATTERN, 1.3, a, z=3)
            arrow(av, mu[i], mv[i], MASS, 1.3, a, z=3)
        if final:
            arrow(av, pu, pv, PATTERN, 3.0, z=6, scale=16)
            arrow(av, cu, cv, MASS, 3.0, z=6, scale=16)
            arrow(av, cu, cv, INK, 1.5, ls=(0, (4, 3)), z=5, x0=pu, y0=pv, scale=12)
            key = [Line2D([], [], color=PATTERN, lw=3.0, label=f"Rain pattern  {np.hypot(pu, pv):.1f} m/s"),
                   Line2D([], [], color=MASS, lw=3.0, label=f"Rain mass  {np.hypot(cu, cv):.1f} m/s"),
                   Line2D([], [], color=INK, lw=1.5, ls=(0, (4, 3)),
                          label=f"Propagation  {np.hypot(cu - pu, cv - pv):.1f} m/s")]
            leg = av.legend(handles=key, loc="upper right", fontsize=9.5, labelcolor=INK, frameon=True,
                            facecolor=SURFACE, edgecolor=RULE, framealpha=0.95, handlelength=2.4, borderpad=0.7)
            leg.set_zorder(13)
        av.set_xlim(-lim, lim)
        av.set_ylim(-lim, lim)
        av.set_aspect("equal")
        av.set_xticks([])
        av.set_yticks([])
        for sp in av.spines.values():
            sp.set_color(RULE)
            sp.set_linewidth(1.0)
        av.text(0.0, 1.025, "Velocity (m/s)", transform=av.transAxes, ha="left", va="bottom", fontsize=13,
                fontweight="bold", color=INK, zorder=12)
        av.text(0.03, 0.035, "Thin arrows: velocity of each hourly step" if not final else
                "Propagation = rain mass minus rain pattern", transform=av.transAxes, ha="left",
                va="bottom", fontsize=9, color=INK2, path_effects=_TEXT_HALO, zorder=12)

    frames = []
    for k in range(n):
        final = k == n - 1
        _panel(ax, s, k, final=final, half=half, scale_km=scale_km, pattern=True, pat=pat)
        velocity_panel(k, final)
        frames.append(_grab(fig))
    return _save(frames, save_path, fps, hold_frames)
