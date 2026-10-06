"""Catalog-diagnostics pipeline.

Three stages, callable from notebooks and the CLI:

1. ``run_catalog_tracking(cfg)``  -> dict[event_name -> event_dict]
2. ``summarize_motion(results, cfg)`` -> MotionSummary (trajectories, df, aggregates)
3. ``generate_catalog_diagnostics(summary, cfg)`` -> list[Path] of saved figures

``run(cfg)`` chains all three. This replaces the ~90-line loop that was previously
copy-pasted across the notebooks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pyproj import Transformer

from .config import Config
from . import io as scio
from .tracking.dataset import (
    storm_tracking_event,
    continuos_storm,
    storm_tracking_features,
)
from .motion.direction import (
    mean_precipitation_within_ellipse,
    get_max_accumulated_value,
    compute_line_length,
    mean_velocity,
    mean_direction,
    mean_direction_weighted,
    mean_direction_ln,
    mean_bearing,
    endpoint_bearing,
    bearing_to_math,
    line_length_geodesic,
    mean_speed_geodesic,
)

# Projected (EPSG:2163) -> geographic (EPSG:4326)
_PCS_TO_GCS = Transformer.from_crs("EPSG:2163", "EPSG:4326", always_xy=True)


def _parse_storm_id(name: str, fallback: int) -> int:
    try:
        return int(name.split("_storm_")[1].split("_")[0])
    except (IndexError, ValueError):
        return fallback


def _axial_mean_deg(angles_deg) -> float:
    """Mean of axial angles (ellipse orientations, period 180 deg), in (-90, 90].

    Doubling maps the 180-deg-periodic axis angles onto the full circle, so that,
    e.g., +89 and -89 deg (both nearly north-south) average to ~90, not to 0.
    """
    a = np.radians(2 * np.asarray(angles_deg, dtype=float))
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.degrees(0.5 * np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))))


# --------------------------------------------------------------------------
# Stage 1: tracking
# --------------------------------------------------------------------------
def run_catalog_tracking(cfg: Config, verbose: bool = True) -> Dict[str, dict]:
    """Track every event in the catalog; return only events that pass selection."""
    t = cfg.tracking
    results: Dict[str, dict] = {}
    for event_path in scio.discover_events(cfg):
        name = event_path.name
        ds = scio.open_event(event_path, var_name=t.rainfall_var_name)
        try:
            event_dict: dict = {}
            storm_tracking_event(
                ds, event_dict,
                morph_radius=t.morph_radius_cells,
                high_threshold=t.rainfall_threshold_mmhr,
                var_name=t.rainfall_var_name,
                ratio_threshold=t.overlap_ratio_threshold,
                dry_spell_time=t.dry_spell_hr,
                area_fraction=cfg.selection.min_area_fraction,
            )
            if int(event_dict["longest_duration"]) < cfg.selection.min_duration_hr:
                if verbose:
                    print(f"skip {name}: duration {event_dict['longest_duration']} "
                          f"< {cfg.selection.min_duration_hr}")
                continue
            continuos_storm(event_dict, area_fraction=cfg.selection.min_area_fraction)
            storm_tracking_features(event_dict, ellipse_fit=t.ellipse_fit_method)
            results[name] = event_dict
            if verbose:
                print(f"ok   {name}")
        finally:
            ds.close()
    if verbose:
        print(f"\nRetained {len(results)} storms.")
    return results


# --------------------------------------------------------------------------
# Stage 2: motion summary (the extracted notebook glue)
# --------------------------------------------------------------------------
@dataclass
class MotionSummary:
    storm_trajectories: dict
    geographic_trajectories: dict
    storm_properties: pd.DataFrame
    df: pd.DataFrame
    longest_trajectories: dict
    storm_mean_bearing: list          # per-storm geodesic mean bearing (compass deg)
    storm_mean_velocity: list
    mean_angle: float                 # catalog circular mean direction (deg CCW from E)
    mean_vel: float
    mean_p: float


def compute_storm_trajectory(storm: dict, window_hr: int, storm_id: int = 0) -> Optional[dict]:
    """Compute one storm's most-intense-window trajectory + direction/speed.

    ``window_hr`` is the most-intense-window length in hours (``motion.intensity_window_hr``).
    Side effect: writes the per-storm motion metrics back into ``storm`` (the
    event_dict) because the per-event plots read them from there. Returns the public
    trajectory dict (with a private ``_metrics`` key) or None if the storm has no
    valid (>= 2 centroid) trajectory.
    """
    mean_p_ellipse = mean_precipitation_within_ellipse(storm)
    if np.nanmin(mean_p_ellipse) < 0:
        return None
    series = pd.Series(mean_p_ellipse, index=storm["selected_storm_time_steps"].values)
    window = pd.to_timedelta(window_hr, unit="h")
    _, _, _, start_idx, end_idx = get_max_accumulated_value(series, window)

    x = storm["storm_prj_lon_cent_list"][start_idx:end_idx]
    y = storm["storm_prj_lat_cent_list"][start_idx:end_idx]
    lon = np.asarray(storm["storm_lon_cent"][start_idx:end_idx], dtype=float)
    lat = np.asarray(storm["storm_lat_cent"][start_idx:end_idx], dtype=float)
    if len(x) < 2:
        return None

    # motion metrics: geodesic on WGS84, relative to true north. Computed as compass
    # bearings of the direction of motion (0 = N, clockwise); reported as degrees
    # counterclockwise from east (0 = E, 90 = N)
    v = mean_speed_geodesic(lon, lat)
    bearing, bearing_var = mean_bearing(lon, lat, weighted=True)
    bearing_unw, _ = mean_bearing(lon, lat, weighted=False)
    bearing_ep = endpoint_bearing(lon, lat)

    # legacy projected-plane metrics (EPSG:2163 grid; math convention 0 = E, CCW),
    # kept for comparison. Grid north differs from true north away from 100 W.
    d, vd = mean_direction(x, y)
    wd, wv = mean_direction_weighted(x, y)
    lm = mean_direction_ln(x, y)[0]

    # metrics needed by per-event plots live on the event_dict
    storm["mean_velocity"] = v
    storm["mean_bearing"] = bearing
    storm["var_bearing"] = bearing_var
    storm["endpoint_bearing"] = bearing_ep
    storm["mean_direction"] = d
    storm["var_direction"] = vd
    storm["mean_direction_ln"] = lm
    storm["mean_direction_weighted"] = wd
    storm["var_dir_weighted"] = wv

    # aggregate per-storm intensity and ellipse properties over the most-intense window
    win = slice(start_idx, end_idx + 1)
    record = storm["storm_record"]
    max_p = np.asarray(record["max_prcp(mm/h)"])[win]
    major = np.asarray(record["major_axis_length(m)"])[win]
    minor = np.asarray(record["minor_axis_length(m)"])[win]
    angle = np.asarray(record["ellipse_angle (degree)"])[win]
    times = storm["selected_storm_time_steps"]

    mean_precip = float(np.mean(mean_p_ellipse[win]))
    trajectory_length_m = line_length_geodesic(lon, lat)
    return {
        "start_time_index": start_idx,
        "end_time_index": end_idx,
        "x_coords": x,
        "y_coords": y,
        "trajectory_length": trajectory_length_m,
        "mean_precipitation": mean_precip,
        "mean_bearing": bearing,
        "angular_variance": bearing_var,
        "storm_id": storm_id,
        "_metrics": {
            "v": v, "bearing": bearing, "bearing_unw": bearing_unw, "bearing_ep": bearing_ep,
            "bearing_var": bearing_var, "wd_grid": wd, "wv_grid": wv,
            "v_grid": mean_velocity(x, y), "length_grid_km": compute_line_length(x, y) / 1000.0,
            "area": float(np.mean(storm["selected_storm_area"][win])),
            "intensity": mean_precip,
            "peak_intensity": float(np.nanmax(max_p)) if max_p.size else float("nan"),
            "total_rainfall": float(np.nansum(mean_p_ellipse[win])),
            "trajectory_length_km": trajectory_length_m / 1000.0,
            "major_km": float(np.nanmean(major)) / 1000.0,
            "minor_km": float(np.nanmean(minor)) / 1000.0,
            "ellipse_angle": _axial_mean_deg(angle),
            "duration_hours": int(end_idx - start_idx + 1),
            "start_time": times[start_idx].values,
            "end_time": times[end_idx].values,
        },
    }


def summarize_motion(results: Dict[str, dict], cfg: Config) -> MotionSummary:
    """Compute per-storm direction/speed/trajectory and catalog-level aggregates."""
    results = {k: v for k, v in results.items() if v}
    window_hr = cfg.motion.intensity_window_hr

    storm_trajectories: dict = {}
    storm_mean_velocity: list = []
    storm_mean_bearing: list = []
    storm_intensity: list = []
    property_rows: list = []

    for i, (name, storm) in enumerate(results.items()):
        sid = _parse_storm_id(name, i)
        traj = compute_storm_trajectory(storm, window_hr, storm_id=sid)
        if traj is None:
            continue
        m = traj.pop("_metrics")
        storm_trajectories[name] = traj
        storm_mean_velocity.append(m["v"])
        storm_mean_bearing.append(m["bearing"])
        storm_intensity.append(m["intensity"])
        property_rows.append({
            "storm_event": name,
            "storm_id": sid,
            "start_time": m["start_time"],
            "end_time": m["end_time"],
            "duration_hours": m["duration_hours"],
            "mean_speed_ms": m["v"],
            # directions reported counterclockwise from east (0 = E, 90 = N)
            "mean_direction_deg": float(bearing_to_math(m["bearing"])),
            "mean_direction_vector_deg": float(bearing_to_math(m["bearing_unw"])),
            "endpoint_direction_deg": float(bearing_to_math(m["bearing_ep"])),
            "angular_variance": m["bearing_var"],
            "mean_intensity_mmh": m["intensity"],
            "peak_intensity_mmh": m["peak_intensity"],
            "total_rainfall_mm": m["total_rainfall"],
            "storm_area_km2": m["area"],
            "trajectory_length_km": m["trajectory_length_km"],
            "mean_major_axis_km": m["major_km"],
            "mean_minor_axis_km": m["minor_km"],
            "mean_ellipse_angle_deg": m["ellipse_angle"],
            # legacy EPSG:2163 grid-plane values (math convention), for comparison only
            "mean_direction_grid_deg": m["wd_grid"] % 360.0,
            "angular_variance_grid": m["wv_grid"],
            "mean_speed_grid_ms": m["v_grid"],
            "trajectory_length_grid_km": m["length_grid_km"],
        })

    if not storm_trajectories:
        raise RuntimeError("No storms produced a valid trajectory (>=2 centroids).")

    # reproject trajectories to geographic coords
    geographic_trajectories: dict = {}
    for name, traj in storm_trajectories.items():
        lon, lat = _PCS_TO_GCS.transform(traj["x_coords"], traj["y_coords"])
        geographic_trajectories[name] = {
            "start_time_index": traj["start_time_index"],
            "end_time_index": traj["end_time_index"],
            "lon_coords": lon,
            "lat_coords": lat,
            "trajectory_length": traj["trajectory_length"],
        }

    storm_properties = pd.DataFrame(property_rows)
    df = pd.DataFrame.from_dict(storm_trajectories, orient="index")

    longest_trajectories = dict(
        sorted(storm_trajectories.items(),
               key=lambda kv: kv[1]["trajectory_length"], reverse=True)[
            : cfg.motion.n_trajectories_plotted]
    )

    b_arr = np.radians(storm_mean_bearing)
    mean_angle = float(bearing_to_math(np.degrees(np.arctan2(np.mean(np.sin(b_arr)), np.mean(np.cos(b_arr))))))
    mean_vel = float(np.mean(storm_mean_velocity))
    mean_p = float(np.mean(storm_intensity))

    return MotionSummary(
        storm_trajectories=storm_trajectories,
        geographic_trajectories=geographic_trajectories,
        storm_properties=storm_properties,
        df=df,
        longest_trajectories=longest_trajectories,
        storm_mean_bearing=storm_mean_bearing,
        storm_mean_velocity=storm_mean_velocity,
        mean_angle=mean_angle,
        mean_vel=mean_vel,
        mean_p=mean_p,
    )


# --------------------------------------------------------------------------
# Stage 3: diagnostics
# --------------------------------------------------------------------------
def generate_catalog_diagnostics(summary: MotionSummary, cfg: Config) -> dict:
    """Save the per-storm properties CSV + catalog-level figures.

    Returns ``{"figures": [png paths], "table": csv_path}``.
    """
    # import plotting lazily (heavy: cartopy/seaborn/windrose)
    from .plotting.diagnostics import (
        diagnostic_plot_storm_trajectories,
        diagnostic_plot_storm_direction_vectors,
        plot_storm_properties_pairplot,
    )

    paths = scio.ensure_output_dirs(cfg)
    save_dir = str(paths["base"])
    domain = cfg.io.domain_name
    fz = cfg.figure.font_size

    # per-storm properties table (mean speed/direction, intensity, total rainfall,
    # ellipse major/minor/angle, area, duration, ...)
    csv_path = paths["base"] / f"storm_properties_{domain}.csv"
    summary.storm_properties.to_csv(csv_path, index=False)

    transposition_domain = scio.load_vector(cfg.transposition_domain_path)
    control_area = scio.load_vector(cfg.control_area_path)
    grid = scio.load_vector(cfg.grid_path)  # None for non-CONUS catalogs

    diagnostic_plot_storm_trajectories(
        df=summary.df,
        longest_trajectories=summary.longest_trajectories,
        geographic_trajectories=summary.geographic_trajectories,
        transposition_domain=transposition_domain,
        storm_mean_direction_vector=summary.storm_mean_bearing,
        storm_mean_velocity=summary.storm_mean_velocity,
        mean_angle=summary.mean_angle,
        mean_vel=summary.mean_vel,
        mean_p_ellipse=np.float64(summary.mean_p),
        time_window=cfg.motion.intensity_window_hr,
        grid=grid,
        save_path=save_dir,
        wsh=control_area,
        smooth_factor=cfg.figure.smooth_factor,
        fig_name=domain,
        font_size=fz,
    )

    diagnostic_plot_storm_direction_vectors(
        df=summary.df,
        longest_trajectories=summary.longest_trajectories,
        geographic_trajectories=summary.geographic_trajectories,
        transposition_domain=transposition_domain,
        storm_mean_direction_vector=summary.storm_mean_bearing,
        storm_mean_velocity=summary.storm_mean_velocity,
        mean_angle=summary.mean_angle,
        mean_vel=summary.mean_vel,
        mean_p_ellipse=np.float64(summary.mean_p),
        time_window=cfg.motion.intensity_window_hr,
        grid=grid,
        save_path=save_dir,
        wsh=control_area,
        fig_name=domain,
        font_size=fz,
        arrow_scale=cfg.figure.arrow_scale,
        arrow_width=cfg.figure.arrow_width,
        head_width=cfg.figure.head_width,
        head_length=cfg.figure.head_length,
    )

    plot_storm_properties_pairplot(
        summary.storm_properties, save_path=save_dir, domain_name=domain, dpi=cfg.figure.dpi
    )

    expected = [
        paths["base"] / f"storm_trajectories_diagnostic_plot_{domain}.png",
        paths["base"] / f"storm_direction_vector_plot_{domain}.png",
        paths["base"] / f"storm_statistics_pairplot_{domain}.png",
    ]
    return {"figures": [p for p in expected if p.exists()], "table": csv_path}


def run(cfg: Config, verbose: bool = True) -> dict:
    """End-to-end: track -> summarize -> diagnostics. Returns a results dict."""
    results = run_catalog_tracking(cfg, verbose=verbose)
    summary = summarize_motion(results, cfg)
    diag = generate_catalog_diagnostics(summary, cfg)
    figures, table = diag["figures"], diag["table"]
    if verbose:
        print(f"\nSaved storm-properties table:\n  {table}")
        print(f"Saved {len(figures)} figures:")
        for p in figures:
            print(f"  {p}")
    return {"results": results, "summary": summary, "figures": figures, "table": table}


# --------------------------------------------------------------------------
# Phase 2: single-event tracking figures
# --------------------------------------------------------------------------
# window lengths (hours) for which plot_storm_time_steps has a panel layout
_TIME_STEP_WINDOWS = {6, 8, 12, 16, 24}


def track_one_event(cfg: Config, storm: str):
    """Track a single catalog event (by name/substring/id); return (results, trajectories)."""
    path = scio.resolve_event(cfg, storm)
    name = path.name
    t = cfg.tracking
    ds = scio.open_event(path, var_name=t.rainfall_var_name)
    try:
        event_dict: dict = {}
        storm_tracking_event(
            ds, event_dict,
            morph_radius=t.morph_radius_cells, high_threshold=t.rainfall_threshold_mmhr, var_name=t.rainfall_var_name,
            ratio_threshold=t.overlap_ratio_threshold, dry_spell_time=t.dry_spell_hr,
            area_fraction=cfg.selection.min_area_fraction,
        )
        if int(event_dict["longest_duration"]) < cfg.selection.min_duration_hr:
            raise RuntimeError(
                f"{name}: longest duration {event_dict['longest_duration']} "
                f"< min_duration_hr={cfg.selection.min_duration_hr}"
            )
        continuos_storm(event_dict, area_fraction=cfg.selection.min_area_fraction)
        storm_tracking_features(event_dict, ellipse_fit=t.ellipse_fit_method)
    finally:
        ds.close()

    traj = compute_storm_trajectory(
        event_dict, cfg.motion.intensity_window_hr, storm_id=_parse_storm_id(name, 0)
    )
    if traj is None:
        raise RuntimeError(f"{name}: no valid trajectory (>= 2 centroids)")
    traj.pop("_metrics", None)
    return {name: event_dict}, {name: traj}


def generate_event_diagnostics(cfg: Config, storm: str, verbose: bool = True) -> List[Path]:
    """Render the per-event figures (storm track + time-step panels) for one storm."""
    from .plotting.per_event import plot_storm_track, plot_storm_time_steps

    results, trajectories = track_one_event(cfg, storm)
    name = next(iter(results))
    paths = scio.ensure_output_dirs(cfg)

    plot_storm_track(
        storm_tracking_results=results, storm_name=name,
        storm_trajectories=trajectories, save_path=str(paths["storm_track"]),
    )

    interval = cfg.motion.intensity_window_hr
    if interval in _TIME_STEP_WINDOWS:
        st = trajectories[name]["start_time_index"]
        en = trajectories[name]["end_time_index"]
        plot_storm_time_steps(
            storm_tracking_results=results, storm_name=name,
            start_end=(st, en), time_window=interval,
            save_path=str(paths["storm_time_steps"]), font_size=cfg.figure.font_size,
        )
    elif verbose:
        print(f"note: intensity_window_hr {interval!r} not in {sorted(_TIME_STEP_WINDOWS)}; "
              "skipping the time-step panel figure")

    n = name.split("_storm_")[1].split("_")[0] if "_storm_" in name else "0"
    candidates = [
        paths["storm_track"] / f"storm_track_{n}.png",
        paths["storm_time_steps"] / f"storm_step_{n}.png",
    ]
    saved = [p for p in candidates if p.exists()]
    if verbose:
        print(f"Saved {len(saved)} figures for {name}:")
        for p in saved:
            print(f"  {p}")
    return saved


def generate_event_animation(cfg: Config, storm: str, save_path=None,
                             verbose: bool = True, **kwargs) -> str:
    """Render the storm-tracking illustration GIF for a single storm.

    Tracks one event (like ``generate_event_diagnostics``) and writes an animated
    GIF of the three-phase workflow: the storm moving, then the per-step fitted
    ellipse + accumulating centroid trail, then the full trajectory with the
    mean-direction vector and a statistics box. Extra keyword arguments pass
    through to ``plotting.animation.animate_storm_tracking`` (``fps``,
    ``rainfall_threshold``, ``hold_frames``, ``dpi``, ...).
    """
    from .plotting.animation import animate_storm_tracking

    results, trajectories = track_one_event(cfg, storm)
    name = next(iter(results))
    if save_path is None:
        save_path = scio.ensure_output_dirs(cfg)["storm_animation"]
    out = animate_storm_tracking(
        results, name, trajectories, save_path=str(save_path), **kwargs
    )
    if verbose:
        print(f"Saved storm-tracking animation for {name}:\n  {out}")
    return out


def generate_all_event_diagnostics(cfg: Config, results: Dict[str, dict],
                                   summary: "MotionSummary", verbose: bool = True) -> List[Path]:
    """Render per-event figures (storm track + time-step panels) for EVERY storm.

    Reuses the already-computed tracking ``results`` and motion ``summary`` from
    ``run()`` (no re-tracking). Saves to ``<output_dir>/<domain>/StormTrack/`` and
    ``.../StormTimeSteps/``. Slow for large catalogs (2 figures per storm) — subsample
    the catalog with ``io.max_events`` if needed.
    """
    from .plotting.per_event import plot_storm_track, plot_storm_time_steps

    paths = scio.ensure_output_dirs(cfg)
    interval = cfg.motion.intensity_window_hr
    do_steps = interval in _TIME_STEP_WINDOWS
    if not do_steps and verbose:
        print(f"note: intensity_window_hr {interval!r} not in {sorted(_TIME_STEP_WINDOWS)}; "
              "drawing storm-track figures only (no time-step panels)")

    trajectories = summary.storm_trajectories
    total = len(trajectories)
    saved: List[Path] = []
    for k, (name, traj) in enumerate(trajectories.items(), start=1):
        if name not in results:
            continue
        plot_storm_track(
            storm_tracking_results=results, storm_name=name,
            storm_trajectories=trajectories, save_path=str(paths["storm_track"]),
        )
        if do_steps:
            plot_storm_time_steps(
                storm_tracking_results=results, storm_name=name,
                start_end=(traj["start_time_index"], traj["end_time_index"]),
                time_window=interval, save_path=str(paths["storm_time_steps"]),
                font_size=cfg.figure.font_size,
            )
        sid = name.split("_storm_")[1].split("_")[0] if "_storm_" in name else str(k)
        for pth in (paths["storm_track"] / f"storm_track_{sid}.png",
                    paths["storm_time_steps"] / f"storm_step_{sid}.png"):
            if pth.exists():
                saved.append(pth)
        if verbose and (k % 25 == 0 or k == total):
            print(f"  per-event figures: {k}/{total} storms")

    if verbose:
        print(f"Saved {len(saved)} per-event figures for {total} storms under\n"
              f"  {paths['storm_track']}\n  {paths['storm_time_steps']}")
    return saved


# --------------------------------------------------------------------------
# Storm-motion direction-probability field (grid diagnostic)
# --------------------------------------------------------------------------
def build_direction_probability_field(results: Dict[str, dict], cfg: Config,
                                      verbose: bool = True):
    """Aggregate per-storm motion grids into a per-cell direction-probability field.

    Builds one storm-motion grid per storm on a **shared** projected extent (so the
    grids align), then bins the directions into ``cfg.direction_grid.n_sectors`` sectors.
    Grid resolution, sector count, and start angle come from ``cfg.direction_grid``.
    """
    from .motion.grid import build_storm_motion_grid, calculate_direction_probability_field

    dg = cfg.direction_grid
    angle_bin_size = 360.0 / dg.n_sectors
    start_angle = dg.start_angle_deg if dg.start_angle_deg is not None else -(angle_bin_size / 2.0)
    cell_size_km = dg.cell_size_km

    events = {k: v for k, v in results.items() if v and "lon_prj_array" in v}
    if not events:
        raise RuntimeError("no tracked storms with projected grids available")

    xmin = min(np.nanmin(v["lon_prj_array"]) for v in events.values())
    xmax = max(np.nanmax(v["lon_prj_array"]) for v in events.values())
    ymin = min(np.nanmin(v["lat_prj_array"]) for v in events.values())
    ymax = max(np.nanmax(v["lat_prj_array"]) for v in events.values())
    extent = (xmin, xmax, ymin, ymax)

    motion_grids = []
    for name, event_dict in events.items():
        try:
            motion_grids.append(
                build_storm_motion_grid(event_dict, cell_size_m=cell_size_km * 1000.0, extent=extent)
            )
        except Exception as exc:  # a storm may have no cells inside any ellipse
            if verbose:
                print(f"  motion grid skipped for {name}: {type(exc).__name__}")
    if not motion_grids:
        raise RuntimeError("no storm-motion grids could be built")

    if verbose:
        print(f"Built {len(motion_grids)} storm-motion grids "
              f"({cell_size_km:g} km cells); {dg.n_sectors} sectors ({angle_bin_size:g} deg each).")
    return calculate_direction_probability_field(
        motion_grids, angle_bin_size=angle_bin_size, start_angle=start_angle
    )
