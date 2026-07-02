"""Catalog-diagnostics pipeline.

Three stages, callable from notebooks and the CLI:

1. ``run_catalog_tracking(cfg)``  -> dict[event_name -> event_dict]
2. ``summarize_motion(results, cfg)`` -> MotionSummary (trajectories, df, aggregates)
3. ``generate_catalog_diagnostics(summary, cfg)`` -> list[Path] of saved figures

``run(cfg)`` chains all three. This replaces the ~90-line loop that was previously
copy-pasted across the notebooks.
"""
from __future__ import annotations

import copy
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
)

# Projected (EPSG:2163) -> geographic (EPSG:4326)
_PCS_TO_GCS = Transformer.from_crs("EPSG:2163", "EPSG:4326", always_xy=True)


def _parse_storm_id(name: str, fallback: int) -> int:
    try:
        return int(name.split("_storm_")[1].split("_")[0])
    except (IndexError, ValueError):
        return fallback


# --------------------------------------------------------------------------
# Stage 1: tracking
# --------------------------------------------------------------------------
def run_catalog_tracking(cfg: Config, verbose: bool = True) -> Dict[str, dict]:
    """Track every event in the catalog; return only events that pass selection."""
    t = cfg.tracking
    results: Dict[str, dict] = {}
    for event_path in scio.discover_events(cfg):
        name = event_path.name
        ds = scio.open_event(event_path, var_name=t.var_name)
        try:
            event_dict: dict = {}
            storm_tracking_event(
                ds, event_dict,
                morph_radius=t.morph_radius,
                high_threshold=t.high_threshold,
                var_name=t.var_name,
            )
            if int(event_dict["longest_duration"]) < cfg.selection.min_duration_steps:
                if verbose:
                    print(f"skip {name}: duration {event_dict['longest_duration']} "
                          f"< {cfg.selection.min_duration_steps}")
                continue
            continuos_storm(event_dict)
            storm_tracking_features(event_dict, ellipse_fit=t.ellipse_fit)
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
    storm_mean_direction_weighted: list
    storm_mean_velocity: list
    mean_angle: float
    mean_vel: float
    mean_p: float


def compute_storm_trajectory(storm: dict, interval: str, storm_id: int = 0) -> Optional[dict]:
    """Compute one storm's most-intense-window trajectory + direction/speed.

    Side effect: writes the per-storm motion metrics back into ``storm`` (the
    event_dict) because the per-event plots read them from there. Returns the public
    trajectory dict (with a private ``_metrics`` key) or None if the storm has no
    valid (>= 2 centroid) trajectory.
    """
    mean_p_ellipse = mean_precipitation_within_ellipse(storm)
    if np.nanmin(mean_p_ellipse) < 0:
        return None
    series = pd.Series(mean_p_ellipse, index=storm["selected_storm_time_steps"].values)
    _, _, _, start_idx, end_idx = get_max_accumulated_value(series, interval)

    x = storm["storm_prj_lon_cent_list"][start_idx:end_idx]
    y = storm["storm_prj_lat_cent_list"][start_idx:end_idx]
    if len(x) < 2:
        return None

    v = mean_velocity(x, y)
    d, vd = mean_direction(x, y)
    wd, wv = mean_direction_weighted(x, y)
    lm = mean_direction_ln(x, y)[0]

    # metrics needed by per-event plots live on the event_dict
    storm["mean_velocity"] = v
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
    trajectory_length_m = compute_line_length(x, y)
    return {
        "start_time_index": start_idx,
        "end_time_index": end_idx,
        "x_coords": x,
        "y_coords": y,
        "trajectory_length": trajectory_length_m,
        "mean_precipitation": mean_precip,
        "angular_variance": wv,
        "storm_id": storm_id,
        "_metrics": {
            "v": v, "d": d, "wd": wd, "wv": wv,
            "area": float(np.mean(storm["selected_storm_area"][win])),
            "intensity": mean_precip,
            "peak_intensity": float(np.nanmax(max_p)) if max_p.size else float("nan"),
            "total_rainfall": float(np.nansum(mean_p_ellipse[win])),
            "trajectory_length_km": trajectory_length_m / 1000.0,
            "major_km": float(np.nanmean(major)) / 1000.0,
            "minor_km": float(np.nanmean(minor)) / 1000.0,
            "ellipse_angle": float(np.nanmean(angle)),
            "duration_hours": int(end_idx - start_idx + 1),
            "start_time": times[start_idx].values,
            "end_time": times[end_idx].values,
        },
    }


def summarize_motion(results: Dict[str, dict], cfg: Config) -> MotionSummary:
    """Compute per-storm direction/speed/trajectory and catalog-level aggregates."""
    results = {k: v for k, v in results.items() if v}
    interval = cfg.motion.storm_interval

    storm_trajectories: dict = {}
    storm_mean_velocity: list = []
    storm_mean_direction_weighted: list = []
    storm_intensity: list = []
    property_rows: list = []

    for i, (name, storm) in enumerate(results.items()):
        sid = _parse_storm_id(name, i)
        traj = compute_storm_trajectory(storm, interval, storm_id=sid)
        if traj is None:
            continue
        m = traj.pop("_metrics")
        storm_trajectories[name] = traj
        storm_mean_velocity.append(m["v"])
        storm_mean_direction_weighted.append(m["wd"])
        storm_intensity.append(m["intensity"])
        property_rows.append({
            "storm_event": name,
            "storm_id": sid,
            "start_time": m["start_time"],
            "end_time": m["end_time"],
            "duration_hours": m["duration_hours"],
            "mean_speed_ms": m["v"],
            "mean_direction_deg": m["wd"],
            "mean_direction_vector_deg": m["d"],
            "angular_variance": m["wv"],
            "mean_intensity_mmh": m["intensity"],
            "peak_intensity_mmh": m["peak_intensity"],
            "total_rainfall_mm": m["total_rainfall"],
            "storm_area_km2": m["area"],
            "trajectory_length_km": m["trajectory_length_km"],
            "mean_major_axis_km": m["major_km"],
            "mean_minor_axis_km": m["minor_km"],
            "mean_ellipse_angle_deg": m["ellipse_angle"],
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
            : cfg.motion.top_n_trajectories]
    )

    wd_arr = np.radians(storm_mean_direction_weighted)
    mean_angle = float(np.degrees(np.arctan2(np.mean(np.sin(wd_arr)), np.mean(np.cos(wd_arr)))))
    mean_vel = float(np.mean(storm_mean_velocity))
    mean_p = float(np.mean(storm_intensity))

    return MotionSummary(
        storm_trajectories=storm_trajectories,
        geographic_trajectories=geographic_trajectories,
        storm_properties=storm_properties,
        df=df,
        longest_trajectories=longest_trajectories,
        storm_mean_direction_weighted=storm_mean_direction_weighted,
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
        storm_mean_direction_vector=summary.storm_mean_direction_weighted,
        storm_mean_velocity=summary.storm_mean_velocity,
        mean_angle=summary.mean_angle,
        mean_vel=summary.mean_vel,
        mean_p_ellipse=np.float64(summary.mean_p),
        time_window=cfg.motion.storm_interval,
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
        storm_mean_direction_vector=summary.storm_mean_direction_weighted,
        storm_mean_velocity=summary.storm_mean_velocity,
        mean_angle=summary.mean_angle,
        mean_vel=summary.mean_vel,
        mean_p_ellipse=np.float64(summary.mean_p),
        time_window=cfg.motion.storm_interval,
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
_TIME_STEP_WINDOWS = {"6H", "8H", "12H", "16H", "24H"}


def track_one_event(cfg: Config, storm: str):
    """Track a single catalog event (by name/substring/id); return (results, trajectories)."""
    path = scio.resolve_event(cfg, storm)
    name = path.name
    t = cfg.tracking
    ds = scio.open_event(path, var_name=t.var_name)
    try:
        event_dict: dict = {}
        storm_tracking_event(
            ds, event_dict,
            morph_radius=t.morph_radius, high_threshold=t.high_threshold, var_name=t.var_name,
        )
        if int(event_dict["longest_duration"]) < cfg.selection.min_duration_steps:
            raise RuntimeError(
                f"{name}: longest duration {event_dict['longest_duration']} "
                f"< min_duration_steps={cfg.selection.min_duration_steps}"
            )
        continuos_storm(event_dict)
        storm_tracking_features(event_dict, ellipse_fit=t.ellipse_fit)
    finally:
        ds.close()

    traj = compute_storm_trajectory(
        event_dict, cfg.motion.storm_interval, storm_id=_parse_storm_id(name, 0)
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

    interval = cfg.motion.storm_interval
    if interval in _TIME_STEP_WINDOWS:
        st = trajectories[name]["start_time_index"]
        en = trajectories[name]["end_time_index"]
        plot_storm_time_steps(
            storm_tracking_results=results, storm_name=name,
            start_end=(st, en), time_window=interval,
            save_path=str(paths["storm_time_steps"]), font_size=cfg.figure.font_size,
        )
    elif verbose:
        print(f"note: storm_interval {interval!r} not in {sorted(_TIME_STEP_WINDOWS)}; "
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


# --------------------------------------------------------------------------
# Phase 3: parameter sensitivity sweep
# --------------------------------------------------------------------------
_SWEEP_SETTERS = {
    "high_threshold": lambda c, v: setattr(c.tracking, "high_threshold", float(v)),
    "morph_radius": lambda c, v: setattr(c.tracking, "morph_radius", int(v)),
    "min_duration_steps": lambda c, v: setattr(c.selection, "min_duration_steps", int(v)),
    "max_events": lambda c, v: setattr(c.io, "max_events", int(v)),
}


def run_parameter_sweep(cfg: Config, param: str, values, verbose: bool = True) -> pd.DataFrame:
    """Re-run track+summarize for each value of ``param``; return aggregate stats per value.

    Supported params: ``high_threshold``, ``morph_radius``, ``min_duration_steps``,
    ``max_events``. Heavy (re-tracks the catalog per value) — use ``io.max_events`` to
    subsample for quick sensitivity checks.
    """
    if param not in _SWEEP_SETTERS:
        raise ValueError(f"param must be one of {sorted(_SWEEP_SETTERS)}")

    rows = []
    for val in values:
        c = copy.deepcopy(cfg)
        _SWEEP_SETTERS[param](c, val)
        if verbose:
            print(f"\n=== {param} = {val} ===")
        results = run_catalog_tracking(c, verbose=False)
        try:
            summary = summarize_motion(results, c)
        except RuntimeError:
            rows.append({param: val, "n_storms": 0, "mean_direction_deg": np.nan,
                         "mean_speed_ms": np.nan, "mean_intensity_mmh": np.nan,
                         "mean_angular_variance": np.nan})
            continue
        row = {
            param: val,
            "n_storms": len(summary.storm_trajectories),
            "mean_direction_deg": summary.mean_angle,
            "mean_speed_ms": summary.mean_vel,
            "mean_intensity_mmh": summary.mean_p,
            "mean_angular_variance": float(np.mean(summary.storm_properties["angular_variance"])),
        }
        rows.append(row)
        if verbose:
            print(f"  n_storms={row['n_storms']}  dir={row['mean_direction_deg']:.1f} deg  "
                  f"speed={row['mean_speed_ms']:.2f} m/s")
    return pd.DataFrame(rows)


def generate_parameter_analysis(cfg: Config, param: str, values, verbose: bool = True):
    """Run a sweep and save a CSV + comparison figure. Returns (DataFrame, [paths])."""
    from .plotting.diagnostics import plot_parameter_sweep

    sweep = run_parameter_sweep(cfg, param, values, verbose=verbose)
    paths = scio.ensure_output_dirs(cfg)
    domain = cfg.io.domain_name
    csv_path = paths["base"] / f"parameter_sweep_{param}_{domain}.csv"
    sweep.to_csv(csv_path, index=False)
    fig_path = Path(plot_parameter_sweep(sweep, param, str(paths["base"]), domain, dpi=cfg.figure.dpi))
    if verbose:
        print(f"\nSaved:\n  {csv_path}\n  {fig_path}")
    return sweep, [csv_path, fig_path]
