import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Wedge, FancyArrowPatch
from pyproj import Transformer


def _ellipse_mask(x_grid, y_grid, center_x, center_y, major_axis_m, minor_axis_m, angle_deg):
    """Return True where projected grid points fall inside a rotated ellipse."""
    a = major_axis_m / 2.0
    b = minor_axis_m / 2.0
    if not np.isfinite([center_x, center_y, a, b, angle_deg]).all() or a <= 0 or b <= 0:
        return np.zeros_like(x_grid, dtype=bool)

    angle_rad = np.radians(angle_deg)
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)

    x_shifted = x_grid - center_x
    y_shifted = y_grid - center_y
    x_rot = x_shifted * cos_t + y_shifted * sin_t
    y_rot = -x_shifted * sin_t + y_shifted * cos_t

    return (x_rot**2 / a**2 + y_rot**2 / b**2) <= 1.0


def diagnose_storm_motion_grid_alignment(event_dict, motion_grid=None):
    """
    Diagnose whether fitted ellipses and the motion grid share the same coordinates.

    This is useful when ``inside_ellipse`` has zero cells for every time step.
    It checks the projected rainfall/grid extent, ellipse-center extent, axis
    lengths, and whether the ellipse centers would fall inside the grid if x/y
    were accidentally swapped.
    """
    record = event_dict["storm_record"].reset_index(drop=True)

    if motion_grid is None:
        x_min = np.nanmin(event_dict["lon_prj_array"])
        x_max = np.nanmax(event_dict["lon_prj_array"])
        y_min = np.nanmin(event_dict["lat_prj_array"])
        y_max = np.nanmax(event_dict["lat_prj_array"])
        cell_size_m = np.nan
        inside_counts = None
    else:
        x_min = np.nanmin(motion_grid["x_edges"])
        x_max = np.nanmax(motion_grid["x_edges"])
        y_min = np.nanmin(motion_grid["y_edges"])
        y_max = np.nanmax(motion_grid["y_edges"])
        cell_size_m = motion_grid["cell_size_m"]
        inside_counts = motion_grid["inside_ellipse"].sum(axis=(1, 2))

    ellipse_x = record["ellipse_cent_prj_lon(m)"].to_numpy(dtype=float)
    ellipse_y = record["ellipse_cent_prj_lat(m)"].to_numpy(dtype=float)
    major = record["major_axis_length(m)"].to_numpy(dtype=float)
    minor = record["minor_axis_length(m)"].to_numpy(dtype=float)

    centers_inside = (
        (ellipse_x >= x_min)
        & (ellipse_x <= x_max)
        & (ellipse_y >= y_min)
        & (ellipse_y <= y_max)
    )
    centers_inside_if_swapped = (
        (ellipse_y >= x_min)
        & (ellipse_y <= x_max)
        & (ellipse_x >= y_min)
        & (ellipse_x <= y_max)
    )

    diagnostics = {
        "cell_size_m": cell_size_m,
        "grid_extent": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        },
        "ellipse_center_extent": {
            "x_min": np.nanmin(ellipse_x),
            "x_max": np.nanmax(ellipse_x),
            "y_min": np.nanmin(ellipse_y),
            "y_max": np.nanmax(ellipse_y),
        },
        "ellipse_axis_summary_m": {
            "major_min": np.nanmin(major),
            "major_median": np.nanmedian(major),
            "major_max": np.nanmax(major),
            "minor_min": np.nanmin(minor),
            "minor_median": np.nanmedian(minor),
            "minor_max": np.nanmax(minor),
        },
        "ellipse_centers_inside_grid": int(np.sum(centers_inside)),
        "ellipse_centers_total": int(len(record)),
        "ellipse_centers_inside_grid_if_xy_swapped": int(np.sum(centers_inside_if_swapped)),
        "inside_cell_counts": inside_counts,
    }

    diagnostics["likely_issue"] = None
    if np.sum(centers_inside) == 0 and np.sum(centers_inside_if_swapped) > 0:
        diagnostics["likely_issue"] = "ellipse center x/y coordinates look swapped relative to the grid"
    elif np.sum(centers_inside) == 0:
        diagnostics["likely_issue"] = "ellipse centers are outside the grid extent"
    elif motion_grid is not None and inside_counts is not None and np.all(inside_counts == 0):
        diagnostics["likely_issue"] = "ellipse centers are inside the grid, but no grid-cell centers fall inside the ellipses; check cell_size_m and ellipse axis lengths"

    return diagnostics


def build_storm_motion_grid(
    event_dict,
    cell_size_m,
    extent=None,
    include_empty_cells=False,
    require_cells_inside=False,
):
    """
    Create a projected grid and assign storm-motion direction to cells inside each ellipse.

    A cell receives the direction from the current ellipse centroid to the next
    ellipse centroid when the cell center is inside the current time-step ellipse.
    The last storm time step is omitted because it has no next centroid.

    Parameters
    ----------
    event_dict : dict
        Storm dictionary after ``storm_tracking_features``. It must contain
        ``storm_record``, ``lon_prj_array``, ``lat_prj_array``, and
        ``selected_storm_time_steps``.
    cell_size_m : float
        Grid-cell size in projected meters.
    extent : tuple, optional
        Projected extent as ``(x_min, x_max, y_min, y_max)``. If omitted, the
        full projected rainfall-domain extent is used.
    include_empty_cells : bool, default False
        If True, return one row per cell per time step, including cells outside
        the ellipse with NaN direction. If False, the tabular output keeps only
        cells inside ellipses.
    require_cells_inside : bool, default False
        If True, raise an error when no grid-cell centers fall inside any
        fitted ellipse.

    Returns
    -------
    dict
        Dictionary with grid coordinates, per-time-step 3D direction arrays,
        unit-vector components, masks, and a compact ``records`` table.

    Notes
    -----
    ``direction_deg`` follows mathematical map coordinates: 0 is east and
    angles increase counterclockwise. ``bearing_deg`` is also returned using
    compass convention: 0 is north and angles increase clockwise.
    """
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive")

    required_keys = {
        "storm_record",
        "lon_prj_array",
        "lat_prj_array",
        "selected_storm_time_steps",
    }
    missing = required_keys.difference(event_dict)
    if missing:
        raise KeyError(f"event_dict is missing required keys: {sorted(missing)}")

    record = event_dict["storm_record"].reset_index(drop=True)
    n_times = len(record)
    if n_times < 2:
        raise ValueError("At least two storm time steps are required")

    if extent is None:
        x_min = np.nanmin(event_dict["lon_prj_array"])
        x_max = np.nanmax(event_dict["lon_prj_array"])
        y_min = np.nanmin(event_dict["lat_prj_array"])
        y_max = np.nanmax(event_dict["lat_prj_array"])
    else:
        x_min, x_max, y_min, y_max = extent

    x_edges = np.arange(x_min, x_max + cell_size_m, cell_size_m)
    y_edges = np.arange(y_min, y_max + cell_size_m, cell_size_m)
    if len(x_edges) < 2 or len(y_edges) < 2:
        raise ValueError("cell_size_m is too large for the requested extent")
    x_centers = x_edges[:-1] + cell_size_m / 2.0
    y_centers = y_edges[:-1] + cell_size_m / 2.0
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)

    n_motion_steps = n_times - 1
    grid_shape = (n_motion_steps, len(y_centers), len(x_centers))
    direction_deg = np.full(grid_shape, np.nan, dtype=float)
    bearing_deg = np.full(grid_shape, np.nan, dtype=float)
    u = np.full(grid_shape, np.nan, dtype=float)
    v = np.full(grid_shape, np.nan, dtype=float)
    inside_ellipse = np.zeros(grid_shape, dtype=bool)
    step_direction_deg = np.full(n_motion_steps, np.nan, dtype=float)
    step_bearing_deg = np.full(n_motion_steps, np.nan, dtype=float)
    step_distance_m = np.full(n_motion_steps, np.nan, dtype=float)

    records = []
    times = np.asarray(event_dict["selected_storm_time_steps"].values)

    for t in range(n_motion_steps):
        cx = record.loc[t, "ellipse_cent_prj_lon(m)"]
        cy = record.loc[t, "ellipse_cent_prj_lat(m)"]
        nx = record.loc[t + 1, "ellipse_cent_prj_lon(m)"]
        ny = record.loc[t + 1, "ellipse_cent_prj_lat(m)"]

        dx = nx - cx
        dy = ny - cy
        distance_m = np.hypot(dx, dy)
        if distance_m == 0 or not np.isfinite(distance_m):
            theta = np.nan
            bearing = np.nan
            unit_u = np.nan
            unit_v = np.nan
        else:
            theta = np.degrees(np.arctan2(dy, dx)) % 360.0
            bearing = (90.0 - theta) % 360.0
            unit_u = dx / distance_m
            unit_v = dy / distance_m

        mask = _ellipse_mask(
            x_grid,
            y_grid,
            cx,
            cy,
            record.loc[t, "major_axis_length(m)"],
            record.loc[t, "minor_axis_length(m)"],
            record.loc[t, "ellipse_angle (degree)"],
        )

        inside_ellipse[t] = mask
        step_direction_deg[t] = theta
        step_bearing_deg[t] = bearing
        step_distance_m[t] = distance_m

        direction_deg[t, mask] = theta
        bearing_deg[t, mask] = bearing
        u[t, mask] = unit_u
        v[t, mask] = unit_v

        record_mask = np.ones_like(mask, dtype=bool) if include_empty_cells else mask
        rows, cols = np.where(record_mask)
        for row, col in zip(rows, cols):
            records.append(
                {
                    "time_index": t,
                    "time": times[t],
                    "next_time": times[t + 1],
                    "row": row,
                    "col": col,
                    "x_center": x_centers[col],
                    "y_center": y_centers[row],
                    "inside_ellipse": bool(mask[row, col]),
                    "direction_deg": direction_deg[t, row, col],
                    "bearing_deg": bearing_deg[t, row, col],
                    "u": u[t, row, col],
                    "v": v[t, row, col],
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "next_centroid_x": nx,
                    "next_centroid_y": ny,
                    "centroid_distance_m": distance_m,
                }
            )

    if require_cells_inside and not np.any(inside_ellipse):
        diagnostics = diagnose_storm_motion_grid_alignment(
            event_dict,
            {
                "x_edges": x_edges,
                "y_edges": y_edges,
                "cell_size_m": cell_size_m,
                "inside_ellipse": inside_ellipse,
            },
        )
        raise ValueError(
            "No grid-cell centers fall inside any fitted ellipse. "
            f"Diagnostics: {diagnostics}"
        )

    return {
        "cell_size_m": cell_size_m,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "time": times[:-1],
        "next_time": times[1:],
        "direction_deg": direction_deg,
        "bearing_deg": bearing_deg,
        "u": u,
        "v": v,
        "inside_ellipse": inside_ellipse,
        "step_direction_deg": step_direction_deg,
        "step_bearing_deg": step_bearing_deg,
        "step_distance_m": step_distance_m,
        "records": pd.DataFrame.from_records(records),
    }


def _circular_mean_deg(angles_deg):
    angles = np.asarray(angles_deg, dtype=float)
    angles = angles[np.isfinite(angles)]
    if angles.size == 0:
        return np.nan, np.nan

    angles_rad = np.radians(angles)
    sin_mean = np.mean(np.sin(angles_rad))
    cos_mean = np.mean(np.cos(angles_rad))
    mean_deg = np.degrees(np.arctan2(sin_mean, cos_mean)) % 360.0
    resultant_length = np.hypot(sin_mean, cos_mean)
    return mean_deg, resultant_length


def store_grid_directions_over_time(motion_grid, mode="all"):
    """
    Store or average all storm-motion directions assigned to each grid cell.

    Parameters
    ----------
    motion_grid : dict
        Output from ``build_storm_motion_grid``.
    mode : {"all", "mean"}, default "all"
        ``"all"`` keeps every direction that occurred in each grid cell across
        all time steps. ``"mean"`` computes a circular mean direction for each
        grid cell.

    Returns
    -------
    dict
        For ``mode="all"``, returns 2D object arrays with lists of directions,
        bearings, and time indices per cell, plus a table with one row per
        occupied cell.

        For ``mode="mean"``, returns 2D numeric arrays with circular mean
        direction, circular mean bearing, mean vector components, counts, and
        mean resultant length, plus a table with one row per occupied cell.

    Notes
    -----
    Direction averaging is circular, so directions near 359 and 1 degrees
    average to 0 degrees rather than 180 degrees.
    """
    if mode not in {"all", "mean"}:
        raise ValueError("mode must be either 'all' or 'mean'")

    direction_deg = np.asarray(motion_grid["direction_deg"], dtype=float)
    bearing_deg = np.asarray(motion_grid["bearing_deg"], dtype=float)
    u = np.asarray(motion_grid["u"], dtype=float)
    v = np.asarray(motion_grid["v"], dtype=float)
    inside = np.asarray(motion_grid["inside_ellipse"], dtype=bool)

    if direction_deg.ndim != 3:
        raise ValueError("motion_grid['direction_deg'] must be a 3D array")

    n_times, n_rows, n_cols = direction_deg.shape
    counts = np.sum(np.isfinite(direction_deg) & inside, axis=0)

    if mode == "all":
        direction_lists = np.empty((n_rows, n_cols), dtype=object)
        bearing_lists = np.empty((n_rows, n_cols), dtype=object)
        time_index_lists = np.empty((n_rows, n_cols), dtype=object)
        time_lists = np.empty((n_rows, n_cols), dtype=object)
        records = []

        for row in range(n_rows):
            for col in range(n_cols):
                valid = np.isfinite(direction_deg[:, row, col]) & inside[:, row, col]
                time_indices = np.where(valid)[0]
                directions = direction_deg[valid, row, col].tolist()
                bearings = bearing_deg[valid, row, col].tolist()
                times = motion_grid["time"][time_indices].tolist()

                direction_lists[row, col] = directions
                bearing_lists[row, col] = bearings
                time_index_lists[row, col] = time_indices.tolist()
                time_lists[row, col] = times

                if directions:
                    records.append(
                        {
                            "row": row,
                            "col": col,
                            "x_center": motion_grid["x_centers"][col],
                            "y_center": motion_grid["y_centers"][row],
                            "count": len(directions),
                            "direction_deg": directions,
                            "bearing_deg": bearings,
                            "time_index": time_indices.tolist(),
                            "time": times,
                        }
                    )

        return {
            "mode": mode,
            "cell_size_m": motion_grid["cell_size_m"],
            "x_centers": motion_grid["x_centers"],
            "y_centers": motion_grid["y_centers"],
            "x_grid": motion_grid["x_grid"],
            "y_grid": motion_grid["y_grid"],
            "count": counts,
            "direction_deg": direction_lists,
            "bearing_deg": bearing_lists,
            "time_index": time_index_lists,
            "time": time_lists,
            "records": pd.DataFrame.from_records(records),
        }

    mean_direction_deg = np.full((n_rows, n_cols), np.nan, dtype=float)
    mean_bearing_deg = np.full((n_rows, n_cols), np.nan, dtype=float)
    mean_u = np.full((n_rows, n_cols), np.nan, dtype=float)
    mean_v = np.full((n_rows, n_cols), np.nan, dtype=float)
    resultant_length = np.full((n_rows, n_cols), np.nan, dtype=float)
    records = []

    for row in range(n_rows):
        for col in range(n_cols):
            valid = np.isfinite(direction_deg[:, row, col]) & inside[:, row, col]
            if not np.any(valid):
                continue

            mean_dir, r_len = _circular_mean_deg(direction_deg[valid, row, col])
            mean_bearing = (90.0 - mean_dir) % 360.0
            mean_direction_deg[row, col] = mean_dir
            mean_bearing_deg[row, col] = mean_bearing
            mean_u[row, col] = np.nanmean(u[valid, row, col])
            mean_v[row, col] = np.nanmean(v[valid, row, col])
            resultant_length[row, col] = r_len

            records.append(
                {
                    "row": row,
                    "col": col,
                    "x_center": motion_grid["x_centers"][col],
                    "y_center": motion_grid["y_centers"][row],
                    "count": int(counts[row, col]),
                    "mean_direction_deg": mean_dir,
                    "mean_bearing_deg": mean_bearing,
                    "mean_u": mean_u[row, col],
                    "mean_v": mean_v[row, col],
                    "resultant_length": r_len,
                }
            )

    return {
        "mode": mode,
        "cell_size_m": motion_grid["cell_size_m"],
        "x_centers": motion_grid["x_centers"],
        "y_centers": motion_grid["y_centers"],
        "x_grid": motion_grid["x_grid"],
        "y_grid": motion_grid["y_grid"],
        "count": counts,
        "mean_direction_deg": mean_direction_deg,
        "mean_bearing_deg": mean_bearing_deg,
        "mean_u": mean_u,
        "mean_v": mean_v,
        "resultant_length": resultant_length,
        "records": pd.DataFrame.from_records(records),
    }


def plot_storm_motion_grid_step(
    event_dict,
    motion_grid,
    time_index,
    ax=None,
    rainfall_threshold=0.2,
    vector_stride=1,
    vector_length_fraction=0.45,
    cmap="Blues",
):
    """
    Plot rainfall, the fitted ellipse, centroid movement, and grid-cell directions.

    The plot uses the same projected coordinate system as the fitted ellipses
    (EPSG:2163 in this codebase), so distances and cell sizes are in meters.
    """
    if time_index < 0 or time_index >= len(motion_grid["time"]):
        raise IndexError("time_index is outside the motion grid time range")

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 8))

    record = event_dict["storm_record"].reset_index(drop=True)
    rainfall = np.asarray(event_dict["selected_storm"][time_index])
    masked_rainfall = np.ma.masked_where(rainfall < rainfall_threshold, rainfall)

    rain_plot = ax.pcolormesh(
        event_dict["lon_prj_array"],
        event_dict["lat_prj_array"],
        masked_rainfall,
        cmap=cmap,
        shading="auto",
    )
    plt.colorbar(rain_plot, ax=ax, label="Rainfall (mm/h)")

    ellipse = Ellipse(
        xy=(
            record.loc[time_index, "ellipse_cent_prj_lon(m)"],
            record.loc[time_index, "ellipse_cent_prj_lat(m)"],
        ),
        width=record.loc[time_index, "major_axis_length(m)"],
        height=record.loc[time_index, "minor_axis_length(m)"],
        angle=record.loc[time_index, "ellipse_angle (degree)"],
        fill=False,
        edgecolor="red",
        linewidth=2,
        label="Current ellipse",
    )
    ax.add_patch(ellipse)

    cx = record.loc[time_index, "ellipse_cent_prj_lon(m)"]
    cy = record.loc[time_index, "ellipse_cent_prj_lat(m)"]
    nx = record.loc[time_index + 1, "ellipse_cent_prj_lon(m)"]
    ny = record.loc[time_index + 1, "ellipse_cent_prj_lat(m)"]
    ax.plot([cx, nx], [cy, ny], "o-", color="black", linewidth=2, label="Ellipse centroid motion")

    mask = motion_grid["inside_ellipse"][time_index]
    stride = (slice(None, None, vector_stride), slice(None, None, vector_stride))
    quiver_mask = mask[stride]
    xq = motion_grid["x_grid"][stride][quiver_mask]
    yq = motion_grid["y_grid"][stride][quiver_mask]
    arrow_length_m = motion_grid["cell_size_m"] * vector_length_fraction
    uq = motion_grid["u"][time_index][stride][quiver_mask] * arrow_length_m
    vq = motion_grid["v"][time_index][stride][quiver_mask] * arrow_length_m
    ax.quiver(
        xq,
        yq,
        uq,
        vq,
        color="darkorange",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.005,
        label="Grid direction",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Projected x (m)")
    ax.set_ylabel("Projected y (m)")
    direction = motion_grid["step_direction_deg"][time_index]
    bearing = motion_grid["step_bearing_deg"][time_index]
    time_label = str(motion_grid["time"][time_index])[:16]
    ax.set_title(
        f"{time_label} | direction={direction:.1f} deg, bearing={bearing:.1f} deg"
    )
    ax.legend(loc="best")
    return ax


def plot_storm_motion_grid_steps(
    event_dict,
    motion_grid,
    time_indices=None,
    ncols=3,
    figsize_per_panel=(5, 5),
    **kwargs,
):
    """
    Plot several storm-motion grid verification panels.

    Extra keyword arguments are passed to ``plot_storm_motion_grid_step``.
    """
    if time_indices is None:
        time_indices = range(len(motion_grid["time"]))
    time_indices = list(time_indices)
    if not time_indices:
        raise ValueError("time_indices must contain at least one time index")

    ncols = min(ncols, len(time_indices))
    nrows = int(np.ceil(len(time_indices) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    for ax, time_index in zip(axes.ravel(), time_indices):
        plot_storm_motion_grid_step(event_dict, motion_grid, time_index, ax=ax, **kwargs)

    for ax in axes.ravel()[len(time_indices):]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig, axes


def plot_storm_motion_grid_all_directions(
    event_dict,
    motion_grid,
    ax=None,
    time_indices=None,
    direction_mode="all",
    vector_stride=1,
    vector_length_fraction=0.35,
    ellipse_alpha=0.35,
    direction_alpha=0.45,
    trajectory_color="black",
    ellipse_color="red",
    direction_color="darkorange",
    rainfall="accumulated",
    rainfall_threshold=0.2,
    cmap="Blues",
):
    """
    Plot all storm direction vectors with fitted ellipses and the storm trajectory.

    Parameters
    ----------
    event_dict : dict
        Storm dictionary after ``storm_tracking_features``.
    motion_grid : dict
        Output from ``build_storm_motion_grid``.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. If omitted, a new figure and axis are created.
    time_indices : iterable, optional
        Motion time indices to plot. If omitted, all motion steps are plotted.
        The final ellipse is also plotted when all time steps are selected.
    direction_mode : {"all", "mean"}, default "all"
        ``"all"`` plots each time-step direction. ``"mean"`` plots one
        circular-mean direction per grid cell across the selected time steps.
    vector_stride : int, default 1
        Plot every nth grid cell in x/y to reduce clutter.
    vector_length_fraction : float, default 0.35
        Arrow length as a fraction of ``cell_size_m``.
    ellipse_alpha : float, default 0.35
        Transparency for fitted ellipses.
    direction_alpha : float, default 0.45
        Transparency for grid direction arrows.
    rainfall : {"accumulated", "max", None}, default "accumulated"
        Optional rainfall background. ``"accumulated"`` sums over time,
        ``"max"`` takes the maximum over time, and ``None`` skips rainfall.
    rainfall_threshold : float, default 0.2
        Mask rainfall values below this threshold in the background.
    cmap : str, default "Blues"
        Rainfall colormap.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 9))
    if direction_mode not in {"all", "mean"}:
        raise ValueError("direction_mode must be either 'all' or 'mean'")

    n_motion_steps = len(motion_grid["time"])
    if time_indices is None:
        time_indices = list(range(n_motion_steps))
    else:
        time_indices = list(time_indices)

    if not time_indices:
        raise ValueError("time_indices must contain at least one time index")
    if min(time_indices) < 0 or max(time_indices) >= n_motion_steps:
        raise IndexError("time_indices contains an index outside the motion grid time range")

    record = event_dict["storm_record"].reset_index(drop=True)

    if rainfall is not None:
        selected_storm = np.asarray(event_dict["selected_storm"])
        if rainfall == "accumulated":
            rainfall_field = np.nansum(selected_storm, axis=0)
            rainfall_label = "Accumulated rainfall"
        elif rainfall == "max":
            rainfall_field = np.nanmax(selected_storm, axis=0)
            rainfall_label = "Maximum rainfall"
        else:
            raise ValueError("rainfall must be 'accumulated', 'max', or None")

        masked_rainfall = np.ma.masked_where(rainfall_field < rainfall_threshold, rainfall_field)
        rain_plot = ax.pcolormesh(
            event_dict["lon_prj_array"],
            event_dict["lat_prj_array"],
            masked_rainfall,
            cmap=cmap,
            shading="auto",
        )
        plt.colorbar(rain_plot, ax=ax, label=f"{rainfall_label} (mm)")

    ellipse_indices = sorted(set(time_indices + [idx + 1 for idx in time_indices]))
    for idx in ellipse_indices:
        ellipse = Ellipse(
            xy=(
                record.loc[idx, "ellipse_cent_prj_lon(m)"],
                record.loc[idx, "ellipse_cent_prj_lat(m)"],
            ),
            width=record.loc[idx, "major_axis_length(m)"],
            height=record.loc[idx, "minor_axis_length(m)"],
            angle=record.loc[idx, "ellipse_angle (degree)"],
            fill=False,
            edgecolor=ellipse_color,
            linewidth=1.5,
            alpha=ellipse_alpha,
        )
        ax.add_patch(ellipse)

    trajectory_x = record["ellipse_cent_prj_lon(m)"].to_numpy()
    trajectory_y = record["ellipse_cent_prj_lat(m)"].to_numpy()
    ax.plot(
        trajectory_x,
        trajectory_y,
        "o-",
        color=trajectory_color,
        linewidth=2,
        markersize=4,
        label="Ellipse centroid trajectory",
        zorder=5,
    )

    arrow_length_m = motion_grid["cell_size_m"] * vector_length_fraction
    stride = (slice(None, None, vector_stride), slice(None, None, vector_stride))
    quiver_handle = None

    if direction_mode == "all":
        for time_index in time_indices:
            mask = motion_grid["inside_ellipse"][time_index][stride]
            if not np.any(mask):
                continue

            xq = motion_grid["x_grid"][stride][mask]
            yq = motion_grid["y_grid"][stride][mask]
            uq = motion_grid["u"][time_index][stride][mask] * arrow_length_m
            vq = motion_grid["v"][time_index][stride][mask] * arrow_length_m
            quiver_handle = ax.quiver(
                xq,
                yq,
                uq,
                vq,
                color=direction_color,
                alpha=direction_alpha,
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.0022,
                headwidth=2.4,
                headlength=3.2,
                headaxislength=2.8,
                zorder=4,
            )
    else:
        selected_u = np.asarray(motion_grid["u"])[time_indices]
        selected_v = np.asarray(motion_grid["v"])[time_indices]
        selected_inside = np.asarray(motion_grid["inside_ellipse"])[time_indices]
        valid = np.isfinite(selected_u) & np.isfinite(selected_v) & selected_inside

        summed_u = np.where(valid, selected_u, 0.0).sum(axis=0)
        summed_v = np.where(valid, selected_v, 0.0).sum(axis=0)
        counts = valid.sum(axis=0)
        occupied = counts > 0
        mean_u = np.full_like(summed_u, np.nan, dtype=float)
        mean_v = np.full_like(summed_v, np.nan, dtype=float)
        mean_u[occupied] = summed_u[occupied] / counts[occupied]
        mean_v[occupied] = summed_v[occupied] / counts[occupied]

        vector_norm = np.hypot(mean_u, mean_v)
        usable = occupied & (vector_norm > 0)
        mean_u[usable] = mean_u[usable] / vector_norm[usable]
        mean_v[usable] = mean_v[usable] / vector_norm[usable]

        mask = usable[stride]
        if np.any(mask):
            xq = motion_grid["x_grid"][stride][mask]
            yq = motion_grid["y_grid"][stride][mask]
            uq = mean_u[stride][mask] * arrow_length_m
            vq = mean_v[stride][mask] * arrow_length_m
            quiver_handle = ax.quiver(
                xq,
                yq,
                uq,
                vq,
                color=direction_color,
                alpha=direction_alpha,
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.0022,
                headwidth=2.4,
                headlength=3.2,
                headaxislength=2.8,
                zorder=4,
            )

    ax.scatter(
        trajectory_x[0],
        trajectory_y[0],
        s=70,
        color="limegreen",
        edgecolor="black",
        zorder=6,
        label="Start",
    )
    ax.scatter(
        trajectory_x[-1],
        trajectory_y[-1],
        s=70,
        color="crimson",
        edgecolor="black",
        zorder=6,
        label="End",
    )

    if quiver_handle is not None:
        label = "Grid-cell directions" if direction_mode == "all" else "Mean grid-cell directions"
        quiver_handle.set_label(label)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Projected x (m)")
    ax.set_ylabel("Projected y (m)")
    title_mode = "All" if direction_mode == "all" else "Mean"
    ax.set_title(f"Storm Trajectory, Fitted Ellipses, and {title_mode} Grid Direction Vectors")
    ax.legend(loc="best")
    return ax


def _as_motion_grid_list(storm_motion_results):
    if isinstance(storm_motion_results, dict):
        motion_grids = []
        for value in storm_motion_results.values():
            if isinstance(value, dict) and "motion_grid" in value:
                motion_grids.append(value["motion_grid"])
            elif isinstance(value, dict) and "direction_deg" in value:
                motion_grids.append(value)
        return motion_grids
    return list(storm_motion_results)


def _angle_range_mask(angles_deg, angle_min, angle_max):
    angles = np.mod(angles_deg, 360.0)
    angle_min = angle_min % 360.0
    angle_max = angle_max % 360.0

    if angle_min == angle_max:
        return np.isfinite(angles)
    if angle_min < angle_max:
        return (angles >= angle_min) & (angles < angle_max)
    return (angles >= angle_min) | (angles < angle_max)


def calculate_direction_probability_field(
    storm_motion_results,
    angle_ranges=None,
    angle_bin_size=45,
    start_angle=0,
    use_bearing=False,
):
    """
    Calculate direction probability fields for each grid cell across many storms.

    Parameters
    ----------
    storm_motion_results : dict or iterable
        Either the dictionary created in the multi-storm loop, where each value
        contains ``"motion_grid"``, or an iterable of ``motion_grid`` objects.
        All grids must have the same x/y edges.
    angle_ranges : list of tuple, optional
        Direction ranges as ``[(min_deg, max_deg), ...]``. Ranges are
        half-open: ``min <= angle < max``. Ranges can wrap through 360, e.g.
        ``(315, 45)``. If omitted, ranges are created from
        ``angle_bin_size``.
    angle_bin_size : float, default 45
        Bin size in degrees used when ``angle_ranges`` is omitted.
    start_angle : float, default 0
        Starting angle for automatic bins when ``angle_ranges`` is omitted.
        Bins are created by adding ``angle_bin_size`` until one full circle is
        covered. For example, ``start_angle=-15`` and ``angle_bin_size=30``
        creates ``(-15, 15)``, ``(15, 45)``, ..., ``(315, 345)``.
    use_bearing : bool, default False
        If False, use mathematical direction: 0 is east, counterclockwise
        positive. If True, use bearing: 0 is north, clockwise positive.

    Returns
    -------
    dict
        Probability field with shape ``(n_ranges, y, x)``, raw bin counts,
        total counts per cell, angle ranges, and grid coordinates.
    """
    motion_grids = _as_motion_grid_list(storm_motion_results)
    if not motion_grids:
        raise ValueError("storm_motion_results does not contain any motion grids")

    automatic_angle_ranges = angle_ranges is None
    if automatic_angle_ranges:
        if angle_bin_size <= 0 or angle_bin_size > 360:
            raise ValueError("angle_bin_size must be in the interval (0, 360]")
        angle_ranges = []
        current_angle = float(start_angle)
        stop_angle = float(start_angle) + 360.0
        while current_angle < stop_angle:
            next_angle = min(current_angle + angle_bin_size, stop_angle)
            angle_ranges.append((float(current_angle), float(next_angle)))
            current_angle = next_angle
    else:
        angle_ranges = [(float(a0), float(a1)) for a0, a1 in angle_ranges]

    reference = motion_grids[0]
    x_edges = reference["x_edges"]
    y_edges = reference["y_edges"]
    x_centers = reference["x_centers"]
    y_centers = reference["y_centers"]
    n_rows = len(y_centers)
    n_cols = len(x_centers)

    bin_counts = np.zeros((len(angle_ranges), n_rows, n_cols), dtype=int)
    total_counts = np.zeros((n_rows, n_cols), dtype=int)
    direction_key = "bearing_deg" if use_bearing else "direction_deg"

    for motion_grid in motion_grids:
        if not np.allclose(motion_grid["x_edges"], x_edges) or not np.allclose(motion_grid["y_edges"], y_edges):
            raise ValueError("All motion grids must use the same x/y grid edges")

        directions = np.asarray(motion_grid[direction_key], dtype=float)
        valid = np.isfinite(directions) & np.asarray(motion_grid["inside_ellipse"], dtype=bool)
        total_counts += valid.sum(axis=0)

        for bin_index, (angle_min, angle_max) in enumerate(angle_ranges):
            in_range = _angle_range_mask(directions, angle_min, angle_max) & valid
            bin_counts[bin_index] += in_range.sum(axis=0)

    probability = np.full(bin_counts.shape, np.nan, dtype=float)
    occupied = total_counts > 0
    probability[:, occupied] = bin_counts[:, occupied] / total_counts[occupied]

    return {
        "angle_ranges": angle_ranges,
        "angle_labels": [f"{a0:g}-{a1:g} deg" for a0, a1 in angle_ranges],
        "start_angle": start_angle if automatic_angle_ranges else None,
        "angle_bin_size": angle_bin_size if automatic_angle_ranges else None,
        "use_bearing": use_bearing,
        "direction_type": "bearing" if use_bearing else "direction",
        "probability": probability,
        "bin_counts": bin_counts,
        "total_counts": total_counts,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "x_grid": reference["x_grid"],
        "y_grid": reference["y_grid"],
        "cell_size_m": reference["cell_size_m"],
    }


def plot_direction_probability_field(
    probability_field,
    bin_index=0,
    ax=None,
    cmap="viridis",
    vmin=0,
    vmax=1,
    probability_range=None,
    show_counts=False,
    min_total_count=None,
    count_threshold=None,
    show_bin_arrow=True,
    bin_arrow_size=0.12,
    bin_arrow_edgecolor="black",
    bin_arrow_linewidth=2.0,
):
    """
    Plot one direction-probability field.

    Parameters
    ----------
    probability_field : dict
        Output from ``calculate_direction_probability_field``.
    bin_index : int, default 0
        Which angle range to plot.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. If omitted, a new figure and axis are created.
    cmap : str, default "viridis"
        Probability colormap.
    vmin, vmax : float, default 0, 1
        Color limits for probability.
    probability_range : tuple, optional
        Probability range to display as ``(min_probability, max_probability)``.
        Cells outside this range are hidden. This is separate from ``vmin`` and
        ``vmax``, which only control the colorbar scaling.
    show_counts : bool, default False
        If True, overlay total direction counts as thin contours.
    min_total_count : int, optional
        Hide cells with fewer than this many total direction samples.
    count_threshold : int, optional
        Hide cells whose total direction count is less than or equal to this
        threshold. For example, ``count_threshold=100`` only shows cells with
        more than 100 samples.
    show_bin_arrow : bool, default True
        If True, draw an outline arrow at the center of the plot showing the
        representative direction of the selected angle bin.
    bin_arrow_size : float, default 0.12
        Arrow length in axes-fraction units. For example, ``0.12`` means the
        arrow spans about 12% of the axis width/height.
    bin_arrow_edgecolor : str, default "black"
        Arrow edge color. The arrow face is transparent.
    bin_arrow_linewidth : float, default 2.0
        Arrow outline width.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the plot.
    """
    probability = np.asarray(probability_field["probability"], dtype=float)
    if bin_index < 0 or bin_index >= probability.shape[0]:
        raise IndexError("bin_index is outside the probability-field range")

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 8))

    plot_data = probability[bin_index].copy()
    total_counts = np.asarray(probability_field["total_counts"])
    if min_total_count is not None:
        plot_data = np.where(total_counts >= min_total_count, plot_data, np.nan)
    if count_threshold is not None:
        plot_data = np.where(total_counts > count_threshold, plot_data, np.nan)
    if probability_range is not None:
        prob_min, prob_max = probability_range
        if prob_min > prob_max:
            raise ValueError("probability_range minimum must be <= maximum")
        plot_data = np.where((plot_data >= prob_min) & (plot_data <= prob_max), plot_data, np.nan)

    mesh = ax.pcolormesh(
        probability_field["x_edges"],
        probability_field["y_edges"],
        plot_data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    plt.colorbar(mesh, ax=ax, label="Direction probability")

    if show_counts and np.nanmax(total_counts) > 0:
        ax.contour(
            probability_field["x_grid"],
            probability_field["y_grid"],
            total_counts,
            colors="black",
            linewidths=0.7,
            alpha=0.45,
        )

    if show_bin_arrow:
        angle_min, angle_max = probability_field["angle_ranges"][bin_index]
        angle_span = (angle_max - angle_min) % 360.0
        if angle_span == 0:
            angle_span = 360.0
        bin_mid_angle = (angle_min + angle_span / 2.0) % 360.0

        if probability_field["use_bearing"]:
            plot_angle = np.radians((90.0 - bin_mid_angle) % 360.0)
        else:
            plot_angle = np.radians(bin_mid_angle)

        dx = bin_arrow_size * np.cos(plot_angle)
        dy = bin_arrow_size * np.sin(plot_angle)
        ax.annotate(
            "",
            xy=(0.5 + dx / 2.0, 0.5 + dy / 2.0),
            xytext=(0.5 - dx / 2.0, 0.5 - dy / 2.0),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-|>",
                edgecolor=bin_arrow_edgecolor,
                facecolor="none",
                linewidth=bin_arrow_linewidth,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=18,
            ),
            zorder=10,
        )

    direction_name = "Bearing" if probability_field["use_bearing"] else "Direction"
    label = probability_field["angle_labels"][bin_index]
    ax.set_title(f"{direction_name} Probability: {label}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    return ax


def _as_event_dict_items(storm_results):
    if isinstance(storm_results, dict):
        return list(storm_results.items())
    return [(str(i), event_dict) for i, event_dict in enumerate(storm_results)]


def _trajectory_xy_from_event(event_dict, centroid_source="ellipse"):
    record = event_dict["storm_record"].reset_index(drop=True)
    if centroid_source == "ellipse":
        x_col = "ellipse_cent_prj_lon(m)"
        y_col = "ellipse_cent_prj_lat(m)"
    elif centroid_source == "storm":
        x_col = "cen_prj_lon(m)"
        y_col = "cen_prj_lat(m)"
    else:
        raise ValueError("centroid_source must be either 'ellipse' or 'storm'")

    missing = [col for col in (x_col, y_col) if col not in record]
    if missing:
        raise KeyError(f"storm_record is missing required centroid columns: {missing}")

    x = record[x_col].to_numpy(dtype=float)
    y = record[y_col].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def _mean_trajectory_direction(x, y, method="weighted"):
    if len(x) < 2:
        return np.nan, np.nan, np.nan, np.nan

    dx = np.diff(x)
    dy = np.diff(y)
    segment_length = np.hypot(dx, dy)
    valid = segment_length > 0
    if not np.any(valid):
        return np.nan, np.nan, np.nan, np.nan

    angles = np.arctan2(dy[valid], dx[valid])
    if method == "weighted":
        weights = segment_length[valid]
    elif method == "unweighted":
        weights = np.ones(np.sum(valid), dtype=float)
    elif method == "end_to_end":
        total_dx = x[-1] - x[0]
        total_dy = y[-1] - y[0]
        total_length = np.hypot(total_dx, total_dy)
        if total_length == 0:
            return np.nan, np.nan, np.nan, np.nan
        direction_deg = np.degrees(np.arctan2(total_dy, total_dx)) % 360.0
        bearing_deg = (90.0 - direction_deg) % 360.0
        return direction_deg, bearing_deg, total_dx / total_length, total_dy / total_length
    else:
        raise ValueError("method must be 'weighted', 'unweighted', or 'end_to_end'")

    sin_sum = np.sum(weights * np.sin(angles))
    cos_sum = np.sum(weights * np.cos(angles))
    norm = np.hypot(cos_sum, sin_sum)
    if norm == 0:
        return np.nan, np.nan, np.nan, np.nan

    direction_deg = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360.0
    bearing_deg = (90.0 - direction_deg) % 360.0
    return direction_deg, bearing_deg, cos_sum / norm, sin_sum / norm


def build_trajectory_direction_grid(
    storm_results,
    cell_size_m,
    extent=None,
    centroid_source="ellipse",
    direction_method="weighted",
    aggregation_mode="mean",
):
    """
    Build a grid of storm-trajectory directions.

    Each storm contributes one trajectory direction. The contribution is assigned
    to the grid cell containing the trajectory centroid, computed as the mean
    projected x/y location of that storm's centroid path. If several trajectory
    centroids fall in the same cell, all directions are stored in that cell.
    With ``aggregation_mode="mean"``, the function also computes the circular
    mean trajectory direction for each occupied cell.

    Parameters
    ----------
    storm_results : dict or iterable
        Dictionary or iterable of storm event dictionaries after
        ``storm_tracking_features``.
    cell_size_m : float
        Grid-cell size in projected meters.
    extent : tuple, optional
        Projected grid extent as ``(x_min, x_max, y_min, y_max)``. If omitted,
        the extent is inferred from available projected rainfall grids, falling
        back to trajectory centroid locations.
    centroid_source : {"ellipse", "storm"}, default "ellipse"
        Which centroid path to use. ``"ellipse"`` uses fitted ellipse centers;
        ``"storm"`` uses precipitation-weighted storm centroids.
    direction_method : {"weighted", "unweighted", "end_to_end"}, default "weighted"
        How to summarize each storm trajectory direction.
    aggregation_mode : {"mean", "all"}, default "mean"
        ``"mean"`` computes one circular mean direction per occupied cell while
        also storing all individual directions. ``"all"`` stores all individual
        directions and leaves the mean fields as NaN.

    Returns
    -------
    dict
        Grid edges/centers, per-cell direction lists/counts, optional per-cell
        mean directions/vectors, and one record per storm trajectory.
    """
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive")
    if aggregation_mode not in {"mean", "all"}:
        raise ValueError("aggregation_mode must be either 'mean' or 'all'")

    event_items = _as_event_dict_items(storm_results)
    trajectory_records = []
    extent_candidates = []

    for storm_name, event_dict in event_items:
        if "storm_record" not in event_dict:
            continue

        x, y = _trajectory_xy_from_event(event_dict, centroid_source=centroid_source)
        if len(x) < 2:
            continue

        direction_deg, bearing_deg, unit_u, unit_v = _mean_trajectory_direction(
            x,
            y,
            method=direction_method,
        )
        if not np.isfinite(direction_deg):
            continue

        trajectory_x = np.nanmean(x)
        trajectory_y = np.nanmean(y)
        trajectory_records.append(
            {
                "storm_name": storm_name,
                "trajectory_centroid_x": trajectory_x,
                "trajectory_centroid_y": trajectory_y,
                "direction_deg": direction_deg,
                "bearing_deg": bearing_deg,
                "u": unit_u,
                "v": unit_v,
                "n_centroids": len(x),
            }
        )

        if "lon_prj_array" in event_dict and "lat_prj_array" in event_dict:
            extent_candidates.append(
                (
                    np.nanmin(event_dict["lon_prj_array"]),
                    np.nanmax(event_dict["lon_prj_array"]),
                    np.nanmin(event_dict["lat_prj_array"]),
                    np.nanmax(event_dict["lat_prj_array"]),
                )
            )

    if not trajectory_records:
        raise ValueError("No valid storm trajectories were found")

    if extent is None:
        if extent_candidates:
            x_min = min(item[0] for item in extent_candidates)
            x_max = max(item[1] for item in extent_candidates)
            y_min = min(item[2] for item in extent_candidates)
            y_max = max(item[3] for item in extent_candidates)
        else:
            trajectory_x = np.array([row["trajectory_centroid_x"] for row in trajectory_records])
            trajectory_y = np.array([row["trajectory_centroid_y"] for row in trajectory_records])
            x_min = np.nanmin(trajectory_x) - cell_size_m / 2.0
            x_max = np.nanmax(trajectory_x) + cell_size_m / 2.0
            y_min = np.nanmin(trajectory_y) - cell_size_m / 2.0
            y_max = np.nanmax(trajectory_y) + cell_size_m / 2.0
    else:
        x_min, x_max, y_min, y_max = extent

    x_edges = np.arange(x_min, x_max + cell_size_m, cell_size_m)
    y_edges = np.arange(y_min, y_max + cell_size_m, cell_size_m)
    if len(x_edges) < 2 or len(y_edges) < 2:
        raise ValueError("cell_size_m is too large for the requested extent")

    x_centers = x_edges[:-1] + cell_size_m / 2.0
    y_centers = y_edges[:-1] + cell_size_m / 2.0
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)
    n_rows = len(y_centers)
    n_cols = len(x_centers)

    count = np.zeros((n_rows, n_cols), dtype=int)
    sum_u = np.zeros((n_rows, n_cols), dtype=float)
    sum_v = np.zeros((n_rows, n_cols), dtype=float)
    direction_lists = np.empty((n_rows, n_cols), dtype=object)
    bearing_lists = np.empty((n_rows, n_cols), dtype=object)
    u_lists = np.empty((n_rows, n_cols), dtype=object)
    v_lists = np.empty((n_rows, n_cols), dtype=object)
    storm_name_lists = np.empty((n_rows, n_cols), dtype=object)
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            direction_lists[row_idx, col_idx] = []
            bearing_lists[row_idx, col_idx] = []
            u_lists[row_idx, col_idx] = []
            v_lists[row_idx, col_idx] = []
            storm_name_lists[row_idx, col_idx] = []
    assigned_records = []

    for row in trajectory_records:
        col_idx = np.searchsorted(x_edges, row["trajectory_centroid_x"], side="right") - 1
        row_idx = np.searchsorted(y_edges, row["trajectory_centroid_y"], side="right") - 1
        if row_idx < 0 or row_idx >= n_rows or col_idx < 0 or col_idx >= n_cols:
            continue

        count[row_idx, col_idx] += 1
        sum_u[row_idx, col_idx] += row["u"]
        sum_v[row_idx, col_idx] += row["v"]
        direction_lists[row_idx, col_idx].append(row["direction_deg"])
        bearing_lists[row_idx, col_idx].append(row["bearing_deg"])
        u_lists[row_idx, col_idx].append(row["u"])
        v_lists[row_idx, col_idx].append(row["v"])
        storm_name_lists[row_idx, col_idx].append(row["storm_name"])
        assigned = row.copy()
        assigned["row"] = row_idx
        assigned["col"] = col_idx
        assigned["cell_x_center"] = x_centers[col_idx]
        assigned["cell_y_center"] = y_centers[row_idx]
        assigned_records.append(assigned)

    mean_u = np.full((n_rows, n_cols), np.nan, dtype=float)
    mean_v = np.full((n_rows, n_cols), np.nan, dtype=float)
    mean_direction_deg = np.full((n_rows, n_cols), np.nan, dtype=float)
    mean_bearing_deg = np.full((n_rows, n_cols), np.nan, dtype=float)
    resultant_length = np.full((n_rows, n_cols), np.nan, dtype=float)

    occupied = count > 0
    if aggregation_mode == "mean":
        vector_norm = np.hypot(sum_u, sum_v)
        usable = occupied & (vector_norm > 0)
        mean_u[usable] = sum_u[usable] / vector_norm[usable]
        mean_v[usable] = sum_v[usable] / vector_norm[usable]
        mean_direction_deg[usable] = np.degrees(np.arctan2(mean_v[usable], mean_u[usable])) % 360.0
        mean_bearing_deg[usable] = (90.0 - mean_direction_deg[usable]) % 360.0
        resultant_length[occupied] = vector_norm[occupied] / count[occupied]

    return {
        "cell_size_m": cell_size_m,
        "centroid_source": centroid_source,
        "direction_method": direction_method,
        "aggregation_mode": aggregation_mode,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "count": count,
        "direction_deg": direction_lists,
        "bearing_deg": bearing_lists,
        "u": u_lists,
        "v": v_lists,
        "storm_name": storm_name_lists,
        "mean_direction_deg": mean_direction_deg,
        "mean_bearing_deg": mean_bearing_deg,
        "mean_u": mean_u,
        "mean_v": mean_v,
        "resultant_length": resultant_length,
        "records": pd.DataFrame.from_records(assigned_records),
    }


def plot_trajectory_direction_grid(
    trajectory_grid,
    ax=None,
    cmap="viridis",
    count_vmin=None,
    count_vmax=None,
    vector_mode="mean",
    vector_stride=1,
    vector_length_fraction=0.45,
    vector_color="black",
    vector_width=0.0022,
    all_vector_offset_fraction=0.18,
    show_colorbar=True,
):
    """
    Plot a trajectory-direction grid.

    Cell color shows the number of storm trajectory centroids that fall in each
    cell. Vectors show either the circular mean direction per occupied cell or
    all individual trajectory directions assigned to each cell.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 9))
    if vector_mode not in {"mean", "all"}:
        raise ValueError("vector_mode must be either 'mean' or 'all'")

    count = np.asarray(trajectory_grid["count"], dtype=float)
    count_plot = np.where(count > 0, count, np.nan)
    mesh = ax.pcolormesh(
        trajectory_grid["x_edges"],
        trajectory_grid["y_edges"],
        count_plot,
        cmap=cmap,
        vmin=count_vmin,
        vmax=count_vmax,
        shading="auto",
    )
    if show_colorbar:
        plt.colorbar(mesh, ax=ax, label="Trajectory centroid count")

    arrow_length_m = trajectory_grid["cell_size_m"] * vector_length_fraction
    if vector_mode == "mean":
        stride = (slice(None, None, vector_stride), slice(None, None, vector_stride))
        mask = (
            (count > 0)[stride]
            & np.isfinite(trajectory_grid["mean_u"][stride])
            & np.isfinite(trajectory_grid["mean_v"][stride])
        )
        if np.any(mask):
            xq = trajectory_grid["x_grid"][stride][mask]
            yq = trajectory_grid["y_grid"][stride][mask]
            uq = trajectory_grid["mean_u"][stride][mask] * arrow_length_m
            vq = trajectory_grid["mean_v"][stride][mask] * arrow_length_m
            ax.quiver(
                xq,
                yq,
                uq,
                vq,
                color=vector_color,
                angles="xy",
                scale_units="xy",
                scale=1,
                width=vector_width,
                headwidth=2.4,
                headlength=3.2,
                headaxislength=2.8,
                label="Mean trajectory direction",
                zorder=4,
            )
    else:
        x_values = []
        y_values = []
        u_values = []
        v_values = []
        offset_radius = trajectory_grid["cell_size_m"] * all_vector_offset_fraction

        for row_idx in range(0, count.shape[0], vector_stride):
            for col_idx in range(0, count.shape[1], vector_stride):
                cell_u = trajectory_grid["u"][row_idx, col_idx]
                cell_v = trajectory_grid["v"][row_idx, col_idx]
                n_vectors = len(cell_u)
                if n_vectors == 0:
                    continue

                if n_vectors == 1:
                    offsets_x = [0.0]
                    offsets_y = [0.0]
                else:
                    offset_angles = np.linspace(0, 2 * np.pi, n_vectors, endpoint=False)
                    offsets_x = offset_radius * np.cos(offset_angles)
                    offsets_y = offset_radius * np.sin(offset_angles)

                for i in range(n_vectors):
                    x_values.append(trajectory_grid["x_grid"][row_idx, col_idx] + offsets_x[i])
                    y_values.append(trajectory_grid["y_grid"][row_idx, col_idx] + offsets_y[i])
                    u_values.append(cell_u[i] * arrow_length_m)
                    v_values.append(cell_v[i] * arrow_length_m)

        if x_values:
            ax.quiver(
                np.asarray(x_values),
                np.asarray(y_values),
                np.asarray(u_values),
                np.asarray(v_values),
                color=vector_color,
                angles="xy",
                scale_units="xy",
                scale=1,
                width=vector_width,
                headwidth=2.4,
                headlength=3.2,
                headaxislength=2.8,
                label="Trajectory directions",
                zorder=4,
            )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Projected x (m)")
    ax.set_ylabel("Projected y (m)")
    title_mode = "Mean" if vector_mode == "mean" else "All"
    ax.set_title(f"Storm Trajectory Direction Grid ({title_mode} Vectors)")
    ax.legend(loc="best")
    return ax


def plot_cardinal_direction_sectors(
    ax=None,
    sector_ranges=None,
    sector_labels=None,
    sector_colors=None,
    arrow_color="black",
    arrow_scale=18,
    radius=1.0,
    alpha=0.35,
    edgecolor="black",
    linewidth=1.2,
    label_radius=0.62,
):
    """
    Plot a circle divided into cardinal direction sectors.

    Angles use the mathematical convention used elsewhere in this module:
    0 degrees points east and angles increase counterclockwise.

    By default the sectors are:
    - East: -45 to 45 degrees
    - North: 45 to 135 degrees
    - West: 135 to 225 degrees
    - South: 225 to 315 degrees
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    if sector_ranges is None:
        sector_ranges = [(-45, 45), (45, 135), (135, 225), (225, 315)]
    if sector_labels is None:
        sector_labels = ["E", "N", "W", "S"]
    if sector_colors is None:
        sector_colors = ["#f4a261", "#2a9d8f", "#457b9d", "#e76f51"]

    if not (len(sector_ranges) == len(sector_labels) == len(sector_colors)):
        raise ValueError("sector_ranges, sector_labels, and sector_colors must have the same length")

    for (theta1, theta2), label, color in zip(sector_ranges, sector_labels, sector_colors):
        wedge = Wedge(
            center=(0, 0),
            r=radius,
            theta1=theta1,
            theta2=theta2,
            facecolor=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
        )
        ax.add_patch(wedge)

        theta_mid = (theta1 + ((theta2 - theta1) % 360.0) / 2.0) % 360.0
        theta_rad = np.radians(theta_mid)
        dx = 0.42 * radius * np.cos(theta_rad)
        dy = 0.42 * radius * np.sin(theta_rad)
        arrow = FancyArrowPatch(
            posA=(0, 0),
            posB=(dx, dy),
            arrowstyle="-|>",
            mutation_scale=arrow_scale,
            linewidth=linewidth + 0.4,
            edgecolor=arrow_color,
            facecolor="none",
            zorder=5,
        )
        ax.add_patch(arrow)

        ax.text(
            label_radius * radius * np.cos(theta_rad),
            label_radius * radius * np.sin(theta_rad),
            label,
            ha="center",
            va="center",
            fontsize=13,
            weight="bold",
            zorder=6,
        )

    outer_circle = plt.Circle((0, 0), radius, fill=False, edgecolor=edgecolor, linewidth=linewidth)
    ax.add_patch(outer_circle)
    #ax.axhline(0, color=edgecolor, linewidth=0.8, alpha=0.5)
    #ax.axvline(0, color=edgecolor, linewidth=0.8, alpha=0.5)
    ax.set_xlim(-1.1 * radius, 1.1 * radius)
    ax.set_ylim(-1.1 * radius, 1.1 * radius)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return ax


def set_projected_axis_geographic_ticks(
    ax,
    source_crs="EPSG:2163",
    n_ticks=5,
    lon_format="{:.1f}°",
    lat_format="{:.1f}°",
    set_axis_labels=True,
):
    """
    Label a projected x/y axis with approximate lon/lat tick labels.

    The plotted data remain in the projected coordinate system. Tick labels are
    selected in geographic coordinates and then projected back to the current
    axis limits. This avoids repeated or missing labels on small domains.
    """
    to_geo = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    from_geo = Transformer.from_crs("EPSG:4326", source_crs, always_xy=True)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_ref = 0.5 * (x_min + x_max)
    y_ref = 0.5 * (y_min + y_max)

    lon_min, _ = to_geo.transform(x_min, y_ref)
    lon_max, _ = to_geo.transform(x_max, y_ref)
    _, lat_min = to_geo.transform(x_ref, y_min)
    _, lat_max = to_geo.transform(x_ref, y_max)

    lon_labels = np.linspace(lon_min, lon_max, n_ticks)
    lat_labels = np.linspace(lat_min, lat_max, n_ticks)

    _, center_lat = to_geo.transform(x_ref, y_ref)
    center_lon, _ = to_geo.transform(x_ref, y_ref)
    x_ticks, _ = from_geo.transform(lon_labels, np.full_like(lon_labels, center_lat))
    _, y_ticks = from_geo.transform(np.full_like(lat_labels, center_lon), lat_labels)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([lon_format.format(lon) for lon in lon_labels])
    ax.set_yticklabels([lat_format.format(lat) for lat in lat_labels])
    ax.tick_params(axis="both", which="major", labelbottom=True, labelleft=True)

    if set_axis_labels:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    return ax
