"""Storm-motion grid computation.

Gridded storm-motion direction / probability fields and trajectory-direction grids.
Plotting for these lives in ``stormcatalog_analyzer.plotting.motion_grid``.
"""
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
