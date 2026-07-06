"""Plotting for the storm-motion grid / trajectory-direction grid.

Moved from motion/grid.py: these render already-computed grid dicts produced by
``stormcatalog_analyzer.motion.grid`` (build_storm_motion_grid,
calculate_direction_probability_field, build_trajectory_direction_grid, ...).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Wedge, FancyArrowPatch
from pyproj import Transformer


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
