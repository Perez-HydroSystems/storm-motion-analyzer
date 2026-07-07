"""Animated GIF illustrating the storm-tracking workflow for a single storm.

``animate_storm_tracking`` reads the same event dictionary and trajectory dict
produced by ``pipeline.track_one_event`` (and used by ``plotting.per_event``) and
renders a three-phase animation:

1. **Storm moving** — the rainfall field stepping through the tracked window.
2. **Tracking** — at each step the current-time-step fitted ellipse and centroid,
   while the centroid trail and the trajectory line accumulate behind it.
3. **Trajectory + motion** — the full trajectory, a mean-direction vector spanning
   the track length, and a statistics box (direction, speed, rainfall intensity,
   angular variance).

The ellipse/centroid drawing matches ``plotting.per_event.plot_storm_track``
(geographic degrees, ``matplotlib_width(m)/111000``) so the GIF is consistent with
the static per-event figures.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Ellipse

# meters per degree of latitude (same rough conversion used by per_event.py)
_M_PER_DEG = 111_000.0


def _storm_number(storm_name):
    try:
        return int(storm_name.split("_storm_")[1].split("_")[0])
    except (IndexError, ValueError):
        return storm_name


def _coords_2d(lon_array, lat_array, field_shape):
    """Return 2D lon/lat grids matching ``field_shape`` for 1D or 2D inputs."""
    lon = np.asarray(lon_array)
    lat = np.asarray(lat_array)
    if lon.ndim == 1 or lat.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon.ravel(), lat.ravel())
    else:
        lon2d, lat2d = lon, lat
    return lon2d, lat2d


def animate_storm_tracking(
    storm_tracking_results,
    storm_name,
    storm_trajectories,
    save_path=None,
    fig_name=None,
    fps=4,
    rainfall_threshold=0.5,
    pad_deg=0.4,
    hold_frames=6,
    dpi=120,
    figsize=(9, 8),
):
    """Create the storm-tracking illustration GIF for one storm.

    Parameters
    ----------
    storm_tracking_results : dict
        ``{storm_name: event_dict}`` from ``pipeline.track_one_event``.
    storm_name : str
        Key into the two dicts.
    storm_trajectories : dict
        ``{storm_name: traj}`` from ``pipeline.track_one_event`` (needs
        ``start_time_index``, ``end_time_index``, ``trajectory_length``,
        ``mean_precipitation``).
    save_path : str, optional
        Directory to write ``storm_tracking_<id>.gif`` into. If omitted, the
        ``FuncAnimation`` object is returned instead (for inline display).
    fps : int, default 4
        Frames per second of the GIF.
    rainfall_threshold : float, default 0.5
        Mask rainfall below this value (mm/h).
    pad_deg : float, default 0.4
        Map padding around the storm footprint / trajectory (degrees).
    hold_frames : int, default 6
        Extra frames held at the end of each phase (the final phase holds
        ``2 * hold_frames``) so the viewer can read each stage.
    """
    event = storm_tracking_results[storm_name]
    traj = storm_trajectories[storm_name]

    start = int(traj["start_time_index"])
    end = int(traj["end_time_index"])
    steps = list(range(start, end))          # per_event uses [start, end)
    if len(steps) < 2:
        steps = list(range(start, end + 1))
    if len(steps) < 2:
        raise ValueError(f"{storm_name}: not enough tracked time steps to animate")

    lon = np.asarray(event["storm_lon_cent"], dtype=float)
    lat = np.asarray(event["storm_lat_cent"], dtype=float)
    selected = np.asarray(event["selected_storm"], dtype=float)
    times = np.asarray(event["selected_storm_time_steps"].values)

    record = event["storm_record"].reset_index(drop=True)
    ell_w = record["matplotlib_width(m)"].to_numpy() / _M_PER_DEG
    ell_h = record["matplotlib_height(m)"].to_numpy() / _M_PER_DEG
    ell_a = record["ellipse_angle (degree)"].to_numpy()

    # motion statistics (written onto the event_dict by compute_storm_trajectory)
    wd = float(event.get("mean_direction_weighted", np.nan))
    speed = float(event.get("mean_velocity", np.nan))
    ang_var = float(event.get("var_dir_weighted", np.nan))
    mean_p = float(traj.get("mean_precipitation", np.nan))
    traj_len_deg = float(traj.get("trajectory_length", np.nan)) / _M_PER_DEG

    lon2d, lat2d = _coords_2d(event["lon_array"], event["lat_array"], selected[0].shape)

    # fixed rainfall color scale across all frames
    win = slice(steps[0], steps[-1] + 1)
    vmin = 0.2
    vmax = max(float(np.nanmax(selected[win])), 1.0)

    # mean-direction vector, centred on the trajectory midpoint
    xa, ya = float(np.mean(lon[steps])), float(np.mean(lat[steps]))
    theta = np.radians(wd) if np.isfinite(wd) else 0.0
    vec_u = np.cos(theta) * traj_len_deg if np.isfinite(traj_len_deg) else 0.0
    vec_v = np.sin(theta) * traj_len_deg if np.isfinite(traj_len_deg) else 0.0

    # map extent: storm footprint + centroid path + the mean-direction arrow
    acc = np.nansum(selected[win], axis=0)
    foot = acc > rainfall_threshold
    if foot.any():
        foot_lon, foot_lat = lon2d[foot], lat2d[foot]
    else:
        foot_lon, foot_lat = lon[steps], lat[steps]
    xs = np.concatenate([foot_lon, lon[steps], [xa - vec_u / 2, xa + vec_u / 2]])
    ys = np.concatenate([foot_lat, lat[steps], [ya - vec_v / 2, ya + vec_v / 2]])
    extent = [xs.min() - pad_deg, xs.max() + pad_deg,
              ys.min() - pad_deg, ys.max() + pad_deg]

    # frame plan: (phase, time_index)
    frames = [("move", t) for t in steps]
    frames += [("move", steps[-1])] * (hold_frames // 2)
    frames += [("track", t) for t in steps]
    frames += [("track", steps[-1])] * (hold_frames // 2)
    frames += [("final", steps[-1])] * (2 * hold_frames)

    n_storm = _storm_number(storm_name)
    phase_titles = {
        "move": "Storm event in motion",
        "track": "Tracking the storm motion",
        "final": "Extracting trajectory and computing storm motion properties",
    }

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap="Blues")
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.08)
    cbar.set_label("Rainfall (mm/h)")

    # Figure-level stage banner + info line, created once and updated per frame so
    # they never stack (ax.clear() only clears the axes, not figure-level text).
    banner = fig.text(0.45, 0.955, "", ha="center", va="center",
                      fontsize=16, fontweight="bold")
    subinfo = fig.text(0.45, 0.910, "", ha="center", va="center",
                       fontsize=10, color="0.35")

    def update(k):
        phase, t = frames[k]
        ax.clear()
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, edgecolor="gray", linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, edgecolor="gray", linewidth=0.5)
        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.4)
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 8}

        # rainfall (faded in the final, trajectory-focused phase)
        field = selected[t]
        ax.pcolormesh(
            lon2d, lat2d, np.ma.masked_where(field < rainfall_threshold, field),
            cmap="Blues", vmin=vmin, vmax=vmax, shading="auto",
            alpha=0.35 if phase == "final" else 1.0, zorder=1,
        )

        if phase in ("track", "final"):
            upto = steps if phase == "final" else [s for s in steps if s <= t]
            # centroid trail + trajectory line accumulated so far
            ax.plot(lon[upto], lat[upto], "o--", color="red", markersize=5,
                    linewidth=1.6, label="Centroid track", zorder=5)

        if phase == "track":
            # only the current time-step ellipse + its centroid
            ax.add_patch(Ellipse(
                (lon[t], lat[t]), width=ell_w[t], height=ell_h[t], angle=ell_a[t],
                edgecolor="red", facecolor="none", linewidth=2.0, zorder=6,
                label="Current ellipse",
            ))
            ax.plot(lon[t], lat[t], "o", color="darkred", markersize=8, zorder=7)

        if phase == "final":
            ax.quiver(
                xa, ya, vec_u, vec_v, angles="xy", scale_units="xy", scale=1,
                pivot="middle", color="blue", width=0.012, zorder=8,
                label="Mean direction",
            )
            # variable names in bold (mathtext), values in normal weight
            stats = "\n".join([
                r"$\mathbf{Direction}$: " + f"{wd:.1f}°",
                r"$\mathbf{Speed}$: " + f"{speed:.2f} m/s",
                r"$\mathbf{Rain\ intensity}$: " + f"{mean_p:.2f} mm/h",
                r"$\mathbf{Angular\ variance}$: " + f"{ang_var:.2f}",
            ])
            # upper-left corner: keeps the (bigger) box clear of the trajectory,
            # arrow (centred) and legend (upper-right)
            ax.text(
                0.03, 0.97, stats, transform=ax.transAxes, fontsize=14,
                va="top", ha="left", zorder=10, linespacing=1.7,
                bbox=dict(boxstyle="round,pad=0.7", facecolor="white",
                          edgecolor="black", linewidth=1.5, alpha=0.95),
            )

        ts = str(times[t])[:16]
        banner.set_text(phase_titles[phase])
        subinfo.set_text(f"Storm {n_storm}  ·  {ts}")
        if phase in ("track", "final"):
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        return []

    anim = manimation.FuncAnimation(fig, update, frames=len(frames), blit=False)

    if save_path is None:
        return anim

    os.makedirs(str(save_path), exist_ok=True)
    out = os.path.join(str(save_path), f"storm_tracking_{fig_name or n_storm}.gif")
    anim.save(out, writer=manimation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out
