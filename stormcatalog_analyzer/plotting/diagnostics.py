"""Catalog-level diagnostic figures.

Migrated from functions/diagnostic_plots.py during the package restructure.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from skimage import measure
import seaborn as sns
from windrose import WindroseAxes
from scipy import stats
from scipy.interpolate import splprep, splev
from pyproj import Transformer

from ..motion.direction import compute_line_length
from .helpers import _plot_geographic_outline, _annotate_conus_grid

def diagnostic_plot_storm_trajectories(df, longest_trajectories, geographic_trajectories,  transposition_domain, 
                                       storm_mean_direction_vector, storm_mean_velocity, mean_angle, mean_vel, 
                                       mean_p_ellipse, time_window, grid, save_path, wsh = None, smooth_factor= 0.1, fig_name=None,
                                       font_size=10):
    """
    Generates diagnostic plots for storm trajectories and wind rose.

    Parameters:
    - df (DataFrame): DataFrame containing storm data.
    - longest_trajectories (dict): Dictionary of longest storm trajectories.
    - geographic_trajectories (dict): Dictionary of geographic trajectories.
    - wsh (GeoDataFrame): GeoDataFrame of the control area.
    - transposition_domain (GeoDataFrame): GeoDataFrame of the transposition domain.
    - storm_mean_direction_vector (array): Array of storm mean direction vectors.
    - storm_mean_velocity (array): Array of storm mean velocities.
    - mean_angle (float): Mean storm direction angle.
    - mean_vel (float): Mean storm velocity.
    - mean_p_ellipse (array): Mean precipitation within the ellipse.
    - time_window (str): Time window for storm duration.
    """
    # Structured layout to avoid axes/text overlapping
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(
        2, 3, figure=fig, width_ratios=[2.4, 1, 1], height_ratios=[2, 1],
        wspace=0.35, hspace=0.35
    )
    # Main trajectories map spans both rows of the first column
    ax1 = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
    ax1.set_title(
        f'Storm Trajectories (Storm duration: {time_window})\nStorm events: {len(longest_trajectories)}',
        fontsize=font_size + 14
    )

    # Plot basin, transposed basin, and transposition domain
    if wsh is not None:
        _plot_geographic_outline(wsh, ax1, label="Control Area", edgecolor='red', linewidth=1.5, zorder=2)
    _plot_geographic_outline(transposition_domain, ax1, label="Transposition Domain", edgecolor='green', linestyle='--', linewidth=1.5)

    # Add grid with labeled coordinates
    gl1 = ax1.gridlines(draw_labels=True, linestyle='--')
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xlabel_style = {'size': 10}  # Set font size for x-axis labels
    gl1.ylabel_style = {'size': 10}  # Set font size for y-axis labels

    # Add U.S. state boundaries
    ax1.add_feature(cfeature.STATES, edgecolor='grey', linewidth=1)

    max_precip = df.loc[longest_trajectories.keys(), 'mean_precipitation'].max()
    min_precip = df.loc[longest_trajectories.keys(), 'mean_precipitation'].min()

    
      # skip negative precipitation values

    # Create a colormap for the trajectories
    norm = Normalize(vmin=min_precip, vmax=max_precip)
    cmap = plt.get_cmap('Blues')

    # Plot the longest 100 trajectories
    for storm_event in longest_trajectories.keys():
        traj = geographic_trajectories[storm_event]
        x_coords = traj['lon_coords']
        y_coords = traj['lat_coords']
        color_value = df.loc[storm_event, 'mean_precipitation']  # Use trajectory length for color mapping
        color = cmap(norm(color_value))  # Normalize and map to color
        unique_points = np.unique(np.array([x_coords, y_coords]), axis=1)
        # Use splines to smooth the trajectory
        if len(x_coords) > 3:  # Ensure there are enough points for spline fitting
            tck, u = splprep(unique_points, s=smooth_factor)
            smooth_coords = splev(np.linspace(0, 1, 100), tck)
            ax1.plot(smooth_coords[0], smooth_coords[1], color=color, label=storm_event[-20:-12], zorder=1)
        else:
            # If not enough points, plot as is
            ax1.plot(x_coords, y_coords, color=color, label=storm_event[-20:-12], zorder=1)

    # Set extent to 0.5 degrees around the transposition domain
    tp_bounds = transposition_domain.total_bounds  # [minx, miny, maxx, maxy]
    lon_min, lat_min, lon_max, lat_max = tp_bounds
    buffer = 0.1
    ax1.set_extent([lon_min - buffer, lon_max + buffer, lat_min - buffer, lat_max + buffer], crs=ccrs.PlateCarree())
    
    # Create a ScalarMappable for the colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # Set an empty array to avoid warnings

    # Add the colorbar to the plot
    cbar = plt.colorbar(sm, ax=ax1, label='Intensity [mm/h]', orientation='vertical', fraction=0.03, pad=0.02)

    # Create a legend using custom handles
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, linestyle='--', label="Transposition Domain"),
        Line2D([0], [0], color='lightblue', lw=2, label="Storm Trajectories"),
    ]

    # Add the legend to ax1
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Labels for rose plots
    new_labels = ["E", "N-E", "N", "N-W", "W", "S-W", "S", "S-E"]

    
    # Create a smaller subplot for the wind rose plot
    rect_speed = gs[0, 1].get_position(fig).bounds
    ax2 = WindroseAxes.from_ax(fig=fig, rect=rect_speed, theta_labels=new_labels)
    mean_dir = np.asarray(storm_mean_direction_vector) % 360
    st_dir_vector = (90 - mean_dir) % 360 # Convert to meteorological convention
    bins = np.arange(0, np.max(storm_mean_velocity), 5)
    ax2.bar(st_dir_vector, storm_mean_velocity, normed=True, bins=bins, cmap=plt.get_cmap("cool"))
    ax2.set_legend(title='Storm Speed (m/s)', loc='upper center', bbox_to_anchor=(0.5, -0.15))
    # Hide default radial tick labels
    ax2.set_yticklabels([])

    # Define your custom radial ticks (e.g., 5%, 10%, 15%)
    r_ticks = ax2.get_yticks()
    for r in r_ticks:
        if r == 0:
            continue  # skip center label
        ax2.text(np.pi/2, r, f"{r:.0f}%", ha='left', va='bottom', fontsize=9)

    # Create a smaller subplot using the mean precipitation within the ellipse
    rect_intensity = gs[0, 2].get_position(fig).bounds
    ax2 = WindroseAxes.from_ax(fig=fig, rect=rect_intensity, theta_labels=new_labels)
    bins = np.linspace(0, df['mean_precipitation'].max(), 5)
    mean_dir = np.asarray(storm_mean_direction_vector) % 360
    st_dir_vector = (90 - mean_dir) % 360
    ax2.bar(st_dir_vector, df['mean_precipitation'], normed=True, bins=bins, cmap=plt.get_cmap("jet"))
    ax2.set_legend(title='Intensity (mm/h)', loc='upper center', bbox_to_anchor=(0.5, -0.15))
    ax2.legend_.get_title().set_horizontalalignment('center')
    # Hide default radial tick labels
    ax2.set_yticklabels([])

    # Define your custom radial ticks (e.g., 5%, 10%, 15%)
    r_ticks = ax2.get_yticks()
    for r in r_ticks:
        if r == 0:
            continue  # skip center label
        ax2.text(np.pi/2, r, f"{r:.0f}%", ha='left', va='bottom', fontsize = 9 + font_size)
        
    # Add a text box above the trajectories
    mean_direction_text = f"Mean Storm Direction: {mean_angle:.2f}°"
    mean_speed_text = f"Mean Storm Speed: {mean_vel:.2f} m/s"
    mean_precipitation_within_ellipse_text = f"Mean Storm Intensity: {mean_p_ellipse.mean():.2f} mm/h"

    # Position the text box above  rose plots
    x_pos = 0.75  # Centered above the two rose plots
    y_pos = 0.85  # Slightly above the rose plots

    fig.text(x_pos, y_pos, f"{mean_direction_text}\n{mean_speed_text}\n{mean_precipitation_within_ellipse_text}",
             fontsize=12, color='black', ha='center', va='center',
             bbox=dict(facecolor='white', edgecolor='black'))

    # add a location plot for the transposition domain
    # Right subplot (1/3 width)
    ax4 = fig.add_subplot(gs[1, 1:], projection=ccrs.PlateCarree())
    ax4.set_title('Transposition Domain Location', fontsize=12 + font_size)
    # Set extent to show the contiguous United States
    ax4.set_extent([-130, -66, 25, 53], crs=ccrs.PlateCarree())
    # Plot the transposition domain
    _plot_geographic_outline(transposition_domain, ax4, label="Transposition Domain", edgecolor='green', linewidth=2)
    # Add grid with labeled coordinates
    gl4 = ax4.gridlines(draw_labels=True, linestyle='--')
    gl4.top_labels = False
    gl4.right_labels = False
    gl4.xlabel_style = {'size': 10}  # Set font size for x-axis labels
    gl4.ylabel_style = {'size': 10}  # Set font size for y-axis labels
    # Add U.S. state boundaries
    ax4.add_feature(cfeature.STATES, edgecolor='lightgray', linewidth=0.5)
    # CONUS 5x5 grid index labels (drawn only when a CONUS reference grid is given)
    if grid is not None:
        _annotate_conus_grid(ax4, grid, font_size)


    # Show plot
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path+f'/storm_trajectories_diagnostic_plot_{fig_name}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def diagnostic_plot_storm_direction_vectors(df, longest_trajectories, geographic_trajectories, transposition_domain,
                                            storm_mean_direction_vector, storm_mean_velocity, mean_angle, mean_vel,
                                            mean_p_ellipse, time_window, grid, save_path, wsh=None, fig_name=None, colorbar_var='mean_precipitation',
                                            font_size=10, arrow_scale = 1, arrow_width = 0.05, head_width=3, head_length=4):
    """
    Same layout as diagnostic_plot_storm_trajectories but draws a single mean-direction
    vector for each storm. The vector magnitude equals the distance between the first
    and last centroid in the trajectory.

    Parameters mirror diagnostic_plot_storm_trajectories. Mean directions are taken from
    `storm_mean_direction_vector` (aligned to df.index) or, if lengths mismatch, from
    df columns 'mean_direction_weighted'/'mean_direction' when available.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(
        2, 3, figure=fig, width_ratios=[2.4, 1, 1], height_ratios=[2, 1],
        wspace=0.35, hspace=0.35
    )
    ax1 = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
    ax1.set_title(
        f'Storm Mean Direction Vectors (Storm duration: {time_window})\nStorm events: {len(longest_trajectories)}',
        fontsize=font_size + 14
    )

    if wsh is not None:
        _plot_geographic_outline(wsh, ax1, label="Control Area", edgecolor='red', linewidth=1.5, zorder=2)
    _plot_geographic_outline(transposition_domain, ax1, label="Transposition Domain", edgecolor='green', linestyle='--', linewidth=1.5)

    gl1 = ax1.gridlines(draw_labels=True, linestyle='--')
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xlabel_style = {'size': 10}
    gl1.ylabel_style = {'size': 10}
    ax1.add_feature(cfeature.STATES, edgecolor='grey', linewidth=1)

    if colorbar_var == 'mean_precipitation':
        max_precip = df.loc[longest_trajectories.keys(), 'mean_precipitation'].max()
        min_precip = df.loc[longest_trajectories.keys(), 'mean_precipitation'].min()
        norm = Normalize(vmin=min_precip, vmax=max_precip)
        cmap = plt.get_cmap('Blues')
        label = 'Intensity [mm/h]'
    else:
        max_var = 1
        min_var = 0
        norm = Normalize(vmin=min_var, vmax=max_var)
        cmap = plt.get_cmap('Reds')
        label = 'Angular Variance'

    # map storm id -> mean direction (deg)
    direction_lookup = {}
    if storm_mean_direction_vector is not None and len(storm_mean_direction_vector) == len(df.index):
        direction_lookup = dict(zip(df.index, np.asarray(storm_mean_direction_vector)))
    if not direction_lookup:
        for col in ('mean_direction_weighted', 'mean_direction'):
            if col in df.columns:
                direction_lookup = df[col].to_dict()
                break

    for storm_event in longest_trajectories.keys():
        traj = geographic_trajectories[storm_event]
        x_coords = np.asarray(traj['lon_coords'])
        y_coords = np.asarray(traj['lat_coords'])
        if x_coords.size < 2 or y_coords.size < 2:
            continue
        traj_len = np.hypot(x_coords[-1] - x_coords[0], y_coords[-1] - y_coords[0])
        if traj_len == 0:
            continue
        direction = direction_lookup.get(storm_event, np.nan)
        if np.isnan(direction):
            continue

        if colorbar_var == 'mean_precipitation':
            color_value = df.loc[storm_event, 'mean_precipitation']
        else:
            color_value = df.loc[storm_event, 'angular_variance']
        
        color = cmap(norm(color_value))
        theta = np.radians(direction % 360)
        dx = np.cos(theta) * traj_len
        dy = np.sin(theta) * traj_len
        ax1.quiver(
            x_coords[0], y_coords[0], dx, dy,
            angles='xy', scale_units='xy', scale=arrow_scale, color=color,
            width=arrow_width, headwidth=head_width, headlength=head_length, zorder=2
        )
        #ax1.scatter(x_coords[0], y_coords[0], s=12, color=color, zorder=3)

    tp_bounds = transposition_domain.total_bounds
    lon_min, lat_min, lon_max, lat_max = tp_bounds
    buffer = 0.1
    ax1.set_extent([lon_min - buffer, lon_max + buffer, lat_min - buffer, lat_max + buffer], crs=ccrs.PlateCarree())

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, label=label, orientation='vertical', fraction=0.05, pad=0.02)

    legend_elements = [
        Line2D([0], [0], color='green', lw=2, linestyle='--', label="Transposition Domain"),
        Line2D([0], [0], color='lightblue', lw=2, label="Mean Direction Vectors"),
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)

    new_labels = ["E", "N-E", "N", "N-W", "W", "S-W", "S", "S-E"]
    rect_speed = gs[0, 1].get_position(fig).bounds
    ax2 = WindroseAxes.from_ax(fig=fig, rect=rect_speed, theta_labels=new_labels)
    mean_dir = np.asarray(storm_mean_direction_vector) % 360
    st_dir_vector = (90 - mean_dir) % 360
    bins = np.arange(0, np.max(storm_mean_velocity), 5)
    ax2.bar(st_dir_vector, storm_mean_velocity, normed=True, bins=bins, cmap=plt.get_cmap("cool"))
    ax2.set_legend(title='Storm Speed (m/s)', loc='upper center', bbox_to_anchor=(0.5, -0.15))
    ax2.set_yticklabels([])
    for r in ax2.get_yticks():
        if r == 0:
            continue
        ax2.text(np.pi/2, r, f"{r:.0f}%", ha='left', va='bottom', fontsize=9)

    rect_intensity = gs[0, 2].get_position(fig).bounds
    ax3 = WindroseAxes.from_ax(fig=fig, rect=rect_intensity, theta_labels=new_labels)
    bins = np.linspace(0, df['mean_precipitation'].max(), 5)
    mean_dir = np.asarray(storm_mean_direction_vector) % 360
    st_dir_vector = (90 - mean_dir) % 360
    ax3.bar(st_dir_vector, df['mean_precipitation'], normed=True, bins=bins, cmap=plt.get_cmap("jet"))
    ax3.set_legend(title='Intensity (mm/h)', loc='upper center', bbox_to_anchor=(0.5, -0.15))
    ax3.legend_.get_title().set_horizontalalignment('center')
    ax3.set_yticklabels([])
    for r in ax3.get_yticks():
        if r == 0:
            continue
        ax3.text(np.pi/2, r, f"{r:.0f}%", ha='left', va='bottom', fontsize=9 + font_size)

    mean_direction_text = f"Mean Storm Direction: {mean_angle:.2f}°"
    mean_speed_text = f"Mean Storm Speed: {mean_vel:.2f} m/s"
    mean_precipitation_within_ellipse_text = f"Mean Storm Intensity: {mean_p_ellipse.mean():.2f} mm/h"
    mean_variance_text = f"Mean Angular Variance: {df['angular_variance'].mean():.2f}"
    fig.text(
        0.75, 0.85,
        f"{mean_direction_text}\n{mean_speed_text}\n{mean_precipitation_within_ellipse_text}\n{mean_variance_text}",
        fontsize=12, color='black', ha='center', va='center',
        bbox=dict(facecolor='white', edgecolor='black')
    )

    ax4 = fig.add_subplot(gs[1, 1:], projection=ccrs.PlateCarree())
    ax4.set_title('Transposition Domain Location', fontsize=12 + font_size)
    ax4.set_extent([-130, -66, 25, 53], crs=ccrs.PlateCarree())
    _plot_geographic_outline(transposition_domain, ax4, label="Transposition Domain", edgecolor='green', linewidth=2)
    gl4 = ax4.gridlines(draw_labels=True, linestyle='--')
    gl4.top_labels = False
    gl4.right_labels = False
    gl4.xlabel_style = {'size': 10}
    gl4.ylabel_style = {'size': 10}
    ax4.add_feature(cfeature.STATES, edgecolor='lightgray', linewidth=0.5)
    # CONUS 5x5 grid index labels (drawn only when a CONUS reference grid is given)
    if grid is not None:
        _annotate_conus_grid(ax4, grid, font_size)

    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path+f'/storm_direction_vector_plot_{fig_name}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_storm_trajectories(df, geographic_trajectories, transposition_domain, longest_trajectories=None,
                            wsh=None, smooth_factor=0.1, figsize=(12, 8), font_size=10,
                            save_path=None, fig_name=None, buffer=0.1):
    """
    Plot storm trajectories only (no roses or ancillary panels).

    Parameters:
    - df (DataFrame): DataFrame containing storm data with a 'mean_precipitation' column.
    - geographic_trajectories (dict): Dict keyed by storm_id with 'lon_coords' and 'lat_coords'.
    - transposition_domain (GeoDataFrame): Domain polygon.
    - longest_trajectories (dict, optional): Keys to plot; defaults to all geographic_trajectories.
    - wsh (GeoDataFrame, optional): Control area outline.
    - smooth_factor (float): Spline smoothness passed to splprep.
    - figsize (tuple): Matplotlib figure size.
    - font_size (int): Base font size.
    - save_path (str, optional): Directory to save the figure.
    - fig_name (str, optional): Figure name suffix; required when save_path is given.
    - buffer (float): Degrees padded around the domain extent.
    """
    keys_to_plot = longest_trajectories.keys() if longest_trajectories else geographic_trajectories.keys()
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_title(f'Storm Trajectories (n={len(list(keys_to_plot))})', fontsize=font_size + 4)

    if wsh is not None:
        _plot_geographic_outline(wsh, ax, label="Control Area", edgecolor='red', linewidth=1.5, zorder=2)
    _plot_geographic_outline(transposition_domain, ax, label="Transposition Domain", edgecolor='green', linestyle='--', linewidth=1.5)
    ax.add_feature(cfeature.STATES, edgecolor='grey', linewidth=1)

    # Color by mean precipitation, guarding against a flat range
    precip_values = df.loc[list(keys_to_plot), 'mean_precipitation']
    vmin, vmax = precip_values.min(), precip_values.max()
    if vmin == vmax:
        vmax = vmin + 1e-6
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('Blues')

    for storm_event in keys_to_plot:
        traj = geographic_trajectories[storm_event]
        x_coords = traj['lon_coords']
        y_coords = traj['lat_coords']
        color_value = df.loc[storm_event, 'mean_precipitation']
        color = cmap(norm(color_value))
        unique_points = np.unique(np.array([x_coords, y_coords]), axis=1)
        if len(x_coords) > 3:
            tck, u = splprep(unique_points, s=smooth_factor)
            smooth_coords = splev(np.linspace(0, 1, 100), tck)
            ax.plot(smooth_coords[0], smooth_coords[1], color=color, zorder=1)
        else:
            ax.plot(x_coords, y_coords, color=color, zorder=1)

    tp_bounds = transposition_domain.total_bounds  # [minx, miny, maxx, maxy]
    lon_min, lat_min, lon_max, lat_max = tp_bounds
    ax.set_extent([lon_min - buffer, lon_max + buffer, lat_min - buffer, lat_max + buffer], crs=ccrs.PlateCarree())

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Mean Intensity [mm/h]', orientation='vertical', fraction=0.03, pad=0.02)

    legend_elements = [
        Line2D([0], [0], color='green', lw=2, linestyle='--', label="Transposition Domain"),
        Line2D([0], [0], color='lightblue', lw=2, label="Storm Trajectories"),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=font_size)

    gl = ax.gridlines(draw_labels=True, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': font_size}
    gl.ylabel_style = {'size': font_size}

    fig.tight_layout()
    if save_path:
        fname = fig_name if fig_name else "storm_trajectories"
        fig.savefig(save_path + f'/{fname}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_storm_properties_pairplot(storm_properties, save_path, domain_name, dpi=300):
    """Seaborn pairplot of numeric storm properties; saves a PNG and returns its path."""
    cols = {
        "mean_direction_deg": "Mean Direction (deg)",
        "mean_speed_ms": "Mean Speed (m/s)",
        "storm_area_km2": "Storm Area (km^2)",
        "mean_intensity_mmh": "Mean Intensity (mm/h)",
        "total_rainfall_mm": "Total Rainfall (mm)",
        "angular_variance": "Angular Variance",
    }
    available = [c for c in cols if c in storm_properties.columns]
    data = storm_properties[available].rename(columns=cols)
    g = sns.pairplot(data, diag_kind="hist", markers="o", corner=True)
    g.figure.suptitle(f"Storm Statistics Pairplot - Domain {domain_name}", y=1.02, fontsize=16)
    out = os.path.join(str(save_path), f"storm_statistics_pairplot_{domain_name}.png")
    g.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(g.figure)
    return out


def plot_parameter_sweep(sweep_df, param, save_path, domain_name, dpi=300):
    """Line plots of catalog aggregates vs a swept parameter; saves a PNG. Returns its path."""
    metrics = [
        ("n_storms", "# storms retained"),
        ("mean_direction_deg", "Mean direction (deg)"),
        ("mean_speed_ms", "Mean speed (m/s)"),
        ("mean_intensity_mmh", "Mean intensity (mm/h)"),
    ]
    metrics = [m for m in metrics if m[0] in sweep_df.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, (col, lab) in zip(axes, metrics):
        ax.plot(sweep_df[param], sweep_df[col], "o-")
        ax.set_xlabel(param)
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Parameter sensitivity: {param}  -  Domain {domain_name}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(str(save_path), f"parameter_sweep_{param}_{domain_name}.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out
