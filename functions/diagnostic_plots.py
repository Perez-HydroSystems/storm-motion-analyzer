## import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.cm import ScalarMappable
from skimage import measure
import seaborn as sns
from windrose import WindroseAxes
from scipy.interpolate import splprep, splev
import matplotlib.gridspec as gridspec
from scipy import stats
try:
    from .storm_direction import compute_line_length
except ImportError:  # Fallback when run as a script
    from storm_direction import compute_line_length


### Plot storm time steps for a given time window and storm event
def plot_storm_time_steps(storm_tracking_results, storm_name, start_end, time_window, figsize = (10, 8), save_path=None, font_size=10):
    """
    Plots the storm precipitation, storm center, and fitted ellipse at a given time index.

    Parameters:
    - event_dict (dict): Dictionary containing storm data.
    - time_window (str): String of the selected time window for the trajectory.
    - start_end (tuple): Indexes values of the start and end time step of the most intensive time window
    """
    # Get the event dictionary for the specified storm name
    event_dict = storm_tracking_results[storm_name]
    
    # Get the shape of the figure using the time window
    time_window_options = ['6H', '8H', '12H', '16H', '24H']
    fig_shape = [(2,3), (3,3), (3,4), (4,4), (4,4)]  # shape of the figure
    dict_shape = dict(zip(time_window_options, fig_shape))  # dictionary of the shape
    
    fig, axes = plt.subplots(*dict_shape[time_window], figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})

    # Get the maximum and minimum values for the colorbar
    time_steps_window = range(start_end[0], start_end[1],1)
    t_list = [f't{i}' for i in range(1, event_dict['selected_storm'].shape[0] + 1)]  # time list for plotting
    for time_index, ax in zip(time_steps_window, axes.flatten()):
        if time_index >= event_dict['selected_storm'].shape[0]:
            break
        
        # Get the current precipitation array
        curr_prcp_array = event_dict['selected_storm'][time_index]
        # Set the threshold level
        cb_max = curr_prcp_array.flatten().max()
        cb_min = 0.2
        # Plot the projected precipitation data
        rain = ax.pcolormesh(event_dict['lon_array'], event_dict['lat_array'], 
                             np.ma.masked_where(curr_prcp_array < 0.5, curr_prcp_array),
                             cmap='Blues', vmax=cb_max, vmin=cb_min)
        
        time_str = str(event_dict['selected_storm_time_steps'].values[time_index])
        ax.set_title(f'{time_str[:16]}', fontsize = 10 + font_size)

        # Plot the storm centroid
        ax.plot(event_dict['storm_lon_cent'][time_index], event_dict['storm_lat_cent'][time_index], 'ro--', 
                label='storm center')

        # Plot the fitted ellipse
        ellipse_artist = Ellipse(
            xy=(event_dict['storm_lon_cent'][time_index], event_dict['storm_lat_cent'][time_index]),
            width=event_dict['storm_record']['matplotlib_width(m)'][time_index] / 111000,  # horizontal axis
            height=event_dict['storm_record']['matplotlib_height(m)'][time_index] / 111000,  # vertical axis
            angle=event_dict['storm_record']['ellipse_angle (degree)'][time_index],
            edgecolor='red',
            fill=False,
            linewidth=2,
        )
        ax.add_artist(ellipse_artist)
        ax.text(
            event_dict['storm_lon_cent'][time_index],
            event_dict['storm_lat_cent'][time_index],
            t_list[time_index],
            fontsize=10 + font_size,
            ha='right',
            va='bottom'
        )
        # Add U.S. state boundaries
        ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=1)

        # Add grid with labeled coordinates
        gl1 = ax.gridlines(draw_labels=False, linestyle='--')
        gl1.top_labels = False
        gl1.right_labels = False
    ## Set colorbar for all plots
    ## min and max values for the colorbar
    cb_max = event_dict['selected_storm'].flatten().max()
    cb_max = 20
    cb_min = 0.2
    n_storm = int(storm_name.split('_storm_')[1].split('_')[0])
    cbar_ax = fig.add_axes([0.18, 0.09, 0.64, 0.022])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(vmin=cb_min, vmax=cb_max), cmap='Blues'),
        cax=cbar_ax,
        orientation='horizontal'
    )
    plt.suptitle(f'Storm No. {n_storm}', fontsize=14 + font_size, y=0.98)
    cbar.set_label('Rainfall (mm/h)', fontsize=12 + font_size)
    morph_radius = event_dict['morph_radius']
    high_threshold = event_dict['high_threshold']
    # Add a text box with parameters for the subplots
    params_text = (
        "Parameters:\n"
        f"Morph radius: {morph_radius}, "
        f"High threshold: {high_threshold} mm/h, "
        f"Time window: {time_window}"
    )
    fig.text(
        0.5,
        0.91,
        params_text,
        fontsize=11 + font_size,
        color='black',
        ha='center',
        va='center',
        linespacing=1.3,
        bbox=dict(facecolor='white', edgecolor='black')
    )
    # Reserve headroom for the title/text box and bottom room for the colorbar.
    fig.subplots_adjust(top=0.80, bottom=0.16, hspace=0.16, wspace=0.10)
    # Save the figure
    if save_path is None:
        # don't save if no path is provided
        plt.show()
    else:
        plt.savefig(save_path + f'/storm_step_{n_storm}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to avoid displaying it in Jupyter Notebook


### diagnostic plot for storm trajectories

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
        wsh.plot(ax=ax1, facecolor='none', edgecolor='red', linewidth=1.5, label="Control Area", zorder=2)
    transposition_domain.plot(ax=ax1, facecolor='none', linestyle='--', edgecolor='green', linewidth=1.5, label="Transposition Domain")

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
    transposition_domain.plot(ax=ax4, facecolor='none', edgecolor='green', linewidth=2, label="Transposition Domain")
    # Add grid with labeled coordinates
    gl4 = ax4.gridlines(draw_labels=True, linestyle='--')
    gl4.top_labels = False
    gl4.right_labels = False
    gl4.xlabel_style = {'size': 10}  # Set font size for x-axis labels
    gl4.ylabel_style = {'size': 10}  # Set font size for y-axis labels
    # Add U.S. state boundaries
    ax4.add_feature(cfeature.STATES, edgecolor='lightgray', linewidth=0.5)
    #put  labels on top of the grid from 1 to 12
    for i in range(1, 13):
        cell = grid[(grid['row_index']==2.0) & (grid['col_index']==float(i-1))]
        x = (cell.right.values[0] + cell.left.values[0]) / 2  
        y = cell.top.values[0] + 0.5
        ax4.text(x, y, str(i), fontsize=10 + font_size, fontweight='bold', ha='center', va='center', transform=ccrs.PlateCarree())

    # put labels on left side from the grid from A to E
    for i in range(1, 13):
        cell = grid[(grid['row_index']==2.0) & (grid['col_index']==float(i-1))]
        x = (cell.right.values[0] + cell.left.values[0]) / 2
        y = cell.top.values[0] + 0.5
        ax4.text(x, y, str(i), fontsize=10 + font_size, fontweight='bold', ha='center', va='center', transform=ccrs.PlateCarree())
    for i, letter in enumerate(['A', 'B', 'C', 'D', 'E']):
        if i < 3:
            cell = grid[(grid['row_index']==float(i+2)) & (grid['col_index']==0.0)]
            
        elif i == 3:
            cell = grid[(grid['row_index']==float(i+2)) & (grid['col_index']==1.0)]
        else:
            cell = grid[(grid['row_index']==float(i+2)) & (grid['col_index']==5.0)]
        x = grid.loc[26].left - 0.9
        y = (cell.top.values[0] + cell.bottom.values[0]) / 2
        ax4.text(x, y, letter, fontsize = 10 + font_size, fontweight='bold',ha='center', va='center', transform=ccrs.PlateCarree())


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
        wsh.plot(ax=ax1, facecolor='none', edgecolor='red', linewidth=1.5, label="Control Area", zorder=2)
    transposition_domain.plot(ax=ax1, facecolor='none', linestyle='--', edgecolor='green', linewidth=1.5, label="Transposition Domain")

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
    transposition_domain.plot(ax=ax4, facecolor='none', edgecolor='green', linewidth=2, label="Transposition Domain")
    gl4 = ax4.gridlines(draw_labels=True, linestyle='--')
    gl4.top_labels = False
    gl4.right_labels = False
    gl4.xlabel_style = {'size': 10}
    gl4.ylabel_style = {'size': 10}
    ax4.add_feature(cfeature.STATES, edgecolor='lightgray', linewidth=0.5)
    for i in range(1, 13):
        cell = grid[(grid['row_index']==2.0) & (grid['col_index']==float(i-1))]
        x = (cell.right.values[0] + cell.left.values[0]) / 2
        y = cell.top.values[0] + 0.5
        ax4.text(x, y, str(i), fontsize=10 + font_size, fontweight='bold', ha='center', va='center', transform=ccrs.PlateCarree())
    for i, letter in enumerate(['A', 'B', 'C', 'D', 'E']):
        if i < 3:
            cell = grid[(grid['row_index']==float(i+2)) & (grid['col_index']==0.0)]
            
        elif i == 3:
            cell = grid[(grid['row_index']==float(i+2)) & (grid['col_index']==1.0)]
        else:
            cell = grid[(grid['row_index']==float(i+2)) & (grid['col_index']==5.0)]
        x = grid.loc[26].left - 0.9
        y = (cell.top.values[0] + cell.bottom.values[0]) / 2
        ax4.text(x, y, letter, fontsize = 10 + font_size, fontweight='bold',ha='center', va='center', transform=ccrs.PlateCarree())

    
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
        wsh.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1.5, label="Control Area", zorder=2)
    transposition_domain.plot(ax=ax, facecolor='none', linestyle='--', edgecolor='green', linewidth=1.5, label="Transposition Domain")
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


def plot_storm_track(storm_tracking_results, storm_name, storm_trajectories, save_path=None):
    """
    Plots an overview of a storm event, including precipitation, storm centroid trajectory, and fitted ellipses.

    Parameters:
    - storm_tracking_results (dict): Dictionary containing storm tracking results.
    - storm_name (str): Name of the storm event.
    - storm_trajectories (dict): Dictionary containing storm trajectory data.
    """
    start_time_index = storm_trajectories[storm_name]['start_time_index']  # start time index for plotting
    end_time_index = storm_trajectories[storm_name]['end_time_index']  # end time index for plotting
    event_dict = storm_tracking_results[storm_name]
    n_storm = int(storm_name.split('_storm_')[1].split('_')[0])
    # Create a list of time steps for labeling
    t_list = [f't{i}' for i in range(1, event_dict['selected_storm'].shape[0] + 1)]  # time list for plotting

    # Sum over all time steps
    acum_prcp = event_dict['selected_storm'][start_time_index: end_time_index].sum(axis=0)

    # Get the maximum and minimum values for the colorbar
    cb_max = acum_prcp.flatten().max()
    cb_min = 0.2

    # Create the figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 8), subplot_kw={'projection': ccrs.PlateCarree()}, gridspec_kw={'width_ratios': [1, 1]})
    
    # First plot: Precipitation and storm centroid
    rain = ax[0].pcolormesh(
        event_dict['lon_array'], event_dict['lat_array'], 
        np.ma.masked_where(acum_prcp < 0.2, acum_prcp),
        cmap='Blues', vmax=cb_max, vmin=cb_min
    )
    cbar = plt.colorbar(rain, ax=ax[0], label='Rainfall (mm)', orientation='vertical', fraction=0.03, pad=0.1)

    # Plot the storm centroid trajectory
    ax[0].plot(
        event_dict['storm_lon_cent'][start_time_index:end_time_index ], 
        event_dict['storm_lat_cent'][start_time_index:end_time_index ], 
        'ro--', label='Storm Center'
    )
    for lon, lat, label in zip(
        event_dict['storm_lon_cent'][start_time_index:end_time_index ], 
        event_dict['storm_lat_cent'][start_time_index:end_time_index ], 
        t_list[start_time_index:end_time_index ]
    ):
        ax[0].text(lon, lat, label, fontsize=10, ha='right', va='bottom')

    # Plot all the fitted ellipses
    for time_index in range(start_time_index, end_time_index):
        ellipse_artist = Ellipse(
            xy=(event_dict['storm_lon_cent'][time_index], event_dict['storm_lat_cent'][time_index]),
            width=event_dict['storm_record']['matplotlib_width(m)'][time_index] / 111000,  # horizontal axis
            height=event_dict['storm_record']['matplotlib_height(m)'][time_index] / 111000,  # vertical axis
            angle=event_dict['storm_record']['ellipse_angle (degree)'][time_index],
            edgecolor='lightcoral',
            fill=False,
            linewidth=1,
        )
        ax[0].add_artist(ellipse_artist)

    # Add U.S. state boundaries
    ax[0].add_feature(cfeature.STATES, edgecolor='black', linewidth=1)

    #Add bounding box for the centroids trajectory
    # Calculate the bounding box for the storm centroid trajectory
    lon_min = np.array(event_dict['storm_lon_cent'][start_time_index:end_time_index ]).min() - 0.15
    lon_max = np.array(event_dict['storm_lon_cent'][start_time_index:end_time_index ]).max() + 0.15
    lat_min = np.array(event_dict['storm_lat_cent'][start_time_index:end_time_index ]).min() - 0.15
    lat_max = np.array(event_dict['storm_lat_cent'][start_time_index:end_time_index ]).max() + 0.15
    # Create a rectangle for the bounding box
    bounding_box = Rectangle(
        (lon_min, lat_min),  # lower left corner
        lon_max - lon_min,  # width
        lat_max - lat_min,  # height
        linewidth=1.5, edgecolor='green', facecolor='none', linestyle='--'
    )
    # Add the bounding box to the plot
    ax[0].add_patch(bounding_box)
    # Add grid with labeled coordinates
    gl1 = ax[0].gridlines(draw_labels=True, linestyle='--')
    gl1.top_labels = False
    gl1.right_labels = False
    font_size = 8  # Set the desired font size
    gl1.xlabel_style = {'size': font_size}  # Set font size for x-axis labels
    gl1.ylabel_style = {'size': font_size}  # Set font size for y-axis labels

    # Set axis labels and title
    ax[0].set_xlabel("Longitude")
    ax[0].set_ylabel("Latitude")
    title = str(event_dict['selected_storm_time_steps'].values[end_time_index])
    ax[0].set_title(f'Storm Event {n_storm} at Time {title}', fontsize=14)
    ax[0].legend()

    # Second plot: Storm centroid trajectory only
    ax[1].set_extent([
        np.array(event_dict['storm_lon_cent'][start_time_index:end_time_index ]).min() - 0.2,
        np.array(event_dict['storm_lon_cent'][start_time_index:end_time_index ]).max() + 0.2,
        np.array(event_dict['storm_lat_cent'][start_time_index:end_time_index ]).min() - 0.2,
        np.array(event_dict['storm_lat_cent'][start_time_index:end_time_index ]).max() + 0.2
    ])
    ax[1].plot(
        event_dict['storm_lon_cent'][start_time_index:end_time_index], 
        event_dict['storm_lat_cent'][start_time_index:end_time_index], 
        'ro--', label='Storm Center'
    )
    # Add the bounding box to the plot
    # Create a rectangle for the bounding box
    bounding_box = Rectangle(
        (lon_min, lat_min),  # lower left corner
        lon_max - lon_min,  # width
        lat_max - lat_min,  # height
        linewidth=1.5, edgecolor='green', facecolor='none', linestyle='--', 
    )
    # Add the bounding box to the plot
    ax[1].add_patch(bounding_box)

    # Calculate the mean direction vector
    mean_track_direction = storm_tracking_results[storm_name]['mean_direction_weighted']
    # Calculate the mean speed
    mean_speed = storm_tracking_results[storm_name]['mean_velocity']
    # Create text for mean direction and speed  
    mean_direction_text = f"Mean Direction: {mean_track_direction:.2f}°"
    var_direction_text = f"Angular Variance: {storm_tracking_results[storm_name]['var_dir_weighted']:.2f}°"
    mean_speed_text = f"Mean Speed: {mean_speed:.2f} m/s"

    
    # Create text for mean precipitation within the ellipse
    mean_p = storm_trajectories[storm_name]['mean_precipitation']
    mean_precipitation_within_ellipse_text = f"Mean Intensity: {mean_p:.2f} mm/h"



    # Plot the mean direction vector
    # ─── compute centroid ───
    x = np.array(event_dict['storm_lon_cent'][start_time_index:end_time_index])
    y = np.array(event_dict['storm_lat_cent'][start_time_index:end_time_index])
    # Calculate the mean coordinates of the storm centroid trajectory
    xa, ya = x.mean(), y.mean()
     # direction (degrees -> radians), unit vector
    theta = np.radians(mean_track_direction)
    cx, sy = np.cos(theta), np.sin(theta)
    # ─── bounding‐box vector ───
    dx = x.max() - x.min()
    dy = y.max() - y.min()


    # compute max half-length that stays inside box for pivot='middle'
    def bound_along(dir_comp, low, mid, high):
        # distance to the nearer boundary in + and - direction for one axis
        if dir_comp == 0:
            return np.inf
        pos = (high - mid) / abs(dir_comp)  # forward
        neg = (mid - low)  / abs(dir_comp)  # backward
        return min(pos, neg)

    tx = bound_along(cx, x.min(), xa, x.max())
    ty = bound_along(sy, y.min(), ya, y.max())
    half_len = min(tx, ty)
        
   

    # build vector components; quiver length is full magnitude when pivot='middle'
    u = cx * (2 * half_len)
    v = sy * (2 * half_len)
    ax[1].quiver(
        xa,  ya, u, v, 
        angles='xy', scale_units='xy', pivot='middle', scale=1, color='blue', label='Mean Direction'
    )

    for lon, lat, label in zip(
        event_dict['storm_lon_cent'][start_time_index:end_time_index ], 
        event_dict['storm_lat_cent'][start_time_index:end_time_index ], 
        t_list[start_time_index:end_time_index]
    ):
        ax[1].text(lon, lat, label, fontsize=10, ha='right', va='bottom')

    # Add U.S. state boundaries
    ax[1].add_feature(cfeature.STATES, edgecolor='black', linewidth=1)

    # Add grid with labeled coordinates
    gl2 = ax[1].gridlines(draw_labels=True, linestyle='--')
    gl2.top_labels = False
    gl2.right_labels = False
    gl2.xlabel_style = {'size': font_size}
    gl2.ylabel_style = {'size': font_size}

    # Set axis labels and title
    ax[1].set_xlabel("Longitude")
    ax[1].set_ylabel("Latitude")
    ax[1].set_title("Storm Centroid Trajectory", fontsize=14)
    ax[1].legend()

    # Add a text box with parameters for the subplots
    fig.text(0.25, 0.92, f"{mean_direction_text}\n{var_direction_text}\n{mean_speed_text}\n{mean_precipitation_within_ellipse_text}", 
             fontsize=13, color='black', ha='center', va='center', 
             bbox=dict(facecolor='white', edgecolor='black'))
    
    
    # Add a text box with parameters for the subplots
    morph_radius = event_dict['morph_radius']
    high_threshold = event_dict['high_threshold']
    fig.text(0.25, 0.1, f"Parameters: -Morph radius: {morph_radius}, -High treshold:{high_threshold} mm/h", 
         fontsize=13, color='black', ha='center', va='center', 
         bbox=dict(facecolor='white', edgecolor='black'))

    # Adjust layout and add a global title
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.suptitle(f'Storm Event {n_storm} Overview', fontsize=16)

    # Save the figure
    if save_path is None:
        # don't save if no path is provided
        plt.show()
    else:
        plt.savefig(save_path + f'/storm_track_{n_storm}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to avoid displaying it in Jupyter Notebook


def circular_mean(angles, unit='degrees', rayleigh_test=False):
    """
    Calculates the mean direction for circular data like angles.

    Args:
        angles (list or np.array): An array of angles.
        unit (str, optional): The unit of the input angles. 
                                Can be 'degrees' (default) or 'radians'.

    Returns:
        float: The mean angle, in the same unit as the input.
               The output is normalized to the range [0, 360) for degrees
               or [0, 2*pi) for radians.
    """
    angles_array = np.asarray(angles)
    
    # --- 1. Convert to radians ---
    if unit == 'degrees':
        angles_rad = np.deg2rad(angles_array)
    elif unit == 'radians':
        angles_rad = angles_array
    else:
        raise ValueError("Unit must be 'degrees' or 'radians'")
        
    # --- 2. Calculate mean x and y components ---
    mean_x = np.mean(np.cos(angles_rad))
    mean_y = np.mean(np.sin(angles_rad))
    
    # --- 3. Convert mean (x, y) back to an angle in radians ---
    mean_angle_rad = np.arctan2(mean_y, mean_x)

    # ---4. Estimate p_value with Rayleigh test (optional)---
    if rayleigh_test:
        Rw = np.sqrt(mean_x**2 + mean_y**2)
        # Rayleigh test p-value
        R = len(angles) * Rw
        p_value = np.exp(-R**2 / len(angles))
    
    # --- 4. Convert back to original unit and normalize range ---
    if unit == 'degrees' and rayleigh_test:
        mean_angle = np.rad2deg(mean_angle_rad)
        return mean_angle % 360, p_value
    elif unit == 'degrees':
        mean_angle = np.rad2deg(mean_angle_rad)
        return mean_angle % 360
    elif unit == 'radians' and rayleigh_test:
        return mean_angle_rad % (2 * np.pi), p_value
    else:        return mean_angle_rad % (2 * np.pi)
    

def circular_percentile(directions_deg, q):
    """
    Compute circular percentiles for a series of directions in degrees.
    Handles wrap-around at 0/360 by rotating around the circular mean.

    Parameters
    ----------
    directions_deg : array-like
        Directions in degrees. Can be a list, numpy array, or pandas Series.
    q : array-like
        Percentiles to compute (0–100). Default: (10, 50, 90).

    Returns
    -------
    np.ndarray
        Circular percentiles in degrees in [0, 360).
    """
    # Convert to numpy array and drop NaNs
    ang_deg = np.asarray(directions_deg, dtype=float)
    ang_deg = ang_deg[~np.isnan(ang_deg)]

    if ang_deg.size == 0:
        raise ValueError("No valid (non-NaN) directions provided.")

    # 1. Normalize to [0, 360)
    ang_deg = ang_deg % 360.0

    # 2. Convert to radians
    ang = np.deg2rad(ang_deg)

    # 3. Circular mean (radians)
    mu = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())

    # 4. Rotate around mean and wrap to (-π, π]
    shift = (ang - mu + np.pi) % (2 * np.pi) - np.pi

    # 5. Percentiles in this "unwrapped" space
    q = np.asarray(q, dtype=float)
    p_shift = np.percentile(shift, q)

    # 6. Rotate back and wrap to [0, 2π)
    p = (p_shift + mu) % (2 * np.pi)

    # 7. Convert to degrees in [0, 360)
    p_deg = (np.rad2deg(p) + 360.0) % 360.0

    return p_deg

def circular_trend(time_vector, angles, unit='degrees'):
    """
    Estimates the magnitude, sign, and significance of a trend
    in circular data using the unwrap method and linear regression.

    Args:
        time_vector (list or np.array): Time or predictor variable
        angles (list or np.array): An array of angles.
        unit (str, optional): The unit of the input angles. 
                                Can be 'degrees' (default) or 'radians'.

    Returns:
        dict: A dictionary containing:
            - slope: rate of change (degrees/time_unit or radians/time_unit)
            - intercept: starting angle in unwrapped space
            - p_value: significance of the slope
            - r_squared: coefficient of determination
            - stderr: standard error of the slope
    """
    
    angles_array = np.asarray(angles, dtype=float)
    time_var = np.asarray(time_vector, dtype=float)
    
    # Validation
    if len(angles_array) != len(time_var):
        raise ValueError("time_vector and angles must have the same length")
    if len(angles_array) < 2:
        raise ValueError("Need at least 2 data points for regression")
    
    # Convert to radians if needed
    if unit == 'degrees':
        angles_rad = np.deg2rad(angles_array)
        conversion_factor = np.rad2deg(1)  # For converting back
    elif unit == 'radians':
        angles_rad = angles_array
        conversion_factor = 1
    else:
        raise ValueError("Unit must be 'degrees' or 'radians'")
    
    # Normalize to [-pi, pi] before unwrapping
    angles_rad = np.arctan2(np.sin(angles_rad), np.cos(angles_rad))
    
    # Unwrap to handle circular discontinuities
    angles_unwrapped_rad = np.unwrap(angles_rad)
    
    # Linear regression on unwrapped angles
    result = stats.linregress(time_var, angles_unwrapped_rad)

    # Convert results back to original units
    slope = result.slope * conversion_factor
    intercept = result.intercept * conversion_factor
        
    return {
        "slope": slope,
        "intercept": intercept,
        "p_value": result.pvalue,
        "r_squared": result.rvalue**2,
        "stderr": result.stderr * conversion_factor
    }

def circular_trend_permutation_test(years, directions_deg, n_permutations=9999, seed=42):
    """
    Test for significant temporal trend in circular (directional) data
    using a permutation test on the U and V components.
    
    Parameters
    ----------
    years : array-like
        The time axis (e.g., [2000, 2001, ..., 2020])
    directions_deg : array-like
        Yearly mean directions in degrees
    n_permutations : int
        Number of permutations
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    dict with observed slopes, combined statistic, and p-value
    """
    rng = np.random.default_rng(seed)
    years = np.array(years)
    dirs_rad = np.deg2rad(np.array(directions_deg))

    # --- Component decomposition ---
    U = np.sin(dirs_rad)
    V = np.cos(dirs_rad)

    # --- Mean dirrection 
    mean_dir = np.arctan2(np.mean(U), np.mean(V))
    U_mean = np.sin(mean_dir)
    V_mean = np.cos(mean_dir)

    # --- Compute observed slopes via OLS ---
    def get_slopes(t, u, v):
        slope_u = stats.linregress(t, u).slope
        slope_v = stats.linregress(t, v).slope
        return slope_u, slope_v

    def combined_statistic(slope_u, slope_v):
        """Euclidean norm of the two slopes as a single test statistic."""
        return np.sqrt(slope_u**2 + slope_v**2)

    obs_slope_u, obs_slope_v = get_slopes(years, U, V)
    obs_stat = combined_statistic(obs_slope_u, obs_slope_v)

    # angular mean direction of the slopes (for interpretability)
    rate_dir_rad = obs_slope_u * V_mean - obs_slope_v * U_mean
    rate_dir_deg = np.rad2deg(rate_dir_rad) ## degrees per year

    # --- Permutation distribution ---
    perm_stats = np.empty(n_permutations)
    for i in range(n_permutations):
        # Shuffle the directions (break the time-direction association)
        perm_idx = rng.permutation(len(dirs_rad))
        U_perm = U[perm_idx]
        V_perm = V[perm_idx]
        sl_u, sl_v = get_slopes(years, U_perm, V_perm)
        perm_stats[i] = combined_statistic(sl_u, sl_v)

    # --- p-value: proportion of permuted stats >= observed ---
    p_value = (np.sum(perm_stats >= obs_stat) + 1) / (n_permutations + 1)

    return {
        "slope_U": obs_slope_u,
        "slope_V": obs_slope_v,
        "observed_statistic": obs_stat,
        "rate_direction_deg": rate_dir_deg,
        "p_value": p_value,
        "perm_stats": perm_stats,
    }
