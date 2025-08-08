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



### Plot storm time steps for a given time window and storm event
def plot_storm_time_steps(storm_tracking_results, storm_name, start_end, time_window, save_path=None):
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
    
    fig, axes = plt.subplots(*dict_shape[time_window], figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

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
        ax.set_title(f'{time_str[:16]}', fontsize=10)

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
            fontsize=10,
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
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=cb_min, vmax=cb_max), cmap='Blues'), 
                        ax=axes, orientation='horizontal', fraction=0.02, pad=0.04)
    plt.suptitle(f'Storm No. {n_storm}', fontsize=14)
    cbar.set_label('Rainfall (mm/h)', fontsize=12)
    morph_radius = event_dict['morph_radius']
    high_threshold = event_dict['high_threshold']
    # Add a text box with parameters for the subplots
    fig.text(0.5, 0.1, f"Parameters: -Morph radius: {morph_radius}, -High treshold:{high_threshold} mm/h, -Time window:{time_window}", 
         fontsize=13, color='black', ha='center', va='center', 
         bbox=dict(facecolor='white', edgecolor='black'))
    # Adjust layout and save the figure
    plt.tight_layout()
    fig.subplots_adjust(top=0.1) # Adjust top to make room for the title
    # Save the figure
    if save_path is None:
        # don't save if no path is provided
        plt.show()
    else:
        plt.savefig(save_path + f'/storm_track_{n_storm}.png', dpi=300, bbox_inches='tight')


### diagnostic plot for storm trajectories

def diagnostic_plot_storm_trajectories(df, longest_trajectories, geographic_trajectories, wsh, transposition_domain, 
                                       storm_mean_direction_vector, storm_mean_velocity, mean_angle, mean_vel, 
                                       mean_p_ellipse, time_window,save_path):
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
    fig = plt.figure(figsize=(12, 8))

    # Create a larger subplot for ax1
    ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree())
    ax1.set_title(f'Storm Trajectories (Storm duration: {time_window})\n Storm events: 400', fontsize=16)

    # Plot basin, transposed basin, and transposition domain
    wsh.plot(ax=ax1, facecolor='none', edgecolor='red', linewidth=1.5, label="Control Area", zorder=2)
    transposition_domain.plot(ax=ax1, facecolor='none', linestyle='--', edgecolor='black', linewidth=1.5, label="Transposition Domain")

    # Add grid with labeled coordinates
    gl1 = ax1.gridlines(draw_labels=True, linestyle='--')
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xlabel_style = {'size': 10}  # Set font size for x-axis labels
    gl1.ylabel_style = {'size': 10}  # Set font size for y-axis labels

    # Add U.S. state boundaries
    ax1.add_feature(cfeature.STATES, edgecolor='black', linewidth=1)

    max_precip = df.loc[longest_trajectories.keys(), 'mean_precipitation'].max()
    min_precip = df.loc[longest_trajectories.keys(), 'mean_precipitation'].min()

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
        # Plot the trajectory with the corresponding color
        ax1.plot(x_coords, y_coords, color=color, label=storm_event[-20:-12], zorder=1) 

    # Create a ScalarMappable for the colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # Set an empty array to avoid warnings

    # Add the colorbar to the plot
    cbar = plt.colorbar(sm, ax=ax1, label='Intensity [mm/h]', orientation='vertical', fraction=0.03, pad=0.04)

    # Create a legend using custom handles
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label="Area of interest"),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label="Transposition Domain"),
        Line2D([0], [0], color='lightblue', lw=2, label="Storm Trajectories"),
    ]

    # Add the legend to ax1
    ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.5, 1.15), fontsize=10)
    new_labels = ["E", "N-E", "N", "N-W", "W", "S-W", "S", "S-E"]

    
    # Create a smaller subplot for the wind rose plot
    ax2 = WindroseAxes.from_ax(fig=fig, rect=[0.55, 0.1, 0.3, 0.25], theta_labels=new_labels)
    st_dir_vector = np.asarray(storm_mean_direction_vector) + 90
    bins = np.arange(0, np.max(storm_mean_velocity), 5)
    ax2.bar(st_dir_vector, storm_mean_velocity, normed=True, bins=bins, cmap=plt.get_cmap("cool"))
    ax2.set_legend(title='Storm Speed (m/s)', loc='lower right', bbox_to_anchor=(2.1, 0.2))
    # Hide default radial tick labels
    ax2.set_yticklabels([])

    # Define your custom radial ticks (e.g., 5%, 10%, 15%)
    r_ticks = ax2.get_yticks()
    for r in r_ticks:
        if r == 0:
            continue  # skip center label
        ax2.text(np.pi/2, r, f"{r:.0f}%", ha='left', va='bottom', fontsize=9)

    # Create a smaller subplot using the mean precipitation within the ellipse
    ax2 = WindroseAxes.from_ax(fig=fig, rect=[0.55, 0.5, 0.3, 0.25], theta_labels=new_labels)
    st_dir_vector = np.asarray(storm_mean_direction_vector) + 90
    bins = np.arange(0, np.max(df['mean_precipitation']), 5)
    ax2.bar(st_dir_vector, df['mean_precipitation'], normed=True, bins=bins, cmap=plt.get_cmap("jet"))
    ax2.set_legend(title='Intensity (mm/h)', loc='lower right', bbox_to_anchor=(2.1, 0.2))
    ax2.legend_.get_title().set_horizontalalignment('center')
    ax2.set_title('Storm Direction', fontsize=10)
    # Hide default radial tick labels
    ax2.set_yticklabels([])

    # Define your custom radial ticks (e.g., 5%, 10%, 15%)
    r_ticks = ax2.get_yticks()
    for r in r_ticks:
        if r == 0:
            continue  # skip center label
        ax2.text(np.pi/2, r, f"{r:.0f}%", ha='left', va='bottom', fontsize=9)
        
    # Add a text box above the trajectories
    mean_direction_text = f"Mean Direction: {mean_angle:.2f}°"
    mean_speed_text = f"Mean Speed: {mean_vel:.2f} m/s"
    mean_precipitation_within_ellipse_text = f"Mean Intensity: {mean_p_ellipse.mean():.2f} mm/h"

    fig.text(0.3, 0.7, f"{mean_direction_text}\n{mean_speed_text}\n{mean_precipitation_within_ellipse_text}", 
             fontsize=12, color='black', ha='center', va='center', 
             bbox=dict(facecolor='white', edgecolor='black'))

    # Show plot
    plt.show()
    plt.savefig(save_path+'/storm_trajectories_diagnostic_plot.png', dpi=300, bbox_inches='tight')



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
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # First plot: Precipitation and storm centroid
    rain = ax[0].pcolormesh(
        event_dict['lon_array'], event_dict['lat_array'], 
        np.ma.masked_where(acum_prcp < 0.2, acum_prcp),
        cmap='Blues', vmax=cb_max, vmin=cb_min
    )
    cbar = plt.colorbar(rain, ax=ax[0], label='Rainfall (mm)', orientation='vertical', fraction=0.03, pad=0.04)

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
    mean_track_direction = storm_tracking_results[storm_name]['mean_direction']
    # Calculate the mean speed
    mean_speed = storm_tracking_results[storm_name]['mean_velocity']
    # Create text for mean direction and speed  
    mean_direction_text = f"Mean Direction: {mean_track_direction:.2f}°"
    var_direction_text = f"Angular Variance: {storm_tracking_results[storm_name]['var_direction']:.2f}°"
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

    # ─── bounding‐box vector ───
    dx = x.max() - x.min()
    dy = y.max() - y.min()
    length = np.hypot(dx, dy)

    ax[1].quiver(
        xa, 
        ya, 
        np.cos(np.radians(mean_track_direction)), 
        np.sin(np.radians(mean_track_direction)), 
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
    fig.text(0.25, 0.83, f"{mean_direction_text}\n{var_direction_text}\n{mean_speed_text}\n{mean_precipitation_within_ellipse_text}", 
             fontsize=13, color='black', ha='center', va='center', 
             bbox=dict(facecolor='white', edgecolor='black'))
    
    
    # Add a text box with parameters for the subplots
    morph_radius = event_dict['morph_radius']
    high_threshold = event_dict['high_threshold']
    fig.text(0.2, 0.2, f"Parameters: -Morph radius: {morph_radius}, -High treshold:{high_threshold} mm/h", 
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






