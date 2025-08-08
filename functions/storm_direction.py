import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def mean_direction_ln(x, y):
     # Perform linear regression to find the slope
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    
    # Compute the angle in degrees
    angle = np.degrees(np.arctan(m))
    
    return angle, m, b

def mean_direction(x, y):
    angles = []
    for i in range(len(x) - 1):
        x1 = x[i]
        y1 = y[i]
        x2 = x[i+1]
        y2 = y[i+1]
        # Compute the angle using arctan2
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)
    # Compute circular mean of angles
    mean_angle = np.degrees(np.arctan2(np.mean(np.sin(np.radians(angles))), 
                                       np.mean(np.cos(np.radians(angles)))))
    R = np.sqrt((np.mean(np.sin(np.radians(angles)))**2 + 
                 np.mean(np.cos(np.radians(angles)))**2))
    angle_variance = 1 - R  # Variance of the circular distribution
    
    return mean_angle, angle_variance

def mean_velocity(x, y):
    velocities = [] 
    for i in range(len(x) - 1):
        x1 = x[i]
        y1 = y[i]
        x2 = x[i+1]
        y2 = y[i+1]
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        velocities.append(distance)
    
    # Compute the average velocity
    avg_velocity = np.mean(velocities) if velocities else 0
    avg_velocity = avg_velocity/3600 # convert units to m/s
    return avg_velocity

def plot_mean_direction(x,y):
    """
    Plot the list of points and overlay the mean direction as an arrow.
    """
    mean_angle = mean_direction(x, y)
    # Compute vector for the mean direction
    length = max(x) - min(x)  # Scale arrow length
    dx = length * np.cos(np.radians(mean_angle))
    dy = length * np.sin(np.radians(mean_angle))

    # Plot points
    #plt.scatter(x, y, color='blue', label="Points")
    #plt.plot(x, y, linestyle='--', alpha=0.5)  # Connect points
    
    # Draw arrow from the first point
    plt.arrow(x[0], y[0], dx, dy, 
              head_width=0.5, head_length=0.5, fc='red', ec='red', label="Mean Direction")
    
    # Labels and display
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Mean Direction of Points")
    plt.legend()
    plt.grid()
    plt.show()

def mean_precipitation_within_ellipse(event_dict):
    """
    Compute the mean precipitation within a fixed ellipse mask for each time step of a storm event.

    Parameters:
    - event_dict (dict): Must contain:
    - 'selected_storm': 3D array [time, rows, cols] of precipitation
    - 'lon_prj_array': 2D array [rows, cols] of projected longitudes
    - 'lat_prj_array': 2D array [rows, cols] of projected latitudes

    Returns:
    - list of float: mean precipitation values within the ellipse at each time step
    """

    selected_storm = event_dict['selected_storm']
    lon_prj_array = event_dict['lon_prj_array']
    lat_prj_array = event_dict['lat_prj_array']
    
    # Compute mean precipitation inside ellipse for each time step
    mean_precip = []
    df = event_dict['storm_record']
    for t in range(selected_storm.shape[0]):
        # Unpack ellipse parameters
        (cx, cy) = df.loc[t, 'ellipse_cent_prj_lon(m)'], df.loc[t, 'ellipse_cent_prj_lat(m)']
        (width, height) = df.loc[t, 'major_axis_length(m)'], df.loc[t, 'minor_axis_length(m)']
        rotation_deg = df.loc[t, 'ellipse_angle (degree)']
        a, b = width / 2, height / 2
        angle_rad = np.radians(rotation_deg)
        cos_t = np.cos(angle_rad)
        sin_t = np.sin(angle_rad)

        # Shift and rotate coordinate grids
        x_shifted = lon_prj_array - cx
        y_shifted = lat_prj_array - cy

        x_rot = x_shifted * cos_t + y_shifted * sin_t
        y_rot = -x_shifted * sin_t + y_shifted * cos_t

        # Generate static ellipse mask
        ellipse_mask = (x_rot**2 / a**2 + y_rot**2 / b**2) <= 1

        # Apply the mask to the current time step
        m = selected_storm[t][ellipse_mask].mean()
        mean_precip.append(m)
    # Convert to numpy array
    mean_precip = np.array(mean_precip) 

    return mean_precip

def get_max_accumulated_value(series, window):
    """
    Finds the time window with the highest accumulated value in a time series.

    Parameters:
    - series (pd.Series): A time series with a DatetimeIndex.
    - window (str or pd.Timedelta): A string like '3H', '1D', or a Timedelta object representing the window size.

    Returns:
    - max_sum (float): The highest accumulated sum found.
    - start_time (pd.Timestamp): Start time of the window with max sum.
    - end_time (pd.Timestamp): End time of the window with max sum.
    """

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex")

    # Ensure the time series is sorted
    series = series.sort_index()

    # Convert window string to timedelta if necessary
    window = pd.to_timedelta(window)

    max_sum = float('-inf')
    start_time = None
    end_time = None

    # Slide over time index
    for i in range(len(series)):
        current_start = series.index[i]
        current_end = current_start + window

        # Get window slice
        window_slice = series.loc[current_start:current_end]

        # Compute sum
        current_sum = window_slice.sum()

        if current_sum > max_sum:
            max_sum = current_sum
            start_time = current_start
            end_time = current_end
            index_start = i
            index_end = i + len(window_slice) - 1

    return max_sum, start_time, end_time, index_start, index_end

def compute_line_length(x_coords, y_coords):
    """
    Compute the length of the line created by the x and y points.

    Parameters:
    - x_coords (numpy.ndarray): Array of x coordinates.
    - y_coords (numpy.ndarray): Array of y coordinates.

    Returns:
    - float: Total length of the line.
    """
    distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
    return np.sum(distances)



