import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Geod

from ..stats.circular import RAYLEIGH_BASE_SEED, rayleigh_p_unweighted, weighted_rayleigh_test

# WGS84 ellipsoid used for all geodesic motion metrics (bearings, distances)
_GEOD = Geod(ellps="WGS84")

# Segments shorter than this (m) are zero-length (repeated centroid) and have no bearing.
# pyproj returns ~1e-10 m, not 0, for two identical points, so a strict ``dist > 0`` kept
# spurious segments with an arbitrary azimuth.
MIN_SEGMENT_LENGTH_M = 1.0


def segment_geodesics(lon, lat, min_length=MIN_SEGMENT_LENGTH_M):
    """
    Geodesic forward azimuth and length of each centroid-to-centroid segment on WGS84.

    Azimuths are compass bearings of the direction of motion (0 = north, 90 = east,
    clockwise), measured relative to true north at each segment's start point.
    Segments shorter than ``min_length`` (m) have no defined bearing and are dropped
    (speed and track length still use every segment). Returns (azimuth_deg, distance_m).
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if lon.size < 2:
        return np.array([]), np.array([])
    az, _, dist = _GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    az = np.atleast_1d(az) % 360.0
    dist = np.atleast_1d(dist)
    keep = np.isfinite(az) & np.isfinite(dist) & (dist >= min_length)
    return az[keep], dist[keep]


def mean_bearing(lon, lat, weighted=True):
    """
    Circular mean of the geodesic segment bearings (compass degrees, direction of motion).

    With ``weighted=True`` each segment is weighted by its geodesic length. Returns
    (mean_bearing_deg in [0, 360), angular_variance = 1 - R). Unlike the projected-plane
    ``mean_direction_weighted``, the weighted mean is not in general identical to the
    first-to-last bearing (see ``endpoint_bearing``), because each azimuth refers to
    local north at its own segment.
    """
    az, dist = segment_geodesics(lon, lat)
    if az.size == 0:
        return np.nan, np.nan
    if not weighted:
        dist = np.ones_like(dist)
    az_rad = np.radians(az)
    S = np.sum(dist * np.sin(az_rad))
    C = np.sum(dist * np.cos(az_rad))
    mean_bearing = np.degrees(np.arctan2(S, C)) % 360.0
    Rw = np.hypot(S, C) / np.sum(dist)
    return mean_bearing, 1 - Rw


def endpoint_bearing(lon, lat):
    """Geodesic forward azimuth (compass, degrees) from the first to the last centroid."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if lon.size < 2:
        return np.nan
    az, _, dist = _GEOD.inv(lon[0], lat[0], lon[-1], lat[-1])
    return float(az % 360.0) if np.isfinite(dist) and dist >= MIN_SEGMENT_LENGTH_M else np.nan


def bearing_to_math(bearing_deg):
    """Compass bearing (0 = N, clockwise) -> reported direction (0 = E, counterclockwise)."""
    return (90.0 - np.asarray(bearing_deg, dtype=float)) % 360.0


def line_length_geodesic(lon, lat):
    """Total geodesic length (m) of the centroid trajectory (all segments)."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if lon.size < 2:
        return 0.0
    _, _, dist = _GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    dist = np.atleast_1d(dist)
    return float(np.sum(dist[np.isfinite(dist)]))


def coherence_test(lon, lat, n_sim=10_000, seed=None, min_segments=3, alpha=0.05,
                   refine_n_sim=200_000, refine_se=3.0):
    """
    Directional-coherence test of one lon/lat centroid track (geodesic segments, WGS84).

    Uses the defined segments only (see ``segment_geodesics``) and runs the length-weighted
    Rayleigh test with a Monte Carlo null conditional on the segment lengths
    (``stats.circular.weighted_rayleigh_test``), plus the classical unweighted Rayleigh test
    for reference. Borderline cases, whose Monte Carlo p-value lies within ``refine_se``
    standard errors of ``alpha``, are recomputed with ``refine_n_sim`` replicates
    (``refine_n_sim=0`` disables this). Tracks with fewer than ``min_segments`` defined
    segments are not tested.

    The classification describes directional concentration of the hourly movements; it
    is not a test of trajectory straightness or of tracking accuracy.

    Returns a dict with n_segments, r_weighted, p_weighted, n_sim_used, r_unweighted,
    p_unweighted, tested, coherent.
    """
    az, dist = segment_geodesics(lon, lat)
    n = int(az.size)
    out = {"n_segments": n, "r_weighted": np.nan, "p_weighted": np.nan, "n_sim_used": 0,
           "r_unweighted": np.nan, "p_unweighted": np.nan, "tested": False, "coherent": False}
    if n == 0:
        return out
    theta = np.radians(np.asarray(bearing_to_math(az), dtype=float))
    out["r_weighted"] = float(np.hypot(np.sum(dist * np.cos(theta)), np.sum(dist * np.sin(theta))) / dist.sum())
    R_u = np.hypot(np.sum(np.cos(theta)), np.sum(np.sin(theta)))
    out["r_unweighted"] = float(R_u / n)
    if n < min_segments:
        return out
    rng = np.random.default_rng(RAYLEIGH_BASE_SEED if seed is None else seed)
    _, p_w = weighted_rayleigh_test(theta, dist, n_sim=n_sim, rng=rng)
    n_used = n_sim
    if refine_n_sim and abs(p_w - alpha) < refine_se * np.sqrt(alpha * (1 - alpha) / n_sim):
        _, p_w = weighted_rayleigh_test(theta, dist, n_sim=refine_n_sim, rng=rng)
        n_used = refine_n_sim
    out["p_weighted"] = p_w
    out["n_sim_used"] = n_used
    out["p_unweighted"] = float(rayleigh_p_unweighted(n, R_u))
    out["tested"] = True
    out["coherent"] = bool(p_w < alpha)
    return out


def select_analysis_window(values, max_positions=12):
    """
    Analysis window of a tracked storm: at most ``max_positions`` consecutive hourly fields.

    A track with ``max_positions`` or fewer positions is kept in full; a longer track is
    reduced to the ``max_positions`` consecutive fields with the largest sum of ``values``
    (the hourly ellipse-mean precipitation), the first such window if several tie.

    Returns (start, end) with ``end`` EXCLUSIVE, so ``values[start:end]`` holds the analysed
    fields and the same slice is used for the centroids, times, intensity and area.

    Replaces ``get_max_accumulated_value(series, '12H')`` in the pipeline: that window
    included both ends (13 hourly fields) while the centroids were sliced ``[start:end]``,
    so the last centroid of every window was dropped, and intensity and area used one
    field more than the trajectory.
    """
    v = np.nan_to_num(np.asarray(values, dtype=float))
    n = v.size
    if n <= max_positions:
        return 0, n
    sums = np.convolve(v, np.ones(max_positions), mode="valid")   # sums[i] = v[i:i + max_positions].sum()
    start = int(np.argmax(sums))
    return start, start + max_positions


def mean_speed_geodesic(lon, lat):
    """Mean geodesic step length per hour, converted to m/s (hourly centroids)."""
    # stationary steps count as zero speed, as in ``mean_velocity``
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if lon.size < 2:
        return 0
    _, _, dist = _GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    return float(np.mean(dist)) / 3600


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

def mean_direction_weighted(x, y):
    """
    Weighted mean direction of successive vectors defined by points (x[i], y[i]) -> (x[i+1], y[i+1]).
    Weights are the segment lengths. Returns (mean_angle_deg, angle_variance); (nan, nan) when fewer than two
    points or no segment of non-zero length (previously a 3-tuple there, which broke the callers' unpacking).

    Rw in [0,1] is the (weighted) mean resultant length; angle_variance = 1 - Rw.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return np.nan, np.nan

    dx = np.diff(x)
    dy = np.diff(y)

    # segment directions (radians) and lengths (weights)
    theta = np.arctan2(dy, dx)                  # shape (n-1,)
    w = np.hypot(dx, dy)                        # segment lengths

    # guard against zero-length segments
    mask = w > 0
    if not np.any(mask):
        return np.nan, np.nan
    theta = theta[mask]
    w = w[mask]

    # weighted circular mean
    S = np.sum(w * np.sin(theta))
    C = np.sum(w * np.cos(theta))
    mean_angle_rad = np.arctan2(S, C)
    mean_angle_deg = np.degrees(mean_angle_rad)

    # weighted mean resultant length
    r = np.hypot(C, S)
    Rw = r / np.sum(w)
    angle_variance = 1 - Rw

    return mean_angle_deg, angle_variance

def percentile_direction(angles_deg, percentiles, wrap_to_360=True):
    """
    Compute percentiles for direction values in degrees using circular wrapping.

    Parameters:
    - angles_deg (array-like): Direction values in degrees.
    - percentiles (float or array-like): Percentiles in the range [0, 100].
    - wrap_to_360 (bool): If True, return results in [0, 360). If False, return
      results in [-180, 180).

    Returns:
    - float or numpy.ndarray: Circular percentiles of the input angles.
    """

    angles = np.asarray(angles_deg, dtype=float)
    angles = angles[np.isfinite(angles)]

    if angles.size == 0:
        raise ValueError("angles_deg must contain at least one finite value")

    percentiles = np.asarray(percentiles, dtype=float)
    if np.any((percentiles < 0) | (percentiles > 100)):
        raise ValueError("percentiles must be within [0, 100]")

    # Map angles to [0, 360) and choose a circular reference close to the data.
    wrapped = np.mod(angles, 360.0)
    circular_mean = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(wrapped)))))

    # Shift the data so the percentile calculation is continuous around the mean.
    shifted = (wrapped - circular_mean + 180.0) % 360.0 - 180.0
    pct_shifted = np.percentile(shifted, percentiles)
    result = pct_shifted + circular_mean

    if wrap_to_360:
        result = np.mod(result, 360.0)
    else:
        result = (result + 180.0) % 360.0 - 180.0

    if result.ndim == 0:
        return float(result)
    return result

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
    # mask -9999 as nan
    selected_storm = np.where(selected_storm == -9999, np.nan, selected_storm)
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



