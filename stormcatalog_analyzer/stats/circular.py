"""Circular (directional) statistics.

Migrated from functions/diagnostic_plots.py during the package restructure.
"""
import numpy as np
from scipy import stats

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
