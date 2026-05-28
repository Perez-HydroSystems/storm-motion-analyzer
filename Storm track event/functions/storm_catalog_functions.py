# Functions to compute the storm catalog
# Yuan Liu
# 03/27/2023


import numpy as np
from pyproj import Transformer
import cv2
from skimage import measure


def compute_projected_coordinate_system(lon_data, lat_data):
    """
    Convert the lat/lon coordinates into projected coordinates
    :param lon_data: an 1-d array of longitude data
    :param lat_data: an 1-d array of latitude data
    :return:
            lon_prj_array: an 2-d array of projected coordinates in x direction
            lat_prj_array: an 2-d array of projected coordinates in y direction
    """
    # get projected coordinates
    lon_array, lat_array = np.meshgrid(lon_data, lat_data)
    lon_array_flat = lon_array.flatten()
    lat_array_flat = lat_array.flatten()
    points = []
    for i in range(lon_array_flat.shape[0]):
        points.append((lat_array_flat[i], lon_array_flat[i]))
    # transform lat lon to projected coordinate system
    transformer = Transformer.from_crs(4326, 2163)
    # The Transformer.from_crs() function in the PyProj library is used to create a coordinate transformation
    # between two coordinate reference systems (CRS).
    # Here, you want to create a transformer to convert coordinates
    # from EPSG:4326 (WGS 84) to EPSG:2163 (US National Atlas Equal Area).

    lat_prj_array = []
    lon_prj_array = []
    for pt in transformer.itransform(points):
        lat_prj_array.append(pt[1])
        lon_prj_array.append(pt[0])
    lat_prj_array = np.reshape(np.array(lat_prj_array), lat_array.shape)
    lon_prj_array = np.reshape(np.array(lon_prj_array), lat_array.shape)

    return lon_prj_array, lat_prj_array


def transform_gcs_to_pcs(lon_data, lat_data):
    """
    Transform any lon/lat coordinate data into projected coordinates
    :param lon_data: an 1-d array of longitude data
    :param lat_data: an 1-d array of latitude data
    :return:
            lon_prj_data: an 1-d array of projected coordinates along x axis (unit: m)
            lat_prj_data: an 1-d array of projected coordinates along y axis (unit: m)
    """
    # convert to array
    lon_data = np.array(lon_data)
    lat_data = np.array(lat_data)

    points = []
    for i in range(lon_data.shape[0]):
        points.append((lat_data[i], lon_data[i]))
    # transform lat lon to projected coordinate system
    transformer = Transformer.from_crs(4326, 2163)
    # The Transformer.from_crs() function in the PyProj library is used to create a coordinate transformation
    # between two coordinate reference systems (CRS).
    # Here, you want to create a transformer to convert coordinates
    # from EPSG:4326 (WGS 84) to EPSG:2163 (US National Atlas Equal Area).

    lat_prj_data = []
    lon_prj_data = []
    for pt in transformer.itransform(points):
        lat_prj_data.append(pt[1])
        lon_prj_data.append(pt[0])

    lon_prj_data = np.array(lon_prj_data)
    lat_prj_data = np.array(lat_prj_data)
    return lon_prj_data, lat_prj_data


# compute the precipitation weighted centroid
def compute_weighted_centroid(curr_prcp_array, lon_array, lat_array):
    """
    Compute the storm centroid weighted by precipitation intensity
    :param curr_prcp_array: precipitation array at the current time step with dim (latitude, longitude)
    :param lon_array: 2-dimension array of longitude coordinates
    :param lat_array: 2-dimension array of latitude coordinates
    :return:
    """
    # find the sum of the precipitation values belonging to the storm
    sum_ivt = np.sum(curr_prcp_array)
    # and its intensity weighted centroid
    x_avg = np.sum((lon_array * curr_prcp_array) / sum_ivt)
    y_avg = np.sum((lat_array * curr_prcp_array) / sum_ivt)
    return x_avg, y_avg


# compute the area weighted precipitation
def compute_weighted_average(var_array, pixel_area):
    """
    Compute average precipitation weighted by the area (in km2) of each pixel
    :param var_array: precipitation array at the current time step with dim (latitude, longitude)
    :param pixel_area: an array of projected area at each pixel with dim (latitude, longitude)
    :return:
    """
    curr_area = np.sum(np.where(var_array > 0, pixel_area, 0))
    if curr_area == 0:
        weighted_avg = 0
    else:
        weighted_avg = np.sum(var_array * pixel_area) / curr_area
    return weighted_avg# , curr_area


def fit_ellipse_by_contour(curr_prcp_array, lon_prj_array, lat_prj_array, threshold_level):
    """
    Fit an ellipse based on the longest contour of the current precipitation field
    :param curr_prcp_array: precipitation array at the current time step with dim (latitude, longitude)
    :param lon_prj_array: 2-dimension array of projected longitude coordinates
    :param lat_prj_array: 2-dimension array of projected latitude coordinates
    :param threshold_level: function will find precipitation contours above this threshold
    :return: a dictionary containing key information of the fitted ellipse
    """

    # Add a border of 0 around the input array:
    border_value = 0
    data_with_border = np.pad(curr_prcp_array, pad_width=1, mode='constant', constant_values=border_value)

    # find the contours of precipitation
    contours = measure.find_contours(data_with_border,
                                     level=threshold_level)
    # for each contour, count the number of points in this contour
    point_list = []
    for contour in contours:
        # Offset the contours by -1 in both x and y directions to account for the added border:
        contour -= 1
        # compute the number of points
        point_num = contour.shape[0]
        point_list.append(point_num)

    # find the contour with the largest number of points
    point_list = np.array(point_list)
    max_contours = contours[np.argwhere(point_list == point_list.max())[0][0]].astype('int')

    # convert the contour indices to projected coordinates
    ellipse_contour_indices = (max_contours[:, 0], max_contours[:, 1])

    # stack geographical projected coordinates of the contour
    projected_coords = np.column_stack(
        (lon_prj_array[ellipse_contour_indices], lat_prj_array[ellipse_contour_indices])).astype(np.float32)

    # fit ellipse based on projected precipitation coordinates
    ellipse = cv2.fitEllipse(projected_coords)

    # get the major and minor axis (unit: m)
    semi_major_axis_length = max(ellipse[1])
    semi_minor_axis_length = min(ellipse[1])
    # get the length of horizontal axis (unit: m)
    matplotlib_ellipse_width = ellipse[1][0]
    # get the length of vertical axis (unit: m)
    matplotlib_ellipse_height = ellipse[1][1]

    # angle: rotation angle of the ellipse's major axis relative to the X-axis of the coordinate system, clockwise direction
    ellipse_angle = ellipse[2]
    # get the center location
    ellipse_cent_prj_lon = ellipse[0][0]  # (unit: m)
    ellipse_cent_prj_lat = ellipse[0][1]  # (unit: m)

    # create a dictionary to save ellipse data
    ellipse_dict = {}
    ellipse_dict['major_axis_length'] = semi_major_axis_length
    ellipse_dict['minor_axis_length'] = semi_minor_axis_length
    ellipse_dict['matplotlib_width'] = matplotlib_ellipse_width
    ellipse_dict['matplotlib_height'] = matplotlib_ellipse_height
    ellipse_dict['angle'] = ellipse_angle
    ellipse_dict['cent_prj_lon'] = ellipse_cent_prj_lon
    ellipse_dict['cent_prj_lat'] = ellipse_cent_prj_lat

    return ellipse_dict

def fit_ellipse_to_rainfall(rainfall_field, x_coords, y_coords, threshold=1.0):
    """
    Fits an ellipse to a 2D rainfall field using the method of moments.

    Args:
        rainfall_field (np.ndarray): A 2D NumPy array where each value is the
                                    rainfall intensity.
        threshold (float): The minimum rainfall intensity to consider.
                           Values below this will be set to zero.

    Returns:
        dict: A dictionary containing the ellipse parameters:
              'center' (tuple), 'major_axis' (float), 'minor_axis' (float),
              'angle_deg' (float). Returns None if no data is above the threshold.
    """
    # Ensure input is a numpy array
    field = np.array(rainfall_field)
    
    # Apply threshold
    field_thresholded = np.where(field >= threshold, field, 0)

    

    # Calculate zeroth moment (total intensity)
    m00 = np.sum(field_thresholded)

    # Check if there is any rainfall above the threshold
    if m00 == 0:
        return None

    # First moments (to find the centroid)
    m10 = np.sum(x_coords * field_thresholded)
    m01 = np.sum(y_coords * field_thresholded)

    # --- Step 2: Calculate Ellipse Center (Centroid) ---
    x_bar = m10 / m00
    y_bar = m01 / m00

    # Second central moments (to find size and orientation)
    mu20 = np.sum((x_coords - x_bar)**2 * field) / m00
    mu02 = np.sum((y_coords - y_bar)**2 * field) / m00
    mu11 = np.sum((x_coords - x_bar) * (y_coords - y_bar) * field) / m00

    # --- Step 3: Calculate Ellipse Axes and Angle ---
    # Common term
    common_term = np.sqrt((mu20 - mu02)**2 + 4 * mu11**2)

    # Major and minor axes (proportional to standard deviations)
    # The factor 2*sqrt(2) scales the ellipse to contain ~86.5% of the total intensity
    # under a Gaussian assumption, a common convention.
    major_axis = 2 * np.sqrt(2) * np.sqrt(0.5 * ((mu20 + mu02) + common_term))
    minor_axis = 2 * np.sqrt(2) * np.sqrt(0.5 * ((mu20 + mu02) - common_term))

    # Orientation angle
    angle_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    angle_deg = np.degrees(angle_rad)

     # create a dictionary to save ellipse data
    ellipse_dict = {}
    ellipse_dict['major_axis_length'] = major_axis
    ellipse_dict['minor_axis_length'] = minor_axis
    ellipse_dict['matplotlib_width'] = major_axis
    ellipse_dict['matplotlib_height'] = minor_axis
    ellipse_dict['angle'] = angle_deg
    ellipse_dict['cent_prj_lon'] = x_bar
    ellipse_dict['cent_prj_lat'] = y_bar
    return ellipse_dict
        

    




