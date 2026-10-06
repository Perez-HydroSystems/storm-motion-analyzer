#import basic packages
import numpy as np
import pandas as pd
import mpu # compute storm centorid movement
from pyproj import Transformer
from geographiclib.geodesic import Geodesic

# import storm tracking and identification functions
from . import identification as storm_identification
from . import tracking as storm_tracking
from . import catalog as storm_catalog_functions

def storm_tracking_event(rain_dataset, storm_dict, morph_radius = 4, high_threshold = 0.2, var_name = 'rain',
                         ratio_threshold = 0.2, dry_spell_time = 0, area_fraction = 0.05):

    # extract basic attributes of the raw data
    # time variables dimension (350, 72)
    storm_time = rain_dataset.time 

    storm_prcp = rain_dataset.variables[var_name] # unit: mm/h, dimension (time, lat, lon)
    storm_catalog_latitude = rain_dataset.variables['latitude'] # length: 114
    storm_catalog_longitude = rain_dataset.variables['longitude'] # length: 155
    # get the array of precipitation
    rain_array = rain_dataset[var_name]
    # get the storm duration (72 hour)
    storm_duration = rain_array.shape[0]
    # Apply the storm identification and tracking code:

    # parameters: 
    # morph_radius: used to connect isolated precipitation areas if they are close to each other. 
    # high_threshold: remove precipitation grids with value below this threshold in the identification
    identification_array = storm_identification.storm_area_identification(rain_array, morph_radius, high_threshold)

    # The second step is tracking, it tracks the storm area over time
    # parameters:
    # ratio_threshold: if the storm areas in t-1 and t overlap by at least this ratio (default 0.2 = 20%), they are considered as a consecutive storm object
    # dry_spell_time: if the tracking algorithm can't find a suitable object at t, it will search at t+1 for potential suitable object
    # I suggest using dry spell time as zero. Because if the storm loses its tracking in the next time step, it is difficult to
    # calculate correct direction and moving distance
    track_array = storm_tracking.track(identification_array, ratio_threshold=ratio_threshold, dry_spell_time=dry_spell_time)
    #Based on the tracking array, we find out the longest, continous storm event during this period
    # longest_event_label: each continous storm is labeled with a unique number, we find out this number for the longest storm we want
    # area_fraction: at one step, the storm area should be greater than this fraction (default 5%) of the entire domain to be counted in its duration
    longest_event_label, longest_duration = storm_tracking.single_event_selection(track_array, area_portion_threshold=area_fraction)

    print("The longest storm has a valid duration of {0} hours".format(longest_duration))

    duration_ratio = 0.1
    if longest_duration >= storm_duration * duration_ratio:
        print("The selected duration is longer than {0} of {1} hours".format(duration_ratio, storm_duration))
    else:
        # if the selected storm after post-processing is less than 1/4 of the nominated duration (e.g., 72-hour)
        # we think it is too short and should be removed from the dataset
    
        print("The selected duration is shorted than {0} of {1} hours, skip this event".format(duration_ratio, storm_duration))
    
    # retrieve the latitude and longitude array
    lat_vector = storm_catalog_latitude[:].data
    lon_vector = storm_catalog_longitude[:].data
    # expand each coordinate to 2-dimesnion using numpy meshgrid 
    lon_array, lat_array = np.meshgrid(lon_vector, lat_vector)

    # compute the projected coordinates (unit: m)
    lon_prj_array, lat_prj_array = storm_catalog_functions.compute_projected_coordinate_system(lon_vector, lat_vector)
    storm_dict['storm_pcrp'] = storm_prcp
    storm_dict['identification_array'] = identification_array
    storm_dict['track_array'] = track_array
    storm_dict['lon_array'] = lon_array
    storm_dict['lat_array'] = lat_array
    storm_dict['storm_time'] = storm_time
    storm_dict['longest_event_label'] = longest_event_label
    storm_dict['lon_prj_array'] = lon_prj_array
    storm_dict['lat_prj_array'] = lat_prj_array
    storm_dict['storm_duration'] = storm_duration
    storm_dict['morph_radius'] = morph_radius
    storm_dict['high_threshold'] = high_threshold
    storm_dict['longest_duration'] = longest_duration

    return storm_dict

def continuos_storm(storm_dict, area_fraction = 0.05):
# compute projected area of each pixel (unit: km^2)
    
    lat_cell_degree = 0.04166412
    lon_cell_degree = 0.04166412
    pixel_area = np.cos(storm_dict['lat_array'] * np.pi / 180) * 111 * 111 * lat_cell_degree * lon_cell_degree
    # compute the total area of the study domain
    domain_total_area = np.sum(pixel_area)
    # compute the storm area at each time step
    storm_area_array = np.sum(np.where(storm_dict['track_array'] == storm_dict['longest_event_label'], pixel_area, 0), axis = (1, 2))

    # set a mask of True and False
    area_mask = storm_area_array > area_fraction * domain_total_area
    # This is to identify time steps when the storm area is greater than area_fraction (default 5%) of the total domain area

    # Find the index of the first non-zero element
    start_index = next((i for i, x in enumerate(area_mask) if x), None)
    # Find the index of the last non-zero element
    end_index = next((i for i, x in reversed(list(enumerate(area_mask))) if x), None)
    time_steps = storm_dict['storm_time']
    # get the storm event time step 
    selected_storm_time_steps = time_steps[start_index: end_index+1]
    # get the storm event precipitation series
    selected_storm_prcp_array = np.where(storm_dict['track_array'] == storm_dict['longest_event_label'], storm_dict['storm_pcrp'], 0)
    selected_storm_prcp_array = selected_storm_prcp_array[start_index: end_index+1]
    selected_storm_duration = selected_storm_time_steps.shape[0]
    selected_storm_area_list = storm_area_array[start_index: end_index+1]

    print("Selected storm starts from {0} to {1}".format(selected_storm_time_steps.values[0], selected_storm_time_steps.values[-1]))
    print("Selected storm duration {0} hours".format(selected_storm_duration))
    # Note that this duration might be different from the "longest" duration we found in the tracking step
    # This is because we may select additional time steps where the storm area is lower than the threshold, while it should
    # be kept to maintain the continous time steps
    storm_dict['selected_storm'] = selected_storm_prcp_array
    storm_dict['selected_storm_area'] = selected_storm_area_list
    storm_dict['selected_storm_duration'] = selected_storm_duration
    storm_dict['selected_storm_time_steps'] = selected_storm_time_steps  
    return storm_dict

def storm_tracking_features(storm_dict, ellipse_fit = "moments"):    
    # initialize the list to save storm bearing (degree)
    bearing_list = []

    # initialize the list to save move distance storm centroid per hour (km)
    distance_list = []

    # initialize lists to save storm centroids
    storm_lon_cent_list = []
    storm_lat_cent_list = []
            
    # initialize list to save precipitation 
    storm_avg_prcp_list = []
    storm_max_prcp_list = []

    # initialize list to save ellipse parameters
    major_axis_length_list = []
    minor_axis_length_list = []
    matplotlib_width_list = []
    matplotlib_height_list = []
    ellipse_angle_list = []
    ellipse_prj_lon_list = []
    ellipse_prj_lat_list = []
        
        
    # for each time step
    for time_index in range(storm_dict['selected_storm'].shape[0]):
        
        # get current precipitation array
        curr_prcp_array = storm_dict['selected_storm'][time_index]
        # compute the storm centroid weighted by precipitaiton (unit: degree)
        storm_lon_cent, storm_lat_cent = storm_catalog_functions.compute_weighted_centroid(curr_prcp_array, storm_dict['lon_array'], storm_dict['lat_array'])
        # compute the storm area weighted average precipitation (unit: mm/h)
        lat_cell_degree = 0.04166412
        lon_cell_degree = 0.04166412
        pixel_area = np.cos(storm_dict['lat_array'] * np.pi / 180) * 111 * 111 * lat_cell_degree * lon_cell_degree
        storm_avg_prcp = storm_catalog_functions.compute_weighted_average(curr_prcp_array, pixel_area)
        # compute the maximum precipitation (unit: mm/h)
        storm_max_prcp = np.max(curr_prcp_array)
        
        # set a threshold as 50th quantile of rainfall values at this step
        precipitation_threshold = np.quantile(curr_prcp_array[curr_prcp_array!=0].flatten(), q = 0.5)
        
        # get the ellipse parameter
        if ellipse_fit != "contour":
            ellipse_dict = storm_catalog_functions.fit_ellipse_to_rainfall(curr_prcp_array, storm_dict['lon_prj_array'], storm_dict['lat_prj_array'], precipitation_threshold)
        else:
            # fit the ellipse by contour, if it fails, reduce the threshold level by half
            try:
                ellipse_dict = storm_catalog_functions.fit_ellipse_by_contour(curr_prcp_array, storm_dict['lon_prj_array'], storm_dict['lat_prj_array'], precipitation_threshold)
            except:
                # reduce the threshold level by half
                precipitation_threshold = precipitation_threshold/2
                ellipse_dict = storm_catalog_functions.fit_ellipse_by_contour(curr_prcp_array, storm_dict['lon_prj_array'], storm_dict['lat_prj_array'], precipitation_threshold)
        
        # append the variables of storm centroid
        storm_lon_cent_list.append(storm_lon_cent)
        storm_lat_cent_list.append(storm_lat_cent)
        storm_dict['storm_lon_cent'] = storm_lon_cent_list
        storm_dict['storm_lat_cent'] = storm_lat_cent_list
        
        # append storm precipitation data
        storm_avg_prcp_list.append(storm_avg_prcp)
        storm_max_prcp_list.append(storm_max_prcp)
        
        # append the ellipse attributes
        major_axis_length_list.append(ellipse_dict['major_axis_length']) # get the major and minor axis (unit: m)
        minor_axis_length_list.append(ellipse_dict['minor_axis_length'])
        matplotlib_width_list.append(ellipse_dict['matplotlib_width']) # get the length of horizontal axis for matplotlib plot (unit: m)
        matplotlib_height_list.append(ellipse_dict['matplotlib_height']) # get the length of vertical axis for matplotlib plot (unit: m)
        # angle: rotation angle of the ellipse's major axis relative to the X-axis of the coordinate system, clockwise direction
        ellipse_angle_list.append(ellipse_dict['angle']) 
        # get the ellipse centroid
        ellipse_prj_lon_list.append(ellipse_dict['cent_prj_lon'])
        ellipse_prj_lat_list.append(ellipse_dict['cent_prj_lat'])
        

    # Compute projected coordinates of storm centroid
    storm_prj_lon_cent_list, storm_prj_lat_cent_list = storm_catalog_functions.transform_gcs_to_pcs(storm_lon_cent_list, storm_lat_cent_list)
    storm_dict['storm_prj_lon_cent_list'] = storm_prj_lon_cent_list
    storm_dict['storm_prj_lat_cent_list'] = storm_prj_lat_cent_list

    # compute the bearing and moving distance
    for i in range(len(storm_lon_cent_list)):
        if i == 0:
            # the distance and bearing are 0 for the first time step
            distance = 0
            bearing = 0
            distance_list.append(distance)
            bearing_list.append(bearing)
        else:
            # obtain the centroids for current and previous time steps
            lat1 = storm_lat_cent_list[i-1]
            lon1 = storm_lon_cent_list[i-1]
            lat2 = storm_lat_cent_list[i]
            lon2 = storm_lon_cent_list[i]
            # compute the distance between two centroids (sqkm)
            # https: // stackoverflow.com / questions / 19412462 / getting - distance - between - two - points - based - on - latitude - longitude
            distance = mpu.haversine_distance((lat1, lon1), (lat2, lon2))
            distance_list.append(distance)
            # compute the bearing between two centroids (degree)
            # https://stackoverflow.com/questions/54873868/python-calculate-bearing-between-two-lat-long
            # Move direction of the storm centroid from the previous to the current time step
            # From point at t-1 to point at t, closewise is positive, anti-closewise is negative
            bearing = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)['azi1']  # return the bearing in degrees
            bearing_list.append(bearing)

    # Define a dataframe of storm
    storm_record = pd.DataFrame()
    selected_storm_prcp_array = storm_dict['selected_storm']
    #storm_record['storm_id'] = event[-19:-12]
    storm_record['duration(hour)'] = [selected_storm_prcp_array.shape[0]] * selected_storm_prcp_array.shape[0]
    storm_time = storm_dict['selected_storm_time_steps']
    storm_record['time'] = storm_time.values

    storm_record['year'] = [pd.Timestamp(storm_time.values[0]).year] * selected_storm_prcp_array.shape[0]
    storm_record['month'] = [pd.Timestamp(storm_time.values[0]).month] * selected_storm_prcp_array.shape[0]


    storm_record['avg_prcp(mm/h)'] = storm_avg_prcp_list
    storm_record['max_prcp(mm/h)'] = storm_max_prcp_list
    #storm_record['area(km2)'] = storm_dict['selected_storm_area_list']

    storm_record['cent_lon(degree)'] = storm_lon_cent_list
    storm_record['cent_lat(degree)'] = storm_lat_cent_list

    storm_record['cen_prj_lon(m)'] = storm_prj_lon_cent_list
    storm_record['cen_prj_lat(m)'] = storm_prj_lat_cent_list
    storm_record['ellipse_lon'] = ellipse_prj_lon_list
    storm_record['ellipse_lat'] = ellipse_prj_lat_list

    storm_record['bearing(degree)'] = bearing_list
    storm_record['distance(km)'] = distance_list

    storm_record['major_axis_length(m)'] = major_axis_length_list
    storm_record['minor_axis_length(m)'] = minor_axis_length_list
    storm_record['ellipse_angle (degree)'] = ellipse_angle_list
    storm_record['ellipse_cent_prj_lon(m)'] = ellipse_prj_lon_list
    storm_record['ellipse_cent_prj_lat(m)'] = ellipse_prj_lat_list
    transformer = Transformer.from_crs("EPSG:2163", "EPSG:4326", always_xy=True)
    lon_ellipse, lat_ellipse = transformer.transform(ellipse_prj_lon_list, ellipse_prj_lat_list)
    storm_record['ellipse_cent_lon'] = lon_ellipse
    storm_record['ellipse_cent_lat'] = lat_ellipse

    storm_record['matplotlib_width(m)'] = matplotlib_width_list
    storm_record['matplotlib_height(m)'] = matplotlib_height_list

    storm_dict['storm_record'] = storm_record
    return storm_dict
