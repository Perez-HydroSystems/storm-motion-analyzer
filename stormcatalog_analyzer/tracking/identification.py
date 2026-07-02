# Functions to perform storm identification
# Yuan Liu
# 03/27/2023


import copy
import numpy as np
from scipy.ndimage import label
from scipy.ndimage import generate_binary_structure
from skimage import morphology
from skimage import draw


def perform_connected_components(to_be_connected: np.ndarray,
                                 connectivity_type: np.ndarray) -> None:
    """Higher order function used to label connected-components on all time slices of a dataset.
    :param to_be_connected: the data to perform the operation on, given as an array of dimensions Time x Rows x Cols.
    :param result: where the result of the operation will be stored, with the same dimensions as to_be_connected.
    :param lifetime: the number of time slices in the data, given as an integer.
    :param connectivity_type: an array representing the type of connectivity to be used by the labeling algorithm. See
    scipy.ndimage.measurements.label for more information.
    :return: (None - the operation is performed on the result in place.)
    """
    # for index in range(lifetime):
    cc_output, _ = label(to_be_connected, connectivity_type)  # label also returns # of labels found

    return cc_output


def perform_morph_op(morph_function: object, to_be_morphed: np.ndarray,
                     structure: np.ndarray) -> None:
    """Higher order function used to perform a morphological operation on all time slices of a dataset.
    :param morph_function: the morphological operation to perform, given as an object (function).
    :param to_be_morphed: the data to perform the operation on, given as an array of dimensions Time x Rows x Cols.
    :param result: where the result of the operation will be store, with the same dimensions as to_be_morphed.
    :param lifetime: the number of time slices in the data, given as an integer.
    :param structure: the structural set used to perform the operation, given as an array. See scipy.morphology for more
    information.
    :return: (None - the operation is performed on the result in place.)
    """
    operation = morph_function(to_be_morphed, structure)

    return operation


def build_morph_structure(radius: int):
    """
    Create an array for morphological operation
    :param radius: the radius of the circle area in the array, where in the circle the element is 1, outside the circle the element is 0
    :return: an array for morphological operation
    """
    struct = np.zeros((2 * radius, 2 * radius))
    rr, cc = draw.disk(center=(radius - 0.5, radius - 0.5), radius=radius)
    struct[rr, cc] = 1  # data in the circle equals 1
    return struct


def storm_area_identification(prcp_data: np.ndarray, morph_radius: int,
                       high_threshold: float):
    """
    Identify isolated storm areas at each time step
    :param prcp_data: raw precipitation data of the storm event dim: (duration, latitude, longitude)
    :param morph_radius: the strength of merging close storm area as a single storm. If the radius is larger, more
                         close areas will be merged as a single storm.
    :param high_threshold: set precipitation grids below this threshold as 0, which means only precipitation area above
                         this threshold is used in the identification
    :return: return an array having the same dimension as the raw precipitation array
             at each time step of the array, each isolated storm area is labeled with an unique number to show that it is
             an individual storm area
    """
    # define a morph structure
    morph_structure = build_morph_structure(radius=morph_radius)
    # filter with high threhsold
    high_threshold_data = np.where(prcp_data >= high_threshold, prcp_data, 0)

    # use 8-connectivity for determining connectedness below
    connectivity = generate_binary_structure(2, 2)

    # use identification to detect structure
    identification_array = np.zeros(high_threshold_data.shape)

    for time_index in range(prcp_data.shape[0]):

        # find connected components based on high-threshold filtered precipitation area
        label_array = perform_connected_components(high_threshold_data[time_index], connectivity)

        # perform a morph dialation
        filtered_label_array = perform_morph_op(morphology.dilation, label_array, morph_structure)

        # perform connected labeling to merge close components
        filtered_label_array = perform_connected_components(filtered_label_array, connectivity)

        # apply it to raw labelled array
        processed_label_array = np.where((label_array != 0), filtered_label_array, 0)

        # make a copy
        temp_label_array = copy.deepcopy(processed_label_array)
        eroded_label_array = perform_morph_op(morphology.erosion, temp_label_array, morph_structure)

        # keep only those storm ids in eroded_labeled_array
        unique_storm_labels = np.unique(eroded_label_array)
        unique_storm_labels = unique_storm_labels[unique_storm_labels != 0]
        processed_label_array = np.where(np.isin(processed_label_array, unique_storm_labels), processed_label_array, 0)

        identification_array[time_index] = processed_label_array

    return identification_array


