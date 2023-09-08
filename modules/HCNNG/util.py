import numpy as np
from numba import jit, njit


@jit(nopython=True)
def squared_euclidean_distance(vector1, vector2):
    """Calculate the squared Euclidean distance between two vectors.

    Parameters:
        vector1 (numpy.ndarray): The first vector.
        vector2 (numpy.ndarray): The second vector.

    Returns:
        float: The Euclidean distance between vector1 and vector2.
    """
    distance_sum = 0.0
    for index in range(vector1.shape[0]):
        distance_sum += (vector1[index] - vector2[index]) ** 2
    return distance_sum


@jit(nopython=True)
def euclidean_distance(vector1, vector2):
    """Calculate the Euclidean distance between two vectors.

    Parameters:
        vector1: The first vector.
        vector2: The second vector.

    Returns:
        The Euclidean distance between vector1 and vector2.
    """
    distance_sum = 0.0
    for index in range(vector1.shape[0]):
        distance_sum += (vector1[index] - vector2[index]) ** 2
    return np.sqrt(distance_sum)
