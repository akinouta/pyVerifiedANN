import numpy as np
from numba import jit, njit, guvectorize, float64, float32
from .data_structure import *

@njit(cache=True, fastmath=True, nogil=True)
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


# @njit(float32(float32[:], float32[:]), cache=True, fastmath=True, nogil=True)
@njit(cache=True, fastmath=True, nogil=True)
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


@guvectorize([(float32[:], float32[:], float32[:])], '(n),(n)->()', target='parallel')
def euclidean_distance_guvectorize(vector1, vector2, result):
    distance_sum = 0.0
    for i in range(vector1.shape[0]):
        distance_sum += (vector1[i] - vector2[i]) ** 2
    result[0] = np.sqrt(distance_sum)


def tree_to_dict(trie_node :Trie):
    if trie_node.is_leaf:
        return list(trie_node.children.keys())
    return {
        "k": list(trie_node.children.keys()),
        "v": [tree_to_dict(child) for child in trie_node.children.values()]
    }