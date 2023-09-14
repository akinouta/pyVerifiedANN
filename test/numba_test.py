from modules.HCNNG.util import squared_euclidean_distance,euclidean_distance
from modules.HCNNG.load_dataset import read_fvecs
import timeit
import time
import numpy as np
from numba import jit, njit


def test_sq_d(vectors):
    for j in range(0, 200):
        for i in range(0, 10000):
            np.linalg.norm(vectors[0] - vectors[i])


def test_d(vectors):
    for j in range(0, 200):
        for i in range(0, 10000):
            euclidean_distance(vectors[0], vectors[i])


vectors = read_fvecs("../resource/siftsmall/siftsmall_base.fvecs")
time_test_numpy_start = time.time()
test_sq_d(vectors)
time_test_numpy_end = time.time()
sq_d_time = time_test_numpy_end - time_test_numpy_start
print(rf"numpy:{sq_d_time}")

time_test_numba_start = time.time()
test_d(vectors)
time_test_numba_end = time.time()
d_time = time_test_numba_end - time_test_numba_start
print(rf"numba:{d_time}")
