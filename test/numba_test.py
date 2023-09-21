from modules.HCNNG.util import *
from modules.HCNNG.load_dataset import read_fvecs
import timeit
import time
import numpy as np
from numba import jit, njit


def test_1(vectors):
    result = np.zeros(vectors.shape[0])
    for j in range(0, 200):
        for i in range(0, 1000):
            euclidean_distance_guvectorize(vectors[0], vectors[i], result)


def test_2(vectors):
    for j in range(0, 200):
        for i in range(0, 1000):
            euclidean_distance(vectors[0], vectors[i])


vectors = read_fvecs("../resource/siftsmall/siftsmall_base.fvecs")
time_test_numpy_start = time.time()
test_1(vectors)
time_test_numpy_end = time.time()
sq_d_time = time_test_numpy_end - time_test_numpy_start
print(rf"test1:{sq_d_time}")

time_test_numba_start = time.time()
test_2(vectors)
time_test_numba_end = time.time()
d_time = time_test_numba_end - time_test_numba_start
print(rf"test2:{d_time}")
