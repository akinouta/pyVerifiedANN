
from modules.HCNNG.util import squared_euclidean_distance,euclidean_distance
from modules.HCNNG.load_dataset import read_fvecs
from modules.HCNNG.minimum_spanning_tree import *
import timeit
from time import time
import timeit

from numba.typed import List
import numpy as np


vectors = read_fvecs("../resource/siftsmall/siftsmall_base.fvecs")

# 示例
indexes = np.array(range(10))
result = complete_graph(indexes, vectors)


mst = prim(result, indexes)
for edge in mst:
    print(edge)




