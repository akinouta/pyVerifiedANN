
from modules.HCNNG.util import squared_euclidean_distance,euclidean_distance
from modules.HCNNG.load_dataset import read_fvecs
from modules.HCNNG.minimum_spanning_tree import complete_graph,prim
import timeit
from time import time
import timeit

from numba.typed import List
import numpy as np


vectors = read_fvecs("../resource/siftsmall/siftsmall_base.fvecs")

# 示例
index = np.array(range(100))
result = complete_graph(index, vectors)

time1=time();
for i in range(200):
    prim(result, index)
time2=time();
print(rf"prim1:{time2-time1}")






