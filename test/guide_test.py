import numpy as np

from modules.HCNNG.hcnng import *
from modules.HCNNG.load_dataset import read_fvecs
from modules.HCNNG.guide_search import *

vectors = read_fvecs("../resource/siftsmall/siftsmall_base.fvecs")
indexes = range(vectors.shape[0])

hcnng = createHCNNG(vectors, indexes, 5, 20)
# print(hcnng)
num_vertices=vectors.shape[0]
neighborss = get_all_points_neighbors(hcnng,num_vertices=num_vertices)
# print(neighborss)

spts = get_all_points_guided_tuples(vectors,hcnng)
test_point=5
print(neighborss[test_point])
print("spt")
print(spts[test_point])

print(search_neighbors(spts[test_point],vectors[test_point],vectors[0],0))