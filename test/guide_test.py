import numpy as np

from modules.HCNNG.hcnng import *
from modules.HCNNG.load_dataset import read_fvecs
from modules.HCNNG.guide_search import *

vectors = read_fvecs("../resource/siftsmall/siftsmall_base.fvecs")[:20]
indexes = range(vectors.shape[0])

hcnng = createHCNNG(vectors, indexes, 5, 20)
num_vertices = vectors.shape[0]

# spt test

spts = get_spts(vectors, hcnng)
test_start = 5
test_query = 0
k = 10
test_query_vector = np.array([0, 16, 35, 5, 32, 31, 14, 10, 11, 78, 55, 10, 45, 83,
                              11, 6, 14, 57, 102, 75, 20, 8, 3, 5, 67, 17, 19, 26,
                              5, 0, 1, 22, 60, 26, 7, 1, 18, 22, 84, 53, 85, 119,
                              119, 4, 24, 18, 7, 7, 1, 81, 106, 102, 72, 30, 6, 0,
                              9, 1, 9, 119, 72, 1, 4, 33, 119, 29, 6, 1, 0, 1,
                              14, 52, 119, 30, 3, 0, 0, 55, 92, 10, 2, 5, 4, 9,
                              22, 89, 96, 14, 1, 0, 1, 82, 59, 16, 25, 5, 26, 158,
                              11, 4, 0, 0, 1, 26, 47, 23, 4, 0, 0, 4, 38, 83,
                              30, 14, 9, 4, 9, 17, 23, 41, 0, 0, 2, 8, 19, 25,
                              23, 1])
# print(search_inner(vectors, spts, k, test_start, test_query))

# gt test

gts = get_gts(vectors, hcnng)
# tries = build_tries(gts)
#
# print(search_by_gts(vectors, tries, gts, k, test_start, test_query_vector))
search_neighbors_by_gt2(gts[1], vectors[1], test_query_vector)
