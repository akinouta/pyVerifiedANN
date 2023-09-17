import pickle

import numpy as np

from modules.HCNNG.hcnng import *
from modules.HCNNG.load_dataset import read_fvecs
from modules.HCNNG.guide_search import *
from modules.outsource.MHT import *
from modules.outsource.verified_search import *
from modules.outsource.VO import *

vectors = read_fvecs("../resource/siftsmall/siftsmall_base.fvecs")[:20]
indexes = range(vectors.shape[0])

# hcnng = createHCNNG(vectors, indexes, 5, 3)
# with open("hcnng.pkl", "wb") as f:
#     pickle.dump(hcnng, f)

with open("hcnng.pkl", "rb") as f:
    hcnng = pickle.load(f)
print(hcnng)
gts = get_gts(vectors, hcnng)

print(gts)

test_start = 5
test_query = 0
k = 5
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

tries = build_tries(gts)



visited, knn = verified_search(vectors, tries, gts, k, test_start, test_query_vector)


print(visited)
print(knn)

hash_list = gts_to_hash(gts)
root_hash_DO = get_merkle_root(hash_list)
vos = vo_construction(gts, visited)
root_hash_Client = vo_compute(vos).data

print(root_hash_Client == root_hash_DO)



