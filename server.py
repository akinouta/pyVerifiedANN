import pickle

from modules.HCNNG.guide_search import *
from modules.outsource.VO import *
from modules.outsource.verified_search import *


def server(k, start_index, query_vector):
    with open("./communication_file/vectors.pkl", "rb") as f:
        vectors = pickle.load(f)

    with open("./communication_file/hcnng.pkl", "rb") as f:
        hcnng = pickle.load(f)

    with open("./communication_file/gts.pkl", "rb") as f:
        gts = pickle.load(f)

    tries = build_tries(gts)

    visited, knn = verified_search(vectors, tries, gts, k, start_index, query_vector)

    vos = vo_construction(gts, visited, vectors)

    with open("./communication_file/knn.pkl", "wb") as f:
        pickle.dump(knn, f)

    with open("./communication_file/vos.pkl", "wb") as f:
        pickle.dump(vos, f)