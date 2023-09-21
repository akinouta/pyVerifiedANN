import pickle

from modules.HCNNG.guide_search import *
from modules.outsource.VO import *
from modules.outsource.verified_search import *


def server(k, start_index, query_vector):
    with open("./communication_file/vectors.pkl", "rb") as f:
        vectors = pickle.load(f)

    with open("./communication_file/gts.pkl", "rb") as f:
        gts = pickle.load(f)

    tries = build_tries(gts)
    print("build tries")

    visited, knn = verified_search(vectors, tries, gts, k, start_index, query_vector)
    print("find knn")

    vos = vo_construction(gts, visited, vectors)
    print("vo construct")


    with open("./communication_file/knn.pkl", "wb") as f:
        pickle.dump(knn, f)

    with open("./communication_file/vos.pkl", "wb") as f:
        pickle.dump(vos, f)