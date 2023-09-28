import random

from joblib import Parallel, delayed

from .minimum_spanning_tree import *
from .util import *


def get_two_random_index(indexes):
    if len(indexes) < 2:
        print("聚类范围小于2")
        raise Exception
    index1 = random.choice(indexes)
    index2 = random.choice(indexes)

    while index1 == index2:
        index2 = random.choice(indexes)

    return index1, index2


def hierarchical_clustering(vectors, indexes, min_size_cluster):
    size = len(indexes)

    indexes1 = []
    indexes2 = []

    if size < min_size_cluster:
        return mst3(vectors, np.array(indexes, dtype=np.int))
    else:
        index1, index2 = get_two_random_index(indexes)
        for index in indexes:
            if euclidean_distance(vectors[index], vectors[index1]) < euclidean_distance(vectors[index],
                                                                                        vectors[index2]):
                indexes1.append(index)
            else:
                indexes2.append(index)

    edges1 = hierarchical_clustering(vectors, indexes1, min_size_cluster)
    edges2 = hierarchical_clustering(vectors, indexes2, min_size_cluster)
    return edges1.union(edges2)


def createHCNNG(vectors, indexes, min_size_cluster, num_cluster):
    hcnng = set()
    for num in range(num_cluster):
        hcnng.update(
            hierarchical_clustering(vectors, indexes, min_size_cluster)
        )
        print(num)
    return hcnng


def createHCNNG_parallel(vectors, indexes, min_size_cluster, num_cluster):
    hcnng = set()

    def inner_loop(num):
        print(f"{num}down")
        return hierarchical_clustering(vectors, indexes, min_size_cluster)

    results = Parallel(n_jobs=-1, batch_size='auto', backend='loky')(
        delayed(inner_loop)(num) for num in range(num_cluster)
    )

    for result in results:
        hcnng.update(result)

    return hcnng
