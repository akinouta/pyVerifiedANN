import heapq

from .util import *
from .data_structure import *


def get_all_neighbors(hcnng, num_vertices):
    neighborss = []
    for num in range(num_vertices):
        neighborss.append([])
    for edge in hcnng:
        neighborss[edge.start].append(edge.end)
        neighborss[edge.end].append(edge.start)

    return neighborss


def get_spt(vectors, index, neighbors, cur_dim):
    neg_neighbors = []
    pos_neighbors = []
    spt = SPT(None, None, None, False, cur_dim)

    for neighbor in neighbors:
        if vectors[neighbor][cur_dim] < vectors[index][cur_dim]:
            neg_neighbors.append(neighbor)
        else:
            pos_neighbors.append(neighbor)

    if not neg_neighbors or not pos_neighbors:
        spt.is_leaf = True
        spt.neighbors = neighbors
        return spt
    else:
        spt.neg = get_spt(vectors, index, neg_neighbors, cur_dim + 1)
        spt.pos = get_spt(vectors, index, pos_neighbors, cur_dim + 1)

    return spt


def get_spts(vectors, hcnng):
    neighborss = get_all_neighbors(hcnng, vectors.shape[0])
    spts = []
    for index, neighbors in enumerate(neighborss):
        spts.append(get_spt(vectors, index, neighbors, 0))
    return spts


def search_neighbors(spt, vector, query, cur_dim):
    if spt.is_leaf:
        return spt.neighbors
    else:
        if query[cur_dim] < vector[cur_dim]:
            return search_neighbors(spt.neg, vector, query, cur_dim + 1)
        else:
            return search_neighbors(spt.pos, vector, query, cur_dim + 1)


def search(vectors, spts, k, start_index, query_vector):
    visited = set()
    # 最小堆-候选集
    candidate = []
    # 最大吨-最近邻
    nearest_neighbors = []

    visited.add(start_index)
    start_dist = euclidean_distance(vectors[start_index], query_vector)
    heapq.heappush(candidate, (start_dist, start_index))

    while candidate:
        now_dist, now_index = heapq.heappop(candidate)

        if len(nearest_neighbors) == k and -nearest_neighbors[0][0] < now_dist:
            return [nearest_neighbor[1] for nearest_neighbor in sorted(nearest_neighbors, key=lambda x: -x[0])]
        else:
            heapq.heappush(nearest_neighbors, (-now_dist, now_index))

        if len(nearest_neighbors) > k:
            heapq.heappop(nearest_neighbors)

        neighbors = search_neighbors(spts[now_index], vectors[now_index], query_vector, 0)
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                dist = euclidean_distance(vectors[neighbor], query_vector)
                heapq.heappush(candidate, (dist, neighbor))


def search_inner(vectors, spts, k, start_index, query_index):
    visited = set()
    # 最小堆-候选集
    candidate = []
    # 最大吨-最近邻
    nearest_neighbors = []

    visited.add(start_index)
    start_dist = euclidean_distance(vectors[start_index], vectors[query_index])
    heapq.heappush(candidate, (start_dist, start_index))

    while candidate:
        now_dist, now_index = heapq.heappop(candidate)

        if len(nearest_neighbors) == k and -nearest_neighbors[0][0] < now_dist:
            # return sorted(nearest_neighbors, key=lambda x: -x[0])
            return [nearest_neighbor[1] for nearest_neighbor in sorted(nearest_neighbors, key=lambda x: -x[0])]
        else:
            heapq.heappush(nearest_neighbors, (-now_dist, now_index))

        if len(nearest_neighbors) > k:
            heapq.heappop(nearest_neighbors)

        neighbors = search_neighbors(spts[now_index], vectors[now_index], vectors[query_index], 0)
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                dist = euclidean_distance(vectors[neighbor], vectors[query_index])
                heapq.heappush(candidate, (dist, neighbor))


def get_guided_tuple(vectors, index, neighbors, cur_dim, judge):
    guided_tuple = dict()
    neg_neighbors = []
    pos_neighbors = []

    for neighbor in neighbors:
        if vectors[neighbor][cur_dim] < vectors[index][cur_dim]:
            neg_neighbors.append(neighbor)
        else:
            pos_neighbors.append(neighbor)

    if not neg_neighbors or not pos_neighbors:
        return {judge: neighbors}
    else:
        guided_tuple.update(get_guided_tuple(vectors, index, neg_neighbors, cur_dim + 1, judge + "0"))
        guided_tuple.update(get_guided_tuple(vectors, index, pos_neighbors, cur_dim + 1, judge + "1"))

    return guided_tuple


def get_gts(vectors, hcnng):
    neighborss = get_all_neighbors(hcnng, vectors.shape[0])
    gts = []
    for index, neighbors in enumerate(neighborss):
        gts.append(get_guided_tuple(vectors, index, neighbors, 0, ""))
    return gts


def build_tries(gts):
    tries = []
    for gt in gts:
        trie = Trie()
        for key in gt.keys():
            trie.insert(key)
        tries.append(trie)

    return tries


def search_neighbors_by_gt(tire: Trie, gt, vector, query, cur_dim, judge):
    if tire.is_leaf:
        return gt[judge]
    else:
        if query[cur_dim] < vector[cur_dim]:
            return search_neighbors_by_gt(tire.children["0"], gt, vector, query, cur_dim + 1, judge + "0")
        else:
            return search_neighbors_by_gt(tire.children["1"], gt, vector, query, cur_dim + 1, judge + "1")


def search_by_gts(vectors, tries, gts, k, start_index, query_vector):
    visited = set()
    # 最小堆-候选集
    candidate = []
    # 最大吨-最近邻
    nearest_neighbors = []

    visited.add(start_index)
    start_dist = euclidean_distance(vectors[start_index], query_vector)
    heapq.heappush(candidate, (start_dist, start_index))

    while candidate:
        now_dist, now_index = heapq.heappop(candidate)

        if len(nearest_neighbors) == k and -nearest_neighbors[0][0] < now_dist:
            return [nearest_neighbor[1] for nearest_neighbor in sorted(nearest_neighbors, key=lambda x: -x[0])]
        else:
            heapq.heappush(nearest_neighbors, (-now_dist, now_index))

        if len(nearest_neighbors) > k:
            heapq.heappop(nearest_neighbors)

        neighbors = search_neighbors_by_gt(tries[now_index], gts[now_index], vectors[now_index], query_vector, 0, "")
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                dist = euclidean_distance(vectors[neighbor], query_vector)
                heapq.heappush(candidate, (dist, neighbor))
