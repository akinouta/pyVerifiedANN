from modules.HCNNG.guide_search import *


def verified_search(vectors, tries, gts, k, start_index, query_vector):
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
            break
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

    return visited, [nearest_neighbor[1] for nearest_neighbor in sorted(nearest_neighbors, key=lambda x: -x[0])]



def verified_search_without_guide(vectors, neighborss, k, start_index, query_vector):
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
            break
        else:
            heapq.heappush(nearest_neighbors, (-now_dist, now_index))

        if len(nearest_neighbors) > k:
            heapq.heappop(nearest_neighbors)

        neighbors = neighborss[now_index]
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                dist = euclidean_distance(vectors[neighbor], query_vector)
                heapq.heappush(candidate, (dist, neighbor))

    return visited, [nearest_neighbor[1] for nearest_neighbor in sorted(nearest_neighbors, key=lambda x: -x[0])]