from .data_struct import SPT


def get_all_points_neighbors(hcnng, num_vertices):
    neighborss = []
    for num in range(num_vertices):
        neighborss.append([])

    for edge in hcnng:
        neighborss[edge.start].append(edge.end)
        neighborss[edge.end].append(edge.start)

    return neighborss


def get_guided_tuple(vectors, point, neighbors, cur_dim):
    neg_neighbors = []
    pos_neighbors = []
    spt = SPT(None, None, None, False, cur_dim)

    for neighbor in neighbors:
        if vectors[neighbor][cur_dim] < vectors[point][cur_dim]:
            neg_neighbors.append(neighbor)
        else:
            pos_neighbors.append(neighbor)


    if not neg_neighbors or not pos_neighbors:
        spt.is_leaf = True
        spt.neighbors = neighbors
        return spt
    else:
        spt.neg = get_guided_tuple(vectors, point, neg_neighbors, cur_dim + 1)
        spt.pos = get_guided_tuple(vectors, point, pos_neighbors, cur_dim + 1)

    return spt


def get_all_points_guided_tuples(vectors, hcnng):
    neighborss = get_all_points_neighbors(hcnng, vectors.shape[0])
    spts = []
    for point, neighbors in enumerate(neighborss):
        spts.append(get_guided_tuple(vectors, point, neighbors, 0))
    return spts


def search_neighbors(spt, point, query, cur_dim):
    if spt.is_leaf:
        return spt.neighbors
    else:
        if query[cur_dim] < point[cur_dim]:
            return search_neighbors(spt.neg, point, query, cur_dim+1)
        else:
            return search_neighbors(spt.pos, point, query, cur_dim+1)
