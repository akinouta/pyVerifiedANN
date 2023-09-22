import numpy as np
from numba import int32

from .util import *
from .data_structure import *


# @njit(float32[:,:](int32[:], float32[:,:]), cache=True, fastmath=True, nogil=True)
@njit(cache=True, fastmath=True, nogil=True)
def complete_graph(indexes, vectors):
    """
    生成完全图的邻接矩阵。

    参数:
        index: 索引集合，每个元素代表对应向量的索引编号。
        vectors: 向量集合。

    返回:
        完全图的邻接矩阵。
    """
    n = indexes.shape[0]

    max_float = np.finfo(np.float32).max  # 获取float32类型的最大值
    adjacency_matrix = np.full((n, n), max_float, dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            distance = euclidean_distance(vectors[indexes[i]], vectors[indexes[j]])
            adjacency_matrix[i][j] = distance
            adjacency_matrix[j][i] = distance  # 图是无向的

    return adjacency_matrix


def minimum(closedges, degrees):
    size = len(closedges)
    min_mark = -1
    mincost = np.finfo(np.float32).max  # 获取float32类型的最大值

    for i in range(size):
        if closedges[i].lowcost < mincost and closedges[i].is_tree is False and degrees[i] <= Edge.MAX_DEGREE:
            min_mark = i
            mincost = closedges[i].lowcost

    return min_mark


def prim(graph, indexes):
    """
    使用Prim算法生成最小生成树。

    参数:
        graph (numpy.ndarray): 表示图的邻接矩阵。

    返回:
        list: 最小生成树的边列表，每条边表示为（权重，起点，终点）。
    """
    # 从k出发
    k = 0;
    # 获取顶点数量
    num_vertices = graph.shape[0]
    # 初始化访问标记数组，所有顶点初始为未访问（False）
    closedges = [Closedge(k, graph[k][j], False) for j in range(num_vertices)]
    degrees = np.zeros(num_vertices, dtype=np.int32)
    # 最小生成树
    mst = set()

    closedges[k].is_tree = True

    for _ in range(num_vertices - 1):
        pre = k
        k = minimum(closedges, degrees)
        mst.add(Edge(graph[pre][k], indexes[pre], indexes[k]))
        degrees[pre] += 1
        degrees[k] += 1
        closedges[k].is_tree = True
        for j in range(num_vertices):
            tmp = graph[k][j]
            if tmp < closedges[j].lowcost:
                closedges[j].end = k
                closedges[j].lowcost = tmp

    return mst


# @njit(int32(int32[:], float32[:], int32[:], int32[:]), cache=True, fastmath=True, nogil=True)
@njit(cache=True, fastmath=True, nogil=True)
def minimum_numba(end_array, lowcost_array, is_tree_array, degrees):
    size = end_array.shape[0]
    min_mark = -1
    mincost = np.finfo(np.float32).max  # 获取float32类型的最大值

    for i in np.arange(size):
        if lowcost_array[i] < mincost and is_tree_array[i] == 0 and degrees[i] <= 3:
            min_mark = i
            mincost = lowcost_array[i]

    return min_mark


@njit(cache=True, fastmath=True, nogil=True)
def prim_numba(graph, indexes):
    """
    使用Prim算法生成最小生成树。

    参数:
        graph (numpy.ndarray): 表示图的邻接矩阵。

    返回:
        list: 最小生成树的边列表，每条边表示为（权重，起点，终点）。
    """
    # 从k出发
    k = 0;
    # 获取顶点数量
    num_vertices = graph.shape[0]
    # 初始化访问标记数组，所有顶点初始为未访问（False）
    end_array = np.full(num_vertices, k, dtype=np.int32)
    lowcost_array = np.zeros(num_vertices, dtype=np.float32)
    is_tree_array = np.zeros(num_vertices, dtype=np.int32)
    degrees = np.zeros(num_vertices, dtype=np.int32)

    for j in np.arange(num_vertices):
        lowcost_array[j] = graph[k][j]
    # 最小生成树
    mst_weight = np.zeros(num_vertices - 1, dtype=np.float32)
    mst_start = np.zeros(num_vertices - 1, dtype=np.int32)
    mst_end = np.zeros(num_vertices - 1, dtype=np.int32)

    is_tree_array[k] = 1

    for i in np.arange(num_vertices - 1):
        pre = k
        k = minimum_numba(end_array, lowcost_array, is_tree_array, degrees)
        mst_weight[i] = graph[pre][k]
        mst_start[i] = indexes[pre]
        mst_end[i] = indexes[k]

        degrees[pre] += 1
        degrees[k] += 1
        is_tree_array[k] = 1
        for j in np.arange(num_vertices):
            tmp = graph[k][j]
            if tmp < lowcost_array[j]:
                end_array[j] = k
                lowcost_array[j] = tmp

    return mst_weight, mst_start, mst_end


def mst3(vectors, indexes):
    edges = set()
    mst_weight, mst_start, mst_end = prim_numba(complete_graph(indexes, vectors), indexes)
    for i in np.arange(mst_weight.shape[0]):
        edges.add(Edge(mst_weight[i], mst_start[i], mst_end[i]))

    return edges
