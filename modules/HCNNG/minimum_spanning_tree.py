import numpy as np

from .util import *
from .data_structure import *


@jit(nopython=True)
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


def mst3(vectors, indexes):
    return prim(complete_graph(indexes, vectors), indexes)
