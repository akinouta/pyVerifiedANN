import sys

import numpy as np
from numba import jit

from .util import euclidean_distance
from numba.typed import List


class Edge:

    MAX_DEGREE = 3

    def __init__(self, weight, start, end):
        self.weight = weight
        self.start = min(start, end)  # 无向边，所以起点和终点是可交换的
        self.end = max(start, end)  # 无向边，所以起点和终点是可交换的

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return self.weight == other.weight and self.start == other.start and self.end == other.end

    def __lt__(self, other):
        if not isinstance(other, Edge):
            return NotImplemented
        return (self.weight, self.start, self.end) < (other.weight, other.start, other.end)

    def __hash__(self):
        return hash((self.weight, self.start, self.end))


@jit(nopython=True)
def complete_graph(index, vectors):
    """
    生成完全图的邻接矩阵。

    参数:
        index: 索引集合，每个元素代表对应向量的索引编号。
        vectors: 向量集合。

    返回:
        完全图的邻接矩阵。
    """
    n = index.shape[0]

    max_float = np.finfo(np.float32).max  # 获取float64类型的最大值
    adjacency_matrix = np.full((n, n), max_float, dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            distance = euclidean_distance(vectors[index[i]], vectors[index[j]])
            adjacency_matrix[i][j] = distance
            adjacency_matrix[j][i] = distance  # 图是无向的

    return adjacency_matrix


def prim(graph, index):
    """
    使用Prim算法生成最小生成树。

    参数:
        graph (numpy.ndarray): 表示图的邻接矩阵。

    返回:
        list: 最小生成树的边列表，每条边表示为（权重，起点，终点）。
    """
    # 获取顶点数量
    num_vertices = graph.shape[0]

    # 初始化访问标记数组，所有顶点初始为未访问（False）
    visited = np.full(num_vertices, False, dtype=bool)

    # 初始化最小生成树（MST）的边列表
    mst = []

    # 从第一个顶点（索引为0）开始，标记为已访问
    visited[0] = True

    # 主循环，迭代V-1次（V为顶点数量）
    for _ in range(num_vertices - 1):
        # 初始化最小边权重为无穷大，起点和终点为None
        min_edge = (sys.maxsize, None, None)

        # 遍历所有已访问的顶点
        for frm in range(num_vertices):
            if visited[frm]:
                # 遍历与已访问顶点相邻的所有顶点
                for to in range(num_vertices):
                    # 获取边的权重
                    weight = graph[frm, to]

                    # 如果该顶点未被访问且权重小于当前最小权重
                    if not visited[to] and weight < min_edge[0]:
                        # 更新最小边
                        min_edge = (weight, frm, to)

        # 解构最小边的权重、起点和终点
        weight, frm, to = min_edge

        # 标记终点为已访问
        visited[to] = True
        mst.append((weight, index[frm], index[to]))

    return mst


def prim2(graph, index):
    """
    使用Prim算法生成最小生成树。

    参数:
        graph (numpy.ndarray): 表示图的邻接矩阵。

    返回:
        list: 最小生成树的边列表，每条边表示为（权重，起点，终点）。
    """
    # 获取顶点数量
    num_vertices = graph.shape[0]

    # 初始化访问标记数组，所有顶点初始为未访问（False）
    visited = np.full(num_vertices, False, dtype=bool)

    # 初始化最小生成树（MST）的边列表
    mst = []

    # 从第一个顶点（索引为0）开始，标记为已访问
    visited[0] = True

    # 主循环，迭代V-1次（V为顶点数量）
    for _ in range(num_vertices - 1):
        # 初始化最小边权重为无穷大，起点和终点为None
        min_edge = (sys.maxsize, None, None)

        # 遍历所有已访问的顶点
        for frm in range(num_vertices):
            if visited[frm]:
                # 遍历与已访问顶点相邻的所有顶点
                for to in range(num_vertices):
                    # 获取边的权重
                    weight = graph[frm, to]

                    # 如果该顶点未被访问且权重小于当前最小权重
                    if not visited[to] and weight < min_edge[0]:
                        # 更新最小边
                        min_edge = (weight, frm, to)

        # 解构最小边的权重、起点和终点
        weight, frm, to = min_edge

        # 标记终点为已访问
        visited[to] = True
        mst.append((weight, index[frm], index[to]))

    return mst
