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

    def __repr__(self):
        return rf"{self.start}-{self.end}:{self.weight}"


class Closedge:

    def __init__(self, end, lowcost, is_tree):
        self.end = end
        self.lowcost = lowcost
        self.is_tree = is_tree

    def __repr__(self):
        return rf"->{self.end}:[lowcost:{self.lowcost},is_tree:{self.is_tree}]"


class SPT:

    def __init__(self, neg, pos, neighbors, is_leaf, dim):
        self.neg = neg
        self.pos = pos
        self.neighbors = neighbors
        self.is_leaf = is_leaf
        self.dim = dim

    def __str__(self, s=""):

        if self.is_leaf:
            neighbors = '[' + ', '.join(str(neighbor) for neighbor in self.neighbors) + ']'
            return neighbors
        else:
            s = f"{self.dim}->neg:{str(self.neg)}\n{self.dim}->pos:{str(self.pos)}"
            return s
