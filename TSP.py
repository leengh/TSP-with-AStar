from sys import maxsize
from itertools import permutations
import numpy as np


# implementation of traveling Salesman Problem
def TSP(graph, s):
    # store all vertex apart from source vertex
    V = graph.shape[0]
    vertex = []
    for i in range(V):
        if i != s:
            vertex.append(i)

    # store minimum weight Hamiltonian Cycle
    min_path = maxsize
    path = []
    next_permutation = permutations(vertex)
    for i in next_permutation:
        if i[-1] != V-1:
            continue

        # store current Path weight(cost)
        current_pathweight = 0

        # compute current path weight
        k = s
        i = (0,) + i
        for j in i:
            current_pathweight += graph[k][j]
            k = j
        current_pathweight += graph[k][s]
        if current_pathweight < min_path:
            min_path = current_pathweight
            path = list(i)
        # update minimum
        # min_path = min(min_path, current_pathweight)

    return path, min_path