import numpy as np


class AStar:
    def __init__(self):
        pass

    def get_vertex_neighbours(self, pos):
        n = []
        # Moves allow link a chess king
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            x2 = pos[0] + dx
            y2 = pos[1] + dy
            if x2 < 0 or x2 >= 100 or y2 < 0 or y2 >= 100:
                continue
            n.append((x2, y2))
        return n

    def move_cost(self, graph,  b):
        if graph[int(b[0]), int(b[1])] == 1:
            return 100  # Extremely high cost to enter barrier squares
        return 1

    def heuristic(self, start, goal):
        # Use octile distance heuristic if we can move one square either
        # adjacent or diagonal
        # D = 1
        # D2 = np.sqrt(2) # 1
        dx = start[0] - goal[0]
        dy = start[1] - goal[1]
        # return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
        return np.sqrt((dx*dx) + (dy*dy))

    def search(self, graph, cost, start, end):
        G = {}  # Actual movement cost to each position from the start position
        F = {}  # Estimated movement cost of start to end going via this position

        # Initialize starting values
        G[start] = 0
        F[start] = self.heuristic(start, end)

        closedVertices = set()
        openVertices = set([start])
        cameFrom = {}

        while len(openVertices) > 0:
            # Get the vertex in the open list with the lowest F score
            current = None
            currentFscore = None
            for pos in openVertices:
                if current is None or F[pos] < currentFscore:
                    currentFscore = F[pos]
                    current = pos

            # Check if we have reached the goal
            if current == end:
                # Retrace our route backward
                path = [current]
                while current in cameFrom:
                    current = cameFrom[current]
                    path.append(current)
                path.reverse()
                # print("Astar cost: ", F[end], "- path length : ", len(path))
                return len(openVertices), path, F[end]  # Done!

            # Mark the current vertex as closed
            openVertices.remove(current)
            closedVertices.add(current)

            # Update scores for vertices near the current position
            for neighbour in self.get_vertex_neighbours(current):
                if neighbour in closedVertices:
                    continue  # We have already processed this node exhaustively
                candidateG = G[current] + self.move_cost(graph, neighbour)

                if neighbour not in openVertices:
                    openVertices.add(neighbour)  # Discovered a new vertex
                elif candidateG >= G[neighbour]:
                    continue  # This G score is worse than previously found

                # Adopt this G score
                cameFrom[neighbour] = current
                G[neighbour] = candidateG
                H = self.heuristic(neighbour, end)
                F[neighbour] = G[neighbour] + H

        raise RuntimeError("A* failed to find a solution")

