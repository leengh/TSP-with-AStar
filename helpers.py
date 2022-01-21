import numpy as np
from AStar import AStar
import matplotlib.pyplot as plt
from enviroment import get_enviroment


def update_obstacles_in_maze(maze, key=None):
    _, goals, category1, category2, _, start_point, _, combinations = get_enviroment()

    if key is None:
        return maze
    temp = maze.copy()

    if key == "path1":
        obstacles = np.hstack((category2, goals[:, 1].reshape(2, 1)))
    elif key == "path2":
        obstacles = np.hstack((category1, goals[:, 0].reshape(2, 1)))
    elif key == "path3":
        obstacles = start_point.reshape(2, 1)
    elif key == "path4":
        obstacles = start_point.reshape(2, 1)

    for i in range(obstacles.shape[1]):
        obstacle = obstacles[:, i]
        temp[obstacle[0], obstacle[1]] = 1

    return temp


def get_path_from_indices(path, indices):
    real_path = np.zeros((2, 1))
    for i in indices:
        real_path = np.hstack((real_path, path[:, i].reshape(2, 1)))

    return real_path[:, 1:]


def path_cost_matrix(paths, maze, cost, key=None):
    astar = AStar()

    if type(paths) is dict:
        # calculate matrices for all paths
        dist_matrices = {}
        for key, value in paths.items():
            graph = np.zeros((value.shape[1], value.shape[1]))
            temp_maze = update_obstacles_in_maze(maze, key)
            for this in range(value.shape[1]):
                for another_point in range(value.shape[1]):
                    if this != another_point:
                        _, _, astar_cost = astar.search(temp_maze, 1, tuple(value[:, this]),
                                                        tuple(value[:, another_point]))

                        graph[this][another_point] = astar_cost
                        graph[another_point][this] = astar_cost
            dist_matrices[key] = graph
        return dist_matrices
    else:
        # calculate matrix for one path
        graph = np.zeros((paths.shape[1], paths.shape[1]))
        temp_maze = update_obstacles_in_maze(maze, key)
        for this in range(paths.shape[1]):
            for another_point in range(paths.shape[1]):
                if this != another_point and graph[this][another_point] == 0:
                    _, _, astar_cost = astar.search(temp_maze, 1, tuple(paths[:, this]), tuple(paths[:, another_point]))
                    graph[this][another_point] = astar_cost
                    graph[another_point][this] = astar_cost
        return graph


def get_full_path(maze, costs, combinations, cost, plot=False):
    full_path = {}
    for i in range(combinations.shape[0]):
        cost_full = costs[combinations[i, 0]]["cost"] + costs[combinations[i, 1]]["cost"]
        full_path[i] = cost_full
    if full_path[0] < full_path[1]:
        path1_key = combinations[0, 0]
        path2_key = combinations[0, 1]
        p1 = costs[path1_key]["path"]
        p2 = costs[path2_key]["path"]
    else:
        path1_key = combinations[1, 0]
        path2_key = combinations[1, 1]
        p1 = costs[path1_key]["path"]
        p2 = costs[path2_key]["path"]

    total_cost = 0
    total_nodes_explored = 0
    if plot:
        temp = update_obstacles_in_maze(maze, path1_key)
        pat, nodes_explored = get_path_coordinates_using_astar(p1, temp, cost)
        total_nodes_explored += nodes_explored
        total_cost += pat.shape[1]
        plt.plot(pat[0, :], pat[1, :], label="y = x")
        print("Path 1 length : ", pat.shape[1])
        temp = update_obstacles_in_maze(maze, path2_key)
        pat, nodes_explored = get_path_coordinates_using_astar(p2, temp, cost)
        total_nodes_explored += nodes_explored
        total_cost += pat.shape[1]
        plt.plot(pat[0, :], pat[1, :], label="y = x")
        print("Path 2 length : ", pat.shape[1])
    print("total nodes explored:", total_nodes_explored)

    return p1, p2, total_cost


def get_path_coordinates_using_astar(path, maze, cost):
    full_path = None
    astar = AStar()
    total_nodes_explored = 0

    for i in range(0, path.shape[1] - 1):
        start = tuple(path[:, i].astype(int))
        end = tuple(path[:, i + 1].astype(int))
        nodes_explored, astar_path, _ = astar.search(maze, cost, start, end)
        new_path = np.zeros((2, len(astar_path)))
        total_nodes_explored += nodes_explored

        for j in range(len(astar_path)):
            new_path[0, j] = astar_path[j][0]
            new_path[1, j] = astar_path[j][1]

        if full_path is None:
            full_path = new_path
        else:
            full_path = np.hstack((full_path, new_path[:, 1:]))
    total_nodes_explored /= path.shape[1]
    return full_path, total_nodes_explored


def build_graph(data):
    data = data.T
    graph = np.sqrt(((data[:, None, :] - data) ** 2).sum(-1))
    return graph


def get_length(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1.0 / 2.0)


def check_collision(path_key, start, goal):
    x = start[0]
    y = start[1]
    x_goal = goal[0]
    y_goal = goal[1]
    if x_goal == x:
        slope = 1
    else:
        slope = (y_goal - y) / (x_goal - x)

    maze = update_obstacles_in_maze(path_key)
    coor = np.array(np.where(maze == 1))

    for i in range(coor.shape[1]):
        obstacle_x = coor[0, i]
        obstacle_y = coor[1, i]
        is_point_on_line = (obstacle_y - y) == slope * (obstacle_x - x)
        is_point_between = (min(x, x_goal) <= obstacle_x <= max(x, x_goal)) and (
                min(y, y_goal) <= obstacle_y <= max(y, y_goal))
        if is_point_between or is_point_on_line:
            return True

    return False
