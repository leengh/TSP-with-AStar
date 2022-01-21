import matplotlib.pyplot as plt
import numpy as np
from AStar import AStar
from TSP import TSP
from helpers import build_graph, get_path_from_indices, path_cost_matrix, get_full_path, get_length
from enviroment import print_enviroment, setup_enviroment, get_enviroment
from timeit import default_timer as timer
import networkx as nx
from networkx.algorithms import approximation as approx

def method1():
    start = timer()
    print("--------Method 1---------")
    # Get Path using TSP only and then plot the path using A*
    print_enviroment(obstacles, start_point, goals, category1, category2, "Method 1")
    costs = {}

    for key, path in paths.items():
        distance_matrix = build_graph(path)
        path, path_cost = TSP(distance_matrix, 0)
        actual_path = get_path_from_indices(paths[key], path)
        costs[key] = {"path": actual_path, "cost": path_cost}

    _, _, final_cost = get_full_path(maze, costs, combinations, cost, plot=True)

    plt.show()
    print("Final Cost", final_cost)
    end = timer()
    print("Time (seconds) ", end - start)


def method2():
    # get heuristics paths costs matrices from A*
    # use them in TSP
    start = timer()
    print("--------Method 2---------")
    dist_matrices = path_cost_matrix(paths, maze, cost)
    print_enviroment(obstacles, start_point, goals, category1, category2, "Method 2")

    costs = {}

    for key, value in dist_matrices.items():
        start_city = 0
        distance_matrix = dist_matrices[key]
        shortest_path, path_cost = TSP(distance_matrix, start_city)
        actual_path = get_path_from_indices(paths[key], shortest_path)
        costs[key] = {"path": actual_path, "cost": path_cost}

    _, _, final_cost = get_full_path(maze, costs, combinations, cost, plot=True)

    plt.show()
    print("Final Cost", final_cost)
    end = timer()
    print("Time (seconds) ", end - start)


def method3():
    # Get TSP path
    # plot path : if cost of A* is less than path of TSP => recalculate
    print_enviroment(obstacles, start_point, goals, category1, category2, "Method 3")

    start_time = timer()
    print("--------Method 3---------")
    # Prepare environment to display
    # print_enviroment(obstacles, start_point, goals, category1, category2, "Method 3")
    # Initialize A* distance matrices for each path
    distance_matrices = {}  # path_cost_matrix(paths, maze, cost)
    # Save the cost for each path
    costs = {}

    for key, path in paths.items():
        path_cost = 0
        # euclidean distance matrix
        graph = build_graph(path)
        # Calculate TSP route for every possible path using euclidean distance matrix
        actual_tsp_path, _ = TSP(graph, 0)
        # Convert indices to actual nodes' coordinates
        initial_path = get_path_from_indices(paths[key], actual_tsp_path)
        distance_matrices[key] = path_cost_matrix(initial_path.astype(int), maze, 1, key)

        # Keep track of the updated TSP route
        final_path = initial_path

        for index in range(initial_path.shape[1] - 1):
            start = list(final_path[:, index].astype(int))
            goal = list(final_path[:, index + 1].astype(int))
            # get euclidean distance between node and the one after it the in TSP route
            euclidean_distance = get_length(start[0], start[1], goal[0], goal[1])
            # get A* cost between node and the one after it the in TSP route
            astar_cost = distance_matrices[key][index, index + 1]

            # if A* cost is larger than the euclidean_distance
            # then we replace the euclidean heuristics with A* heuristics for the current start point only
            # so it can select the actual closest neighbor if it results in a shorter path
            if astar_cost > euclidean_distance:
                # Path from the current start point to the goal node
                distance_matrix = graph
                distance_matrix[index, :] = distance_matrices[key][index, :]
                distance_matrix[:, index] = distance_matrices[key][:, index]

                updated_tsp_path, _ = TSP(distance_matrix[index:, index:], 0)

                path_from_this_point = final_path[:, index:]
                updated_tsp_path_coordinates = np.zeros((2, 1))
                for i in updated_tsp_path:
                    updated_tsp_path_coordinates = np.hstack(
                        (updated_tsp_path_coordinates, path_from_this_point[:, i].reshape(2, 1)))
                updated_tsp_path_coordinates = updated_tsp_path_coordinates[:, 1:]
                # If TSP doesn't change the next point
                # ignore it and move to the next point
                if np.array_equal(path_from_this_point[0, :2], updated_tsp_path_coordinates[0, :2]) and np.array_equal(
                        path_from_this_point[1, :2], updated_tsp_path_coordinates[1, :2]):
                    path_cost += astar_cost
                    continue

                astar_cost = distance_matrices[key][index, index + updated_tsp_path[1]]
                path_cost += astar_cost
                final_path = np.hstack((final_path[:, :index], updated_tsp_path_coordinates))
                distance_matrices[key] = path_cost_matrix(final_path.astype(int), maze, 1, key)
            else:
                path_cost += astar_cost

        costs[key] = {"path": final_path, "cost": path_cost}
    _, _, final_cost = get_full_path(maze, costs, combinations, cost, plot=True)
    plt.show()

    print("Final Cost", final_cost)
    end = timer()
    print("Time (seconds) ", end - start_time)


def method4():
    print_enviroment(obstacles, start_point, goals, category1, category2, "Method 4")
    start_time = timer()
    print("--------Method 4---------")
    distance_matrices = {}  # path_cost_matrix(paths, maze, cost)
    # Save the cost for each path
    costs = {}

    for key, path in paths.items():
        path_cost = 0
        # euclidean distance matrix
        graph = build_graph(path)
        # Calculate TSP route for every possible path using euclidean distance matrix
        actual_tsp_path, _ = TSP(graph, 0)
        # Convert indices to actual nodes' coordinates
        initial_path = get_path_from_indices(paths[key], actual_tsp_path)
        distance_matrices[key] = path_cost_matrix(initial_path.astype(int), maze, 1, key)

        # Keep track of the updated TSP route
        final_path = initial_path

        for index in range(initial_path.shape[1] - 1):
            start = list(final_path[:, index].astype(int))
            goal = list(final_path[:, index + 1].astype(int))

            # get euclidean distance between node and the one after it the in TSP route
            euclidean_distance = get_length(start[0], start[1], goal[0], goal[1])
            # get A* cost between node and the one after it the in TSP route
            astar_cost = distance_matrices[key][index, index + 1]

            # if A* cost is larger than the euclidean_distance
            # then we replace the euclidean heuristics with A* heuristics for the current start point only
            # so it can select the actual closest neighbor if it results in a shorter path
            if astar_cost > euclidean_distance:
                # Path from the current start point to the goal node
                distance_matrix = distance_matrices[key]
                neighbors = distance_matrix[index, index + 1:]
                nearest_distance = min(neighbors)

                path_cost += nearest_distance

                nearest_neighbor_index = np.where(neighbors == nearest_distance)[0]
                nearest_neighbor_index = (index + 1) + nearest_neighbor_index

                if nearest_neighbor_index == index + 1:
                    continue

                temp_neighbor = final_path[:, nearest_neighbor_index]
                temp_next_neighbor = final_path[:, index + 1]

                final_path[:, nearest_neighbor_index] = temp_next_neighbor.reshape(2, 1)

                final_path[:, index + 1] = temp_neighbor.reshape(2, )

                # paths[key] = final_path
                distance_matrices[key] = path_cost_matrix(final_path.astype(int), maze, 1, key)

                updated_tsp_path, _ = TSP(distance_matrix[index + 1:, index + 1:], 0)
                path_from_this_point = final_path[:, index + 1:]
                updated_tsp_path_coordinates = np.zeros((2, 1))
                for i in updated_tsp_path:
                    updated_tsp_path_coordinates = np.hstack(
                        (updated_tsp_path_coordinates, path_from_this_point[:, i].reshape(2, 1)))
                updated_tsp_path_coordinates = updated_tsp_path_coordinates[:, 1:]
                final_path = np.hstack((final_path[:, :index + 1], updated_tsp_path_coordinates))
                # paths[key] = final_path
                distance_matrices[key] = path_cost_matrix(final_path.astype(int), maze, 1, key)
            else:
                path_cost += astar_cost

        costs[key] = {"path": final_path, "cost": path_cost}
    _, _, final_cost = get_full_path(maze, costs, combinations, cost, plot=True)
    plt.show()

    print("Final Cost", final_cost)
    end = timer()
    print("Time (seconds) ", end - start_time)


def method5():
    start = timer()
    print("--------Method 5---------")
    # Get Path using TSP only and then plot the path using A*
    print_enviroment(obstacles, start_point, goals, category1, category2, "Method 5")
    costs = {}

    for key, path in paths.items():
        distance_matrix = build_graph(path)
        distance_matrix = np.hstack((distance_matrix, np.full((distance_matrix.shape[0], 1), -1)))
        distance_matrix = np.vstack((distance_matrix, np.full((1, distance_matrix.shape[1]), -1)))
        distance_matrix[1:-2, -1] = float('inf')
        distance_matrix[-1, 1:-2] = float('inf')
        distance_matrix[-1, -1] = 0

        ## GREEEDY
        G = nx.cycle_graph(distance_matrix.shape[1])
        add_edges_from = []

        for i in range(distance_matrix.shape[1]):
            for j in range(distance_matrix.shape[1]):
                add_edges_from.append((i, j, distance_matrix[i, j]))
        G.add_weighted_edges_from(add_edges_from)
        cycle = approx.greedy_tsp(G, source=0)
        path = [ele for ele in reversed(cycle)]
        path = path[:-2]
        actual_path = get_path_from_indices(paths[key], path)
        path_cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))
        costs[key] = {"path": actual_path, "cost": path_cost}
        ## END GREEEDY

    _, _, final_cost = get_full_path(maze, costs, combinations, cost, plot=True)

    plt.show()
    print("Final Cost", final_cost)
    end = timer()
    print("Time (seconds) ", end - start)


def TSP_only():
    costs = {}

    for key, path in paths.items():
        distance_matrix = build_graph(path)
        path, path_cost = TSP(distance_matrix, 0)
        actual_path = get_path_from_indices(paths[key], path)
        costs[key] = {"path": actual_path, "cost": path_cost}

    p1, p2, _ = get_full_path(maze, costs, combinations, cost, plot=False)
    plt.plot(p1[0, :], p1[1, :], 'r')
    plt.plot(p2[0, :], p2[1, :], 'b')
    plt.show()


def plot_paths_combinations():
    # For visualizing only
    costs = {}

    for key, path in paths.items():
        distance_matrix = build_graph(path)
        path, path_cost = TSP(distance_matrix, 0)
        actual_path = get_path_from_indices(paths[key], path)
        costs[key] = {"path": actual_path, "cost": path_cost}
    i = 1

    for combination in combinations:
        title = "Path Combination " + str(i)
        print_enviroment(obstacles, start_point, goals, category1, category2, title)
        path1 = costs[combination[0]]["path"]
        path2 = costs[combination[1]]["path"]
        plt.plot(path1[0, :], path1[1, :])
        plt.plot(path2[0, :], path2[1, :])
        plt.show()
        i += 1


def AStar_only():
    print_enviroment(obstacles, start_point, goals, category1, category2, "A*")
    print("--------A* Alone---------")

    distance_matrices = {}
    costs = {}
    start_time = timer()

    for key, path in paths.items():
        distance_matrices[key] = path_cost_matrix(path, maze, cost, key)
        # Keep track of the updated TSP route
        final_path = path
        path_cost = 0
        for index in range(final_path.shape[1] - 1):
            distance_matrix = distance_matrices[key]
            neighbors = distance_matrix[index, index + 1:]
            if len(neighbors) > 1:
                neighbors = neighbors[:-1]

            nearest_distance = min(neighbors)
            path_cost += nearest_distance

            nearest_neighbor_index = np.where(neighbors == nearest_distance)[0]
            nearest_neighbor_index = (index + 1) + nearest_neighbor_index
            if len(nearest_neighbor_index) > 1:
                nearest_neighbor_index = nearest_neighbor_index[:1]

            if nearest_neighbor_index == (index + 1):
                continue


            temp_neighbor = final_path[:, nearest_neighbor_index].reshape(2,1)
            temp_next_neighbor = final_path[:, index + 1]

            final_path[:, nearest_neighbor_index] = temp_next_neighbor.reshape(2,1)

            final_path[:, index + 1] = temp_neighbor.reshape(2,)

            paths[key] = final_path
            distance_matrices[key] = path_cost_matrix(final_path.astype(int), maze, 1, key)

        costs[key] = {"path": final_path, "cost": path_cost}
    _, _, final_cost = get_full_path(maze, costs, combinations, cost, plot=True)
    plt.show()

    print("Final Cost", final_cost)
    end = timer()
    print("Time (seconds) ", end - start_time)


setup_enviroment()
obstacles, goals, category1, category2, paths, start_point, maze, combinations = get_enviroment()
cost = 1  # cost per movement
astar = AStar()
TSP_only()
method1()
method2()
method3()
method4()
method5()
AStar_only()


