import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

category1 = None
category2 = None

def get_enviroment():
    num_of_rows = 100
    num_of_columns = 100
    global category1
    global category2

    obstacle1 = np.array([[20, 13], [50, 5]])
    obstacle2 = np.array([[60, 45], [30, 5]])
    obstacle3 = np.array([[20, 70], [50, 5]])
    # obstacle4 = np.array([[10, 45], [30, 5]])
    # obstacle5 = np.array([[70, 80], [20, 5]])
    # obstacle6 = np.array([[10, 80], [20, 5]])
    # obstacle7 = np.array([[45, 25], [5, 40]])


    obstacles = {
        "obstacle1": obstacle1,
        "obstacle2": obstacle2,
        "obstacle3": obstacle3,
        # "obstacle4": obstacle4,
        # "obstacle5": obstacle5,
        # "obstacle6": obstacle6,
        # "obstacle7": obstacle7
    }

    # Goals: 2 goals 1 for each category [x; y]
    goals = np.array([[10, 80], [90, 90]])

    # Start Point
    start_point = np.array([[40], [10]])

    # combinations per row
    combinations = np.array([["path1", "path3"], ["path2", "path4"]])

    path1 = np.hstack((start_point, category1, goals[:, 0].reshape(2, 1)))
    path2 = np.hstack((start_point, category2, goals[:, 1].reshape(2, 1)))
    path3 = np.hstack((goals[:, 0].reshape(2, 1), category2, goals[:, 1].reshape(2, 1)))
    path4 = np.hstack((goals[:, 1].reshape(2, 1), category1, goals[:, 0].reshape(2, 1)))

    paths = {
        "path1": path1,
        "path2": path2,
        "path3": path3,
        "path4": path4,
    }

    # maze : 1s for obstacles
    maze = np.zeros((num_of_rows, num_of_columns))
    for key in obstacles:
        obstacle = obstacles[key]
        x = obstacle[0, :][0]
        y = obstacle[0, :][1]
        width = obstacle[1, :][0]
        height = obstacle[1, :][1]
        maze[x:x + width + 1, y:y + height + 1] = 1

    return obstacles, goals, category1, category2, paths, start_point, maze, combinations


def setup_enviroment():
    global category1
    global category2

    # Categories [x ; y]

    # Environment 1
    category1 = np.array([[62, 41, 50, 33, 61], [10, 80, 21, 37, 67]])
    category2 = np.array([[78, 38, 26, 3, 9], [36, 64, 29, 81, 37]])

    # Environment 2
    # category1 = np.array([[18, 67, 23, 57, 34], [6, 13, 81, 89, 77]])
    # category2 = np.array([[86, 79, 21, 47, 41], [2, 71, 41, 51, 47]])
    #
    # Environment 3
    # category1 = np.array([[41, 68, 90, 45, 90, 18, 15, 52, 10, 22, 75, 40, 84], [45, 77, 90, 24, 40, 70, 12, 64, 60, 40, 24, 80, 74]])
    # category2 = np.array([[13, 56, 77, 17, 80, 28, 40, 64, 70, 17, 81, 50, 10], [53, 31, 36, 77, 20, 44, 60, 80, 43, 43, 84, 90, 24]])

    #Environment 4
    # category1 = np.array([[3, 68, 95, 60, 28, 96, 48, 25], [43, 64, 98, 41, 5, 46, 79, 35]])
    # category2 = np.array([[21, 40, 49, 38, 10, 80, 15, 95], [28, 20, 87, 52, 70, 23, 42, 20]])


def print_enviroment(obstacles, start_point, goals, category1, category2, title=None):
    fig, ax = plt.subplots()
    plt.ylim([0, 100])
    plt.xlim([0, 100])

    # Print Obstacles
    for obstacle_vertex in obstacles:
        obstacle = obstacles[obstacle_vertex]
        x = obstacle[0, :][0]
        y = obstacle[0, :][1]
        width = obstacle[1, :][0]
        height = obstacle[1, :][1]
        ax.add_patch(Rectangle((x, y), width, height, fill=True, fc='black'))

    # Plot goals
    plt.scatter(goals[0, 0], goals[1, 0], marker="*", color='b', label="Category 1 Goal")
    plt.scatter(goals[0, 1], goals[1, 1], marker="*", color='m', label="Category 2 Goal")

    # Plot points
    ax.scatter(category1[0, :], category1[1, :], marker='x', color='b', label="Category 1")
    ax.scatter(category2[0, :], category2[1, :], marker='o', color='m', label="Category 2")
    plt.scatter(start_point[0], start_point[1], marker='*', color='r', label="Starting Point")

    ax.legend()
    # Plot title
    if title is not None:
        plt.title(title)
