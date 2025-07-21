import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)
n = 100

data = pd.DataFrame({
    'x': np.random.normal(0, 1, n),
    'y': np.random.normal(0, 1, n),
    'z': np.random.normal(0, 1, n),
    'category': np.random.choice(['A', 'B', 'C'], n)
})

def surfaceplot():
    fig = plt.figure(figsize= (8,6))
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(-3,3,100)
    y = np.linspace(-3,3,100)
    X,Y = np.meshgrid(x,y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    surf = ax.plot_surface(X,Y,Z, cmap='viridis', edgecolor='none')
    fig.colorbar(surf)
    ax.set_title("Surface Plot")
    plt.show()

surfaceplot()


# best first search

from queue import PriorityQueue

def bfs(graph, heuristics, start, goal):
    visited = set()
    pq = PriorityQueue()
    pq.put((heuristics[start],start))
    while not pq.empty():
        h,current = pq.get()
        print("Visited: ", current)

        if goal == current:
            print("Goal reached")
            return

        visited.add(current)
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                pq.put((heuristics[neighbor],neighbor))
    print("Goal not found")

graph = {}
n = int(input("Enter the number of nodes: "))
print("Enter the adjuscents: \n")
for _ in range(n):
    node = input("Node: ")
    neighbors = input(f"Neighbors of node {node} [space separated]: ").split()
    graph[node] = neighbors

heuristics = {}
for node in graph:
    h = int(input(f"Heuristic for node {node}: "))
    heuristics[node] = h

start = input("Enter the start node: ")
goal = input("Enter the goal node: ")

bfs(graph, heuristics, start, goal)