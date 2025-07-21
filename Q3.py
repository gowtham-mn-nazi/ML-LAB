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

def contourplot():
    x = np.linspace(-3,3,100)
    y = np.linspace(-3,3,100)
    X,Y = np.meshgrid(x,y)
    Z = np.sin(X**2 + Y**2)
    plt.figure(figsize = (6,5))
    cp = plt.contourf(X,Y,Z,cmap='viridis')
    plt.colorbar(cp)
    plt.title("Contour plot")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

contourplot()

# A star algorithm

from queue import PriorityQueue

def a_star(graph, heuristics, start, goal):
    pq = PriorityQueue()
    pq.put((heuristics[start], 0, start))  # (f = g + h, g, node)
    came_from = {}
    g_score = {start: 0}
    visited = set()

    while not pq.empty():
        f, g, current = pq.get()
        print("Visited:", current)

        if current == goal:
            print("Goal reached!")
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            print("Best Path:", ' -> '.join(path))
            return

        visited.add(current)

        for neighbor, cost in graph.get(current, []):
            if neighbor in visited:
                continue
            tentative_g = g + cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristics[neighbor]
                pq.put((f_score, tentative_g, neighbor))
                came_from[neighbor] = current

    print("Goal not reachable.")

# ----------- Input Section -----------

graph = {}
n = int(input("Enter number of nodes: "))
print("Enter adjacency list with cost (format: neighbor1:cost1 neighbor2:cost2):")
for _ in range(n):
    node = input("Node: ")
    neighbors_input = input(f"Enter neighbors of {node}: ").split()
    neighbors = []
    for entry in neighbors_input:
        neighbor, cost = entry.split(":")
        neighbors.append((neighbor, int(cost)))
    graph[node] = neighbors

heuristics = {}
print("Enter heuristic values:")
for node in graph:
    h = int(input(f"Heuristic for {node}: "))
    heuristics[node] = h

start = input("Enter start node: ")
goal = input("Enter goal node: ")

# ----------- Run the algorithm -----------

print("\nA* Search Path:")
a_star(graph, heuristics, start, goal)