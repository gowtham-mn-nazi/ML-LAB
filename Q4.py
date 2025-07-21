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

def heatmap():
    correlation = data[['x','y','z']].corr()
    plt.figure(figsize=(5,4))
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title("Heat Map")
    plt.show()

heatmap()

# min max algorithm

def minimax(depth, node_index, is_maximizing_player, scores, target_depth):
    if depth == target_depth:
        return scores[node_index]
    if is_maximizing_player:
        return max(
            minimax(depth + 1, node_index * 2, False, scores, target_depth),
            minimax(depth + 1, node_index * 2 + 1, False, scores, target_depth)
        )
    else:
        return min(
            minimax(depth + 1, node_index * 2, True, scores, target_depth),
            minimax(depth + 1, node_index * 2 + 1, True, scores, target_depth)
        )
    
print("Enter the depth of the game tree (e.g., 3 for 8 leaf nodes):")
tree_depth = int(input("Depth: "))
num_leaves = 2 ** tree_depth
print(f"Enter {num_leaves} leaf node scores separated by space:")
scores_input = input("Scores: ")
scores = list(map(int, scores_input.strip().split()))
if len(scores) != num_leaves:
    print(f"Error: Expected {num_leaves} scores, but got {len(scores)}.")
else:
    optimal_value = minimax(0, 0, True, scores, tree_depth)
    print(f"\nOptimal value using Minimax: {optimal_value}")