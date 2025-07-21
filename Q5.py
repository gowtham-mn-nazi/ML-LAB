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

def boxplot():
    plt.figure(figsize=(6,4))
    sns.boxplot(data=data, x='category', y='z')
    plt.title("Box Plot")
    plt.show()

boxplot()

# alpha beta

def alphabeta(depth, node_index, is_maximizing_player, scores, target_depth, alpha, beta):
    if depth == target_depth:
        return scores[node_index]
    if is_maximizing_player:
        max_eval = float('-inf')
        for i in range(2):
            eval = alphabeta(depth + 1, node_index * 2 + i, False, scores, target_depth,alpha, beta)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break # Beta cut-off
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(2):
            eval = alphabeta(depth + 1, node_index * 2 + i, True, scores, target_depth,alpha, beta)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break # Alpha cut-off
        return min_eval

# ---------------- Input Section ----------------
print("Enter the depth of the game tree (e.g., 3 for 8 leaf nodes):")
tree_depth = int(input("Depth: "))
num_leaves = 2 ** tree_depth
print(f"Enter {num_leaves} leaf node scores separated by space:")
scores_input = input("Scores: ")
scores = list(map(int, scores_input.strip().split()))
if len(scores) != num_leaves:
    print(f"Error: Expected {num_leaves} scores, but got {len(scores)}.")
else:
    optimal_value_ab = alphabeta(0, 0, True, scores, tree_depth, float('-inf'),float('inf'))
    print(f"Optimal value using Alpha-Beta Pruning: {optimal_value_ab}")