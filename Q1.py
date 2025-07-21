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

def scatterplot():
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=data, x='x', y='y', hue='category')
    plt.title("2D scatter plot")
    plt.show()

scatterplot()


# hill climbing algorithm

import random

def object(x):
    return (-(x**2) + 5)

def hill(start,size,n):
    current = start
    score = object(current)
    for x in range(n):
        new = current + random.uniform(-size, size)
        newScore = object(new)
        print(f"{x+1}. x={current:.4f}, f(x)={score:.4f}")
        if newScore > score:
            current = new
            score = newScore
        else: 
            pass
    print("\nSolution: ")
    print(f"x={current:.4f}, f(x)={score:.4f}")
    return current, score

best, bestScore = hill(0.1,0.05,5)