import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris

# Load sample Iris data (first 6 samples for simplicity)
iris = load_iris()
data = iris.data[:6]

# Define proximity matrix computation
def proximity_matrix(data):
    n = data.shape[0]
    proximity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(data[i] - data[j])
            proximity_matrix[i, j] = distance
            proximity_matrix[j, i] = distance
    return proximity_matrix

# Define dendrogram plotting function
def plot_dendrogram(data, method):
    linkage_matrix = linkage(data, method=method)
    dendrogram(linkage_matrix)
    plt.title(f'Dendrogram - {method} linkage')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.show()

# Calculate and print the proximity matrix
print("Proximity matrix:")
print(np.round(proximity_matrix(data), 2))

# Plot dendrograms for different linkage methods
plot_dendrogram(data, 'single')    # Single-linkage
plot_dendrogram(data, 'complete')  # Complete-linkage
