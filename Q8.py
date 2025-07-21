import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def kmeans(X, K, max_iters=100):
    # Initialize centroids using the first K points
    centroids = X[:K]
    
    for _ in range(max_iters):
        # Assign each data point to the nearest centroid
        expanded_x = X[:, np.newaxis]
        euc_dist = np.linalg.norm(expanded_x - centroids, axis=2)
        labels = np.argmin(euc_dist, axis=1)

        # Update the centroids based on the assigned points
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])

        # If centroids haven't changed, stop iterating
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels, centroids

# Load the Iris dataset
X = load_iris().data
K = 3
labels, centroids = kmeans(X, K)

# Print results
print("Labels:", labels)
print("Centroids:\n", centroids)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', s=200)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('K-means Clustering of Iris Dataset')
plt.show()