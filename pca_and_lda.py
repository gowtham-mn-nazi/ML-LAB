import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# ------------------ PCA Class ------------------
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        cov = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        idxs = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, idxs[:self.n_components]].T

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)

# ------------------ LDA Class ------------------
class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)
        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))

        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            SW += (X_c - mean_c).T.dot(X_c - mean_c)
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            SB += n_c * mean_diff.dot(mean_diff.T)

        A = np.linalg.inv(SW).dot(SB)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        idxs = np.argsort(eigenvalues)[::-1]
        self.linear_discriminants = eigenvectors[:, idxs[:self.n_components]].T

    def transform(self, X):
        return np.dot(X, self.linear_discriminants.T)

# ------------------ Load Data ------------------
iris = load_iris()
X = iris.data
y = iris.target

# ------------------ PCA ------------------
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

# ------------------ LDA ------------------
lda = LDA(n_components=2)
lda.fit(X, y)
X_lda = lda.transform(X)

# ------------------ Plot ------------------
plt.figure(figsize=(12, 5))

# PCA Plot
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="jet")
plt.title("PCA")
plt.xlabel("PC 1")
plt.ylabel("PC 2")

# LDA Plot
plt.subplot(1, 2, 2)
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap="jet")
plt.title("LDA")
plt.xlabel("LD 1")
plt.ylabel("LD 2")

plt.tight_layout()
plt.show()