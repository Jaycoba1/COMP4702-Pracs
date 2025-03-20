from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from pca import pca

# Load the MNIST dataset
mnist = fetch_openml("mnist_784", version=1, parser="auto")
X = mnist.data.to_numpy()
y = mnist.target.astype(int)

X_pca, X_eigen, _ = pca(X, 100)
total_variance = np.sum(X_eigen)
variance_explained = np.sum(X_eigen[:100]) / total_variance * 100

print("Variance Retained: " + str(variance_explained))
# scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", alpha=0.5)
# plt.colorbar(scatter, label="Digit Class (0-9)")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.title(f"MNIST Data in PCA Space (First Two Principal Components)\nVariance Retained: {variance_explained:.2f}%")
# plt.show()
