import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

def compute_pca_scree(X):
    """Computes PCA eigenvalues and returns variance explained for all components."""
    
    # Step 1: Mean-center the data
    X_meaned = X - np.mean(X, axis=0)
    
    # Step 2: Compute covariance matrix
    covariance_matrix = np.cov(X_meaned, rowvar=False)
    
    # Step 3: Compute eigenvalues
    eigenvalues, _ = np.linalg.eigh(covariance_matrix)
    
    # Step 4: Sort eigenvalues (descending order)
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Step 5: Compute explained variance for each component
    total_variance = np.sum(sorted_eigenvalues)
    explained_variance_ratio = sorted_eigenvalues / total_variance * 100
    
    return explained_variance_ratio

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.to_numpy()

# Compute variance explained for all components
explained_variance_ratio = compute_pca_scree(X)

# Plot Scree Graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='-')
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained (%)")
plt.title("Scree Plot of MNIST PCA")
plt.grid()
plt.show()
