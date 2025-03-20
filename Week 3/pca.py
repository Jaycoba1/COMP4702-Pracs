import numpy as np

def pca(X, k=2):
    """
    Perform PCA on dataset X and reduce its dimensionality to k.

    Parameters:
    - X: Data matrix (n_samples, n_features)
    - k: Number of principal components to keep

    Returns:
    - X_reduced: Transformed data in reduced dimensions
    - eigenvalues: Eigenvalues of the covariance matrix
    - eigenvectors: Corresponding eigenvectors
    """
    # Step 1: Mean center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Step 2: Compute covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Step 3: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Step 4: Sort eigenvectors by descending eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Descending order
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 5: Select top-k eigenvectors
    top_eigenvectors = eigenvectors[:, :k]

    # Step 6: Project data onto new subspace
    X_reduced = np.dot(X_centered, top_eigenvectors)

    return X_reduced, eigenvalues, eigenvectors

