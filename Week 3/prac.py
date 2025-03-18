import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("heightWeightData.csv")

# Extract relevant columns (Height: col 1, Weight: col 2, Class Label: col 0)
X = df.iloc[:, 1:3].values  # Height and Weight
y = df.iloc[:, 0].values    # Class labels

# Apply K-Means clustering (Assuming 2 clusters; adjust K as needed)
kmeans = KMeans(n_clusters=2, n_init=10)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Sort data by class labels
sorted_indices = np.argsort(y)
X_sorted = X[sorted_indices]
y_sorted = y[sorted_indices]

# Plot the data with colors based on the true class labels
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_sorted[:, 0], X_sorted[:, 1], c=y_sorted, cmap='viridis', edgecolor='k', label="Data Points")

# Plot cluster centers
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label="Cluster Centers")

# Labels and legend
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("K-Means Clustering on Height-Weight Data")
plt.legend()
plt.colorbar(scatter, label="Class Labels")
plt.show()
