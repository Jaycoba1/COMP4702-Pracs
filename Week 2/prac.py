import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

regr = pd.read_csv("w3regr.csv", names=["x", "y"], header=None)
classif = pd.read_csv("w3classif.csv", header=None, names=["f1","f2","classification"])
features = classif.iloc[:,:-1]
classification = classif.iloc[:, -1]
train_f, test_f, train_c, test_c = train_test_split(features, classification, test_size=0.3, shuffle=True)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_f, train_c)
train_predict_c = knn.predict(train_f)
test_predict_c = knn.predict(test_f)
train_loss = 1 - accuracy_score(train_c, train_predict_c)
test_loss = 1 - accuracy_score(test_c, test_predict_c)

# Create mesh grid for decision boundary
f1_min, f1_max = features.to_numpy()[:, 0].min() - 1, features.to_numpy()[:, 0].max() + 1
f2_min, f2_max = features.to_numpy()[:, 1].min() - 1, features.to_numpy()[:, 1].max() + 1

f1, f2 = np.meshgrid(np.linspace(f1_min, f1_max, 200),
                     np.linspace(f2_min, f2_max, 200))

# Predict class for each point in the mesh grid
c = knn.predict(np.c_[f1.ravel(), f2.ravel()])
c = c.reshape(f1.shape)

cmap_light = ListedColormap(["#FFAAAA", "#AAAAFF", "#AAFFAA"])  # Light colors for background
plt.contourf(f1, f2, c, cmap=cmap_light)

# Plot decision boundary
plt.figure(figsize=(8, 6))
cmap_light = ListedColormap(["#FFAAAA", "#AAAAFF", "#AAFFAA"])  # Light colors for background
cmap_bold = ListedColormap(["#FF0000", "#0000FF", "#00AA00"])  # Bold colors for points

plt.contourf(f1, f2, c, cmap=cmap_light, alpha=0.5)

# Print results
print(f"Training Loss (Misclassification Rate): {train_loss:.4f}")
print(f"Testing Loss (Misclassification Rate): {test_loss:.4f}")

# train_f.to_csv("train_f.csv", index=False)
# train_c.to_csv("train_c.csv", index=False)

plt.scatter(classif["f1"], classif["f2"], c=classif["classification"])
plt.show()