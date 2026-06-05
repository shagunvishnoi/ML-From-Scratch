import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

from models.kmeans import KMeans


X, _ = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=1.0,
    random_state=42
)

model = KMeans(
    k=3
)

model.fit(X)

labels = model.labels_

plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels
)

plt.scatter(
    model.centroids[:, 0],
    model.centroids[:, 1],
    marker="X",
    s=200
)

plt.title(
    "KMeans From Scratch"
)

plt.savefig(
    "images/kmeans_clusters.png",
    bbox_inches="tight"
)

plt.close()

print(
    "Clusters Found:",
    len(set(labels))
)