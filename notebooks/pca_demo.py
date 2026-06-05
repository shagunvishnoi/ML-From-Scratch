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

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as SklearnPCA

from models.pca import PCA


iris = load_iris()

X = iris.data
y = iris.target

# Your PCA
my_pca = PCA(
    n_components=2
)

X_reduced = my_pca.fit_transform(X)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    c=y
)

plt.title("PCA From Scratch")

plt.savefig(
    "images/pca_from_scratch.png",
    bbox_inches="tight"
)

plt.close()


# sklearn PCA

sk_pca = SklearnPCA(
    n_components=2
)

X_sk = sk_pca.fit_transform(X)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_sk[:, 0],
    X_sk[:, 1],
    c=y
)

plt.title("Sklearn PCA")

plt.savefig(
    "images/pca_sklearn.png",
    bbox_inches="tight"
)

plt.close()

print(
    "My PCA Shape:",
    X_reduced.shape
)

print(
    "Sklearn PCA Shape:",
    X_sk.shape
)