import numpy as np


class KMeans:

    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):

        n_samples, n_features = X.shape

        random_idxs = np.random.choice(
            n_samples,
            self.k,
            replace=False
        )

        self.centroids = X[random_idxs]

        for _ in range(self.max_iters):

            clusters = [[] for _ in range(self.k)]

            for idx, sample in enumerate(X):

                distances = np.linalg.norm(
                    sample - self.centroids,
                    axis=1
                )

                centroid_idx = np.argmin(
                    distances
                )

                clusters[centroid_idx].append(
                    idx
                )

            old_centroids = self.centroids.copy()

            for cluster_idx, cluster in enumerate(clusters):

                if cluster:
                    self.centroids[cluster_idx] = np.mean(
                        X[cluster],
                        axis=0
                    )

            if np.allclose(
                old_centroids,
                self.centroids
            ):
                break

        self.labels_ = np.zeros(
            n_samples,
            dtype=int
        )

        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                self.labels_[sample_idx] = cluster_idx

    def predict(self, X):

        labels = []

        for sample in X:

            distances = np.linalg.norm(
                sample - self.centroids,
                axis=1
            )

            labels.append(
                np.argmin(distances)
            )

        return np.array(labels)