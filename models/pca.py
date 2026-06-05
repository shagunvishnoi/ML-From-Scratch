import numpy as np


class PCA:

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):

        X = X - np.mean(X, axis=0)

        covariance_matrix = np.cov(X.T)

        eigenvalues, eigenvectors = np.linalg.eig(
            covariance_matrix
        )

        idxs = np.argsort(eigenvalues)[::-1]

        eigenvectors = eigenvectors[:, idxs]

        self.components = eigenvectors[
            :,
            :self.n_components
        ]

    def transform(self, X):

        X = X - np.mean(X, axis=0)

        return np.dot(X, self.components)

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)