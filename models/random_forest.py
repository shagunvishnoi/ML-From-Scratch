import numpy as np

from models.decision_tree import DecisionTree


class RandomForest:

    def __init__(
        self,
        n_trees=10,
        max_depth=10
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(
        self,
        X,
        y
    ):

        self.trees = []

        n_samples = X.shape[0]

        for _ in range(self.n_trees):

            idxs = np.random.choice(
                n_samples,
                n_samples,
                replace=True
            )

            X_sample = X[idxs]
            y_sample = y[idxs]

            tree = DecisionTree(
                max_depth=self.max_depth
            )

            tree.fit(
                X_sample,
                y_sample
            )

            self.trees.append(tree)

    def predict(self, X):

        tree_preds = np.array(
            [
                tree.predict(X)
                for tree in self.trees
            ]
        )

        tree_preds = np.swapaxes(
            tree_preds,
            0,
            1
        )

        return np.array(
            [
                np.bincount(row).argmax()
                for row in tree_preds
            ]
        )