import numpy as np


class Node:

    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:

    def __init__(
        self,
        max_depth=10,
        min_samples_split=2
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(
        self,
        X,
        y,
        depth=0
    ):

        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        best_feature, best_threshold = self._best_split(
            X,
            y,
            n_features
        )

        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = X[:, best_feature] > best_threshold

        left = self._grow_tree(
            X[left_idxs],
            y[left_idxs],
            depth + 1
        )

        right = self._grow_tree(
            X[right_idxs],
            y[right_idxs],
            depth + 1
        )

        return Node(
            best_feature,
            best_threshold,
            left,
            right
        )

    def _best_split(
        self,
        X,
        y,
        n_features
    ):

        best_gain = -1
        split_idx = None
        split_threshold = None

        for feature_idx in range(n_features):

            thresholds = np.unique(
                X[:, feature_idx]
            )

            for threshold in thresholds:

                gain = self._information_gain(
                    y,
                    X[:, feature_idx],
                    threshold
                )

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _entropy(self, y):

        hist = np.bincount(y)

        probs = hist / len(y)

        return -np.sum(
            [
                p * np.log2(p)
                for p in probs
                if p > 0
            ]
        )

    def _information_gain(
        self,
        y,
        feature,
        threshold
    ):

        parent_entropy = self._entropy(y)

        left_idxs = feature <= threshold
        right_idxs = feature > threshold

        if (
            len(y[left_idxs]) == 0
            or len(y[right_idxs]) == 0
        ):
            return 0

        n = len(y)

        n_left = len(y[left_idxs])
        n_right = len(y[right_idxs])

        e_left = self._entropy(
            y[left_idxs]
        )

        e_right = self._entropy(
            y[right_idxs]
        )

        child_entropy = (
            n_left / n
        ) * e_left + (
            n_right / n
        ) * e_right

        return parent_entropy - child_entropy

    def _most_common_label(
        self,
        y
    ):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array(
            [
                self._traverse_tree(
                    x,
                    self.root
                )
                for x in X
            ]
        )

    def _traverse_tree(
        self,
        x,
        node
    ):

        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(
                x,
                node.left
            )

        return self._traverse_tree(
            x,
            node.right
        )