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

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from models.decision_tree import DecisionTree


data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

tree = DecisionTree(max_depth=5)

tree.fit(X_train, y_train)

preds = tree.predict(X_test)

print(
    "My Tree Accuracy:",
    accuracy_score(y_test, preds)
)

sk_tree = DecisionTreeClassifier(
    max_depth=5
)

sk_tree.fit(
    X_train,
    y_train
)

sk_preds = sk_tree.predict(
    X_test
)

print(
    "Sklearn Tree Accuracy:",
    accuracy_score(
        y_test,
        sk_preds
    )
)

# ==========================
# Tree Visualization
# ==========================

plt.figure(figsize=(18, 10))

plot_tree(
    sk_tree,
    filled=True,
    feature_names=data.feature_names,
    class_names=["malignant", "benign"]
)

plt.savefig(
    "images/decision_tree_visualization.png",
    bbox_inches="tight"
)

plt.close()

