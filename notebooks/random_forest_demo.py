import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from models.random_forest import RandomForest


data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

forest = RandomForest(
    n_trees=10,
    max_depth=5
)

forest.fit(
    X_train,
    y_train
)

preds = forest.predict(
    X_test
)

print(
    "My Forest Accuracy:",
    accuracy_score(
        y_test,
        preds
    )
)

sk_forest = RandomForestClassifier(
    n_estimators=10,
    max_depth=5
)

sk_forest.fit(
    X_train,
    y_train
)

sk_preds = sk_forest.predict(
    X_test
)

print(
    "Sklearn Forest Accuracy:",
    accuracy_score(
        y_test,
        sk_preds
    )
)
