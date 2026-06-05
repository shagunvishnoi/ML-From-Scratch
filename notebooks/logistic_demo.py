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
from sklearn.linear_model import LogisticRegression as SklearnLogistic

from models.logistic_regression import LogisticRegression


data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(
    lr=0.0001,
    epochs=2000
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

print(
    "My Accuracy:",
    accuracy_score(y_test, preds)
)

sk_model = SklearnLogistic(max_iter=5000)

sk_model.fit(X_train, y_train)

sk_preds = sk_model.predict(X_test)

print(
    "Sklearn Accuracy:",
    accuracy_score(y_test, sk_preds)
)

plt.plot(model.losses)

plt.title("Logistic Regression Loss")

plt.xlabel("Iterations")

plt.ylabel("Binary Cross Entropy")

plt.savefig(
    "images/logistic_loss_curve.png",
    bbox_inches="tight"
)

plt.close()