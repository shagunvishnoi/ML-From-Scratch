import sys
import os

print("SCRIPT STARTED")

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression as SklearnLR

from models.linear_regression import LinearRegression


# Generate dataset
X, y = make_regression(
    n_samples=100,
    n_features=1,
    noise=10,
    random_state=42
)

# Train your model
model = LinearRegression(
    lr=0.01,
    epochs=1000
)

model.fit(X, y)

predictions = model.predict(X)

# ==========================
# Regression Plot
# ==========================

plt.figure(figsize=(8, 6))

plt.scatter(X, y)

plt.plot(X, predictions)

plt.title("Linear Regression From Scratch")

plt.savefig(
    "images/linear_regression_fit.png",
    bbox_inches="tight"
)

plt.close()


# ==========================
# Loss Curve
# ==========================

plt.figure(figsize=(8, 6))

plt.plot(model.losses)

plt.title("Gradient Descent Loss Curve")

plt.xlabel("Iterations")

plt.ylabel("MSE Loss")

plt.savefig(
    "images/loss_curve.png",
    bbox_inches="tight"
)

plt.close()


# ==========================
# Compare with sklearn
# ==========================

sk_model = SklearnLR()

sk_model.fit(X, y)

sk_preds = sk_model.predict(X)

print(
    "My Model MSE:",
    mean_squared_error(y, predictions)
)

print(
    "Sklearn MSE:",
    mean_squared_error(y, sk_preds)
)

print("SCRIPT FINISHED")