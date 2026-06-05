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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from models.neural_network import NeuralNetwork


data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = NeuralNetwork(
    input_size=X_train.shape[1],
    hidden_size=16,
    output_size=1,
    lr=0.1,
    epochs=3000
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

print(
    "My Neural Network Accuracy:",
    accuracy_score(y_test, preds)
)

sk_model = MLPClassifier(
    hidden_layer_sizes=(16,),
    max_iter=3000,
    random_state=42
)

sk_model.fit(X_train, y_train)

sk_preds = sk_model.predict(X_test)

print(
    "Sklearn MLP Accuracy:",
    accuracy_score(y_test, sk_preds)
)

plt.figure(figsize=(8, 6))

plt.plot(model.losses)

plt.title("Neural Network Loss Curve")

plt.xlabel("Iterations")

plt.ylabel("Binary Cross Entropy")

plt.savefig(
    "images/neural_network_loss.png",
    bbox_inches="tight"
)

plt.close()