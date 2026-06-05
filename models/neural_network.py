import numpy as np


class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.losses = []

        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        n_samples = X.shape[0]

        for _ in range(self.epochs):
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = self.sigmoid(Z1)

            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self.sigmoid(Z2)

            loss = -np.mean(
                y * np.log(A2 + 1e-9)
                + (1 - y) * np.log(1 - A2 + 1e-9)
            )
            self.losses.append(loss)

            dZ2 = A2 - y
            dW2 = np.dot(A1.T, dZ2) / n_samples
            db2 = np.sum(dZ2, axis=0, keepdims=True) / n_samples

            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * self.sigmoid_derivative(A1)
            dW1 = np.dot(X.T, dZ1) / n_samples
            db1 = np.sum(dZ1, axis=0, keepdims=True) / n_samples

            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

    def predict_proba(self, X):
        A1 = self.sigmoid(np.dot(X, self.W1) + self.b1)
        A2 = self.sigmoid(np.dot(A1, self.W2) + self.b2)
        return A2

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int).flatten()