import numpy as np

class Preceptron:
    def __init__(self, lr=0.01, loop=1000) -> None:
        self.lr = lr
        self.loop = loop
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y = np.array([1 if i > 0 else 0 for i in Y])

        for _ in range(self.loop):
            for i, e in enumerate(X):
                linear_product = np.dot(e, self.weights) + self.bias
                y_predicted = self.activation_func(linear_product)

                update = self.lr * (y[i] - y_predicted)
                self.weights += update * e
                self.bias += update

    def predict(self, X):
        linear_product = np.dot(X, self.weights) + self.bias
        return self.activation_func(linear_product)
