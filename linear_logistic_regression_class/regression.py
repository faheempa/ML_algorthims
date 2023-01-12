import numpy as np

class Regression:
    def __init__(self, loop=1000, learning_time=0.001):
        self.loop = loop
        self.learning_time = learning_time
        self.weight = None
        self.bias = None

    def fit(self, x, y):
        self.n_sample, self.features = x.shape
        self.weight = np.zeros(self.features)
        self.bias = 0
        for _ in range(self.loop):
            y_predicted = self._approximation(self, x)

        dw = (1 / self.n_sample) * np.dot(x.T, (y_predicted - y))
        db = (1 / self.n_sample) * np.sum(y_predicted - y)

        self.weight -= self.learning_time * dw
        self.bias -= self.learning_time * db

    def _approximation(self, x):
        return NotImplementedError()


class Linear_regression(Regression):
    def __init__(self, loop=1000, learning_time=0.001):
        super().__init__(loop, learning_time)

    def _approximation(self, x):
        return np.dot(x, self.weight) + self.bias

    def predict(self, x):
        return np.dot(x, self.weight) + self.bias


class LogisticRegression(Regression):
    def __init__(self, loop=1000, learning_time=0.001):
        super().__init__(loop, learning_time)

    def _approximation(self, x):
        linear_model = np.dot(x, self.weight) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, x):
        linear_model = np.dot(x, self.weight) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
