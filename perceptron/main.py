from perceptron import Preceptron
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, Y = datasets.make_blobs(
    n_samples=500, n_features=2, centers=2, cluster_std=1.25, random_state=2
)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.1)

model = Preceptron()
model.fit(train_x, train_y)
predictions = model.predict(test_x)

p1 = [np.amin(train_x[:, 0]), np.amax(train_x[:, 0])]
p2 = [
    (-model.weights[0] * p1[0] - model.bias) / model.weights[1],
    (-model.weights[0] * p1[1] - model.bias) / model.weights[1],
]

def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_pred) * 100

print("Accuracy :", accuracy(predictions, test_y))

plt.plot(p1, p2, "k")
plt.scatter(train_x[:, 0], train_x[:,1], c=train_y)
plt.show()




