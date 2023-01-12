from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from linear_reg import Linear_regression
import numpy as np

def MSE(actual_values, predicted_values):
    return np.mean((actual_values-predicted_values)**2)

X,Y = datasets.make_regression(n_samples=200, n_features=1, noise=15, random_state=4)
train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.1)

model=Linear_regression(learning_time=0.01)
model.fit(train_x,train_y)
predictions = model.predict(test_x)

mse = MSE(test_y, predictions)
print(mse)

model_line = model.predict(X)
plt.plot(X,model_line, c="black")
plt.scatter(X,Y)
plt.show()