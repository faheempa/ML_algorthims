from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from logistic_reg import LogisticRegression
import numpy as np

data_set = datasets.load_breast_cancer()
X,Y = data_set.data, data_set.target
train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.1)

model=LogisticRegression()
model.fit(train_x,train_y)
predictions = model.predict(test_x)

def accuracy(predicted, real):
    return np.sum(real==predicted)/len(real)*100

print("classification accuracy =", accuracy(predictions, test_y))

# sample data
# plt.scatter(X[:,0], X[:,1], c=Y)
# plt.show()