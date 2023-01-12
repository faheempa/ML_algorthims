from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from nb import NaiveBayes
import numpy as np

X,Y  = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=1234)
train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.1)

model=NaiveBayes()
model.fit(train_x,train_y)
predictions = model.predict(test_x)

def accuracy(predicted, real):
    return np.sum(real==predicted)/len(real)*100

print("classification accuracy =", accuracy(predictions, test_y))


# sample data
# plt.scatter(X[:,0], X[:,1], c=Y)
# plt.show()