from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from knn_class import KNN

iris_data = datasets.load_iris()
X,Y=iris_data.data, iris_data.target
train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.1)

model = KNN(5)
model.fit(train_x, train_y)
predictions = model.predict(test_x)

score = accuracy_score(predictions, test_y)
print(score*100)

# original data
plt.scatter(X[:,0], X[:,1], c=Y)
plt.show()
# predicted 
plt.scatter(test_x[:,0], test_x[:,1], c=predictions)
plt.show()