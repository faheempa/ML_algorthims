import numpy as np
from collections import Counter


def distance(x1,y1, x2,y2):
    return np.sqrt(np.sum([(x2-x1)**2, (y2-y1)**2]))

class KNN:
    def __init__(self, k=5):
        self.k=k

    def fit(self,X,Y):
        self.train_x=X
        self.train_y=Y

    def predict(self, X):
        return np.array([self.prediction(x[0],x[1]) for x in X])

    def prediction(self, x1,y1):
        # calculate distance
        distances=[distance(x1,y1,sample_x[0],sample_x[1]) for sample_x in self.train_x]
        # find indexes for nearesr neighbor sorted by distance
        k_indexes = np.argsort(distances)[:self.k]
        k_labels = [self.train_y[i] for i in k_indexes]
        # find majority vote
        common_label = Counter(k_labels).most_common(1)
        return common_label[0][0]
