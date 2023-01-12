import numpy as np
from decistion import DecisionTree
from collections import Counter

def bootstrap_sample(X, Y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[idxs], Y[idxs]

def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common

class Random_forest:
    def __init__(
        self, n_trees=10, min_samples_split=2, max_depth=100, n_feats=None
    ) -> None:
        self.n_trees = n_trees
        self.min_sample_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, Y):
        for _ in range(self.n_trees):
            tree = DecisionTree(
                min_samples_split=self.min_sample_split,
                max_depth=self.max_depth,
                n_feats=self.n_feats,
            )
            x_sample, y_sample = bootstrap_sample(X, Y)
            tree.fit(x_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # tree_preds is in this structure: [1111, 0000, 1111]
        tree_preds = np.swapaxes(tree_preds,0, 1)
        # tree_preds is in this structure: [101, 101, 101]
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)