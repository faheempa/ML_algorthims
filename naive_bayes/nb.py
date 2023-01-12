import numpy as np

class NaiveBayes:
    def fit(self, X, Y):
        n_sample, n_feature = X.shape
        self.classes = np.unique(Y)
        n_classes = len(self.classes)

        self._mean = np.zeros((n_classes, n_feature), dtype=np.float64)
        self._var = np.zeros((n_classes, n_feature), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for c in self.classes:
            x_c = X[c == Y]
            self._mean[c, :] = x_c.mean(axis=0)
            self._var[c, :] = x_c.var(axis=0)
            self._priors[c] = x_c.shape[0] / float(n_sample)

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        posteriors = []
        for i, e in enumerate(self.classes):
            prior = np.log(self._priors[i])
            class_conditional = np.sum(np.log(self._pdf(i, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def _pdf(self, cls_i, x):
        mean = self._mean[cls_i]
        var = self._var[cls_i]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
