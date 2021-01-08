import numpy as np

class LinearRegression:
    def fit(self, X, y):
        ones = np.ones(len(X)).reshape(-1, 1)
        X = np.concatenate((ones, X), axis=1)

        slopes = np.matmul(np.linalg.pinv(np.matmul(X.T, X)), np.matmul(X.T, y))

        self.slope = slopes[1:]
        self.intercept = slopes[0]

    def predict(self, X: list):
        self.predicted = np.dot(X, self.slope) + self.intercept
        return self.predicted
