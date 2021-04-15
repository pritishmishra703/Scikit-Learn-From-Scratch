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


class LogisticRegression:
    def __init__(self, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
    intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, 
    multi_class='auto', verbose=0, l1_ratio=None):

        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.l1_ratio = l1_ratio

    def fit(self, X, y):
        
        if self.solver == "newton-cg":
            pass

