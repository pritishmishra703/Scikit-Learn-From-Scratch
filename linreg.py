import numpy as np
from numpy.core.defchararray import center

class LinearRegression:
    def fit(self, X, y):
        ones = np.ones(len(X)).reshape(-1, 1)
        X = np.concatenate((ones, X), axis=1)

        B = np.matmul(np.linalg.pinv(np.matmul(X.T, X)), np.matmul(X.T, y))

        self.slope = B[1:]
        self.intercept = B[0]

    def predict(self, X):
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
        w = 1
        b = 0
        lr = 0.01
        n = len(X)
        print(X[0])
        for i in range(self.max_iter):
            y_pred = X[0][0]*w + b
            md = - ((y[0]*(1 - y_pred)*X) - (1 - y[0])*y_pred*X)
            # md = 1/(1 - np.e**(-md))
            w = w - lr*md
            print(w)


class RidgeRegression() :

    def __init__(self,  alpha=1.0, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = alpha

    def fit(self, X, y) :
        self.W = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(self.max_iter):
            y_pred = self.predict(X)
            LW = (-(2*(X.T).dot(y - y_pred)) + (2*self.alpha*self.W))/X.shape[0]
            Lb = -2*np.sum(y - y_pred )/X.shape[0]
            
            self.W -= self.learning_rate*LW
            self.b -= self.learning_rate*Lb

      
    def predict(self, X) :    
        return np.dot(X, self.W) + self.b


class GradientDescent() :

    def __init__(self, learning_rate=0.001, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit(self, X, y) :
        self.W = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(self.max_iter):
            y_pred = self.predict(X)
            LW = -(2*(X.T).dot(y - y_pred))/X.shape[0]
            Lb = -2*np.sum(y - y_pred)/X.shape[0]
            
            self.W -= self.learning_rate*LW
            self.b -= self.learning_rate*Lb

      
    def predict(self, X) :    
        return np.dot(X, self.W) + self.b


class LassoRegression() :

    def __init__(self,  alpha=1.0, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = alpha

    def fit(self, X, y) :
        self.W = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(self.max_iter):
            y_pred = self.predict(X)
            LW = (-(2*(X.T).dot(y - y_pred)) + (self.alpha*self.W))/X.shape[0]
            Lb = -2*np.sum(y - y_pred )/X.shape[0]
            
            self.W -= self.learning_rate*LW
            self.b -= self.learning_rate*Lb

      
    def predict(self, X) :    
        return np.dot(X, self.W) + self.b

X = list(range(1, 21))
y = []
for i in X:
    y.append(i**2)

X, y = np.array(X).reshape(-1, 1), np.array(y)

reg = GradientDescent(learning_rate=0.0001, max_iter=1000)
reg.fit(X, y)
print(reg.W)
print(reg.b)
print(reg.predict(np.array([12]).reshape(-1, 1)))
