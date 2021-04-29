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


class RidgeRegression() :

    def __init__(self,  alpha=1.0, learning_rate=0.005, max_iter=1000, get_log=False):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = alpha
        self.get_log = get_log

    def fit(self, X, y) :
        self.W = np.zeros(X.shape[1])
        self.b = 0

        if self.get_log == True:
            self.y_pred_log = []
            self.loss_log = []

        for i in range(self.max_iter):
            y_pred = self.predict(X)
            LW = (-(2*(X.T).dot(y - y_pred)) + (2*self.alpha*self.W))/X.shape[0]
            Lb = -2*np.sum(y - y_pred)/X.shape[0]
            
            self.W -= self.learning_rate*LW
            self.b -= self.learning_rate*Lb

            if self.get_log == True:
                if i%10 == 0:
                    self.y_pred_log.append(y_pred)
                self.loss_log.append(np.mean(np.square(y - y_pred)))

      
    def predict(self, X):
        return np.dot(X, self.W) + self.b


class GradientDescent() :

    def __init__(self, learning_rate=0.001, max_iter=1000, get_log=False):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.get_log = get_log

    def fit(self, X, y) :
        self.W = np.zeros(X.shape[1])
        self.b = 0

        for i in range(self.max_iter):
            y_pred = self.predict(X)
            LW = -(2*(X.T).dot(y - y_pred))/X.shape[0]
            Lb = -2*np.sum(y - y_pred)/X.shape[0]
            
            self.W -= self.learning_rate*LW
            self.b -= self.learning_rate*Lb

            if self.get_log == True:
                if i%10 == 0:
                    self.y_pred_log.append(y_pred)
                self.loss_log.append(np.mean(np.square(y - y_pred)))

      
    def predict(self, X) :    
        return np.dot(X, self.W) + self.b


class LassoRegression() :

    def __init__(self,  alpha=1.0, learning_rate=0.005, max_iter=1000, get_log=False):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = alpha
        self.get_log = get_log

    def fit(self, X, y) :
        self.W = np.zeros(X.shape[1])
        self.b = 0

        for i in range(self.max_iter):
            y_pred = self.predict(X)
            LW = (-(2*(X.T).dot(y - y_pred)) + (self.alpha))/X.shape[0]
            Lb = -2*np.sum(y - y_pred )/X.shape[0]
            
            self.W -= self.learning_rate*LW
            self.b -= self.learning_rate*Lb

            if self.get_log == True:
                if i%10 == 0:
                    self.y_pred_log.append(y_pred)
                self.loss_log.append(np.mean(np.square(y - y_pred)))

      
    def predict(self, X) :    
        return np.dot(X, self.W) + self.b


class ElasticNet() :

    def __init__(self,  alpha=1.0, l1_ratio=0.5, learning_rate=0.005, max_iter=1000, get_log=False):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.get_log = get_log

    def fit(self, X, y) :
        self.W = np.zeros(X.shape[1])
        self.b = 0

        for i in range(self.max_iter):
            y_pred = self.predict(X)
            LW = (-(2*(X.T).dot(y - y_pred)) + (self.alpha*self.l1_ratio) + (2*self.alpha*((1 - self.l1_ratio)*0.5)*self.W))/X.shape[0]
            Lb = -2*np.sum(y - y_pred )/X.shape[0]
            
            self.W -= self.learning_rate*LW
            self.b -= self.learning_rate*Lb

            if self.get_log == True:
                if i%10 == 0:
                    self.y_pred_log.append(y_pred)
                self.loss_log.append(np.mean(np.square(y - y_pred)))

      
    def predict(self, X) :    
        return np.dot(X, self.W) + self.b
