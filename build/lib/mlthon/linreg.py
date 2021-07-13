import numpy as np

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


class RidgeRegression:

    def __init__(self,  alpha=1.0, learning_rate=0.005, max_iter=1000, get_log=False):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = alpha
        self.get_log = get_log

    def fit(self, X, y) :
        self.W = np.random.randn(X.shape[1])
        self.b = 3

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


class BatchGradientDescent:

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


class LassoRegression:

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


class ElasticNet:

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
            LW = (-(2*(X.T).dot(y - y_pred)) + (self.alpha*self.l1_ratio) + (self.alpha*(1 - self.l1_ratio)*self.W))/X.shape[0]
            Lb = -2*np.sum(y - y_pred )/X.shape[0]
            
            self.W -= self.learning_rate*LW
            self.b -= self.learning_rate*Lb

            if self.get_log == True and (i + 1) % 10 == 0:
                    self.y_pred_log.append(y_pred)
                    self.loss_log.append(np.mean(np.square(y - y_pred)))

      
    def predict(self, X) :    
        return np.dot(X, self.W) + self.b


class LogisticRegression:

    def __init__(self, penalty='l2', learning_rate=0.0001, C=1.0, fit_intercept=True, max_iter=5000, 
    l1_ratio=None, print_info=False):
        self.penalty = penalty
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.print_info = print_info

    def fit(self, X, y):
        self.weights = np.random.rand(X.shape[1])
        self.m = len(X)

        for i in range(self.max_iter):
            y_pred = self.predict(X)
            Lw = np.dot(X.T, (y_pred - y)) / self.m
            self.weights = self.weights - self.learning_rate * Lw

            if self.print_info == True and (i + 1) % 100 == 0:
                loss = self.BinaryCrossEntropy(y, y_pred)
                print(f"Iteration: {i+1}, Loss: {loss}")

    def BinaryCrossEntropy(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        term_0 = (1 - y_true) * sum(np.log(1 - y_pred + 1e-7))
        term_1 = y_true * sum(np.log(y_pred + 1e-7))
        return -1.0/self.m * np.mean(term_0+term_1, axis=0)
    
    def predict(self, X):
        predictions = self.sigmoid(np.dot(X, self.weights)).astype('float32')
        predictions = np.where(predictions >= 0.5, 1, 0)
        return predictions

    def sigmoid(self, X):
        return 1/(1 + np.e**(-X))
