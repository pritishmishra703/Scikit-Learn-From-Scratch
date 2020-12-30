import numpy as np

class LabelEncoder:
    def fit_transform(self, x):
        X_set = set(x)
        self.classes_ = list(X_set)
        self.classes_.sort()
        X_mod = []

        for i in x:
            X_mod.append(self.classes_.index(i))

        X_mod = np.array(X_mod)
        return X_mod
    
    def inverse_transform(self, x):
        inv_X = []
        for i in x:
            inv_X.append(self.classes_[i])

        inv_X = np.array(inv_X)
        return inv_X

class onehotencoder():
    pass

def train_test_split(X, y, test_size=0.2, shuffle=False):
    if shuffle:
        np.random.shuffle(X)
        np.random.shuffle(y)

    split_val = int(len(X) - len(X)*test_size)
    X_train, X_test = X[:split_val], X[split_val:]
    y_train, y_test = y[:split_val], y[split_val:]
    
    return X_train, X_test, y_train, y_test


class PolyFeatures:
    def checkback(self, vals, degree):
        vals = vals.tolist()
        for i in range(len(vals)):
            sno = degree-vals[i]
            for j in range(i+1, len(vals)):
                if vals[j] == sno:
                    print(vals.index(vals[i]), vals.index(vals[j]))

    def transform(self, X, degree):
        X_copy = X
        self.degree_list = np.ones(X.shape[1])
        print(X.shape[1])

        for i in range(2, degree+1):
            X_copy = np.concatenate((np.array(X_copy), np.array(np.power(X, i))), axis=1)
            self.degree_list = np.append(self.degree_list, i)
            self.checkback(self.degree_list, degree)

        
        return X_copy

class StandardScaler:
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X):
        if self.with_mean == False:
            self.mean_ = 0
        else:
            self.mean_ = np.mean(X)
        
        if self.with_std == False: 
            self.std_ = 1
        else:
            self.std_ = np.std(X)
    
    def fit_transform(self, X):
        self.fit(X)
        return (X - self.mean_)/self.std_
    
    def transform(self, X):
        return (X - self.mean_)/self.std_

def add_dummy_feature(X, value=1):
    return np.array(np.concatenate((np.full((X.shape), value), X), axis=1), dtype=float)

class Binarizer:
    def __init__(self, threshold=0):
        self.threshold = threshold
    
    def transform(self, X):

        if type(X) is list:
            X = np.array(X).reshape(-1, 1)

        binarized = []
        for i in np.nditer(X):
            if i > self.threshold:
                binarized.append(1)
            else:
                binarized.append(0)

        return np.array(binarized).reshape(X.shape, order='F')

class LabelBinarizer:
    def __init__(self, neg_label=0, pos_label=1):
        self.neg_label = neg_label
        self.pos_label = pos_label

        if self.neg_label >= self.pos_label:
            raise ValueError(f"neg_label={self.neg_label} must be strictly less than pos_label={self.pos_label}")

    def fit(self, X):
        self.classes_ = np.array(list(set(X.ravel())))

    def transform(self, X):
        template = np.empty((0, len(self.classes_)), dtype=int)

        for i in np.nditer(X):
            gen_arr = np.full((len(self.classes_)), self.neg_label)
            gen_arr[np.where(self.classes_ == i)[0][0]] = self.pos_label
            template = np.row_stack((template, gen_arr))

        return template 
    
    def fit_transform(self, X):
        self.fit(X)
        template = self.transform(X)
        return template

class MaxAbsScaler:
    def fit(self, X):
        absX = abs(X)
        self.max_value = np.amax(absX, axis=0).astype(float)

    def transform(self, X):
        return X/self.max_value
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        return X*self.max_value
