import numpy as np
from itertools import combinations_with_replacement

class LabelEncoder:
    def fit(self, X):
        X_set = set(X)
        self.classes_ = list(X_set)
        self.classes_.sort()

    def fit_transform(self, X):
        self.fit(X)
        X_mod = []

        for i in X:
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

def train_test_split(X, y, test_size=0.2, shuffle=False, seed=None):
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.shuffle(y)

    split_val = int(len(X) - len(X)*test_size)
    X_train, X_test = X[:split_val], X[split_val:]
    y_train, y_test = y[:split_val], y[split_val:]
    
    return X_train, X_test, y_train, y_test


def polynomial_features(X, degree):
    n_samples, n_features = np.shape(X)

    def index_combinations():
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs
    
    combinations = index_combinations()
    n_output_features = len(combinations)
    X_new = np.empty((n_samples, n_output_features))
    
    for i, index_combs in enumerate(combinations):  
        X_new[:, i] = np.prod(X[:, index_combs], axis=1)

    return X_new


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

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)) -> None:
        self.feature_range = feature_range
    
    def fit(self, X):
        self.data_min_ = np.min(X, axis=0).astype(float)
        self.data_max_ = np.max(X, axis=0).astype(float)
        self.data_range_ = self.data_max_ - self.data_min_
    
    def transform(self, X):
        fr_min = self.feature_range[0]
        fr_max = self.feature_range[1]
        self.scale_ = (fr_max - fr_min)/(self.data_max_ - self.data_min_)
        self.min_ = fr_min - self.data_min_*self.scale_
        return ((X - self.data_min_)/(self.data_max_ - self.data_min_)) * (fr_max - fr_min) + fr_min

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

