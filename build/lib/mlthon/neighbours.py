import numpy as np
from mlthon.mlthon.backend import _check_data_validity, _dim_check

class KNNClassifier:

    def __init__(self, n_neighbours='auto', p=2):
        self.n_neighbours = n_neighbours
        self.p = p


    def fit(self, X, y):
        _check_data_validity(data=[X, y], names=['X', 'y'])
        _dim_check(data=X, dim=2, name='X')
        _dim_check(data=y, dim=1, name='y')

        self.X = X
        self.y = y

        if self.n_neighbours == 'auto':
            self.n_neighbours = int(np.sqrt(len(self.X)))
            if self.n_neighbours % 2 != 0:
                self.n_neighbours += 1
        
        return self


    def predict(self, X):
        _dim_check(X, 2, 'X')
        predictions = []
        self.confidence = []
        for pred_row in X:
            euclidean_distances = []
            for X_row in self.X:
                distance = np.linalg.norm(X_row - pred_row, ord=self.p)
                euclidean_distances.append(distance)

            neighbours = self.y[np.argsort(euclidean_distances)[:self.n_neighbours]]
            neighbours_bc = np.bincount(neighbours)
            prediction = np.argmax(neighbours_bc)
            self.confidence.append(neighbours_bc[prediction]/len(neighbours))
            predictions.append(prediction)

        predictions = np.array(predictions)
        return predictions
