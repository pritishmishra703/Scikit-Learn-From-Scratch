import numpy as np

class KMeans:

    def __init__(self, k=8, n_iter=100, random_state=None):
        self.k = k
        self.n_iter = n_iter
        self.random_state = random_state


    def fit(self, X):
        np.random.seed(self.random_state)
        centroids_index = np.random.choice(X.shape[0], size=self.k, replace=False)
        self.centroids = X[centroids_index]


        for _ in range(self.n_iter):
            # Intialize a dictionary of classifications
            classifications = {}
            for j in range(self.k):
                classifications[j] = []

            # Assigns Every Data point to a class
            for X_row in X:
                distance = np.linalg.norm(X_row - self.centroids, axis=1)
                classification = np.argmin(distance)
                classifications[classification].append(X_row)

            # Calculating the new centroid
            for classification in classifications:
                self.centroids[classification] = np.average(classifications[classification], axis=0)


    def predict(self, X):
        predictions = []
        for pred_row in X:
            distance = np.linalg.norm(pred_row - self.centroids, axis=1)
            prediction = np.argmin(distance)
            predictions.append(prediction)
        return np.array(predictions)
