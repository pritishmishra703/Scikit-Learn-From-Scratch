import numpy as np
from mlthon.mlthon.backend import _dim_check


class GaussianNB:
	def __init__(self) -> None:
		pass

	def fit(self, X, y):
		_dim_check(X, 2, 'X')
		_dim_check(y, 1, 'y')

		self.classes = np.unique(y)
		self.mean_ = {}
		self.var_ = {}
		self.priors_ = {}

		for class_value in self.classes:
			sep_arr = X[np.where(y == class_value)[0]]
			self.priors_[class_value] = len(sep_arr)/X.shape[0]
			self.mean_[class_value] = np.mean(sep_arr, axis=0)
			self.var_[class_value] = np.var(sep_arr, axis=0)
		
		return self
	
	def predict(self, X):
		_dim_check(X, 2, 'X')

		predictions = []
		for row in X:
			posteriors = []
			for class_value in self.classes:
				prior = self.priors_[class_value]
				conditional = np.sum(np.log(self.gaussian_density(class_value, row)))
				posterior = prior + conditional
				posteriors.append(posterior)

			predictions.append(np.argmax(posteriors))
		
		predictions = np.array(predictions)
		return predictions
	
	def gaussian_density(self, class_value, X):
		mean = self.mean_[class_value]
		var = self.var_[class_value]
		return np.exp((-1/2)*((X - mean)**2) / (2 * var)) / np.sqrt(2 * np.pi * var)
