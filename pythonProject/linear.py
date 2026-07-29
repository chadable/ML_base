import numpy as np

class LinearRegression:

    def __init__(self, lr = 0.001, n_iter = 1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        print(X.shape)
        print(y.shape)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        print(self.weights.shape)
        self.bias = 0

        for _ in range(self.n_iter):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(np.transpose(X), (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db



    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred