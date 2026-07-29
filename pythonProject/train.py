import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import KNN
import pandas as pd
from linear import LinearRegression


X, y = datasets.make_regression(n_samples=100, n_features = 1, noise = 20, random_state = 234)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=234)
def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)
# clf = KNN.KNN(k=5)
# clf.fit(X_train, y_train)
# pred = clf.predict(X_test)
#
# print(pred)
#
# mse = mse(y_test,pred)
# print(mse)

reg = LinearRegression()
reg.fit(X_train, y_train)
prediction = reg.predict(X_test)
mse = mse(y_test,prediction)
print(mse)

