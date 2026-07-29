import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import KNN
import pandas as pd

ch = datasets.load_diabetes()
X, y = ch.data, ch.target

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=234)

clf = KNN.KNN(k=5)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print(pred)

error = np.sum(pred - y_test)
print(error)
