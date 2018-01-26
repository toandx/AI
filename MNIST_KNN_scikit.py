from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import neighbors,datasets
from sklearn.metrics import accuracy_score

import time

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     iris_X, iris_y, test_size=50)
start_time = time.time()
clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
end_time = time.time()
print (100*accuracy_score(y_test, y_pred))
print (end_time - start_time)