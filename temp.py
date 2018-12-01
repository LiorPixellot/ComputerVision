from __future__ import print_function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
import numpy as np
import cv2
import sklearn
from sklearn.model_selection import train_test_split

X, y = np.arange(10).reshape((5, 2)), range(5)
print(X)





print(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33)

print(X_train)



print(y_train)

print(X_test)


print(y_test)

