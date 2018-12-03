from __future__ import print_function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
import numpy as np
import cv2
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
import struct
import sys




def main():

    fast = False
    if(len(sys.argv) > 1 and sys.argv[1]=='fast'):
        print ("running fast version")
        fast = True

    print("IRIS KNN")
    testData, testLabels, trainData, trainLabels, valData, valLabels = ExtructIris(fast)
    PerformKnn(testData, testLabels, trainData, trainLabels, valData, valLabels)
    PerformSvm(testData, testLabels, trainData, trainLabels, valData, valLabels)
    print("\n\n MNIST KNN")
    testData, testLabels, trainData, trainLabels, valData, valLabels = ExtructMNIST(fast)
    PerformKnn(testData, testLabels, trainData, trainLabels, valData, valLabels)
    PerformSvm(testData, testLabels, trainData, trainLabels, valData, valLabels)

def ExtructIris(fast):
    iris = datasets.load_iris()
    trainData, trainLabels = iris.data ,iris.target
    (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1)
    (trainData, testData, trainLabels, testLabels) = train_test_split(trainData, trainLabels)
    return testData, testLabels, trainData, trainLabels, valData, valLabels

def ExtructMNIST(fast):
    trainData, trainLabels = loadlocal_mnist(images_path='train-images.idx3-ubyte',
                                             labels_path='train-labels.idx1-ubyte')
    testData, testLabels = loadlocal_mnist(images_path='t10k-images.idx3-ubyte', labels_path='t10k-labels.idx1-ubyte')
    if fast:
        # to make it fast we take only the 10% for the training
        (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.9)
    # we take 10% of the training data and use that for validation
    (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1)
    return testData, testLabels, trainData, trainLabels, valData, valLabels




def PerformKnn(testData, testLabels, trainData, trainLabels, valData, valLabels):
    BestK = FindBestK(trainData, trainLabels, valData, valLabels)
    print("best K is %d" % (BestK))
    # re-train our classifier using the best k value and predict the labels of the
    # test data
    model = KNeighborsClassifier(n_neighbors=BestK)
    model.fit(trainData, trainLabels)
    predictions = model.predict(testData)
    # show a final classification report demonstrating the accuracy of the classifier
    # for each of the digits
    print("Classification report for classifier %s:\n%s"
      % (model, metrics.classification_report(testLabels, predictions)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(testLabels, predictions))


def PerformSvm(testData, testLabels, trainData, trainLabels, valData, valLabels):
    BestC ,kernel = FindBestCAndKernel(trainData, trainLabels, valData, valLabels)
    print("best C is %f best kernel is %s" % (BestC,kernel))
    # re-train our classifier using the best k value and predict the labels of the
    # test data
    model = svm.SVC(kernel=kernel, C=BestC)
    model.fit(trainData, trainLabels)
    predictions = model.predict(testData)
    # show a final classification report demonstrating the accuracy of the classifier
    # for each of the digits
    print("Classification report for classifier %s:\n%s"
      % (model, metrics.classification_report(testLabels, predictions)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(testLabels, predictions))


def FindBestK(trainData, trainLabels, valData, valLabels):
    print("FindBestK")
    MaxScore = 0.0
    minK = 0
    # try different values of K for the best classification results
    for k in range(1, 15, 2):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(trainData, trainLabels)
        score = model.score(valData, valLabels)
        print("k=" ,k,"score=", score)
        if MaxScore < score:
            MaxScore = score
            minK = k
    return minK

def FindBestCAndKernel(trainData, trainLabels, valData, valLabels):
    print("FindBestCAndKernel")
    MaxScore = 0.0
    minC = 0
    minKern = 'linear'
    # try different values of C and kernel for the best classification results
    for kern in ['linear','rbf','poly']:
        for c in np.linspace(0.1,1,5):
            model = svm.SVC(kernel=kern, C=c)
            model.fit(trainData, trainLabels)
            score = model.score(valData, valLabels)
            print("C=",c,"kernel= ",kern,"score=", score)
            if MaxScore < score:
                MaxScore = score
                minC = c
                minKern = kern
    return minC , minKern

def loadlocal_mnist(images_path, labels_path):
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

main()