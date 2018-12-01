from __future__ import print_function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
import numpy as np
import cv2
import sklearn
from sklearn.model_selection import train_test_split
import struct
import sys


def loadlocal_mnist(images_path, labels_path):
    """ Read MNIST from ubyte files.
    Parameters
    ----------
    images_path : str
        path to the test or train MNIST ubyte file
    labels_path : str
        path to the test or train MNIST class labels file
    Returns
    --------
    images : [n_samples, n_pixels] numpy.array
        Pixel values of the images.
    labels : [n_samples] numpy array
        Target class labels
    Examples
    """
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







def main():
    fast = False
    if(len(sys.argv) > 1 and sys.argv[1]=='fast'):
        print ("running fast version")
        fast = True

    trainData, trainLabels = loadlocal_mnist(images_path='train-images.idx3-ubyte',labels_path='train-labels.idx1-ubyte')
    testData, testLabels = loadlocal_mnist(images_path='t10k-images.idx3-ubyte',labels_path='t10k-labels.idx1-ubyte')

    if fast:
        # to make it fast we take only the 10% for the training
        (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,test_size=0.9)


    # we take 10% of the training data and use that for validation
    (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,test_size=0.1)

    BestK = FindBestK(trainData, trainLabels, valData, valLabels)

    print("best K is %d" %( BestK))

    # re-train our classifier using the best k value and predict the labels of the
    # test data
    model = KNeighborsClassifier(n_neighbors=BestK)
    model.fit(trainData, trainLabels)
    predictions = model.predict(testData)

    # show a final classification report demonstrating the accuracy of the classifier
    # for each of the digits
    print("EVALUATION ON TESTING DATA")
    print(classification_report(testLabels, predictions))

def FindBestK(trainData, trainLabels, valData, valLabels):
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

main()