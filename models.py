import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from kNN import kNN
from kernalised_SVM import SVM
from check_performance_utility import baseline, ConfusionMatrix, ROC

if __name__ == '__main__':
    df = pd.read_csv('dataset.csv', sep=',', header=0)
    X = np.column_stack((df['X1'], df['X3'], df['X6'], df['X8'], df['X9'], df['X10']))
    Y = df['y']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    model_knn = kNN(x_train, y_train)
    modelKernalisedSVM = SVM(x_train, y_train)
    dummyClassifier = baseline(x_train, y_train)
    knnAccuracy, SVMAccuracy, dummyAccuracy = ConfusionMatrix(model_knn, modelKernalisedSVM, dummyClassifier, x_test, y_test)
    ROC(x_test, y_test, model_knn, modelKernalisedSVM, dummyClassifier)