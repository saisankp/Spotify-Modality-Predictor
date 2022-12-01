import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from itertools import combinations

# Comment the line below if you are not using an M1 (ARM-based) machine
matplotlib.use('TkAgg')

# Feature selection #1 (plotting features vs target value to see their dependence)
# Conclusion from this function: Features X1, X2, X3, X4, X6, X8, X9 and X10 are dependent features
def select_features_with_dependency():
    # Use 20 songs (with 10 major songs, and 10 minor songs) from the first 50 songs in dataset.csv
    df = pd.read_csv('data_for_dependency_graphs.csv', sep=',', header=0)
    # Setup plots
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 20
    fig, ax = plt.subplots(2, 2)

    # Plot feature X1 vs target value (modality)
    ax[0][0].scatter(df['X1'], df['y'])
    ax[0][0].set_title("X1 (Danceability) vs Y (Target value)")
    ax[0][0].set_xlabel("X1")
    ax[0][0].set_ylabel("y")

    # Plot feature X2 vs target value (modality)
    ax[0][1].scatter(df['X2'], df['y'])
    ax[0][1].set_title("X2 (Energy) vs Y (Target value)")
    ax[0][1].set_xlabel("X2")
    ax[0][1].set_ylabel("y")

    # Plot feature X3 vs target value (modality)
    ax[1][0].scatter(df['X3'], df['y'])
    ax[1][0].set_title("X3 (Key) vs Y (Target value)")
    ax[1][0].set_xlabel("X3")
    ax[1][0].set_ylabel("y")

    # Plot feature X4 vs target value (modality)
    ax[1][1].scatter(df['X4'], df['y'])
    ax[1][1].set_title("X4 (Loudness) vs Y (Target value)")
    ax[1][1].set_xlabel("X4")
    ax[1][1].set_ylabel("y")
    fig.tight_layout()
    fig.set_figheight(50)
    fig.set_figwidth(50)
    # Show subplots with X1, X2, X3, and X4
    plt.show()

    fig, ax = plt.subplots(2, 2)
    # Plot feature X5 vs target value (modality)
    ax[0][0].scatter(df['X5'], df['y'])
    ax[0][0].set_title("X5 (Speechiness) vs Y (Target value)")
    ax[0][0].set_xlabel("X5")
    ax[0][0].set_ylabel("y")

    # Plot feature X6 vs target value (modality)
    ax[0][1].scatter(df['X6'], df['y'])
    ax[0][1].set_title("X6 (Acousticness) vs Y (Target value)")
    ax[0][1].set_xlabel("X6")
    ax[0][1].set_ylabel("y")

    # Plot feature X7 vs target value (modality)
    ax[1][0].scatter(df['X7'], df['y'])
    ax[1][0].set_title("X7 (Instrumentalness) vs Y (Target value)")
    ax[1][0].set_xlabel("X7")
    ax[1][0].set_ylabel("y")

    # Plot feature X8 vs target value (modality)
    ax[1][1].scatter(df['X8'], df['y'])
    ax[1][1].set_title("X8 (Liveness) vs Y (Target value)")
    ax[1][1].set_xlabel("X8")
    ax[1][1].set_ylabel("y")
    fig.tight_layout()
    fig.set_figheight(50)
    fig.set_figwidth(50)
    # Show subplots with X5, X6, X7, and X8
    plt.show()

    fig, ax = plt.subplots(2, 2)
    # Plot feature X9 vs target value (modality)
    ax[0][0].scatter(df['X9'], df['y'])
    ax[0][0].set_title("X9 (Tempo) vs Y (Target value)")
    ax[0][0].set_xlabel("X9")
    ax[0][0].set_ylabel("y")

    # Plot feature X10 vs target value (modality)
    ax[0][1].scatter(df['X10'], df['y'])
    ax[0][1].set_title("X10 (Valence) vs Y (Target value)")
    ax[0][1].set_xlabel("X10")
    ax[0][1].set_ylabel("y")

    # Plot feature X11 vs target value (modality)
    ax[1][0].scatter(df['X11'], df['y'])
    ax[1][0].set_title("X11 (Duration_ms) vs Y (Target value)")
    ax[1][0].set_xlabel("X11")
    ax[1][0].set_ylabel("y")

    # Plot feature X12 vs target value (modality)
    ax[1][1].scatter(df['X12'], df['y'])
    ax[1][1].set_title("X12 (Time_signature) vs Y (Target value)")
    ax[1][1].set_xlabel("X12")
    ax[1][1].set_ylabel("y")
    fig.tight_layout()
    fig.set_figheight(50)
    fig.set_figwidth(50)
    # Show subplots with X9, X10, X11, and X12
    plt.show()

# Feature selection #2 (brute forcing every combination of feature combinations to get the highest accuracy)
# Conclusion from this function (resulting plot in report):
# 1. The maximum kNN accuracy of 0.7508771929824561 happened at index 2877 [X1, X3, X6, X8, X9, X10, X12]
# 1. The maximum kernalised SVM accuracy of 0.7947368421052632 happened at index 2751 [X1, X2, X6, X8, X9, X10, X12]
def select_features_with_best_accuracy():
    # Setup plots
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(96, 64))
    ax[0].set_title("Kernalized SVM Accuracy vs feature combination index", fontsize=17)
    ax[0].set_xlabel("Index of feature combination", fontsize=13)
    ax[0].set_ylabel("Kernalized SVM Accuracy", fontsize=13)
    ax[1].set_title("kNN Accuracy vs feature combination index", fontsize=17)
    ax[1].set_xlabel("Index of feature combination", fontsize=13)
    ax[1].set_ylabel("kNN Accuracy", fontsize=13)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

    # Get every combination of 12 features (X1 -> X12)
    df = pd.read_csv('dataset.csv', sep=',', header=0)
    features = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12']
    tmp = []
    for i in range(len(features)):
        oc = combinations(features, i + 1)
        for c in oc:
            tmp.append(list(c))
    overallListOfTupleCombinationsWithNoDataFrame = []
    overallListOfTupleCombinationsWithDataFrame = []
    for combination in range(len(tmp)):
        tupleCombination = ()
        tupleNames = ()
        for oneFeature in range(len(tmp[combination])):
            tupleCombination += (df[tmp[combination][oneFeature]],)
            tupleNames += (tmp[combination][oneFeature],)
        overallListOfTupleCombinationsWithDataFrame.append(tupleCombination)
        overallListOfTupleCombinationsWithNoDataFrame.append(tupleNames)

    print(len(overallListOfTupleCombinationsWithDataFrame))
    # With all combinations, train kernalised SVM and kNN classifiers to get index with the best accuracy
    maxKNNAccuracy = 0
    maxKNNAccuracyIndex = 0
    maxSVMAccuracy = 0
    maxSVMAccuracyIndex = 0
    maxCombinedAccuracy = 0
    maxCombinedAccuracyIndex = 0
    for oneCombination in range(len(overallListOfTupleCombinationsWithDataFrame)):
        print("Using combination index : " + str(oneCombination) + " [features : " + str(
            overallListOfTupleCombinationsWithNoDataFrame[oneCombination]) + "]")
        if len(overallListOfTupleCombinationsWithDataFrame[oneCombination]) == 1:
            X = np.array(np.column_stack(overallListOfTupleCombinationsWithDataFrame[oneCombination])).reshape(-1, 1)
        else:
            X = np.column_stack(overallListOfTupleCombinationsWithDataFrame[oneCombination])

        # Calculate the confusion matrix and accuracy from this particular combination of features.
        Y = df['y']
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        model_knn = kNN(x_train, y_train)
        modelKernalisedSVM = SVM(x_train, y_train)
        dummyClassifier = baseline(x_train, y_train)
        knnAccuracy, SVMAccuracy, dummyAccuracy = ConfusionMatrix(model_knn, modelKernalisedSVM, dummyClassifier,
                                                                  x_train, y_train, x_test, y_test)
        combinedAccuracy = knnAccuracy + SVMAccuracy
        if combinedAccuracy > maxCombinedAccuracy:
            print("A new maximum combined accuracy was found: " + str(combinedAccuracy))
            maxCombinedAccuracy = combinedAccuracy
            maxCombinedAccuracyIndex = oneCombination

        if knnAccuracy > maxKNNAccuracy:
            print("A new maximum kNN accuracy was found: " + str(knnAccuracy))
            maxKNNAccuracy = knnAccuracy
            maxKNNAccuracyIndex = oneCombination

        if SVMAccuracy > maxSVMAccuracy:
            print("A new maximum kernalised SVM accuracy was found: " + str(SVMAccuracy))
            maxSVMAccuracy = SVMAccuracy
            maxSVMAccuracyIndex = oneCombination
        ax[0].plot(oneCombination, SVMAccuracy, c="red", marker="x")
        ax[1].plot(oneCombination, knnAccuracy, c="green", marker="x")

    # Show 2 plots (one for kNN, another for kernalised SVM) with 5700 points each representing how the accuracy
    # changes as new features are introduced to each classifier.
    fig.show()
    fig.waitforbuttonpress()

    print("The overall maximum kNN accuracy of " + str(maxKNNAccuracy) + " occurred at index " + str(
        maxKNNAccuracyIndex))
    print("The overall maximum kernalised SVM accuracy of " + str(maxSVMAccuracy) + " occurred at index " + str(
        maxSVMAccuracyIndex))
    print("The maximum combined accuracy of " + str(maxCombinedAccuracy) + " occurred at index " + str(
        maxCombinedAccuracyIndex))


if __name__ == '__main__':
    select_features_with_dependency()
    select_features_with_best_accuracy()