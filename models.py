import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from itertools import combinations

# Comment the line below if you are not using an M1 (ARM-based) machine
matplotlib.use('TkAgg')


# STEP 1: Feature selection

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


# for kNN
###############################
# Selecting range of k values #
###############################
# Conclusion from this function: Best range is between 600 and 1000
def select_k_range(X, Y):
    plt.rcParams["figure.constrained_layout.use"] = True
    mean_error = []
    std_error = []
    k_range = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k, weights="uniform")
        scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.errorbar(k_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("k")
    plt.ylabel("F1 Score")
    plt.title("kNN k vs F1 Score (Selecting k-range for CV)")
    plt.show()


# for kNN
##################################################
# Cross validation on range of k values selected #
##################################################
# Conclusion from this function: Best value for k is 800
def choose_k_using_CV(X, Y):
    mean_error = []
    std_error = []
    k_range = [600, 700, 800, 900, 1000]
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k, weights="uniform")
        scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.errorbar(k_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("k");
    plt.ylabel("F1 Score")
    plt.title("kNN k vs F1 Score (performing CV)")
    plt.show()


def gaussian_kernel10(distances):
    weights = np.exp((-10 * (distances ** 2)))
    return weights / np.sum(weights)


def gaussian_kernel100(distances):
    weights = np.exp((-100 * (distances ** 2)))
    return weights / np.sum(weights)


def gaussian_kernel1000(distances):
    weights = np.exp((-1000 * (distances ** 2)))
    return weights / np.sum(weights)


# for kNN
###################################
# Selecting range of gamma values #
###################################
# Conclusion from this function: The best range to use CV for gamma is between 10 and 50.
def select_kNN_gamma_range_for_CV(X, Y):
    mean_error = []
    std_error = []
    model = KNeighborsClassifier(n_neighbors=800, weights=gaussian_kernel10)
    scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())
    model = KNeighborsClassifier(n_neighbors=800, weights=gaussian_kernel100)
    scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())
    model = KNeighborsClassifier(n_neighbors=800, weights=gaussian_kernel1000)
    scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())

    plt.errorbar([10, 100, 1000], mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("Gamma")
    plt.ylabel("F1 Score")
    plt.title("kNN gamma vs F1 Score (Selecting gamma range for CV)")
    plt.show()


def gaussian_kernel10(distances):
    weights = np.exp((-10 * (distances ** 2)))
    return weights / np.sum(weights)


def gaussian_kernel30(distances):
    weights = np.exp((-30 * (distances ** 2)))
    return weights / np.sum(weights)


def gaussian_kernel50(distances):
    weights = np.exp((-50 * (distances ** 2)))
    return weights / np.sum(weights)

# for kNN
###################################################
# Cross validating range of gamma values selected #
###################################################
# Conclusion from this function: The best value for gamma is 10.
def choose_kNN_gamma_using_CV(X, Y):
    mean_error = []
    std_error = []
    model = KNeighborsClassifier(n_neighbors=800, weights=gaussian_kernel10)
    scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())
    model = KNeighborsClassifier(n_neighbors=800, weights=gaussian_kernel30)
    scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())
    model = KNeighborsClassifier(n_neighbors=800, weights=gaussian_kernel50)
    scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())

    plt.errorbar([10, 30, 50], mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("Gamma")
    plt.ylabel("F1 Score")
    plt.title("kNN gamma vs F1 Score (performing CV)")
    plt.show()


#########################################################################
# kNN classifier with hyper-parameters k=800 & gamma=10 selected via CV #
#########################################################################
# N.B USING WEIGHTS="DISTANCE" MAKES THE KNN MODEL BE 0.74 ACCURATE
def kNN(x_train, y_train):
    model_knn = KNeighborsClassifier(n_neighbors=800, weights=gaussian_kernel10).fit(x_train, y_train)
    return model_knn

###################################################################################################################
# for SVM
##################################################
# Selecting range of C values for kernalised SVM #
##################################################
# Conclusion from this function: The best range for C is between 0.01 and 10.
def select_C_range(X, Y, x_train, y_train):
    warnings.filterwarnings('ignore', 'Solver terminated early.*')
    mean_error = []
    std_error = []
    c_range = [0.01, 0.1, 1, 2, 10, 100, 200]
    for c in c_range:
        model = SVC(C=c, kernel='rbf', gamma=5, max_iter=1000).fit(x_train, y_train)
        scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.errorbar(c_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("c")
    plt.ylabel("F1 Score")
    plt.title("kernalised SVM Classifier c vs F1 Score (Selecting c-range for CV)")
    plt.show()

# for SVM
#####################################################################
# Cross validation on range of C values selected for kernalised SVM #
#####################################################################
# Conclusion from this function: The best value for C is 1.
# RBF IS DEFAULT, SO WHY WE USING IT?
def choose_C_using_CV(X, Y, x_train, y_train):
    warnings.filterwarnings('ignore', 'Solver terminated early.*')
    mean_error = []
    std_error = []
    c_range = [0.01, 0.05, 0.1, 0.5, 1, 2, 10]
    for c in c_range:
        model = SVC(C=c, kernel='rbf', gamma=5, max_iter=1000).fit(x_train, y_train)
        scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.errorbar(c_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("c")
    plt.ylabel("F1 Score")
    plt.title("kernalised SVM Classifier c vs F1 Score (Performing CV)")
    plt.show()

# for SVM
######################################################
# Selecting range of gamma values for kernalised SVM #
######################################################
# Conclusion from this function: The best range for gamma is between 3000 and 4000.
def select_SVM_gamma_range_for_CV(X, Y, x_train, y_train):
    warnings.filterwarnings('ignore', 'Solver terminated early.*')
    mean_error = []
    std_error = []
    g_range = [1, 5, 10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000]
    for g in g_range:
        model = SVC(kernel='rbf', gamma=g, max_iter=1000).fit(x_train, y_train)
        scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.errorbar(g_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("g")
    plt.ylabel("F1 Score")
    plt.title("kernalised SVM Classifier Gamma vs F1 Score (Selecting g-range for CV)")
    plt.show()

# for SVM
######################################################
# Selecting range of gamma values for kernalised SVM #
######################################################
# Conclusion from this function: The best value for gamma is 3250.
def choose_SVM_gamma_using_CV(X, Y, x_train, y_train):
    warnings.filterwarnings('ignore', 'Solver terminated early.*')
    mean_error = []
    std_error = []
    g_range = [3000, 3250, 3500, 3750, 4000]
    for g in g_range:
        model = SVC(kernel='rbf', gamma=g, max_iter=1000).fit(x_train, y_train)
        scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.errorbar(g_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("g")
    plt.ylabel("F1 Score")
    plt.title("kernalised SVM Classifier Gamma vs F1 Score (Selecting g-range for CV)")
    plt.show()


######################################################################################
# Kernalised SVM classifier with hyper-parameters C=1 and gamma=3250 selected via CV #
######################################################################################
# RBF = gaussian kernel
def SVM(x_train, y_train):
    modelKernalisedSVM = SVC(C=1, kernel='rbf', gamma=3250, probability=True).fit(x_train, y_train)
    return modelKernalisedSVM


def baseline(x_train, y_train):
    dummy = DummyClassifier(strategy="uniform").fit(x_train, y_train)
    return dummy


#################################################################################
# Confusion Matrix for kNN classifier where k=1 (selected via cross validation) #
#################################################################################
def ConfusionMatrix(model_knn, modelKernalisedSVM, dummyClassifier, x_train, y_train, x_test, y_test):
    y_pred = model_knn.predict(x_test)
    print("Confusion Matrix kNN Classifier:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy: %.2f" % (accuracy_score(y_test, y_pred)))
    knnAccuracy = accuracy_score(y_test, y_pred)
    ###########################################################################################################
    # Confusion matrix for kernalised SVM classifier where C=0.1 and gamma=25 (selected via cross validation) #
    ###########################################################################################################
    y_pred = modelKernalisedSVM.predict(x_test)
    print("Confusion Matrix kernalised SVM Classifier:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy: %.2f" % (accuracy_score(y_test, y_pred)))
    SVMAccuracy = accuracy_score(y_test, y_pred)

    #########################################
    # Confusion matrix for dummy classifier #
    #########################################
    ydummy = dummyClassifier.predict(x_test)
    print("Dummy Confusion Matrix:\n", confusion_matrix(y_test, ydummy))
    print("Accuracy: %.2f" % (accuracy_score(y_test, ydummy)))
    dummyAccuracy = accuracy_score(y_test, ydummy)
    return knnAccuracy, SVMAccuracy, dummyAccuracy


################################################################################
# ROC curve for kNN classifier, kernalised SVM classifier and dummy classifier #
################################################################################
def ROC(model_knn, modelKernalisedSVM, dummy):
    knn = "kNN Classifier ROC"
    kSVM = "kernalised SVM Classifier ROC"
    dumb = "Dummy Classifier ROC"
    for model, title in zip([model_knn, modelKernalisedSVM, dummy], [knn, kSVM, dumb]):
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label="AUC = %f" % (roc_auc))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.title(title)
        plt.plot([0, 1], [0, 1], color="green", linestyle="--")
        plt.show()


if __name__ == '__main__':
    # UNCOMMENT TO SEE DATA PLOT

    input = input('Do you wish to redo feature selection? (Y/N) [this may take a while]: ')
    if input == "Y" or "y":
        # Conclusion from feature selection: use features [X1, X3, X6, X8, X9, X10]
        select_features_with_dependency()
        select_features_with_best_accuracy()

    # Train a kernalised SVM, kNN, and dummy classifier using the features from feature selection.
    df = pd.read_csv('dataset.csv', sep=',', header=0)
    X = np.column_stack((df['X1'], df['X3'], df['X6'], df['X8'], df['X9'], df['X10']))
    Y = df['y']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    model_knn = kNN(x_train, y_train)
    modelKernalisedSVM = SVM(x_train, y_train)
    dummyClassifier = baseline(x_train, y_train)
    knnAccuracy, SVMAccuracy, dummyAccuracy = ConfusionMatrix(model_knn, modelKernalisedSVM, dummyClassifier, x_train,
                                                              y_train, x_test, y_test)

    terminateProgram = False

    while not terminateProgram:
        print("You can run the following experiments:")
        print("")
        input = input("")

    # EXPERIMENTS (UNCOMMENT TO RUN):
    # ROC(model_knn, modelKernalisedSVM, dummyClassifier)
    # select_k_range(X, Y)
    # choose_k_using_CV(X, Y)
    # select_kNN_gamma_range_for_CV(X, Y)
    # choose_kNN_gamma_using_CV(X,Y)
    # select_C_range(X, Y, x_train, y_train)
    # choose_C_using_CV(X, Y, x_train, y_train)
    # select_SVM_gamma_range_for_CV(X, Y, x_train, y_train)
    # choose_SVM_gamma_using_CV(X, Y, x_train, y_train)
