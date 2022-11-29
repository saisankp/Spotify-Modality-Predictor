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

# uncomment if using m1 machine
matplotlib.use('TkAgg')


###############################
#        Plot of data         #
###############################
def plot():
    df = pd.read_csv('dataset.csv', sep=',', header=0)

    fig, ax = plt.subplots(2, 2)
    ax[0][0].scatter(df['X1'], df['y'])
    ax[0][0].set_title("X1 vs Y")
    ax[0][1].scatter(df['X2'], df['y'])
    ax[0][1].set_title("X2 vs Y")
    ax[1][0].scatter(df['X3'], df['y'])
    ax[1][0].set_title("X3 vs Y")
    ax[1][1].scatter(df['X4'], df['y'])
    ax[1][1].set_title("X4 vs Y")
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 3)
    ax[0].scatter(df['X5'], df['y'])
    ax[0].set_title("X5 vs Y")
    ax[1].scatter(df['X7'], df['y'])
    ax[1].set_title("X7 vs Y")
    ax[2].scatter(df['X8'], df['y'])
    ax[2].set_title("X8 vs Y")
    fig.tight_layout()
    plt.show()


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
# DO NOT USE PROBABILITY=TRUE ON SVM
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


def select_features_with_best_accuracy():
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(96, 64))
    ax[0].set_title("Kernalized SVM Accuracy vs feature combination index", fontsize=17)
    ax[0].set_xlabel("Index of feature combination", fontsize=13)
    ax[0].set_ylabel("Kernalized SVM Accuracy", fontsize=13)

    ax[1].set_title("kNN Accuracy vs feature combination index", fontsize=17)
    ax[1].set_xlabel("Index of feature combination", fontsize=13)
    ax[1].set_ylabel("kNN Accuracy", fontsize=13)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    df = pd.read_csv('dataset.csv', sep=',', header=0)
    features = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12']
    tmp = []
    for i in range(len(features)):
        oc = combinations(features, i + 1)
        for c in oc:
            tmp.append(list(c))
    overallListOfTupleCombinations = []
    for combination in range(len(tmp)):
        tupleCombination = ()
        for oneFeature in range(len(tmp[combination])):
            tupleCombination += (df[tmp[combination][oneFeature]],)
        overallListOfTupleCombinations.append(tupleCombination)
    maxKNNAccuracy = 0
    maxKNNAccuracyIndex = 0
    maxSVMAccuracy = 0
    maxSVMAccuracyIndex = 0
    maxCombinedAccuracy = 0
    maxCombinedAccuracyIndex = 0
    for oneCombination in range(len(overallListOfTupleCombinations)):
        print("USING COMBINATION INDEX: " + str(oneCombination))
        if len(overallListOfTupleCombinations[oneCombination]) == 1:
            X = np.array(np.column_stack(overallListOfTupleCombinations[oneCombination])).reshape(-1, 1)
        else:
            X = np.column_stack(overallListOfTupleCombinations[oneCombination])

        Y = df['y']
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        model_knn = kNN(x_train, y_train)
        modelKernalisedSVM = SVM(x_train, y_train)
        dummyClassifier = baseline(x_train, y_train)
        knnAccuracy, SVMAccuracy, dummyAccuracy = ConfusionMatrix(model_knn, modelKernalisedSVM, dummyClassifier, x_train, y_train, x_test, y_test)
        combinedAccuracy = knnAccuracy + SVMAccuracy
        print(combinedAccuracy)
        print(knnAccuracy)
        print(SVMAccuracy)
        if combinedAccuracy > maxCombinedAccuracy:
            print("NEW COMBINED ACCURACY FOUND: " + str(combinedAccuracy))
            maxCombinedAccuracy = combinedAccuracy
            maxCombinedAccuracyIndex = oneCombination

        if knnAccuracy > maxKNNAccuracy:
            print("NEW KNN ACCURACY FOUND: " + str(knnAccuracy))
            maxKNNAccuracy = knnAccuracy
            maxKNNAccuracyIndex = oneCombination

        if SVMAccuracy > maxSVMAccuracy:
            print("NEW SVM ACCURACY FOUND: " + str(SVMAccuracy))
            maxSVMAccuracy = SVMAccuracy
            maxSVMAccuracyIndex = oneCombination
        ax[0].plot(oneCombination, SVMAccuracy, c="red", marker="x")
        ax[1].plot(oneCombination, knnAccuracy, c="green", marker="x")

    fig.show()
    fig.waitforbuttonpress()

    print("THE MAXIMUM KNN ACCURACY OF " + str(maxKNNAccuracy) + " HAPPENED AT INDEX " + str(
        maxKNNAccuracyIndex))
    print("THE MAXIMUM SVM ACCURACY OF " + str(maxSVMAccuracy) + " HAPPENED AT INDEX " + str(
        maxSVMAccuracyIndex))
    print("THE MAXIMUM COMBINED ACCURACY OF " + str(maxCombinedAccuracy) + " HAPPENED AT INDEX " + str(
        maxCombinedAccuracyIndex))
    return overallListOfTupleCombinations, maxKNNAccuracy, maxKNNAccuracyIndex, maxSVMAccuracy, maxSVMAccuracyIndex, maxCombinedAccuracy, maxCombinedAccuracyIndex


if __name__ == '__main__':
    # UNCOMMENT TO SEE DATA PLOT
    #plot()
    df = pd.read_csv('dataset.csv', sep=',', header=0)
    overallListOfTupleCombinations, maxKNNAccuracy, maxKNNAccuracyIndex, maxSVMAccuracy, maxSVMAccuracyIndex, maxCombinedAccuracy, maxCombinedAccuracyIndex = select_features_with_best_accuracy()
    # RESULTS:
    # THE MAXIMUM KNN ACCURACY OF 0.7508771929824561 HAPPENED AT INDEX 2515 [Hence features X1, X2, X3, X4, X5, X7, X8]
    # THE MAXIMUM SVM ACCURACY OF 0.7947368421052632 HAPPENED AT INDEX 832 [Hence features X1, X2, X4, X5, X9]
    # THE MAXIMUM COMBINED ACCURACY OF 1.5447368421052632 HAPPENED AT INDEX 2515 [Hence features X1, X2, X3, X4, X5, X7, X8]

    # THE MAXIMUM KNN ACCURACY OF 0.7508771929824561 HAPPENED AT INDEX 3072
    # THE MAXIMUM SVM ACCURACY OF 0.7859649122807018 HAPPENED AT INDEX 1686
    # THE MAXIMUM COMBINED ACCURACY OF 1.5140350877192983 HAPPENED AT INDEX 3852

    # THE MAXIMUM KNN ACCURACY OF 0.7508771929824561 HAPPENED AT INDEX 2877
    # THE MAXIMUM SVM ACCURACY OF 0.7921052631578948 HAPPENED AT INDEX 2751
    # THE MAXIMUM COMBINED ACCURACY OF 1.5280701754385966 HAPPENED AT INDEX 2877

    # X = np.column_stack((df['X1'], df['X2'], df['X3'], df['X4'], df['X5'], df['X7'], df['X8'], df['X12']))
    # Y = df['y']
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    # model_knn = kNN(x_train, y_train)
    # modelKernalisedSVM = SVM(x_train, y_train)
    # dummyClassifier = baseline(x_train, y_train)
    # knnAccuracy, SVMAccuracy, dummyAccuracy = ConfusionMatrix(model_knn, modelKernalisedSVM, dummyClassifier, x_train, y_train, x_test, y_test)

    # EXPERIMENTS (UNCOMMENT TO RUN):
    #ROC(model_knn, modelKernalisedSVM, dummyClassifier)
    #select_k_range(X, Y)
    #choose_k_using_CV(X, Y)
    #select_kNN_gamma_range_for_CV(X, Y)
    #choose_kNN_gamma_using_CV(X,Y)
    #select_C_range(X, Y, x_train, y_train)
    #choose_C_using_CV(X, Y, x_train, y_train)
    #select_SVM_gamma_range_for_CV(X, Y, x_train, y_train)
    #choose_SVM_gamma_using_CV(X, Y, x_train, y_train)
