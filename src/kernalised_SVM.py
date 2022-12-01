import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split

# Comment the line below if you are not using an M1 (ARM-based) machine
matplotlib.use('TkAgg')


# Step 1: Selecting C

# Selecting range of C values for the kernalised SVM
# Conclusion from this function: The best range for C is between 0.01 and 2
def select_C_range(X, Y, x_train, y_train):
    mean_error = []
    std_error = []
    c_range = [0.01, 0.1, 1, 2, 10]
    for c in c_range:
        model = SVC(C=c, kernel='rbf', gamma=5, max_iter=10000).fit(x_train, y_train)
        scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.errorbar(c_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("C")
    plt.ylabel("F1 Score")
    plt.title("Kernalised SVM Classifier c vs F1 Score (Selecting c-range for CV)")
    plt.show()


# Cross validation on range of C values selected previously (0.01 and 2)
# Conclusion from this function: The best value for C is 0.6.
def choose_C_using_CV(X, Y, x_train, y_train):
    mean_error = []
    std_error = []
    c_range = [0.01, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]
    for c in c_range:
        model = SVC(C=c, kernel='rbf', gamma=5, max_iter=10000).fit(x_train, y_train)
        scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.errorbar(c_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("C")
    plt.ylabel("F1 Score")
    plt.title("Kernalised SVM Classifier c vs F1 Score (Performing CV)")
    plt.show()


# Step 2: Selecting gamma (for kernel)

# Selecting range of gamma values for the kernalised SVM
# Conclusion from this function: The best range for gamma is between 1 and 100.
def select_SVM_gamma_range_for_CV(X, Y, x_train, y_train):
    mean_error = []
    std_error = []
    g_range = [1, 5, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    for g in g_range:
        model = SVC(C=0.6, kernel='rbf', gamma=g, max_iter=10000).fit(x_train, y_train)
        scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.errorbar(g_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("Gamma")
    plt.ylabel("F1 Score")
    plt.title("kernalised SVM Classifier gamma vs F1 Score (Selecting g-range for CV)")
    plt.show()


# Selecting range of gamma values for the kernalised SVM
# Conclusion from this function: The best value for gamma is 100.
def choose_SVM_gamma_using_CV(X, Y, x_train, y_train):
    mean_error = []
    std_error = []
    g_range = [1, 25, 50, 75, 100]
    for g in g_range:
        model = SVC(C=0.6, kernel='rbf', gamma=g, max_iter=10000).fit(x_train, y_train)
        scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.errorbar(g_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("Gamma")
    plt.ylabel("F1 Score")
    plt.title("kernalised SVM Classifier gamma vs F1 Score (Selecting g-range for CV)")
    plt.show()


# Kernalised SVM classifier with chosen hyperparameters C=0.6 & gamma=100 selected via cross-validation
def SVM(x_train, y_train):
    modelKernalisedSVM = SVC(C=0.6, kernel='rbf', gamma=100, probability=True).fit(x_train, y_train)
    return modelKernalisedSVM


if __name__ == '__main__':
    df = pd.read_csv('../data/dataset.csv', sep=',', header=0)
    X = np.column_stack((df['X1'], df['X3'], df['X6'], df['X8'], df['X9'], df['X10']))
    Y = df['y']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    #select_C_range(X, Y, x_train, y_train)
    #choose_C_using_CV(X, Y, x_train, y_train)
    select_SVM_gamma_range_for_CV(X, Y, x_train, y_train)
    choose_SVM_gamma_using_CV(X, Y, x_train, y_train)
