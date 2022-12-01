import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from gaussian_kernel_utility import *

# Comment the line below if you are not using an M1 (ARM-based) machine
matplotlib.use('TkAgg')


# Step 1: Selecting k

# Selecting range of k values for the kNN classifier
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


# Cross validation on range of k values selected previously (600 to 1000)
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


# Step 2: Selecting gamma (for weights)

# Selecting range of gamma values for the kNN classifier
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


# Cross validation on range of gamma values selected previously (10 to 50)
# Conclusion from this function: The best value for gamma is 30.
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


# kNN classifier with chosen hyperparameters k=800 & gamma=30 selected via cross-validation
def kNN(x_train, y_train):
    model_knn = KNeighborsClassifier(n_neighbors=800, weights=gaussian_kernel30).fit(x_train, y_train)
    return model_knn


if __name__ == '__main__':
    df = pd.read_csv('../data/dataset.csv', sep=',', header=0)
    X = np.column_stack((df['X1'], df['X3'], df['X6'], df['X8'], df['X9'], df['X10']))
    Y = df['y']
    select_k_range(X, Y)
    choose_k_using_CV(X, Y)
    select_kNN_gamma_range_for_CV(X, Y)
    choose_kNN_gamma_using_CV(X, Y)
