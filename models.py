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

matplotlib.use('TkAgg')

#############################################
# Read data and initialise global variables #
#############################################

df = pd.read_csv('dataset.csv', sep=',', header=0)
X = np.column_stack((df['X1'], df['X2'], df['X3']))
Y = df['y']

###############################
#        Plot of data         #
###############################
fig, ax = plt.subplots(1, 3)
ax[0].scatter(df['X1'], df['y']);
ax[0].set_title("Dancibility vs Y")
ax[1].scatter(df['X2'], df['y']);
ax[1].set_title("Energy vs Y")
ax[2].scatter(df['X3'], df['y']);
ax[2].set_title("Valence vs Y")
fig.tight_layout()
plt.show()

###############################
# Selecting range of k values #
###############################
mean_error = []
std_error = []
k_range = [1, 5, 11, 51, 101]
for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k, weights="uniform")
    scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())

plt.errorbar(k_range, mean_error, yerr=std_error, linewidth=3)
plt.xlabel("k");
plt.ylabel("F1 Score")
plt.title("kNN k vs F1 Score (Selecting k-range for CV)")
plt.show()

##################################################
# Cross validation on range of k values selected #
##################################################
mean_error = []
std_error = []
k_range = [41, 43, 45, 47, 49, 51]
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


###################################
# Selecting range of gamma values #
###################################
def gaussian_kernel10(distances):
    weights = np.exp((-10 * (distances ** 2)))
    return weights / np.sum(weights)


def gaussian_kernel100(distances):
    weights = np.exp((-100 * (distances ** 2)))
    return weights / np.sum(weights)


def gaussian_kernel1000(distances):
    weights = np.exp((-1000 * (distances ** 2)))
    return weights / np.sum(weights)


mean_error = []
std_error = []
model = KNeighborsClassifier(n_neighbors=41, weights=gaussian_kernel10)
scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
mean_error.append(np.array(scores).mean())
std_error.append(np.array(scores).std())
model = KNeighborsClassifier(n_neighbors=41, weights=gaussian_kernel100)
scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
mean_error.append(np.array(scores).mean())
std_error.append(np.array(scores).std())
model = KNeighborsClassifier(n_neighbors=41, weights=gaussian_kernel1000)
scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
mean_error.append(np.array(scores).mean())
std_error.append(np.array(scores).std())

plt.errorbar([10, 100, 1000], mean_error, yerr=std_error, linewidth=3)
plt.xlabel("Gamma");
plt.ylabel("F1 Score")
plt.title("kNN gamma vs F1 Score (performing CV)")
plt.show()


###################################################
# Cross validating range of gamma values selected #
###################################################
def gaussian_kernel10(distances):
    weights = np.exp((-10 * (distances ** 2)))
    return weights / np.sum(weights)


def gaussian_kernel25(distances):
    weights = np.exp((-25 * (distances ** 2)))
    return weights / np.sum(weights)


def gaussian_kernel50(distances):
    weights = np.exp((-50 * (distances ** 2)))
    return weights / np.sum(weights)


mean_error = []
std_error = []
model = KNeighborsClassifier(n_neighbors=41, weights=gaussian_kernel10)
scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
mean_error.append(np.array(scores).mean())
std_error.append(np.array(scores).std())
model = KNeighborsClassifier(n_neighbors=41, weights=gaussian_kernel25)
scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
mean_error.append(np.array(scores).mean())
std_error.append(np.array(scores).std())
model = KNeighborsClassifier(n_neighbors=41, weights=gaussian_kernel50)
scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
mean_error.append(np.array(scores).mean())
std_error.append(np.array(scores).std())

plt.errorbar([10, 25, 50], mean_error, yerr=std_error, linewidth=3)
plt.xlabel("Gamma");
plt.ylabel("F1 Score")
plt.title("kNN gamma vs F1 Score (performing CV)")
plt.show()

#########################################################################
# kNN classifier with hyper-parameters k=41 & gamma=10 selected via CV  #
#########################################################################
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
model_knn = KNeighborsClassifier(n_neighbors=41, weights=gaussian_kernel10).fit(x_train, y_train)
y_pred = model_knn.predict(x_test)
zipped = zip(x_test[:, 0], y_pred)
zipped = sorted(zipped, key=lambda x: x[0])
x_test_sorted = [i for i, _ in zipped]
y_pred_sorted = [j for _, j in zipped]

fig, ax = plt.subplots(1, 3)
ax[0].scatter(x_train[:, 0], y_train, color="red");
ax[0].set_title("Dancibility vs Y")
ax[0].plot(x_test_sorted, y_pred_sorted, color="green")
ax[1].scatter(x_train[:, 1], y_train, color="red");
ax[1].set_title("Energy vs Y")
ax[1].plot(x_test_sorted, y_pred_sorted, color="green")
ax[2].scatter(x_train[:, 2], y_train, color="red");
ax[2].set_title("Valence vs Y")
ax[2].plot(x_test_sorted, y_pred_sorted, color="green")
fig.tight_layout()
fig.suptitle("kNN Classifier k=41")
fig.legend(['train', 'predict'])
plt.show()

##################################################
# Selecting range of C values for kernalised SVM #
##################################################
mean_error = []
std_error = []
c_range = [0.01, 0.1, 1, 10, 100, ]
for c in c_range:
    model = SVC(C=c, kernel='rbf', gamma=50).fit(x_train, y_train)
    scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())
plt.errorbar(c_range, mean_error, yerr=std_error, linewidth=3)
plt.xlabel("c");
plt.ylabel("F1 Score")
plt.title("kernalised SVM Classifier c vs F1 Score (Selecting c-range for CV)")
plt.show()

#####################################################################
# Cross validation on range of C values selected for kernalised SVM #
#####################################################################
mean_error = []
std_error = []
c_range = [0.01, 0.05, 0.1, 0.5, 1]
for c in c_range:
    model = SVC(C=c, kernel='rbf', gamma=50).fit(x_train, y_train)
    scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())
plt.errorbar(c_range, mean_error, yerr=std_error, linewidth=3)
plt.xlabel("c");
plt.ylabel("F1 Score")
plt.title("kernalised SVM Classifier c vs F1 Score (Performing CV)")
plt.show()

######################################################
# Selecting range of gamma values for kernalised SVM #
######################################################
mean_error = []
std_error = []
g_range = [1, 5, 10, 50, 100, 500, 1000]
for g in g_range:
    model = SVC(C=0.1, kernel='rbf', gamma=g).fit(x_train, y_train)
    scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())
plt.errorbar(g_range, mean_error, yerr=std_error, linewidth=3)
plt.xlabel("g");
plt.ylabel("F1 Score")
plt.title("kernalised SVM Classifier Gamma vs F1 Score (Selecting g-range for CV)")
plt.show()

######################################################
# Selecting range of gamma values for kernalised SVM #
######################################################
mean_error = []
std_error = []
g_range = [1, 5, 10, 25, 50]
for g in g_range:
    model = SVC(C=0.1, kernel='rbf', gamma=g).fit(x_train, y_train)
    scores = cross_val_score(model, X, Y, cv=5, scoring="f1")
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())
plt.errorbar(g_range, mean_error, yerr=std_error, linewidth=3)
plt.xlabel("g");
plt.ylabel("F1 Score")
plt.title("kernalised SVM Classifier Gamma vs F1 Score (Selecting g-range for CV)")
plt.show()

######################################################################################
# Kernalised SVM classifier with hyper-parameters C=0.1 and gamma=25 selected via CV #
######################################################################################
modelKernalisedSVM = SVC(C=0.1, kernel='rbf', gamma=25, probability=True).fit(x_train, y_train)
y_pred = modelKernalisedSVM.predict(x_test)
zipped = zip(x_test[:, 0], y_pred)
zipped = sorted(zipped, key=lambda x: x[0])
x_test_sorted = [i for i, _ in zipped]
y_pred_sorted = [j for _, j in zipped]

fig, ax = plt.subplots(1, 3)
ax[0].scatter(x_train[:, 0], y_train, color="red");
ax[0].set_title("Dancibility vs Y")
ax[0].plot(x_test_sorted, y_pred_sorted, color="green")
ax[1].scatter(x_train[:, 1], y_train, color="red");
ax[1].set_title("Energy vs Y")
ax[1].plot(x_test_sorted, y_pred_sorted, color="green")
ax[2].scatter(x_train[:, 2], y_train, color="red");
ax[2].set_title("Valence vs Y")
ax[2].plot(x_test_sorted, y_pred_sorted, color="green")
fig.tight_layout()
fig.suptitle("kernalised SVM Classifier C=0.1 and Gamma=25")
fig.legend(['train', 'predict'])
plt.show()

#################################################################################
# Confusion Matrix for kNN classifier where k=1 (selected via cross validation) #
#################################################################################
y_pred = model_knn.predict(x_test)
print("Confusion Matrix kNN Classifier:\n", confusion_matrix(y_test, y_pred))
print("Accuracy: %.2f" % (accuracy_score(y_test, y_pred)))

###########################################################################################################
# Confusion matrix for kernalised SVM classifier where C=0.1 and gamma=25 (selected via cross validation) #
###########################################################################################################
y_pred = modelKernalisedSVM.predict(x_test)
print("Confusion Matrix kernalised SVM Classifier:\n", confusion_matrix(y_test, y_pred))
print("Accuracy: %.2f" % (accuracy_score(y_test, y_pred)))

#########################################
# Confusion matrix for dummy classifier #
#########################################
dummy = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
ydummy = dummy.predict(x_test)
print("Dummy Confusion Matrix:\n", confusion_matrix(y_test, ydummy))
print("Accuracy: %.2f" % (accuracy_score(y_test, ydummy)))

################################################################################
# ROC curve for kNN classifier, kernalised SVM classifier and dummy classifier #
################################################################################
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