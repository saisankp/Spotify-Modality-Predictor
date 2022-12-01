import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Get a dummy classifier that predicts the target value randomly.
def baseline(x_train, y_train):
    dummy = DummyClassifier(strategy="uniform").fit(x_train, y_train)
    return dummy


# Get confusion matrices and accuracy for kNN, kernalised SVM, and dummy classifiers.
def ConfusionMatrix(model_knn, modelKernalisedSVM, dummyClassifier, x_test, y_test):
    # Confusion matrix and accuracy for kNN classifier.
    y_pred = model_knn.predict(x_test)
    print("Confusion Matrix kNN Classifier:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy: %.2f" % (accuracy_score(y_test, y_pred)))
    knnAccuracy = accuracy_score(y_test, y_pred)

    # Confusion matrix and accuracy for kernalised SVM classifier.
    y_pred = modelKernalisedSVM.predict(x_test)
    print("Confusion Matrix kernalised SVM Classifier:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy: %.2f" % (accuracy_score(y_test, y_pred)))
    SVMAccuracy = accuracy_score(y_test, y_pred)

    # Confusion matrix and accuracy for dummy classifier.
    ydummy = dummyClassifier.predict(x_test)
    print("Dummy Confusion Matrix:\n", confusion_matrix(y_test, ydummy))
    print("Accuracy: %.2f" % (accuracy_score(y_test, ydummy)))
    dummyAccuracy = accuracy_score(y_test, ydummy)
    return knnAccuracy, SVMAccuracy, dummyAccuracy


# Plot ROC curve for kNN, kernalised SVM, and dummy classifiers.
def ROC(x_test, y_test, model_knn, modelKernalisedSVM, dummy):
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
