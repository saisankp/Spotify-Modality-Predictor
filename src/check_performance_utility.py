import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve, classification_report, accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix


# Get a dummy classifier that predicts the target value randomly.
def baseline(x_train, y_train):
    dummy = DummyClassifier(strategy="uniform").fit(x_train, y_train)
    return dummy


# Print confusion matrices and classification reports (accuracy, precision, recall, f1-score, support) for kNN,
# kernalised SVM and dummy classifiers, returning the accuracy of each.
def compareModels(model_knn, modelKernalisedSVM, dummyClassifier, x_test, y_test):
    # Print confusion matrix and classification report for kNN classifier
    y_predkNN = model_knn.predict(x_test)
    print("kNN Classifier Confusion Matrix:\n", confusion_matrix(y_test, y_predkNN))
    print("kNN Classification Report:\n", classification_report(y_test, y_predkNN, zero_division=0))
    # Store accuracy of kNN classifier
    kNNAccuracy = accuracy_score(y_test, y_predkNN)

    # Print confusion matrix and classification report for kernalised SVM classifier.
    y_predSVM = modelKernalisedSVM.predict(x_test)
    print("Kernalised SVM Classifier Confusion Matrix :\n", confusion_matrix(y_test, y_predSVM))
    print("Kernalised SVM Classification Report:\n", classification_report(y_test, y_predSVM, zero_division=0))
    # Store accuracy of kernalised SVM classifier
    kernalisedSVMAccuracy = accuracy_score(y_test, y_predSVM)

    # Print confusion matrix and classification report for kernalised SVM classifier.
    ydummy = dummyClassifier.predict(x_test)
    print("Dummy Classifier Confusion Matrix:\n", confusion_matrix(y_test, ydummy))
    print("Dummy Classification Report:\n", classification_report(y_test, ydummy, zero_division=0))
    # Store accuracy of dummy classifier
    dummyAccuracy = accuracy_score(y_test, ydummy)
    return kNNAccuracy, kernalisedSVMAccuracy, dummyAccuracy


# Plot ROC curves for kNN, kernalised SVM, and dummy classifiers.
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
