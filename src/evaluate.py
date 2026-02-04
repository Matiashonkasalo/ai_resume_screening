import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc


def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Create confusion matrix plot and return matplotlib figure.
    """

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots()
    disp.plot(ax=ax)

    return fig

def generate_classification_report(y_true, y_pred):
    """
    Return classification report as string.
    """

    return classification_report(y_true, y_pred)

def plot_roc_curve(model, X_test, y_test):
    """
    Plot ROC curve using predicted probabilities.
    """

    # Get probability predictions
    y_probs = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()

    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")  # random baseline

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    return fig, roc_auc
