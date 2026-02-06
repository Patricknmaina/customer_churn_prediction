# scripts/modeling/evaluate.py

"""
Functions for evaluating model performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def plot_confusion_matrix(y_true, y_pred, class_labels=None, title='Confusion Matrix', figsize=(6, 5), cmap='viridis'):
    """
    Plots a confusion matrix using a Seaborn heatmap.

    Parameters:
        y_true: array-like of shape (n_samples,) - True labels
        y_pred: array-like of shape (n_samples,) - Predicted labels
        class_labels: list of strings - Names of the target classes
        title: string - Title of the plot
        figsize: tuple - Size of the plot
        cmap: string - Color map for the heatmap
    """

    # Define the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=figsize)
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap=cmap,
        xticklabels=class_labels if class_labels else 'auto',
        yticklabels=class_labels if class_labels else 'auto'
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


def compute_classification_metrics(y_true, y_pred, y_proba=None):
    """
    Compute common classification metrics.

    Parameters:
        y_true: array-like - True labels
        y_pred: array-like - Predicted labels
        y_proba: array-like, optional - Predicted probabilities for the positive class

    Returns:
        dict: Dictionary with accuracy, recall, precision, f1, and optionally roc_auc.
    """
    
    # Define the performance metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    return metrics
