"""
Model evaluation module.

Provides functions to compute accuracy, confusion matrix,
precision/recall/F1 classification report, and a convenience
function that runs the full evaluation suite on a trained pipeline.
"""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate classification accuracy.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Accuracy as a float between 0.0 and 1.0.
    """
    return float(accuracy_score(y_true, y_pred))


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Compute the confusion matrix.

    Layout::

        [[TN, FP],
         [FN, TP]]

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Confusion matrix of shape (n_classes, n_classes).
    """
    return confusion_matrix(y_true, y_pred)


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str] | None = None,
) -> str:
    """Generate a full-text classification report with precision,
    recall, and F1-score per class.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        target_names: Optional list of class label names.

    Returns:
        Formatted classification report string.
    """
    return classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )


def evaluate_model(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: list[str] | None = None,
) -> dict[str, Any]:
    """Run the full evaluation suite on a trained pipeline.

    Args:
        pipeline: A fitted sklearn Pipeline.
        X_test: Test feature matrix.
        y_test: Test ground-truth labels.
        target_names: Optional list of class label names
                      (e.g. ['malignant', 'benign']).

    Returns:
        Dictionary containing:
            - accuracy (float)
            - confusion_matrix (np.ndarray)
            - classification_report (str)
            - y_pred (np.ndarray)
    """
    y_pred: np.ndarray = pipeline.predict(X_test)

    accuracy = compute_accuracy(y_test, y_pred)
    cm = compute_confusion_matrix(y_test, y_pred)
    report = generate_classification_report(y_test, y_pred, target_names)

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_pred": y_pred,
    }
