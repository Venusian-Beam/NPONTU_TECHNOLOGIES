"""
Data preprocessing module.

Handles loading the Breast Cancer Wisconsin dataset and preparing
it for training, including stratified train/test splitting and
computation of training-set statistics for downstream drift detection.

The dataset loader is isolated so it can be swapped for a different
data source in the future without touching the rest of the pipeline.
"""

from typing import Tuple

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_dataset() -> Tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load the Breast Cancer Wisconsin diagnostic dataset.

    Returns:
        Tuple containing:
            - X: Feature matrix of shape (569, 30).
            - y: Binary target vector of shape (569,).
                 0 = malignant, 1 = benign.
            - feature_names: List of 30 feature column names.
            - target_names: List of class label names
                            ['malignant', 'benign'].
    """
    data = load_breast_cancer()
    X: np.ndarray = data.data
    y: np.ndarray = data.target
    feature_names: list[str] = list(data.feature_names)
    target_names: list[str] = list(data.target_names)
    return X, y, feature_names, target_names


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into training and testing sets with stratification.

    Stratification preserves the class distribution in both splits,
    which is important for imbalanced medical datasets.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Proportion of data to hold out for testing.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def get_training_statistics(X_train: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-feature mean and standard deviation of training data.

    These statistics are persisted alongside the model and used at
    inference time for data drift detection.

    Args:
        X_train: Training feature matrix of shape (n_samples, n_features).

    Returns:
        Dictionary with 'mean' and 'std' arrays, each of shape (n_features,).
    """
    return {
        "mean": np.mean(X_train, axis=0),
        "std": np.std(X_train, axis=0),
    }
