"""
Data drift detection module.

Detects distribution shift between live inference inputs and the
training data by comparing per-feature means and standard deviations
using a z-score approach.

When a feature's z-score exceeds the configured threshold, an alert
is printed to stdout (suitable for log aggregation in production).
"""

import numpy as np

# Default threshold: alert if any feature mean shifts by more than
# this many training-set standard deviations.
DEFAULT_DRIFT_THRESHOLD: float = 2.0


def detect_feature_drift(
    live_features: np.ndarray,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> dict:
    """Check whether a single live observation has drifted from
    the training distribution.

    For each feature the z-score is computed as::

        z = |live_value - train_mean| / train_std

    Features whose z-score exceeds ``threshold`` are flagged.

    Args:
        live_features: Feature vector of shape (n_features,).
        train_mean: Per-feature mean from the training set.
        train_std: Per-feature standard deviation from the training set.
        threshold: Number of standard deviations that triggers an alert.

    Returns:
        Dictionary with:
            - ``drifted`` (bool): True if any feature exceeds threshold.
            - ``z_scores`` (list[float]): Per-feature z-scores.
            - ``drifted_features`` (list[int]): Indices of drifted features.
            - ``alert_message`` (str | None): Human-readable alert or None.
    """
    # Guard against zero std (constant features in training)
    safe_std: np.ndarray = np.where(train_std == 0, 1e-8, train_std)
    z_scores: np.ndarray = np.abs((live_features - train_mean) / safe_std)

    drifted_indices: list[int] = [int(i) for i in np.where(z_scores > threshold)[0]]
    is_drifted: bool = len(drifted_indices) > 0

    alert_message: str | None = None
    if is_drifted:
        alert_message = (
            f"DATA DRIFT ALERT: {len(drifted_indices)} feature(s) "
            f"exceed threshold ({threshold:.1f} std). "
            f"Indices: {drifted_indices}. "
            f"Max z-score: {z_scores.max():.2f}"
        )
        print(alert_message)

    return {
        "drifted": is_drifted,
        "z_scores": [round(float(z), 4) for z in z_scores],
        "drifted_features": drifted_indices,
        "alert_message": alert_message,
    }


def detect_batch_drift(
    live_batch: np.ndarray,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> dict:
    """Detect drift across a batch of observations.

    Computes the mean of the batch and compares it against the
    training distribution using the same z-score method.

    Args:
        live_batch: Feature matrix of shape (n_samples, n_features).
        train_mean: Per-feature mean from the training set.
        train_std: Per-feature standard deviation from the training set.
        threshold: Number of standard deviations that triggers an alert.

    Returns:
        Same structure as ``detect_feature_drift``, computed on
        the batch mean vector.
    """
    batch_mean: np.ndarray = np.mean(live_batch, axis=0)
    return detect_feature_drift(batch_mean, train_mean, train_std, threshold)
