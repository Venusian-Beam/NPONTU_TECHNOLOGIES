"""
Performance tracking and prediction drift module.

Provides:
    - Rolling accuracy computation when ground-truth labels are available.
    - Prediction class distribution tracking with imbalance alerting.
    - A summary method that returns all monitoring metrics in one call.

Alerts are printed to stdout; in a production deployment these would
be routed to a structured logging / alerting backend.
"""

from collections import Counter
from typing import Any

ACCURACY_ALERT_THRESHOLD: float = 0.85
PREDICTION_IMBALANCE_THRESHOLD: float = 0.80  # one class > 80% of predictions


class PerformanceTracker:
    """Accumulates predictions and optional ground-truth labels to
    compute rolling performance metrics and detect prediction drift.

    Attributes:
        predictions: List of predicted class indices.
        true_labels: List of ground-truth labels (may lag predictions).
        target_names: Human-readable class label names.
    """

    def __init__(self, target_names: list[str] | None = None) -> None:
        """Initialise the tracker.

        Args:
            target_names: Class label names for display purposes
                          (e.g. ``['malignant', 'benign']``).
        """
        self.predictions: list[int] = []
        self.true_labels: list[int] = []
        self.target_names: list[str] = target_names or []

    # ── Recording ────────────────────────────────────────────────

    def record_prediction(
        self, prediction: int, true_label: int | None = None
    ) -> None:
        """Record a single prediction and optional true label.

        Args:
            prediction: Predicted class index.
            true_label: Ground-truth label, if available.
        """
        self.predictions.append(prediction)
        if true_label is not None:
            self.true_labels.append(true_label)

    # ── Rolling accuracy ─────────────────────────────────────────

    def rolling_accuracy(self, window: int = 50) -> float | None:
        """Compute accuracy over the most recent *window* labelled
        predictions.

        If accuracy drops below ``ACCURACY_ALERT_THRESHOLD`` an alert
        is printed.

        Args:
            window: Number of recent samples to consider.

        Returns:
            Accuracy as a float, or ``None`` if no labels are recorded.
        """
        if not self.true_labels:
            return None

        recent_preds: list[int] = self.predictions[-window:]
        recent_labels: list[int] = self.true_labels[-window:]

        # Align lengths -- labels may lag behind predictions.
        n: int = min(len(recent_preds), len(recent_labels))
        if n == 0:
            return None

        correct: int = sum(
            p == t for p, t in zip(recent_preds[-n:], recent_labels[-n:])
        )
        accuracy: float = correct / n

        if accuracy < ACCURACY_ALERT_THRESHOLD:
            print(
                f"PERFORMANCE ALERT: Rolling accuracy ({accuracy:.2%}) "
                f"dropped below threshold ({ACCURACY_ALERT_THRESHOLD:.0%})."
            )

        return accuracy

    # ── Prediction distribution ──────────────────────────────────

    def prediction_distribution(self) -> dict[str, Any]:
        """Compute the distribution of predicted classes and check
        for prediction drift (class imbalance).

        Returns:
            Dictionary with:
                - ``counts``: Per-class prediction counts.
                - ``proportions``: Per-class proportions.
                - ``imbalanced``: True if any class exceeds the
                  imbalance threshold.
        """
        if not self.predictions:
            return {"counts": {}, "proportions": {}, "imbalanced": False}

        counter: Counter = Counter(self.predictions)
        total: int = len(self.predictions)
        proportions: dict[int, float] = {k: v / total for k, v in counter.items()}

        max_proportion: float = max(proportions.values())
        imbalanced: bool = max_proportion > PREDICTION_IMBALANCE_THRESHOLD

        if imbalanced:
            dominant_class: int = max(proportions, key=proportions.get)  # type: ignore[arg-type]
            label: str = (
                self.target_names[dominant_class]
                if dominant_class < len(self.target_names)
                else str(dominant_class)
            )
            print(
                f"PREDICTION DRIFT ALERT: Class '{label}' accounts for "
                f"{max_proportion:.1%} of predictions "
                f"(threshold: {PREDICTION_IMBALANCE_THRESHOLD:.0%})."
            )

        return {
            "counts": dict(counter),
            "proportions": {str(k): round(v, 4) for k, v in proportions.items()},
            "imbalanced": imbalanced,
        }

    # ── Summary ──────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return a complete monitoring summary.

        Returns:
            Dictionary with rolling accuracy, prediction distribution,
            and sample counts.
        """
        return {
            "total_predictions": len(self.predictions),
            "total_labelled": len(self.true_labels),
            "rolling_accuracy": self.rolling_accuracy(),
            "prediction_distribution": self.prediction_distribution(),
        }
