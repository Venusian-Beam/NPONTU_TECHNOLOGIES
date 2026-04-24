"""
Model loading and inference logic.

Encapsulates all model I/O in a single class so the API layer
never touches ``joblib`` or ``numpy`` directly.  The model and
metadata are loaded **once** at construction time and reused for
every inference call.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline


class Predictor:
    """Load a persisted sklearn pipeline and expose a ``predict`` method.

    Attributes:
        pipeline: The trained sklearn Pipeline (scaler + classifier).
        metadata: Training metadata dictionary (feature names, target
                  names, training statistics, etc.).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ) -> None:
        """Load the model and metadata from disk.

        Args:
            model_path: Path to ``model.pkl``.
                        Defaults to ``<project>/models/model.pkl``.
            metadata_path: Path to ``metadata.pkl``.
                           Defaults to ``<project>/models/metadata.pkl``.

        Raises:
            FileNotFoundError: If model or metadata files are missing.
        """
        project_root: Path = Path(__file__).resolve().parent.parent

        if model_path is None:
            model_path = project_root / "models" / "model.pkl"
        if metadata_path is None:
            metadata_path = project_root / "models" / "metadata.pkl"

        model_path = Path(model_path)
        metadata_path = Path(metadata_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Train the model first with: python -m ml.train"
            )
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}. "
                "Train the model first with: python -m ml.train"
            )

        self.pipeline: Pipeline = joblib.load(model_path)
        self.metadata: dict[str, Any] = joblib.load(metadata_path)

    # ── Convenience properties ───────────────────────────────────

    @property
    def target_names(self) -> list[str]:
        """Return class label names (['malignant', 'benign'])."""
        return self.metadata.get("target_names", [])

    @property
    def feature_names(self) -> list[str]:
        """Return the 30 feature names from the dataset."""
        return self.metadata.get("feature_names", [])

    @property
    def n_features(self) -> int:
        """Return the expected number of input features."""
        return int(self.metadata.get("n_features", 30))

    @property
    def train_stats(self) -> dict[str, np.ndarray]:
        """Return training-set mean and std for drift detection."""
        return self.metadata.get("train_stats", {})

    # ── Inference ────────────────────────────────────────────────

    def predict(self, features: list[float]) -> dict[str, Any]:
        """Run inference on a single observation.

        Args:
            features: List of numeric feature values (length must
                      equal ``n_features``).

        Returns:
            Dictionary with:
                - ``prediction`` (int): 0 (malignant) or 1 (benign).
                - ``class_name`` (str): Human-readable label.

        Raises:
            ValueError: If the feature vector length is incorrect.
        """
        expected: int = self.n_features
        if len(features) != expected:
            raise ValueError(
                f"Expected {expected} features, received {len(features)}."
            )

        X: np.ndarray = np.array(features).reshape(1, -1)
        prediction: int = int(self.pipeline.predict(X)[0])

        class_name: str = (
            self.target_names[prediction]
            if prediction < len(self.target_names)
            else f"class_{prediction}"
        )

        return {
            "prediction": prediction,
            "class_name": class_name,
        }
