"""
Pydantic request and response schemas for the Breast Cancer
prediction API.

All input validation (feature count, numeric types) and output
serialisation is handled here, keeping the endpoint logic clean.
"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input schema for the ``POST /predict`` endpoint.

    Attributes:
        features: A list of exactly 30 numeric values representing
                  the Breast Cancer Wisconsin diagnostic features.
    """

    features: list[float] = Field(
        ...,
        examples=[
            [
                17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
                0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
                0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119,
                0.2654, 0.4601, 0.1189,
            ]
        ],
        description="Numeric features from the dataset. Dataset replacement doesn't require schema changes.",
    )


class FeedbackRequest(BaseModel):
    """Schema for submitting ground-truth labels for performance monitoring."""
    
    prediction_id: int | None = Field(None, description="Optional ID of the prediction")
    true_label: int = Field(..., description="Ground-truth class label (e.g. 0 or 1)")


class PredictionResponse(BaseModel):
    """Output schema for the ``POST /predict`` endpoint.

    Attributes:
        prediction: Integer class index (0 = malignant, 1 = benign).
        class_name: Human-readable diagnosis label.
    """

    prediction: int = Field(..., description="Predicted class index (0 or 1).")
    class_name: str = Field(
        ..., description="Predicted class name ('malignant' or 'benign')."
    )


class PredictionWithMonitoring(PredictionResponse):
    """Extended prediction response that includes drift metadata.

    Attributes:
        drift_detected: Whether data drift was detected for this input.
        drift_details: Detailed drift analysis (z-scores, indices, etc.).
    """

    drift_detected: bool = Field(
        False, description="True if data drift was detected."
    )
    drift_details: dict | None = Field(
        None, description="Per-feature drift analysis details."
    )


class HealthResponse(BaseModel):
    """Response schema for the ``GET /health`` endpoint."""

    status: str = Field(..., description="Service health status.")
    model_loaded: bool = Field(
        ..., description="Whether the model is loaded and ready."
    )
    total_predictions: int = Field(
        ..., description="Total number of predictions served since startup."
    )


class MonitoringSummary(BaseModel):
    """Response schema for the ``GET /monitoring`` endpoint."""

    total_predictions: int = Field(
        ..., description="Total predictions since startup."
    )
    total_labelled: int = Field(
        ..., description="Predictions for which ground truth is available."
    )
    rolling_accuracy: float | None = Field(
        None, description="Rolling accuracy over the latest labelled window."
    )
    prediction_distribution: dict = Field(
        ..., description="Counts and proportions per predicted class."
    )
