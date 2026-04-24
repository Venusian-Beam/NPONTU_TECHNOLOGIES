"""
Unit tests for the Breast Cancer prediction API.

Tests cover:
    - ``GET  /health``    — readiness check
    - ``POST /predict``   — valid input, invalid input, edge cases
    - ``GET  /monitoring`` — metrics endpoint

The TestClient is used as a context manager so that the FastAPI
lifespan (model loading) is triggered before tests run.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# ── Ensure project root is importable ────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app

# ── Sample feature vectors for testing ───────────────────────────
# Typical malignant sample (class 0)
MALIGNANT_FEATURES: list[float] = [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
    0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
    0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119,
    0.2654, 0.4601, 0.1189,
]

# Typical benign sample (class 1)
BENIGN_FEATURES: list[float] = [
    13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664,
    0.04781, 0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56,
    0.008462, 0.01460, 0.02387, 0.01315, 0.01980, 0.002300,
    15.11, 19.26, 99.70, 711.2, 0.1440, 0.1773, 0.2390,
    0.1288, 0.2977, 0.07259,
]


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a TestClient with lifespan so the model loads."""
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    """Tests for the GET /health endpoint."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health endpoint should return 200 with correct structure."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert isinstance(data["total_predictions"], int)


class TestPredictEndpoint:
    """Tests for the POST /predict endpoint."""

    def test_predict_valid_malignant(self, client: TestClient) -> None:
        """Valid malignant features should return a valid prediction."""
        response = client.post(
            "/predict", json={"features": MALIGNANT_FEATURES}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in [0, 1]
        assert data["class_name"] in ["malignant", "benign"]
        assert isinstance(data["drift_detected"], bool)

    def test_predict_valid_benign(self, client: TestClient) -> None:
        """Valid benign features should return a valid prediction."""
        response = client.post(
            "/predict", json={"features": BENIGN_FEATURES}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in [0, 1]
        assert data["class_name"] in ["malignant", "benign"]

    def test_predict_response_includes_drift(self, client: TestClient) -> None:
        """Response must include drift detection metadata."""
        response = client.post(
            "/predict", json={"features": MALIGNANT_FEATURES}
        )
        data = response.json()
        assert "drift_detected" in data
        assert "drift_details" in data
        assert isinstance(data["drift_details"], dict)

    def test_predict_wrong_feature_count_too_few(self, client: TestClient) -> None:
        """Fewer than 30 features should return 422."""
        response = client.post(
            "/predict", json={"features": [1.0, 2.0, 3.0]}
        )
        assert response.status_code == 422

    def test_predict_wrong_feature_count_too_many(self, client: TestClient) -> None:
        """More than 30 features should return 422."""
        response = client.post(
            "/predict", json={"features": [0.0] * 31}
        )
        assert response.status_code == 422

    def test_predict_empty_features(self, client: TestClient) -> None:
        """Empty feature list should return 422."""
        response = client.post("/predict", json={"features": []})
        assert response.status_code == 422

    def test_predict_missing_body(self, client: TestClient) -> None:
        """Missing request body should return 422."""
        response = client.post("/predict")
        assert response.status_code == 422

    def test_predict_non_numeric_features(self, client: TestClient) -> None:
        """Non-numeric feature values should return 422."""
        response = client.post(
            "/predict", json={"features": ["a"] * 30}
        )
        assert response.status_code == 422


class TestMonitoringEndpoint:
    """Tests for the GET /monitoring endpoint."""

    def test_monitoring_returns_200(self, client: TestClient) -> None:
        """Monitoring endpoint should return current metrics."""
        # Make at least one prediction so there is data
        client.post("/predict", json={"features": BENIGN_FEATURES})

        response = client.get("/monitoring")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "prediction_distribution" in data
        assert data["total_predictions"] > 0
