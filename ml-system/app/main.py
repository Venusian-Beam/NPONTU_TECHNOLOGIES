"""
FastAPI entry point.

Exposes the Breast Cancer classification REST API with three endpoints:

    - ``POST /predict``  — classify a tumour as malignant or benign.
    - ``GET  /health``   — service health and readiness check.
    - ``GET  /monitoring`` — live monitoring metrics summary.

The model is loaded **once** at startup via the ASGI lifespan hook;
every request reuses the same in-memory pipeline for low-latency
inference.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks

# ── Ensure project root is importable ────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.predictor import Predictor
from app.schema import (
    HealthResponse,
    MonitoringSummary,
    PredictionRequest,
    PredictionWithMonitoring,
    FeedbackRequest,
)
from monitoring.drift import detect_feature_drift
from monitoring.logger import log_prediction
from monitoring.metrics import PerformanceTracker

# ── Global state (populated at startup) ──────────────────────────
predictor: Predictor | None = None
performance_tracker: PerformanceTracker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """ASGI lifespan: load model on startup, cleanup on shutdown."""
    global predictor, performance_tracker

    print("[STARTUP] Loading model ...")
    try:
        predictor = Predictor()
        performance_tracker = PerformanceTracker(
            target_names=predictor.target_names
        )
        print(
            f"[OK] Model loaded successfully. "
            f"Classes: {predictor.target_names} | "
            f"Features: {predictor.n_features}"
        )
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        print("   Train the model first:  python -m ml.train")
        raise

    yield  # ── Application is running ──

    print("[SHUTDOWN] Shutting down.")


from fastapi.responses import RedirectResponse

# ── FastAPI application ─────────────────────────────────────────
app: FastAPI = FastAPI(
    title="Breast Cancer Prediction API",
    description=(
        "Production-style REST API for breast cancer tumour classification "
        "(malignant vs benign) powered by a Logistic Regression pipeline, "
        "with integrated data drift detection, prediction logging, and "
        "performance monitoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect root to /docs."""
    return RedirectResponse(url="/docs")


# ═══════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════════════════════════


@app.post(
    "/predict",
    response_model=PredictionWithMonitoring,
    summary="Classify a tumour sample",
    tags=["Prediction"],
)
async def predict(
    request: PredictionRequest, background_tasks: BackgroundTasks
) -> PredictionWithMonitoring:
    """Classify a breast tumour as **malignant** (0) or **benign** (1)
    based on 30 numeric diagnostic features.

    The response includes real-time data drift analysis and the
    prediction is automatically logged for monitoring.
    """
    if predictor is None or performance_tracker is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable.",
        )

    try:
        # ── Inference ───────────────────────────────────────────
        result: dict = predictor.predict(request.features)

        # ── Drift detection ─────────────────────────────────────
        train_stats: dict = predictor.train_stats
        drift_result: dict = detect_feature_drift(
            live_features=np.array(request.features),
            train_mean=train_stats["mean"],
            train_std=train_stats["std"],
        )

        # ── Logging & tracking ──────────────────────────────────
        background_tasks.add_task(
            log_prediction,
            features=request.features,
            prediction=result["prediction"],
            class_name=result["class_name"],
        )
        performance_tracker.record_prediction(result["prediction"])

        return PredictionWithMonitoring(
            prediction=result["prediction"],
            class_name=result["class_name"],
            drift_detected=drift_result["drifted"],
            drift_details=drift_result,
        )

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Operations"],
)
async def health() -> HealthResponse:
    """Return service health status and readiness information."""
    total: int = (
        len(performance_tracker.predictions) if performance_tracker else 0
    )
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        model_loaded=predictor is not None,
        total_predictions=total,
    )


@app.get(
    "/monitoring",
    response_model=MonitoringSummary,
    summary="Monitoring metrics",
    tags=["Operations"],
)
async def monitoring() -> MonitoringSummary:
    """Return current monitoring metrics including prediction
    distribution, rolling accuracy, and sample counts."""
    if performance_tracker is None:
        raise HTTPException(
            status_code=503,
            detail="Performance tracker not initialised.",
        )

    summary: dict = performance_tracker.summary()
    return MonitoringSummary(**summary)


@app.post(
    "/feedback",
    summary="Submit ground truth",
    tags=["Monitoring"],
)
async def feedback(request: FeedbackRequest) -> dict:
    """Submit a true label for a previous prediction to compute rolling accuracy.
    
    In a real system, you'd link this via `prediction_id`. Here, we just append
    to the performance tracker.
    """
    if performance_tracker is None:
        raise HTTPException(status_code=503, detail="Tracker not initialised.")
        
    performance_tracker.record_prediction(
        prediction=-1,  # Placeholder, usually you'd update an existing record
        true_label=request.true_label
    )
    
    return {"message": "Feedback recorded."}

