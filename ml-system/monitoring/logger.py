"""
Prediction logging module.

Provides thread-safe CSV logging of every prediction made by
the API, including timestamp, all 30 input features, the numeric
prediction, and the human-readable class name.

Logs are appended to ``logs/predictions.csv`` and can be read
back for monitoring and auditing.
"""

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

# ── Paths ────────────────────────────────────────────────────────
LOGS_DIR: Path = Path(__file__).resolve().parent.parent / "logs"
PREDICTIONS_LOG: Path = LOGS_DIR / "predictions.csv"

# Thread-safety for concurrent writes from async workers
_write_lock: Lock = Lock()

# CSV column headers — feature columns are named generically so
# the logger stays dataset-agnostic.
CSV_HEADERS: list[str] = [
    "timestamp",
    "features",
    "prediction",
    "class_name",
]


def _ensure_log_file() -> None:
    """Create the logs directory and CSV header row if they
    don't already exist."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    if not PREDICTIONS_LOG.exists():
        with open(PREDICTIONS_LOG, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)


def log_prediction(
    features: list[float],
    prediction: int,
    class_name: str,
) -> None:
    """Append a single prediction record to the CSV log.

    Args:
        features: Input feature values (length 30 for breast cancer).
        prediction: Numeric class prediction (0 or 1).
        class_name: Human-readable class label.
    """
    _ensure_log_file()
    timestamp: str = datetime.now(timezone.utc).isoformat()

    # Store features as a JSON array string so the CSV
    # remains one row per prediction and is easy to parse.
    features_str: str = json.dumps([round(v, 6) for v in features])
    row: list = [timestamp, features_str, prediction, class_name]

    with _write_lock:
        with open(PREDICTIONS_LOG, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)


def get_prediction_history(n_last: int | None = None) -> list[dict]:
    """Read prediction history from the log file.

    Args:
        n_last: If provided, return only the N most recent records.

    Returns:
        List of dictionaries, one per logged prediction.
    """
    _ensure_log_file()
    records: list[dict] = []
    with open(PREDICTIONS_LOG, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))

    if n_last is not None:
        records = records[-n_last:]
    return records
