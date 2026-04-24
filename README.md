# Breast Cancer Prediction System

A complete, easy-to-use machine learning system that classes breast tumours as **malignant** (harmful) or **benign** (safe). It uses the famous Wisconsin Diagnostic Breast Cancer dataset.

Built with **scikit-learn** and **FastAPI**, it includes tools to monitor the system's health.

---

##  What's Inside?

This project shows what a real-world machine learning app looks like under the hood:

| Part | What it Does |
|-------|------------|
| **Data & Training** (`ml/`) | Loads the data, trains the math model, and tests its accuracy |
| **API** (`app/`) | The web service that takes in numbers and gives back a prediction |
| **Monitoring** (`monitoring/`) | Keeps an eye out for weird data, biased guesses, and dropping accuracy |
| **Storage** (`models/`) | Saves the trained model so it doesn't have to relearn every time |
| **Logging** (`logs/`) | Saves a record of every prediction made |

---

##  Project Structure

```
ml-system/
│
├── app/
│   ├── main.py              # FastAPI entry point with lifespan hooks
│   ├── schema.py            # Pydantic request/response models
│   └── predictor.py         # Model loading and inference class
│
├── ml/
│   ├── train.py             # End-to-end training pipeline
│   ├── preprocess.py        # Data loading and preprocessing
│   ├── evaluate.py          # Accuracy, confusion matrix, F1 report
│   └── model.py             # sklearn Pipeline definition
│
├── monitoring/
│   ├── drift.py             # Z-score based data drift detection
│   ├── metrics.py           # Rolling accuracy & prediction distribution
│   └── logger.py            # Thread-safe CSV prediction logging
│
├── models/
│   ├── model.pkl            # Trained sklearn pipeline
│   └── metadata.pkl         # Training stats, feature/target names
│
├── logs/
│   └── predictions.csv      # Prediction audit log
│
├── tests/
│   └── test_api.py          # Unit tests for API endpoints
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

##  Dataset

**Breast Cancer Wisconsin (Diagnostic)**

- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Samples:** 569
- **Features:** 30 real-valued features computed from digitised images of fine needle aspirates (FNA) of breast masses
- **Classes:** 2 — `malignant` (class 0) and `benign` (class 1)
- **Task:** Binary classification

Key features include mean radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension — each with mean, standard error, and "worst" (largest) variants.

---

##  Setup Instructions

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
cd ml-system
pip install -r requirements.txt
```

---

##  Train the Model

```bash
python -m ml.train
```

**What happens:**
1. Loads the Breast Cancer dataset (569 samples, 30 features)
2. Splits 80/20 with stratification (`random_state=42`)
3. Fits a `StandardScaler → LogisticRegression` pipeline
4. Evaluates on the held-out test set
5. Saves `models/model.pkl` and `models/metadata.pkl`

**Expected output:**
```
============================================================
  BREAST CANCER CLASSIFIER — TRAINING PIPELINE
============================================================

[1/5] Loading Breast Cancer Wisconsin dataset ...
      Samples : 569
      Features: 30
      Classes : ['malignant', 'benign']

[2/5] Splitting data (train 80% / test 20%) ...
      Train samples: 455
      Test  samples: 114

[3/5] Training StandardScaler → LogisticRegression pipeline ...
      Training complete.

[4/5] Evaluating on test set ...
      Accuracy: 0.9737

[5/5] Saving model artefacts ...
      Model    → models/model.pkl
      Metadata → models/metadata.pkl

============================================================
  PIPELINE COMPLETE
============================================================
```

---

##  Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API: `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`
- OpenAPI spec: `http://localhost:8000/openapi.json`

### Screenshots

Here is a screenshot of the FastAPI Interactive Docs (Swagger UI) that is automatically generated for my endpoints:

![FastAPI Swagger UI Docs](docs_screenshot.png)

---

##  API Endpoints

### `POST /predict` — Classify a tumour sample

**Request:**
```json
{
  "features": [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
    0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
    0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119,
    0.2654, 0.4601, 0.1189
  ]
}
```

**Response:**
```json
{
  "prediction": 0,
  "class_name": "malignant",
  "drift_detected": false,
  "drift_details": {
    "drifted": false,
    "z_scores": [0.82, 0.31, ...],
    "drifted_features": [],
    "alert_message": null
  }
}
```

**cURL example:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}'
```

### `GET /health` — Health Check

```json
{
  "status": "healthy",
  "model_loaded": true,
  "total_predictions": 42
}
```

### `GET /monitoring` — Live Monitoring Metrics

```json
{
  "total_predictions": 42,
  "total_labelled": 0,
  "rolling_accuracy": null,
  "prediction_distribution": {
    "counts": {"0": 18, "1": 24},
    "proportions": {"0": 0.4286, "1": 0.5714},
    "imbalanced": false
  }
}
```

---

##  How I Monitor Health

### A. Data Checks (Data Drift) — `monitoring/drift.py`

When data comes in, I match it against the average numbers my model learned during training:

- I use a basic math test (Z-score) on every feature.
- If a feature looks way out of place (more than **2.0** standard deviations away), it triggers a warning.
- This helps me quickly spot if the kind of tumours being checked are changing.

### B. Guess Checks (Prediction Drift) — `monitoring/metrics.py`

I keep a running tally of what the model is predicting:

- It counts how many malignant vs benign guesses happen.
- If the model gets stuck and guesses the same thing over **80%** of the time, it prints a **prediction drift alert**.

### C. Accuracy Tracking (Performance) — `monitoring/metrics.py`

When doctors confirm what the tumour actually was, we can send that back to the `/feedback` endpoint:

- The system matches its past guess with the real answer.
- It calculates a rolling accuracy score for the last 50 checks.
- If this score falls under **85%**, a **performance alert** is printed to say "it's time to retrain the model!"

### D. Saving Records (Logging) — `monitoring/logger.py`

Every single prediction is saved to `logs/predictions.csv` for safety and auditing:

| timestamp | features | prediction | class_name |
|-----------|----------|------------|------------|
| 2026-04-23T00:... | [17.99, 10.38, ...] | 0 | malignant |

---

##  Run Tests

```bash
pytest tests/ -v
```

Tests cover:
-  Health endpoint returns correct structure
-  Valid malignant/benign predictions
-  Drift detection metadata included
-  Invalid inputs (wrong count, empty, non-numeric) return 422
-  Monitoring endpoint returns metrics

---

##  Docker

### Build

```bash
docker build -t breast-cancer-api .
```

### Run

```bash
docker run -p 8000:8000 breast-cancer-api
```

The model is trained at build time, so the container is self-contained.

---

##  Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Framework | scikit-learn 1.3 |
| API Framework | FastAPI |
| ASGI Server | Uvicorn |
| Validation | Pydantic v2 |
| Serialisation | joblib |
| Data Processing | NumPy, Pandas |
| Testing | pytest + httpx |
| Containerisation | Docker |

---

##  License

This project is provided for educational and assessment purposes.
