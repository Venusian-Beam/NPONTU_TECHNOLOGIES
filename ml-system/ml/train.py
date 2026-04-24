"""
Training pipeline module.

Orchestrates the end-to-end model training workflow:
    1. Load the Breast Cancer dataset.
    2. Split into train / test sets (80/20, stratified).
    3. Create and fit the StandardScaler → LogisticRegression pipeline.
    4. Evaluate on the held-out test set.
    5. Persist the trained model and training metadata to disk.

Run directly::

    python -m ml.train
"""

import os
import sys
from pathlib import Path

import joblib

# ── Ensure project root is importable ────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluate import evaluate_model
from ml.model import create_pipeline
from ml.preprocess import get_training_statistics, load_dataset, split_data


def train_and_save(
    model_dir: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Execute the full training pipeline and persist artefacts.

    Args:
        model_dir: Directory to save model artefacts in.
                   Defaults to ``<project_root>/models``.
        test_size: Fraction of data held out for evaluation.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary with evaluation results and saved file paths.
    """
    if model_dir is None:
        model_dir = str(PROJECT_ROOT / "models")
    os.makedirs(model_dir, exist_ok=True)

    # ── 1. Load data ────────────────────────────────────────────
    print("=" * 60)
    print("  BREAST CANCER CLASSIFIER - TRAINING PIPELINE")
    print("=" * 60)
    print("\n[1/5] Loading Breast Cancer Wisconsin dataset ...")
    X, y, feature_names, target_names = load_dataset()
    print(f"      Samples : {X.shape[0]}")
    print(f"      Features: {X.shape[1]}")
    print(f"      Classes : {target_names}")

    # ── 2. Split ────────────────────────────────────────────────
    print(
        f"\n[2/5] Splitting data "
        f"(train {1 - test_size:.0%} / test {test_size:.0%}) ..."
    )
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"      Train samples: {X_train.shape[0]}")
    print(f"      Test  samples: {X_test.shape[0]}")

    # ── 3. Train ────────────────────────────────────────────────
    print("\n[3/5] Training StandardScaler + LogisticRegression pipeline ...")
    pipeline = create_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)
    print("      Training complete.")

    # ── 4. Evaluate ─────────────────────────────────────────────
    print("\n[4/5] Evaluating on test set ...")
    results = evaluate_model(pipeline, X_test, y_test, target_names)

    print(f"      Accuracy: {results['accuracy']:.4f}")
    print(f"\n      Confusion Matrix:\n{results['confusion_matrix']}")
    print(f"\n      Classification Report:\n{results['classification_report']}")

    # ── 5. Persist ──────────────────────────────────────────────
    print("[5/5] Saving model artefacts ...")

    train_stats = get_training_statistics(X_train)

    model_path = os.path.join(model_dir, "model.pkl")
    metadata_path = os.path.join(model_dir, "metadata.pkl")

    metadata: dict = {
        "feature_names": feature_names,
        "target_names": target_names,
        "train_stats": train_stats,
        "accuracy": results["accuracy"],
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "random_state": random_state,
    }

    joblib.dump(pipeline, model_path)
    joblib.dump(metadata, metadata_path)

    print(f"      Model    -> {model_path}")
    print(f"      Metadata -> {metadata_path}")
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)

    return {
        "model_path": model_path,
        "metadata_path": metadata_path,
        "accuracy": results["accuracy"],
        "confusion_matrix": results["confusion_matrix"],
    }


# ── CLI entry point ─────────────────────────────────────────────
if __name__ == "__main__":
    train_and_save()
