"""
ML pipeline definition module.

Defines the scikit-learn Pipeline that combines StandardScaler
feature preprocessing with a Logistic Regression classifier.

Keeping the pipeline definition in its own module ensures the
preprocessing-plus-model combination is consistent across training
and inference — the scaler is always applied before the classifier.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline(
    max_iter: int = 1000,
    random_state: int = 42,
    C: float = 1.0,
) -> Pipeline:
    """Create an unfitted sklearn Pipeline for binary classification.

    Steps:
        1. **StandardScaler** — zero-mean, unit-variance normalisation.
        2. **LogisticRegression** — L2-regularised logistic regression
           with the LBFGS solver.

    Args:
        max_iter: Maximum solver iterations (increased for convergence
                  on 30-feature datasets).
        random_state: Seed for reproducibility.
        C: Inverse of regularisation strength; smaller values mean
           stronger regularisation.

    Returns:
        An unfitted sklearn Pipeline instance.
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=max_iter,
                    random_state=random_state,
                    C=C,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    return pipeline
