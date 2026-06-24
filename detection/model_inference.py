"""Real-time risk scoring using the trained ensemble.

Loads model artifacts from `config.MODEL_DIR` and combines per-model
probabilities into the single LedgerLens Risk Score (0-100) consumed by
the API and the `ledgerlens-score` Soroban contract.

The returned dict's `score`, `benford_flag`, `ml_flag`, and `confidence`
fields match the contract's `RiskScore` struct (the `timestamp` field is
added by the persistence layer when a record is stored).
"""

import os

import joblib
import numpy as np
import pandas as pd

from config import config
from detection.model_training import FEATURE_COLUMNS_EXCLUDE, MODEL_REGISTRY

BENFORD_MAD_FLAG_THRESHOLD = 0.015
ML_FLAG_THRESHOLD = 0.5


def _combine_probabilities(probs: list[float], weights: list[float] | None = None) -> float:
    """Combine per-model probabilities into a single ensemble probability.

    Defaults to a simple average; pass `weights` (same length as `probs`)
    to weight individual models differently.
    """
    if weights is None:
        weights = [1.0] * len(probs)
    return sum(p * w for p, w in zip(probs, weights, strict=True)) / sum(weights)


def _confidence_from_probs(probs: list[float], avg_prob: float) -> int:
    """Derive a 0-100 confidence score from how far the ensemble probability
    is from the decision boundary, discounted by inter-model disagreement."""
    certainty = abs(avg_prob - 0.5) * 2
    if len(probs) > 1:
        agreement = 1.0 - (max(probs) - min(probs))
        certainty *= max(agreement, 0.0)
    return int(round(certainty * 100))


class RiskScorer:
    """Loads trained ensemble models and produces risk scores."""

    def __init__(self, model_dir: str | None = None):
        self.model_dir = model_dir or config.MODEL_DIR
        self.models = self._load_models()

    def _load_models(self) -> dict:
        models = {}
        for name in MODEL_REGISTRY:
            path = os.path.join(self.model_dir, f"{name}.joblib")
            if os.path.exists(path):
                models[name] = joblib.load(path)
        return models

    def _ensemble_probabilities(self, feature_row: pd.Series) -> list[float]:
        """Per-model wash-trade probabilities for a single feature row.

        Raises if no models are loaded so callers (`score`,
        `score_continuous`) surface the same error.
        """
        if not self.models:
            raise RuntimeError(
                f"No trained models found in {self.model_dir}. " "Run model_training.py first."
            )

        feature_cols = [c for c in feature_row.index if c not in FEATURE_COLUMNS_EXCLUDE]
        X = feature_row[feature_cols].to_frame().T.astype(float)
        return [model.predict_proba(X)[0, 1] for model in self.models.values()]

    def score_continuous(self, feature_row: pd.Series) -> float:
        """Continuous ensemble risk score in `[0, 100]` (unrounded).

        `score` rounds this to an int for the on-chain `RiskScore`, which
        makes it locally flat and unusable for gradient-based analysis. The
        adversarial robustness tooling (`detection/adversarial`) estimates
        feature gradients via finite differences against this method, so it
        must stay continuous.
        """
        return _combine_probabilities(self._ensemble_probabilities(feature_row)) * 100

    def score_continuous_batch(self, X: pd.DataFrame) -> np.ndarray:
        """Continuous ensemble scores for a batch of feature rows.

        `X` must contain (at least) the model feature columns; non-feature
        columns (`FEATURE_COLUMNS_EXCLUDE`) are dropped. Vectorised over the
        batch so the adversarial tooling can evaluate every finite-difference
        probe in one `predict_proba` call per model instead of one per row.
        """
        if not self.models:
            raise RuntimeError(
                f"No trained models found in {self.model_dir}. " "Run model_training.py first."
            )
        feature_cols = [c for c in X.columns if c not in FEATURE_COLUMNS_EXCLUDE]
        Xf = X[feature_cols].astype(float)
        per_model = np.column_stack([m.predict_proba(Xf)[:, 1] for m in self.models.values()])
        return per_model.mean(axis=1) * 100

    def score(self, feature_row: pd.Series) -> dict:
        """Score a single wallet's feature row.

        Returns a dict matching the on-chain `RiskScore` shape:
            {score, benford_flag, ml_flag, confidence}
        """
        probs = self._ensemble_probabilities(feature_row)
        avg_prob = _combine_probabilities(probs)

        benford_mad_cols = [c for c in feature_row.index if c.startswith("benford_mad_")]
        benford_flag = bool(
            benford_mad_cols and (feature_row[benford_mad_cols] > BENFORD_MAD_FLAG_THRESHOLD).any()
        )

        return {
            "score": int(round(avg_prob * 100)),
            "benford_flag": benford_flag,
            "ml_flag": bool(avg_prob >= ML_FLAG_THRESHOLD),
            "confidence": _confidence_from_probs(probs, avg_prob),
        }

    def score_matrix(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """Score every row in a feature matrix, returning the matrix with
        `score`, `benford_flag`, `ml_flag`, `confidence` columns appended."""
        scores = feature_matrix.apply(self.score, axis=1, result_type="expand")
        return pd.concat([feature_matrix[["wallet"]], scores], axis=1)
