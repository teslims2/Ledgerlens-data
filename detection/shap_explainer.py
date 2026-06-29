"""SHAP-based interpretability for risk scores.

Wraps each trained ensemble model with a SHAP explainer so that every
risk score can be accompanied by a per-feature attribution, surfaced via
the API for auditors and end-users.
"""

import numpy as np
import pandas as pd
import shap

from config import config
from detection.differential_privacy import (
    feature_sensitivity,
    gaussian_sigma,
    load_shap_sensitivity,
    renyi_noise_multiplier,
)
from detection.model_training import FEATURE_COLUMNS_EXCLUDE


class ShapExplainer:
    """Produces SHAP value explanations for one or more trained models.

    `TreeExplainer` construction is not free, so explainers are cached per
    model id (`id(model)`) and reused across calls.
    """

    def __init__(self, model=None):
        self._explainers: dict[int, shap.TreeExplainer] = {}
        self.model = model
        if model is not None:
            self.explainer = self._get_explainer(model)

    def _get_explainer(self, model) -> shap.TreeExplainer:
        key = id(model)
        if key not in self._explainers:
            self._explainers[key] = shap.TreeExplainer(model)
        return self._explainers[key]

    def _shap_values_for(self, model, X: pd.DataFrame):
        explainer = self._get_explainer(model)
        shap_values = explainer.shap_values(X)
        # Binary classifiers may return a list [class_0, class_1], or a single
        # ndarray shaped (n_samples, n_features, n_classes).
        if isinstance(shap_values, list):
            return shap_values[1][0]
        if shap_values.ndim == 3:
            return shap_values[0, :, 1]
        return shap_values[0]

    def explain(self, feature_row: pd.Series, top_n: int = 5, model=None) -> list[dict]:
        """Return the top `top_n` features driving this wallet's score
        according to a single model.

        Each entry: {"feature": str, "contribution": float, "value": float}
        """
        model = model or self.model
        if model is None:
            raise ValueError("No model provided to explain()")

        feature_cols = [c for c in feature_row.index if c not in FEATURE_COLUMNS_EXCLUDE]
        X = feature_row[feature_cols].to_frame().T.astype(float)

        values = self._shap_values_for(model, X)

        contributions = sorted(
            zip(feature_cols, values, X.iloc[0].values, strict=True),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:top_n]

        return [
            {"feature": name, "contribution": float(value), "value": float(raw)}
            for name, value, raw in contributions
        ]

    def shap_dict(self, X: pd.DataFrame, model=None) -> dict[str, float]:
        """Return exact ``{feature: contribution}`` for the single row in `X`."""
        model = model or self.model
        if model is None:
            raise ValueError("No model provided to shap_dict()")

        feature_cols = [c for c in X.columns if c not in FEATURE_COLUMNS_EXCLUDE]
        X_features = X[feature_cols].astype(float)
        values = self._shap_values_for(model, X_features)
        return {col: float(v) for col, v in zip(feature_cols, values, strict=True)}

    def explain_private(
        self,
        model,
        X: pd.DataFrame,
        wallet: str,
        epsilon: float | None = None,
        delta: float | None = None,
        *,
        private: bool = True,
        sensitivities: dict | None = None,
        query_store=None,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Return SHAP values with calibrated Gaussian noise for (epsilon, delta)-DP.

        The Gaussian mechanism adds ``N(0, sigma_i^2)`` to each feature's SHAP
        value, where ``sigma_i`` is derived from that feature's sensitivity
        (max SHAP change from adding/removing one trade, see
        `scripts/estimate_shap_sensitivity.py`). This makes individual trade
        contributions indistinguishable, defeating model-inversion attacks.

        Audit mode (`private=False`) returns the exact SHAP values unchanged and
        consumes no privacy budget — intended for the authenticated, logged
        internal audit API only.

        When a `query_store` is supplied, the per-wallet query count is
        incremented and Rényi composition scales `sigma` once the wallet exceeds
        `config.DP_RENYI_QUERY_THRESHOLD` queries.
        """
        exact = self.shap_dict(X, model)
        if not private:
            return exact

        epsilon = config.DP_EPSILON if epsilon is None else epsilon
        delta = config.DP_DELTA if delta is None else delta
        if sensitivities is None:
            sensitivities = load_shap_sensitivity()

        query_count = 0
        if query_store is not None:
            query_count = query_store.increment_shap_query(wallet)
        noise_scale = renyi_noise_multiplier(query_count)

        rng = np.random.default_rng(seed)
        private_values: dict[str, float] = {}
        for feature, value in exact.items():
            sensitivity = feature_sensitivity(sensitivities, feature)
            sigma = gaussian_sigma(sensitivity, epsilon, delta) * noise_scale
            private_values[feature] = float(value + rng.normal(0.0, sigma))
        return private_values

    def explain_ensemble(self, feature_row: pd.Series, models: dict, top_n: int = 5) -> list[dict]:
        """Aggregate per-model SHAP contributions across an ensemble into a
        single ranked list.

        `models` maps model name -> fitted estimator (e.g. the `MODEL_REGISTRY`
        models loaded by `RiskScorer`). Contributions for each feature are
        averaged across models, then sorted by absolute magnitude.

        Each entry: {"feature": str, "contribution": float, "value": float}
        """
        if not models:
            raise ValueError("No models provided to explain_ensemble()")

        feature_cols = [c for c in feature_row.index if c not in FEATURE_COLUMNS_EXCLUDE]
        X = feature_row[feature_cols].to_frame().T.astype(float)
        raw_values = X.iloc[0].values

        totals = [0.0] * len(feature_cols)
        for model in models.values():
            values = self._shap_values_for(model, X)
            for i, value in enumerate(values):
                totals[i] += float(value)

        averaged = [total / len(models) for total in totals]

        contributions = sorted(
            zip(feature_cols, averaged, raw_values, strict=True),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:top_n]

        return [
            {"feature": name, "contribution": float(value), "value": float(raw)}
            for name, value, raw in contributions
        ]
