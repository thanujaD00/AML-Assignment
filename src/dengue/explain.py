"""SHAP (global + local) and LIME (local) explanations for the selected model.

Important for interpretation: the models predict **log growth** of next week's
cases relative to this week. A positive contribution therefore pushes the
forecast *above* "same as this week"; a negative one pushes it *below*.
Explanations describe what the model has learned from 2026 weeks 25-33 -
they are associations, not causes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from .config import RANDOM_STATE
from .models import MeanEnsemble


def _underlying(model):
    """Return the final estimator of a pipeline (or the model itself)."""
    return model.steps[-1][1] if hasattr(model, "steps") else model


def shap_values(model, X: pd.DataFrame, background: pd.DataFrame) -> shap.Explanation:
    """SHAP values for tree models or a mean ensemble of tree models.

    For an averaging ensemble the SHAP values are the average of the members'
    TreeSHAP values; this is exact because SHAP values are additive in the model.
    """
    if isinstance(model, MeanEnsemble):
        parts = [shap_values(m, X, background) for _, m in model.fitted_]
        return shap.Explanation(
            values=np.mean([p.values for p in parts], axis=0),
            base_values=np.mean([p.base_values for p in parts], axis=0),
            data=X.to_numpy(),
            feature_names=list(X.columns),
        )
    est = _underlying(model)
    try:
        explainer = shap.TreeExplainer(est, data=background, feature_perturbation="interventional")
    except Exception:  # non-tree model -> model-agnostic fallback
        explainer = shap.Explainer(model.predict, background)
    exp = explainer(X)
    exp.feature_names = list(X.columns)
    return exp


def lime_explain(model, X_train: pd.DataFrame, x_row: pd.Series, num_features: int = 8):
    """Fit a LIME local surrogate around one instance (regression mode)."""
    explainer = LimeTabularExplainer(
        training_data=X_train.to_numpy(),
        feature_names=list(X_train.columns),
        mode="regression",
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )
    predict_fn = lambda arr: model.predict(pd.DataFrame(arr, columns=X_train.columns))
    return explainer.explain_instance(
        x_row.to_numpy(), predict_fn, num_features=num_features, num_samples=5000
    )
