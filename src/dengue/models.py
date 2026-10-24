"""Candidate models: two naive baselines, a linear benchmark and tree ensembles.

All models predict the *log growth* target (see ``features.py``) from the
feature matrix ``X`` (a DataFrame with ``config.FEATURES`` columns).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .config import RANDOM_STATE


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
class PersistenceForecaster(BaseEstimator, RegressorMixin):
    """Naive forecast: next week's cases = this week's cases (growth = 0)."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class MovingAverageForecaster(BaseEstimator, RegressorMixin):
    """Next week's cases = mean of weeks t and t-1 (falls back to week t if t-1 is missing)."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        log_t = X["log_cases_t"].to_numpy()
        y_t = np.expm1(log_t)
        y_prev = np.expm1(log_t - X["growth_1w"].to_numpy())
        return np.log1p((y_t + y_prev) / 2.0) - log_t


class NationalTrendForecaster(BaseEstimator, RegressorMixin):
    """Persistence scaled by last week's national growth: y[t+1] = y[t] * national(t)/national(t-1).

    A stronger, still model-free baseline: it captures the island-wide epidemic
    phase (rising / falling), so an ML model must beat simple trend-following.
    """

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X["national_growth_1w"].to_numpy()


class MeanEnsemble(BaseEstimator, RegressorMixin):
    """Simple averaging ensemble of already-configured regressors."""

    def __init__(self, estimators: list[tuple[str, Any]] | None = None):
        self.estimators = estimators

    def fit(self, X, y):
        self.fitted_ = [(n, clone(e).fit(X, y)) for n, e in self.estimators]
        return self

    def predict(self, X):
        return np.mean([e.predict(X) for _, e in self.fitted_], axis=0)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    name: str
    family: str
    build: Callable[..., Any]
    grid: list[dict] = field(default_factory=lambda: [{}])
    tree_based: bool = False
    note: str = ""


def _ridge(alpha=1.0):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def _rf(max_depth=None, min_samples_leaf=5, max_features=1.0):
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _xgb(max_depth=2, n_estimators=200, learning_rate=0.05):
    return XGBRegressor(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def _lgbm(num_leaves=4, min_child_samples=10, n_estimators=200, learning_rate=0.05):
    return LGBMRegressor(
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=1,
    )


def model_specs() -> list[ModelSpec]:
    """All candidate models with their (deliberately small) hyper-parameter grids.

    Grids are small because the training window has only 52-130 rows; a large
    search would over-fit the two validation weeks.
    """
    return [
        ModelSpec("Persistence (naive)", "baseline", PersistenceForecaster,
                  note="next week = this week"),
        ModelSpec("Moving average (2 wk)", "baseline", MovingAverageForecaster,
                  note="mean of weeks t and t-1"),
        ModelSpec("National-trend naive", "baseline", NationalTrendForecaster,
                  note="this week x last week's national growth"),
        ModelSpec("Ridge regression", "linear", _ridge,
                  grid=[{"alpha": a} for a in (0.3, 1.0, 3.0, 10.0, 30.0)],
                  note="linear benchmark on standardised features"),
        ModelSpec("Random Forest", "bagging", _rf, tree_based=True,
                  grid=[{"max_depth": d, "min_samples_leaf": l, "max_features": f}
                        for d in (3, None) for l in (3, 8) for f in (0.5, 1.0)]),
        ModelSpec("XGBoost", "boosting", _xgb, tree_based=True,
                  grid=[{"max_depth": d, "n_estimators": n, "learning_rate": lr}
                        for d in (2, 3) for n in (100, 300) for lr in (0.03, 0.1)]),
        ModelSpec("LightGBM", "boosting", _lgbm, tree_based=True,
                  grid=[{"num_leaves": nl, "min_child_samples": mc, "n_estimators": n}
                        for nl in (4, 8) for mc in (5, 10) for n in (100, 300)]),
    ]
