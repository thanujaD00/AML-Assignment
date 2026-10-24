"""Time-aware evaluation: rolling-origin (expanding-window) forecasting.

For each forecast origin ``o`` the model is trained only on rows whose origin
is *earlier* than ``o`` (so every training target is already observed by week
``o``) and then predicts week ``o + 1`` for all 26 divisions. No information
from the evaluation week can leak into training.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import FEATURES
from .features import growth_to_cases


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Metrics on the case-count scale plus a log-scale error.

    * MAE / RMSE  - average error in cases (RMSE punishes large misses, e.g. Colombo)
    * WAPE        - sum |error| / sum actual: error as a share of all cases
    * R2          - variance of next-week counts explained
    * MAE_log     - error on log1p scale: treats small and large divisions alike
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "WAPE_%": 100 * np.abs(y_true - y_pred).sum() / y_true.sum(),
        "R2": r2_score(y_true, y_pred),
        "MAE_log": mean_absolute_error(np.log1p(y_true), np.log1p(y_pred)),
    }


def rolling_origin_predict(
    sup: pd.DataFrame,
    make_model: Callable[[], Any],
    origins: Iterable[pd.Timestamp],
    features: list[str] = FEATURES,
) -> pd.DataFrame:
    """Expanding-window forecasts for each origin; returns one row per prediction."""
    out = []
    for o in origins:
        train = sup[sup["origin_date"] < o]
        test = sup[sup["origin_date"] == o]
        if test.empty:
            raise ValueError(f"No rows for origin {o.date()}")
        model = make_model()
        model.fit(train[features], train["target_log_growth"])
        g = model.predict(test[features])
        pred = test[["place", "origin_date", "target_date", "cases_t", "cases_next"]].copy()
        pred["pred_growth"] = g
        pred["pred_cases"] = growth_to_cases(test["cases_t"], g)
        pred["n_train"] = len(train)
        out.append(pred)
    return pd.concat(out, ignore_index=True)


def tune(spec, sup: pd.DataFrame, origins, features: list[str] = FEATURES):
    """Grid search over ``spec.grid`` using rolling-origin MAE on the validation origins."""
    records = []
    for params in spec.grid:
        preds = rolling_origin_predict(sup, lambda: spec.build(**params), origins, features)
        m = regression_metrics(preds["cases_next"], preds["pred_cases"])
        records.append({"params": params, **m})
    table = pd.DataFrame(records).sort_values(["MAE", "RMSE"]).reset_index(drop=True)
    return table.loc[0, "params"], table


def summarise(preds: pd.DataFrame) -> dict[str, float]:
    """Pooled metrics, plus skill relative to persistence on the same rows."""
    m = regression_metrics(preds["cases_next"], preds["pred_cases"])
    naive_mae = np.abs(preds["cases_next"] - preds["cases_t"]).mean()
    m["Skill_vs_persistence_%"] = 100 * (1 - m["MAE"] / naive_mae)
    # National view: sum of division forecasts vs. sum of division actuals per week
    nat = preds.groupby("target_date")[["cases_next", "pred_cases"]].sum()
    m["National_APE_%"] = float((100 * (nat["pred_cases"] - nat["cases_next"]).abs() / nat["cases_next"]).mean())
    return m
