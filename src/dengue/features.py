"""Turn the weekly panel into a supervised next-week forecasting table.

One row = one RDHS division at one forecast origin week *t*.
Features use only information published up to and including week *t*;
the target is the dengue count notified in week *t+1*.

Models learn the **log growth** ``log1p(y[t+1]) - log1p(y[t])`` rather than the
raw count. This makes the problem scale-free across divisions (Colombo reports
~100x more cases than Mullaitivu), anchors every model to the persistence
forecast, and lets tree models generalise across divisions of different size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import SurveillanceData

ONE_WEEK = pd.Timedelta(days=7)


def _reliable_series(weekly: pd.DataFrame, col: str, use_unreliable: bool) -> pd.DataFrame:
    """Pivot a weekly column to place x date, masking unreliable weeks with NaN."""
    piv = weekly.pivot(index="place", columns="date", values=col).astype(float)
    if not use_unreliable and col.startswith("dengue_week"):
        for d in config.UNRELIABLE_DENGUE_WEEKS:
            if d in piv.columns:
                piv[d] = np.nan
    return piv


def build_supervised(
    data: SurveillanceData,
    use_unreliable: bool = False,
    require_target: bool = True,
) -> pd.DataFrame:
    """Build the modelling table.

    Parameters
    ----------
    use_unreliable
        If False (default) the weeks listed in ``config.UNRELIABLE_DENGUE_WEEKS``
        are treated as missing. If True the data are used exactly as printed
        (used only for the sensitivity analysis).
    require_target
        If True only origins with an observed week t+1 are returned (training /
        evaluation). If False rows without a target are kept as well (used to
        produce the genuine forward forecast from the latest week).
    """
    weekly = data.weekly
    cases = _reliable_series(weekly, "dengue_week", use_unreliable)
    ytd = weekly.pivot(index="place", columns="date", values="dengue_ytd").astype(float)
    lepto = weekly.pivot(index="place", columns="date", values="leptospirosis_week").astype(float)
    epi_week = weekly.drop_duplicates("date").set_index("date")["epi_week"]

    ann = data.annual.set_index("place")
    hist_weekly = ann[[c for c in ann.columns if c.startswith("dengue_")]].mean(axis=1) / 52.0

    nat = cases.loc[config.NATIONAL]
    dates = list(cases.columns)
    rows = []
    for place in data.divisions:
        for t in dates:
            y_t = cases.at[place, t]
            if np.isnan(y_t):
                continue  # no reliable current-week count -> cannot forecast
            t_prev, t_next = t - ONE_WEEK, t + ONE_WEEK
            y_prev = cases.at[place, t_prev] if t_prev in cases.columns else np.nan
            y_next = cases.at[place, t_next] if t_next in cases.columns else np.nan
            if require_target and np.isnan(y_next):
                continue
            nat_prev = nat.get(t_prev, np.nan)

            log_t = np.log1p(y_t)
            log_hist = np.log1p(hist_weekly[place])
            log_ytd_rate = np.log1p(ytd.at[place, t] / epi_week[t])
            rows.append(
                {
                    "place": place,
                    "origin_date": t,
                    "target_date": t_next,
                    "origin_epi_week": int(epi_week[t]),
                    "cases_t": y_t,
                    "cases_t_minus_1": y_prev,
                    "log_cases_t": log_t,
                    "growth_1w": 0.0 if np.isnan(y_prev) else log_t - np.log1p(y_prev),
                    "log_hist_weekly_mean": log_hist,
                    "intensity_vs_hist": log_t - log_hist,
                    "log_ytd_weekly_rate": log_ytd_rate,
                    "cases_vs_ytd_rate": log_t - log_ytd_rate,
                    "log_lepto_t": np.log1p(lepto.at[place, t]),
                    "national_growth_1w": 0.0 if np.isnan(nat_prev) else np.log1p(nat[t]) - np.log1p(nat_prev),
                    "cases_next": y_next,
                    "target_log_growth": np.log1p(y_next) - log_t if not np.isnan(y_next) else np.nan,
                }
            )
    sup = pd.DataFrame(rows).sort_values(["origin_date", "place"]).reset_index(drop=True)
    return sup


def growth_to_cases(cases_t: np.ndarray | pd.Series, growth: np.ndarray) -> np.ndarray:
    """Convert a predicted log growth back to a non-negative case count."""
    return np.clip(np.expm1(np.log1p(np.asarray(cases_t, dtype=float)) + np.asarray(growth)), 0, None)
