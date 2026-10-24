"""Project-wide configuration: paths, constants and experiment settings.

Everything that controls an experiment lives here so a run is reproducible
and every decision (e.g. which weeks are excluded and why) is visible in one place.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "disease-surveillance-2026-10-10.xlsx"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
RESULTS = REPORTS / "results"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Dataset facts (taken from the workbook's "About" sheet)
# --------------------------------------------------------------------------- #
NATIONAL = "Sri Lanka"
# Weekly figures are weeks 24-33 of 2026; a week is dated by its last day.
FIRST_WEEK_DATE = pd.Timestamp("2026-06-14")
FIRST_EPI_WEEK = 24

# --------------------------------------------------------------------------- #
# Data-quality decisions (see docs/dataset_limitations.md)
# --------------------------------------------------------------------------- #
# Week 24 (2026-06-14): national total is consistent, but the division-level
# pattern has ~0 correlation with every other week (adjacent weeks: r > 0.9),
# e.g. Colombo 20 -> 1138, Kilinochchi 419 -> 6. Treated as unreliable.
# Week 26 (2026-06-28): the source note says Vol. 53 No. 27 re-printed week 25's
# dengue counts in all 26 divisions. The true week-26 counts are unknown.
UNRELIABLE_DENGUE_WEEKS: tuple[pd.Timestamp, ...] = (
    pd.Timestamp("2026-06-14"),
    pd.Timestamp("2026-06-28"),
)

# --------------------------------------------------------------------------- #
# Evaluation design (rolling-origin / expanding window, time order preserved)
# A forecast "origin" t uses information up to week t and predicts week t+1.
# --------------------------------------------------------------------------- #
VALIDATION_ORIGINS = (pd.Timestamp("2026-07-19"), pd.Timestamp("2026-07-26"))  # -> wk 30, 31
TEST_ORIGINS = (pd.Timestamp("2026-08-02"), pd.Timestamp("2026-08-09"))        # -> wk 32, 33

# Features fed to the ML models (all are known at forecast time t).
FEATURES = [
    "log_cases_t",
    "growth_1w",
    "log_hist_weekly_mean",
    "intensity_vs_hist",
    "log_ytd_weekly_rate",
    "cases_vs_ytd_rate",
    "log_lepto_t",
    "national_growth_1w",
]

FEATURE_DESCRIPTIONS = {
    "log_cases_t": "log(1 + dengue cases notified in the current week t)",
    "growth_1w": "log-growth from week t-1 to week t (0 if week t-1 is unavailable)",
    "log_hist_weekly_mean": "log(1 + mean weekly cases, from the 2023-2025 full-year totals / 52)",
    "intensity_vs_hist": "log_cases_t minus log_hist_weekly_mean (how unusual this week is)",
    "log_ytd_weekly_rate": "log(1 + 2026 year-to-date cases / epidemiological week number)",
    "cases_vs_ytd_rate": "log_cases_t minus log_ytd_weekly_rate (above/below this year's average)",
    "log_lepto_t": "log(1 + leptospirosis cases in week t), a co-reported rain-linked disease",
    "national_growth_1w": "log-growth of the Sri Lanka national dengue total from t-1 to t",
}
