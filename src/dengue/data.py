"""Loading, tidying and quality-checking the Epidemiology Unit surveillance data.

The workbook (compiled by research.dulajmendis.com from the Epidemiology Unit's
Weekly Epidemiological Report, Table 1) is in *long* format: one row per
(date, indicator) where the indicator string encodes disease, measure and place,
e.g. ``"Dengue, cases notified in the week: Colombo"``.

Nothing in this module changes a value. Unreliable weeks are *flagged* and the
modelling code decides whether to use them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

MEASURE_MAP = {
    "cases notified in the week": "week",
    "cases notified this year to date": "ytd",
    "cases notified in weeks 1 to 52": "full_year",
}


def load_raw(path: Path | str = config.DATA_RAW) -> pd.DataFrame:
    """Read the ``Data`` sheet and split the indicator into disease / measure / place."""
    df = pd.read_excel(path, sheet_name="Data")
    df["date"] = pd.to_datetime(df["date"])
    parts = df["indicator"].str.extract(r"^(?P<disease>[^,]+), (?P<measure>.+?): (?P<place>.+)$")
    if parts.isna().any().any():
        bad = df.loc[parts.isna().any(axis=1), "indicator"].unique()
        raise ValueError(f"Unparseable indicator strings: {bad[:5]}")
    df = pd.concat([df, parts], axis=1)
    df["disease"] = df["disease"].str.lower()
    df["measure"] = df["measure"].map(MEASURE_MAP)
    if df["measure"].isna().any():
        raise ValueError("Unknown measure label in indicator column")
    return df


def load_about(path: Path | str = config.DATA_RAW) -> pd.DataFrame:
    """Return the workbook's ``About`` sheet (metadata, source notes, licence)."""
    about = pd.read_excel(path, sheet_name="About", header=None, names=["field", "value"])
    return about


@dataclass
class SurveillanceData:
    """Tidy views of the raw data.

    weekly : one row per (place, date) with dengue/leptospirosis weekly and YTD counts.
    annual : one row per place with full-year dengue totals for 2023, 2024, 2025.
    raw    : the parsed long-format table.
    """

    weekly: pd.DataFrame
    annual: pd.DataFrame
    raw: pd.DataFrame

    @property
    def divisions(self) -> list[str]:
        return sorted(p for p in self.weekly["place"].unique() if p != config.NATIONAL)


def build_tables(raw: pd.DataFrame) -> SurveillanceData:
    """Pivot the long table into a weekly panel and an annual-totals table."""
    wk = raw[raw["measure"].isin(["week", "ytd"])].copy()
    wk["col"] = wk["disease"] + "_" + wk["measure"]
    weekly = (
        wk.pivot_table(index=["place", "date"], columns="col", values="value", aggfunc="first")
        .reset_index()
        .rename_axis(columns=None)
        .sort_values(["place", "date"])
        .reset_index(drop=True)
    )
    weekly["epi_week"] = (
        config.FIRST_EPI_WEEK + (weekly["date"] - config.FIRST_WEEK_DATE).dt.days // 7
    ).astype(int)
    weekly["dengue_week_reliable"] = ~weekly["date"].isin(config.UNRELIABLE_DENGUE_WEEKS)

    fy = raw[(raw["measure"] == "full_year") & (raw["disease"] == "dengue")].copy()
    fy["year"] = fy["date"].dt.year
    annual = fy.pivot_table(index="place", columns="year", values="value", aggfunc="first")
    annual.columns = [f"dengue_{c}" for c in annual.columns]
    annual = annual.reset_index()
    return SurveillanceData(weekly=weekly, annual=annual, raw=raw)


def load_data(path: Path | str = config.DATA_RAW) -> SurveillanceData:
    """Convenience wrapper: read the workbook and build the tidy tables."""
    return build_tables(load_raw(path))


# --------------------------------------------------------------------------- #
# Data-quality checks
# --------------------------------------------------------------------------- #
def quality_report(data: SurveillanceData) -> dict[str, pd.DataFrame | dict]:
    """Run the integrity checks used to justify the preprocessing decisions.

    Returns a dict of small tables that are saved to ``reports/results``.
    """
    raw, weekly, annual = data.raw, data.weekly, data.annual
    out: dict[str, pd.DataFrame | dict] = {}

    out["overview"] = {
        "rows": int(len(raw)),
        "indicators": int(raw["indicator"].nunique()),
        "places": int(raw["place"].nunique()),
        "divisions_excl_national": len(data.divisions),
        "weekly_dates": [d.date().isoformat() for d in sorted(weekly["date"].unique())],
        "full_year_dates": [d.date().isoformat() for d in sorted(raw.loc[raw.measure == "full_year", "date"].unique())],
        "missing_values": int(raw["value"].isna().sum()),
        "duplicate_rows": int(raw.duplicated(["date", "indicator"]).sum()),
        "negative_values": int((raw["value"] < 0).sum()),
        "status_values": raw["status"].value_counts().to_dict(),
    }

    # 1) Divisions sum to the national figure?
    rows = []
    for col in ["dengue_week", "dengue_ytd", "leptospirosis_week", "leptospirosis_ytd"]:
        piv = weekly.pivot(index="place", columns="date", values=col)
        diff = piv.drop(index=config.NATIONAL).sum() - piv.loc[config.NATIONAL]
        rows.append({"series": col, "max_abs_gap_divisions_vs_national": int(diff.abs().max())})
    out["national_sum_check"] = pd.DataFrame(rows)

    # 2) YTD(t) - YTD(t-1) == week(t)?
    rows = []
    for dis in ["dengue", "leptospirosis"]:
        w = weekly.pivot(index="place", columns="date", values=f"{dis}_week")
        y = weekly.pivot(index="place", columns="date", values=f"{dis}_ytd")
        gap = (y.diff(axis=1) - w).iloc[:, 1:]
        rows.append({"disease": dis, "max_abs_gap_ytd_increment_vs_week": int(gap.abs().max().max())})
    out["ytd_consistency_check"] = pd.DataFrame(rows)

    # 3) Week-to-week correlation of the division pattern (log scale).
    w = weekly[weekly.place != config.NATIONAL].pivot(index="place", columns="date", values="dengue_week")
    lw = np.log1p(w)
    rows = []
    dates = list(lw.columns)
    for a, b in zip(dates[:-1], dates[1:]):
        rows.append(
            {
                "week_a": a.date().isoformat(),
                "week_b": b.date().isoformat(),
                "pearson_r_log_cases": round(float(np.corrcoef(lw[a], lw[b])[0, 1]), 3),
                "identical_in_all_divisions": bool((w[a] == w[b]).all()),
            }
        )
    out["adjacent_week_correlation"] = pd.DataFrame(rows)
    return out
