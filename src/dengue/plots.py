"""Matplotlib figures for EDA, model comparison and explainability.

All functions take tidy DataFrames, draw one figure and save it as PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: works in scripts, notebooks and CI
import matplotlib.dates  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

# Palette (validated categorical order; grey = baselines / context)
BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9890", "#0b0b0b", "#52514e"
FLAG = "#e34948"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#c9c8c2",
        "axes.labelcolor": MUTED,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.grid": True,
        "grid.color": "#ecebe7",
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 10,
        "legend.frameon": False,
    }
)


def _save(fig, name: str, outdir: Path = config.FIGURES) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.savefig(path)
    plt.close(fig)
    return path


def _fmt_dates(ax, dates):
    ax.set_xticks(dates)
    ax.set_xticklabels([f"{d:%d %b}\nwk {config.FIRST_EPI_WEEK + i}" for i, d in enumerate(dates)], fontsize=8)


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #
def national_trend(weekly: pd.DataFrame, outdir=config.FIGURES):
    nat = weekly[weekly.place == config.NATIONAL].sort_values("date")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(nat["date"], nat["dengue_week"], color=BLUE, lw=2, marker="o", ms=6, label="Dengue (as printed)")
    bad = nat[~nat["dengue_week_reliable"]]
    ax.scatter(bad["date"], bad["dengue_week"], s=120, facecolors="none", edgecolors=FLAG, lw=2,
               label="Flagged week (excluded from modelling)", zorder=5)
    for _, r in nat.iterrows():
        ax.annotate(f"{int(r.dengue_week):,}", (r.date, r.dengue_week), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=MUTED)
    ax.set_ylabel("Cases notified in the week")
    ax.set_title("Sri Lanka - weekly dengue notifications, weeks 24-33 of 2026")
    _fmt_dates(ax, list(nat["date"]))
    ax.set_ylim(0, nat["dengue_week"].max() * 1.15)
    ax.legend(loc="upper right")
    return _save(fig, "01_national_weekly_trend.png", outdir)


def division_heatmap(weekly: pd.DataFrame, outdir=config.FIGURES):
    w = weekly[weekly.place != config.NATIONAL].pivot(index="place", columns="date", values="dengue_week")
    w = w.loc[w.iloc[:, 1:].sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(np.log10(w + 1), aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(w)), w.index, fontsize=8)
    ax.set_xticks(range(w.shape[1]), [f"{d:%d %b}" for d in w.columns], rotation=45, fontsize=8)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            v = w.iat[i, j]
            ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=6.5,
                    color="white" if np.log10(v + 1) > 2.3 else INK)
    for j, d in enumerate(w.columns):
        if d in config.UNRELIABLE_DENGUE_WEEKS:
            ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, len(w), fill=False, ec=FLAG, lw=2))
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("log10(cases + 1)")
    ax.set_title("Weekly dengue cases by RDHS division (red frame = flagged week)")
    return _save(fig, "02_division_heatmap.png", outdir)


def annual_totals(annual: pd.DataFrame, outdir=config.FIGURES):
    a = annual[annual.place != config.NATIONAL].set_index("place")
    a = a.loc[a.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8, 8))
    y = np.arange(len(a))
    for k, (col, c) in enumerate(zip(a.columns, [BLUE, ORANGE, AQUA])):
        ax.barh(y + (k - 1) * 0.27, a[col], height=0.25, color=c, label=col.replace("dengue_", ""))
    ax.set_yticks(y, a.index, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Dengue cases, weeks 1-52 (log scale)")
    ax.set_title("Full-year dengue totals by RDHS division, 2023-2025")
    ax.legend(title="Year", loc="lower right")
    return _save(fig, "03_annual_totals_by_division.png", outdir)


def week24_diagnostic(weekly: pd.DataFrame, outdir=config.FIGURES):
    w = weekly[weekly.place != config.NATIONAL].pivot(index="place", columns="date", values="dengue_week")
    d = list(w.columns)
    pairs = [(d[0], d[1], "Week 24 vs 25 (flagged)"), (d[1], d[2], "Week 25 vs 26 (duplicate)"),
             (d[4], d[5], "Week 28 vs 29 (typical)")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for ax, (a, b, title) in zip(axes, pairs):
        x, y = np.log1p(w[a]), np.log1p(w[b])
        r = np.corrcoef(x, y)[0, 1]
        ax.scatter(x, y, s=40, color=FLAG if "typical" not in title else BLUE, edgecolor="white", lw=1)
        ax.plot([0, 8], [0, 8], color=GREY, lw=1, ls="--")
        ax.set_title(f"{title}\nr = {r:.2f}", fontsize=10)
        ax.set_xlabel(f"log(1+cases) {a:%d %b}")
    axes[0].set_ylabel("log(1+cases) next week")
    fig.suptitle("Division-level consistency between consecutive weeks", fontweight="bold", y=1.04)
    return _save(fig, "04_week_consistency_diagnostic.png", outdir)


def distributions(sup: pd.DataFrame, outdir=config.FIGURES):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(sup["cases_t"], bins=30, color=BLUE, edgecolor="white")
    axes[0].set_title("Weekly cases per division (raw)")
    axes[0].set_xlabel("Cases in week")
    axes[1].hist(sup["target_log_growth"].dropna(), bins=25, color=AQUA, edgecolor="white")
    axes[1].axvline(0, color=INK, lw=1)
    axes[1].set_title("Target: log growth to next week")
    axes[1].set_xlabel("log1p(cases t+1) - log1p(cases t)")
    for ax in axes:
        ax.set_ylabel("Division-weeks")
    return _save(fig, "05_distributions.png", outdir)


def dengue_vs_lepto(weekly: pd.DataFrame, outdir=config.FIGURES):
    w = weekly[(weekly.place != config.NATIONAL) & weekly.dengue_week_reliable]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(w["leptospirosis_week"], w["dengue_week"], s=25, color=BLUE, alpha=0.7, edgecolor="white", lw=0.5)
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    r = np.corrcoef(np.log1p(w["leptospirosis_week"]), np.log1p(w["dengue_week"]))[0, 1]
    ax.set_xlabel("Leptospirosis cases in week")
    ax.set_ylabel("Dengue cases in week")
    ax.set_title(f"Dengue vs leptospirosis per division-week (log r = {r:.2f})")
    return _save(fig, "06_dengue_vs_leptospirosis.png", outdir)


def feature_correlation(sup: pd.DataFrame, features, outdir=config.FIGURES):
    cols = features + ["target_log_growth"]
    c = sup[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{c.iat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(c.iat[i, j]) > 0.6 else INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Pearson correlation: features and target")
    return _save(fig, "07_feature_correlation.png", outdir)


# --------------------------------------------------------------------------- #
# Model comparison
# --------------------------------------------------------------------------- #
def model_comparison(table: pd.DataFrame, selected: str, outdir=config.FIGURES):
    t = table.sort_values("MAE", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors = [ORANGE if m == selected else (GREY if f == "baseline" else BLUE) for m, f in zip(t.index, t["family"])]
    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        ax.barh(t.index, t[metric], color=colors, height=0.6)
        for i, v in enumerate(t[metric]):
            ax.text(v, i, f" {v:.1f}", va="center", fontsize=8, color=MUTED)
        ax.set_xlabel(f"{metric} (cases, lower is better)")
        ax.grid(axis="y", visible=False)
    axes[0].set_title("Test MAE")
    axes[1].set_title("Test RMSE")
    fig.suptitle("Model comparison - rolling-origin test (forecasts for weeks 32 & 33)\n"
                 "grey = baseline, blue = ML candidate, orange = selected", fontsize=10)
    return _save(fig, "08_model_comparison.png", outdir)


def predicted_vs_actual(preds: pd.DataFrame, model_name: str, outdir=config.FIGURES):
    fig, ax = plt.subplots(figsize=(6, 6))
    for (d, g), c in zip(preds.groupby("target_date"), [BLUE, ORANGE]):
        ax.scatter(g["cases_next"], g["pred_cases"], s=40, color=c, edgecolor="white", lw=1, label=f"Week ending {d:%d %b}")
    lim = max(preds["cases_next"].max(), preds["pred_cases"].max()) * 1.3
    ax.plot([0.8, lim], [0.8, lim], color=GREY, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Actual cases")
    ax.set_ylabel("Forecast cases")
    ax.set_title(f"{model_name}: forecast vs actual (test weeks)")
    ax.legend()
    for _, r in preds.nlargest(3, "cases_next").iterrows():
        ax.annotate(r.place, (r.cases_next, r.pred_cases), fontsize=7, color=MUTED, xytext=(4, -10), textcoords="offset points")
    return _save(fig, "09_predicted_vs_actual.png", outdir)


def residuals(preds: pd.DataFrame, model_name: str, outdir=config.FIGURES):
    p = preds.assign(err=preds.pred_cases - preds.cases_next)
    by = p.groupby("place")["err"].mean().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1, 1.2]})
    axes[0].scatter(np.log1p(p["cases_next"]), p["err"], color=BLUE, s=30, edgecolor="white")
    axes[0].axhline(0, color=INK, lw=1)
    axes[0].set_xlabel("log(1 + actual cases)")
    axes[0].set_ylabel("Forecast - actual (cases)")
    axes[0].set_title("Residuals vs size")
    axes[1].barh(by.index, by.values, color=[ORANGE if v > 0 else BLUE for v in by.values], height=0.7)
    axes[1].axvline(0, color=INK, lw=1)
    axes[1].tick_params(axis="y", labelsize=7)
    axes[1].set_xlabel("Mean error over test weeks (cases)\norange = over-forecast, blue = under-forecast")
    axes[1].set_title("Bias by division")
    fig.suptitle(f"{model_name}: residual analysis", fontweight="bold")
    return _save(fig, "10_residuals.png", outdir)


def division_forecast_panels(sup_all: pd.DataFrame, preds: pd.DataFrame, places, outdir=config.FIGURES):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, place in zip(axes.flat, places):
        h = sup_all[sup_all.place == place]
        ax.plot(h["origin_date"], h["cases_t"], color=INK, marker="o", ms=4, lw=1.5, label="Actual")
        p = preds[preds.place == place]
        ax.plot(p["target_date"], p["pred_cases"], color=ORANGE, marker="s", ms=6, lw=0, label="Forecast")
        ax.set_title(place, fontsize=10)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.set_ylim(bottom=0)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("One-week-ahead test forecasts for selected divisions", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "11_division_forecasts.png", outdir)
