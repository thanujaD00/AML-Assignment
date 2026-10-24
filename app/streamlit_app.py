"""Small demo UI: next-week dengue forecast per RDHS division, with a SHAP explanation.

Run from the project root (after `python scripts/run_pipeline.py`):
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dengue import config  # noqa: E402
from dengue.data import load_data  # noqa: E402
from dengue.explain import shap_values  # noqa: E402
from dengue.features import build_supervised, growth_to_cases  # noqa: E402

st.set_page_config(page_title="Sri Lanka Dengue Forecast", layout="wide")


@st.cache_resource
def load_artifacts():
    final_path = config.MODELS_DIR / "final_model.joblib"
    if not final_path.exists():
        st.error("No trained model found. Run `python scripts/run_pipeline.py` first.")
        st.stop()
    data = load_data()
    sup = build_supervised(data, require_target=False)
    return data, sup, joblib.load(final_path), joblib.load(config.MODELS_DIR / "explain_model.joblib")


data, sup, final, explain = load_artifacts()
model, feats = final["model"], final["features"]

st.title("Sri Lankan dengue: next-week case forecast by RDHS division")
st.caption(
    "Data: Epidemiology Unit, Ministry of Health, Weekly Epidemiological Report (Table 1), 2026 weeks 24-33. "
    "This is a model estimate trained on a very short period - not an official forecast."
)

c1, c2 = st.columns([1, 1])
place = c1.selectbox("RDHS division", sorted(sup.place.unique()), index=sorted(sup.place.unique()).index("Colombo"))
origins = sorted(sup.origin_date.unique())
origin = c2.selectbox("Forecast from week ending", origins, index=len(origins) - 1,
                      format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"))

row = sup[(sup.place == place) & (sup.origin_date == origin)].iloc[0].copy()
what_if = st.slider("What-if: cases notified this week", 0, int(max(50, row.cases_t * 3)), int(row.cases_t))
if what_if != int(row.cases_t):  # recompute the features that depend on this week's count
    new_log = np.log1p(what_if)
    delta = new_log - row.log_cases_t
    row["log_cases_t"] = new_log
    row["cases_t"] = what_if
    for f in ("growth_1w", "intensity_vs_hist", "cases_vs_ytd_rate"):
        row[f] += delta

X = pd.DataFrame([row[feats].astype(float)])
growth = float(model.predict(X)[0])
forecast = float(growth_to_cases([row.cases_t], [growth])[0])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Cases this week", f"{int(row.cases_t):,}")
m2.metric(f"Forecast ({final['name']})", f"{forecast:,.0f}", f"{100 * (np.expm1(growth)):+.0f}% vs this week")
m3.metric("Persistence baseline", f"{int(row.cases_t):,}")
actual = row.cases_next
m4.metric("Actual next week", "not yet in dataset" if pd.isna(actual) else f"{int(actual):,}")

left, right = st.columns([1.1, 1])
with left:
    st.subheader("History")
    hist = data.weekly[data.weekly.place == place][["date", "dengue_week", "dengue_week_reliable"]]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ok = hist[hist.dengue_week_reliable]
    ax.plot(ok.date, ok.dengue_week, marker="o", color="#2a78d6", label="Notified (reliable weeks)")
    ax.scatter(hist[~hist.dengue_week_reliable].date, hist[~hist.dengue_week_reliable].dengue_week,
               color="#e34948", marker="x", label="Flagged week (not used)")
    ax.scatter([row.target_date], [forecast], color="#eb6834", marker="s", s=60, zorder=5, label="Forecast")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    st.pyplot(fig)

with right:
    st.subheader(f"Why? SHAP explanation ({explain['name']})")
    bg = sup.dropna(subset=["target_log_growth"])[feats]
    sv = shap_values(explain["model"], X, background=bg)
    fig = plt.figure()
    shap.plots.waterfall(sv[0], max_display=len(feats), show=False)
    st.pyplot(plt.gcf())
    st.caption("Bars show how each feature moves the predicted log growth (next week vs this week) "
               "away from the average. Associations learned from 2026 weeks 25-33, not causes.")

with st.expander("Feature definitions"):
    st.table(pd.DataFrame.from_dict(config.FEATURE_DESCRIPTIONS, orient="index", columns=["meaning"]))
