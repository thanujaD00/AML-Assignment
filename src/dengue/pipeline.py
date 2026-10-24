"""End-to-end experiment: data checks -> EDA -> tuning -> comparison -> explainability -> forecast.

Run with ``python scripts/run_pipeline.py``. Every table and figure in the
report is written to ``reports/`` by this module, so results are reproducible
from the raw workbook alone.
"""

from __future__ import annotations

import json
import logging
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import config, plots
from .data import load_data, quality_report
from .evaluation import rolling_origin_predict, summarise, tune
from .explain import lime_explain, shap_values
from .features import build_supervised, growth_to_cases
from .models import MeanEnsemble, ModelSpec, model_specs

log = logging.getLogger("dengue")
warnings.filterwarnings("ignore", category=UserWarning)

LOCAL_CASE = "Colombo"  # division explained with LIME / SHAP waterfall


def _dirs():
    for p in (config.FIGURES, config.RESULTS, config.MODELS_DIR, config.DATA_PROCESSED):
        p.mkdir(parents=True, exist_ok=True)


def _evaluate_specs(sup: pd.DataFrame, specs: list[ModelSpec]):
    """Tune each spec on validation origins, then evaluate on test origins."""
    tuned, val_rows, test_rows, test_preds = {}, [], [], []
    for spec in specs:
        params, table = tune(spec, sup, config.VALIDATION_ORIGINS)
        if len(spec.grid) > 1:
            table.assign(params=table["params"].astype(str)).to_csv(
                config.RESULTS / f"tuning_{spec.name.split(' (')[0].lower().replace(' ', '_')}.csv", index=False)
        tuned[spec.name] = (spec, params)
        val = summarise(rolling_origin_predict(sup, lambda: spec.build(**params), config.VALIDATION_ORIGINS))
        preds = rolling_origin_predict(sup, lambda: spec.build(**params), config.TEST_ORIGINS)
        preds["model"] = spec.name
        test_preds.append(preds)
        val_rows.append({"model": spec.name, "family": spec.family, **val})
        test_rows.append({"model": spec.name, "family": spec.family, **summarise(preds), "params": json.dumps(params)})
        log.info("%-24s val MAE %.2f | test MAE %.2f", spec.name, val["MAE"], test_rows[-1]["MAE"])
    return tuned, pd.DataFrame(val_rows), pd.DataFrame(test_rows), pd.concat(test_preds, ignore_index=True)


def _ensemble_spec(tuned) -> ModelSpec:
    members = [(n, tuned[n][0].build(**tuned[n][1])) for n in ("Random Forest", "XGBoost", "LightGBM")]
    return ModelSpec("Ensemble (RF+XGB+LGBM mean)", "ensemble", lambda: MeanEnsemble(members), tree_based=True,
                     note="average of the three tuned tree models")


def run(make_figures: bool = True) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _dirs()

    # ---------------- 1. Data + quality checks ----------------
    data = load_data()
    q = quality_report(data)
    json.dump(q["overview"], open(config.RESULTS / "data_overview.json", "w"), indent=2)
    for k in ("national_sum_check", "ytd_consistency_check", "adjacent_week_correlation"):
        q[k].to_csv(config.RESULTS / f"quality_{k}.csv", index=False)
    data.weekly.to_csv(config.DATA_PROCESSED / "weekly_panel.csv", index=False)
    data.annual.to_csv(config.DATA_PROCESSED / "annual_totals.csv", index=False)

    sup = build_supervised(data)
    sup.to_csv(config.DATA_PROCESSED / "supervised_table.csv", index=False)
    log.info("Supervised table: %d rows (%d origins x %d divisions)", len(sup), sup.origin_date.nunique(), sup.place.nunique())

    # ---------------- 2. EDA figures ----------------
    if make_figures:
        plots.national_trend(data.weekly)
        plots.division_heatmap(data.weekly)
        plots.annual_totals(data.annual)
        plots.week24_diagnostic(data.weekly)
        plots.distributions(sup)
        plots.dengue_vs_lepto(data.weekly)
        plots.feature_correlation(sup, config.FEATURES)

    # ---------------- 3. Tune + compare ----------------
    specs = model_specs()
    tuned, val_tab, test_tab, test_preds = _evaluate_specs(sup, specs)
    ens = _ensemble_spec(tuned)
    t2, v2, te2, p2 = _evaluate_specs(sup, [ens])
    tuned.update(t2)
    val_tab = pd.concat([val_tab, v2], ignore_index=True)
    test_tab = pd.concat([test_tab, te2], ignore_index=True)
    test_preds = pd.concat([test_preds, p2], ignore_index=True)

    # Selection uses VALIDATION performance only (the test weeks are never used to choose).
    candidates = val_tab[val_tab.family != "baseline"]
    selected = candidates.sort_values("MAE").iloc[0]["model"]
    val_tab.round(3).to_csv(config.RESULTS / "model_comparison_validation.csv", index=False)
    test_tab.round(3).to_csv(config.RESULTS / "model_comparison_test.csv", index=False)
    test_preds.to_csv(config.RESULTS / "test_predictions_all_models.csv", index=False)
    per_week = (
        test_preds.groupby(["model", "target_date"])
        .apply(lambda g: pd.Series({"MAE": (g.cases_next - g.pred_cases).abs().mean(),
                                    "actual_total": g.cases_next.sum(), "forecast_total": g.pred_cases.sum()}),
               include_groups=False)
        .round(1).reset_index()
    )
    per_week.to_csv(config.RESULTS / "test_metrics_per_week.csv", index=False)
    log.info("Selected (lowest validation MAE among ML models): %s", selected)

    sel_spec, sel_params = tuned[selected]
    sel_preds = test_preds[test_preds.model == selected]

    # ---------------- 4. Robustness: seeds + as-printed sensitivity ----------------
    seed_rows = []
    for name in ("Random Forest", "XGBoost", "LightGBM"):
        spec, params = tuned[name]
        for seed in range(5):
            def mk(spec=spec, params=params, seed=seed):
                m = spec.build(**params)
                m.set_params(random_state=seed)
                return m
            m = summarise(rolling_origin_predict(sup, mk, config.TEST_ORIGINS))
            seed_rows.append({"model": name, "seed": seed, "MAE": m["MAE"], "RMSE": m["RMSE"]})
    seeds = pd.DataFrame(seed_rows)
    seeds.groupby("model")[["MAE", "RMSE"]].agg(["mean", "std", "min", "max"]).round(2).to_csv(
        config.RESULTS / "seed_stability.csv")

    sup_printed = build_supervised(data, use_unreliable=True)
    sens = []
    for name in ("Persistence (naive)", "National-trend naive", "Random Forest", "XGBoost", "LightGBM"):
        spec, params = tuned[name]
        m = summarise(rolling_origin_predict(sup_printed, lambda: spec.build(**params), config.TEST_ORIGINS))
        sens.append({"model": name, "n_rows_available": len(sup_printed), **m})
    pd.DataFrame(sens).round(3).to_csv(config.RESULTS / "sensitivity_as_printed.csv", index=False)

    # ---------------- 5. Final model for explanation (train < last test origin) ----------------
    last_origin = config.TEST_ORIGINS[-1]
    train = sup[sup.origin_date < last_origin]
    test_last = sup[sup.origin_date == last_origin]
    explain_spec = sel_spec if sel_spec.tree_based else tuned["LightGBM"][0]
    explain_params = sel_params if sel_spec.tree_based else tuned["LightGBM"][1]
    explain_name = explain_spec.name
    model = explain_spec.build(**explain_params).fit(train[config.FEATURES], train["target_log_growth"])

    X_all = pd.concat([train, test_last])[config.FEATURES]
    sv = shap_values(model, X_all, background=train[config.FEATURES])
    shap_df = pd.DataFrame(sv.values, columns=config.FEATURES)
    imp = shap_df.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").round(4)
    imp.to_csv(config.RESULTS / "shap_global_importance.csv")

    sv_test = sv[len(train):]
    idx = list(test_last.place).index(LOCAL_CASE)
    row = test_last.iloc[idx]
    lime_exp = lime_explain(model, train[config.FEATURES], row[config.FEATURES])
    pred_growth = float(model.predict(test_last[config.FEATURES].iloc[[idx]])[0])
    local = {
        "division": LOCAL_CASE,
        "model": explain_name,
        "origin_week_ending": row.origin_date.date().isoformat(),
        "forecast_week_ending": row.target_date.date().isoformat(),
        "cases_this_week": int(row.cases_t),
        "predicted_log_growth": round(pred_growth, 4),
        "predicted_cases": round(float(growth_to_cases([row.cases_t], [pred_growth])[0]), 1),
        "actual_cases": int(row.cases_next),
        "shap_base_value": round(float(np.ravel(sv_test.base_values)[idx]), 4),
        "shap_contributions": {f: round(float(v), 4) for f, v in zip(config.FEATURES, sv_test.values[idx])},
        "lime_weights": [(f, round(float(w), 4)) for f, w in lime_exp.as_list()],
        "lime_local_r2": round(float(lime_exp.score), 3),
        "lime_local_prediction": round(float(np.ravel(lime_exp.local_pred)[0]), 4),
    }
    json.dump(local, open(config.RESULTS / "local_explanation_colombo.json", "w"), indent=2)

    if make_figures:
        plots.model_comparison(test_tab.set_index("model"), selected)
        plots.predicted_vs_actual(sel_preds, selected)
        plots.residuals(sel_preds, selected)
        sup_hist = build_supervised(data, require_target=False)
        plots.division_forecast_panels(sup_hist, sel_preds,
                                       ["Colombo", "Gampaha", "Kandy", "Kalutara", "Jaffna", "Batticaloa"])

        plt.figure()
        shap.plots.beeswarm(sv, max_display=len(config.FEATURES), show=False)
        plt.title(f"SHAP summary - {explain_name}\n(impact on predicted log growth of next-week cases)", fontsize=10)
        plots._save(plt.gcf(), "12_shap_beeswarm.png")
        plt.figure()
        shap.plots.bar(sv, max_display=len(config.FEATURES), show=False)
        plt.title("Global importance: mean |SHAP value|", fontsize=10)
        plots._save(plt.gcf(), "13_shap_bar.png")
        for k, feat in enumerate(imp.index[:2]):
            plt.figure()
            shap.plots.scatter(sv[:, feat], color=sv, show=False)
            plt.title(f"SHAP dependence: {feat}", fontsize=10)
            plots._save(plt.gcf(), f"14_shap_dependence_{k + 1}_{feat}.png")
        plt.figure()
        shap.plots.waterfall(sv_test[idx], max_display=len(config.FEATURES), show=False)
        plt.title(f"SHAP local explanation - {LOCAL_CASE}, forecast for week ending {row.target_date:%d %b %Y}", fontsize=10)
        plots._save(plt.gcf(), "15_shap_waterfall_colombo.png")
        fig = lime_exp.as_pyplot_figure()
        fig.set_size_inches(8, 4.5)
        plt.title(f"LIME local explanation - {LOCAL_CASE} (local R2 = {lime_exp.score:.2f})", fontsize=10)
        plots._save(fig, "16_lime_colombo.png")
        lime_exp.save_to_file(str(config.FIGURES / "16_lime_colombo.html"))

    # ---------------- 6. Final fit on all data + genuine forward forecast ----------------
    final = sel_spec.build(**sel_params).fit(sup[config.FEATURES], sup["target_log_growth"])
    sup_fc = build_supervised(data, require_target=False)
    latest = sup_fc[sup_fc.origin_date == sup_fc.origin_date.max()].copy()
    latest["forecast_cases"] = growth_to_cases(latest["cases_t"], final.predict(latest[config.FEATURES])).round(1)
    latest["persistence_forecast"] = latest["cases_t"]
    fc = latest[["place", "origin_date", "target_date", "cases_t", "persistence_forecast", "forecast_cases"]]
    fc.to_csv(config.RESULTS / "forecast_next_week.csv", index=False)

    # Model used by the Streamlit demo (+ the explanation model if different)
    joblib.dump({"model": final, "name": selected, "features": config.FEATURES, "params": sel_params},
                config.MODELS_DIR / "final_model.joblib")
    explain_final = explain_spec.build(**explain_params).fit(sup[config.FEATURES], sup["target_log_growth"])
    joblib.dump({"model": explain_final, "name": explain_name, "features": config.FEATURES},
                config.MODELS_DIR / "explain_model.joblib")

    summary = {
        "selected_model": selected,
        "selected_params": sel_params,
        "explained_model": explain_name,
        "n_supervised_rows": len(sup),
        "validation": val_tab.round(3).to_dict("records"),
        "test": test_tab.drop(columns="params").round(3).to_dict("records"),
        "shap_importance": imp.to_dict(),
        "forecast_origin": str(latest.origin_date.max().date()),
        "forecast_target": str(latest.target_date.max().date()),
        "national_forecast_sum": float(fc.forecast_cases.sum()),
    }
    json.dump(summary, open(config.RESULTS / "summary.json", "w"), indent=2, default=str)
    _write_results_md(summary, val_tab, test_tab, imp, local, seeds, pd.DataFrame(sens), fc)
    return summary


def _md_table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def _write_results_md(summary, val_tab, test_tab, imp, local, seeds, sens, fc):
    cols = ["model", "family", "MAE", "RMSE", "WAPE_%", "R2", "MAE_log", "Skill_vs_persistence_%", "National_APE_%"]
    seed_tab = seeds.groupby("model")["MAE"].agg(["mean", "std", "min", "max"]).reset_index()
    text = f"""# Results (auto-generated by `scripts/run_pipeline.py`)

Do not edit by hand - re-run the pipeline to regenerate.

**Selected model:** {summary['selected_model']} (lowest *validation* MAE among the ML models)
- Parameters: `{json.dumps(summary['selected_params'])}`
- Supervised rows: {summary['n_supervised_rows']} (26 divisions x 6 forecast origins)

## Validation (forecasts for weeks 30-31; used for tuning and model selection)
{_md_table(val_tab[cols])}

## Test (forecasts for weeks 32-33; never used for any choice)
{_md_table(test_tab[cols])}

## Seed stability of the tree models on the test weeks (MAE, 5 seeds)
{_md_table(seed_tab)}

## Sensitivity: data used exactly as printed (weeks 24 and 26 kept)
{_md_table(sens[['model', 'MAE', 'RMSE', 'R2', 'Skill_vs_persistence_%']])}

## SHAP global importance ({summary['explained_model']})
{_md_table(imp.reset_index().rename(columns={'index': 'feature'}), '.4f')}

## Local explanation: {local['division']}, forecast for week ending {local['forecast_week_ending']}
- Cases this week: {local['cases_this_week']} | forecast: {local['predicted_cases']} | actual: {local['actual_cases']}
- SHAP contributions (log-growth units): {local['shap_contributions']}
- LIME weights: {local['lime_weights']} (local R2 = {local['lime_local_r2']})

## Forward forecast (no actual value in the dataset yet)
Origin week ending {summary['forecast_origin']} -> forecast for week ending {summary['forecast_target']}.
Sum of division forecasts: {summary['national_forecast_sum']:.0f} cases.

{_md_table(fc.assign(origin_date=fc.origin_date.dt.date, target_date=fc.target_date.dt.date), '.1f')}
"""
    (config.RESULTS / "RESULTS.md").write_text(text, encoding="utf-8")
