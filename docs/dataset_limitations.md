# Dataset limitations and data-quality decisions

All checks below are reproduced by `dengue.data.quality_report()` and saved to `reports/results/quality_*.csv`.
No value in the raw workbook is modified. Unreliable weeks are only masked inside the modelling code
(`config.UNRELIABLE_DENGUE_WEEKS`).

## 1. What the data are (and are not)

| Item | Detail |
|---|---|
| Source | Epidemiology Unit, Ministry of Health, Sri Lanka: *Weekly Epidemiological Report*, Table 1 |
| Compiled by | research.dulajmendis.com, dataset `disease-surveillance`, version 2026-10-10 |
| Weekly coverage | **10 weeks only**: 2026 epidemiological weeks 24-33 (week ending 14 Jun to 16 Aug 2026), Vol. 53 Nos. 25-34 |
| Annual coverage | Full-year (weeks 1-52) totals for 2023, 2024, 2025 |
| Places | 26 RDHS divisions + Sri Lanka total |
| Measure | Cases **notified** by Medical Officers of Health. This is not the same as infections or confirmed cases. |
| Not included | Rainfall, temperature, humidity, mosquito (vector) indices, population, hospital admissions, deaths |

## 2. Integrity checks that pass

* 1,242 rows, no missing values, no duplicate (date, indicator) pairs, no negative values; every `status` is "unflagged".
* The 26 divisions sum exactly to the national figure for every week and every measure.
* For every division and both diseases, year-to-date(t) - year-to-date(t-1) = cases in week t.

## 3. Problems found and how they are handled

### Week 24 (ending 2026-06-14): division values look misaligned. Excluded.
* The national total (3,973) fits the season curve, but the **pattern across divisions** does not:
  the correlation of log cases with the next week is **r = 0.04**. Every other pair of adjacent reliable weeks has r = 0.95-0.98.
* Examples: Colombo 20 -> 1,138; Gampaha 24 -> 1,294; Kilinochchi 419 -> 6; Mannar 387 -> 5; Matale 889 -> 49.
* The most likely cause is that values were attached to the wrong rows somewhere between the PDF and the compiled table. This could not be confirmed, because the original PDF could not be downloaded from this environment.
* **Decision:** the week-24 dengue counts are not used as a feature or target. They stay in the raw file.
  *Action for the author:* check the printed report (Vol. 53 No. 25). If it shows the same numbers, the anomaly is in the source itself.

### Week 26 (ending 2026-06-28): duplicate of week 25. Excluded.
* The dataset's own source note: *"Vol. 53 No. 27 prints the same dengue counts for week 26 as No. 26 does for week 25, in all 26 divisions; its year-to-date column adds them again."*
* Verified: the two weeks are identical in all 26 divisions.
* **Decision:** treated as missing. The true week-26 counts are unknown, so they are not imputed.
* **Side effect:** 2026 year-to-date figures from week 26 onward count week 25 twice. The feature `log_ytd_weekly_rate` therefore carries a small, roughly constant upward bias. SHAP shows this feature has the lowest importance.

### Origin 2026-07-05 has no previous week
Week 26 is excluded, so for this origin `growth_1w` and `national_growth_1w` are set to 0 ("no change"). This affects 26 of the 156 rows, all of them in the training period.

### Week-numbering convention
Up to 2025 the reports' weeks ran Saturday-Friday; in 2026 they run Monday-Sunday. Totals to week 52 leave out the last days of December. This only affects the annual-total features, and only slightly.

### Vol. 53 Nos. 32-33 date typos
The dataset already corrected misprinted week-ending dates (2026-07-02 -> 2026-08-02, 2026-07-09 -> 2026-08-09). No further action was needed.

## 4. Consequences for the analysis

1. **Small sample.** 156 supervised rows (6 origins x 26 divisions), with 52 test forecasts. Model differences of 1-3 cases MAE are within noise.
2. **Single season phase.** Every evaluated week is in the *declining* part of the 2026 mid-year season. A model that learned "national decline -> lower forecast" may fail at the start of an outbreak.
3. **The `national_growth_1w` feature has only 6 distinct values** (one per week). Its high SHAP importance mostly reflects season phase, learned from very few independent observations.
4. **No seasonality can be learned.** Less than one season of weekly data is available, so no seasonal or calendar terms are used.
5. **No external drivers.** The most important causes of dengue (lagged rainfall and temperature, vector density, population density) are absent. Explanations therefore describe surveillance-count dynamics only.
6. **Reporting artefacts** (holidays, delayed returns, changes in testing) are part of the target and cannot be separated from real changes in transmission.

## 5. Recommended extensions

* Extend the weekly series back to 2010+ from the Weekly Epidemiological Report archive (the same Table 1). This allows proper seasonal models and evaluation across several seasons.
* Add lagged rainfall and temperature per district (Department of Meteorology), plus population from the Department of Census and Statistics, to model incidence rates.
* Forecast 2-4 weeks ahead and report prediction intervals (e.g. quantile LightGBM).

## 6. Ethical notes

* The data are aggregated counts with no personal information.
* A forecast should support resource planning, not replace epidemiological judgement. Under-forecasting the start of an outbreak is the costly error, and that is the situation this model has *not* been tested on.
* Acknowledge the Epidemiology Unit, Ministry of Health, as the primary data source, as the reports request.
