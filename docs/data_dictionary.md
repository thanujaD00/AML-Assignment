# Data dictionary

## Raw file: `data/raw/disease-surveillance-2026-10-10.xlsx`

Sheet **Data** (1,242 rows, long format):

| Column | Type | Meaning |
|---|---|---|
| `date` | date | Last day of the reporting week. For full-year totals, the last day of week 52. |
| `indicator` | text | `"<Disease>, <measure>: <place>"`, e.g. `Dengue, cases notified in the week: Colombo` |
| `value` | integer | Number of notified cases |
| `unit` | text | Always `cases` |
| `geography` | text | Always `LK` (ISO 3166-1 alpha-2) |
| `status` | text | Source flag. Always `unflagged`. |
| `source` | text | Source record id (`epid-weekly-epidemiological-report`) |

Sheet **About** holds the metadata, licence, citation and source notes.

Parsed from `indicator`:

| Field | Values |
|---|---|
| `disease` | `dengue`, `leptospirosis` |
| `measure` | `week` (cases in the week), `ytd` (2026 year to date), `full_year` (weeks 1-52 of 2023/2024/2025) |
| `place` | 26 RDHS divisions (Ampara RDHS, Anuradhapura, Badulla, Batticaloa, Colombo, Galle, Gampaha, Hambantota, Jaffna, Kalmunai RDHS, Kalutara, Kandy, Kegalle, Kilinochchi, Kurunegala, Mannar, Matale, Matara, Moneragala, Mullaitivu, Nuwara Eliya, Polonnaruwa, Puttalam, Ratnapura, Trincomalee, Vavuniya) and `Sri Lanka` |

## Processed tables (created by the pipeline in `data/processed/`)

### `weekly_panel.csv`: one row per place and week

| Column | Meaning |
|---|---|
| `place`, `date` | Division (or `Sri Lanka`) and week-ending date |
| `epi_week` | Epidemiological week number (24-33) |
| `dengue_week`, `dengue_ytd` | Dengue cases in the week / 2026 year to date |
| `leptospirosis_week`, `leptospirosis_ytd` | Same for leptospirosis |
| `dengue_week_reliable` | False for weeks 24 and 26 (see `dataset_limitations.md`) |

### `annual_totals.csv`
`dengue_2023`, `dengue_2024`, `dengue_2025`: full-year dengue cases per place.

### `supervised_table.csv`: one row per division and forecast origin week *t*

| Column | Meaning |
|---|---|
| `origin_date` / `target_date` | Week *t* (information cut-off) / week *t+1* (forecast week) |
| `cases_t`, `cases_t_minus_1` | Dengue cases in weeks *t* and *t-1* (NaN if *t-1* is unreliable) |
| `log_cases_t` | log(1 + cases_t) |
| `growth_1w` | log1p(cases_t) - log1p(cases_t-1); 0 when *t-1* is unavailable |
| `log_hist_weekly_mean` | log(1 + mean of 2023-2025 full-year totals / 52) |
| `intensity_vs_hist` | `log_cases_t - log_hist_weekly_mean` |
| `log_ytd_weekly_rate` | log(1 + YTD_t / epi_week_t), the average weekly count so far in 2026 |
| `cases_vs_ytd_rate` | `log_cases_t - log_ytd_weekly_rate` |
| `log_lepto_t` | log(1 + leptospirosis cases in week *t*) |
| `national_growth_1w` | Log growth of the national dengue total from *t-1* to *t* |
| `cases_next` | **Actual** dengue cases in week *t+1* |
| `target_log_growth` | **Model target**: log1p(cases_next) - log1p(cases_t) |

All features use only information published by the end of week *t*.
