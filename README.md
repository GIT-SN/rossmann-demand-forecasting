# Retail Demand Forecasting — Rossmann Store Sales

A retail demand forecasting project that predicts store-level sales over a 42-calendar-day horizon using historical sales, calendar features, promotion information, and store metadata.

The project focuses on leakage-aware time-series validation, comparison against historical baselines, transparent holdout evaluation, and an interactive Streamlit dashboard.

## Project Highlights

- Forecast horizon: 42 calendar days
- Target population: Open store-days
- Primary metric: Weighted Absolute Percentage Error (WAPE)
- Main model: HistGradientBoostingRegressor
- Validation: Four expanding-window, rolling-origin folds
- Final evaluation: Separate 42-day holdout
- Dashboard: Store-level predictions, baseline comparison, and promotion error analysis

## Results

### Rolling Validation

The reproducible validation pipeline is the canonical source for validation metrics.

| Metric | Result |
|---|---:|
| Mean model WAPE | 9.92% |
| Best baseline mean WAPE (lag 364 days) | 14.65% |
| Improvement over best baseline | 4.73 percentage points |
| Number of validation folds | 4 |

Validation windows:

1. 2015-01-03 to 2015-02-13
2. 2015-02-14 to 2015-03-27
3. 2015-03-28 to 2015-05-08
4. 2015-05-09 to 2015-06-19

### Final Holdout

The final holdout covers 2015-06-20 to 2015-07-31.

| Metric | Result |
|---|---:|
| Model WAPE on all eligible rows | 9.33% |
| Model WAPE on common comparison rows | 9.12% |
| Lag-364 baseline WAPE on common rows | 11.28% |
| Lag-42 baseline WAPE on common rows | 14.03% |
| Weekday-mean baseline WAPE on common rows | 16.99% |
| Improvement over lag-364 on common rows | 2.17 percentage points |

The all-row model WAPE and common-row baseline comparison use different row populations and should not be compared as if they were calculated on identical observations.

The store-level bootstrap 95% interval for improvement over the lag-364 baseline is approximately 1.93 to 2.40 percentage points, conditional on this holdout period.

## Methodology

### Leakage Prevention

For every target date `t`, sales-derived features use observations no later than `t - 42 days`. The validation folds are chronological and each covers 42 calendar days.

### Features

The feature pipeline includes:

- Historical sales lags, including 42-day and 364-day lags
- Historical same-weekday averages
- Lagged rolling sales summaries
- Calendar features
- Promotion and holiday indicators
- Store metadata

Closed store-days are excluded from the sales target population. Observed open days with zero sales are preserved. Missing historical observations are not automatically imputed as zero sales.

### Baselines

The main model is compared with:

- Sales from 42 days earlier
- Sales from 364 days earlier
- Historical same-weekday average

Baseline comparisons use the same eligible rows within each evaluation window.

## Streamlit Dashboard

The dashboard reads saved evaluation artifacts. It does not retrain the model or rescore the final holdout.

Features include:

- Holdout performance KPIs
- Interactive store-level actual-versus-predicted sales
- Model versus baseline comparison
- Rolling validation results
- Promotion versus non-promotion error analysis
- CSV download for selected-store holdout predictions
- Methodology and limitations

## Setup

Use Python 3.11 and a terminal in the project root.

Install dependencies:

```bash
python -m pip install -r requirements.txt
