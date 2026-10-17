Decisions Log
Data Audit — 17 October 2026
- Dataset: Use the Rossmann training data containing 1,017,209 rows, 1,115 distinct stores, and observed dates from 2013-01-01 to 2015-07-31.
- Forecast horizon: Use a direct 42-calendar-day forecast. Derive validation and holdout boundaries from the observed dataset date range.
- Forecast population: Evaluate sales on open store-days only. Closed store-days are excluded from the sales target population.
- Open zero-sales days: Preserve all 54 observed open store-days with zero sales. Do not silently remove or replace them. Their underlying causes remain unverified.
- Long observation gaps: Preserve missing observations as missing; do not impute them as zero sales. The 180 observed 185-day intervals all end on 2015-01-01. Their underlying cause remains unverified.
- Holiday encoding: StateHoliday contains both string and integer representations of zero in the raw data. Normalize the category explicitly during preprocessing while preserving the raw data.
- Missing values: No missing values were found in the nine original training columns.
- Date coverage: The dataset contains 942 distinct dates between 2013-01-01 and 2015-07-31.
- Validation design: Use four expanding-window rolling-origin validation folds of 42 calendar days each, followed by one final 42-day holdout. Derive fold boundaries from the observed date range.
- Primary metric: Use WAPE (Weighted Absolute Percentage Error), with forecast-to-actual ratio/bias and improvement over the best baseline as supporting measures.
- Model selection: Select the main model only if its mean validation WAPE improves on the best baseline by at least 1 percentage point; otherwise prefer the simpler baseline.
- Leakage prevention: For every target date t, every sales-derived feature must use sales observations dated no later than t - 42 days.
Findings from the Data Audit
- Long gaps: Found 180 intervals of 185 days, all ending on 2015-01-01. The cause remains unverified. Do not impute sales or treat missing observations as zero.
- Open zero-sales observations: Found 54 open store-days with zero sales. Preserve these observations. Their underlying causes remain unverified.
- Holiday encoding: Normalize StateHoliday to a consistent string representation during preprocessing.
- Missing values: The original nine training columns contain no missing values.
- Date coverage: There are 942 distinct dates between 2013-01-01 and 2015-07-31.
- Target population: The training data contains 844,392 open store-day rows and 172,817 closed store-day rows.
SQL Analysis
- Database: Use Python's built-in sqlite3 to create a local SQLite database containing the training, store metadata, and test tables.
- Store-level aggregates: Calculate recorded rows, open and closed days, total sales on open days, average open-day sales, and open zero-sales counts using SQL.
- Cross-check: SQL and pandas agree on the counts of 844,392 open store-days, 172,817 closed store-days, and 54 open store-days with zero sales.
- Scope: Test-period rows contain forecast inputs, not observed target sales. Do not use the test table for training or historical model evaluation.
Validation Design and Leakage Prevention
- Forecast horizon: 42 calendar days, using a direct rather than recursive forecast setup.
- Validation windows:
  1. 2015-01-03 to 2015-02-13
  2. 2015-02-14 to 2015-03-27
  3. 2015-03-28 to 2015-05-08
  4. 2015-05-09 to 2015-06-19
- Final holdout: 2015-06-20 to 2015-07-31.
- Training-label cutoff: For each validation fold, training labels may use observations through the day before validation starts. For the final holdout, the training-label cutoff is 2015-06-19.
- Feature cutoff: For each training or validation row with target date t, all sales-derived predictors must use sales observations dated no later than t - 42 days. This feature-level cutoff is distinct from the training-label cutoff.
- Baseline history: Historical baseline predictions use observations before the corresponding validation window. Each baseline's target-date lag or historical average uses only dates at least 42 days before the target.
- Fair comparison: Within each fold, calculate model and baseline metrics on the same intersection of eligible open store-days where all methods have predictions.
- Fold derivation: The validation pipeline derives the four folds from the observed date range. Automated tests verify 42-day fold lengths, contiguity, and separation from the holdout.
- Leakage tests: Automated feature tests verify lag behavior and that changing sales strictly after the target's t - 42 cutoff does not change that target's sales-derived features. A dedicated test checks the 56-calendar-day rolling-window boundary.
Model Selection Rule
Compare models using mean WAPE across the four 42-day rolling-origin validation folds. Report fold-level WAPE, mean and range of model WAPE, forecast-to-actual ratio, and baseline-relative improvement.
The primary model is selected only if it improves mean validation WAPE by at least 1 percentage point over the best baseline. Otherwise, prefer the simpler baseline.
The main model is HistGradientBoostingRegressor with the following parameters:
- learning_rate=0.08
- max_iter=200
- max_leaf_nodes=31
- l2_regularization=1.0
- early_stopping=False
- random_state=42
Reproducible Validation Results
The validation-only pipeline writes its results to reports/validation_results.json. It does not score the final holdout.
- Mean model WAPE: 9.921%.
- WAPE range across folds: 8.934% to 11.257%.
- Best baseline: pred_lag364, with mean fold WAPE of 14.655%.
- Eight-week same-weekday mean: 17.265% mean fold WAPE.
- Lag-42 baseline: 23.562% mean fold WAPE.
- Improvement over the best baseline: 4.734 percentage points, exceeding the pre-established 1.0-point selection threshold.
- Common comparison rows per fold: 30,832; 39,703; 33,600; and 31,279.
- Automated tests: Seven tests pass with python -m pytest -q at the time of this log update.
The validation pipeline's results are the canonical validation report. Earlier notebook results differ slightly from the reproducible pipeline results; that discrepancy remains to be reconciled before claiming exact notebook-to-pipeline reproducibility. Do not conceal or silently overwrite this discrepancy.
Final Holdout Evaluation — Decision Recorded Before Scoring
The final holdout was evaluated separately in the notebook. Preserve the recorded results and do not rerun the holdout for tuning.
- Holdout: 2015-06-20 to 2015-07-31, 42 calendar days.
- Training-label cutoff: 2015-06-19.
- Selected model: HistGradientBoostingRegressor, selected under the pre-established validation rule.
- Validation selection result: The reproducible pipeline reports mean WAPE of 9.921%, compared with 14.655% for pred_lag364, an improvement of 4.734 percentage points.
- Holdout target population: 40,282 open store-days.
- Model holdout WAPE on all open target rows: 9.326%.
- Fair baseline comparison population: 32,603 open store-days with predictions available from the model and all three baselines.
- Fair comparison WAPE: Model 9.117%; pred_lag364 11.283%; pred_lag42 14.026%; eight-week same-weekday mean 16.995%.
- Improvement over the best baseline on common rows: 2.166 percentage points.
- Store-level bootstrap interval: The 95% bootstrap interval for improvement over the best baseline is approximately 1.932 to 2.402 percentage points. This describes store-resampling uncertainty for this holdout period; it is not a measure of uncertainty across future time periods.
- Prediction bands: The empirical 90% band covered 92.13% of holdout actuals; the empirical 95% band covered 96.32%. These are observed coverage rates, not guarantees of future coverage.
- Saved artifacts: reports/holdout_predictions.csv and reports/results.json.
Holdout metrics are for final evaluation, not further tuning. Do not change features, parameters, or model choice based on these results. Any subsequent changes must be explicitly labelled post-hoc and must not be presented as a clean, untouched holdout evaluation.
Interpretation and Limitations
- The model outperformed the best historical baseline on both the validation comparison and the recorded holdout comparison.
- The holdout baseline comparison uses 32,603 rows, while the model's all-row WAPE uses 40,282 rows. These figures use different populations and must not be compared as if they were calculated on identical rows.
- The 90% and 95% prediction-band coverage figures are empirical. They do not guarantee those coverage rates on future dates or under distribution shift.
- The dataset ends on 2015-07-31 and spans about 2.5 years. The validation windows do not test a full Christmas period, so performance around Christmas and other unrepresented seasonal conditions is uncertain.
- The causes of the long observation gaps and open zero-sales observations remain unverified.
- Exact notebook-to-pipeline validation reproducibility remains to be reconciled.
- The validation pipeline is not a holdout-evaluation script. Keep the final holdout artifacts separate from validation runs.
Next Engineering Steps
- Reconcile the small differences between notebook and pipeline validation results.
- Add or confirm pipeline-level checks for the per-row 42-day sales-feature cutoff and comparison-row consistency.
- Audit pinned dependencies and verify the project from a clean environment or fresh clone.
- Complete and test the Streamlit application, capture a dashboard screenshot, and document the user workflow.
- Write the README only after the implementation, tests, results, and app are complete.



## Validation Pipeline Reproducibility

- The validation pipeline is the authoritative source for reported cross-validation metrics.
- Four rolling validation folds are used, each covering 42 calendar days.
- The pipeline's mean validation WAPE is 9.9211%; the mean WAPE recorded in the saved notebook results is 9.9385%.
- Validation row counts and baseline WAPE scores match between the notebook and pipeline, but the exact cause of the small model-prediction discrepancy has not been established.
- The pipeline's temporary prediction-export and fingerprint diagnostics were removed after investigation.
- Automated tests passed after cleanup: 7 passed.
- The final holdout results remain unchanged and must not be used for further model tuning.
- No model changes were made as a result of the unresolved discrepancy.