# Retail Demand Forecasting — Streamlit Dashboard

import json
from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. Page configuration and file paths
# --------------------------------------------------

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📈",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

st.title("Retail Demand Forecasting")
st.caption(
    "Rossmann Store Sales | 42-day forecasting horizon | "
    "Historical holdout evaluation"
)


# --------------------------------------------------
# 2. Load saved evaluation artifacts
# --------------------------------------------------

@st.cache_data
def load_artifacts():
    with open(REPORTS / "results.json", encoding="utf-8") as file:
        holdout_results = json.load(file)

    with open(
        REPORTS / "validation_results.json", encoding="utf-8"
    ) as file:
        validation_results = json.load(file)

    predictions = pd.read_csv(
        REPORTS / "holdout_predictions.csv",
        parse_dates=["Date"],
    )

    return holdout_results, validation_results, predictions


try:
    holdout_results, validation_results, predictions = load_artifacts()
except (FileNotFoundError, json.JSONDecodeError, pd.errors.ParserError) as error:
    st.error(f"Could not load project results: {error}")
    st.stop()


# --------------------------------------------------
# 3. Holdout KPI cards
# --------------------------------------------------

holdout = holdout_results["holdout"]
uncertainty = holdout_results["uncertainty"]

st.subheader("Final Holdout Performance")
st.caption(
    f"Evaluation period: {holdout['period_start']} to "
    f"{holdout['period_end']}. These are historical test results, "
    "not a live future forecast."
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Model WAPE",
    f"{holdout['model_wape_all_rows_pct']:.2f}%",
    help="Calculated on all eligible open store-days in the holdout.",
)

c2.metric(
    "Common-row WAPE",
    f"{holdout['model_wape_common_rows_pct']:.2f}%",
    help="Model and all baselines are compared on the same rows.",
)

c3.metric(
    "Improvement vs lag-364",
    f"{holdout['improvement_vs_lag_364_pp']:.2f} pp",
    help="Absolute WAPE reduction, in percentage points, on common rows.",
)

bias_pct = (holdout["forecast_actual_ratio_all_rows"] - 1) * 100

c4.metric(
    "Forecast / actual",
    f"{holdout['forecast_actual_ratio_all_rows']:.3f}",
    f"{bias_pct:+.2f}% overall",
    help="A ratio above 1 indicates aggregate overforecasting.",
)

st.info(
    f"Store-level bootstrap 95% interval for improvement over lag-364: "
    f"{uncertainty['ci_95_lower_pp']:.2f} to "
    f"{uncertainty['ci_95_upper_pp']:.2f} percentage points. "
    "This reflects resampling across stores for this holdout period."
)


# --------------------------------------------------
# 4. Store-level actual vs predicted sales
# --------------------------------------------------

st.subheader("Explore Store-Level Predictions")

stores = sorted(predictions["Store"].dropna().unique().tolist())
selected_store = st.selectbox("Select a store", stores)

store_data = (
    predictions.loc[predictions["Store"] == selected_store]
    .sort_values("Date")
    .copy()
)

chart_data = (
    store_data.set_index("Date")[["Sales", "model_prediction"]]
    .rename(
        columns={
            "Sales": "Actual sales",
            "model_prediction": "Model prediction",
        }
    )
)

st.line_chart(chart_data, y=["Actual sales", "Model prediction"])

store_actual = store_data["Sales"].sum()
store_predicted = store_data["model_prediction"].sum()

left, right = st.columns(2)
left.metric("Actual sales in period", f"{store_actual:,.0f}")
right.metric("Predicted sales in period", f"{store_predicted:,.0f}")

if store_actual > 0:
    store_wape = (
        (store_data["Sales"] - store_data["model_prediction"])
        .abs()
        .sum()
        / store_actual
        * 100
    )
    st.metric("Selected store WAPE", f"{store_wape:.2f}%")

with st.expander("View selected store's daily records"):
    st.dataframe(
        store_data[
            [
                "Date",
                "Sales",
                "model_prediction",
                "pred_lag42",
                "pred_lag364",
                "pred_weekday_mean",
                "Promo",
                "SchoolHoliday",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

# Download selected store's historical predictions
download_data = store_data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download selected store predictions (CSV)",
    data=download_data,
    file_name=f"store_{selected_store}_holdout_predictions.csv",
    mime="text/csv",
)


# --------------------------------------------------
# 5. Fair holdout baseline comparison
# --------------------------------------------------

st.subheader("Holdout: Model vs Baselines")
st.caption(
    "All methods below are evaluated on the same "
    f"{holdout['common_comparison_rows']:,} eligible rows."
)

comparison = pd.DataFrame(
    {
        "Method": [
            "HistGradientBoosting",
            "Lag 364 days",
            "Lag 42 days",
            "Weekday mean",
        ],
        "WAPE (%)": [
            holdout["model_wape_common_rows_pct"],
            holdout["lag_364_wape_common_rows_pct"],
            holdout["lag_42_wape_common_rows_pct"],
            holdout["weekday_mean_wape_common_rows_pct"],
        ],
    }
).set_index("Method")

st.bar_chart(comparison, y="WAPE (%)")
st.dataframe(comparison, use_container_width=True)


# Promo vs non-promo error analysis

st.subheader("Error Analysis: Promotion vs Non-Promotion")

promo_analysis = predictions.copy()
promo_analysis["Scenario"] = promo_analysis["Promo"].map(
    {0: "No promotion", 1: "Promotion"}
)
promo_analysis["Absolute error"] = (
    promo_analysis["Sales"] - promo_analysis["model_prediction"]
).abs()

promo_summary = (
    promo_analysis.groupby("Scenario")
    .agg(
        Store_days=("Sales", "size"),
        Actual_sales=("Sales", "sum"),
        Absolute_error=("Absolute error", "sum"),
    )
    .reset_index()
)

promo_summary["WAPE (%)"] = (
    promo_summary["Absolute_error"]
    / promo_summary["Actual_sales"]
    * 100
)

st.dataframe(
    promo_summary[
        ["Scenario", "Store_days", "WAPE (%)"]
    ].round(2),
    hide_index=True,
    use_container_width=True,
)

promo_chart = promo_summary.set_index("Scenario")[["WAPE (%)"]]
st.bar_chart(promo_chart)



# --------------------------------------------------
# 6. Rolling validation results
# --------------------------------------------------

st.subheader("Rolling-Origin Validation")
st.caption(
    "Four sequential 42-day validation folds. "
    "These results come from the validation-only pipeline."
)

folds = pd.DataFrame(validation_results["folds"])

fold_chart = folds[
    ["fold", "wape_pct"]
].rename(
    columns={
        "fold": "Fold",
        "wape_pct": "Model WAPE (%)",
    }
).set_index("Fold")

st.bar_chart(fold_chart, y="Model WAPE (%)")

st.dataframe(
    folds[
        [
            "fold",
            "validation_start",
            "validation_end",
            "common_comparison_rows",
            "wape_pct",
        ]
    ].rename(
        columns={
            "fold": "Fold",
            "validation_start": "Start date",
            "validation_end": "End date",
            "common_comparison_rows": "Common rows",
            "wape_pct": "Model WAPE (%)",
        }
    ),
    hide_index=True,
    use_container_width=True,
)

st.metric(
    "Mean validation WAPE",
    f"{validation_results['model_mean_wape_pct']:.2f}%",
)

st.caption(
    "Validation metrics are kept separate from final holdout metrics. "
    "The holdout was not used for model selection or tuning."
)


# --------------------------------------------------
# 7. Methodology and limitations
# --------------------------------------------------

with st.expander("Methodology and limitations"):
    st.markdown(
        """
        **Model:** HistGradientBoostingRegressor

        **Forecast horizon:** 42 calendar days.

        **Target population:** Open store-days only.

        **Primary metric:** Weighted Absolute Percentage Error (WAPE).

        **Leakage prevention:** Sales-derived features for each target date
        use historical observations no later than 42 days before that date.

        **Important limitations:**
        - The final holdout covers only 42 days in 2015.
        - Bootstrap uncertainty is across stores, conditional on this period.
        - Prediction-band coverage is empirical, not a guarantee of future coverage.
        - The dataset does not evaluate a full Christmas period.
        - The saved notebook and validation pipeline have a small,
          unresolved prediction discrepancy; pipeline validation metrics
          are treated as canonical.
        """
    )

st.caption(
    "Portfolio project | Results are from saved historical evaluations; "
    "this dashboard does not train a model or generate live forecasts."
)