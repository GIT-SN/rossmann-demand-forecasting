# src/train.py

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.features import (
    build_features,
    get_feature_columns,
    prepare_sales_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

HORIZON_DAYS = 42
MODEL_PARAMS = {
    "learning_rate": 0.08,
    "max_iter": 200,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 42,
}

FOLDS = [
    ("2015-01-03", "2015-02-13"),
    ("2015-02-14", "2015-03-27"),
    ("2015-03-28", "2015-05-08"),
    ("2015-05-09", "2015-06-19"),
]

# Derive rolling validation folds from the observed data range
def derive_folds(dates, horizon_days=42, n_folds=4):
    dates = pd.to_datetime(dates).dropna()

    if dates.empty:
        raise ValueError("Cannot derive folds from empty dates.")

    latest_date = dates.max().normalize()
    holdout_start = latest_date - pd.Timedelta(days=horizon_days - 1)

    # Validation finishes the day before the untouched holdout begins.
    last_validation_end = holdout_start - pd.Timedelta(days=1)

    folds = []
    for i in range(n_folds - 1, -1, -1):
        end = last_validation_end - pd.Timedelta(days=i * horizon_days)
        start = end - pd.Timedelta(days=horizon_days - 1)
        folds.append((start, end))

    return folds


def wape(y_true, y_pred):
    """Weighted absolute percentage error, expressed as a percentage."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true).sum()

    if denominator == 0:
        raise ValueError("WAPE is undefined when total absolute actuals are zero.")

    return float(np.abs(y_true - y_pred).sum() / denominator * 100)


def build_baselines(history, targets):
    """Generate historical baselines using only history before target dates."""
    history = history.loc[history["Open"] == 1].copy()
    history["Date"] = pd.to_datetime(history["Date"])
    targets = targets[["Store", "Date", "Sales"]].copy()
    targets["Date"] = pd.to_datetime(targets["Date"])

    lookup = history.set_index(["Store", "Date"])["Sales"].sort_index()
    result = targets.copy()

    for lag, name in [(42, "pred_lag42"), (364, "pred_lag364")]:
        index = pd.MultiIndex.from_arrays(
            [
                result["Store"].to_numpy(),
                (result["Date"] - pd.Timedelta(days=lag)).to_numpy(),
            ],
            names=["Store", "Date"],
        )
        result[name] = lookup.reindex(index).to_numpy()

    weekday_values = []

    for store_id, date in zip(result["Store"], result["Date"]):
        dates = [date - pd.Timedelta(days=lag) for lag in range(42, 92, 7)]
        values = [
            lookup.get((store_id, historical_date), np.nan)
            for historical_date in dates
        ]
        weekday_values.append(
            np.nanmean(values) if not np.isnan(values).all() else np.nan
        )

    result["pred_weekday_mean"] = weekday_values
    return result

# Temporary diagnostic helper
import hashlib

def dataframe_fingerprint(df):
    normalized = df.reset_index(drop=True).astype("float64")
    hashes = pd.util.hash_pandas_object(
        normalized, index=False
    ).to_numpy()
    return hashlib.sha256(hashes.tobytes()).hexdigest()[:16]



# src/train.py — corrected main function

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(
        DATA_DIR / "train.csv",
        dtype={"StateHoliday": str},
    )
    store_metadata = pd.read_csv(DATA_DIR / "store.csv")

    # Derive folds from the observed dataset dates.
    folds = derive_folds(
        train_raw["Date"],
        horizon_days=HORIZON_DAYS,
        n_folds=4,
    )

    print("Derived validation folds:")
    for start, end in folds:
        print(f"  {start.date()} to {end.date()}")

    sales_data = prepare_sales_data(train_raw)
    features = build_features(sales_data, store_metadata)
    feature_columns = get_feature_columns(features)

    print("Feature table:", features.shape)
    print("Predictors:", len(feature_columns))

    fold_results = []





    for fold_number, (validation_start, validation_end) in enumerate(
        folds, start=1
    ):
        forecast_origin = validation_start - pd.Timedelta(days=1)

        # Labels can be observed through the day before validation begins.
        train_rows = features.loc[
            features["Date"] <= forecast_origin
        ].copy()

        validation_rows = features.loc[
            features["Date"].between(validation_start, validation_end)
        ].copy()

        if train_rows.empty or validation_rows.empty:
            raise ValueError(f"Fold {fold_number} has no training or validation rows.")

        model = HistGradientBoostingRegressor(**MODEL_PARAMS)
        model.fit(train_rows[feature_columns], train_rows["Sales"])

        predictions = validation_rows[["Store", "Date", "Sales"]].copy()
        # Match the notebook's non-negative sales prediction rule
        predictions["model_prediction"] = np.clip(
            model.predict(validation_rows[feature_columns]),
            a_min=0,
            a_max=None,
        )

        # Baselines may use only data before the validation window.
        history = train_raw.loc[
            pd.to_datetime(train_raw["Date"]) < validation_start
        ].copy()
        history["Date"] = pd.to_datetime(history["Date"])

        baselines = build_baselines(
            history,
            predictions[["Store", "Date", "Sales"]],
        )

        predictions = predictions.merge(
            baselines.drop(columns=["Sales"]),
            on=["Store", "Date"],
            how="left",
            validate="one_to_one",
        )

        baseline_columns = [
            "pred_lag42",
            "pred_lag364",
            "pred_weekday_mean",
        ]

        # Compare every model and baseline on exactly the same rows.
        common = predictions.dropna(
            subset=["Sales", "model_prediction"] + baseline_columns
        )

        if common.empty:
            raise ValueError(f"Fold {fold_number} has no common comparison rows.")

        metrics = {
            "fold": fold_number,
            "validation_start": validation_start.strftime("%Y-%m-%d"),
            "validation_end": validation_end.strftime("%Y-%m-%d"),
            "training_rows": int(len(train_rows)),
            "validation_open_store_days": int(len(validation_rows)),
            "common_comparison_rows": int(len(common)),
            "wape_pct": wape(common["Sales"], common["model_prediction"]),
            "forecast_actual_ratio": float(
                common["model_prediction"].sum() / common["Sales"].sum()
            ),
            "baseline_wape_pct": {
                name: wape(common["Sales"], common[name])
                for name in baseline_columns
            },
        }


        fold_results.append(metrics)
        print(
            f"Fold {fold_number}: model WAPE={metrics['wape_pct']:.3f}%, "
            f"common rows={len(common):,}"
        )

    baseline_means = {
        name: float(
            np.mean([fold["baseline_wape_pct"][name] for fold in fold_results])
        )
        for name in ["pred_lag42", "pred_lag364", "pred_weekday_mean"]
    }

    model_mean = float(
        np.mean([fold["wape_pct"] for fold in fold_results])
    )
    best_baseline_name = min(baseline_means, key=baseline_means.get)
    best_baseline_mean = baseline_means[best_baseline_name]

    results = {
        "evaluation_type": "validation_only",
        "holdout_scored": False,
        "forecast_horizon_days": HORIZON_DAYS,
        "model": "HistGradientBoostingRegressor",
        "model_parameters": MODEL_PARAMS,
        "folds": fold_results,
        "model_mean_wape_pct": model_mean,
        "model_wape_min_pct": float(min(f["wape_pct"] for f in fold_results)),
        "model_wape_max_pct": float(max(f["wape_pct"] for f in fold_results)),
        "baseline_mean_wape_pct": baseline_means,
        "best_baseline": best_baseline_name,
        "best_baseline_mean_wape_pct": best_baseline_mean,
        "improvement_over_best_baseline_pp": float(
            best_baseline_mean - model_mean
        ),
        "selection_threshold_pp": 1.0,
    }



    output_path = REPORTS_DIR / "validation_results.json"
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nValidation-only pipeline complete.")
    print(f"Mean model WAPE: {model_mean:.3f}%")
    print(
        f"Best baseline: {best_baseline_name} "
        f"({best_baseline_mean:.3f}%)"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
