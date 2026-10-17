# src/features.py

from __future__ import annotations

import numpy as np
import pandas as pd


SALES_LAGS = (42, 49, 364)
WEEKDAY_MEAN_LAGS = tuple(range(42, 92, 7))
FORECAST_EMBARGO_DAYS = 42


def prepare_sales_data(train: pd.DataFrame) -> pd.DataFrame:
    """Normalize dates and categories; retain open store-days only."""
    required = {"Store", "Date", "Sales", "Open"}
    missing = required - set(train.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = train.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.loc[data["Open"] == 1].copy()
    data["StateHoliday"] = data["StateHoliday"].astype(str)

    if data.duplicated(["Store", "Date"]).any():
        raise ValueError("Duplicate Store-Date observations found.")

    return data.sort_values(["Store", "Date"]).reset_index(drop=True)


def build_features(
    sales_data: pd.DataFrame,
    store_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build direct 42-day-horizon features.

    For target date t, sales-derived features use dates no later than t - 42.
    Missing or closed dates are not converted to zero sales.
    """
    data = prepare_sales_data(sales_data)

    base_columns = [
        "Store", "Date", "Sales", "Promo", "StateHoliday", "SchoolHoliday"
    ]
    features = data[base_columns].copy()

    features["DayOfWeek"] = features["Date"].dt.dayofweek + 1
    features["Month"] = features["Date"].dt.month
    features["DayOfMonth"] = features["Date"].dt.day
    features["WeekOfYear"] = (
        features["Date"].dt.isocalendar().week.astype(int)
    )
    features["IsWeekend"] = (
        features["Date"].dt.dayofweek >= 5
    ).astype(int)

    sales_lookup = data.set_index(["Store", "Date"])["Sales"].sort_index()

    # Exact-date lags
    for lag in SALES_LAGS:
        lookup_index = pd.MultiIndex.from_arrays(
            [
                features["Store"].to_numpy(),
                (features["Date"] - pd.Timedelta(days=lag)).to_numpy(),
            ],
            names=["Store", "Date"],
        )
        features[f"Sales_lag_{lag}"] = (
            sales_lookup.reindex(lookup_index).to_numpy()
        )

    # Eight same-weekday observations, from t-42 through t-91
    weekday_columns = []

    for lag in WEEKDAY_MEAN_LAGS:
        column = f"_weekday_lag_{lag}"
        lookup_index = pd.MultiIndex.from_arrays(
            [
                features["Store"].to_numpy(),
                (features["Date"] - pd.Timedelta(days=lag)).to_numpy(),
            ],
            names=["Store", "Date"],
        )
        features[column] = sales_lookup.reindex(lookup_index).to_numpy()
        weekday_columns.append(column)

    features["Sales_same_weekday_mean_8w"] = features[
        weekday_columns
    ].mean(axis=1)
    features.drop(columns=weekday_columns, inplace=True)

    # Rolling mean over the 56 calendar days ending at t-42.
    # Missing dates stay NaN; they are not treated as zero sales.
    rolling_parts = []

    for store_id, group in data.groupby("Store", sort=False):
        daily = group.set_index("Date")["Sales"].sort_index()
        calendar = daily.reindex(
            pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        )

        rolling = calendar.rolling(
            "56D", closed="right", min_periods=1
        ).mean()

        rolling.index = rolling.index + pd.Timedelta(days=42)

        rolling_parts.append(
            pd.DataFrame(
                {
                    "Store": store_id,
                    "Date": rolling.index,
                    "Sales_rolling_mean_56d": rolling.to_numpy(),
                }
            )
        )

    rolling_features = pd.concat(rolling_parts, ignore_index=True)

    features = features.merge(
        rolling_features,
        on=["Store", "Date"],
        how="left",
        validate="one_to_one",
    )

    metadata = store_metadata.drop_duplicates("Store").copy()

    features = features.merge(
        metadata,
        on="Store",
        how="left",
        validate="many_to_one",
        suffixes=("", "_store"),
    )

    categorical_columns = [
        col
        for col in ["StateHoliday", "StoreType", "Assortment", "PromoInterval"]
        if col in features.columns
    ]

    for column in categorical_columns:
        features[column] = features[column].astype("string")

    features = pd.get_dummies(
        features,
        columns=categorical_columns,
        dummy_na=True,
        dtype=int,
    )

    return features


def get_feature_columns(features: pd.DataFrame) -> list[str]:
    """Return numeric predictors without target, IDs, or unavailable fields."""
    excluded = {"Sales", "Customers", "Date", "Open", "Store"}
    columns = [
        col for col in features.columns
        if col not in excluded
    ]

    non_numeric = features[columns].select_dtypes(exclude="number").columns
    if len(non_numeric):
        raise ValueError(
            f"Non-numeric model features: {non_numeric.tolist()}"
        )

    return columns