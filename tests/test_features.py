# tests/test_features.py

import numpy as np
import pandas as pd
import pytest

from src.features import (
    prepare_sales_data,
    build_features,
    get_feature_columns,
)


def test_prepare_sales_data_keeps_only_open_days():
    data = pd.DataFrame(
        {
            "Store": [1, 1, 1],
            "Date": ["2015-01-01", "2015-01-02", "2015-01-03"],
            "Sales": [100, 0, 0],
            "Open": [1, 0, 1],
            "StateHoliday": [0, 0, 0],
        }
    )

    result = prepare_sales_data(data)

    assert len(result) == 2
    assert (result["Open"] == 1).all()
    assert pd.api.types.is_datetime64_any_dtype(result["Date"])
    assert result["StateHoliday"].astype(str).tolist() == ["0", "0"]


def test_sales_lags_use_only_prior_sales():
    dates = pd.date_range("2023-01-01", periods=500, freq="D")

    train = pd.DataFrame(
        {
            "Store": 1,
            "Date": dates,
            "Sales": np.arange(500, dtype=float) + 100,
            "Open": 1,
            "Promo": 0,
            "StateHoliday": "0",
            "SchoolHoliday": 0,
        }
    )

    store_metadata = pd.DataFrame(
        {
            "Store": [1],
            "StoreType": ["a"],
            "Assortment": ["a"],
            "CompetitionDistance": [1000.0],
            "CompetitionOpenSinceMonth": [1],
            "CompetitionOpenSinceYear": [2010],
            "Promo2": [0],
            "Promo2SinceWeek": [np.nan],
            "Promo2SinceYear": [np.nan],
            "PromoInterval": [np.nan],
        }
    )

    features = build_features(train, store_metadata)

    target_date = pd.Timestamp("2024-01-01")
    row = features.loc[features["Date"] == target_date].iloc[0]

    # Date 2024-01-01 is 365 days after 2023-01-01.
    # Each lag must match the exact historical date.
    source = train.set_index("Date")["Sales"]

    for lag in [42, 49, 364]:
        expected_date = target_date - pd.Timedelta(days=lag)
        assert row[f"Sales_lag_{lag}"] == source.loc[expected_date]

    # Explicitly verify the 42-day embargo for the lag-42 feature.
    assert row["Sales_lag_42"] == source.loc[
        target_date - pd.Timedelta(days=42)
    ]

    predictors = get_feature_columns(features)
    assert "Customers" not in predictors
    assert "Sales" not in predictors
    assert "Store" not in predictors


def test_duplicate_store_dates_are_rejected():
    data = pd.DataFrame(
        {
            "Store": [1, 1],
            "Date": ["2015-01-01", "2015-01-01"],
            "Sales": [100, 120],
            "Open": [1, 1],
            "StateHoliday": [0, 0],
        }
    )

    with pytest.raises(ValueError, match="Duplicate Store-Date"):
        prepare_sales_data(data)


# tests/test_features.py — add this test at the bottom

def test_future_sales_cannot_change_target_features():
    dates = pd.date_range("2023-01-01", periods=500, freq="D")

    train = pd.DataFrame({
        "Store": 1,
        "Date": dates,
        "Sales": np.arange(500, dtype=float) + 100,
        "Open": 1,
        "Promo": 0,
        "StateHoliday": "0",
        "SchoolHoliday": 0,
    })

    store_metadata = pd.DataFrame({
        "Store": [1],
        "StoreType": ["a"],
        "Assortment": ["a"],
        "CompetitionDistance": [1000.0],
        "CompetitionOpenSinceMonth": [1],
        "CompetitionOpenSinceYear": [2010],
        "Promo2": [0],
        "Promo2SinceWeek": [np.nan],
        "Promo2SinceYear": [np.nan],
        "PromoInterval": [np.nan],
    })

    target_date = pd.Timestamp("2024-01-01")
    cutoff = target_date - pd.Timedelta(days=42)

    original_features = build_features(train, store_metadata)

    # Change all sales strictly after the permitted cutoff.
    modified_train = train.copy()
    modified_train.loc[modified_train["Date"] > cutoff, "Sales"] += 100_000

    modified_features = build_features(modified_train, store_metadata)

    original_row = original_features.loc[
        original_features["Date"] == target_date
    ].iloc[0]
    modified_row = modified_features.loc[
        modified_features["Date"] == target_date
    ].iloc[0]

    sales_feature_columns = [
        "Sales_lag_42",
        "Sales_lag_49",
        "Sales_lag_364",
        "Sales_same_weekday_mean_8w",
        "Sales_rolling_mean_56d",
    ]

    np.testing.assert_allclose(
        original_row[sales_feature_columns].to_numpy(dtype=float),
        modified_row[sales_feature_columns].to_numpy(dtype=float),
        equal_nan=True,
    )

# Verify the rolling mean uses only the 56-day window ending at t - 42.



from src.features import build_features


def test_rolling_mean_uses_correct_56_day_window():
    dates = pd.date_range("2023-01-01", periods=150, freq="D")
    target_date = pd.Timestamp("2023-05-30")
    cutoff = target_date - pd.Timedelta(days=42)
    window_start = cutoff - pd.Timedelta(days=56)

    train = pd.DataFrame({
        "Store": 1,
        "Date": dates,
        "Sales": np.arange(len(dates), dtype=float),
        "Open": 1,
        "Promo": 0,
        "StateHoliday": "0",
        "SchoolHoliday": 0,
    })

    metadata = pd.DataFrame({
        "Store": [1],
        "StoreType": ["a"],
        "Assortment": ["a"],
        "PromoInterval": [""],
    })

    features = build_features(train, metadata)
    row = features.loc[features["Date"] == target_date].iloc[0]

    # The intended window is (cutoff - 56 days, cutoff],
    # matching rolling("56D", closed="right").
    expected = train.loc[
        (train["Date"] > window_start) &
        (train["Date"] <= cutoff),
        "Sales",
    ].mean()

    assert row["Sales_rolling_mean_56d"] == expected
