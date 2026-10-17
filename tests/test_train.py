
# tests/test_train.py

import pandas as pd

from src.train import derive_folds


def test_derive_folds_creates_four_42_day_windows():
    dates = pd.date_range("2013-01-01", "2015-07-31", freq="D")

    folds = derive_folds(dates, horizon_days=42, n_folds=4)

    assert len(folds) == 4

    for start, end in folds:
        assert (end - start).days + 1 == 42

    assert folds[0][0] == pd.Timestamp("2015-01-03")
    assert folds[-1][1] == pd.Timestamp("2015-06-19")


def test_validation_folds_are_contiguous_and_end_before_holdout():
    dates = pd.date_range("2013-01-01", "2015-07-31", freq="D")

    folds = derive_folds(dates, horizon_days=42, n_folds=4)
    holdout_start = dates.max() - pd.Timedelta(days=41)

    for previous, current in zip(folds, folds[1:]):
        assert current[0] == previous[1] + pd.Timedelta(days=1)

    assert folds[-1][1] + pd.Timedelta(days=1) == holdout_start
