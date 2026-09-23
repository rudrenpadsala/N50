"""
Sprint 1 - Step 14: Tests for the data foundation pipeline.

Run with:
    pytest
"""

import os
import sys

import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import (  # noqa: E402
    discover_raw_files,
    load_raw_csv,
    validate_raw_columns,
    load_dataset,
    DataLoadError,
)
from src.data.cleaner import (  # noqa: E402
    standardize_columns,
    clean_dataset,
    _parse_price,
    _parse_volume,
)
from src.data.validator import validate_dataset  # noqa: E402
from src.data.splitter import (  # noqa: E402
    chronological_split,
    TRAIN_START,
    TRAIN_END,
    TEST_START,
    TEST_END,
)

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_filenames():
    return discover_raw_files(RAW_DIR)


@pytest.fixture(scope="module")
def sample_raw_df(raw_filenames):
    return load_raw_csv(os.path.join(RAW_DIR, raw_filenames[0]))


@pytest.fixture
def messy_raw_df():
    """A small synthetic raw-format dataframe with duplicates, a missing
    volume, and out-of-order (descending) dates, mirroring the real files."""
    return pd.DataFrame({
        "Date": ["03-01-2015", "02-01-2015", "02-01-2015", "01-01-2015"],
        "Price": ["101.50", "100.00", "100.00", "99.00"],
        "Open": ["100.00", "99.50", "99.50", "98.50"],
        "High": ["102.00", "100.50", "100.50", "99.50"],
        "Low": ["99.50", "99.00", "99.00", "98.00"],
        "Vol.": ["1.5M", "", "1.2M", "1.0M"],
        "Change %": ["1.5%", "0.5%", "0.5%", "1.0%"],
    })


# ---------------------------------------------------------------------------
# Dataset discovery / loading
# ---------------------------------------------------------------------------

def test_discovers_50_companies(raw_filenames):
    assert len(raw_filenames) == 50


def test_load_raw_csv_returns_dataframe(sample_raw_df):
    assert isinstance(sample_raw_df, pd.DataFrame)
    assert len(sample_raw_df) > 0


def test_validate_raw_columns_passes_for_real_file(sample_raw_df):
    validate_raw_columns(sample_raw_df, "sample.csv")  # should not raise


def test_validate_raw_columns_fails_on_missing_column():
    bad_df = pd.DataFrame({"Date": ["01-01-2015"], "Open": ["1"]})
    with pytest.raises(DataLoadError):
        validate_raw_columns(bad_df, "bad.csv")


def test_load_dataset_returns_company_and_symbol(raw_filenames):
    loaded = load_dataset(RAW_DIR, raw_filenames[0])
    assert loaded.symbol
    assert loaded.company
    assert len(loaded.df) > 0


def test_discover_raw_files_raises_on_missing_dir():
    with pytest.raises(DataLoadError):
        discover_raw_files("/path/does/not/exist")


# ---------------------------------------------------------------------------
# Column standardization
# ---------------------------------------------------------------------------

def test_standardize_columns_renames_correctly(sample_raw_df):
    std = standardize_columns(sample_raw_df)
    assert list(std.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]


def test_standardize_columns_raises_when_column_missing():
    bad_df = pd.DataFrame({"Date": ["01-01-2015"], "Open": ["1"], "High": ["2"]})
    with pytest.raises(ValueError):
        standardize_columns(bad_df)


def test_parse_price_handles_commas():
    assert _parse_price("1,307.80") == 1307.80


def test_parse_price_handles_blank():
    assert pd.isna(_parse_price(""))


def test_parse_volume_handles_million_suffix():
    assert _parse_volume("8.62M") == 8_620_000.0


def test_parse_volume_handles_thousand_suffix():
    assert _parse_volume("750K") == 750_000.0


def test_parse_volume_handles_blank():
    assert pd.isna(_parse_volume(""))


# ---------------------------------------------------------------------------
# Cleaning: dates, sorting, duplicates, missing values
# ---------------------------------------------------------------------------

def test_clean_dataset_converts_date_to_datetime(messy_raw_df):
    cleaned, _ = clean_dataset(messy_raw_df, "TEST")
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Date"])


def test_clean_dataset_sorts_dates_ascending(messy_raw_df):
    cleaned, _ = clean_dataset(messy_raw_df, "TEST")
    assert cleaned["Date"].is_monotonic_increasing


def test_clean_dataset_removes_duplicate_dates(messy_raw_df):
    # messy_raw_df has two rows for 02-01-2015 (one with blank volume)
    cleaned, report = clean_dataset(messy_raw_df, "TEST")
    assert cleaned["Date"].duplicated().sum() == 0
    assert report.duplicate_dates_removed >= 1


def test_clean_dataset_imputes_isolated_missing_volume_via_ffill():
    df = pd.DataFrame({
        "Date": ["01-01-2015", "02-01-2015", "03-01-2015"],
        "Price": ["100.0", "101.0", "102.0"],
        "Open": ["99.0", "100.0", "101.0"],
        "High": ["101.0", "102.0", "103.0"],
        "Low": ["98.0", "99.0", "100.0"],
        "Vol.": ["1.0M", "", "1.2M"],
        "Change %": ["0%", "0%", "0%"],
    })
    cleaned, report = clean_dataset(df, "TEST")
    assert cleaned["Volume"].isna().sum() == 0
    # forward-filled the middle day's volume from the first day (1.0M)
    assert cleaned.loc[cleaned["Date"] == pd.Timestamp("2015-01-02"), "Volume"].iloc[0] == 1_000_000.0
    assert report.volume_values_imputed == 1


def test_clean_dataset_drops_rows_with_missing_price_not_imputed():
    df = pd.DataFrame({
        "Date": ["01-01-2015", "02-01-2015"],
        "Price": ["100.0", ""],  # missing close on day 2
        "Open": ["99.0", "100.0"],
        "High": ["101.0", "102.0"],
        "Low": ["98.0", "99.0"],
        "Vol.": ["1.0M", "1.1M"],
        "Change %": ["0%", "0%"],
    })
    cleaned, report = clean_dataset(df, "TEST")
    assert len(cleaned) == 1
    assert report.missing_price_rows_dropped == 1


def test_clean_dataset_preserves_raw_input(messy_raw_df):
    before = messy_raw_df.copy(deep=True)
    clean_dataset(messy_raw_df, "TEST")
    pd.testing.assert_frame_equal(messy_raw_df, before)


def test_full_pipeline_no_missing_values_after_cleaning(raw_filenames):
    """Every real dataset should come out of cleaning with zero NaNs."""
    for filename in raw_filenames[:5]:  # sample for speed
        loaded = load_dataset(RAW_DIR, filename)
        cleaned, _ = clean_dataset(loaded.df, loaded.symbol)
        assert cleaned.isna().sum().sum() == 0, f"{loaded.symbol} still has missing values"


# ---------------------------------------------------------------------------
# OHLC validation
# ---------------------------------------------------------------------------

def test_validate_dataset_flags_high_less_than_low():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2015-01-01", "2015-01-02"]),
        "Open": [10.0, 10.0],
        "High": [9.0, 12.0],   # first row: High < Low -> invalid
        "Low": [11.0, 9.0],
        "Close": [10.5, 10.5],
        "Volume": [100.0, 100.0],
    })
    result = validate_dataset(df, "TEST")
    assert result.invalid_ohlc_rows == 1
    assert result.status == "FAIL"


def test_validate_dataset_flags_non_positive_price():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2015-01-01"]),
        "Open": [-1.0],
        "High": [10.0],
        "Low": [1.0],
        "Close": [5.0],
        "Volume": [100.0],
    })
    result = validate_dataset(df, "TEST")
    assert result.non_positive_open == 1
    assert result.status == "FAIL"


def test_validate_dataset_passes_clean_data():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2015-01-01", "2015-01-02"]),
        "Open": [10.0, 10.5],
        "High": [11.0, 11.5],
        "Low": [9.5, 10.0],
        "Close": [10.5, 11.0],
        "Volume": [100.0, 120.0],
    })
    result = validate_dataset(df, "TEST")
    assert result.invalid_ohlc_rows == 0
    assert result.status == "PASS"


def test_validate_dataset_flags_negative_volume():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2015-01-01"]),
        "Open": [10.0],
        "High": [11.0],
        "Low": [9.0],
        "Close": [10.5],
        "Volume": [-5.0],
    })
    result = validate_dataset(df, "TEST")
    assert result.negative_volume == 1


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

@pytest.fixture
def full_range_df():
    dates = pd.date_range("2014-06-01", "2026-12-31", freq="90D")
    return pd.DataFrame({
        "Date": dates,
        "Open": range(len(dates)),
        "High": range(len(dates)),
        "Low": range(len(dates)),
        "Close": range(len(dates)),
        "Volume": range(len(dates)),
    }).astype({"Open": float, "High": float, "Low": float, "Close": float, "Volume": float})


def test_split_train_boundaries(full_range_df):
    result = chronological_split(full_range_df, "TEST")
    assert result.train_df["Date"].min() >= TRAIN_START
    assert result.train_df["Date"].max() <= TRAIN_END


def test_split_test_boundaries(full_range_df):
    result = chronological_split(full_range_df, "TEST")
    assert result.test_df["Date"].min() >= TEST_START
    assert result.test_df["Date"].max() <= TEST_END


def test_split_no_overlap(full_range_df):
    result = chronological_split(full_range_df, "TEST")
    train_dates = set(result.train_df["Date"])
    test_dates = set(result.test_df["Date"])
    assert train_dates.isdisjoint(test_dates)


def test_split_excludes_rows_outside_both_windows(full_range_df):
    result = chronological_split(full_range_df, "TEST")
    total_in_windows = len(result.train_df) + len(result.test_df)
    assert total_in_windows <= len(full_range_df)
    # rows before 2015-01-01 or after 2026-07-31 must be excluded
    out_of_window = full_range_df[
        (full_range_df["Date"] < TRAIN_START) | (full_range_df["Date"] > TEST_END)
    ]
    assert len(out_of_window) > 0  # sanity: fixture does include out-of-window rows
    for d in out_of_window["Date"]:
        assert d not in set(result.train_df["Date"]) | set(result.test_df["Date"])


def test_split_is_not_random_same_result_every_call(full_range_df):
    r1 = chronological_split(full_range_df, "TEST")
    r2 = chronological_split(full_range_df, "TEST")
    pd.testing.assert_frame_equal(r1.train_df, r2.train_df)
    pd.testing.assert_frame_equal(r1.test_df, r2.test_df)


def test_split_preserves_chronological_order(full_range_df):
    result = chronological_split(full_range_df, "TEST")
    assert result.train_df["Date"].is_monotonic_increasing
    assert result.test_df["Date"].is_monotonic_increasing


def test_split_raises_if_unsorted():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2020-01-02", "2020-01-01"]),
        "Open": [1.0, 1.0], "High": [1.0, 1.0], "Low": [1.0, 1.0],
        "Close": [1.0, 1.0], "Volume": [1.0, 1.0],
    })
    with pytest.raises(ValueError):
        chronological_split(df, "TEST")


# ---------------------------------------------------------------------------
# No data-leakage sanity checks (Step 7)
# ---------------------------------------------------------------------------

def test_no_leakage_test_dates_all_after_train_dates(full_range_df):
    result = chronological_split(full_range_df, "TEST")
    if len(result.train_df) and len(result.test_df):
        assert result.train_df["Date"].max() < result.test_df["Date"].min()
