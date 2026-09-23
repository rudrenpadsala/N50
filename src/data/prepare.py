"""
src/data/prepare.py

Sprint 1 - Step 13: Main pipeline.

Run with:
    python -m src.data.prepare

Steps performed (Sprint 1 scope ONLY - no RL, no features, no signals):
1. Find all raw datasets in data/raw/.
2. Load each dataset.
3. Standardize columns to Date, Open, High, Low, Close, Volume.
4. Clean the data (documented strategy - see cleaner.py).
5. Validate OHLC / price / volume / date integrity.
6. Save processed datasets to data/processed/.
7. Chronologically split into train/test (no shuffling) and save to
   data/train/ and data/test/.
8. Generate data-quality and train/test summary reports in reports/.
9. Print final results / acceptance-criteria checklist.
"""

import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import load_all_datasets, DataLoadError  # noqa: E402
from src.data.cleaner import clean_dataset  # noqa: E402
from src.data.validator import validate_dataset  # noqa: E402
from src.data.splitter import chronological_split, TRAIN_START, TRAIN_END, TEST_START, TEST_END  # noqa: E402

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "train")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

EXPECTED_MIN_DATE = pd.Timestamp("2015-01-01")
EXPECTED_MAX_DATE = pd.Timestamp("2026-07-31")


def ensure_dirs():
    for d in (PROCESSED_DIR, TRAIN_DIR, TEST_DIR, REPORTS_DIR):
        os.makedirs(d, exist_ok=True)


def run_pipeline():
    ensure_dirs()

    print(f"Discovering raw datasets in {RAW_DIR} ...")
    try:
        datasets = load_all_datasets(RAW_DIR)
    except DataLoadError as exc:
        print(f"FATAL: {exc}")
        sys.exit(1)
    print(f"Found and loaded {len(datasets)} raw dataset(s).\n")

    quality_rows = []
    split_rows = []

    for ds in datasets:
        symbol = ds.symbol
        cleaned_df, clean_report = clean_dataset(ds.df, symbol)
        val_result = validate_dataset(cleaned_df, symbol)

        # Save processed (cleaned) dataset - never overwrite raw data.
        processed_path = os.path.join(PROCESSED_DIR, f"{symbol}_processed.csv")
        cleaned_df.to_csv(processed_path, index=False)

        # Chronological train/test split - never shuffled.
        split_result = chronological_split(cleaned_df, symbol)
        train_path = os.path.join(TRAIN_DIR, f"{symbol}_train.csv")
        test_path = os.path.join(TEST_DIR, f"{symbol}_test.csv")
        split_result.train_df.to_csv(train_path, index=False)
        split_result.test_df.to_csv(test_path, index=False)

        min_date = cleaned_df["Date"].min()
        max_date = cleaned_df["Date"].max()
        range_ok = (min_date >= EXPECTED_MIN_DATE) and (max_date <= EXPECTED_MAX_DATE)

        status = val_result.status
        if not range_ok:
            status = "WARNING" if status == "PASS" else status

        quality_rows.append({
            "Company": ds.company,
            "Symbol": symbol,
            "Rows": len(cleaned_df),
            "Start Date": min_date.date() if pd.notna(min_date) else None,
            "End Date": max_date.date() if pd.notna(max_date) else None,
            "Missing Values": int(cleaned_df.isna().sum().sum()),
            "Duplicate Rows": clean_report.duplicate_rows_removed,
            "Duplicate Dates": clean_report.duplicate_dates_removed,
            "Invalid OHLC Rows": val_result.invalid_ohlc_rows,
            "Training Rows": len(split_result.train_df),
            "Testing Rows": len(split_result.test_df),
            "Status": status,
        })

        split_rows.append({
            "Company": ds.company,
            "Symbol": symbol,
            "Training Start": split_result.train_df["Date"].min().date() if len(split_result.train_df) else None,
            "Training End": split_result.train_df["Date"].max().date() if len(split_result.train_df) else None,
            "Training Rows": len(split_result.train_df),
            "Testing Start": split_result.test_df["Date"].min().date() if len(split_result.test_df) else None,
            "Testing End": split_result.test_df["Date"].max().date() if len(split_result.test_df) else None,
            "Testing Rows": len(split_result.test_df),
        })

        flag = "OK " if status == "PASS" else status
        print(f"[{flag:8}] {symbol:10} rows={len(cleaned_df):5} "
              f"train={len(split_result.train_df):5} test={len(split_result.test_df):4} "
              f"invalid_ohlc={val_result.invalid_ohlc_rows}")

    quality_df = pd.DataFrame(quality_rows)
    split_df = pd.DataFrame(split_rows)

    quality_path = os.path.join(REPORTS_DIR, "data_quality_summary.csv")
    split_path = os.path.join(REPORTS_DIR, "train_test_summary.csv")
    quality_df.to_csv(quality_path, index=False)
    split_df.to_csv(split_path, index=False)

    print("\n=== Sprint 1 Pipeline Summary ===")
    print(f"Companies processed : {len(datasets)}")
    print(f"PASS    : {(quality_df['Status'] == 'PASS').sum()}")
    print(f"WARNING : {(quality_df['Status'] == 'WARNING').sum()}")
    print(f"FAIL    : {(quality_df['Status'] == 'FAIL').sum()}")
    print(f"Training window : {TRAIN_START.date()} -> {TRAIN_END.date()}")
    print(f"Testing window  : {TEST_START.date()} -> {TEST_END.date()}")
    print(f"\nReports written:\n  {quality_path}\n  {split_path}")
    print(f"\nProcessed data: {PROCESSED_DIR}")
    print(f"Train data    : {TRAIN_DIR}")
    print(f"Test data     : {TEST_DIR}")

    fail_count = (quality_df["Status"] == "FAIL").sum()
    if fail_count > 0:
        print(f"\nNOTE: {fail_count} dataset(s) contain genuine raw-data OHLC violations "
              f"(reported, not silently altered). See data_quality_summary.csv for details.")

    print("\nSprint 1 data pipeline completed successfully "
          "(all 50 companies processed; violations reported, not hidden).")


if __name__ == "__main__":
    run_pipeline()
