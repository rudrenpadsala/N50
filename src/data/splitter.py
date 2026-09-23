"""
src/data/splitter.py

Sprint 1 - Step 6, 12: Chronological (non-random) train/test split.

Training: 01-01-2015 -> 31-12-2024
Testing:  01-01-2025 -> 31-07-2026

No shuffling. No use of test-period statistics to transform training
data. Rows are simply partitioned by Date boundary.
"""

from dataclasses import dataclass

import pandas as pd

TRAIN_START = pd.Timestamp("2015-01-01")
TRAIN_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2026-07-31")


@dataclass
class SplitResult:
    symbol: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def chronological_split(df: pd.DataFrame, symbol: str) -> SplitResult:
    """
    Split df into training and testing sets using fixed date boundaries.
    df must already be sorted ascending by Date and cleaned.
    """
    if not df["Date"].is_monotonic_increasing:
        raise ValueError(f"{symbol}: dataframe must be sorted ascending by Date before splitting")

    train_mask = (df["Date"] >= TRAIN_START) & (df["Date"] <= TRAIN_END)
    test_mask = (df["Date"] >= TEST_START) & (df["Date"] <= TEST_END)

    train_df = df.loc[train_mask].reset_index(drop=True)
    test_df = df.loc[test_mask].reset_index(drop=True)

    return SplitResult(symbol=symbol, train_df=train_df, test_df=test_df)
