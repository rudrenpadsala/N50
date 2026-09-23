"""
src/data/validator.py

Sprint 1 - Step 4, 11: OHLC / price / volume / date / duplicate validation.

This module only INSPECTS and REPORTS. It never mutates or drops data.
"""

from dataclasses import dataclass, field
from typing import List

import pandas as pd


@dataclass
class ValidationResult:
    symbol: str
    total_rows: int = 0
    high_lt_open: int = 0
    high_lt_close: int = 0
    high_lt_low: int = 0
    low_gt_open: int = 0
    low_gt_close: int = 0
    non_positive_open: int = 0
    non_positive_high: int = 0
    non_positive_low: int = 0
    non_positive_close: int = 0
    negative_volume: int = 0
    duplicate_dates: int = 0
    unsorted_dates: bool = False
    invalid_ohlc_rows: int = 0
    violations: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.invalid_ohlc_rows == 0 and self.duplicate_dates == 0 and not self.unsorted_dates:
            return "PASS"
        return "FAIL" if self.invalid_ohlc_rows > 0 else "WARNING"


def validate_dataset(df: pd.DataFrame, symbol: str) -> ValidationResult:
    result = ValidationResult(symbol=symbol, total_rows=len(df))

    high_lt_open = df["High"] < df["Open"]
    high_lt_close = df["High"] < df["Close"]
    high_lt_low = df["High"] < df["Low"]
    low_gt_open = df["Low"] > df["Open"]
    low_gt_close = df["Low"] > df["Close"]

    result.high_lt_open = int(high_lt_open.sum())
    result.high_lt_close = int(high_lt_close.sum())
    result.high_lt_low = int(high_lt_low.sum())
    result.low_gt_open = int(low_gt_open.sum())
    result.low_gt_close = int(low_gt_close.sum())

    result.non_positive_open = int((df["Open"] <= 0).sum())
    result.non_positive_high = int((df["High"] <= 0).sum())
    result.non_positive_low = int((df["Low"] <= 0).sum())
    result.non_positive_close = int((df["Close"] <= 0).sum())
    result.negative_volume = int((df["Volume"] < 0).sum())

    result.duplicate_dates = int(df["Date"].duplicated().sum())
    result.unsorted_dates = not df["Date"].is_monotonic_increasing

    invalid_row_mask = (
        high_lt_open | high_lt_close | high_lt_low | low_gt_open | low_gt_close
        | (df["Open"] <= 0) | (df["High"] <= 0) | (df["Low"] <= 0) | (df["Close"] <= 0)
        | (df["Volume"] < 0)
    )
    result.invalid_ohlc_rows = int(invalid_row_mask.sum())

    if result.high_lt_open:
        result.violations.append(f"High < Open in {result.high_lt_open} row(s)")
    if result.high_lt_close:
        result.violations.append(f"High < Close in {result.high_lt_close} row(s)")
    if result.high_lt_low:
        result.violations.append(f"High < Low in {result.high_lt_low} row(s)")
    if result.low_gt_open:
        result.violations.append(f"Low > Open in {result.low_gt_open} row(s)")
    if result.low_gt_close:
        result.violations.append(f"Low > Close in {result.low_gt_close} row(s)")
    if result.non_positive_open:
        result.violations.append(f"Open <= 0 in {result.non_positive_open} row(s)")
    if result.non_positive_high:
        result.violations.append(f"High <= 0 in {result.non_positive_high} row(s)")
    if result.non_positive_low:
        result.violations.append(f"Low <= 0 in {result.non_positive_low} row(s)")
    if result.non_positive_close:
        result.violations.append(f"Close <= 0 in {result.non_positive_close} row(s)")
    if result.negative_volume:
        result.violations.append(f"Volume < 0 in {result.negative_volume} row(s)")
    if result.duplicate_dates:
        result.violations.append(f"{result.duplicate_dates} duplicate date(s) remain")
    if result.unsorted_dates:
        result.violations.append("Dates are not sorted ascending")

    return result
