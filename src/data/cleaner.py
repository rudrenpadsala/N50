"""
src/data/cleaner.py

Sprint 1 - Steps 2, 3, 10: Standardize columns, clean data.

Missing-value strategy (documented, not "fill everything blindly"):
- Volume is the ONLY field observed to have isolated single missing
  values in this dataset (one blank day per affected company, verified
  during inspection). For such isolated gaps, the missing Volume is
  forward-filled from the prior trading day's Volume for that same
  company, since price data for that day is fully valid and dropping
  a real trading day just because volume is blank would discard usable
  OHLC information. Every imputed cell is logged.
- Any missing Date/Open/High/Low/Close is NOT imputed. Rows with a
  missing price field or an unparseable Date are dropped as invalid,
  and this is logged and reported per company.
- No future information is ever used: forward-fill only looks
  backward in time (uses only rows before it in ascending date order).
"""

from dataclasses import dataclass, field
from typing import List

import pandas as pd

# Standard target schema for this project.
STANDARD_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

# investing.com raw column -> standard column
RAW_TO_STANDARD = {
    "Date": "Date",
    "Open": "Open",
    "High": "High",
    "Low": "Low",
    "Price": "Close",   # investing.com calls the close "Price"
    "Vol.": "Volume",
}

DATE_FORMAT = "%d-%m-%Y"


@dataclass
class CleaningReport:
    symbol: str
    rows_in: int = 0
    rows_out: int = 0
    duplicate_rows_removed: int = 0
    duplicate_dates_removed: int = 0
    unparseable_dates_dropped: int = 0
    missing_price_rows_dropped: int = 0
    volume_values_imputed: int = 0
    notes: List[str] = field(default_factory=list)


def _parse_price(value):
    """Parse a price string like '1,307.80' -> 1307.80. Returns NaN if unparseable."""
    if pd.isna(value):
        return float("nan")
    s = str(value).strip().replace(",", "")
    if s == "":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _parse_volume(value):
    """Parse a volume string like '8.62M', '750K', '' -> float. Returns NaN if blank/unparseable."""
    if pd.isna(value):
        return float("nan")
    s = str(value).strip().replace(",", "")
    if s == "":
        return float("nan")
    multiplier = 1.0
    suffix = s[-1].upper() if s else ""
    if suffix == "K":
        multiplier, s = 1_000.0, s[:-1]
    elif suffix == "M":
        multiplier, s = 1_000_000.0, s[:-1]
    elif suffix == "B":
        multiplier, s = 1_000_000_000.0, s[:-1]
    try:
        return float(s) * multiplier
    except ValueError:
        return float("nan")


def standardize_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename raw investing.com columns to the standard schema and drop any
    columns not part of the standard schema (e.g. 'Change %').
    Raises if a required raw column is absent.
    """
    missing = [c for c in RAW_TO_STANDARD if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Cannot standardize: missing raw column(s) {missing}")

    df = raw_df.rename(columns=RAW_TO_STANDARD)
    df = df[STANDARD_COLUMNS].copy()
    return df


def clean_dataset(raw_df: pd.DataFrame, symbol: str) -> "tuple[pd.DataFrame, CleaningReport]":
    """
    Full cleaning pipeline for a single company's raw dataframe.
    Returns (cleaned_df, report). Does not mutate raw_df.
    """
    report = CleaningReport(symbol=symbol, rows_in=len(raw_df))

    df = standardize_columns(raw_df)

    # Parse numeric fields
    df["Open"] = df["Open"].apply(_parse_price)
    df["High"] = df["High"].apply(_parse_price)
    df["Low"] = df["Low"].apply(_parse_price)
    df["Close"] = df["Close"].apply(_parse_price)
    df["Volume"] = df["Volume"].apply(_parse_volume)

    # Parse dates
    parsed_dates = pd.to_datetime(df["Date"], format=DATE_FORMAT, errors="coerce")
    bad_date_mask = parsed_dates.isna()
    report.unparseable_dates_dropped = int(bad_date_mask.sum())
    if report.unparseable_dates_dropped:
        report.notes.append(f"Dropped {report.unparseable_dates_dropped} row(s) with unparseable dates")
    df = df.loc[~bad_date_mask].copy()
    df["Date"] = parsed_dates.loc[~bad_date_mask]

    # Sort ascending by date (raw source is newest-first)
    df = df.sort_values("Date", ascending=True).reset_index(drop=True)

    # Drop rows with missing/invalid price fields (never imputed)
    price_cols = ["Open", "High", "Low", "Close"]
    missing_price_mask = df[price_cols].isna().any(axis=1)
    report.missing_price_rows_dropped = int(missing_price_mask.sum())
    if report.missing_price_rows_dropped:
        report.notes.append(
            f"Dropped {report.missing_price_rows_dropped} row(s) with missing OHLC price field(s)"
        )
    df = df.loc[~missing_price_mask].copy()

    # Documented, targeted imputation: forward-fill isolated missing Volume only.
    vol_missing_before = int(df["Volume"].isna().sum())
    if vol_missing_before:
        df["Volume"] = df["Volume"].ffill()
        vol_missing_after = int(df["Volume"].isna().sum())
        imputed = vol_missing_before - vol_missing_after
        report.volume_values_imputed = imputed
        if imputed:
            report.notes.append(
                f"Forward-filled {imputed} isolated missing Volume value(s) from prior trading day"
            )
        if vol_missing_after:
            # Volume missing at the very start of series with nothing to forward-fill from.
            df = df.loc[df["Volume"].notna()].copy()
            report.notes.append(
                f"Dropped {vol_missing_after} leading row(s) with no prior Volume to forward-fill from"
            )

    # Duplicate rows (exact duplicates across all standard columns)
    dup_row_mask = df.duplicated(subset=STANDARD_COLUMNS, keep="first")
    report.duplicate_rows_removed = int(dup_row_mask.sum())
    if report.duplicate_rows_removed:
        report.notes.append(f"Removed {report.duplicate_rows_removed} exact duplicate row(s)")
    df = df.loc[~dup_row_mask].copy()

    # Duplicate dates (same Date, differing values) - keep first chronological occurrence
    dup_date_mask = df.duplicated(subset=["Date"], keep="first")
    report.duplicate_dates_removed = int(dup_date_mask.sum())
    if report.duplicate_dates_removed:
        report.notes.append(f"Removed {report.duplicate_dates_removed} duplicate-date row(s), kept first occurrence")
    df = df.loc[~dup_date_mask].copy()

    df = df.sort_values("Date", ascending=True).reset_index(drop=True)
    report.rows_out = len(df)

    if not report.notes:
        report.notes.append("No cleaning actions were necessary")

    return df, report
