"""
src/data/loader.py

Sprint 1 - Step 9: Data loader.

Responsible ONLY for:
- Discovering raw dataset files.
- Loading a raw CSV as-is into a pandas DataFrame.
- Validating that the required raw columns are present.

Does NOT clean, standardize, validate values, or transform anything.
That belongs to cleaner.py / validator.py.
"""

import os
from dataclasses import dataclass
from typing import List

import pandas as pd

from src.data.company_map import code_from_filename, company_for_code

# Raw investing.com export columns we expect to see before standardization.
RAW_REQUIRED_COLUMNS = ["Date", "Price", "Open", "High", "Low", "Vol."]


class DataLoadError(Exception):
    """Raised when a raw dataset cannot be loaded or is missing required columns."""


@dataclass
class LoadedDataset:
    company: str
    symbol: str
    filename: str
    filepath: str
    df: pd.DataFrame


def discover_raw_files(raw_dir: str) -> List[str]:
    """Return sorted list of raw CSV filenames found in raw_dir."""
    if not os.path.isdir(raw_dir):
        raise DataLoadError(f"Raw data directory not found: {raw_dir}")
    files = sorted(f for f in os.listdir(raw_dir) if f.lower().endswith(".csv"))
    if not files:
        raise DataLoadError(f"No CSV files found in raw data directory: {raw_dir}")
    return files


def load_raw_csv(filepath: str) -> pd.DataFrame:
    """Load a single raw CSV file exactly as-is (utf-8-sig to strip BOM)."""
    try:
        df = pd.read_csv(filepath, encoding="utf-8-sig")
    except Exception as exc:  # noqa: BLE001 - surface a clear loader error
        raise DataLoadError(f"Failed to read {filepath}: {exc}") from exc
    return df


def validate_raw_columns(df: pd.DataFrame, filename: str) -> None:
    """Ensure all required raw columns exist before we proceed."""
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataLoadError(
            f"{filename}: missing required raw column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def load_dataset(raw_dir: str, filename: str) -> LoadedDataset:
    """Load and minimally validate a single company's raw dataset."""
    filepath = os.path.join(raw_dir, filename)
    df = load_raw_csv(filepath)
    validate_raw_columns(df, filename)

    code = code_from_filename(filename)
    company, symbol = company_for_code(code)

    return LoadedDataset(
        company=company,
        symbol=symbol,
        filename=filename,
        filepath=filepath,
        df=df,
    )


def load_all_datasets(raw_dir: str) -> List[LoadedDataset]:
    """Discover and load every raw dataset in raw_dir. Raises on the first hard failure."""
    filenames = discover_raw_files(raw_dir)
    datasets = []
    for filename in filenames:
        datasets.append(load_dataset(raw_dir, filename))
    return datasets
