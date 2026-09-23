"""
src/features/build_features.py

Sprint 2 - Step 25: Feature engineering script.

Run with:
    python -m src.features.build_features

For every company:
1. Load the Sprint 1 processed dataset (full 2015-2026 cleaned series -
   NOT the train file alone) so that rolling indicators are correctly
   warmed up before the testing period begins, without ever looking at
   future values relative to any given row.
2. Calculate MA5, MA20, RSI, Volatility.
3. Fit the volatility LOW/HIGH threshold using ONLY the training-period
   (<= 2024-12-31) portion of that company's volatility values.
4. Apply Trend / RSI_Condition / Volatility_Condition classifications to
   the full series using that training-fit threshold (and the fixed
   RSI 30/70 rule).
5. Validate the resulting features (RSI bounds, no leakage indicators).
6. Save one feature file per company to data/features/.
7. Save the fitted volatility thresholds for transparency/reuse.
8. Print a summary.

Sprint 1's processed data is never modified.
"""

import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import load_config  # noqa: E402
from src.data.company_map import company_for_code, code_from_filename  # noqa: E402
from src.features.indicators import (  # noqa: E402
    FeatureConfig,
    compute_all_features,
    compute_volatility,
    fit_volatility_threshold,
)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

FEATURE_COLUMNS = [
    "Date", "Open", "High", "Low", "Close", "Volume",
    "MA5", "MA20", "RSI", "Volatility",
    "Trend", "RSI_Condition", "Volatility_Condition",
]


def build_feature_config(cfg: dict) -> FeatureConfig:
    f = cfg["features"]
    return FeatureConfig(
        ma_short_window=f["ma_short_window"],
        ma_long_window=f["ma_long_window"],
        rsi_period=f["rsi_period"],
        rsi_low_threshold=f["rsi_low_threshold"],
        rsi_high_threshold=f["rsi_high_threshold"],
        volatility_window=f["volatility_window"],
    )


def discover_processed_files():
    if not os.path.isdir(PROCESSED_DIR):
        raise FileNotFoundError(
            f"Sprint 1 processed data not found at {PROCESSED_DIR}. "
            "Run `python -m src.data.prepare` (Sprint 1) first."
        )
    files = sorted(f for f in os.listdir(PROCESSED_DIR) if f.endswith("_processed.csv"))
    if not files:
        raise FileNotFoundError(f"No processed CSV files found in {PROCESSED_DIR}")
    return files


def run_pipeline():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    cfg = load_config()
    fcfg = build_feature_config(cfg)
    train_end = pd.Timestamp(cfg["data"]["train_end"])

    files = discover_processed_files()
    print(f"Found {len(files)} Sprint 1 processed dataset(s) in {PROCESSED_DIR}\n")

    threshold_rows = []
    summary_rows = []

    for filename in files:
        symbol = filename.replace("_processed.csv", "")
        company, _ = company_for_code(symbol)

        df = pd.read_csv(os.path.join(PROCESSED_DIR, filename), parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        # Fit volatility threshold using TRAINING PERIOD ONLY.
        train_close = df.loc[df["Date"] <= train_end, "Close"]
        train_volatility = compute_volatility(train_close, fcfg.volatility_window)
        try:
            vol_threshold = fit_volatility_threshold(train_volatility)
        except ValueError as exc:
            print(f"[SKIP] {symbol}: {exc}")
            continue

        featured = compute_all_features(df, fcfg, vol_threshold)

        # Validation: RSI must be within [0, 100] wherever it's not NaN.
        rsi_valid = featured["RSI"].dropna().between(0, 100).all()
        # Validation: no feature computed for a row uses data beyond that
        # row's own Date (structural guarantee from rolling(), spot-checked
        # here by re-deriving MA5 for a late row and comparing).
        leakage_check_ok = True
        if len(featured) > fcfg.ma_short_window:
            probe_idx = len(featured) - 1
            manual_ma5 = df["Close"].iloc[probe_idx - fcfg.ma_short_window + 1: probe_idx + 1].mean()
            leakage_check_ok = abs(featured["MA5"].iloc[probe_idx] - manual_ma5) < 1e-9

        warmup_rows = featured["MA20"].isna().sum()

        out_path = os.path.join(FEATURES_DIR, f"{symbol}_features.csv")
        featured[FEATURE_COLUMNS].to_csv(out_path, index=False)

        threshold_rows.append({
            "Company": company,
            "Symbol": symbol,
            "Volatility_Threshold_Train_Median": vol_threshold,
        })
        summary_rows.append({
            "Company": company,
            "Symbol": symbol,
            "Rows": len(featured),
            "Warmup_Rows_NaN": int(warmup_rows),
            "RSI_In_Bounds": bool(rsi_valid),
            "Leakage_Spot_Check_Passed": bool(leakage_check_ok),
        })

        flag = "OK" if (rsi_valid and leakage_check_ok) else "CHECK"
        print(f"[{flag:5}] {symbol:10} rows={len(featured):5} warmup_nan={int(warmup_rows):3} "
              f"vol_threshold={vol_threshold:.6f}")

    threshold_df = pd.DataFrame(threshold_rows)
    summary_df = pd.DataFrame(summary_rows)
    threshold_path = os.path.join(REPORTS_DIR, "volatility_thresholds.csv")
    summary_path = os.path.join(REPORTS_DIR, "feature_engineering_summary.csv")
    threshold_df.to_csv(threshold_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n=== Sprint 2 Feature Engineering Summary ===")
    print(f"Companies processed : {len(summary_df)}")
    print(f"RSI in [0,100] for all : {summary_df['RSI_In_Bounds'].all()}")
    print(f"Leakage spot-check passed for all : {summary_df['Leakage_Spot_Check_Passed'].all()}")
    print(f"\nFeature files: {FEATURES_DIR}")
    print(f"Reports: {threshold_path}\n         {summary_path}")


if __name__ == "__main__":
    run_pipeline()
