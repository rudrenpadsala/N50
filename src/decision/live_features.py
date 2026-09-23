"""
src/decision/live_features.py

THE PROBLEM THIS SOLVES
------------------------
`engine.get_latest_feature_row()` reads data/features/<SYMBOL>_features.csv
and takes its LAST row - which is frozen at whatever date the static
historical dataset ends on (e.g. 31 Jul 2026), no matter what today's
actual date is. Once Angel One is connected, that's misleading: the
"AI Decision" page should reflect TODAY's actual market state, not a
snapshot from weeks ago.

Indicators like MA20 / RSI(14) / 20-day Volatility / MACD need a
trailing WINDOW of recent daily closes (not just today's single price)
to compute correctly - so "using live data" here means fetching the
last ~4 months of DAILY CANDLES from Angel One (not just today's LTP),
recomputing MA5/MA20/RSI/Volatility/Trend/MACD on that live series with
the EXACT SAME formulas as src/features/indicators.py, and returning
today's row in the same shape `engine.py` already expects. This keeps
the model's indicator logic identical between training-time (Sprint 2)
and live use - only the price series being fed into it changes. Today's
High/Low/Volume come along for free from that same candle series (no
separate API call needed).

VOLATILITY THRESHOLD: reused, never refit. The LOW/HIGH volatility
split threshold must be the one already fit from each company's
TRAINING period (see indicators.fit_volatility_threshold and Sprint
2's data-leakage rule) - refitting it on live data here would be a
form of data leakage against the trained Q-tables. It's loaded from
reports/volatility_thresholds.csv, which build_features.py already
produces.

FAILS CLOSED: any problem (no Angel One credentials, no symbol
mapping, network failure, insufficient candle history to fill the
20-day warm-up window, missing volatility threshold) returns
(None, reason) - callers (engine.get_decision) are expected to fall
back to the static historical feature row in that case, not crash or
show a fabricated "today".
"""

from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd

from src.config import load_config
from src.features.indicators import FeatureConfig, compute_all_features
from src.market import angelone_client
from src.market.angelone_symbol_map import get_broker_symbol

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
VOLATILITY_THRESHOLDS_PATH = os.path.join(REPORTS_DIR, "volatility_thresholds.csv")

# Calendar days of history to request. Needs to comfortably cover the
# biggest rolling window - MACD's signal line needs slow+signal-1 = 26+9-1
# = 34 TRADING days, the largest requirement (bigger than MA20/RSI14/
# Volatility20's 20 days) - plus NSE holidays/weekends. 120 calendar days
# is roughly 80 trading days, generous headroom for a 34-trading-day window.
HISTORY_CALENDAR_DAYS = 120


def _feature_config() -> FeatureConfig:
    cfg = load_config()
    f = cfg["features"]
    return FeatureConfig(
        ma_short_window=f["ma_short_window"],
        ma_long_window=f["ma_long_window"],
        rsi_period=f["rsi_period"],
        rsi_low_threshold=f["rsi_low_threshold"],
        rsi_high_threshold=f["rsi_high_threshold"],
        volatility_window=f["volatility_window"],
    )


_threshold_cache: Optional[dict] = None


def _load_volatility_threshold(symbol: str) -> Tuple[Optional[float], Optional[str]]:
    """Reads the TRAINING-period-fit threshold for `symbol` from
    reports/volatility_thresholds.csv (produced by build_features.py) -
    never refit here, per the module docstring."""
    global _threshold_cache
    if _threshold_cache is None:
        if not os.path.isfile(VOLATILITY_THRESHOLDS_PATH):
            return None, f"{VOLATILITY_THRESHOLDS_PATH} not found - run the Sprint 2 feature pipeline first."
        df = pd.read_csv(VOLATILITY_THRESHOLDS_PATH)
        _threshold_cache = dict(zip(df["Symbol"], df["Volatility_Threshold_Train_Median"]))

    if symbol not in _threshold_cache:
        return None, f"No trained volatility threshold found for '{symbol}'."
    return float(_threshold_cache[symbol]), None


def _resolve_instrument(symbol: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Returns ({"exchange":..., "symboltoken":..., "tradingsymbol":...,
    "is_future": bool, "expiry": date|None}, None) or (None, reason).
    Handles both static equity mappings and the dynamically-resolved
    futures marker from angelone_symbol_map.py.
    """
    mapping, reason = get_broker_symbol(symbol)
    if mapping is None:
        return None, reason

    kind, value = mapping
    if kind == "NFO_FUT":
        contract, err = angelone_client.resolve_front_month_future(value)
        if contract is None:
            return None, err
        return {
            "exchange": "NFO", "symboltoken": contract["token"],
            "tradingsymbol": contract["tradingsymbol"], "is_future": True,
            "expiry": contract["expiry"],
        }, None

    exchange, tradingsymbol = kind, value
    token, err = angelone_client.lookup_symbol_token(exchange, tradingsymbol)
    if not token:
        return None, err
    return {
        "exchange": exchange, "symboltoken": token, "tradingsymbol": tradingsymbol,
        "is_future": False, "expiry": None,
    }, None


def _candles_to_dataframe(candles: list) -> pd.DataFrame:
    rows = []
    for c in candles:
        # Angel One candle: [timestamp, open, high, low, close, volume]
        timestamp, o, h, l, close, volume = c
        rows.append({
            "Date": pd.Timestamp(timestamp).tz_localize(None),
            "Open": float(o), "High": float(h), "Low": float(l),
            "Close": float(close), "Volume": float(volume),
        })
    df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    return df


def get_live_feature_row(symbol: str) -> Tuple[Optional[pd.Series], Optional[str]]:
    """
    Fetches recent Angel One daily candles for `symbol`, recomputes
    MA5/MA20/RSI/Volatility/Trend with the same logic and thresholds
    used at training time, and returns TODAY's (or the most recent
    trading day's) row in the same shape as
    engine.get_latest_feature_row() - or (None, reason) if anything
    prevented a trustworthy live row from being produced.
    """
    threshold, reason = _load_volatility_threshold(symbol)
    if threshold is None:
        return None, reason

    instrument, reason = _resolve_instrument(symbol)
    if instrument is None:
        return None, reason
    exchange, symboltoken = instrument["exchange"], instrument["symboltoken"]

    to_date = datetime.now()
    from_date = to_date - timedelta(days=HISTORY_CALENDAR_DAYS)
    candles, reason = angelone_client.get_historical_candles(exchange, symboltoken, from_date, to_date)
    if candles is None:
        return None, reason

    df = _candles_to_dataframe(candles)
    config = _feature_config()
    min_required = max(config.ma_long_window, config.volatility_window, config.rsi_period) 
    if len(df) < min_required:
        return None, (f"Only {len(df)} live trading day(s) of history available for '{symbol}' "
                       f"(need at least {min_required}) - Angel One may not have enough listed history yet.")

    featured = compute_all_features(df, config, threshold)
    clean = featured.dropna(subset=["Trend", "RSI_Condition", "Volatility_Condition"])
    if clean.empty:
        return None, f"Live indicators for '{symbol}' could not be fully computed (still warming up)."

    last_row = clean.iloc[-1].copy()
    last_pos = clean.index[-1]  # positional index shared with `featured`/`df` (dropna preserves it)
    if last_pos > 0:
        last_row["PreviousClose"] = float(featured.loc[last_pos - 1, "Close"])
        prev_volume = featured.loc[last_pos - 1, "Volume"]
        last_row["PreviousVolume"] = float(prev_volume) if pd.notna(prev_volume) else float("nan")
    else:
        last_row["PreviousClose"] = float("nan")
        last_row["PreviousVolume"] = float("nan")

    today_volume = last_row.get("Volume")
    prev_volume = last_row.get("PreviousVolume")
    if pd.notna(today_volume) and pd.notna(prev_volume) and prev_volume:
        last_row["VolumeChangePct"] = float((today_volume - prev_volume) / prev_volume * 100.0)
    else:
        last_row["VolumeChangePct"] = float("nan")

    # MACD is a display/reasoning-only indicator (see compute_all_features's
    # docstring) - if it's still NaN (e.g. a newly-listed futures contract
    # without 34 trading days of history yet), that must not block the
    # core live decision. Its own NaN-ness already means "not shown".
    last_row["MACD_Available"] = bool(pd.notna(last_row.get("MACD")) and pd.notna(last_row.get("MACD_Signal")))

    # Recent-high / drawdown - for the News & Risk Layer's market-shock
    # detector (src/risk/market_shock.py). Reuses the SAME live candle
    # series already fetched above - no extra API call.
    try:
        from src.config import load_config
        drawdown_window = int(load_config()["news_risk"]["drawdown_window_days"])
    except Exception:
        drawdown_window = 20
    recent_window = featured["Close"].iloc[max(0, last_pos - drawdown_window + 1): last_pos + 1]
    recent_high = float(recent_window.max()) if not recent_window.empty else float("nan")
    if pd.notna(recent_high) and recent_high > 0:
        last_row["RecentHigh"] = recent_high
        last_row["DrawdownPct"] = float((recent_high - last_row["Close"]) / recent_high * 100.0)
    else:
        last_row["RecentHigh"] = float("nan")
        last_row["DrawdownPct"] = float("nan")

    last_row["InstrumentExchange"] = exchange
    last_row["InstrumentTradingSymbol"] = instrument["tradingsymbol"]
    last_row["InstrumentIsFuture"] = instrument["is_future"]
    last_row["InstrumentExpiry"] = instrument["expiry"]

    return last_row, None
