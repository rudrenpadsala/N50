"""
src/features/indicators.py

Sprint 2 - Step 2: Feature engineering.

All indicators are computed using pandas rolling windows that only ever
look at the CURRENT row and PAST rows (never future rows), so nothing
here can leak future information. Warm-up rows (where a rolling window
does not yet have enough history) are left as NaN and are documented,
never fabricated - see `README.md` and `build_features.py`.

Volatility classification is the one feature with a data-driven
threshold. That threshold must be fit using TRAINING-period data only
(see `fit_volatility_threshold`) and then applied unchanged to the
rest of the series (including the testing period) - this is the
critical data-leakage safeguard called out in the Sprint 2 spec.
RSI classification and Trend use fixed, non-fitted rules.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Trend labels
BULLISH = "BULLISH"
BEARISH = "BEARISH"
NEUTRAL = "NEUTRAL"

# RSI condition labels
RSI_LOW = "RSI_LOW"
RSI_NORMAL = "RSI_NORMAL"
RSI_HIGH = "RSI_HIGH"

# MACD condition labels
POSITIVE_MOMENTUM = "POSITIVE_MOMENTUM"
NEGATIVE_MOMENTUM = "NEGATIVE_MOMENTUM"

# Volatility condition labels
LOW_VOLATILITY = "LOW_VOLATILITY"
HIGH_VOLATILITY = "HIGH_VOLATILITY"


@dataclass
class FeatureConfig:
    ma_short_window: int = 5
    ma_long_window: int = 20
    rsi_period: int = 14
    rsi_low_threshold: float = 30.0
    rsi_high_threshold: float = 70.0
    volatility_window: int = 20
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9


def compute_moving_average(close: pd.Series, window: int) -> pd.Series:
    """
    Simple moving average over `window` days, inclusive of the current
    day (i.e. MA5 on day t = mean(Close[t-4..t])). Uses only current +
    past prices. min_periods=window means the first (window-1) rows are
    NaN rather than computed from a partial/incomplete window.
    """
    return close.rolling(window=window, min_periods=window).mean()


def compute_daily_returns(close: pd.Series) -> pd.Series:
    """Daily return_t = (Close_t - Close_(t-1)) / Close_(t-1). First row is NaN."""
    return close.pct_change()


def compute_volatility(close: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation of daily returns over `window` days."""
    returns = compute_daily_returns(close)
    return returns.rolling(window=window, min_periods=window).std()


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Standard `period`-period RSI using a simple (non-exponential) rolling
    average of gains and losses:

        RS  = avg_gain(period) / avg_loss(period)
        RSI = 100 - 100 / (1 + RS)

    Result is bounded to [0, 100]. Only current and past prices are used.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # If avg_loss is 0 but avg_gain > 0 -> RSI should be 100 (no losses at all).
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    # If both avg_gain and avg_loss are 0 (flat prices) -> RSI is neutral (50).
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)

    return rsi.clip(lower=0.0, upper=100.0)


def classify_rsi(rsi: pd.Series, low_threshold: float, high_threshold: float) -> pd.Series:
    """RSI < low -> RSI_LOW; RSI > high -> RSI_HIGH; else RSI_NORMAL. NaN stays NaN."""
    conditions = [rsi < low_threshold, rsi > high_threshold]
    choices = [RSI_LOW, RSI_HIGH]
    result = pd.Series(np.select(conditions, choices, default=RSI_NORMAL), index=rsi.index)
    result[rsi.isna()] = np.nan
    return result


def fit_volatility_threshold(volatility_train: pd.Series) -> float:
    """
    Fit the LOW/HIGH volatility split threshold using TRAINING data only.
    Uses the median of non-NaN training-period volatility values. This
    threshold must be reused unchanged for the testing period - it must
    never be refit using test-period volatility.
    """
    clean = volatility_train.dropna()
    if clean.empty:
        raise ValueError("Cannot fit volatility threshold: no non-NaN training volatility values")
    return float(clean.median())


def classify_volatility(volatility: pd.Series, threshold: float) -> pd.Series:
    """volatility < threshold -> LOW_VOLATILITY; else HIGH_VOLATILITY. NaN stays NaN."""
    result = pd.Series(
        np.where(volatility < threshold, LOW_VOLATILITY, HIGH_VOLATILITY),
        index=volatility.index,
    )
    result[volatility.isna()] = np.nan
    return result


def compute_trend(close: pd.Series, ma_long: pd.Series) -> pd.Series:
    """Close > MA20 -> BULLISH; Close < MA20 -> BEARISH; Close == MA20 -> NEUTRAL."""
    conditions = [close > ma_long, close < ma_long]
    choices = [BULLISH, BEARISH]
    result = pd.Series(np.select(conditions, choices, default=NEUTRAL), index=close.index)
    result[ma_long.isna()] = np.nan
    return result


def compute_ema(close: pd.Series, span: int) -> pd.Series:
    """
    Exponential moving average with `span`-period smoothing.
    min_periods=span means (like every other indicator here) the first
    (span-1) rows are explicit NaN rather than a fabricated/under-warmed
    EMA - pandas' recursive ewm formula technically produces *a* number
    from the first row onward, but that number isn't a trustworthy
    `span`-period EMA until it has seen `span` observations.
    """
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def compute_macd(close: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    """
    Standard MACD: EMA(fast) - EMA(slow), plus a `signal`-period EMA of
    that MACD line. Only uses current + past prices (ewm is causal/
    recursive over the given chronological order). Returns a DataFrame
    with columns EMA_Fast, EMA_Slow, MACD, MACD_Signal.
    """
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    macd_line = ema_fast - ema_slow
    # The signal line needs `signal` MACD observations, which in turn each
    # need `slow` price observations - so it stays NaN until slow+signal-1
    # prices have been seen, same "no fabricated warm-up" rule as elsewhere.
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    signal_line = signal_line.where(macd_line.notna())
    return pd.DataFrame({
        "EMA_Fast": ema_fast, "EMA_Slow": ema_slow,
        "MACD": macd_line, "MACD_Signal": signal_line,
    })


def classify_macd(macd: pd.Series, signal: pd.Series) -> pd.Series:
    """MACD > Signal -> POSITIVE_MOMENTUM; MACD <= Signal -> NEGATIVE_MOMENTUM. NaN stays NaN."""
    result = pd.Series(
        np.where(macd > signal, POSITIVE_MOMENTUM, NEGATIVE_MOMENTUM),
        index=macd.index,
    )
    result[macd.isna() | signal.isna()] = np.nan
    return result


def compute_all_features(df: pd.DataFrame, config: FeatureConfig, volatility_threshold: float) -> pd.DataFrame:
    """
    Compute MA5, MA20, RSI, Volatility, Trend, RSI_Condition,
    Volatility_Condition, and MACD (EMA_Fast/EMA_Slow/MACD/MACD_Signal/
    MACD_Condition) for a single company's full chronological OHLCV
    dataframe (sorted ascending by Date). `volatility_threshold` must
    already be fit from that company's TRAINING period only.

    NOTE: MACD is intentionally NOT part of the discrete RL state (see
    src/environment/state.py) - the 5 trained algorithms were trained on
    a 36-state space (Trend x RSI_Condition x Volatility_Condition x
    Position) that does not include MACD, and folding it in would
    require retraining every saved model. MACD here is a display/
    reasoning indicator only (shown to the user, included in "why this
    decision" reasons) - it augments the explanation, not the trained
    models' inputs. `build_features.py`'s FEATURE_COLUMNS selection
    deliberately excludes these columns from the static historical
    per-company files for the same reason - they're computed live-only.
    """
    out = df.copy()
    out["MA5"] = compute_moving_average(out["Close"], config.ma_short_window)
    out["MA20"] = compute_moving_average(out["Close"], config.ma_long_window)
    out["RSI"] = compute_rsi(out["Close"], config.rsi_period)
    out["Volatility"] = compute_volatility(out["Close"], config.volatility_window)

    out["Trend"] = compute_trend(out["Close"], out["MA20"])
    out["RSI_Condition"] = classify_rsi(out["RSI"], config.rsi_low_threshold, config.rsi_high_threshold)
    out["Volatility_Condition"] = classify_volatility(out["Volatility"], volatility_threshold)

    macd_df = compute_macd(out["Close"], config.macd_fast, config.macd_slow, config.macd_signal)
    out["EMA_Fast"] = macd_df["EMA_Fast"]
    out["EMA_Slow"] = macd_df["EMA_Slow"]
    out["MACD"] = macd_df["MACD"]
    out["MACD_Signal"] = macd_df["MACD_Signal"]
    out["MACD_Condition"] = classify_macd(out["MACD"], out["MACD_Signal"])

    return out
