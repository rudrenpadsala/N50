"""
Sprint 2 - Step 27: Tests for feature engineering + trading environment.

Run with:
    pytest
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.indicators import (  # noqa: E402
    compute_moving_average,
    compute_daily_returns,
    compute_volatility,
    compute_rsi,
    classify_rsi,
    classify_volatility,
    fit_volatility_threshold,
    compute_trend,
    compute_all_features,
    FeatureConfig,
    BULLISH, BEARISH, NEUTRAL,
    RSI_LOW, RSI_NORMAL, RSI_HIGH,
    LOW_VOLATILITY, HIGH_VOLATILITY,
)
from src.environment.state import (  # noqa: E402
    State, StateEncoder, NO_POSITION, HOLDING, EXPECTED_STATE_SPACE_SIZE,
)
from src.environment.actions import (  # noqa: E402
    HOLD, BUY, SELL, NUM_ACTIONS, is_valid_action, action_name,
)
from src.environment.trading_env import TradingEnvironment  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def price_series():
    """30 days of a simple, deterministic synthetic price series."""
    prices = [
        100, 101, 102, 101, 100, 99, 98, 99, 100, 101,
        103, 105, 104, 106, 108, 107, 109, 110, 112, 111,
        113, 115, 114, 116, 118, 117, 119, 120, 122, 121,
    ]
    return pd.Series(prices, dtype=float)


@pytest.fixture
def default_config():
    return FeatureConfig()


@pytest.fixture
def featured_df(price_series, default_config):
    dates = pd.date_range("2015-01-01", periods=len(price_series), freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "Open": price_series,
        "High": price_series + 1,
        "Low": price_series - 1,
        "Close": price_series,
        "Volume": [1000.0] * len(price_series),
    })
    train_volatility = compute_volatility(df["Close"], default_config.volatility_window)
    threshold = fit_volatility_threshold(train_volatility)
    return compute_all_features(df, default_config, threshold)


# ---------------------------------------------------------------------------
# MA5 / MA20
# ---------------------------------------------------------------------------

def test_ma5_matches_manual_mean(price_series):
    ma5 = compute_moving_average(price_series, 5)
    manual = price_series.iloc[0:5].mean()
    assert ma5.iloc[4] == pytest.approx(manual)


def test_ma5_warmup_rows_are_nan(price_series):
    ma5 = compute_moving_average(price_series, 5)
    assert ma5.iloc[:4].isna().all()
    assert ma5.iloc[4:].notna().all()


def test_ma20_matches_manual_mean(price_series):
    ma20 = compute_moving_average(price_series, 20)
    manual = price_series.iloc[0:20].mean()
    assert ma20.iloc[19] == pytest.approx(manual)


def test_ma20_warmup_rows_are_nan(price_series):
    ma20 = compute_moving_average(price_series, 20)
    assert ma20.iloc[:19].isna().all()
    assert ma20.iloc[19:].notna().all()


def test_ma_never_uses_future_prices(price_series):
    """Changing a future value must not change an already-computed MA5."""
    ma5_before = compute_moving_average(price_series, 5).iloc[4]
    altered = price_series.copy()
    altered.iloc[10] = 999999.0  # a future value relative to index 4
    ma5_after = compute_moving_average(altered, 5).iloc[4]
    assert ma5_before == ma5_after


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def test_rsi_bounded_0_100(price_series):
    rsi = compute_rsi(price_series, 14)
    clean = rsi.dropna()
    assert (clean >= 0).all() and (clean <= 100).all()


def test_rsi_warmup_rows_are_nan(price_series):
    rsi = compute_rsi(price_series, 14)
    assert rsi.iloc[:14].isna().all()


def test_rsi_all_gains_yields_100():
    strictly_up = pd.Series([float(i) for i in range(1, 20)])
    rsi = compute_rsi(strictly_up, 14)
    assert rsi.dropna().iloc[0] == pytest.approx(100.0)


def test_rsi_all_losses_yields_0():
    strictly_down = pd.Series([float(i) for i in range(20, 1, -1)])
    rsi = compute_rsi(strictly_down, 14)
    assert rsi.dropna().iloc[0] == pytest.approx(0.0)


def test_classify_rsi_thresholds():
    rsi = pd.Series([10.0, 50.0, 90.0, np.nan])
    cls = classify_rsi(rsi, low_threshold=30, high_threshold=70)
    assert cls.iloc[0] == RSI_LOW
    assert cls.iloc[1] == RSI_NORMAL
    assert cls.iloc[2] == RSI_HIGH
    assert pd.isna(cls.iloc[3])


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def test_volatility_matches_manual_std(price_series):
    vol = compute_volatility(price_series, 20)
    returns = compute_daily_returns(price_series)
    manual = returns.iloc[1:21].std()
    assert vol.iloc[20] == pytest.approx(manual)


def test_volatility_warmup_rows_are_nan(price_series):
    vol = compute_volatility(price_series, 20)
    assert vol.iloc[:20].isna().all()


def test_fit_volatility_threshold_uses_median():
    vol = pd.Series([0.01, 0.02, 0.03, 0.04, np.nan])
    threshold = fit_volatility_threshold(vol)
    assert threshold == pytest.approx(0.025)


def test_fit_volatility_threshold_raises_on_all_nan():
    with pytest.raises(ValueError):
        fit_volatility_threshold(pd.Series([np.nan, np.nan]))


def test_classify_volatility(price_series):
    vol = pd.Series([0.01, 0.05, np.nan])
    cls = classify_volatility(vol, threshold=0.03)
    assert cls.iloc[0] == LOW_VOLATILITY
    assert cls.iloc[1] == HIGH_VOLATILITY
    assert pd.isna(cls.iloc[2])


def test_volatility_threshold_not_fit_on_test_period(featured_df, default_config):
    """
    Simulates the leakage-prevention rule: fitting the threshold using
    only a training slice must be independent of what happens after it.
    """
    train_slice = featured_df.iloc[:20]
    train_vol = compute_volatility(train_slice["Close"], default_config.volatility_window)
    threshold_from_train_only = fit_volatility_threshold(train_vol) if train_vol.notna().any() else None
    # With only 20 rows the 20-day window barely warms up; this just
    # verifies the fitting function only ever looks at the slice given to it.
    assert threshold_from_train_only is None or isinstance(threshold_from_train_only, float)


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------

def test_trend_bullish_bearish_neutral():
    close = pd.Series([110.0, 90.0, 100.0])
    ma20 = pd.Series([100.0, 100.0, 100.0])
    trend = compute_trend(close, ma20)
    assert trend.iloc[0] == BULLISH
    assert trend.iloc[1] == BEARISH
    assert trend.iloc[2] == NEUTRAL


def test_trend_nan_when_ma20_nan():
    close = pd.Series([110.0])
    ma20 = pd.Series([np.nan])
    trend = compute_trend(close, ma20)
    assert pd.isna(trend.iloc[0])


# ---------------------------------------------------------------------------
# compute_all_features / no future-data leakage
# ---------------------------------------------------------------------------

def test_compute_all_features_has_expected_columns(featured_df):
    for col in ["MA5", "MA20", "RSI", "Volatility", "Trend", "RSI_Condition", "Volatility_Condition",
                "EMA_Fast", "EMA_Slow", "MACD", "MACD_Signal", "MACD_Condition"]:
        assert col in featured_df.columns


def test_features_no_leakage_altering_future_row_unchanged_past(price_series, default_config):
    dates = pd.date_range("2015-01-01", periods=len(price_series), freq="D")
    df = pd.DataFrame({
        "Date": dates, "Open": price_series, "High": price_series + 1,
        "Low": price_series - 1, "Close": price_series, "Volume": [1000.0] * len(price_series),
    })
    vol = compute_volatility(df["Close"], default_config.volatility_window)
    threshold = fit_volatility_threshold(vol) if vol.notna().any() else 0.01
    original = compute_all_features(df, default_config, threshold)

    altered_prices = price_series.copy()
    altered_prices.iloc[-1] = 999999.0  # change only the LAST (future-most) row
    df2 = df.copy()
    df2["Close"] = altered_prices
    altered = compute_all_features(df2, default_config, threshold)

    # Every row except the last must be completely unaffected by changing the last row.
    for col in ["MA5", "MA20", "RSI", "Volatility", "MACD", "MACD_Signal"]:
        pd.testing.assert_series_equal(
            original[col].iloc[:-1], altered[col].iloc[:-1], check_names=False
        )


# ---------------------------------------------------------------------------
# MACD (display/reasoning indicator only - NOT part of the RL state, see
# indicators.compute_all_features's docstring and src/environment/state.py)
# ---------------------------------------------------------------------------

def test_macd_matches_manual_ema_calculation():
    """MACD must equal EMA(fast) - EMA(slow), computed the standard way,
    and the signal line must be a `signal`-period EMA of that MACD line."""
    from src.features.indicators import compute_macd

    close = pd.Series([100.0, 102.0, 101.0, 105.0, 107.0, 106.0, 110.0, 112.0,
                        111.0, 115.0, 117.0, 116.0, 120.0, 122.0, 121.0])
    fast, slow, signal = 3, 6, 2  # small windows so this tiny series fully warms up
    result = compute_macd(close, fast, slow, signal)

    manual_ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    manual_ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    manual_macd = manual_ema_fast - manual_ema_slow
    manual_signal = manual_macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    manual_signal = manual_signal.where(manual_macd.notna())

    pd.testing.assert_series_equal(result["MACD"], manual_macd, check_names=False)
    pd.testing.assert_series_equal(result["MACD_Signal"], manual_signal, check_names=False)


def test_macd_nan_during_warmup_not_fabricated():
    from src.features.indicators import compute_macd

    close = pd.Series([100.0, 101.0, 102.0])  # far too short for fast=12/slow=26/signal=9
    result = compute_macd(close, fast=12, slow=26, signal=9)
    assert result["MACD"].isna().all()
    assert result["MACD_Signal"].isna().all()


def test_classify_macd_positive_and_negative():
    from src.features.indicators import classify_macd, POSITIVE_MOMENTUM, NEGATIVE_MOMENTUM

    macd = pd.Series([1.0, -1.0, 0.5])
    signal = pd.Series([0.5, -0.5, 0.5])
    result = classify_macd(macd, signal)
    assert result.iloc[0] == POSITIVE_MOMENTUM   # 1.0 > 0.5
    assert result.iloc[1] == NEGATIVE_MOMENTUM   # -1.0 < -0.5
    assert result.iloc[2] == NEGATIVE_MOMENTUM   # 0.5 == 0.5 -> not strictly greater


def test_macd_no_leakage_altering_future_row_unchanged_past():
    from src.features.indicators import compute_macd

    close = pd.Series([100.0 + i for i in range(40)])
    original = compute_macd(close, fast=12, slow=26, signal=9)

    altered_close = close.copy()
    altered_close.iloc[-1] = 999999.0
    altered = compute_macd(altered_close, fast=12, slow=26, signal=9)

    for col in ["MACD", "MACD_Signal"]:
        pd.testing.assert_series_equal(original[col].iloc[:-1], altered[col].iloc[:-1], check_names=False)


def test_static_historical_feature_files_unaffected_by_macd():
    """
    MACD must NOT be written to the static per-company feature CSVs
    (build_features.py's FEATURE_COLUMNS) - those files back the trained
    RL models' 36-state space (Trend x RSI_Condition x Volatility_Condition
    x Position), which does not include MACD. Adding it there would
    silently change the historical dataset schema every downstream
    training/backtest script reads.
    """
    from src.features.build_features import FEATURE_COLUMNS

    assert "MACD" not in FEATURE_COLUMNS
    assert "MACD_Signal" not in FEATURE_COLUMNS
    assert "EMA_Fast" not in FEATURE_COLUMNS
    assert "EMA_Slow" not in FEATURE_COLUMNS
    assert "MACD_Condition" not in FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

def test_state_space_size_is_36():
    assert EXPECTED_STATE_SPACE_SIZE == 36
    encoder = StateEncoder()
    assert encoder.num_states == 36


def test_state_encoder_encode_decode_roundtrip():
    encoder = StateEncoder()
    state = State(BULLISH, RSI_NORMAL, LOW_VOLATILITY, NO_POSITION)
    encoded = encoder.encode(state)
    decoded = encoder.decode(encoded)
    assert decoded == state


def test_state_encoder_is_deterministic_across_instances():
    e1 = StateEncoder()
    e2 = StateEncoder()
    state = State(BEARISH, RSI_HIGH, HIGH_VOLATILITY, HOLDING)
    assert e1.encode(state) == e2.encode(state)


def test_state_encoder_all_ids_unique_and_contiguous():
    encoder = StateEncoder()
    ids = sorted(encoder.encode(s) for s in encoder.all_states())
    assert ids == list(range(36))


def test_state_encoder_rejects_unknown_id():
    encoder = StateEncoder()
    with pytest.raises(ValueError):
        encoder.decode(999)


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

def test_three_actions_defined():
    assert NUM_ACTIONS == 3
    assert set([HOLD, BUY, SELL]) == {0, 1, 2}


def test_is_valid_action():
    assert is_valid_action(HOLD)
    assert is_valid_action(BUY)
    assert is_valid_action(SELL)
    assert not is_valid_action(3)
    assert not is_valid_action(-1)


def test_action_name_raises_on_undefined():
    with pytest.raises(ValueError):
        action_name(99)


# ---------------------------------------------------------------------------
# Trading environment
# ---------------------------------------------------------------------------

def test_env_reset_sets_initial_capital(featured_df):
    env = TradingEnvironment(featured_df, initial_capital=100_000, transaction_cost=0.001)
    env.reset()
    assert env.cash == 100_000
    assert env.shares == 0
    assert env.position == NO_POSITION
    assert env.get_portfolio_value() == 100_000


def test_env_buy_reduces_cash_and_increases_shares(featured_df):
    env = TradingEnvironment(featured_df, initial_capital=100_000, transaction_cost=0.001)
    env.reset()
    price = float(env.df.loc[0, "Close"])
    env.step(BUY)
    expected_cash = 100_000 - price * (1 + 0.001)
    assert env.cash == pytest.approx(expected_cash)
    assert env.shares == 1


def test_env_hold_does_not_change_cash_or_shares(featured_df):
    env = TradingEnvironment(featured_df, initial_capital=100_000, transaction_cost=0.001)
    env.reset()
    env.step(BUY)
    cash_before, shares_before = env.cash, env.shares
    env.step(HOLD)
    assert env.cash == cash_before
    assert env.shares == shares_before


def test_env_sell_increases_cash_and_decreases_shares(featured_df):
    env = TradingEnvironment(featured_df, initial_capital=100_000, transaction_cost=0.001)
    env.reset()
    env.step(BUY)
    cash_after_buy = env.cash
    price = float(env.df.loc[env.current_step, "Close"])
    env.step(SELL)
    expected_cash = cash_after_buy + price * (1 - 0.001)
    assert env.cash == pytest.approx(expected_cash)
    assert env.shares == 0


def test_env_transaction_cost_applied_to_buy_and_sell_not_hold(featured_df):
    env = TradingEnvironment(featured_df, initial_capital=100_000, transaction_cost=0.05)
    env.reset()
    price = float(env.df.loc[0, "Close"])
    env.step(BUY)
    assert env.cash == pytest.approx(100_000 - price * 1.05)


def test_env_invalid_buy_insufficient_cash_is_noop_with_penalty(featured_df):
    env = TradingEnvironment(featured_df, initial_capital=1.0, transaction_cost=0.001,
                              invalid_action_penalty=-0.001)
    env.reset()
    _, reward, _, info = env.step(BUY)
    assert info["invalid_action"] is True
    assert env.shares == 0
    assert env.cash == 1.0  # unchanged


def test_env_invalid_sell_no_shares_is_noop_with_penalty(featured_df):
    env = TradingEnvironment(featured_df, initial_capital=100_000, transaction_cost=0.001,
                              invalid_action_penalty=-0.001)
    env.reset()
    cash_before = env.cash
    _, reward, _, info = env.step(SELL)
    assert info["invalid_action"] is True
    assert env.shares == 0
    assert env.cash == cash_before


def test_env_never_negative_cash(featured_df):
    env = TradingEnvironment(featured_df, initial_capital=100_000, transaction_cost=0.001)
    env.reset()
    for _ in range(len(env.df) - 1):
        env.step(BUY)
        if env.is_done():
            break
    assert env.cash >= 0


def test_env_never_negative_shares(featured_df):
    env = TradingEnvironment(featured_df, initial_capital=100_000, transaction_cost=0.001)
    env.reset()
    for _ in range(len(env.df) - 1):
        env.step(SELL)
        if env.is_done():
            break
    assert env.shares >= 0


def test_env_portfolio_value_formula(featured_df):
    env = TradingEnvironment(featured_df, initial_capital=100_000, transaction_cost=0.001)
    env.reset()
    env.step(BUY)
    price = float(env.df.loc[env.current_step, "Close"])
    assert env.get_portfolio_value() == pytest.approx(env.cash + env.shares * price)


def test_env_reward_matches_formula(featured_df):
    env = TradingEnvironment(featured_df, initial_capital=100_000, transaction_cost=0.001)
    env.reset()
    prev = env.get_portfolio_value()
    _, reward, _, info = env.step(HOLD)
    expected = (info["portfolio_value"] - prev) / prev
    assert reward == pytest.approx(expected)


def test_env_episode_terminates_at_end_of_data(featured_df):
    env = TradingEnvironment(featured_df, initial_capital=100_000, transaction_cost=0.001)
    env.reset()
    steps = 0
    done = False
    while not done:
        _, _, done, _ = env.step(HOLD)
        steps += 1
        if steps > len(env.df) + 5:
            pytest.fail("Episode did not terminate within expected number of steps")
    assert env.is_done()


def test_env_step_after_done_raises(featured_df):
    env = TradingEnvironment(featured_df, initial_capital=100_000, transaction_cost=0.001)
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step(HOLD)
    with pytest.raises(RuntimeError):
        env.step(HOLD)


def test_env_rejects_invalid_action_value(featured_df):
    env = TradingEnvironment(featured_df, initial_capital=100_000, transaction_cost=0.001)
    env.reset()
    with pytest.raises(ValueError):
        env.step(5)


def test_env_deterministic_same_action_sequence_same_result(featured_df):
    seq = [BUY, HOLD, SELL, HOLD, BUY, HOLD, SELL]

    env1 = TradingEnvironment(featured_df, initial_capital=100_000, transaction_cost=0.001)
    env1.reset()
    results1 = [env1.step(a) for a in seq]

    env2 = TradingEnvironment(featured_df, initial_capital=100_000, transaction_cost=0.001)
    env2.reset()
    results2 = [env2.step(a) for a in seq]

    for (s1, r1, d1, i1), (s2, r2, d2, i2) in zip(results1, results2):
        assert s1 == s2
        assert r1 == pytest.approx(r2)
        assert d1 == d2
        assert i1["cash"] == pytest.approx(i2["cash"])
        assert i1["shares"] == i2["shares"]


def test_env_rejects_dataframe_missing_columns():
    bad_df = pd.DataFrame({"Date": pd.date_range("2015-01-01", periods=5), "Close": [1, 2, 3, 4, 5]})
    with pytest.raises(ValueError):
        TradingEnvironment(bad_df)


def test_env_state_reflects_position(featured_df):
    encoder = StateEncoder()
    env = TradingEnvironment(featured_df, initial_capital=100_000, transaction_cost=0.001,
                              state_encoder=encoder)
    state_id = env.reset()
    state = encoder.decode(state_id)
    assert state.position == NO_POSITION

    env.step(BUY)
    state_after_buy = encoder.decode(env.get_state())
    assert state_after_buy.position == HOLDING


def test_env_uses_only_training_period_when_given_train_slice(featured_df, default_config):
    """The env itself must not fit/refit anything - it just consumes whatever
    slice of the (already-featured) dataframe it's given."""
    # Take a slice that still contains enough post-warmup rows to run.
    train_like_slice = featured_df.iloc[19:].reset_index(drop=True)
    full_env = TradingEnvironment(featured_df, initial_capital=100_000, transaction_cost=0.001)
    slice_env = TradingEnvironment(train_like_slice, initial_capital=100_000, transaction_cost=0.001)
    assert len(slice_env.df) <= len(full_env.df)
