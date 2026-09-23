"""
src/backtesting/metrics.py

Phase 1 - Sprint 5 - Steps 6-11: Performance metrics computed from a
single backtest's portfolio history and trade log. Every function here
is a pure calculation over already-recorded numbers (produced by
src/backtesting/backtester.py) - none of them touch the environment,
a model, or the test dataframe directly, so the SAME formulas apply
identically to all 5 RL algorithms and to the Buy & Hold baseline
(Sprint 5 Section 8: "Do not change the methodology between
algorithms.").

Every function returns a plain float and is explicit about its
divide-by-zero convention, so no metric in reports/final_results.csv
is ever NaN or +/-inf (Sprint 5 Section 36).
"""

from typing import Sequence

import numpy as np


def total_return_pct(initial_value: float, final_value: float) -> float:
    """Sprint 5 Section 7: ((Final - Initial) / Initial) * 100."""
    if initial_value == 0:
        return 0.0
    return ((final_value - initial_value) / initial_value) * 100.0


def annualized_return_pct(initial_value: float, final_value: float, num_calendar_days: int) -> float:
    """
    Sprint 5 Section 11: (Final / Initial) ^ (365 / calendar_days) - 1,
    expressed as a percentage. `num_calendar_days` is the ACTUAL number
    of calendar days spanned by the test period (last date - first
    date), not a fixed trading-day count.
    """
    if initial_value <= 0 or num_calendar_days <= 0:
        return 0.0
    ratio = final_value / initial_value
    if ratio <= 0:
        # A wiped-out portfolio (final <= 0) has no well-defined real-valued
        # fractional power - report the floor case explicitly instead of NaN.
        return -100.0
    return ((ratio ** (365.0 / num_calendar_days)) - 1.0) * 100.0


def sharpe_ratio(daily_returns: Sequence[float], risk_free_rate: float = 0.0,
                  trading_days_per_year: int = 252) -> float:
    """
    Sprint 5 Section 8: Sharpe = Mean(Return - daily_rf) / Std(Return) * sqrt(252).

    `risk_free_rate` is an ANNUAL rate; it is converted to a daily rate
    by dividing by `trading_days_per_year` (kept as a simple, documented
    convention - Section 8 explicitly allows Risk-Free Rate = 0 when
    none is otherwise defined, which is this project's default via
    config.yaml's `backtest.risk_free_rate`).

    Returns 0.0 (never NaN) when there are fewer than 2 return
    observations or the return series has zero variance (e.g. a
    portfolio that never traded) - documented convention, not a crash.
    """
    returns = np.asarray(list(daily_returns), dtype=float)
    if returns.size < 2:
        return 0.0
    daily_rf = risk_free_rate / trading_days_per_year
    excess = returns - daily_rf
    std = np.std(excess, ddof=0)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(trading_days_per_year))


def max_drawdown_pct(drawdowns: Sequence[float]) -> float:
    """
    Sprint 5 Section 9: Maximum Drawdown = min(drawdown) over the whole
    series, expressed as a percentage (drawdowns are already fractional,
    e.g. -0.08 for an 8% drop from the running peak).
    """
    values = list(drawdowns)
    if not values:
        return 0.0
    return float(min(values) * 100.0)


def win_rate_pct(winning_trades: int, losing_trades: int) -> float:
    """
    Sprint 5 Section 10: Winning trades / total CLOSED trades * 100.
    HOLD is never a trade (enforced by the caller, which only ever
    increments winning_trades/losing_trades for a completed BUY->SELL
    round trip - see backtester.py). Returns 0.0 (never NaN) when no
    trade has closed yet.
    """
    total_closed = winning_trades + losing_trades
    if total_closed == 0:
        return 0.0
    return (winning_trades / total_closed) * 100.0
