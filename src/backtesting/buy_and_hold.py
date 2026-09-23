"""
src/backtesting/buy_and_hold.py

Phase 1 - Sprint 5 - Step 5: Buy & Hold baseline.

At the start of the TESTING period, spend the initial capital on the
maximum whole number of shares affordable (including transaction
cost), then hold for the entire test period with no further trades.
This deliberately does NOT reuse `TradingEnvironment` (which always
trades exactly `shares_per_trade` shares per BUY/SELL - Sprint 2/3/4's
fixed unit-trade convention): Buy & Hold is a fundamentally different
strategy (spend everything, once) rather than a 6th policy playing the
same discrete-action game, so it needs its own direct calculation - not
a duplicate trading environment, just the closed-form arithmetic for
this specific, non-discrete strategy.

Uses ONLY the first test-period day's price to size the position -
never a future price (Sprint 5 Section 5: "Do not use future
information to determine the number of shares.").

Returns the SAME `BacktestResult` shape as
`src/backtesting/backtester.py`'s `run_backtest`, so it slots into the
same downstream reporting/metrics code without any special-casing.
"""

import pandas as pd

from src.backtesting.backtester import BacktestResult


def run_buy_and_hold(symbol: str, test_df: pd.DataFrame,
                      initial_capital: float, transaction_cost: float) -> BacktestResult:
    df = test_df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
    if len(df) < 2:
        raise ValueError(f"Not enough test-period rows to run Buy & Hold for {symbol}")

    first_price = float(df.loc[0, "Close"])
    cost_per_share = first_price * (1 + transaction_cost)
    shares = int(initial_capital // cost_per_share)  # maximum whole shares affordable
    spent = shares * cost_per_share
    cash = initial_capital - spent

    result = BacktestResult(
        company=symbol, algorithm="Buy & Hold",
        initial_portfolio_value=initial_capital,
    )

    result.trade_log.append({
        "Date": pd.Timestamp(df.loc[0, "Date"]),
        "Company": symbol,
        "Algorithm": "Buy & Hold",
        "Action": "BUY",
        "Price": first_price,
        "Shares": shares,
        "Cash": cash,
        "Portfolio_Value": cash + shares * first_price,
        "Reward": 0.0,
        "Position": "HOLDING" if shares > 0 else "NO_POSITION",
    })
    result.num_trades = 1  # the single opening purchase; never counted as a "win" or "loss" - see module docstring

    running_peak = initial_capital
    prev_value = initial_capital
    for i in range(len(df)):
        price = float(df.loc[i, "Close"])
        portfolio_value = cash + shares * price
        running_peak = max(running_peak, portfolio_value)
        drawdown = (portfolio_value - running_peak) / running_peak if running_peak > 0 else 0.0
        daily_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0.0
        result.portfolio_history.append({
            "Date": pd.Timestamp(df.loc[i, "Date"]),
            "Portfolio_Value": portfolio_value,
            "Daily_Return": daily_return,
            "Drawdown": drawdown,
        })
        prev_value = portfolio_value

    result.final_portfolio_value = result.portfolio_history[-1]["Portfolio_Value"]
    result.total_reward = sum(row["Daily_Return"] for row in result.portfolio_history)
    # winning_trades / losing_trades stay 0: the position is never closed during
    # the test period, so Win Rate is 0.0 for Buy & Hold by the same "completed
    # trade" definition used everywhere else (see backtester.py docstring) -
    # not a penalty, just "not applicable, no round trip occurred."
    return result
