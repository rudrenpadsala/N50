"""
src/backtesting/backtester.py

Phase 1 - Sprint 5 - Step 3: Runs a TRAINED, FROZEN policy through the
existing `TradingEnvironment` (src/environment/trading_env.py) over the
TESTING period only, and records everything needed to compute the
Section 6 metrics. This does not create a second environment or a
second set of action rules - it is the exact same `TradingEnvironment`
Sprint 2/3/4 already use, just fed a policy array instead of a learning
agent, and never asked to update anything.

EVALUATION ONLY - NO LEARNING
-------------------------------
`run_backtest()` only ever calls `policy[state]` to choose an action and
`env.step(action)` to execute it. It never touches a Q-table, never
calls anything resembling `.update()` or `.learn()`, and the policy
array handed in is read-only (see `model_loader.load_greedy_policy`,
which returns a `.setflags(write=False)` array) - so it is
mechanically impossible for a backtest to retrain a model, update a
Q-table, or change a policy (Sprint 5 Sections 2, 35).

WHAT GETS RECORDED
---------------------
- Trade log (Sprint 5 Section 12): one row per VALID, EXECUTED
  BUY/SELL (never HOLD, and never an invalid no-op action) - Date,
  Company, Algorithm, Action, Price, Shares, Cash, Portfolio_Value,
  Reward, Position.
- Portfolio history (Sprint 5 Section 13): one row per trading day in
  the test period - Date, Portfolio_Value, Daily_Return, Drawdown -
  regardless of what action was taken that day.

TRADE / WIN-LOSS DEFINITION (Sprint 5 Section 10)
----------------------------------------------------
A "completed trade" is a valid SELL that closes against an earlier
valid BUY, matched FIFO (oldest open BUY first). Its outcome is a WIN
if the SELL's execution price is higher than the matched BUY's
execution price, else a LOSS. HOLD actions and invalid (no-op) BUY/SELL
actions are never counted as trades, and an unmatched open BUY at the
end of the test period is left open (neither a win nor a loss) - this
one definition is used for every algorithm and for Buy & Hold, so
Win Rate is directly comparable across all of them.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.environment.actions import HOLD, BUY, SELL, action_name
from src.environment.trading_env import TradingEnvironment
from src.environment.state import StateEncoder
from src.backtesting import metrics as m


@dataclass
class BacktestResult:
    company: str
    algorithm: str
    trade_log: list = field(default_factory=list)
    portfolio_history: list = field(default_factory=list)
    total_reward: float = 0.0
    num_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    initial_portfolio_value: float = 0.0
    final_portfolio_value: float = 0.0

    def metrics(self, risk_free_rate: float, trading_days_per_year: int) -> dict:
        first_date = self.portfolio_history[0]["Date"]
        last_date = self.portfolio_history[-1]["Date"]
        num_calendar_days = max((pd.Timestamp(last_date) - pd.Timestamp(first_date)).days, 1)
        daily_returns = [row["Daily_Return"] for row in self.portfolio_history]
        drawdowns = [row["Drawdown"] for row in self.portfolio_history]

        return {
            "Company": self.company,
            "Algorithm": self.algorithm,
            "Initial Capital": self.initial_portfolio_value,
            "Final Portfolio Value": self.final_portfolio_value,
            "Total Return %": m.total_return_pct(self.initial_portfolio_value, self.final_portfolio_value),
            "Annualized Return %": m.annualized_return_pct(
                self.initial_portfolio_value, self.final_portfolio_value, num_calendar_days
            ),
            "Sharpe Ratio": m.sharpe_ratio(daily_returns, risk_free_rate, trading_days_per_year),
            "Maximum Drawdown %": m.max_drawdown_pct(drawdowns),
            "Number of Trades": self.num_trades,
            "Winning Trades": self.winning_trades,
            "Losing Trades": self.losing_trades,
            "Win Rate %": m.win_rate_pct(self.winning_trades, self.losing_trades),
            "Total Reward": self.total_reward,
        }


def run_backtest(
    symbol: str,
    algorithm: str,
    policy: np.ndarray,
    test_df: pd.DataFrame,
    initial_capital: float,
    transaction_cost: float,
    invalid_action_penalty: float,
    shares_per_trade: int = 1,
    encoder: StateEncoder = None,
) -> BacktestResult:
    """
    Run `policy` (a fixed, read-only array: policy[state] -> action)
    chronologically over `test_df` (already restricted to the TESTING
    period by the caller - see scripts/run_backtest.py) using the
    existing TradingEnvironment, and return a BacktestResult.
    """
    encoder = encoder or StateEncoder()
    env = TradingEnvironment(
        df=test_df,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        invalid_action_penalty=invalid_action_penalty,
        shares_per_trade=shares_per_trade,
        state_encoder=encoder,
        symbol=symbol,
    )

    state = env.reset()
    result = BacktestResult(
        company=symbol, algorithm=algorithm,
        initial_portfolio_value=env.initial_capital,
    )

    running_peak = env.portfolio_value
    prev_portfolio_value = env.portfolio_value
    open_buy_prices = deque()  # FIFO queue of unmatched BUY execution prices

    # Day-zero portfolio row, so a portfolio that never trades still has
    # a well-defined history and dates for metrics().
    first_date = env.df.loc[env.current_step, "Date"]

    while True:
        execution_date = env.df.loc[env.current_step, "Date"]
        execution_price = float(env.df.loc[env.current_step, "Close"])
        action = int(policy[state])

        next_state, reward, done, info = env.step(action)  # environment only - no learning call anywhere here
        result.total_reward += reward

        if action != HOLD and not info["invalid_action"]:
            result.num_trades += 1
            if action == BUY:
                open_buy_prices.append(execution_price)
            elif action == SELL and open_buy_prices:
                entry_price = open_buy_prices.popleft()
                if execution_price > entry_price:
                    result.winning_trades += 1
                else:
                    result.losing_trades += 1

            result.trade_log.append({
                "Date": pd.Timestamp(execution_date),
                "Company": symbol,
                "Algorithm": algorithm,
                "Action": action_name(action),
                "Price": execution_price,
                "Shares": env.shares,
                "Cash": env.cash,
                "Portfolio_Value": env.portfolio_value,
                "Reward": reward,
                "Position": env.position,
            })

        running_peak = max(running_peak, env.portfolio_value)
        drawdown = (env.portfolio_value - running_peak) / running_peak if running_peak > 0 else 0.0
        daily_return = (
            (env.portfolio_value - prev_portfolio_value) / prev_portfolio_value
            if prev_portfolio_value > 0 else 0.0
        )
        result.portfolio_history.append({
            "Date": pd.Timestamp(info["date"]),
            "Portfolio_Value": env.portfolio_value,
            "Daily_Return": daily_return,
            "Drawdown": drawdown,
        })
        prev_portfolio_value = env.portfolio_value

        state = next_state
        if done:
            break

    result.final_portfolio_value = env.portfolio_value
    if not result.portfolio_history:
        # Defensive only - TradingEnvironment already requires >= 2 rows,
        # so this branch is unreachable in practice.
        result.portfolio_history.append({
            "Date": first_date, "Portfolio_Value": env.portfolio_value,
            "Daily_Return": 0.0, "Drawdown": 0.0,
        })

    return result
