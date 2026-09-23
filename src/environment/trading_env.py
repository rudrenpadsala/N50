"""
src/environment/trading_env.py

Sprint 2 - Steps 8-22: The trading environment.

The environment moves forward exactly one trading day per step() call,
never backward, never shuffled. It consumes a pre-computed feature
dataframe (see src/features/build_features.py) for a single company and
a single date range (train slice or test slice) - it never fits or
recomputes feature thresholds itself, so pointing it at test-period data
for evaluation (Sprint 5) cannot leak into feature engineering.

Episode step timing (t = current_step index into the dataframe):
    1. Action is taken using the price at day t (`current_price`).
    2. The environment advances to day t+1.
    3. The resulting portfolio is marked-to-market at day t+1's price
       (or day t's price again if t+1 doesn't exist / episode is done).
    4. Reward = (new_portfolio_value - previous_portfolio_value) / previous_portfolio_value,
       plus a small penalty if the action taken at day t was invalid
       (BUY with insufficient cash, SELL with zero shares).
    5. The state returned is built from day t+1's features + the new position.

This means reward at step t only ever depends on information available
after acting at day t (day t's and day t+1's already-realized prices) -
never on anything beyond day t+1.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.environment.actions import HOLD, BUY, SELL, is_valid_action, action_name
from src.environment.state import State, StateEncoder, NO_POSITION, HOLDING

REQUIRED_COLUMNS = ["Date", "Close", "Trend", "RSI_Condition", "Volatility_Condition"]


class TradingEnvironment:
    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.001,
        invalid_action_penalty: float = -0.001,
        shares_per_trade: int = 1,
        state_encoder: Optional[StateEncoder] = None,
        symbol: str = "",
    ):
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Feature dataframe missing required column(s): {missing}")

        # Only rows with fully warmed-up features can be used (no NaN
        # Trend/RSI_Condition/Volatility_Condition rows - see build_features.py).
        clean_df = df.dropna(subset=["Trend", "RSI_Condition", "Volatility_Condition"]).copy()
        clean_df = clean_df.sort_values("Date").reset_index(drop=True)
        if not clean_df["Date"].is_monotonic_increasing:
            raise ValueError("Environment requires a chronologically sorted, non-shuffled dataframe")
        if len(clean_df) < 2:
            raise ValueError("Not enough warmed-up rows to run an episode (need at least 2)")

        self.df = clean_df
        self.symbol = symbol
        self.initial_capital = float(initial_capital)
        self.transaction_cost = float(transaction_cost)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.shares_per_trade = int(shares_per_trade)
        self.state_encoder = state_encoder or StateEncoder()

        # Episode-tracking attributes, set by reset().
        self.current_step = 0
        self.cash = self.initial_capital
        self.shares = 0
        self.position = NO_POSITION
        self.portfolio_value = self.initial_capital
        self.previous_portfolio_value = self.initial_capital
        self.done = False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> int:
        """Start a fresh episode at the beginning of the dataset. Returns the initial state ID."""
        self.current_step = 0
        self.cash = self.initial_capital
        self.shares = 0
        self.position = NO_POSITION
        self.portfolio_value = self.initial_capital
        self.previous_portfolio_value = self.initial_capital
        self.done = False
        return self.get_state()

    def step(self, action: int):
        """
        Apply `action` at the current day, advance one trading day, and
        return (next_state, reward, done, info).
        """
        if self.done:
            raise RuntimeError("step() called after episode termination; call reset() first")
        if not is_valid_action(action):
            raise ValueError(f"Undefined action: {action!r}")

        current_price = float(self.df.loc[self.current_step, "Close"])
        self.previous_portfolio_value = self.portfolio_value

        invalid = self._apply_action(action, current_price)

        # Advance one trading day (never backward, never shuffled).
        next_step = self.current_step + 1
        if next_step >= len(self.df):
            # No further data: episode ends, mark-to-market at the last known price.
            mark_price = current_price
            self.done = True
        else:
            mark_price = float(self.df.loc[next_step, "Close"])
            self.current_step = next_step
            self.done = self.current_step >= len(self.df) - 1

        self.portfolio_value = self.cash + self.shares * mark_price

        reward = (self.portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
        if invalid:
            reward += self.invalid_action_penalty

        next_state = self.get_state()

        info = {
            "date": self.df.loc[self.current_step, "Date"],
            "price": mark_price,
            "action": action_name(action),
            "cash": self.cash,
            "shares": self.shares,
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "invalid_action": invalid,
        }

        return next_state, reward, self.done, info

    def get_state(self) -> int:
        """Return the integer-encoded state for the current step + current position."""
        row = self.df.loc[self.current_step]
        state = State(
            trend=row["Trend"],
            rsi_condition=row["RSI_Condition"],
            volatility_condition=row["Volatility_Condition"],
            position=self.position,
        )
        return self.state_encoder.encode(state)

    def get_portfolio_value(self) -> float:
        return self.portfolio_value

    def is_done(self) -> bool:
        return self.done

    # ------------------------------------------------------------------
    # Internal action logic
    # ------------------------------------------------------------------

    def _apply_action(self, action: int, price: float) -> bool:
        """
        Execute BUY/HOLD/SELL at `price`. Returns True if the action was
        invalid (no-op), False otherwise. Never allows negative cash or
        negative shares.
        """
        if action == HOLD:
            return False

        if action == BUY:
            cost = price * self.shares_per_trade * (1 + self.transaction_cost)
            if self.cash >= cost:
                self.cash -= cost
                self.shares += self.shares_per_trade
                self.position = HOLDING
                return False
            return True  # insufficient cash: invalid no-op

        if action == SELL:
            if self.shares >= self.shares_per_trade:
                proceeds = price * self.shares_per_trade * (1 - self.transaction_cost)
                self.cash += proceeds
                self.shares -= self.shares_per_trade
                self.position = HOLDING if self.shares > 0 else NO_POSITION
                return False
            return True  # no shares to sell: invalid no-op

        raise ValueError(f"Undefined action: {action!r}")
