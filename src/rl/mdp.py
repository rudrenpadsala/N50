"""
src/rl/mdp.py

Phase 1 - Sprint 4 - Steps 3, 4: Builds a tabular Markov Decision
Process (transition model P(s'|s,a) and reward model R(s,a)) purely
from TRAINING-period data, so Value Iteration and Policy Iteration
(src/rl/value_iteration.py, src/rl/policy_iteration.py) can run offline
against a fixed model instead of interacting with a live environment.

WHY AN EMPIRICAL MODEL IS NECESSARY
------------------------------------
The 36-state encoding (Trend, RSI_Condition, Volatility_Condition,
Position - see src/environment/state.py) intentionally discards a lot
of information (exact calendar date, exact cash balance, exact share
count beyond the binary Position flag) so that Q-Learning, SARSA and
Monte Carlo Control (Sprint 3) can learn a small, tabular Q(s,a). This
module reuses that SAME 36-state / 3-action abstraction (Sprint 4
Section 19: all five algorithms must be directly comparable) - but it
means that, at the abstracted state level, the same encoded state s can
legitimately be followed by different next states s' and different
rewards on different historical days, because real price movements are
not a deterministic function of (Trend, RSI, Volatility, Position)
alone.

Given that, this module treats the training-period rows as a fixed
dataset (never reordered, never sampled with an RNG) and estimates:

    P(s'|s,a) = count(s -> s' via a) / count(s via a)   [empirical frequency]
    R(s,a)    = mean observed reward for (s,a) across all its
                occurrences in the training data

This is a standard way to derive a tabular MDP approximation from a
fixed batch of transitions (a certainty-equivalence / empirical model).
Crucially, it never draws a random number, so re-running it on the same
training data always reproduces the exact same P and R tables - i.e.
the MODEL-BUILDING PROCESS is itself deterministic, matching the
underlying trading environment's own determinism (Sprint 4 Section 3).

HOW EACH TRANSITION SAMPLE IS OBSERVED
-----------------------------------------
For every day t in the training data (t = 0 .. T-2, so a "next day"
t+1 always exists inside the training slice) and for BOTH hypothetical
positions the agent could be in at day t (NO_POSITION and HOLDING -
both are considered for every day, since Trend/RSI/Volatility at day t
do not depend on position; doing both means the model isn't limited to
whichever position a single pass through the data happened to be in,
which would otherwise leave most (state, action) pairs with zero
observations), and for each of the 3 actions:

    1. Build a small, self-consistent hypothetical portfolio for that
       position (`_hypothetical_portfolio`) using ONLY day t's own
       Close price - never a future price.
    2. Apply the action using the IDENTICAL invalid-action rules as
       `TradingEnvironment._apply_action` (src/environment/
       trading_env.py) - BUY never creates negative cash, SELL never
       creates negative shares (Sprint 4 Section 9) - see
       `_apply_action` below, which mirrors that method exactly.
    3. Mark the resulting portfolio to market at day t+1's Close price
       (never a later price - no future leakage) and compute
           reward = (new_portfolio_value - previous_portfolio_value) / previous_portfolio_value
       plus the same invalid_action_penalty used by the environment -
       the IDENTICAL Sprint 2 reward definition (Sprint 4 Section 4),
       never a different one built specially for Value/Policy Iteration.
    4. Encode the resulting (Trend, RSI, Volatility, Position) at day
       t+1 as the next state s', using the SAME StateEncoder as the
       rest of the project.
    5. Record one occurrence of (s, a) -> (s', r).

Every one of these steps reads only day t and day t+1 of whatever
dataframe it is given - the testing period (2025-01-01 -> 2026-07-31)
is never read here at all, because the CALLER (scripts/
train_dynamic_programming.py) only ever passes in a dataframe already
restricted to the training window (Sprint 4 Sections 3, 12, 13).

FALLBACK FOR UNOBSERVED (state, action) PAIRS
-----------------------------------------------
Some (state, action) combinations may never occur in a given company's
training data (e.g. a rare Trend/RSI/Volatility combination). For
those, R(s,a) defaults to 0.0 and P(.|s,a) is a self-loop (probability
1.0 of staying in s) - a conservative default for an unvisited
state-action pair (it makes that estimated action neither attractive
nor punished, and never raises an error), applied identically for
Value Iteration and Policy Iteration.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.environment.actions import ACTIONS, NUM_ACTIONS, HOLD, BUY, SELL
from src.environment.state import State, StateEncoder, NO_POSITION, HOLDING

REQUIRED_COLUMNS = ["Date", "Close", "Trend", "RSI_Condition", "Volatility_Condition"]


@dataclass
class MDP:
    """
    Tabular MDP estimated from training data.

    transition_probs[s, a, s'] = P(s' | s, a)   shape (num_states, num_actions, num_states)
    rewards[s, a]               = R(s, a)        shape (num_states, num_actions)
    observed_counts[s, a]       = number of TRAINING-data occurrences of (s, a) used to
                                   estimate the row above (0 means the fallback self-loop
                                   described in the module docstring was used instead)
    """

    num_states: int
    num_actions: int
    transition_probs: np.ndarray = field(default=None)
    rewards: np.ndarray = field(default=None)
    observed_counts: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.transition_probs is None:
            self.transition_probs = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=float)
        if self.rewards is None:
            self.rewards = np.zeros((self.num_states, self.num_actions), dtype=float)
        if self.observed_counts is None:
            self.observed_counts = np.zeros((self.num_states, self.num_actions), dtype=np.int64)

    def is_observed(self, state: int, action: int) -> bool:
        return bool(self.observed_counts[state, action] > 0)


def _hypothetical_portfolio(position: int, price: float, initial_capital: float,
                             transaction_cost: float, shares_per_trade: int):
    """
    A self-consistent (cash, shares) pair for a hypothetical position at
    day t's price, used ONLY to determine action validity / resulting
    reward for MDP construction - never used to track a real episode.

        NO_POSITION -> cash = initial_capital, shares = 0
        HOLDING     -> as if `shares_per_trade` shares were bought at
                       day t's own price (never a future price)
    """
    if position == NO_POSITION:
        return float(initial_capital), 0
    cost = price * shares_per_trade * (1.0 + transaction_cost)
    cash = max(initial_capital - cost, 0.0)
    return cash, shares_per_trade


def _apply_action(action: int, price: float, cash: float, shares: int,
                   transaction_cost: float, shares_per_trade: int):
    """
    Mirrors TradingEnvironment._apply_action exactly (src/environment/
    trading_env.py) - same invalid-action rules, so Value/Policy
    Iteration are evaluated against identical constraints as
    Q-Learning/SARSA/Monte Carlo (Sprint 4 Section 9).

    Returns (new_cash, new_shares, invalid).
    """
    if action == HOLD:
        return cash, shares, False

    if action == BUY:
        cost = price * shares_per_trade * (1.0 + transaction_cost)
        if cash >= cost:
            return cash - cost, shares + shares_per_trade, False
        return cash, shares, True  # insufficient cash: invalid no-op

    if action == SELL:
        if shares >= shares_per_trade:
            proceeds = price * shares_per_trade * (1.0 - transaction_cost)
            return cash + proceeds, shares - shares_per_trade, False
        return cash, shares, True  # no shares to sell: invalid no-op

    raise ValueError(f"Undefined action: {action!r}")


def build_mdp(
    train_df: pd.DataFrame,
    encoder: StateEncoder,
    initial_capital: float,
    transaction_cost: float,
    invalid_action_penalty: float,
    shares_per_trade: int = 1,
) -> MDP:
    """
    Build an empirical tabular MDP purely from `train_df`.

    `train_df` must already be restricted to the TRAINING period by the
    caller (see scripts/train_dynamic_programming.py) - this function
    performs no date filtering itself and, for any row t it processes,
    never looks past row t+1.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in train_df.columns]
    if missing:
        raise ValueError(f"Training dataframe missing required column(s): {missing}")

    df = train_df.dropna(subset=["Trend", "RSI_Condition", "Volatility_Condition"]).copy()
    df = df.sort_values("Date").reset_index(drop=True)
    if not df["Date"].is_monotonic_increasing:
        raise ValueError("MDP construction requires a chronologically sorted, non-shuffled dataframe")
    if len(df) < 2:
        raise ValueError("Not enough warmed-up training rows to build an MDP (need at least 2)")

    num_states = encoder.num_states
    transition_counts = np.zeros((num_states, NUM_ACTIONS, num_states), dtype=np.int64)
    reward_sums = np.zeros((num_states, NUM_ACTIONS), dtype=float)

    for t in range(len(df) - 1):
        row_t = df.loc[t]
        row_t1 = df.loc[t + 1]
        price_t = float(row_t["Close"])
        price_t1 = float(row_t1["Close"])

        for position in (NO_POSITION, HOLDING):
            state = State(
                trend=row_t["Trend"],
                rsi_condition=row_t["RSI_Condition"],
                volatility_condition=row_t["Volatility_Condition"],
                position=position,
            )
            s = encoder.encode(state)
            cash0, shares0 = _hypothetical_portfolio(
                position, price_t, initial_capital, transaction_cost, shares_per_trade
            )
            previous_portfolio_value = cash0 + shares0 * price_t

            for action in ACTIONS:
                cash1, shares1, invalid = _apply_action(
                    action, price_t, cash0, shares0, transaction_cost, shares_per_trade
                )
                next_position = HOLDING if shares1 > 0 else NO_POSITION
                portfolio_value = cash1 + shares1 * price_t1

                reward = (portfolio_value - previous_portfolio_value) / previous_portfolio_value
                if invalid:
                    reward += invalid_action_penalty

                next_state = State(
                    trend=row_t1["Trend"],
                    rsi_condition=row_t1["RSI_Condition"],
                    volatility_condition=row_t1["Volatility_Condition"],
                    position=next_position,
                )
                s_next = encoder.encode(next_state)

                transition_counts[s, action, s_next] += 1
                reward_sums[s, action] += reward

    mdp = MDP(num_states=num_states, num_actions=NUM_ACTIONS)
    observed_counts = transition_counts.sum(axis=2)
    mdp.observed_counts = observed_counts

    for s in range(num_states):
        for a in ACTIONS:
            total = observed_counts[s, a]
            if total > 0:
                mdp.transition_probs[s, a] = transition_counts[s, a] / total
                mdp.rewards[s, a] = reward_sums[s, a] / total
            else:
                # Fallback: self-loop, zero reward (see module docstring).
                mdp.transition_probs[s, a, s] = 1.0
                mdp.rewards[s, a] = 0.0

    return mdp
