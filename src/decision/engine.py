"""
src/decision/engine.py

Dashboard redesign - "INTERNAL MODEL LOGIC": combines the 7 already-
trained RL algorithms (Q-Learning, SARSA, Monte Carlo, Value Iteration,
Policy Iteration, Expected SARSA, Double Q-Learning) into ONE final
BUY/HOLD/SELL recommendation per company, via majority voting, so the
normal-user dashboard never has to show an algorithm name, a Q-table,
or a Sharpe Ratio. Expected SARSA and Double Q-Learning were added
after the original 5 specifically to make this ensemble's BUY/HOLD/SELL
vote more accurate: more independent, low-variance/bias-corrected
voters make the majority vote less sensitive to any single algorithm's
noise (see src/rl/expected_sarsa.py / src/rl/double_q_learning.py for
why each one was chosen). Nothing below needed to change to support
them - `ALGORITHMS` (imported from model_loader) already drives every
loop in this file, so it works unmodified for however many algorithms
have trained models.

This module does not retrain, backtest, or otherwise modify Sprints
1-5 - it only READS:
    - the company's latest feature row (data/features/<SYMBOL>_features.csv)
    - each algorithm's already-trained Q-table (src/backtesting/model_loader.py)
    - each algorithm's most recent simulated position, if a completed
      Sprint 5 backtest exists for this company (reports/trades/) -
      see `get_current_position` for the documented fallback when it
      doesn't.

VOTING RULE
-----------
Each of the (up to 7) algorithms proposes its own greedy action
(HOLD/BUY/SELL) for its own current state. The final decision is
whichever action gets the most votes.

TIE-BREAK RULE (documented, deterministic - never random)
-----------------------------------------------------------
With 7 voters and 3 actions a tie can take more shapes than the old
5-voter 2-2-1 case (e.g. 3-3-1, 3-2-2) - `combine_votes` below handles
any of them generically via `max_votes`/`tied_actions`, so the exact
split never matters. Two cases:
  1. HOLD is one of the tied top actions -> HOLD wins. HOLD carries no
     transaction cost and no execution risk, making it the appropriate
     conservative default whenever the internal models disagree.
  2. The tie is BUY vs SELL only (HOLD is not among the tied top
     actions) -> the action with the higher AVERAGE Q-value across the
     participating algorithms' own Q-tables (each evaluated at that
     algorithm's own current state) wins - reusing the same Bellman
     values Sprints 3-4 already computed, rather than an arbitrary coin
     flip.
If fewer than 7 algorithms have a trained model for this company, the
same rules apply over however many voted (documented, not hidden -
`get_decision`'s `skipped_algorithms` records which ones were missing).

CONFIDENCE
----------
Confidence % = (votes for the final action / number of algorithms that
voted) * 100 - the tie-break decides WHICH action wins, not how
confident the ensemble is in it, so a tie-broken decision is always
reported at its true (lower) vote share.

    80-100% -> "High"
    60-79%  -> "Medium"
    below 60% -> "Low"

SIGNAL-STRENGTH DOWNGRADE
--------------------------
Vote share alone can overstate confidence: this project's Q-values are
bounded by a single day's fractional portfolio return, so they are
naturally small, and an algorithm's top action can beat its own
second-best action by a razor-thin margin - a near coin flip rather
than a real learned preference. If that happens for most of the
algorithms that voted for the winning action (their average top-2
Q-value margin falls below MARGIN_THRESHOLD, see below), the reported
confidence label is downgraded one tier (High -> Medium, Medium ->
Low). This never changes WHICH action wins, only how confidently that
win is reported, and is exposed as `weak_signal` / `avg_winning_margin`
in `get_decision()`'s result.

POSITION ASSUMPTION
--------------------
Each algorithm's "current state" needs a Position bit (NO_POSITION or
HOLDING) matching what it was trained on. Since the dashboard doesn't
track whether the user personally owns shares, this uses each
algorithm's own MOST RECENT SIMULATED POSITION from its Sprint 5
backtest trade log - i.e. "if this algorithm had been trading this
stock throughout the test period, what position would it be in today,
and what would it do next?" - falling back to NO_POSITION only when no
backtest trade log exists yet for that company/algorithm.
"""

import os

import numpy as np
import pandas as pd

from src.environment.actions import HOLD, BUY, SELL, ACTION_NAMES, NUM_ACTIONS
from src.environment.state import State, StateEncoder, NO_POSITION
from src.backtesting.model_loader import ALGORITHMS, load_q_table, ModelLoadError, file_stub
from src.decision import live_features

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
TRADES_DIR = os.path.join(REPORTS_DIR, "trades")

CONFIDENCE_THRESHOLDS = ((80.0, "High"), (60.0, "Medium"), (0.0, "Low"))

# Signal-strength check (in addition to vote-share confidence above).
# Q-values in this project are bounded by a single day's fractional
# portfolio return (see src/rl/mdp.py / the reward definition), so they
# are naturally small in magnitude - it is common for an algorithm's
# top action and its next-best alternative to differ by a tiny amount.
# When that happens, the "winning" action for that algorithm is close
# to a coin flip rather than a real learned preference, and a vote
# count alone (e.g. "3 of 5 voted SELL") can look more decisive than
# it actually is if those 3 votes were themselves near-ties. This
# threshold - the minimum average gap, for the algorithms that voted
# for the FINAL action, between their top action's Q-value and their
# own second-best action's Q-value - downgrades the reported confidence
# by one tier (High->Medium, Medium->Low) whenever that gap is smaller
# than this. It never changes WHICH action wins, only how confidently
# that win is reported.
MARGIN_THRESHOLD = 0.001
CONFIDENCE_DOWNGRADE = {"High": "Medium", "Medium": "Low", "Low": "Low"}

# Simple-indicator thresholds (documented here, used by build_reasons /
# app.py) - separate from Sprint 2's own RSI_LOW/RSI_HIGH thresholds
# (30/70), which classify RSI for the RL STATE space, not for this
# "momentum" label. RSI_OVERSOLD/OVERBOUGHT below reuse the SAME 30/70
# values as Sprint 2 for the user-facing RSI text (they happen to match
# the RL state thresholds here, but are defined independently since
# they serve a different, display-only purpose).
MOMENTUM_POSITIVE_THRESHOLD = 55.0
MOMENTUM_NEGATIVE_THRESHOLD = 45.0
RSI_OVERSOLD_THRESHOLD = 30.0
RSI_OVERBOUGHT_THRESHOLD = 70.0

DECISION_MESSAGES = {
    BUY: "The AI analysis currently indicates a BUY signal.",
    HOLD: "The AI analysis currently suggests waiting and holding.",
    SELL: "The AI analysis currently indicates a SELL signal.",
}


class DecisionError(Exception):
    """Raised when no final decision can be produced for a company (e.g. no trained models at all)."""


def _none_if_nan(value):
    """pd.isna-safe unwrap: NaN/None -> None, otherwise a plain float.
    Used for the live-only optional fields (MACD, volume, etc.) that a
    historical-fallback row simply won't have."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def confidence_label(confidence_pct: float) -> str:
    for threshold, label in CONFIDENCE_THRESHOLDS:
        if confidence_pct >= threshold:
            return label
    return "Low"


def action_margin(q_row) -> float:
    """Gap between an algorithm's top action's Q-value and its own
    second-best action's Q-value, for one state. A small margin means
    that algorithm barely prefers its chosen action over the alternative -
    see MARGIN_THRESHOLD above."""
    sorted_q = sorted(q_row, reverse=True)
    return float(sorted_q[0] - sorted_q[1])


def list_companies() -> list:
    if not os.path.isdir(FEATURES_DIR):
        return []
    return sorted(
        f[: -len("_features.csv")] for f in os.listdir(FEATURES_DIR) if f.endswith("_features.csv")
    )


def get_latest_feature_row(symbol: str) -> pd.Series:
    """Most recent row with a fully-computed state (Trend/RSI/Volatility not NaN)."""
    path = os.path.join(FEATURES_DIR, f"{symbol}_features.csv")
    if not os.path.isfile(path):
        raise DecisionError(f"No feature data found for {symbol}")
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    clean = df.dropna(subset=["Trend", "RSI_Condition", "Volatility_Condition"])
    if clean.empty:
        raise DecisionError(f"No usable feature rows for {symbol}")
    return clean.iloc[-1]


def get_previous_close(symbol: str, latest_date) -> float:
    path = os.path.join(FEATURES_DIR, f"{symbol}_features.csv")
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    earlier = df[df["Date"] < pd.Timestamp(latest_date)]
    if earlier.empty:
        return float("nan")
    return float(earlier.iloc[-1]["Close"])


def get_current_position(symbol: str, algorithm: str) -> int:
    """
    This algorithm's most recent simulated position from its Sprint 5
    backtest trade log, or NO_POSITION if no backtest trades exist yet
    for this company/algorithm (never backtested, or the algorithm
    never executed a single trade during the test period).
    """
    path = os.path.join(TRADES_DIR, f"{symbol}_{file_stub(algorithm)}_trades.csv")
    if not os.path.isfile(path):
        return NO_POSITION
    trades = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    if trades.empty:
        return NO_POSITION
    return int(trades.iloc[-1]["Position"])


def is_holding_position(symbol: str) -> bool:
    """
    Whether this company is currently considered 'held', for the News &
    Risk Layer's SELL-vs-AVOID distinction (see src/risk/decision_guard.py).
    This project has no real user portfolio - it's a majority vote of
    the trained algorithms' own simulated Sprint 5 backtest positions (see
    get_current_position above), same source the individual algorithm
    predictions already use for their own state's Position bit.
    """
    holding_count = sum(
        1 for algorithm in ALGORITHMS if get_current_position(symbol, algorithm) != NO_POSITION
    )
    return holding_count > len(ALGORITHMS) / 2


def predict_algorithm_action(symbol: str, algorithm: str, latest_row: pd.Series, encoder: StateEncoder):
    """Returns (action:int, q_row:np.ndarray) for one algorithm. Raises ModelLoadError if untrained."""
    position = get_current_position(symbol, algorithm)
    state = State(
        trend=latest_row["Trend"], rsi_condition=latest_row["RSI_Condition"],
        volatility_condition=latest_row["Volatility_Condition"], position=position,
    )
    state_id = encoder.encode(state)
    q_table = load_q_table(MODELS_DIR, symbol, algorithm, encoder.num_states, NUM_ACTIONS)
    q_row = q_table[state_id]
    return int(np.argmax(q_row)), q_row


def combine_votes(actions_by_algorithm: dict, q_rows_by_algorithm: dict) -> dict:
    """
    actions_by_algorithm: {algorithm_name: action_int}
    q_rows_by_algorithm:  {algorithm_name: q_row} for the SAME algorithms

    Returns {"action", "votes" ({0,1,2}->count), "confidence_pct",
    "confidence_label", "tie_broken", "weak_signal", "avg_winning_margin"}.
    """
    votes = {HOLD: 0, BUY: 0, SELL: 0}
    for action in actions_by_algorithm.values():
        votes[action] += 1

    max_votes = max(votes.values())
    tied_actions = [a for a, v in votes.items() if v == max_votes]
    tie_broken = len(tied_actions) > 1

    if not tie_broken:
        final_action = tied_actions[0]
    elif HOLD in tied_actions:
        final_action = HOLD
    else:
        avg_q = {
            a: float(np.mean([q_rows_by_algorithm[algo][a] for algo in q_rows_by_algorithm]))
            for a in tied_actions
        }
        final_action = max(avg_q, key=avg_q.get)

    total_votes = sum(votes.values())
    confidence_pct = (votes[final_action] / total_votes) * 100.0 if total_votes else 0.0
    label = confidence_label(confidence_pct)

    # Signal-strength downgrade - see MARGIN_THRESHOLD docstring above.
    winning_margins = [
        action_margin(q_rows_by_algorithm[algo])
        for algo, action in actions_by_algorithm.items() if action == final_action
    ]
    avg_winning_margin = float(np.mean(winning_margins)) if winning_margins else 0.0
    weak_signal = avg_winning_margin < MARGIN_THRESHOLD
    if weak_signal:
        label = CONFIDENCE_DOWNGRADE[label]

    return {
        "action": final_action,
        "votes": dict(votes),
        "confidence_pct": confidence_pct,
        "confidence_label": label,
        "tie_broken": tie_broken,
        "weak_signal": weak_signal,
        "avg_winning_margin": avg_winning_margin,
    }


def get_decision(symbol: str, prefer_live: bool = True) -> dict:
    """
    Full pipeline for one company: latest market data + every trained
    algorithm's vote -> one final decision. Raises DecisionError if NO
    algorithm has a trained model for this company.

    DATA SOURCE (result["data_source"] is "live" or "historical")
    -----------------------------------------------------------------
    When `prefer_live` (default), this first tries to compute TODAY's
    actual Trend/RSI/Volatility state from live Angel One daily candles
    (see src/decision/live_features.py) - the same indicator formulas
    used at training time, just fed with a live-fetched price series
    instead of the static dataset. If that isn't possible for any
    reason (Angel One not configured, market/company not mapped,
    network failure, not enough live history yet to fill the 20-day
    warm-up window), it falls back to the static historical feature
    file's last row (frozen at whatever date that dataset ends on) -
    result["live_unavailable_reason"] explains why in that case.
    """
    encoder = StateEncoder()

    data_source = "historical"
    live_unavailable_reason = None
    previous_close = None
    latest_row = None
    instrument_info = None

    if prefer_live:
        live_row, reason = live_features.get_live_feature_row(symbol)
        if live_row is not None:
            latest_row = live_row
            data_source = "live"
            previous_close = live_row.get("PreviousClose")
            instrument_info = {
                "exchange": live_row.get("InstrumentExchange"),
                "tradingsymbol": live_row.get("InstrumentTradingSymbol"),
                "is_future": bool(live_row.get("InstrumentIsFuture")),
                "expiry": live_row.get("InstrumentExpiry"),
            }
        else:
            live_unavailable_reason = reason

    if latest_row is None:
        latest_row = get_latest_feature_row(symbol)
        previous_close = get_previous_close(symbol, latest_row["Date"])

    actions, q_rows, skipped = {}, {}, []
    for algorithm in ALGORITHMS:
        try:
            action, q_row = predict_algorithm_action(symbol, algorithm, latest_row, encoder)
            actions[algorithm] = action
            q_rows[algorithm] = q_row
        except ModelLoadError as exc:
            skipped.append({"algorithm": algorithm, "reason": str(exc)})

    if not actions:
        raise DecisionError(
            f"No trained models available for {symbol} yet - train models before requesting a decision."
        )

    result = combine_votes(actions, q_rows)
    result.update({
        "company": symbol,
        "latest_price": float(latest_row["Close"]),
        "latest_date": pd.Timestamp(latest_row["Date"]),
        "previous_close": previous_close,
        "trend": latest_row["Trend"],
        "rsi_condition": latest_row["RSI_Condition"],
        "rsi_value": float(latest_row["RSI"]),
        "volatility_condition": latest_row["Volatility_Condition"],
        "ma5": float(latest_row["MA5"]),
        "ma20": float(latest_row["MA20"]),
        "volatility_value": None if pd.isna(latest_row.get("Volatility")) else float(latest_row["Volatility"]),
        "data_source": data_source,                           # "live" or "historical"
        "live_unavailable_reason": live_unavailable_reason,    # None when data_source == "live"
        "instrument_info": instrument_info,                    # None when data_source == "historical"
        # MACD/Volume/High/Low - live-only display/reasoning indicators
        # (see indicators.compute_all_features's docstring: these are NOT
        # part of the trained models' 36-state space, so they're None
        # whenever data_source == "historical").
        "macd": _none_if_nan(latest_row.get("MACD")),
        "macd_signal": _none_if_nan(latest_row.get("MACD_Signal")),
        "macd_condition": latest_row.get("MACD_Condition") if pd.notna(latest_row.get("MACD_Condition")) else None,
        "today_high": _none_if_nan(latest_row.get("High")),
        "today_low": _none_if_nan(latest_row.get("Low")),
        "volume": _none_if_nan(latest_row.get("Volume")),
        "previous_volume": _none_if_nan(latest_row.get("PreviousVolume")),
        "volume_change_pct": _none_if_nan(latest_row.get("VolumeChangePct")),
        "recent_high": _none_if_nan(latest_row.get("RecentHigh")),
        "drawdown_pct": _none_if_nan(latest_row.get("DrawdownPct")),
        "actions_by_algorithm": actions,     # Developer Analysis only
        "q_rows_by_algorithm": q_rows,       # Developer Analysis only
        "skipped_algorithms": skipped,       # Developer Analysis only
    })
    return result


# ---------------------------------------------------------------------------
# Simple, non-technical labels (used by both the decision above and the
# Stock Decision / Stock History pages directly)
# ---------------------------------------------------------------------------

def classify_price_trend(trend: str) -> str:
    return {"BULLISH": "UP", "BEARISH": "DOWN", "NEUTRAL": "SIDEWAYS"}.get(trend, "SIDEWAYS")


def classify_momentum(rsi_value: float) -> str:
    if rsi_value > MOMENTUM_POSITIVE_THRESHOLD:
        return "POSITIVE"
    if rsi_value < MOMENTUM_NEGATIVE_THRESHOLD:
        return "NEGATIVE"
    return "NEUTRAL"


def classify_volatility_simple(volatility_condition: str) -> str:
    """This project's feature engineering only has 2 volatility buckets
    (LOW_VOLATILITY / HIGH_VOLATILITY - see src/features/indicators.py),
    so there is no MEDIUM level to show without recomputing features."""
    return "HIGH" if volatility_condition == "HIGH_VOLATILITY" else "LOW"


def rsi_text(rsi_value: float) -> str:
    if rsi_value < RSI_OVERSOLD_THRESHOLD:
        return "Potentially oversold"
    if rsi_value > RSI_OVERBOUGHT_THRESHOLD:
        return "Potentially overbought"
    return "Neutral range"


def build_reasons(decision: dict) -> list:
    """3-6 plain-language reasons behind `decision` (from get_decision()).
    Returns a list of (icon, title, text) tuples."""
    reasons = []

    trend_label = classify_price_trend(decision["trend"])
    if trend_label == "UP":
        reasons.append(("📈", "Price Trend", "The recent price trend is positive."))
    elif trend_label == "DOWN":
        reasons.append(("📉", "Price Trend", "The recent price trend is negative."))
    else:
        reasons.append(("➡️", "Price Trend", "The recent price trend is sideways."))

    if decision["ma5"] > decision["ma20"]:
        reasons.append(("📊", "Short-Term Movement", "The short-term average is above the long-term average."))
    else:
        reasons.append(("📊", "Short-Term Movement", "The short-term average is below the long-term average."))

    momentum = classify_momentum(decision["rsi_value"])
    if momentum == "POSITIVE":
        reasons.append(("⚡", "Market Momentum", "The stock currently shows positive momentum."))
    elif momentum == "NEGATIVE":
        reasons.append(("⚡", "Market Momentum", "The stock currently shows negative momentum."))
    else:
        reasons.append(("⚡", "Market Momentum", "The stock currently shows steady, neutral momentum."))

    macd_condition = decision.get("macd_condition")
    if macd_condition == "POSITIVE_MOMENTUM":
        reasons.append(("〰️", "MACD", "MACD is above its signal line, indicating positive momentum."))
    elif macd_condition == "NEGATIVE_MOMENTUM":
        reasons.append(("〰️", "MACD", "MACD is below its signal line, indicating negative momentum."))
    # else: MACD isn't available for this decision (historical fallback, or still
    # warming up on a newly-listed instrument) - omit rather than fabricate.

    action_name = ACTION_NAMES[decision["action"]]
    n_support = decision["votes"][decision["action"]]
    n_total = sum(decision["votes"].values())
    reasons.append(("🤖", "AI Analysis", f"{n_support} of {n_total} internal AI models currently support a {action_name} decision."))

    if decision.get("weak_signal"):
        reasons.append((
            "⚠️", "Signal Strength",
            "The internal models show little preference between actions for this stock right now - "
            "treat this decision with extra caution.",
        ))

    return reasons


# ---------------------------------------------------------------------------
# Stock History page support
# ---------------------------------------------------------------------------

def historical_decisions(symbol: str, num_days: int = 30) -> pd.DataFrame:
    """
    A SIMPLIFIED day-by-day illustration of what the ensemble would have
    suggested on each of the last `num_days` available days, assuming
    NO_POSITION every day (i.e. "what would the AI suggest if you were
    looking at this stock fresh that day") - this is NOT the audited,
    position-aware Sprint 5 backtest simulation (see `backtest_summary`
    / the Developer Analysis page for that). It exists purely as an
    illustrative history table for non-technical users, computed cheaply
    without re-running a full backtest.
    """
    encoder = StateEncoder()
    path = os.path.join(FEATURES_DIR, f"{symbol}_features.csv")
    if not os.path.isfile(path):
        return pd.DataFrame(columns=["Date", "Price", "AI Decision"])

    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    clean = df.dropna(subset=["Trend", "RSI_Condition", "Volatility_Condition"])
    recent = clean.tail(num_days)

    q_tables = {}
    for algorithm in ALGORITHMS:
        try:
            q_tables[algorithm] = load_q_table(MODELS_DIR, symbol, algorithm, encoder.num_states, NUM_ACTIONS)
        except ModelLoadError:
            continue
    if not q_tables:
        return pd.DataFrame(columns=["Date", "Price", "AI Decision"])

    rows = []
    for _, row in recent.iterrows():
        state = State(trend=row["Trend"], rsi_condition=row["RSI_Condition"],
                      volatility_condition=row["Volatility_Condition"], position=NO_POSITION)
        state_id = encoder.encode(state)
        actions = {algo: int(np.argmax(q[state_id])) for algo, q in q_tables.items()}
        q_rows = {algo: q[state_id] for algo, q in q_tables.items()}
        combined = combine_votes(actions, q_rows)
        rows.append({
            "Date": row["Date"], "Price": float(row["Close"]),
            "AI Decision": ACTION_NAMES[combined["action"]],
        })
    return pd.DataFrame(rows)


def backtest_summary(symbol: str) -> dict:
    """
    Simple historical-performance summary for `symbol`, AVERAGED across
    all trained RL algorithms' Sprint 5 backtest results (Buy & Hold excluded -
    it's the baseline, not one of the AI's own strategies). Returns None
    if reports/final_results.csv doesn't exist yet or has no rows for
    this company (i.e. the backtest pipeline hasn't been run).
    """
    path = os.path.join(REPORTS_DIR, "final_results.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    company_rows = df[(df["Company"] == symbol) & (df["Algorithm"] != "Buy & Hold")]
    if company_rows.empty:
        return None
    return {
        "starting_capital": float(company_rows["Initial Capital"].mean()),
        "ending_value": float(company_rows["Final Portfolio Value"].mean()),
        "return_pct": float(company_rows["Total Return %"].mean()),
        "max_drawdown_pct": float(company_rows["Maximum Drawdown %"].mean()),
        "num_algorithms": len(company_rows),
    }