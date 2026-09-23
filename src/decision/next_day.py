"""
src/decision/next_day.py

Dashboard update - "Next Day Strategy" page support.

This module answers "what should I do for the next trading day?" for a
company, given a price the USER expects to enter at. It deliberately
separates two different things that must never be confused (see the
spec's "IMPORTANT TECHNICAL RULE"):

    1. TRADING ACTION (BUY/HOLD/SELL) - this reuses
       src/decision/engine.py's existing ensemble/voting decision
       UNCHANGED. The RL agents operate on a discretized market STATE
       (Trend/RSI/Volatility/Position), not on an arbitrary continuous
       price, so a user-entered price does not and cannot change which
       action the models recommend - it only changes how that
       recommendation is framed relative to what the user personally
       plans to pay.

    2. ESTIMATED PRICE RANGE - this project has no trained, validated
       price-prediction model (Sprints 1-5 only ever produce a
       discrete action, never a price forecast). Rather than inventing
       one, `estimate_price_range()` uses a single well-established,
       simple statistical technique: a +/-1 standard deviation band
       around the latest available Close price, using the SAME
       `Volatility` feature already computed in Sprint 2
       (src/features/indicators.py - a rolling standard deviation of
       daily returns). This is explicitly labeled everywhere as an
       ESTIMATE/RANGE, never a predicted price, and if that Volatility
       figure isn't available for a company (e.g. too little history),
       `estimate_price_range` returns None and the caller must not
       display a range at all - never a fabricated one.

RANGE VS. ENTRY PRICE (why the example numbers look "asymmetric")
--------------------------------------------------------------------
The estimated range is centered on the LATEST AVAILABLE market price
(an objective figure), not on whatever price the user happens to type
in. "Potential Upside/Downside %" are then computed relative to the
USER's own entry price - so if their entry price differs even slightly
from the latest available price, the upside and downside percentages
will differ in magnitude. This is intentional and correctly reflects
what the range means FOR THAT USER's specific entry price.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.decision import engine

RANGE_STD_MULTIPLIER = 1.0  # +/- 1 standard deviation (~68% coverage under a rough normal assumption)


def validate_price_input(raw_value) -> tuple:
    """
    Returns (price: float|None, error_message: str|None). Exactly one
    of the two is not-None. Never raises - handles None, empty string,
    non-numeric text, zero, and negative values, all as one clear,
    user-facing error (Section "INPUT VALIDATION").
    """
    error = "Please enter a valid stock price greater than ₹0."
    if raw_value is None:
        return None, error
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if raw_value == "":
            return None, error
    try:
        price = float(raw_value)
    except (TypeError, ValueError):
        return None, error
    if not (price > 0) or pd.isna(price):  # covers negative, zero, and NaN
        return None, error
    return price, None


def estimate_price_range(symbol: str, std_multiplier: float = RANGE_STD_MULTIPLIER) -> Optional[dict]:
    """
    +/-`std_multiplier` standard deviations around the latest available
    Close price, using Sprint 2's `Volatility` feature (rolling std of
    daily returns) for that same latest row. Returns None - never a
    fabricated range - if the latest row's Volatility is unavailable
    (NaN, e.g. not enough trading history yet for this company).
    """
    latest_row = engine.get_latest_feature_row(symbol)
    volatility = latest_row.get("Volatility")
    if volatility is None or pd.isna(volatility):
        return None

    basis_price = float(latest_row["Close"])
    offset = basis_price * float(volatility) * std_multiplier
    return {
        "basis_price": basis_price,
        "low": basis_price - offset,
        "high": basis_price + offset,
        "volatility_pct": float(volatility) * 100.0,
        "std_multiplier": std_multiplier,
    }


def _price_range_from_decision(decision: dict, std_multiplier: float) -> Optional[dict]:
    """
    Same +/-`std_multiplier`-std-dev calculation as `estimate_price_range`,
    but built from an already-computed `engine.get_decision()` result
    instead of re-reading feature data separately. This guarantees the
    range is centered on the SAME price/volatility (live or historical)
    the AI decision itself used - re-fetching separately here could
    otherwise show a live decision next to a stale historical range, or
    trigger a second live Angel One fetch that inconsistently succeeds/
    fails relative to the first.
    """
    volatility = decision.get("volatility_value")
    if volatility is None:
        return None
    basis_price = float(decision["latest_price"])
    offset = basis_price * float(volatility) * std_multiplier
    return {
        "basis_price": basis_price,
        "low": basis_price - offset,
        "high": basis_price + offset,
        "volatility_pct": float(volatility) * 100.0,
        "std_multiplier": std_multiplier,
    }


def next_day_strategy(symbol: str, entry_price: float) -> dict:
    """
    Full "Next Day Strategy" result for `symbol` given the user's
    `entry_price`. Raises engine.DecisionError if no trained model is
    available for this company (same as the Stock Decision page).

    Returns a dict with the AI decision fields from `engine.get_decision`
    (action, confidence_label, votes, weak_signal, data_source, ...) PLUS:
        entry_price, price_range (dict or None), upside_pct (float or
        None), downside_pct (float or None).
    """
    decision = engine.get_decision(symbol)
    price_range = _price_range_from_decision(decision, RANGE_STD_MULTIPLIER)

    upside_pct = downside_pct = None
    if price_range is not None and entry_price > 0:
        upside_pct = (price_range["high"] - entry_price) / entry_price * 100.0
        downside_pct = (price_range["low"] - entry_price) / entry_price * 100.0

    decision.update({
        "entry_price": entry_price,
        "price_range": price_range,
        "upside_pct": upside_pct,
        "downside_pct": downside_pct,
    })
    return decision