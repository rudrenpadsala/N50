"""
src/risk/market_shock.py

Detects abnormal market movements for one company USING DATA ALREADY
FETCHED by the existing Angel One integration (src/decision/engine.py's
get_decision() result) - no separate market API, no extra network call.
Per project spec #8: intraday/daily return, volume spike, volatility,
drawdown.

MARKET RISK SCORE (0-3): Low / Moderate / High / Extreme - see
src/risk/risk_engine.py for how this and news_risk combine into a
final decision. This module only looks at MARKET data - it knows
nothing about news.

VOLUME SPIKE CAVEAT (documented per spec #8's "don't hard-code
assumptions without documenting them"): a true "spike vs recent
average" would need a rolling N-day average volume. This project
currently only carries forward TODAY's volume and the PREVIOUS
trading day's volume (see live_features.py), not a full rolling
average - so "volume spike" here is approximated as today's volume
vs. the single previous trading day, which is a noisier signal than a
20-day average would be. If a company has a naturally volatile daily
volume, this may over/under-trigger. Documented rather than hidden.
"""

from dataclasses import dataclass
from typing import Optional

from src.config import load_config

RISK_LOW = 0
RISK_MODERATE = 1
RISK_HIGH = 2
RISK_EXTREME = 3

RISK_LABELS = {RISK_LOW: "LOW", RISK_MODERATE: "MODERATE", RISK_HIGH: "HIGH", RISK_EXTREME: "EXTREME"}

HIGH_VOLATILITY = "HIGH_VOLATILITY"  # matches src/features/indicators.py's condition label


@dataclass
class MarketShockAssessment:
    available: bool
    daily_return_pct: Optional[float] = None
    price_shock: bool = False           # |return| >= price_shock_threshold
    extreme_price_shock: bool = False   # |return| >= extreme_price_shock_threshold
    volume_spike: bool = False
    volume_change_pct: Optional[float] = None
    high_volatility: bool = False
    drawdown_pct: Optional[float] = None
    significant_drawdown: bool = False
    market_risk_score: int = RISK_LOW
    market_risk_label: str = "LOW"
    reasons: list = None
    unavailable_reason: Optional[str] = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


def assess_market_shock(decision: dict) -> MarketShockAssessment:
    """
    `decision` is the dict returned by src/decision/engine.get_decision()
    - reused directly so this never issues its own market-data request.
    Works whether decision["data_source"] is "live" or "historical";
    the resulting risk score is just less current in the historical case
    (same honesty rule as everywhere else in this project - no live
    market data is not invented).
    """
    cfg = load_config()["news_risk"]
    price_shock_threshold = float(cfg["price_shock_threshold"])
    extreme_threshold = float(cfg["extreme_price_shock_threshold"])
    volume_spike_multiple = float(cfg["volume_spike_multiple"])
    drawdown_threshold = float(cfg["drawdown_threshold"])

    latest_price = decision.get("latest_price")
    previous_close = decision.get("previous_close")
    if latest_price is None or previous_close is None or previous_close in (0, None):
        return MarketShockAssessment(available=False, unavailable_reason="No previous close available to compute a daily return.")

    daily_return = (latest_price - previous_close) / previous_close
    daily_return_pct = daily_return * 100.0
    price_shock = abs(daily_return) >= price_shock_threshold
    extreme_price_shock = abs(daily_return) >= extreme_threshold

    volume_change_pct = decision.get("volume_change_pct")
    volume_spike = (
        volume_change_pct is not None
        and volume_change_pct >= (volume_spike_multiple - 1.0) * 100.0
    )

    high_volatility = decision.get("volatility_condition") == HIGH_VOLATILITY

    drawdown_pct = decision.get("drawdown_pct")
    significant_drawdown = drawdown_pct is not None and (drawdown_pct / 100.0) >= drawdown_threshold

    score = RISK_LOW
    reasons = []
    if extreme_price_shock:
        score = RISK_EXTREME
        reasons.append(f"Extreme price movement today ({daily_return_pct:+.1f}%).")
    elif price_shock:
        score = max(score, RISK_HIGH)
        reasons.append(f"Large price movement today ({daily_return_pct:+.1f}%).")

    if volume_spike:
        score = max(score, RISK_MODERATE if score < RISK_HIGH else score)
        reasons.append(f"Trading volume spiked ({volume_change_pct:+.1f}% vs. the previous session).")

    if high_volatility:
        score = max(score, RISK_MODERATE)
        reasons.append("20-day volatility is elevated for this stock.")

    if significant_drawdown:
        score = max(score, RISK_MODERATE)
        reasons.append(f"Price is {drawdown_pct:.1f}% below its recent high.")

    # Two or more moderate-or-worse signals together push risk up a notch,
    # even if no single signal alone was extreme.
    moderate_or_worse_signals = sum([price_shock, volume_spike, high_volatility, significant_drawdown])
    if moderate_or_worse_signals >= 3 and score < RISK_EXTREME:
        score = RISK_EXTREME
        reasons.append("Multiple risk signals are elevated at the same time.")
    elif moderate_or_worse_signals >= 2 and score < RISK_HIGH:
        score = RISK_HIGH

    return MarketShockAssessment(
        available=True,
        daily_return_pct=daily_return_pct,
        price_shock=price_shock,
        extreme_price_shock=extreme_price_shock,
        volume_spike=volume_spike,
        volume_change_pct=volume_change_pct,
        high_volatility=high_volatility,
        drawdown_pct=drawdown_pct,
        significant_drawdown=significant_drawdown,
        market_risk_score=score,
        market_risk_label=RISK_LABELS[score],
        reasons=reasons,
    )
