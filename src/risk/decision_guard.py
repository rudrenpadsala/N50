"""
src/risk/decision_guard.py

The RULE HIERARCHY that turns (RL signal + news risk + market risk +
event severity + current position) into ONE final decision. This is
the piece the project spec is most emphatic about (#12): "Do NOT
simply average everything" - RL=BUY + News=SELL is never averaged into
HOLD. Instead, each rule below is evaluated in a fixed priority order
and the FIRST one that applies wins; every override carries an
explicit, readable reason string, so "why did it not follow the RL
signal" is always answerable from the output alone.

RULE ORDER (highest priority first) - mirrors project spec #13 exactly:
  1. Critical negative event detected -> maximum risk protection
     (SELL-if-holding / AVOID-if-not, regardless of what the RL signal said)
  2. High negative news risk + high-or-worse market risk -> a BUY signal
     is downgraded to HOLD (a SELL/HOLD signal is left alone - risk
     doesn't need to "soften" a signal that's already cautious)
  3. Moderate negative news risk -> a BUY signal is downgraded to HOLD
  4. Positive/neutral news + calm-or-moderate market -> RL signal kept as-is
  5. Otherwise -> RL signal kept as-is

POSITION-AWARE SELL/AVOID (project spec #13): after the rule above
decides an underlying BUY/HOLD/SELL action, a SELL is relabeled to
AVOID whenever there's no existing position to exit - "SELL" without
a position to sell doesn't mean anything actionable; the honest label
is "don't enter this one right now".

CONFIDENCE -> "MODEL AGREEMENT" (project spec #15): the RL ensemble's
confidence_pct (see engine.combine_votes) is literally "% of the
algorithms that voted for the winning action" - a genuine agreement
statistic, not a probability of anything. This module labels it
accordingly and applies a documented, deterministic penalty when risk
is elevated or when this guard overrides the RL signal (both indicate
LOWER overall agreement between "what the models say" and "what the
current situation supports") - see PENALTY_TABLE below. It is never
inflated, and it is never presented as a probability of profit.
"""

from dataclasses import dataclass
from typing import Optional

from src.risk import market_shock, news_risk

BUY, HOLD, SELL, AVOID = "BUY", "HOLD", "SELL", "AVOID"

# Documented confidence-penalty formula (project spec #15). Applied
# additively, then the result is floored at MIN_DISPLAYED_CONFIDENCE and
# capped at the original RL confidence (a penalty never INCREASES confidence).
NEWS_RISK_PENALTY = {
    news_risk.RISK_CRITICAL: 30, news_risk.RISK_HIGH: 20,
    news_risk.RISK_MODERATE: 10, news_risk.RISK_LOW: 5, news_risk.RISK_NONE: 0,
}
MARKET_RISK_PENALTY = {
    market_shock.RISK_EXTREME: 20, market_shock.RISK_HIGH: 15,
    market_shock.RISK_MODERATE: 5, market_shock.RISK_LOW: 0,
}
OVERRIDE_PENALTY = 10  # additional penalty when this guard changes the RL signal
MIN_DISPLAYED_CONFIDENCE = 10.0


@dataclass
class RiskGuardResult:
    rl_action: str
    final_action: str                # BUY | HOLD | SELL | AVOID
    overridden: bool
    reason: str
    model_agreement_pct: float        # risk-adjusted "% of models that agreed" - NOT a probability
    model_agreement_raw_pct: float    # the original, unadjusted RL confidence_pct
    holding_position: bool


def _combined_risk_penalty(news_risk_score: int, market_risk_score: int) -> int:
    return NEWS_RISK_PENALTY.get(news_risk_score, 0) + MARKET_RISK_PENALTY.get(market_risk_score, 0)


def apply_risk_guard(
    rl_action: str,
    rl_confidence_pct: float,
    news_risk_score: int,
    market_risk_score: int,
    top_event_severity: int,
    holding_position: bool,
    aggregated_sentiment_score: Optional[float] = None,
) -> RiskGuardResult:
    """
    `rl_action` must already be one of BUY/HOLD/SELL (the RL ensemble's
    own final vote - see engine.get_decision()["action"], mapped to its
    name). Returns the FINAL action (which may be AVOID, a label this
    guard introduces - the RL layer itself never produces it).
    """
    from src.news import event_detector as _ed  # local import to avoid a hard news->risk->news cycle at module load

    action = rl_action
    overridden = False
    reason = "No elevated risk detected - the model signal is used as-is."

    # --- Rule 1: critical negative event -> maximum risk protection ---
    if top_event_severity == _ed.SEVERITY_CRITICAL:
        action = SELL
        overridden = (action != rl_action)
        reason = ("A critical-severity negative event was detected in recent news "
                   "(e.g. fraud, bankruptcy, or a major accounting issue) - maximum risk "
                   "protection applied regardless of the model signal.")

    # --- Rule 2: high negative news + high-or-worse market risk -> downgrade BUY ---
    elif (news_risk_score >= news_risk.RISK_HIGH and market_risk_score >= market_shock.RISK_HIGH
          and rl_action == BUY):
        action = HOLD
        overridden = True
        reason = ("Negative high-impact news combined with elevated market risk - a BUY "
                   "signal was downgraded to HOLD rather than entering into current risk.")

    # --- Rule 3: moderate negative news -> downgrade BUY, prefer caution ---
    elif news_risk_score >= news_risk.RISK_MODERATE and rl_action == BUY:
        action = HOLD
        overridden = True
        reason = "Negative news risk detected - a BUY signal was downgraded to HOLD as a precaution."

    # --- Rule 4/5: positive/neutral news & calm-or-moderate market, or nothing elevated -> keep RL signal ---
    else:
        action = rl_action
        overridden = False
        if news_risk_score <= news_risk.RISK_LOW and market_risk_score <= market_shock.RISK_MODERATE:
            reason = "News and market conditions are normal - the model signal is used as-is."
        else:
            reason = "No rule required overriding the model signal in this situation."

    # --- Position-aware SELL/AVOID relabeling (applies no matter which rule fired) ---
    if action == SELL and not holding_position:
        action = AVOID
        if not overridden:
            reason = "The model signal was SELL, but with no existing position this means: do not enter right now."

    # --- Confidence -> risk-adjusted "Model Agreement" ---
    penalty = _combined_risk_penalty(news_risk_score, market_risk_score)
    if overridden:
        penalty += OVERRIDE_PENALTY
    adjusted = max(MIN_DISPLAYED_CONFIDENCE, min(rl_confidence_pct, rl_confidence_pct - penalty))

    return RiskGuardResult(
        rl_action=rl_action,
        final_action=action,
        overridden=overridden,
        reason=reason,
        model_agreement_pct=adjusted,
        model_agreement_raw_pct=rl_confidence_pct,
        holding_position=holding_position,
    )
