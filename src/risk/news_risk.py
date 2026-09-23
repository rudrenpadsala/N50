"""
src/risk/news_risk.py

Turns a src/news/news_analyzer.NewsAnalysis into a single news_risk_score
(0-4: None/Low/Moderate/High/Critical - see project spec #10), from:
  - aggregated (time-weighted) sentiment across recent articles
  - the single worst event severity detected in any article
  - how many articles carried a NEGATIVE high-impact event
  - recency - a recent (<=6h old) high-severity negative event is
    treated as at least as urgent as an older one of the same severity

This is a transparent RULE TABLE (spec #12: "do not simply average
everything") - each threshold is named and documented below, not a
black-box combination.

3-TIER ADDITION: `assess_news_risk()` itself is UNCHANGED and still
scores one NewsAnalysis (one tier) at a time - it has no idea whether
it's being called on company, sector, or macro news. What's new is
`combine_tiered_news_risk()` at the bottom of this file, which calls
it three times (once per tier - see src/news/news_analyzer.py's
TieredNewsAnalysis) and combines the three results per project spec
#12's explicit priority rules, not a plain average.
"""

from dataclasses import dataclass
from typing import Optional

from src.news import event_detector
from src.news.news_analyzer import NewsAnalysis

RISK_NONE = 0
RISK_LOW = 1
RISK_MODERATE = 2
RISK_HIGH = 3
RISK_CRITICAL = 4

RISK_LABELS = {
    RISK_NONE: "NONE", RISK_LOW: "LOW", RISK_MODERATE: "MODERATE",
    RISK_HIGH: "HIGH", RISK_CRITICAL: "CRITICAL",
}

# Thresholds on the -1..+1 aggregated sentiment score.
_VERY_NEGATIVE = -0.7
_NEGATIVE = -0.5
_MILD_NEGATIVE = -0.25
_SLIGHT_NEGATIVE = -0.1

_RECENT_HOURS = 6.0  # an event this fresh is treated as at least as urgent as its severity implies


@dataclass
class NewsRiskAssessment:
    available: bool
    news_risk_score: int = RISK_NONE
    news_risk_label: str = "NONE"
    reasons: list = None
    unavailable_reason: Optional[str] = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


@dataclass
class TieredNewsRiskAssessment:
    """
    Company/Sector/Macro risk combined per project spec #12 ("Do NOT
    simply average everything... Company news has highest importance.
    Sector news has second-level importance. Macro news has lower
    importance unless event severity is high/critical.") - see
    combine_tiered_news_risk() below for the actual rule.
    """
    company: NewsRiskAssessment
    sector: NewsRiskAssessment
    macro: NewsRiskAssessment
    overall_news_risk_score: int
    overall_news_risk_label: str
    reasons: list
    weights_used: dict


def assess_news_risk(analysis: NewsAnalysis) -> NewsRiskAssessment:
    if not analysis.available:
        return NewsRiskAssessment(available=False, unavailable_reason=analysis.reason)

    if not analysis.articles:
        return NewsRiskAssessment(available=True, news_risk_score=RISK_NONE, news_risk_label="NONE",
                                   reasons=["No recent company news found - nothing to flag."])

    sentiment_score = analysis.aggregated_sentiment_score or 0.0
    top_severity = analysis.top_event_severity
    negative_high_impact = analysis.negative_high_impact_count

    # Recency check: any NEGATIVE event of High+ severity within the last _RECENT_HOURS?
    recent_high_severity_negative = any(
        a.event_severity >= event_detector.SEVERITY_HIGH and a.hours_ago <= _RECENT_HOURS
        and any(e.direction == event_detector.NEGATIVE for e in a.events)
        for a in analysis.articles
    )

    reasons = []
    score = RISK_NONE

    if top_severity == event_detector.SEVERITY_CRITICAL:
        score = RISK_CRITICAL
        reasons.append("A critical-severity negative event was detected in recent news.")
    elif sentiment_score <= _VERY_NEGATIVE and negative_high_impact >= 1:
        score = RISK_CRITICAL
        reasons.append("News sentiment is very negative alongside a high-impact negative event.")
    elif top_severity == event_detector.SEVERITY_HIGH:
        score = RISK_HIGH
        reasons.append("A high-severity negative event was detected in recent news.")
    elif sentiment_score <= _NEGATIVE:
        score = RISK_HIGH
        reasons.append("Aggregated recent news sentiment is strongly negative.")
    elif negative_high_impact >= 2:
        score = RISK_HIGH
        reasons.append("Multiple negative high-impact articles were found.")
    elif top_severity == event_detector.SEVERITY_MEDIUM:
        score = RISK_MODERATE
        reasons.append("A medium-severity event was detected in recent news.")
    elif sentiment_score <= _MILD_NEGATIVE:
        score = RISK_MODERATE
        reasons.append("Aggregated recent news sentiment is negative.")
    elif negative_high_impact >= 1:
        score = RISK_MODERATE
        reasons.append("A negative high-impact article was found.")
    elif top_severity == event_detector.SEVERITY_LOW or sentiment_score <= _SLIGHT_NEGATIVE:
        score = RISK_LOW
        reasons.append("Recent news sentiment is mildly negative.")
    else:
        score = RISK_NONE
        reasons.append("No significant negative news signal found.")

    if recent_high_severity_negative and score < RISK_HIGH:
        score = RISK_HIGH
        reasons.append(f"A high-severity negative event appeared within the last {_RECENT_HOURS:.0f} hours.")

    return NewsRiskAssessment(available=True, news_risk_score=score, news_risk_label=RISK_LABELS[score], reasons=reasons)


# ---------------------------------------------------------------------------
# 3-tier combination (project spec's COMPANY / SECTOR / MACRO hierarchy)
# ---------------------------------------------------------------------------
# Event severity above which a tier's own severity can escalate the
# OVERALL score even if its weight is small (spec: "macro news has
# lower importance unless event severity is high/critical").
_ESCALATION_SEVERITY = 3  # matches event_detector.SEVERITY_HIGH


def combine_tiered_news_risk(company: NewsRiskAssessment, sector: NewsRiskAssessment,
                              macro: NewsRiskAssessment, weights: dict,
                              sector_top_event_severity: int = 0,
                              macro_top_event_severity: int = 0) -> TieredNewsRiskAssessment:
    """
    Combines three already-computed per-tier NewsRiskAssessments
    (each produced by assess_news_risk() unchanged - see module
    docstring) into ONE overall_news_risk_score, honoring project spec
    #12's explicit rule: "Do NOT simply average everything."

    RULE (priority order, first match wins - same style as
    src/risk/decision_guard.py's rule hierarchy):
      1. ANY tier at CRITICAL -> overall = CRITICAL immediately,
         regardless of weight (a critical company fraud story and a
         critical sector-wide regulatory crackdown are both
         disqualifying on their own).
      2. Sector or macro's OWN independent risk score is HIGH-or-worse,
         or carries a HIGH-or-worse severity EVENT -> overall is
         floored at HIGH (or that tier's own score if higher) even
         though its blended weight is small (spec: "unless event
         severity is high/critical", and the worked example where
         company=positive but sector=HIGH still needs to matter - a
         pure 30%-weighted average would understate a tier that's
         independently in bad shape).
      3. Otherwise -> a WEIGHTED score computed on the 0-4 scale
         (weights renormalized over available tiers), rounded to the
         nearest whole risk level - company dominates by weight alone
         in the normal case (spec example: RL positive, company
         positive, sector very negative, macro neutral -> should still
         be pulled toward caution, which this weighted formula does
         via the sector term even without a special-cased event).
    """
    tiers = {"company": company, "sector": sector, "macro": macro}
    available_scores = {name: t.news_risk_score for name, t in tiers.items() if t.available}
    reasons = []

    if not available_scores:
        return TieredNewsRiskAssessment(
            company=company, sector=sector, macro=macro,
            overall_news_risk_score=RISK_NONE, overall_news_risk_label=RISK_LABELS[RISK_NONE],
            reasons=["No news data was available for any tier (company, sector, or macro)."],
            weights_used=weights,
        )

    # --- Rule 1: any tier CRITICAL -> overall CRITICAL ---
    critical_tiers = [name for name, score in available_scores.items() if score == RISK_CRITICAL]
    if critical_tiers:
        overall = RISK_CRITICAL
        reasons.append(f"{'/'.join(critical_tiers).capitalize()}-level news risk is CRITICAL.")
        for name in critical_tiers:
            reasons.extend(tiers[name].reasons[:1])
        return TieredNewsRiskAssessment(
            company=company, sector=sector, macro=macro,
            overall_news_risk_score=overall, overall_news_risk_label=RISK_LABELS[overall],
            reasons=reasons[:3], weights_used=weights,
        )

    # --- Rule 2: sector/macro tier is independently HIGH+ risk, OR carries
    # a HIGH-severity event -> floors the overall score at that level, even
    # though its blended weight is small (spec: "sector conditions are
    # negative... therefore the final signal can be reduced/overridden" -
    # the worked example has company=positive/NONE but sector=HIGH on its
    # own, which a pure weighted average at 30% weight would understate).
    floor_score = RISK_NONE
    if sector.available and sector.news_risk_score >= RISK_HIGH:
        floor_score = max(floor_score, sector.news_risk_score)
    if macro.available and macro.news_risk_score >= RISK_HIGH:
        floor_score = max(floor_score, macro.news_risk_score)
    if sector.available and sector_top_event_severity >= _ESCALATION_SEVERITY:
        floor_score = max(floor_score, RISK_HIGH)
    if macro.available and macro_top_event_severity >= _ESCALATION_SEVERITY:
        floor_score = max(floor_score, RISK_HIGH)

    # --- Rule 3: weighted combination over available tiers ---
    total_weight = sum(weights[name] for name in available_scores)
    weighted = (sum(available_scores[name] * weights[name] for name in available_scores) / total_weight
                if total_weight > 0 else 0.0)
    overall = max(round(weighted), floor_score)
    overall = min(RISK_CRITICAL, max(RISK_NONE, overall))

    if floor_score > round(weighted):
        reasons.append("A high-severity sector or macro event kept overall risk elevated "
                        "even though its weight in the blended score is small.")
    reasons.append(f"Company={RISK_LABELS[company.news_risk_score] if company.available else 'N/A'}, "
                    f"Sector={RISK_LABELS[sector.news_risk_score] if sector.available else 'N/A'}, "
                    f"Macro={RISK_LABELS[macro.news_risk_score] if macro.available else 'N/A'} "
                    f"(weights {int(weights['company']*100)}/{int(weights['sector']*100)}/{int(weights['macro']*100)}%).")
    if company.available and company.reasons:
        reasons.append(company.reasons[0])

    return TieredNewsRiskAssessment(
        company=company, sector=sector, macro=macro,
        overall_news_risk_score=overall, overall_news_risk_label=RISK_LABELS[overall],
        reasons=reasons[:3], weights_used=weights,
    )
