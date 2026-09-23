"""
src/risk/risk_engine.py

Top-level entry point for the News & Risk Layer (project spec #29
architecture). Ties together:

    engine.get_decision()       -> RL signal (UNCHANGED, separate system)
    market_shock.assess_market_shock()   -> market risk (reuses the SAME live fetch)
    news_analyzer.analyze_company_tiered() -> real COMPANY + SECTOR + MACRO
                                               news, sentiment, events (3-tier addition)
    news_risk.assess_news_risk() x3 + combine_tiered_news_risk() -> one overall news risk score
    decision_guard.apply_risk_guard()    -> FINAL decision (rule hierarchy, not averaging)

`evaluate()` is the one function the dashboard (app.py) should call -
it never trains or touches the RL models, only reads their already-
computed decision. Every sub-assessment inside the result is honestly
labeled "unavailable" (never fabricated) when its own data source
fails - see each module's own docstring for its specific fallback rule.

3-TIER ADDITION: `evaluate()` now uses `analyze_company_tiered(symbol)`
(src/news/news_analyzer.py) instead of the old company-only
`analyze_company_news()`, and combines company/sector/macro news risk
via `news_risk.combine_tiered_news_risk()` before handing ONE
overall_news_risk_score to `decision_guard.apply_risk_guard()` -
decision_guard.py's own rule hierarchy is otherwise UNCHANGED, it just
receives a risk-aware overall score instead of a company-only one.
`RiskAssessment.news` is kept as the COMPANY tier (so existing "Latest
News" style display code keeps working) with `.tiered_news` added
alongside it for the full 3-section breakdown.
"""

from dataclasses import dataclass
from typing import Optional

from src.decision import engine
from src.news.news_analyzer import NewsAnalysis, TieredNewsAnalysis, analyze_company_tiered
from src.risk import decision_guard, market_shock, news_risk

ACTION_NAME_BY_ID = {0: "HOLD", 1: "BUY", 2: "SELL"}

# Combined overall risk level (project spec #11) - a 4-point scale that
# folds market_risk_score (0-3) and news_risk_score (0-4) into one
# label by taking whichever is proportionally worse on its own scale.
OVERALL_LOW, OVERALL_MODERATE, OVERALL_HIGH, OVERALL_EXTREME = "LOW", "MODERATE", "HIGH", "EXTREME"
_LEVEL_ORDER = [OVERALL_LOW, OVERALL_MODERATE, OVERALL_HIGH, OVERALL_EXTREME]


def _market_level_to_overall(market_risk_score: int) -> str:
    return {market_shock.RISK_LOW: OVERALL_LOW, market_shock.RISK_MODERATE: OVERALL_MODERATE,
            market_shock.RISK_HIGH: OVERALL_HIGH, market_shock.RISK_EXTREME: OVERALL_EXTREME}[market_risk_score]


def _news_level_to_overall(news_risk_score: int) -> str:
    return {news_risk.RISK_NONE: OVERALL_LOW, news_risk.RISK_LOW: OVERALL_LOW,
            news_risk.RISK_MODERATE: OVERALL_MODERATE, news_risk.RISK_HIGH: OVERALL_HIGH,
            news_risk.RISK_CRITICAL: OVERALL_EXTREME}[news_risk_score]


@dataclass
class RiskAssessment:
    symbol: str
    company_display_name: str
    rl_decision: dict                                  # the full, unmodified engine.get_decision() result
    market: market_shock.MarketShockAssessment
    news: NewsAnalysis                                    # COMPANY tier only (back-compat with older display code)
    tiered_news: TieredNewsAnalysis                       # full COMPANY/SECTOR/MACRO breakdown
    news_risk: news_risk.NewsRiskAssessment               # COMPANY tier only (back-compat)
    tiered_news_risk: news_risk.TieredNewsRiskAssessment  # full 3-tier risk breakdown
    guard: decision_guard.RiskGuardResult
    overall_risk_level: str                              # LOW | MODERATE | HIGH | EXTREME
    final_action: str                                    # BUY | HOLD | SELL | AVOID
    reasons: list                                          # short, dashboard-friendly bullet list


def evaluate(symbol: str, company_display_name: str, prefer_live: bool = True,
             rl_decision: Optional[dict] = None) -> RiskAssessment:
    """
    Full News & Risk Layer evaluation for one company. `symbol` is the
    internal code (e.g. "APLH") used to look up sector/ticker/alias
    metadata (src/news/company_sector_map.py) for the 3-tier news
    search. `company_display_name` is kept for the COMPANY-tier back-
    compat NewsAnalysis's display label.

    `rl_decision`: pass an already-computed engine.get_decision() result
    (e.g. from the dashboard's own st.cache_data-wrapped call) to avoid
    a second, redundant live-market fetch. If omitted, this calls
    engine.get_decision() itself.
    """
    if rl_decision is None:
        rl_decision = engine.get_decision(symbol, prefer_live=prefer_live)

    market = market_shock.assess_market_shock(rl_decision)
    tiered_news = analyze_company_tiered(symbol)
    holding = engine.is_holding_position(symbol)

    company_risk = news_risk.assess_news_risk(tiered_news.company)
    sector_risk = news_risk.assess_news_risk(tiered_news.sector_news)
    macro_risk = news_risk.assess_news_risk(tiered_news.macro)
    tiered_risk = news_risk.combine_tiered_news_risk(
        company_risk, sector_risk, macro_risk, tiered_news.weights_used,
        sector_top_event_severity=tiered_news.sector_news.top_event_severity if tiered_news.sector_news.available else 0,
        macro_top_event_severity=tiered_news.macro.top_event_severity if tiered_news.macro.available else 0,
    )

    rl_action_name = ACTION_NAME_BY_ID[rl_decision["action"]]
    guard = decision_guard.apply_risk_guard(
        rl_action=rl_action_name,
        rl_confidence_pct=rl_decision["confidence_pct"],
        news_risk_score=tiered_risk.overall_news_risk_score,
        market_risk_score=market.market_risk_score if market.available else market_shock.RISK_LOW,
        top_event_severity=max(
            tiered_news.company.top_event_severity if tiered_news.company.available else 0,
            tiered_news.sector_news.top_event_severity if tiered_news.sector_news.available else 0,
            tiered_news.macro.top_event_severity if tiered_news.macro.available else 0,
        ),
        holding_position=holding,
        aggregated_sentiment_score=tiered_news.company.aggregated_sentiment_score,
    )

    market_level = _market_level_to_overall(market.market_risk_score) if market.available else OVERALL_LOW
    news_level = _news_level_to_overall(tiered_risk.overall_news_risk_score)
    overall_level = max(market_level, news_level, key=_LEVEL_ORDER.index)

    reasons = []
    reasons.append(f"RL models currently favor {rl_action_name}.")
    if tiered_news.company.available and tiered_news.company.articles:
        reasons.append(f"Company news sentiment: {tiered_news.company.aggregated_sentiment_label or 'NEUTRAL'}.")
    elif tiered_news.sector_news.available and tiered_news.sector_news.articles:
        reasons.append(f"No company-specific news, but {tiered_news.sector} sector sentiment: "
                        f"{tiered_news.sector_news.aggregated_sentiment_label or 'NEUTRAL'}.")
    elif tiered_news.company.available:
        reasons.append("No recent company or sector news was found.")
    else:
        reasons.append("News data unavailable.")
    if market.available and market.reasons:
        reasons.append(market.reasons[0])
    elif not market.available:
        reasons.append("Live market data unavailable for risk checks.")
    if guard.overridden:
        reasons.append(guard.reason)

    return RiskAssessment(
        symbol=symbol,
        company_display_name=company_display_name,
        rl_decision=rl_decision,
        market=market,
        news=tiered_news.company,
        tiered_news=tiered_news,
        news_risk=company_risk,
        tiered_news_risk=tiered_risk,
        guard=guard,
        overall_risk_level=overall_level,
        final_action=guard.final_action,
        reasons=reasons[:3],  # dashboard shows exactly 3 simple reasons (project spec #16)
    )
