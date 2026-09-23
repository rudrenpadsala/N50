"""
src/news/news_analyzer.py

Orchestrates the News & Risk Layer's "news" side. Two entry points:

  analyze_company_news(company_name)  - ORIGINAL single-tier pipeline
      (company news only), kept unchanged for backward compatibility.

  analyze_company_tiered(symbol)      - NEW 3-tier pipeline: fetches
      COMPANY + SECTOR + MACRO news (src/news/company_sector_map.py
      supplies each company's aliases/sector_terms/macro_terms,
      src/news/news_api.py fetches all three), classifies EVERY
      resulting article with src/news/relevance.py (independent of
      which query found it - see that module's docstring for why),
      buckets by the classifier's own verdict, and returns a
      TieredNewsAnalysis with one NewsAnalysis-shaped result per tier
      plus company_news_impact/sector_news_impact/macro_news_impact/
      overall_news_impact (configurable weights - see config.yaml's
      news_risk.company_weight/sector_weight/macro_weight).

Both entry points reuse the SAME per-article sentiment
(src/news/sentiment.py) and event detection (src/news/event_detector.py)
- nothing about how a single article is scored changes; only how
articles are gathered, classified, and bucketed is new. The
news_risk_score itself (which also folds in market context) is
computed separately by src/risk/news_risk.py - this module's job stops
at "what do the actual articles say, and which tier do they belong to".

NEVER FABRICATES: if the API fails or returns zero articles for a
tier, that tier's own `available`/`reason` says so rather than
inventing sentiment - see `available` below on each tier's NewsAnalysis.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from src.config import load_config
from src.news import event_detector, news_api, relevance, sentiment
from src.news.company_sector_map import get_metadata


@dataclass
class AnalyzedArticle:
    headline: str
    description: Optional[str]
    source: str
    published_at: datetime
    url: str
    sentiment_label: str
    sentiment_score: float
    sentiment_model: str
    events: list
    event_severity: int  # top severity across this article's own matched events
    hours_ago: float
    relevance_type: str = relevance.COMPANY   # COMPANY | SECTOR | MACRO - which tier this article belongs to
    relevance_score: int = 0                  # 0-100, see src/news/relevance.py
    matched_terms: list = field(default_factory=list)


@dataclass
class NewsAnalysis:
    available: bool
    company_query: str
    lookback_hours: int
    tier: str = relevance.COMPANY             # COMPANY | SECTOR | MACRO
    articles: List[AnalyzedArticle] = field(default_factory=list)
    aggregated_sentiment_score: Optional[float] = None   # -1..+1, time-weighted average
    aggregated_sentiment_label: Optional[str] = None
    top_event_severity: int = 0
    negative_high_impact_count: int = 0   # articles with a NEGATIVE event of severity >= HIGH
    reason: Optional[str] = None          # populated when available=False


@dataclass
class TieredNewsAnalysis:
    """The full 3-level result for one company (project spec's
    COMPANY / SECTOR / MACRO hierarchy)."""
    symbol: str
    company_name: str
    ticker: str
    sector: str
    industry: str
    company: NewsAnalysis
    sector_news: NewsAnalysis
    macro: NewsAnalysis
    company_news_impact: Optional[float]   # -1..+1, or None if that tier is unavailable
    sector_news_impact: Optional[float]
    macro_news_impact: Optional[float]
    overall_news_impact: Optional[float]   # weighted combination (see combine weights below)
    weights_used: dict                       # {"company": .., "sector": .., "macro": ..} - always shown, never hidden


def _dedupe(articles: list) -> list:
    """Drop articles with the same normalized headline, keeping the first
    (most recent, since news_api returns newest-first)."""
    seen = set()
    unique = []
    for article in articles:
        key = " ".join(article.headline.lower().split())
        if key in seen:
            continue
        seen.add(key)
        unique.append(article)
    return unique


def _recency_weight(hours_ago: float, lookback_hours: int) -> float:
    """
    Newer articles count more. Halves roughly every 12 hours - an
    article from right now gets weight 1.0, one from 12h ago gets ~0.5,
    one from 24h ago ~0.33, etc. Never reaches exactly zero, so even the
    oldest article inside the lookback window still contributes a
    little to the aggregate (per spec #5: "prefer newer articles to
    have higher weight", not "ignore older ones").
    """
    return 1.0 / (1.0 + max(0.0, hours_ago) / 12.0)


def _analyze_one(article, metadata: dict, now: datetime) -> AnalyzedArticle:
    """Sentiment + events + relevance classification for one raw article -
    the single per-article pipeline shared by every tier and every
    call site, so a headline is scored identically no matter which
    query happened to surface it."""
    text = article.headline + (f". {article.description}" if article.description else "")
    sentiment_result = sentiment.analyze_sentiment(text)
    events = event_detector.detect_events(text, sentiment_result.score)
    hours_ago = max(0.0, (now - article.published_at).total_seconds() / 3600.0)
    rel = relevance.classify_article(text, metadata)
    return AnalyzedArticle(
        headline=article.headline, description=article.description, source=article.source,
        published_at=article.published_at, url=article.url,
        sentiment_label=sentiment_result.label, sentiment_score=sentiment_result.score,
        sentiment_model=sentiment_result.model_name,
        events=events, event_severity=event_detector.top_severity(events),
        hours_ago=hours_ago,
        relevance_type=rel.relevance_type, relevance_score=rel.relevance_score,
        matched_terms=rel.matched_terms,
    )


def _aggregate(tier: str, query_label: str, lookback_hours: int, analyzed: List[AnalyzedArticle],
                max_articles: int) -> NewsAnalysis:
    """Builds the tier's NewsAnalysis from an already-filtered,
    already-sorted (most-recent-first) list of AnalyzedArticle."""
    if not analyzed:
        return NewsAnalysis(available=True, company_query=query_label, lookback_hours=lookback_hours,
                             tier=tier, articles=[], aggregated_sentiment_score=None,
                             aggregated_sentiment_label=None, top_event_severity=0,
                             negative_high_impact_count=0)

    weights = [_recency_weight(a.hours_ago, lookback_hours) for a in analyzed]
    weighted_sum = sum(w * a.sentiment_score for w, a in zip(weights, analyzed))
    total_weight = sum(weights)
    aggregated_score = weighted_sum / total_weight if total_weight else 0.0

    top_severity = max((a.event_severity for a in analyzed), default=0)
    negative_high_impact = sum(
        1 for a in analyzed
        if a.event_severity >= event_detector.SEVERITY_HIGH
        and any(e.direction == event_detector.NEGATIVE for e in a.events)
    )

    return NewsAnalysis(
        available=True,
        company_query=query_label,
        lookback_hours=lookback_hours,
        tier=tier,
        articles=analyzed[:max_articles],
        aggregated_sentiment_score=aggregated_score,
        aggregated_sentiment_label=sentiment.label_for_score(aggregated_score),
        top_event_severity=top_severity,
        negative_high_impact_count=negative_high_impact,
    )


def analyze_company_news(company_name: str, lookback_hours: Optional[int] = None,
                          use_cache: bool = True) -> NewsAnalysis:
    """
    ORIGINAL single-tier (company-only) pipeline - unchanged behavior,
    kept for any existing call site. `company_name` should be a plain
    search phrase (e.g. "Reliance Industries"), not an internal code.
    Prefer analyze_company_tiered() (below) for the full COMPANY/
    SECTOR/MACRO breakdown.
    """
    cfg = load_config()["news_risk"]
    if lookback_hours is None:
        lookback_hours = int(cfg["lookback_hours"])
    max_articles = int(cfg["max_articles"])

    raw_articles, reason = news_api.fetch_company_news(
        company_name, lookback_hours, max_articles=max_articles * 4, use_cache=use_cache,
    )
    if raw_articles is None:
        return NewsAnalysis(available=False, company_query=company_name,
                             lookback_hours=lookback_hours, reason=reason)

    unique_articles = _dedupe(raw_articles)
    now = datetime.now(timezone.utc)
    # No company_sector_map metadata for a bare display-name string, so a
    # minimal ad-hoc metadata dict is used - only `aliases` matters for
    # relevance_type here, and the rest of this pipeline (sentiment,
    # events, aggregation) doesn't depend on relevance_type at all.
    minimal_metadata = dict(ticker=company_name, sector="", industry="", aliases=[company_name],
                             sector_terms=[], macro_terms=[])
    analyzed = [_analyze_one(a, minimal_metadata, now) for a in unique_articles[: max_articles * 3]]
    analyzed.sort(key=lambda a: a.published_at, reverse=True)

    return _aggregate(relevance.COMPANY, company_name, lookback_hours, analyzed, max_articles)


def _weighted_overall(company_impact, sector_impact, macro_impact, weights: dict) -> Optional[float]:
    """
    overall_news_impact (project spec: "using configurable weights...
    do NOT hide these weights in the code"). Missing (None) tiers are
    excluded and the remaining weights renormalized, rather than being
    silently treated as 0 impact - a tier with no data should not drag
    the overall score toward neutral by fiat.
    """
    pairs = [(company_impact, weights["company"]), (sector_impact, weights["sector"]),
             (macro_impact, weights["macro"])]
    available = [(v, w) for v, w in pairs if v is not None]
    if not available:
        return None
    total_weight = sum(w for _, w in available)
    if total_weight <= 0:
        return None
    return sum(v * w for v, w in available) / total_weight


def analyze_company_tiered(symbol: str, lookback_hours: Optional[int] = None,
                            use_cache: bool = True) -> TieredNewsAnalysis:
    """
    Full 3-tier (COMPANY/SECTOR/MACRO) news analysis for `symbol` (an
    internal code like "APLH" - looked up via
    src.news.company_sector_map.get_metadata, NOT a display name).
    """
    cfg = load_config()["news_risk"]
    if lookback_hours is None:
        lookback_hours = int(cfg["lookback_hours"])
    max_articles_per_tier = int(cfg.get("max_articles_per_tier", cfg["max_articles"]))
    relevance_threshold = int(cfg.get("relevance_threshold", 30))
    weights = {
        "company": float(cfg.get("company_weight", 0.6)),
        "sector": float(cfg.get("sector_weight", 0.3)),
        "macro": float(cfg.get("macro_weight", 0.1)),
    }

    metadata = get_metadata(symbol)
    fetch_budget = max_articles_per_tier * 4

    company_raw, company_reason = news_api.fetch_company_tier_news(
        metadata["aliases"], lookback_hours, max_articles=fetch_budget, use_cache=use_cache)
    sector_raw, sector_reason = news_api.fetch_sector_tier_news(
        metadata["sector_terms"], lookback_hours, max_articles=fetch_budget, use_cache=use_cache)
    macro_raw, macro_reason = news_api.fetch_macro_tier_news(
        metadata["macro_terms"], lookback_hours, max_articles=fetch_budget, use_cache=use_cache)

    fetch_ok = {"COMPANY": company_raw is not None, "SECTOR": sector_raw is not None, "MACRO": macro_raw is not None}
    fetch_reason = {"COMPANY": company_reason, "SECTOR": sector_reason, "MACRO": macro_reason}

    # Pool every successfully-fetched article across all three queries,
    # then classify EACH one independently (see module docstring) -
    # the tier it lands in is decided by relevance.classify_article,
    # not by which query happened to surface it.
    pooled = _dedupe([a for raw in (company_raw, sector_raw, macro_raw) if raw for a in raw])
    now = datetime.now(timezone.utc)
    analyzed_all = [_analyze_one(a, metadata, now) for a in pooled]
    analyzed_all.sort(key=lambda a: a.published_at, reverse=True)

    buckets = {"COMPANY": [], "SECTOR": [], "MACRO": []}
    for a in analyzed_all:
        if a.relevance_type == relevance.IRRELEVANT or a.relevance_score < relevance_threshold:
            continue  # per spec: "Reject articles below the configured threshold"
        buckets[a.relevance_type].append(a)

    def _tier_result(tier: str, raw_available: bool, reason: Optional[str], query_label: str) -> NewsAnalysis:
        if not raw_available and not buckets[tier]:
            # This tier's own dedicated fetch failed AND nothing from the
            # other two queries happened to classify into it either -
            # honestly report unavailable rather than showing a fake "0 articles".
            return NewsAnalysis(available=False, company_query=query_label, lookback_hours=lookback_hours,
                                 tier=tier, reason=reason)
        return _aggregate(tier, query_label, lookback_hours, buckets[tier], max_articles_per_tier)

    company_result = _tier_result("COMPANY", fetch_ok["COMPANY"], fetch_reason["COMPANY"], metadata["aliases"][0])
    sector_result = _tier_result("SECTOR", fetch_ok["SECTOR"], fetch_reason["SECTOR"], metadata["sector"])
    macro_result = _tier_result("MACRO", fetch_ok["MACRO"], fetch_reason["MACRO"], "Macro/Market")

    company_impact = company_result.aggregated_sentiment_score if company_result.available else None
    sector_impact = sector_result.aggregated_sentiment_score if sector_result.available else None
    macro_impact = macro_result.aggregated_sentiment_score if macro_result.available else None
    overall_impact = _weighted_overall(company_impact, sector_impact, macro_impact, weights)

    return TieredNewsAnalysis(
        symbol=symbol,
        company_name=metadata["company_name"],
        ticker=metadata["ticker"],
        sector=metadata["sector"],
        industry=metadata["industry"],
        company=company_result,
        sector_news=sector_result,
        macro=macro_result,
        company_news_impact=company_impact,
        sector_news_impact=sector_impact,
        macro_news_impact=macro_impact,
        overall_news_impact=overall_impact,
        weights_used=weights,
    )
