"""
src/news/news_api.py

Thin client for NewsAPI.org's "everything" endpoint - the News & Risk
Layer's ONLY external data source besides the existing Angel One market
API (see project rule #29: exactly one market API, one news API).

Docs: https://newsapi.org/docs/endpoints/everything

WHY NEWSAPI.ORG: a single well-documented REST endpoint, a generous
free tier for personal/research projects, and a simple apiKey-based
auth model - no OAuth/session flow needed, unlike Angel One. If you
later want a different provider, this module is the only place that
needs to change; everything downstream (src/news/news_analyzer.py)
consumes the plain `NewsArticle` list this returns, not raw API JSON.

NEVER FABRICATES: on any failure (missing key, network error, rate
limit, malformed response) this returns (None, reason) - callers must
show "News data unavailable" and must NOT invent sentiment for a
company with no real articles. See src/news/news_analyzer.py.

CACHING: NewsAPI's free tier is rate-limited (100 requests/day at time
of writing) and this project's own spec asks for 10-30 minute caching
regardless of tier - so results are cached in-process per (query,
lookback_hours) key for NEWS_CACHE_MINUTES (config: news_risk.cache_minutes).

THREE-TIER NEWS & RISK LAYER: this module fetches for all three tiers
(company/sector/macro - see src/news/company_sector_map.py for what
each company's tier queries are, and src/news/news_analyzer.py for how
the three tiers get combined). `fetch_news_for_query` is the one
underlying HTTP call; fetch_company_tier_news/fetch_sector_tier_news/
fetch_macro_tier_news just build the right OR'd query string per tier.
`fetch_company_news` is kept as a plain-string wrapper for backward
compatibility with existing single-phrase call sites.
"""

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import requests

BASE_URL = "https://newsapi.org/v2/everything"
REQUEST_TIMEOUT_SECONDS = 10


@dataclass
class NewsArticle:
    headline: str
    description: Optional[str]
    source: str
    published_at: datetime  # timezone-aware, UTC
    url: str


def _load_api_key() -> Optional[str]:
    key = os.environ.get("NEWS_API_KEY", "").strip()
    return key or None


_cache: dict = {}  # (company, lookback_hours) -> (fetched_at: float, articles: list[NewsArticle])


def _cache_minutes() -> int:
    try:
        from src.config import load_config
        return int(load_config()["news_risk"]["cache_minutes"])
    except Exception:
        return 20  # safe default if config is missing/unreadable - never crash on a config read


def _parse_published_at(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    try:
        # NewsAPI uses ISO 8601 with a trailing "Z" (UTC).
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def fetch_news_for_query(
    query: str, lookback_hours: int, max_articles: int = 20, use_cache: bool = True,
) -> Tuple[Optional[list], Optional[str]]:
    """
    Fetch recent news matching `query` (any plain search phrase or
    NewsAPI boolean query string - e.g. one company alias, or several
    sector terms OR'd together). This is the ONE underlying fetch used
    by every tier (company/sector/macro) - see
    fetch_company_news/fetch_sector_news/fetch_macro_news below, which
    just build the right `query` string and cache key for their tier.
    Returns (list[NewsArticle], None) - possibly empty - or (None, reason)
    on failure. Never returns fabricated articles.
    """
    cache_key = (query, lookback_hours)
    if use_cache and cache_key in _cache:
        fetched_at, articles = _cache[cache_key]
        if (time.time() - fetched_at) < _cache_minutes() * 60:
            return articles, None

    api_key = _load_api_key()
    if api_key is None:
        return None, "NEWS_API_KEY is not configured."

    since = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    params = {
        "q": query,
        "from": since.strftime("%Y-%m-%dT%H:%M:%S"),
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": min(max_articles, 100),
        "apiKey": api_key,
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        body = response.json()
    except (requests.RequestException, ValueError):
        return None, "Could not reach the News API."

    if not isinstance(body, dict) or body.get("status") != "ok":
        message = body.get("message", "unknown error") if isinstance(body, dict) else "unexpected response"
        return None, f"News API error: {message}"

    articles = []
    for raw in body.get("articles", []):
        headline = (raw.get("title") or "").strip()
        published_at = _parse_published_at(raw.get("publishedAt"))
        url = raw.get("url") or ""
        if not headline or published_at is None or not url:
            continue  # skip malformed entries rather than showing a broken card
        source = ((raw.get("source") or {}).get("name") or "Unknown source").strip()
        articles.append(NewsArticle(
            headline=headline,
            description=(raw.get("description") or "").strip() or None,
            source=source,
            published_at=published_at,
            url=url,
        ))

    _cache[cache_key] = (time.time(), articles)
    return articles, None


def _build_or_query(phrases: list, max_terms: int = 6) -> str:
    """NewsAPI's `q` supports boolean OR with quoted phrases:
    '"a" OR "b" OR "c"'. Capped at `max_terms` because very long OR
    queries both exceed NewsAPI's query-length limit and dilute
    precision - the terms lists in company_sector_map.py are already
    curated short, so this cap is rarely the binding constraint."""
    terms = [p for p in phrases[:max_terms] if p]
    return " OR ".join(f'"{t}"' for t in terms)


def fetch_company_news(
    company_name: str, lookback_hours: int, max_articles: int = 20, use_cache: bool = True,
) -> Tuple[Optional[list], Optional[str]]:
    """
    Fetch recent news for `company_name` (a plain search phrase, e.g.
    "Reliance Industries" - NOT an internal code like "RELI"). Thin
    wrapper over fetch_news_for_query for backward compatibility with
    the single-company-string call sites; prefer fetch_company_tier_news
    below (which searches ALL of a company's known aliases, not just one
    phrase) for new code.
    """
    return fetch_news_for_query(company_name, lookback_hours, max_articles, use_cache)


def fetch_company_tier_news(aliases: list, lookback_hours: int, max_articles: int = 20,
                             use_cache: bool = True) -> Tuple[Optional[list], Optional[str]]:
    """COMPANY tier: OR's together every known alias/ticker for the company
    (see src/news/company_sector_map.py) so e.g. both "Reliance Industries"
    and "RELIANCE" surface results, not just whichever one phrase was guessed."""
    query = _build_or_query(aliases)
    return fetch_news_for_query(query, lookback_hours, max_articles, use_cache)


def fetch_sector_tier_news(sector_terms: list, lookback_hours: int, max_articles: int = 20,
                            use_cache: bool = True) -> Tuple[Optional[list], Optional[str]]:
    """SECTOR tier: OR's together the sector's curated search terms
    (e.g. "Indian healthcare sector" OR "India hospital industry" ...).
    Deliberately narrower than `category=business` (project spec:
    "this produces too much unrelated news")."""
    query = _build_or_query(sector_terms)
    return fetch_news_for_query(query, lookback_hours, max_articles, use_cache)


def fetch_macro_tier_news(macro_terms: list, lookback_hours: int, max_articles: int = 20,
                           use_cache: bool = True) -> Tuple[Optional[list], Optional[str]]:
    """MACRO tier: OR's together only the macro topics judged relevant
    to this company's sector (plus the common India/global list) - NOT
    every RBI or Nifty headline (project spec: "Do not treat every RBI
    or Nifty article as relevant" - that filtering happens again,
    per-article, in src/news/relevance.py; this is just the search)."""
    query = _build_or_query(macro_terms)
    return fetch_news_for_query(query, lookback_hours, max_articles, use_cache)


def clear_cache() -> None:
    """Force the next fetch to hit the API again. Used by the dashboard's
    'Refresh News' button and by tests."""
    _cache.clear()
