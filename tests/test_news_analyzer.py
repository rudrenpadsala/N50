"""
tests/test_news_analyzer.py

Tests for src/news/news_analyzer.py - dedup, aggregation, event
detection wiring, and honest "unavailable"/"no news" handling. All
news_api calls mocked (via monkeypatching news_api.fetch_company_news).
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.news import news_analyzer, news_api  # noqa: E402


def _article(headline, hours_ago=1.0, description=None, source="Example News"):
    return news_api.NewsArticle(
        headline=headline, description=description, source=source,
        published_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
        url=f"https://example.com/{abs(hash(headline))}",
    )


@pytest.fixture(autouse=True)
def _clear_cache():
    news_api.clear_cache()
    yield
    news_api.clear_cache()


def test_unavailable_when_api_fails(monkeypatch):
    monkeypatch.setattr(news_api, "fetch_company_news", lambda *a, **k: (None, "News API error: rate limited"))
    result = news_analyzer.analyze_company_news("Reliance Industries")
    assert result.available is False
    assert "rate limited" in result.reason


def test_no_articles_is_available_but_empty(monkeypatch):
    monkeypatch.setattr(news_api, "fetch_company_news", lambda *a, **k: ([], None))
    result = news_analyzer.analyze_company_news("Reliance Industries")
    assert result.available is True
    assert result.articles == []
    assert result.aggregated_sentiment_score is None


def test_duplicate_headlines_are_removed(monkeypatch):
    articles = [
        _article("Company reports strong earnings", hours_ago=1),
        _article("company reports strong earnings", hours_ago=2),  # same, different case
        _article("Company reports strong earnings  ", hours_ago=3),  # same, extra whitespace
        _article("Different headline entirely", hours_ago=4),
    ]
    monkeypatch.setattr(news_api, "fetch_company_news", lambda *a, **k: (articles, None))
    result = news_analyzer.analyze_company_news("Reliance Industries")
    assert len(result.articles) == 2


def test_aggregated_sentiment_weights_recent_articles_more(monkeypatch):
    articles = [
        _article("Terrible fraud scandal disaster catastrophe", hours_ago=0.1),   # very recent, very negative
        _article("Amazing record profit surge fantastic success", hours_ago=100),  # old, very positive
    ]
    monkeypatch.setattr(news_api, "fetch_company_news", lambda *a, **k: (articles, None))
    result = news_analyzer.analyze_company_news("Reliance Industries", lookback_hours=120)
    # The recent negative article should dominate the time-weighted aggregate.
    assert result.aggregated_sentiment_score < 0


def test_event_detection_wired_through(monkeypatch):
    articles = [_article("Company faces fraud investigation", hours_ago=1)]
    monkeypatch.setattr(news_api, "fetch_company_news", lambda *a, **k: (articles, None))
    result = news_analyzer.analyze_company_news("Reliance Industries")
    assert result.top_event_severity == 4  # SEVERITY_CRITICAL
    assert result.articles[0].event_severity == 4


def test_negative_high_impact_count(monkeypatch):
    articles = [
        _article("Company faces fraud investigation", hours_ago=1),
        _article("Regulatory investigation launched into company", hours_ago=2),
        _article("Company announces routine board meeting", hours_ago=3),
    ]
    monkeypatch.setattr(news_api, "fetch_company_news", lambda *a, **k: (articles, None))
    result = news_analyzer.analyze_company_news("Reliance Industries")
    assert result.negative_high_impact_count == 2


def test_max_articles_limit_respected(monkeypatch):
    articles = [_article(f"Headline number {i}", hours_ago=i) for i in range(20)]
    monkeypatch.setattr(news_api, "fetch_company_news", lambda *a, **k: (articles, None))
    result = news_analyzer.analyze_company_news("Reliance Industries")
    from src.config import load_config
    max_articles = load_config()["news_risk"]["max_articles"]
    assert len(result.articles) <= max_articles


def test_articles_sorted_most_recent_first(monkeypatch):
    articles = [
        _article("Older headline", hours_ago=10),
        _article("Newer headline", hours_ago=1),
    ]
    monkeypatch.setattr(news_api, "fetch_company_news", lambda *a, **k: (articles, None))
    result = news_analyzer.analyze_company_news("Reliance Industries")
    assert result.articles[0].headline == "Newer headline"
