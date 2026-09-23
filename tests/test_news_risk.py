"""
tests/test_news_risk.py

Tests for src/risk/news_risk.py.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.news import event_detector  # noqa: E402
from src.news.news_analyzer import AnalyzedArticle, NewsAnalysis  # noqa: E402
from src.risk import news_risk  # noqa: E402


def _article(sentiment_score=0.0, events=None, hours_ago=1.0, headline="Test headline"):
    events = events or []
    return AnalyzedArticle(
        headline=headline, description=None, source="Example", url="https://x.com",
        published_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
        sentiment_label="NEUTRAL", sentiment_score=sentiment_score, sentiment_model="test",
        events=events, event_severity=event_detector.top_severity(events), hours_ago=hours_ago,
    )


def _analysis(articles, aggregated_sentiment_score=0.0, top_event_severity=0, negative_high_impact_count=0):
    return NewsAnalysis(
        available=True, company_query="Test Co", lookback_hours=48, articles=articles,
        aggregated_sentiment_score=aggregated_sentiment_score,
        aggregated_sentiment_label="NEUTRAL",
        top_event_severity=top_event_severity, negative_high_impact_count=negative_high_impact_count,
    )


def test_unavailable_propagates():
    analysis = NewsAnalysis(available=False, company_query="X", lookback_hours=48, reason="News API error: boom")
    result = news_risk.assess_news_risk(analysis)
    assert result.available is False
    assert "boom" in result.unavailable_reason


def test_no_articles_is_none_risk():
    analysis = _analysis(articles=[], aggregated_sentiment_score=None)
    result = news_risk.assess_news_risk(analysis)
    assert result.available is True
    assert result.news_risk_score == news_risk.RISK_NONE


def test_critical_event_gives_critical_risk():
    fraud_event = event_detector.EventMatch("fraud", "negative", event_detector.SEVERITY_CRITICAL,
                                             event_detector.SEVERITY_CRITICAL, "fraud")
    article = _article(sentiment_score=-0.8, events=[fraud_event], hours_ago=1)
    analysis = _analysis([article], aggregated_sentiment_score=-0.8, top_event_severity=4,
                          negative_high_impact_count=1)
    result = news_risk.assess_news_risk(analysis)
    assert result.news_risk_score == news_risk.RISK_CRITICAL


def test_very_negative_sentiment_with_high_impact_is_critical():
    result = news_risk.assess_news_risk(_analysis(
        [_article(sentiment_score=-0.8)], aggregated_sentiment_score=-0.8,
        top_event_severity=0, negative_high_impact_count=1,
    ))
    assert result.news_risk_score == news_risk.RISK_CRITICAL


def test_positive_sentiment_no_events_is_none_risk():
    result = news_risk.assess_news_risk(_analysis(
        [_article(sentiment_score=0.6)], aggregated_sentiment_score=0.6,
        top_event_severity=0, negative_high_impact_count=0,
    ))
    assert result.news_risk_score == news_risk.RISK_NONE


def test_moderate_negative_sentiment_gives_moderate_risk():
    result = news_risk.assess_news_risk(_analysis(
        [_article(sentiment_score=-0.3)], aggregated_sentiment_score=-0.3,
        top_event_severity=0, negative_high_impact_count=0,
    ))
    assert result.news_risk_score == news_risk.RISK_MODERATE


def test_recent_high_severity_negative_event_forces_at_least_high():
    """Even if aggregated sentiment alone looks moderate, a HIGH-severity
    negative event within the recency window must push risk to at least HIGH."""
    inv_event = event_detector.EventMatch("regulatory_investigation", "negative",
                                           event_detector.SEVERITY_HIGH, event_detector.SEVERITY_HIGH, "investigation")
    article = _article(sentiment_score=-0.3, events=[inv_event], hours_ago=1.0)  # within _RECENT_HOURS
    analysis = _analysis([article], aggregated_sentiment_score=-0.3, top_event_severity=3,
                          negative_high_impact_count=1)
    result = news_risk.assess_news_risk(analysis)
    assert result.news_risk_score >= news_risk.RISK_HIGH


def test_reasons_are_nonempty():
    result = news_risk.assess_news_risk(_analysis([_article()], aggregated_sentiment_score=0.0))
    assert len(result.reasons) > 0
