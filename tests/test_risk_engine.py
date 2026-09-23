"""
tests/test_risk_engine.py

Integration tests for src/risk/risk_engine.py - the single entry point
the dashboard calls. engine.get_decision(), engine.is_holding_position(),
and news_analyzer.analyze_company_news() are all monkeypatched so this
never touches real models, files, or the network.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.decision import engine  # noqa: E402
from src.news import event_detector  # noqa: E402
from src.news.news_analyzer import AnalyzedArticle, NewsAnalysis  # noqa: E402
from src.risk import risk_engine  # noqa: E402


def _fake_decision(action=1, confidence_pct=80.0, data_source="live", **extra):
    base = {
        "action": action, "confidence_pct": confidence_pct, "confidence_label": "High",
        "data_source": data_source, "latest_price": 105.0, "previous_close": 100.0,
        "volume_change_pct": 0.0, "volatility_condition": "LOW_VOLATILITY", "drawdown_pct": 0.0,
        "votes": {0: 1, 1: 4, 2: 0}, "weak_signal": False,
    }
    base.update(extra)
    return base


def _fake_news(available=True, articles=None, aggregated_sentiment_score=0.0,
               aggregated_sentiment_label="NEUTRAL", top_event_severity=0,
               negative_high_impact_count=0, reason=None):
    return NewsAnalysis(
        available=available, company_query="Test Co", lookback_hours=48,
        articles=articles or [], aggregated_sentiment_score=aggregated_sentiment_score,
        aggregated_sentiment_label=aggregated_sentiment_label,
        top_event_severity=top_event_severity, negative_high_impact_count=negative_high_impact_count,
        reason=reason,
    )


class TestRiskEngineEndToEnd:

    def test_calm_buy_stays_buy(self, monkeypatch):
        monkeypatch.setattr(engine, "get_decision",
                             lambda symbol, prefer_live=True: _fake_decision(action=1, latest_price=100.5, previous_close=100.0))
        monkeypatch.setattr(engine, "is_holding_position", lambda symbol: False)
        monkeypatch.setattr(risk_engine, "analyze_company_news",
                             lambda name, **k: _fake_news(aggregated_sentiment_score=0.4, aggregated_sentiment_label="POSITIVE"))

        result = risk_engine.evaluate("TEST", "Test Company")
        assert result.final_action == "BUY"
        assert result.overall_risk_level in (risk_engine.OVERALL_LOW, risk_engine.OVERALL_MODERATE)
        assert len(result.reasons) <= 3

    def test_critical_news_overrides_buy(self, monkeypatch):
        import datetime
        fraud_event = event_detector.EventMatch("fraud", "negative", event_detector.SEVERITY_CRITICAL,
                                                  event_detector.SEVERITY_CRITICAL, "fraud")
        article = AnalyzedArticle(
            headline="Company accused of accounting fraud", description=None, source="Example",
            url="https://x.com", published_at=datetime.datetime.now(datetime.timezone.utc),
            sentiment_label="NEGATIVE", sentiment_score=-0.9, sentiment_model="test",
            events=[fraud_event], event_severity=event_detector.SEVERITY_CRITICAL, hours_ago=1.0,
        )
        monkeypatch.setattr(engine, "get_decision",
                             lambda symbol, prefer_live=True: _fake_decision(action=1, latest_price=100.5, previous_close=100.0))
        monkeypatch.setattr(engine, "is_holding_position", lambda symbol: False)
        monkeypatch.setattr(risk_engine, "analyze_company_news",
                             lambda name, **k: _fake_news(articles=[article], aggregated_sentiment_score=-0.9,
                                                           aggregated_sentiment_label="NEGATIVE",
                                                           top_event_severity=4, negative_high_impact_count=1))

        result = risk_engine.evaluate("TEST", "Test Company")
        assert result.final_action == "AVOID"  # not holding, critical event
        assert result.overall_risk_level == risk_engine.OVERALL_EXTREME

    def test_news_unavailable_does_not_crash_and_uses_rl_signal(self, monkeypatch):
        monkeypatch.setattr(engine, "get_decision", lambda symbol, prefer_live=True: _fake_decision(action=1))
        monkeypatch.setattr(engine, "is_holding_position", lambda symbol: False)
        monkeypatch.setattr(risk_engine, "analyze_company_news",
                             lambda name, **k: _fake_news(available=False, reason="NEWS_API_KEY is not configured."))

        result = risk_engine.evaluate("TEST", "Test Company")
        assert result.final_action == "BUY"  # no news risk data -> RL signal used as-is
        assert result.news.available is False
        assert any("unavailable" in r.lower() for r in result.reasons)

    def test_market_data_unavailable_does_not_crash(self, monkeypatch):
        monkeypatch.setattr(engine, "get_decision",
                             lambda symbol, prefer_live=True: _fake_decision(action=1, previous_close=None))
        monkeypatch.setattr(engine, "is_holding_position", lambda symbol: False)
        monkeypatch.setattr(risk_engine, "analyze_company_news", lambda name, **k: _fake_news())

        result = risk_engine.evaluate("TEST", "Test Company")
        assert result.market.available is False
        assert result.final_action == "BUY"

    def test_rl_decision_itself_is_unmodified(self, monkeypatch):
        """The RL layer's own output must pass through completely
        untouched - the risk layer only adds a final_action on top."""
        fake = _fake_decision(action=2, confidence_pct=55.0)
        monkeypatch.setattr(engine, "get_decision", lambda symbol, prefer_live=True: fake)
        monkeypatch.setattr(engine, "is_holding_position", lambda symbol: True)
        monkeypatch.setattr(risk_engine, "analyze_company_news", lambda name, **k: _fake_news())

        result = risk_engine.evaluate("TEST", "Test Company")
        assert result.rl_decision is fake
        assert result.rl_decision["action"] == 2

    def test_sell_no_position_shown_as_avoid_end_to_end(self, monkeypatch):
        monkeypatch.setattr(engine, "get_decision", lambda symbol, prefer_live=True: _fake_decision(action=2))
        monkeypatch.setattr(engine, "is_holding_position", lambda symbol: False)
        monkeypatch.setattr(risk_engine, "analyze_company_news", lambda name, **k: _fake_news())

        result = risk_engine.evaluate("TEST", "Test Company")
        assert result.final_action == "AVOID"

    def test_reasons_limited_to_three(self, monkeypatch):
        monkeypatch.setattr(engine, "get_decision", lambda symbol, prefer_live=True: _fake_decision(action=1))
        monkeypatch.setattr(engine, "is_holding_position", lambda symbol: False)
        monkeypatch.setattr(risk_engine, "analyze_company_news",
                             lambda name, **k: _fake_news(aggregated_sentiment_score=-0.3,
                                                           aggregated_sentiment_label="NEGATIVE",
                                                           top_event_severity=2))
        result = risk_engine.evaluate("TEST", "Test Company")
        assert len(result.reasons) <= 3
