"""
tests/test_news_api.py

Tests for src/news/news_api.py - all HTTP calls mocked, no real
network access or API key needed.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.news import news_api  # noqa: E402


class FakeResponse:
    def __init__(self, json_body):
        self._json_body = json_body

    def json(self):
        return self._json_body


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    news_api.clear_cache()
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    yield
    news_api.clear_cache()


def test_missing_api_key_returns_unavailable():
    articles, reason = news_api.fetch_company_news("Reliance Industries", 48)
    assert articles is None
    assert "not configured" in reason


def test_network_failure_returns_unavailable(monkeypatch):
    monkeypatch.setenv("NEWS_API_KEY", "test_key")

    def fake_get(url, params, timeout):
        raise news_api.requests.ConnectionError("boom")

    monkeypatch.setattr(news_api.requests, "get", fake_get)
    articles, reason = news_api.fetch_company_news("Reliance Industries", 48)
    assert articles is None
    assert "Could not reach" in reason


def test_api_error_status_returns_unavailable(monkeypatch):
    monkeypatch.setenv("NEWS_API_KEY", "test_key")

    def fake_get(url, params, timeout):
        return FakeResponse({"status": "error", "message": "Invalid API key"})

    monkeypatch.setattr(news_api.requests, "get", fake_get)
    articles, reason = news_api.fetch_company_news("Reliance Industries", 48)
    assert articles is None
    assert "Invalid API key" in reason


def test_successful_fetch_parses_articles(monkeypatch):
    monkeypatch.setenv("NEWS_API_KEY", "test_key")

    def fake_get(url, params, timeout):
        assert params["q"] == "Reliance Industries"
        return FakeResponse({
            "status": "ok",
            "articles": [
                {
                    "title": "Reliance posts strong quarterly profit",
                    "description": "Profit beats estimates.",
                    "source": {"name": "Example News"},
                    "publishedAt": "2026-09-09T10:00:00Z",
                    "url": "https://example.com/a1",
                },
            ],
        })

    monkeypatch.setattr(news_api.requests, "get", fake_get)
    articles, reason = news_api.fetch_company_news("Reliance Industries", 48)
    assert reason is None
    assert len(articles) == 1
    assert articles[0].headline == "Reliance posts strong quarterly profit"
    assert articles[0].source == "Example News"
    assert articles[0].url == "https://example.com/a1"


def test_malformed_articles_are_skipped(monkeypatch):
    monkeypatch.setenv("NEWS_API_KEY", "test_key")

    def fake_get(url, params, timeout):
        return FakeResponse({
            "status": "ok",
            "articles": [
                {"title": "", "publishedAt": "2026-09-09T10:00:00Z", "url": "https://x.com"},  # no title
                {"title": "Valid headline", "publishedAt": None, "url": "https://x.com"},  # no date
                {"title": "Valid headline 2", "publishedAt": "2026-09-09T10:00:00Z", "url": ""},  # no url
                {"title": "Actually valid", "publishedAt": "2026-09-09T10:00:00Z", "url": "https://x.com",
                 "source": {"name": "Src"}},
            ],
        })

    monkeypatch.setattr(news_api.requests, "get", fake_get)
    articles, reason = news_api.fetch_company_news("Reliance Industries", 48)
    assert reason is None
    assert len(articles) == 1
    assert articles[0].headline == "Actually valid"


def test_caching_avoids_second_network_call(monkeypatch):
    monkeypatch.setenv("NEWS_API_KEY", "test_key")
    calls = {"count": 0}

    def fake_get(url, params, timeout):
        calls["count"] += 1
        return FakeResponse({"status": "ok", "articles": []})

    monkeypatch.setattr(news_api.requests, "get", fake_get)
    news_api.fetch_company_news("Reliance Industries", 48)
    news_api.fetch_company_news("Reliance Industries", 48)
    assert calls["count"] == 1  # second call served from cache


def test_use_cache_false_bypasses_cache(monkeypatch):
    monkeypatch.setenv("NEWS_API_KEY", "test_key")
    calls = {"count": 0}

    def fake_get(url, params, timeout):
        calls["count"] += 1
        return FakeResponse({"status": "ok", "articles": []})

    monkeypatch.setattr(news_api.requests, "get", fake_get)
    news_api.fetch_company_news("Reliance Industries", 48, use_cache=True)
    news_api.fetch_company_news("Reliance Industries", 48, use_cache=False)
    assert calls["count"] == 2
