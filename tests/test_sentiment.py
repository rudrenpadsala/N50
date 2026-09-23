"""
tests/test_sentiment.py

Tests for src/news/sentiment.py - real VADER (no mocking needed, it's
a local lexicon model with no network dependency), plus tests that
FinBERT's absence is handled gracefully.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.news import sentiment  # noqa: E402


def test_clearly_positive_headline():
    result = sentiment.analyze_sentiment("Company reports record profit, beats all estimates", prefer_finbert=False)
    assert result.label == sentiment.POSITIVE
    assert result.score > 0
    assert result.model_name == sentiment.VADER_MODEL_NAME


def test_clearly_negative_headline():
    result = sentiment.analyze_sentiment("Company faces fraud investigation, shares plunge", prefer_finbert=False)
    assert result.label == sentiment.NEGATIVE
    assert result.score < 0


def test_neutral_headline():
    result = sentiment.analyze_sentiment("Company to hold its annual general meeting next week", prefer_finbert=False)
    assert result.label == sentiment.NEUTRAL


def test_empty_text_is_neutral_zero():
    result = sentiment.analyze_sentiment("", prefer_finbert=False)
    assert result.label == sentiment.NEUTRAL
    assert result.score == 0.0


def test_score_always_in_valid_range():
    for text in ["Amazing record-breaking historic profit surge!!!", "Catastrophic fraud collapse disaster bankruptcy"]:
        result = sentiment.analyze_sentiment(text, prefer_finbert=False)
        assert -1.0 <= result.score <= 1.0


def test_label_for_score_thresholds():
    assert sentiment.label_for_score(0.5) == sentiment.POSITIVE
    assert sentiment.label_for_score(0.0) == sentiment.NEUTRAL
    assert sentiment.label_for_score(-0.5) == sentiment.NEGATIVE
    assert sentiment.label_for_score(0.25) == sentiment.POSITIVE   # boundary
    assert sentiment.label_for_score(-0.25) == sentiment.NEGATIVE  # boundary


def test_finbert_unavailable_falls_back_to_vader(monkeypatch):
    """If transformers/torch aren't installed (the common case for this
    project), analyze_sentiment must still return a real VADER result,
    never raise, even when prefer_finbert=True."""
    sentiment._finbert_pipeline = None
    sentiment._finbert_unavailable = False

    def fake_get_finbert():
        return None  # simulates a failed/unavailable model load

    monkeypatch.setattr(sentiment, "_get_finbert", fake_get_finbert)
    result = sentiment.analyze_sentiment("Company reports strong earnings", prefer_finbert=True)
    assert result.model_name == sentiment.VADER_MODEL_NAME


def test_finbert_import_error_is_caught_and_sticky(monkeypatch):
    """A real ImportError (transformers not installed) must be caught,
    not propagate, and must not be retried on every single call."""
    sentiment._finbert_pipeline = None
    sentiment._finbert_unavailable = False

    call_count = {"n": 0}
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers":
            call_count["n"] += 1
            raise ImportError("no module named transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    result1 = sentiment.analyze_sentiment("Some headline", prefer_finbert=True)
    result2 = sentiment.analyze_sentiment("Another headline", prefer_finbert=True)
    assert result1.model_name == sentiment.VADER_MODEL_NAME
    assert result2.model_name == sentiment.VADER_MODEL_NAME
    assert call_count["n"] == 1  # sticky - only tried to import once, not on every call
