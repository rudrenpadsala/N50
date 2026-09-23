"""
tests/test_event_detector.py

Tests for src/news/event_detector.py.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.news import event_detector  # noqa: E402


def test_no_match_returns_empty():
    matches = event_detector.detect_events("Company holds routine board meeting", sentiment_score=0.0)
    assert matches == []


def test_fraud_keyword_detected_as_critical():
    matches = event_detector.detect_events("Company accused of accounting fraud", sentiment_score=-0.6)
    assert any(m.event_type == "fraud" and m.severity == event_detector.SEVERITY_CRITICAL for m in matches)


def test_bankruptcy_keyword_detected_as_critical():
    matches = event_detector.detect_events("Firm files for bankruptcy protection", sentiment_score=-0.7)
    assert any(m.event_type == "bankruptcy" and m.severity == event_detector.SEVERITY_CRITICAL for m in matches)


def test_positive_event_major_acquisition():
    matches = event_detector.detect_events("Company to acquire smaller rival in $2B deal", sentiment_score=0.5)
    assert any(m.event_type == "major_acquisition" and m.direction == event_detector.POSITIVE for m in matches)


def test_severity_reduced_when_sentiment_conflicts_with_keyword():
    """'lawsuit' keyword normally implies negative severity 3 (High), but a
    headline that clearly reads as a WIN should have its severity reduced."""
    matches = event_detector.detect_events("Company wins major lawsuit victory", sentiment_score=0.6)
    lawsuit_match = next(m for m in matches if m.event_type == "major_lawsuit")
    assert lawsuit_match.base_severity == event_detector.SEVERITY_HIGH
    assert lawsuit_match.severity < lawsuit_match.base_severity


def test_severity_increased_when_sentiment_strongly_reinforces():
    matches = event_detector.detect_events("Regulatory investigation deepens into company practices",
                                            sentiment_score=-0.8)
    inv_match = next(m for m in matches if m.event_type == "regulatory_investigation")
    assert inv_match.severity >= inv_match.base_severity


def test_severity_never_exceeds_critical_or_below_none():
    matches = event_detector.detect_events("Company accused of accounting fraud", sentiment_score=-0.9)
    for m in matches:
        assert event_detector.SEVERITY_NONE <= m.severity <= event_detector.SEVERITY_CRITICAL


def test_top_severity_helper():
    matches = event_detector.detect_events(
        "Company faces regulatory investigation and also reports a product recall", sentiment_score=-0.4
    )
    assert event_detector.top_severity(matches) == max(m.severity for m in matches)
    assert event_detector.top_severity([]) == event_detector.SEVERITY_NONE


def test_multiple_events_in_one_headline():
    matches = event_detector.detect_events(
        "Company CEO resigns amid fraud investigation", sentiment_score=-0.7
    )
    event_types = {m.event_type for m in matches}
    assert "ceo_resignation" in event_types
    assert "fraud" in event_types
