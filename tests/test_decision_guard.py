"""
tests/test_decision_guard.py

Tests for src/risk/decision_guard.py - the rule hierarchy that must
NEVER simply average RL action + news sentiment (project spec #12).
These tests directly cover the required scenarios from spec #26.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.news import event_detector  # noqa: E402
from src.risk import decision_guard, market_shock, news_risk  # noqa: E402


class TestNormalAndPositiveCases:

    def test_buy_plus_positive_news_stays_buy(self):
        result = decision_guard.apply_risk_guard(
            rl_action="BUY", rl_confidence_pct=80.0,
            news_risk_score=news_risk.RISK_NONE, market_risk_score=market_shock.RISK_LOW,
            top_event_severity=0, holding_position=False, aggregated_sentiment_score=0.5,
        )
        assert result.final_action == "BUY"
        assert result.overridden is False

    def test_hold_stays_hold_when_calm(self):
        result = decision_guard.apply_risk_guard(
            rl_action="HOLD", rl_confidence_pct=60.0,
            news_risk_score=news_risk.RISK_NONE, market_risk_score=market_shock.RISK_LOW,
            top_event_severity=0, holding_position=True,
        )
        assert result.final_action == "HOLD"


class TestNegativeNewsOverrides:

    def test_buy_plus_moderate_negative_news_becomes_hold(self):
        result = decision_guard.apply_risk_guard(
            rl_action="BUY", rl_confidence_pct=75.0,
            news_risk_score=news_risk.RISK_MODERATE, market_risk_score=market_shock.RISK_LOW,
            top_event_severity=2, holding_position=False,
        )
        assert result.final_action == "HOLD"
        assert result.overridden is True
        assert result.model_agreement_pct < result.model_agreement_raw_pct  # confidence reduced

    def test_buy_plus_high_news_and_high_market_risk_becomes_hold(self):
        result = decision_guard.apply_risk_guard(
            rl_action="BUY", rl_confidence_pct=80.0,
            news_risk_score=news_risk.RISK_HIGH, market_risk_score=market_shock.RISK_HIGH,
            top_event_severity=3, holding_position=False,
        )
        assert result.final_action == "HOLD"
        assert result.overridden is True

    def test_hold_is_not_pushed_around_by_moderate_news(self):
        """Risk only ever downgrades a BUY - it doesn't need to further
        soften a signal that's already cautious (HOLD)."""
        result = decision_guard.apply_risk_guard(
            rl_action="HOLD", rl_confidence_pct=50.0,
            news_risk_score=news_risk.RISK_MODERATE, market_risk_score=market_shock.RISK_LOW,
            top_event_severity=2, holding_position=False,
        )
        assert result.final_action == "HOLD"


class TestCriticalEventProtection:

    def test_critical_event_forces_sell_when_holding(self):
        result = decision_guard.apply_risk_guard(
            rl_action="BUY", rl_confidence_pct=80.0,
            news_risk_score=news_risk.RISK_CRITICAL, market_risk_score=market_shock.RISK_MODERATE,
            top_event_severity=event_detector.SEVERITY_CRITICAL, holding_position=True,
        )
        assert result.final_action == "SELL"
        assert result.overridden is True
        assert "critical" in result.reason.lower()

    def test_critical_event_forces_avoid_when_not_holding(self):
        result = decision_guard.apply_risk_guard(
            rl_action="BUY", rl_confidence_pct=80.0,
            news_risk_score=news_risk.RISK_CRITICAL, market_risk_score=market_shock.RISK_MODERATE,
            top_event_severity=event_detector.SEVERITY_CRITICAL, holding_position=False,
        )
        assert result.final_action == "AVOID"

    def test_critical_event_overrides_even_an_rl_hold(self):
        result = decision_guard.apply_risk_guard(
            rl_action="HOLD", rl_confidence_pct=60.0,
            news_risk_score=news_risk.RISK_CRITICAL, market_risk_score=market_shock.RISK_LOW,
            top_event_severity=event_detector.SEVERITY_CRITICAL, holding_position=True,
        )
        assert result.final_action == "SELL"


class TestPositionAwareSellVsAvoid:

    def test_rl_sell_with_no_position_becomes_avoid(self):
        result = decision_guard.apply_risk_guard(
            rl_action="SELL", rl_confidence_pct=70.0,
            news_risk_score=news_risk.RISK_NONE, market_risk_score=market_shock.RISK_LOW,
            top_event_severity=0, holding_position=False,
        )
        assert result.final_action == "AVOID"

    def test_rl_sell_with_position_stays_sell(self):
        result = decision_guard.apply_risk_guard(
            rl_action="SELL", rl_confidence_pct=70.0,
            news_risk_score=news_risk.RISK_NONE, market_risk_score=market_shock.RISK_LOW,
            top_event_severity=0, holding_position=True,
        )
        assert result.final_action == "SELL"


class TestNeverJustAverages:

    def test_buy_and_extreme_negative_is_not_a_blind_average_to_hold(self):
        """This must go through the explicit rule hierarchy (ending in
        SELL/AVOID for a critical event), never a naive
        'BUY + SELL-ish news = HOLD' average."""
        result = decision_guard.apply_risk_guard(
            rl_action="BUY", rl_confidence_pct=90.0,
            news_risk_score=news_risk.RISK_CRITICAL, market_risk_score=market_shock.RISK_EXTREME,
            top_event_severity=event_detector.SEVERITY_CRITICAL, holding_position=False,
        )
        assert result.final_action in ("AVOID", "SELL")
        assert result.final_action != "HOLD"


class TestModelAgreementConfidence:

    def test_confidence_never_increases(self):
        result = decision_guard.apply_risk_guard(
            rl_action="BUY", rl_confidence_pct=80.0,
            news_risk_score=news_risk.RISK_HIGH, market_risk_score=market_shock.RISK_HIGH,
            top_event_severity=3, holding_position=False,
        )
        assert result.model_agreement_pct <= result.model_agreement_raw_pct

    def test_confidence_floor_respected(self):
        result = decision_guard.apply_risk_guard(
            rl_action="BUY", rl_confidence_pct=25.0,
            news_risk_score=news_risk.RISK_CRITICAL, market_risk_score=market_shock.RISK_EXTREME,
            top_event_severity=4, holding_position=False,
        )
        assert result.model_agreement_pct >= decision_guard.MIN_DISPLAYED_CONFIDENCE

    def test_no_penalty_when_calm(self):
        result = decision_guard.apply_risk_guard(
            rl_action="HOLD", rl_confidence_pct=65.0,
            news_risk_score=news_risk.RISK_NONE, market_risk_score=market_shock.RISK_LOW,
            top_event_severity=0, holding_position=False,
        )
        assert result.model_agreement_pct == pytest.approx(65.0)
