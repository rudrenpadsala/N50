"""
tests/test_market_shock.py

Tests for src/risk/market_shock.py - built entirely on a fake
engine.get_decision()-shaped dict, so no real market API is involved.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.risk import market_shock  # noqa: E402


def _decision(latest_price=100.0, previous_close=100.0, volume_change_pct=0.0,
              volatility_condition="LOW_VOLATILITY", drawdown_pct=0.0):
    return {
        "latest_price": latest_price, "previous_close": previous_close,
        "volume_change_pct": volume_change_pct, "volatility_condition": volatility_condition,
        "drawdown_pct": drawdown_pct,
    }


def test_unavailable_without_previous_close():
    result = market_shock.assess_market_shock({"latest_price": 100.0, "previous_close": None})
    assert result.available is False
    assert result.unavailable_reason is not None


def test_normal_day_is_low_risk():
    result = market_shock.assess_market_shock(_decision(latest_price=101.0, previous_close=100.0))
    assert result.available is True
    assert result.market_risk_score == market_shock.RISK_LOW
    assert result.price_shock is False


def test_large_price_move_flagged():
    result = market_shock.assess_market_shock(_decision(latest_price=106.0, previous_close=100.0))  # +6%
    assert result.price_shock is True
    assert result.extreme_price_shock is False
    assert result.market_risk_score >= market_shock.RISK_HIGH


def test_extreme_price_move_flagged():
    result = market_shock.assess_market_shock(_decision(latest_price=90.0, previous_close=100.0))  # -10%
    assert result.extreme_price_shock is True
    assert result.market_risk_score == market_shock.RISK_EXTREME


def test_volume_spike_detected():
    result = market_shock.assess_market_shock(_decision(volume_change_pct=150.0))  # 2.5x previous day
    assert result.volume_spike is True
    assert result.market_risk_score >= market_shock.RISK_MODERATE


def test_high_volatility_detected():
    result = market_shock.assess_market_shock(_decision(volatility_condition="HIGH_VOLATILITY"))
    assert result.high_volatility is True
    assert result.market_risk_score >= market_shock.RISK_MODERATE


def test_significant_drawdown_detected():
    result = market_shock.assess_market_shock(_decision(drawdown_pct=15.0))
    assert result.significant_drawdown is True
    assert result.market_risk_score >= market_shock.RISK_MODERATE


def test_multiple_moderate_signals_escalate_to_extreme():
    result = market_shock.assess_market_shock(_decision(
        latest_price=106.0, previous_close=100.0,  # price shock (high)
        volume_change_pct=150.0,                    # volume spike
        volatility_condition="HIGH_VOLATILITY",      # high volatility
        drawdown_pct=15.0,                           # drawdown
    ))
    assert result.market_risk_score == market_shock.RISK_EXTREME


def test_reasons_are_human_readable_and_nonempty_when_risky():
    result = market_shock.assess_market_shock(_decision(latest_price=110.0, previous_close=100.0))
    assert len(result.reasons) > 0
    assert all(isinstance(r, str) and r for r in result.reasons)


def test_missing_optional_fields_dont_crash():
    """volume_change_pct/volatility_condition/drawdown_pct may all be
    None (e.g. a historical-fallback decision with no live extras)."""
    result = market_shock.assess_market_shock({
        "latest_price": 100.0, "previous_close": 100.0,
        "volume_change_pct": None, "volatility_condition": None, "drawdown_pct": None,
    })
    assert result.available is True
    assert result.market_risk_score == market_shock.RISK_LOW
