"""
tests/test_next_day.py

Tests for src/decision/next_day.py - price input validation, the
volatility-based price range estimate, and the Next Day Strategy
pipeline that combines them with the existing ensemble decision.
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.environment.actions import BUY  # noqa: E402
from src.environment.state import StateEncoder  # noqa: E402
from src.decision import engine, next_day  # noqa: E402


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidatePriceInput:

    def test_valid_price(self):
        price, error = next_day.validate_price_input("2850")
        assert price == pytest.approx(2850.0)
        assert error is None

    def test_valid_float_price(self):
        price, error = next_day.validate_price_input(2850.50)
        assert price == pytest.approx(2850.50)
        assert error is None

    def test_zero_is_invalid(self):
        price, error = next_day.validate_price_input(0)
        assert price is None
        assert error is not None

    def test_negative_is_invalid(self):
        price, error = next_day.validate_price_input(-100)
        assert price is None
        assert error is not None

    def test_empty_string_is_invalid(self):
        price, error = next_day.validate_price_input("")
        assert price is None
        assert error is not None

    def test_none_is_invalid(self):
        price, error = next_day.validate_price_input(None)
        assert price is None
        assert error is not None

    def test_text_is_invalid(self):
        price, error = next_day.validate_price_input("not a number")
        assert price is None
        assert error is not None

    def test_never_raises_on_garbage_input(self):
        for garbage in [object(), [1, 2, 3], {"a": 1}, float("nan")]:
            price, error = next_day.validate_price_input(garbage)
            assert price is None
            assert error is not None


# ---------------------------------------------------------------------------
# Price range estimate
# ---------------------------------------------------------------------------

class TestEstimatePriceRange:

    @pytest.fixture
    def fake_features(self, tmp_path, monkeypatch):
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        monkeypatch.setattr(engine, "FEATURES_DIR", str(features_dir))
        n = 30
        df = pd.DataFrame({
            "Date": pd.date_range("2025-01-01", periods=n),
            "Close": [100.0] * n,
            "MA5": [100.0] * n, "MA20": [100.0] * n, "RSI": [50.0] * n,
            "Volatility": [0.02] * n,  # 2% daily std
            "Trend": ["NEUTRAL"] * n, "RSI_Condition": ["RSI_NORMAL"] * n,
            "Volatility_Condition": ["LOW_VOLATILITY"] * n,
        })
        df.to_csv(features_dir / "TEST_features.csv", index=False)
        return features_dir

    def test_range_centered_on_latest_price(self, fake_features):
        result = next_day.estimate_price_range("TEST")
        assert result["basis_price"] == pytest.approx(100.0)
        assert result["low"] == pytest.approx(98.0)   # 100 * (1 - 0.02)
        assert result["high"] == pytest.approx(102.0)  # 100 * (1 + 0.02)

    def test_returns_none_when_volatility_missing(self, tmp_path, monkeypatch):
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        monkeypatch.setattr(engine, "FEATURES_DIR", str(features_dir))
        n = 5
        df = pd.DataFrame({
            "Date": pd.date_range("2025-01-01", periods=n),
            "Close": [100.0] * n, "MA5": [100.0] * n, "MA20": [100.0] * n, "RSI": [50.0] * n,
            "Volatility": [np.nan] * n,
            "Trend": ["NEUTRAL"] * n, "RSI_Condition": ["RSI_NORMAL"] * n,
            "Volatility_Condition": ["LOW_VOLATILITY"] * n,
        })
        df.to_csv(features_dir / "TEST_features.csv", index=False)
        assert next_day.estimate_price_range("TEST") is None

    def test_custom_std_multiplier_widens_range(self, fake_features):
        r1 = next_day.estimate_price_range("TEST", std_multiplier=1.0)
        r2 = next_day.estimate_price_range("TEST", std_multiplier=2.0)
        assert (r2["high"] - r2["low"]) > (r1["high"] - r1["low"])


# ---------------------------------------------------------------------------
# Next Day Strategy pipeline
# ---------------------------------------------------------------------------

class TestNextDayStrategy:

    @pytest.fixture
    def fake_project(self, tmp_path, monkeypatch):
        features_dir = tmp_path / "features"
        models_dir = tmp_path / "models"
        trades_dir = tmp_path / "trades"
        features_dir.mkdir()
        models_dir.mkdir()
        trades_dir.mkdir()
        monkeypatch.setattr(engine, "FEATURES_DIR", str(features_dir))
        monkeypatch.setattr(engine, "MODELS_DIR", str(models_dir))
        monkeypatch.setattr(engine, "TRADES_DIR", str(trades_dir))

        n = 30
        df = pd.DataFrame({
            "Date": pd.date_range("2025-01-01", periods=n),
            "Close": [100.0] * n, "MA5": [101.0] * n, "MA20": [99.0] * n, "RSI": [60.0] * n,
            "Volatility": [0.02] * n,
            "Trend": ["BULLISH"] * n, "RSI_Condition": ["RSI_NORMAL"] * n,
            "Volatility_Condition": ["LOW_VOLATILITY"] * n,
        })
        df.to_csv(features_dir / "TEST_features.csv", index=False)

        encoder = StateEncoder()
        q_table = np.zeros((encoder.num_states, 3))
        q_table[:, BUY] = 1.0
        symbol_dir = models_dir / "TEST"
        symbol_dir.mkdir()
        with open(symbol_dir / "q_learning.pkl", "wb") as f:
            pickle.dump({"q_table": q_table}, f)

        return features_dir

    def test_entry_price_does_not_change_the_action(self, fake_project):
        """The AI action must be identical regardless of the entry price
        the user typed in - only the framing (upside/downside) changes."""
        r1 = next_day.next_day_strategy("TEST", entry_price=90.0)
        r2 = next_day.next_day_strategy("TEST", entry_price=110.0)
        assert r1["action"] == r2["action"] == BUY

    def test_upside_downside_relative_to_entry_price(self, fake_project):
        result = next_day.next_day_strategy("TEST", entry_price=100.0)
        assert result["price_range"]["low"] == pytest.approx(98.0)
        assert result["price_range"]["high"] == pytest.approx(102.0)
        assert result["upside_pct"] == pytest.approx(2.0)
        assert result["downside_pct"] == pytest.approx(-2.0)

    def test_asymmetric_when_entry_differs_from_latest_price(self, fake_project):
        result = next_day.next_day_strategy("TEST", entry_price=99.0)
        # range is [98, 102], entry 99 -> upside (102-99)/99, downside (98-99)/99
        assert result["upside_pct"] == pytest.approx((102.0 - 99.0) / 99.0 * 100.0)
        assert result["downside_pct"] == pytest.approx((98.0 - 99.0) / 99.0 * 100.0)
        assert result["upside_pct"] != pytest.approx(abs(result["downside_pct"]))

    def test_raises_when_no_models_trained(self, tmp_path, monkeypatch):
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        n = 5
        pd.DataFrame({
            "Date": pd.date_range("2025-01-01", periods=n),
            "Close": [100] * n, "MA5": [100] * n, "MA20": [100] * n, "RSI": [50] * n,
            "Volatility": [0.02] * n,
            "Trend": ["NEUTRAL"] * n, "RSI_Condition": ["RSI_NORMAL"] * n,
            "Volatility_Condition": ["LOW_VOLATILITY"] * n,
        }).to_csv(features_dir / "NOPE_features.csv", index=False)
        monkeypatch.setattr(engine, "FEATURES_DIR", str(features_dir))
        monkeypatch.setattr(engine, "MODELS_DIR", str(tmp_path / "empty_models"))
        monkeypatch.setattr(engine, "TRADES_DIR", str(tmp_path / "empty_trades"))
        with pytest.raises(engine.DecisionError):
            next_day.next_day_strategy("NOPE", entry_price=100.0)