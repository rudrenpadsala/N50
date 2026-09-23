"""
tests/test_decision_engine.py

Tests for src/decision/engine.py - the ensemble/voting layer that turns
5 algorithms' individual actions into ONE final BUY/HOLD/SELL decision
for the redesigned dashboard.
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

from src.environment.actions import HOLD, BUY, SELL  # noqa: E402
from src.environment.state import StateEncoder, NO_POSITION, HOLDING  # noqa: E402
from src.decision import engine  # noqa: E402


# ---------------------------------------------------------------------------
# Confidence labeling
# ---------------------------------------------------------------------------

class TestConfidenceLabel:

    @pytest.mark.parametrize("pct,expected", [
        (100.0, "High"), (80.0, "High"), (79.9, "Medium"),
        (60.0, "Medium"), (59.9, "Low"), (0.0, "Low"), (40.0, "Low"),
    ])
    def test_thresholds(self, pct, expected):
        assert engine.confidence_label(pct) == expected


# ---------------------------------------------------------------------------
# Voting + tie-break
# ---------------------------------------------------------------------------

class TestCombineVotes:
    """
    Q-rows in this class use `decisive_q(winner)` (a clear top-2 margin
    of 1.0, well above MARGIN_THRESHOLD) unless a test specifically
    exercises the weak-signal downgrade - so these tests isolate the
    vote-counting/tie-break logic from the separate signal-strength
    check (covered in TestWeakSignal below).
    """

    @staticmethod
    def decisive_q(winner: int) -> np.ndarray:
        q = np.array([0.0, 0.0, 0.0])
        q[winner] = 1.0
        return q

    def test_clear_majority_wins(self):
        actions = {"Q-Learning": BUY, "SARSA": BUY, "Monte Carlo": BUY,
                   "Value Iteration": HOLD, "Policy Iteration": SELL}
        q_rows = {k: self.decisive_q(a) for k, a in actions.items()}
        result = engine.combine_votes(actions, q_rows)
        assert result["action"] == BUY
        assert result["votes"] == {HOLD: 1, BUY: 3, SELL: 1}
        assert result["confidence_pct"] == pytest.approx(60.0)
        assert result["confidence_label"] == "Medium"
        assert result["tie_broken"] is False
        assert result["weak_signal"] is False

    def test_unanimous_is_high_confidence(self):
        actions = {f"a{i}": HOLD for i in range(5)}
        q_rows = {k: self.decisive_q(HOLD) for k in actions}
        result = engine.combine_votes(actions, q_rows)
        assert result["action"] == HOLD
        assert result["confidence_pct"] == pytest.approx(100.0)
        assert result["confidence_label"] == "High"
        assert result["tie_broken"] is False
        assert result["weak_signal"] is False

    def test_tie_with_hold_resolves_to_hold(self):
        # 2 BUY, 2 SELL, 1 HOLD -> tied top is {BUY, SELL} actually...
        # construct a genuine HOLD-involved tie: 2 HOLD, 2 BUY, 1 SELL
        actions = {"a": HOLD, "b": HOLD, "c": BUY, "d": BUY, "e": SELL}
        q_rows = {k: self.decisive_q(a) for k, a in actions.items()}
        result = engine.combine_votes(actions, q_rows)
        assert result["action"] == HOLD
        assert result["tie_broken"] is True
        assert result["confidence_pct"] == pytest.approx(40.0)
        assert result["confidence_label"] == "Low"

    def test_buy_sell_tie_uses_average_q_value(self):
        # 2 BUY, 2 SELL, 1 HOLD -> BUY vs SELL tie, broken by average Q.
        actions = {"a": BUY, "b": BUY, "c": SELL, "d": SELL, "e": HOLD}
        # Make SELL's average Q-value clearly higher than BUY's.
        q_rows = {
            "a": np.array([0.0, 1.0, 5.0]),   # HOLD, BUY, SELL
            "b": np.array([0.0, 1.0, 5.0]),
            "c": np.array([0.0, 1.0, 5.0]),
            "d": np.array([0.0, 1.0, 5.0]),
            "e": np.array([0.0, 1.0, 5.0]),
        }
        result = engine.combine_votes(actions, q_rows)
        assert result["action"] == SELL
        assert result["tie_broken"] is True

    def test_buy_sell_tie_favors_buy_when_buy_q_higher(self):
        actions = {"a": BUY, "b": BUY, "c": SELL, "d": SELL, "e": HOLD}
        q_rows = {k: np.array([0.0, 5.0, 1.0]) for k in actions}  # BUY higher
        result = engine.combine_votes(actions, q_rows)
        assert result["action"] == BUY

    def test_confidence_uses_only_participating_algorithms(self):
        """With only 3 algorithms voting (2 missing), confidence is out
        of 3, not out of 5."""
        actions = {"a": BUY, "b": BUY, "c": HOLD}
        q_rows = {k: self.decisive_q(a) for k, a in actions.items()}
        result = engine.combine_votes(actions, q_rows)
        assert result["votes"] == {HOLD: 1, BUY: 2, SELL: 0}
        assert result["confidence_pct"] == pytest.approx((2 / 3) * 100)


# ---------------------------------------------------------------------------
# Signal-strength downgrade
# ---------------------------------------------------------------------------

class TestWeakSignal:

    def test_near_tied_q_values_downgrade_confidence(self):
        """Unanimous vote (would normally be 100% / High), but every
        algorithm's top-2 Q-value margin is far below MARGIN_THRESHOLD -
        confidence must be downgraded, action must NOT change."""
        actions = {f"a{i}": HOLD for i in range(5)}
        q_rows = {k: np.array([0.00021, 0.0002, 0.00019]) for k in actions}  # margin ~0.00001
        result = engine.combine_votes(actions, q_rows)
        assert result["action"] == HOLD  # winning action unchanged
        assert result["confidence_pct"] == pytest.approx(100.0)  # raw vote share unchanged
        assert result["weak_signal"] is True
        assert result["confidence_label"] == "Medium"  # downgraded from High

    def test_decisive_q_values_do_not_downgrade(self):
        actions = {f"a{i}": BUY for i in range(5)}
        q_rows = {k: np.array([0.0, 1.0, 0.0]) for k in actions}  # margin = 1.0
        result = engine.combine_votes(actions, q_rows)
        assert result["weak_signal"] is False
        assert result["confidence_label"] == "High"

    def test_downgrade_only_considers_winning_action_voters(self):
        """The margin check averages over algorithms that voted for the
        WINNING action only - a dissenting voter's tiny margin should
        not affect the winner's reported confidence."""
        actions = {"a": BUY, "b": BUY, "c": BUY, "d": HOLD, "e": SELL}
        q_rows = {
            "a": np.array([0.0, 1.0, 0.0]),      # decisive BUY
            "b": np.array([0.0, 1.0, 0.0]),      # decisive BUY
            "c": np.array([0.0, 1.0, 0.0]),      # decisive BUY
            "d": np.array([0.0001, 0.0, 0.0]),   # near-tied HOLD (dissenter, ignored)
            "e": np.array([0.0, 0.0, 0.0001]),   # near-tied SELL (dissenter, ignored)
        }
        result = engine.combine_votes(actions, q_rows)
        assert result["action"] == BUY
        assert result["weak_signal"] is False
        assert result["confidence_label"] == "Medium"  # 3/5 = 60%, not downgraded further

    def test_low_stays_low_when_downgraded(self):
        actions = {"a": HOLD, "b": HOLD, "c": BUY, "d": BUY, "e": SELL}
        q_rows = {k: np.array([0.0001, 0.0, 0.0]) for k in actions}
        result = engine.combine_votes(actions, q_rows)
        assert result["confidence_label"] == "Low"

    def test_action_margin_helper(self):
        assert engine.action_margin([0.0, 1.0, 0.5]) == pytest.approx(0.5)
        assert engine.action_margin([0.2, 0.2, 0.2]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Position lookup
# ---------------------------------------------------------------------------

class TestCurrentPosition:

    def test_no_trades_file_defaults_to_no_position(self, tmp_path, monkeypatch):
        monkeypatch.setattr(engine, "TRADES_DIR", str(tmp_path))
        assert engine.get_current_position("NOPE", "Q-Learning") == NO_POSITION

    def test_reads_last_trade_row_position(self, tmp_path, monkeypatch):
        monkeypatch.setattr(engine, "TRADES_DIR", str(tmp_path))
        trades = pd.DataFrame({
            "Date": pd.date_range("2025-01-01", periods=3),
            "Action": ["BUY", "HOLD", "SELL"],
            "Position": [HOLDING, HOLDING, NO_POSITION],
        })
        trades.to_csv(tmp_path / "TEST_q_learning_trades.csv", index=False)
        assert engine.get_current_position("TEST", "Q-Learning") == NO_POSITION

    def test_empty_trades_file_defaults_to_no_position(self, tmp_path, monkeypatch):
        monkeypatch.setattr(engine, "TRADES_DIR", str(tmp_path))
        pd.DataFrame(columns=["Date", "Action", "Position"]).to_csv(
            tmp_path / "TEST_sarsa_trades.csv", index=False
        )
        assert engine.get_current_position("TEST", "SARSA") == NO_POSITION


# ---------------------------------------------------------------------------
# Simple labels
# ---------------------------------------------------------------------------

class TestSimpleLabels:

    def test_classify_price_trend(self):
        assert engine.classify_price_trend("BULLISH") == "UP"
        assert engine.classify_price_trend("BEARISH") == "DOWN"
        assert engine.classify_price_trend("NEUTRAL") == "SIDEWAYS"

    def test_classify_momentum(self):
        assert engine.classify_momentum(60) == "POSITIVE"
        assert engine.classify_momentum(40) == "NEGATIVE"
        assert engine.classify_momentum(50) == "NEUTRAL"

    def test_classify_volatility_simple(self):
        assert engine.classify_volatility_simple("HIGH_VOLATILITY") == "HIGH"
        assert engine.classify_volatility_simple("LOW_VOLATILITY") == "LOW"

    def test_rsi_text(self):
        assert engine.rsi_text(20) == "Potentially oversold"
        assert engine.rsi_text(80) == "Potentially overbought"
        assert engine.rsi_text(50) == "Neutral range"

    def test_build_reasons_count_and_mentions_action(self):
        decision = {
            "trend": "BULLISH", "ma5": 105.0, "ma20": 100.0, "rsi_value": 60.0,
            "action": BUY, "votes": {HOLD: 1, BUY: 3, SELL: 1},
        }
        reasons = engine.build_reasons(decision)
        assert 3 <= len(reasons) <= 5
        assert any("BUY" in text for _, _, text in reasons)

    def test_build_reasons_includes_macd_when_present(self):
        decision = {
            "trend": "BULLISH", "ma5": 105.0, "ma20": 100.0, "rsi_value": 60.0,
            "action": BUY, "votes": {HOLD: 1, BUY: 3, SELL: 1},
            "macd_condition": "POSITIVE_MOMENTUM",
        }
        reasons = engine.build_reasons(decision)
        assert any("MACD" in title for _, title, _ in reasons)

    def test_build_reasons_omits_macd_when_absent(self):
        """A historical-fallback decision has no macd_condition key at all -
        must not fabricate a MACD reason or raise a KeyError."""
        decision = {
            "trend": "BULLISH", "ma5": 105.0, "ma20": 100.0, "rsi_value": 60.0,
            "action": BUY, "votes": {HOLD: 1, BUY: 3, SELL: 1},
        }
        reasons = engine.build_reasons(decision)
        assert not any("MACD" in title for _, title, _ in reasons)


# ---------------------------------------------------------------------------
# End-to-end get_decision() against a small fake models/features setup
# ---------------------------------------------------------------------------

class TestGetDecisionEndToEnd:

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
        dates = pd.date_range("2025-01-01", periods=n, freq="D")
        df = pd.DataFrame({
            "Date": dates, "Close": np.linspace(100, 110, n),
            "MA5": np.linspace(101, 109, n), "MA20": np.linspace(99, 108, n),
            "RSI": np.linspace(40, 65, n),
            "Trend": ["BULLISH"] * n,
            "RSI_Condition": ["RSI_NORMAL"] * n,
            "Volatility_Condition": ["LOW_VOLATILITY"] * n,
        })
        df.to_csv(features_dir / "TEST_features.csv", index=False)

        encoder = StateEncoder()
        for algo, stub, favored_action in [
            ("Q-Learning", "q_learning", BUY),
            ("SARSA", "sarsa", BUY),
            ("Monte Carlo", "monte_carlo", HOLD),
        ]:
            q_table = np.zeros((encoder.num_states, 3))
            q_table[:, favored_action] = 1.0
            symbol_dir = models_dir / "TEST"
            symbol_dir.mkdir(exist_ok=True)
            with open(symbol_dir / f"{stub}.pkl", "wb") as f:
                pickle.dump({"q_table": q_table}, f)

        return features_dir, models_dir, trades_dir

    def test_raises_when_no_models_trained(self, tmp_path, monkeypatch):
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        n = 5
        pd.DataFrame({
            "Date": pd.date_range("2025-01-01", periods=n),
            "Close": [100] * n, "MA5": [100] * n, "MA20": [100] * n, "RSI": [50] * n,
            "Trend": ["NEUTRAL"] * n, "RSI_Condition": ["RSI_NORMAL"] * n,
            "Volatility_Condition": ["LOW_VOLATILITY"] * n,
        }).to_csv(features_dir / "NOPE_features.csv", index=False)
        monkeypatch.setattr(engine, "FEATURES_DIR", str(features_dir))
        monkeypatch.setattr(engine, "MODELS_DIR", str(tmp_path / "empty_models"))
        monkeypatch.setattr(engine, "TRADES_DIR", str(tmp_path / "empty_trades"))
        with pytest.raises(engine.DecisionError):
            engine.get_decision("NOPE")

    def test_partial_models_still_produce_a_decision(self, fake_project):
        result = engine.get_decision("TEST")
        assert result["action"] in (HOLD, BUY, SELL)
        assert len(result["skipped_algorithms"]) == 2  # Value Iteration, Policy Iteration missing
        assert sum(result["votes"].values()) == 3
        assert result["action"] == BUY  # 2 of 3 voted BUY
        assert result["confidence_pct"] == pytest.approx((2 / 3) * 100)
        assert result["company"] == "TEST"
        assert result["latest_price"] == pytest.approx(110.0)

    def test_result_includes_display_fields(self, fake_project):
        result = engine.get_decision("TEST")
        for key in ("trend", "rsi_value", "ma5", "ma20", "latest_date", "previous_close"):
            assert key in result

    def test_falls_back_to_historical_when_live_unavailable(self, fake_project):
        """'TEST' has no Angel One symbol mapping, so the live path must
        fail cleanly and get_decision must fall back to the static
        feature file - never crash, never silently show no data."""
        result = engine.get_decision("TEST")
        assert result["data_source"] == "historical"
        assert result["live_unavailable_reason"] is not None

    def test_uses_live_row_when_available(self, fake_project, monkeypatch):
        """When live_features succeeds, get_decision must use its price/
        indicators/PreviousClose instead of the static feature file, and
        report data_source == 'live'."""
        from src.decision import live_features

        live_row = pd.Series({
            "Date": pd.Timestamp("2026-09-07"), "Close": 250.0,
            "MA5": 248.0, "MA20": 240.0, "RSI": 65.0, "Volatility": 0.015,
            "Trend": "BULLISH", "RSI_Condition": "RSI_NORMAL",
            "Volatility_Condition": "LOW_VOLATILITY", "PreviousClose": 245.0,
        })
        monkeypatch.setattr(live_features, "get_live_feature_row", lambda symbol: (live_row, None))

        result = engine.get_decision("TEST")
        assert result["data_source"] == "live"
        assert result["live_unavailable_reason"] is None
        assert result["latest_price"] == pytest.approx(250.0)
        assert result["previous_close"] == pytest.approx(245.0)
        assert result["latest_date"] == pd.Timestamp("2026-09-07")

    def test_live_row_exposes_macd_and_volume_fields(self, fake_project, monkeypatch):
        """MACD/volume/high/low are live-only reasoning fields (not part of
        the trained models' state) - they must surface on the decision
        dict when the live row provides them."""
        from src.decision import live_features

        live_row = pd.Series({
            "Date": pd.Timestamp("2026-09-07"), "Close": 250.0,
            "MA5": 248.0, "MA20": 240.0, "RSI": 65.0, "Volatility": 0.015,
            "Trend": "BULLISH", "RSI_Condition": "RSI_NORMAL",
            "Volatility_Condition": "LOW_VOLATILITY", "PreviousClose": 245.0,
            "MACD": 1.5, "MACD_Signal": 1.1, "MACD_Condition": "POSITIVE_MOMENTUM",
            "High": 252.0, "Low": 246.0, "Volume": 500000.0,
            "PreviousVolume": 400000.0, "VolumeChangePct": 25.0,
        })
        monkeypatch.setattr(live_features, "get_live_feature_row", lambda symbol: (live_row, None))

        result = engine.get_decision("TEST")
        assert result["macd"] == pytest.approx(1.5)
        assert result["macd_signal"] == pytest.approx(1.1)
        assert result["macd_condition"] == "POSITIVE_MOMENTUM"
        assert result["today_high"] == pytest.approx(252.0)
        assert result["today_low"] == pytest.approx(246.0)
        assert result["volume"] == pytest.approx(500000.0)
        assert result["previous_volume"] == pytest.approx(400000.0)
        assert result["volume_change_pct"] == pytest.approx(25.0)

    def test_historical_decision_has_no_macd(self, fake_project):
        """The static historical feature files (see build_features.py's
        FEATURE_COLUMNS) never contain MACD - a historical-fallback
        decision must report it as None, never a fabricated value."""
        result = engine.get_decision("TEST", prefer_live=False)
        assert result["data_source"] == "historical"
        assert result["macd"] is None
        assert result["macd_signal"] is None
        assert result["macd_condition"] is None

    def test_prefer_live_false_always_uses_historical(self, fake_project, monkeypatch):
        from src.decision import live_features

        live_row = pd.Series({
            "Date": pd.Timestamp("2026-09-07"), "Close": 250.0,
            "MA5": 248.0, "MA20": 240.0, "RSI": 65.0, "Volatility": 0.015,
            "Trend": "BULLISH", "RSI_Condition": "RSI_NORMAL",
            "Volatility_Condition": "LOW_VOLATILITY", "PreviousClose": 245.0,
        })
        monkeypatch.setattr(live_features, "get_live_feature_row", lambda symbol: (live_row, None))

        result = engine.get_decision("TEST", prefer_live=False)
        assert result["data_source"] == "historical"
        assert result["latest_price"] == pytest.approx(110.0)  # from the static fixture file


class TestBacktestSummary:

    def test_returns_none_when_no_report(self, tmp_path, monkeypatch):
        monkeypatch.setattr(engine, "REPORTS_DIR", str(tmp_path))
        assert engine.backtest_summary("TEST") is None

    def test_averages_across_rl_algorithms_excluding_buy_and_hold(self, tmp_path, monkeypatch):
        monkeypatch.setattr(engine, "REPORTS_DIR", str(tmp_path))
        df = pd.DataFrame([
            {"Company": "TEST", "Algorithm": "Q-Learning", "Initial Capital": 100000,
             "Final Portfolio Value": 110000, "Total Return %": 10.0, "Maximum Drawdown %": -5.0},
            {"Company": "TEST", "Algorithm": "SARSA", "Initial Capital": 100000,
             "Final Portfolio Value": 120000, "Total Return %": 20.0, "Maximum Drawdown %": -10.0},
            {"Company": "TEST", "Algorithm": "Buy & Hold", "Initial Capital": 100000,
             "Final Portfolio Value": 200000, "Total Return %": 100.0, "Maximum Drawdown %": -50.0},
        ])
        df.to_csv(tmp_path / "final_results.csv", index=False)
        summary = engine.backtest_summary("TEST")
        assert summary["return_pct"] == pytest.approx(15.0)  # avg of 10, 20 - Buy & Hold excluded
        assert summary["num_algorithms"] == 2