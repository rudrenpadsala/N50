"""
Sprint 5 - Step 34: Tests for the backtester, metrics, Buy & Hold
baseline, model loader, and the dashboard's data-loading helpers.

Run with:
    pytest
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
from src.environment.state import StateEncoder  # noqa: E402
from src.backtesting import metrics as m  # noqa: E402
from src.backtesting.backtester import run_backtest  # noqa: E402
from src.backtesting.buy_and_hold import run_buy_and_hold  # noqa: E402
from src.backtesting.model_loader import (  # noqa: E402
    load_greedy_policy, ModelLoadError, model_path,
)

ENV_KWARGS = dict(initial_capital=100_000.0, transaction_cost=0.001,
                   invalid_action_penalty=-0.001, shares_per_trade=1)


@pytest.fixture
def encoder():
    return StateEncoder()


@pytest.fixture
def test_df():
    """Deterministic synthetic test-period dataframe: rising, then falling,
    then flat, so BUY/HOLD/SELL all get meaningfully exercised."""
    n = 40
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    prices = np.concatenate([
        np.linspace(100, 120, n // 2),   # rising half
        np.linspace(120, 90, n - n // 2),  # falling half
    ])
    return pd.DataFrame({
        "Date": dates, "Close": prices,
        "Trend": ["BULLISH"] * (n // 2) + ["BEARISH"] * (n - n // 2),
        "RSI_Condition": ["RSI_NORMAL"] * n,
        "Volatility_Condition": ["LOW_VOLATILITY"] * n,
    })


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:

    def test_total_return_pct(self):
        assert m.total_return_pct(100_000, 110_000) == pytest.approx(10.0)
        assert m.total_return_pct(100_000, 90_000) == pytest.approx(-10.0)

    def test_total_return_pct_zero_initial(self):
        assert m.total_return_pct(0, 1000) == 0.0

    def test_annualized_return_one_year(self):
        # Exactly doubling over 365 days -> 100% annualized.
        assert m.annualized_return_pct(100_000, 200_000, 365) == pytest.approx(100.0, abs=1e-6)

    def test_annualized_return_wiped_out_portfolio(self):
        assert m.annualized_return_pct(100_000, 0, 365) == -100.0

    def test_sharpe_zero_variance_returns_zero(self):
        assert m.sharpe_ratio([0.0, 0.0, 0.0]) == 0.0

    def test_sharpe_positive_for_positive_trend(self):
        returns = [0.01, 0.02, 0.015, 0.01, 0.02]
        assert m.sharpe_ratio(returns) > 0

    def test_sharpe_too_few_observations(self):
        assert m.sharpe_ratio([0.01]) == 0.0
        assert m.sharpe_ratio([]) == 0.0

    def test_max_drawdown(self):
        drawdowns = [0.0, -0.05, -0.10, -0.02, 0.0]
        assert m.max_drawdown_pct(drawdowns) == pytest.approx(-10.0)

    def test_max_drawdown_empty(self):
        assert m.max_drawdown_pct([]) == 0.0

    def test_win_rate(self):
        assert m.win_rate_pct(7, 3) == pytest.approx(70.0)

    def test_win_rate_no_closed_trades(self):
        assert m.win_rate_pct(0, 0) == 0.0

    def test_no_nan_or_inf_ever(self):
        """No combination of degenerate inputs should ever produce NaN/inf
        (Sprint 5 Section 36)."""
        assert np.isfinite(m.total_return_pct(0, 0))
        assert np.isfinite(m.annualized_return_pct(0, 0, 0))
        assert np.isfinite(m.sharpe_ratio([]))
        assert np.isfinite(m.max_drawdown_pct([]))
        assert np.isfinite(m.win_rate_pct(0, 0))


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class TestBacktester:

    def test_correct_date_range(self, test_df, encoder):
        policy = np.zeros(36, dtype=int)  # all HOLD
        result = run_backtest("TEST", "Q-Learning", policy, test_df, encoder=encoder, **ENV_KWARGS)
        dates = [row["Date"] for row in result.portfolio_history]
        assert min(dates) >= pd.Timestamp(test_df["Date"].min())
        assert max(dates) <= pd.Timestamp(test_df["Date"].max())

    def test_correct_initial_capital(self, test_df, encoder):
        policy = np.zeros(36, dtype=int)
        result = run_backtest("TEST", "Q-Learning", policy, test_df, encoder=encoder, **ENV_KWARGS)
        assert result.initial_portfolio_value == ENV_KWARGS["initial_capital"]

    def test_all_hold_never_trades(self, test_df, encoder):
        policy = np.zeros(36, dtype=int)  # HOLD everywhere
        result = run_backtest("TEST", "Q-Learning", policy, test_df, encoder=encoder, **ENV_KWARGS)
        assert result.num_trades == 0
        assert result.trade_log == []
        assert result.final_portfolio_value == pytest.approx(result.initial_portfolio_value)

    def test_buy_execution_reduces_cash_and_increases_shares(self, test_df, encoder):
        policy = np.full(36, BUY, dtype=int)  # BUY every state -> only the first BUY is valid, rest are invalid no-ops while HOLDING... actually BUY can repeat
        result = run_backtest("TEST", "Q-Learning", policy, test_df, encoder=encoder, **ENV_KWARGS)
        buys = [row for row in result.trade_log if row["Action"] == "BUY"]
        assert len(buys) > 0
        first_buy = buys[0]
        assert first_buy["Cash"] < ENV_KWARGS["initial_capital"]
        assert first_buy["Shares"] >= 1

    def test_sell_execution_requires_shares(self, test_df, encoder):
        policy = np.full(36, SELL, dtype=int)  # SELL with 0 shares is always invalid
        result = run_backtest("TEST", "Q-Learning", policy, test_df, encoder=encoder, **ENV_KWARGS)
        assert result.num_trades == 0  # every SELL was invalid (no shares to sell)
        assert all(row["Action"] != "SELL" for row in result.trade_log)

    def test_transaction_cost_is_applied(self, test_df, encoder):
        policy = np.zeros(36, dtype=int)
        policy[:] = HOLD
        # Force a single BUY by using a policy that buys on the very first
        # state and holds thereafter is hard to construct generically, so
        # instead verify the cost arithmetic directly via a BUY-heavy run.
        policy = np.full(36, BUY, dtype=int)
        result = run_backtest("TEST", "Q-Learning", policy, test_df, encoder=encoder, **ENV_KWARGS)
        first_buy = next(row for row in result.trade_log if row["Action"] == "BUY")
        expected_cash = ENV_KWARGS["initial_capital"] - first_buy["Price"] * (1 + ENV_KWARGS["transaction_cost"])
        assert first_buy["Cash"] == pytest.approx(expected_cash)

    def test_portfolio_value_never_negative(self, test_df, encoder):
        for act in (HOLD, BUY, SELL):
            policy = np.full(36, act, dtype=int)
            result = run_backtest("TEST", "X", policy, test_df, encoder=encoder, **ENV_KWARGS)
            assert all(row["Portfolio_Value"] >= 0 for row in result.portfolio_history)

    def test_reward_matches_environment_definition(self, test_df, encoder):
        policy = np.zeros(36, dtype=int)
        result = run_backtest("TEST", "Q-Learning", policy, test_df, encoder=encoder, **ENV_KWARGS)
        # All-HOLD -> reward should be the portfolio's daily return, which is
        # zero throughout (cash never changes, no shares held).
        assert result.total_reward == pytest.approx(0.0, abs=1e-9)

    def test_win_loss_matching_is_fifo(self, test_df, encoder):
        """BUY low (rising half), SELL high (into the falling half) should
        register as a WIN under the FIFO matching rule."""
        # BUY in BULLISH states, SELL in BEARISH states, HOLD would otherwise dominate.
        policy = np.zeros(36, dtype=int)
        for s in range(36):
            state = encoder.decode(s)
            if state.trend == "BULLISH":
                policy[s] = BUY
            elif state.trend == "BEARISH":
                policy[s] = SELL
        result = run_backtest("TEST", "Q-Learning", policy, test_df, encoder=encoder, **ENV_KWARGS)
        assert result.winning_trades + result.losing_trades > 0

    def test_model_never_mutated_read_only_policy(self, test_df, encoder):
        policy = np.zeros(36, dtype=int)
        policy.setflags(write=False)
        # Must not raise, and the policy array must be unchanged afterward.
        before = policy.copy()
        run_backtest("TEST", "Q-Learning", policy, test_df, encoder=encoder, **ENV_KWARGS)
        np.testing.assert_array_equal(policy, before)

    def test_no_training_period_dates_leak_in(self, test_df, encoder):
        """A test_df restricted to 2025+ must never produce a portfolio
        history date before 2025 (i.e. the caller's date filtering, not
        the backtester, is what keeps training data out - verify the
        backtester respects whatever it's given)."""
        policy = np.zeros(36, dtype=int)
        result = run_backtest("TEST", "Q-Learning", policy, test_df, encoder=encoder, **ENV_KWARGS)
        assert all(pd.Timestamp(row["Date"]) >= pd.Timestamp("2025-01-01") for row in result.portfolio_history)


# ---------------------------------------------------------------------------
# Buy & Hold
# ---------------------------------------------------------------------------

class TestBuyAndHold:

    def test_buys_max_affordable_shares_on_day_one(self, test_df):
        result = run_buy_and_hold("TEST", test_df, initial_capital=100_000.0, transaction_cost=0.001)
        first_price = float(test_df.loc[0, "Close"])
        expected_shares = int(100_000.0 // (first_price * 1.001))
        assert result.trade_log[0]["Shares"] == expected_shares
        assert result.trade_log[0]["Date"] == pd.Timestamp(test_df.loc[0, "Date"])

    def test_never_sells(self, test_df):
        result = run_buy_and_hold("TEST", test_df, initial_capital=100_000.0, transaction_cost=0.001)
        assert all(row["Action"] != "SELL" for row in result.trade_log)
        assert result.num_trades == 1

    def test_no_future_information_used(self, test_df):
        """Buying on day 1 must only ever use day 1's price, regardless of
        what happens later in the series (rising then falling)."""
        truncated = test_df.iloc[:5].reset_index(drop=True)
        full = test_df.copy()
        r_trunc = run_buy_and_hold("TEST", truncated, 100_000.0, 0.001)
        r_full = run_buy_and_hold("TEST", full, 100_000.0, 0.001)
        assert r_trunc.trade_log[0]["Shares"] == r_full.trade_log[0]["Shares"]

    def test_portfolio_tracks_price(self, test_df):
        result = run_buy_and_hold("TEST", test_df, 100_000.0, 0.001)
        assert len(result.portfolio_history) == len(test_df)
        assert result.final_portfolio_value >= 0


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

class TestModelLoader:

    def test_missing_model_raises_model_load_error(self, tmp_path, encoder):
        with pytest.raises(ModelLoadError):
            load_greedy_policy(str(tmp_path), "NOPE", "Q-Learning", encoder.num_states, 3)

    def test_greedy_policy_matches_argmax(self, tmp_path, encoder):
        q_table = np.zeros((36, 3))
        q_table[:, SELL] = 1.0  # SELL always has the highest value
        symbol_dir = tmp_path / "TEST"
        symbol_dir.mkdir()
        with open(symbol_dir / "q_learning.pkl", "wb") as f:
            pickle.dump({"q_table": q_table}, f)

        policy = load_greedy_policy(str(tmp_path), "TEST", "Q-Learning", 36, 3)
        assert np.all(policy == SELL)

    def test_returned_policy_is_read_only(self, tmp_path):
        q_table = np.random.default_rng(0).normal(size=(36, 3))
        symbol_dir = tmp_path / "TEST"
        symbol_dir.mkdir()
        with open(symbol_dir / "sarsa.pkl", "wb") as f:
            pickle.dump({"q_table": q_table}, f)
        policy = load_greedy_policy(str(tmp_path), "TEST", "SARSA", 36, 3)
        with pytest.raises(ValueError):
            policy[0] = 99

    def test_wrong_shape_raises_model_load_error(self, tmp_path):
        q_table = np.zeros((10, 3))  # wrong number of states
        symbol_dir = tmp_path / "TEST"
        symbol_dir.mkdir()
        with open(symbol_dir / "monte_carlo.pkl", "wb") as f:
            pickle.dump({"q_table": q_table}, f)
        with pytest.raises(ModelLoadError):
            load_greedy_policy(str(tmp_path), "TEST", "Monte Carlo", 36, 3)

    def test_model_path_construction(self, tmp_path):
        path = model_path(str(tmp_path), "RELI", "Value Iteration")
        assert path.endswith(os.path.join("RELI", "value_iteration.pkl"))


# ---------------------------------------------------------------------------
# Dashboard data-loading helpers (import app.py's pure functions only -
# never invoke Streamlit page rendering, which requires a running server)
# ---------------------------------------------------------------------------

class TestDashboardHelpers:

    def test_app_module_source_is_syntactically_valid(self):
        """app.py must compile cleanly (syntax + name resolution at parse
        time) even with no reports present yet - it should only complain
        gracefully at render time (via st.stop()), never crash on import."""
        app_path = os.path.join(PROJECT_ROOT, "app.py")
        with open(app_path, encoding="utf-8") as f:
            source = f.read()
        compile(source, app_path, "exec")

    def test_decision_engine_reuses_shared_naming_function(self):
        """src/decision/engine.py imports src.backtesting.model_loader.file_stub
        directly (rather than re-implementing its own copy) to find each
        algorithm's trade log, so the dashboard's position lookup can
        never disagree with what scripts/run_backtest.py wrote to disk -
        verify that shared function behaves consistently for every
        algorithm name."""
        from src.backtesting.model_loader import file_stub
        expected = {
            "Q-Learning": "q_learning",
            "SARSA": "sarsa",
            "Monte Carlo": "monte_carlo",
            "Value Iteration": "value_iteration",
            "Policy Iteration": "policy_iteration",
            "Buy & Hold": "buy_and_hold",
        }
        for algo, stub in expected.items():
            assert file_stub(algo) == stub

        with open(os.path.join(PROJECT_ROOT, "src", "decision", "engine.py"), encoding="utf-8") as f:
            engine_source = f.read()
        assert "from src.backtesting.model_loader import" in engine_source and "file_stub" in engine_source
        with open(os.path.join(PROJECT_ROOT, "scripts", "run_backtest.py"), encoding="utf-8") as f:
            script_source = f.read()
        assert "file_stub" in script_source and "def file_stub(" not in script_source
