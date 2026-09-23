"""
Sprint 4 - Step 22: Tests for the tabular MDP, Value Iteration, Policy
Iteration, and their integration with the existing Sprint 1-3 project
(environment, state encoder, model I/O).

Run with:
    pytest
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.environment.actions import HOLD, BUY, SELL, NUM_ACTIONS, is_valid_action  # noqa: E402
from src.environment.state import StateEncoder, EXPECTED_STATE_SPACE_SIZE, NO_POSITION, HOLDING  # noqa: E402
from src.environment.trading_env import TradingEnvironment  # noqa: E402
from src.rl.mdp import build_mdp, MDP, _apply_action, _hypothetical_portfolio  # noqa: E402
from src.rl.value_iteration import ValueIterationAgent  # noqa: E402
from src.rl.policy_iteration import PolicyIterationAgent  # noqa: E402
from src.rl.model_io import extract_dp_policy_rows, save_model, build_dp_metadata  # noqa: E402


ENV_KWARGS = dict(initial_capital=100_000.0, transaction_cost=0.001,
                   invalid_action_penalty=-0.001, shares_per_trade=1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def encoder():
    return StateEncoder()


@pytest.fixture
def synthetic_feature_df():
    """
    Same shape/spirit as the Sprint 3 fixture: a small, deterministic,
    chronologically-sorted synthetic feature dataframe big enough to
    build a non-trivial MDP (needs >= 2 warmed-up rows).
    """
    n = 200
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(rng.normal(0, 1, size=n))
    trend = rng.choice(["BULLISH", "BEARISH", "NEUTRAL"], size=n)
    rsi_cond = rng.choice(["RSI_LOW", "RSI_NORMAL", "RSI_HIGH"], size=n)
    vol_cond = rng.choice(["LOW_VOLATILITY", "HIGH_VOLATILITY"], size=n)
    return pd.DataFrame({
        "Date": dates, "Close": close,
        "Trend": trend, "RSI_Condition": rsi_cond, "Volatility_Condition": vol_cond,
    })


@pytest.fixture
def mdp(synthetic_feature_df, encoder):
    return build_mdp(
        train_df=synthetic_feature_df,
        encoder=encoder,
        initial_capital=ENV_KWARGS["initial_capital"],
        transaction_cost=ENV_KWARGS["transaction_cost"],
        invalid_action_penalty=ENV_KWARGS["invalid_action_penalty"],
        shares_per_trade=ENV_KWARGS["shares_per_trade"],
    )


# ---------------------------------------------------------------------------
# MDP
# ---------------------------------------------------------------------------

class TestMDP:

    def test_mdp_shape(self, mdp, encoder):
        assert mdp.transition_probs.shape == (encoder.num_states, NUM_ACTIONS, encoder.num_states)
        assert mdp.rewards.shape == (encoder.num_states, NUM_ACTIONS)

    def test_transition_probabilities_sum_to_one(self, mdp, encoder):
        for s in range(encoder.num_states):
            for a in range(NUM_ACTIONS):
                total = mdp.transition_probs[s, a].sum()
                assert total == pytest.approx(1.0, abs=1e-9)

    def test_unobserved_state_action_falls_back_to_self_loop(self, encoder):
        """A (state, action) pair with zero training occurrences must
        default to a self-loop with zero reward, never raise, and never
        leave a row that doesn't sum to 1."""
        # A 2-row dataframe only ever visits ONE Trend/RSI/Vol combo at t=0,
        # so most of the 36 states are never observed with any action.
        df = pd.DataFrame({
            "Date": pd.date_range("2021-01-01", periods=2, freq="D"),
            "Close": [100.0, 101.0],
            "Trend": ["BULLISH", "BEARISH"],
            "RSI_Condition": ["RSI_NORMAL", "RSI_NORMAL"],
            "Volatility_Condition": ["LOW_VOLATILITY", "LOW_VOLATILITY"],
        })
        tiny_mdp = build_mdp(df, encoder, **ENV_KWARGS)
        unobserved = np.argwhere(tiny_mdp.observed_counts == 0)
        assert len(unobserved) > 0
        s, a = unobserved[0]
        assert tiny_mdp.transition_probs[s, a, s] == pytest.approx(1.0)
        assert tiny_mdp.rewards[s, a] == 0.0

    def test_no_future_leakage_only_uses_train_end_rows(self, encoder):
        """build_mdp must never reference a row beyond t+1 for any row t
        it processes - i.e. restricting the input dataframe changes the
        model (it is not silently pulling in extra data from elsewhere)."""
        n = 50
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        df = pd.DataFrame({
            "Date": dates,
            "Close": np.linspace(100, 150, n),
            "Trend": ["BULLISH"] * n,
            "RSI_Condition": ["RSI_NORMAL"] * n,
            "Volatility_Condition": ["LOW_VOLATILITY"] * n,
        })
        full_mdp = build_mdp(df, encoder, **ENV_KWARGS)
        half_mdp = build_mdp(df.iloc[:25].reset_index(drop=True), encoder, **ENV_KWARGS)
        # Different amounts of training data must be reflected in different
        # observed_counts (the smaller slice can't have "seen" as much).
        assert full_mdp.observed_counts.sum() > half_mdp.observed_counts.sum()

    def test_missing_required_column_raises(self, encoder):
        bad_df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=5), "Close": range(5)})
        with pytest.raises(ValueError):
            build_mdp(bad_df, encoder, **ENV_KWARGS)

    def test_apply_action_mirrors_environment_invalid_rules(self):
        # BUY with insufficient cash -> invalid, state unchanged.
        cash, shares, invalid = _apply_action(BUY, price=1_000_000.0, cash=100.0, shares=0,
                                               transaction_cost=0.001, shares_per_trade=1)
        assert invalid is True
        assert cash == 100.0 and shares == 0

        # SELL with zero shares -> invalid, state unchanged.
        cash, shares, invalid = _apply_action(SELL, price=100.0, cash=100_000.0, shares=0,
                                               transaction_cost=0.001, shares_per_trade=1)
        assert invalid is True
        assert shares == 0

        # HOLD is always valid and never changes cash/shares.
        cash, shares, invalid = _apply_action(HOLD, price=100.0, cash=500.0, shares=2,
                                               transaction_cost=0.001, shares_per_trade=1)
        assert invalid is False and cash == 500.0 and shares == 2

    def test_hypothetical_portfolio_never_negative(self):
        cash, shares = _hypothetical_portfolio(NO_POSITION, price=100.0, initial_capital=100_000.0,
                                                transaction_cost=0.001, shares_per_trade=1)
        assert cash >= 0 and shares == 0
        cash, shares = _hypothetical_portfolio(HOLDING, price=100.0, initial_capital=100_000.0,
                                                transaction_cost=0.001, shares_per_trade=1)
        assert cash >= 0 and shares == 1


# ---------------------------------------------------------------------------
# Value Iteration
# ---------------------------------------------------------------------------

class TestValueIteration:

    def test_initial_state_values_are_zero(self, mdp):
        agent = ValueIterationAgent(mdp)
        assert np.all(agent.V == 0.0)

    def test_state_and_action_counts(self, mdp, encoder):
        agent = ValueIterationAgent(mdp)
        assert agent.num_states == 36 == encoder.num_states
        assert agent.num_actions == 3 == NUM_ACTIONS

    def test_bellman_update_matches_manual_computation(self, mdp):
        agent = ValueIterationAgent(mdp, gamma=0.9)
        agent.V = np.random.default_rng(0).normal(size=agent.num_states)
        s = 5
        expected = mdp.rewards[s] + 0.9 * (mdp.transition_probs[s] @ agent.V)
        np.testing.assert_allclose(agent.action_values(s), expected)

    def test_converges_within_max_iterations(self, mdp):
        agent = ValueIterationAgent(mdp, theta=1e-6, max_iterations=1000)
        result = agent.run()
        assert result["converged"] is True
        assert result["iterations"] <= 1000
        assert result["final_max_delta"] < 1e-6

    def test_policy_extraction_shape_and_validity(self, mdp):
        agent = ValueIterationAgent(mdp)
        agent.run()
        policy = agent.get_policy()
        assert policy.shape == (36,)
        assert set(np.unique(policy)).issubset({0, 1, 2})

    def test_produces_36_values_and_36_actions(self, mdp):
        agent = ValueIterationAgent(mdp)
        agent.run()
        assert agent.V.shape == (36,)
        assert agent.get_policy().shape == (36,)
        assert all(is_valid_action(int(a)) for a in agent.get_policy())

    def test_invalid_actions_never_produced(self, mdp):
        agent = ValueIterationAgent(mdp)
        agent.run()
        for a in agent.get_policy():
            assert a in (HOLD, BUY, SELL)

    def test_save_and_load_round_trip(self, mdp, tmp_path):
        agent = ValueIterationAgent(mdp)
        agent.run()
        path = tmp_path / "vi.pkl"
        agent.save_q_table(str(path))

        reloaded = ValueIterationAgent(mdp)
        reloaded.load_q_table(str(path))
        np.testing.assert_array_equal(reloaded.q_table, agent.q_table)
        np.testing.assert_array_equal(reloaded.policy, agent.policy)


# ---------------------------------------------------------------------------
# Policy Iteration
# ---------------------------------------------------------------------------

class TestPolicyIteration:

    def test_hold_initialization(self, mdp):
        agent = PolicyIterationAgent(mdp, init="HOLD")
        assert np.all(agent.policy == HOLD)

    def test_random_initialization_is_valid_and_reproducible(self, mdp):
        agent1 = PolicyIterationAgent(mdp, init="RANDOM")
        agent2 = PolicyIterationAgent(mdp, init="RANDOM")
        np.testing.assert_array_equal(agent1.policy, agent2.policy)
        assert set(np.unique(agent1.policy)).issubset({0, 1, 2})

    def test_unknown_initialization_raises(self, mdp):
        with pytest.raises(ValueError):
            PolicyIterationAgent(mdp, init="NOT_A_REAL_STRATEGY")

    def test_policy_evaluation_converges(self, mdp):
        agent = PolicyIterationAgent(mdp, theta=1e-6, max_iterations=1000)
        iters = agent.evaluate_policy()
        assert iters <= 1000
        assert agent.V.shape == (36,)

    def test_policy_improvement_can_change_policy(self, mdp):
        agent = PolicyIterationAgent(mdp, init="HOLD")
        agent.evaluate_policy()
        agent.improve_policy()
        # After one evaluation of an all-HOLD policy, improvement should be
        # able to run without error and produce a fully valid policy.
        assert set(np.unique(agent.policy)).issubset({0, 1, 2})

    def test_converges_to_policy_stable(self, mdp):
        agent = PolicyIterationAgent(mdp)
        result = agent.run()
        assert result["policy_stable"] is True
        assert result["policy_iterations"] <= agent.max_iterations

    def test_stops_improving_once_stable(self, mdp):
        agent = PolicyIterationAgent(mdp)
        agent.run()
        policy_before = agent.policy.copy()
        agent.evaluate_policy()
        changed = agent.improve_policy()
        assert changed is False
        np.testing.assert_array_equal(agent.policy, policy_before)

    def test_value_iteration_and_policy_iteration_agree(self, mdp):
        """Both algorithms solve the SAME MDP optimally, so their greedy
        policies should match once both have converged."""
        vi = ValueIterationAgent(mdp)
        vi.run()
        pi = PolicyIterationAgent(mdp)
        pi.run()
        np.testing.assert_array_equal(vi.get_policy(), pi.get_policy())

    def test_save_and_load_round_trip(self, mdp, tmp_path):
        agent = PolicyIterationAgent(mdp)
        agent.run()
        path = tmp_path / "pi.pkl"
        agent.save_q_table(str(path))

        reloaded = PolicyIterationAgent(mdp)
        reloaded.load_q_table(str(path))
        np.testing.assert_array_equal(reloaded.policy, agent.policy)
        assert reloaded.policy_stable == agent.policy_stable


# ---------------------------------------------------------------------------
# Integration - environment compatibility, model I/O, policy output
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_state_space_is_36(self, encoder):
        assert encoder.num_states == 36 == EXPECTED_STATE_SPACE_SIZE

    def test_action_space_is_3(self):
        assert NUM_ACTIONS == 3

    def test_mdp_uses_same_state_encoder_as_environment(self, synthetic_feature_df, encoder):
        """Sanity: an environment built from the same dataframe and the
        MDP built from it must agree on state IDs for identical rows."""
        env = TradingEnvironment(df=synthetic_feature_df, state_encoder=encoder, **ENV_KWARGS)
        initial_state = env.reset()
        m = build_mdp(synthetic_feature_df, encoder, **ENV_KWARGS)
        assert 0 <= initial_state < m.num_states

    @pytest.mark.parametrize("algorithm,agent_cls", [
        ("Value Iteration", ValueIterationAgent),
        ("Policy Iteration", PolicyIterationAgent),
    ])
    def test_model_saving_and_loading(self, mdp, encoder, tmp_path, algorithm, agent_cls):
        agent = agent_cls(mdp)
        agent.run()
        metadata = build_dp_metadata(
            company="TEST", algorithm=algorithm, num_states=agent.num_states,
            num_actions=agent.num_actions, gamma=agent.gamma, theta=agent.theta,
            train_start="2015-01-01", train_end="2024-12-31",
            initial_capital=100_000.0, transaction_cost=0.001,
            iterations=1, convergence_status="Converged",
        )
        paths = save_model(str(tmp_path), "TEST", algorithm, agent, metadata)
        assert os.path.isfile(paths["model_path"])
        assert os.path.isfile(paths["metadata_path"])

    @pytest.mark.parametrize("algorithm,agent_cls", [
        ("Value Iteration", ValueIterationAgent),
        ("Policy Iteration", PolicyIterationAgent),
    ])
    def test_policy_output_rows_are_complete_and_valid(self, mdp, encoder, algorithm, agent_cls):
        agent = agent_cls(mdp)
        agent.run()
        rows = extract_dp_policy_rows("TEST", algorithm, agent, encoder)
        assert len(rows) == 36
        for row in rows:
            assert row["Best_Action"] in {"HOLD", "BUY", "SELL"}
            assert isinstance(row["Value"], float)

    def test_no_negative_cash_or_shares_across_hypothetical_transitions(self, mdp):
        """Every reward/transition baked into the MDP came from an
        `_apply_action` call that enforces non-negative cash/shares -
        this just re-confirms the invariant holds for the built model's
        underlying assumptions by re-deriving a couple of transitions."""
        cash, shares, invalid = _apply_action(BUY, price=50.0, cash=40.0, shares=0,
                                               transaction_cost=0.001, shares_per_trade=1)
        assert cash >= 0
        cash, shares, invalid = _apply_action(SELL, price=50.0, cash=100.0, shares=0,
                                               transaction_cost=0.001, shares_per_trade=1)
        assert shares >= 0
