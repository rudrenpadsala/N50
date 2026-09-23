"""
src/backtesting/model_loader.py

Phase 1 - Sprint 5 - Step 1: A single, uniform way to load a TRAINED
model/policy for any of the 7 algorithms, for backtesting only.

Every algorithm's saved pickle (Sprint 3's `BaseAgent.save_q_table` /
`MonteCarloAgent.save_q_table`, Sprint 4's `ValueIterationAgent.save_q_table`
/ `PolicyIterationAgent.save_q_table`, and the `ExpectedSarsaAgent` /
`DoubleQLearningAgent` addition - see src/rl/base_agent.py,
src/rl/monte_carlo.py, src/rl/value_iteration.py, src/rl/policy_iteration.py,
src/rl/expected_sarsa.py, src/rl/double_q_learning.py)
stores a `"q_table"` array of shape (num_states, num_actions). That
shared shape means backtesting does NOT need 7 separate code paths: for
every algorithm, the greedy backtesting policy is simply

    policy[s] = argmax_a q_table[s, a]

This module loads the pickle and derives that greedy policy - it never
mutates the loaded q_table (backtesting is evaluation-only, Sprint 5
Section 2/35: no retraining, no Q-table updates, no policy updates).
"""

import os
import pickle

import numpy as np

ALGORITHM_FILE_NAMES = {
    "Q-Learning": "q_learning",
    "SARSA": "sarsa",
    "Monte Carlo": "monte_carlo",
    "Value Iteration": "value_iteration",
    "Policy Iteration": "policy_iteration",
    # Ensemble-accuracy addition (see src/rl/expected_sarsa.py,
    # src/rl/double_q_learning.py): two more low-variance / low-bias
    # tabular voters for src/decision/engine.py's majority vote. Both
    # save the same (num_states, num_actions) "q_table" shape as every
    # other algorithm above, so no other code in this module changes.
    "Expected SARSA": "expected_sarsa",
    "Double Q-Learning": "double_q_learning",
}

ALGORITHMS = list(ALGORITHM_FILE_NAMES.keys())


def file_stub(algorithm: str) -> str:
    """
    Canonical on-disk filename fragment for an algorithm name, shared by
    every writer/reader of per-company backtest files (trade logs,
    portfolio histories, plots): scripts/run_backtest.py and app.py both
    import THIS function rather than each re-deriving their own naming,
    so the dashboard can never disagree with what the backtest script
    actually wrote to disk.
    """
    return algorithm.lower().replace(" & ", "_and_").replace(" ", "_").replace("-", "_")


class ModelLoadError(Exception):
    """Raised when a trained model/policy cannot be found or loaded for backtesting."""


def model_path(models_dir: str, symbol: str, algorithm: str) -> str:
    if algorithm not in ALGORITHM_FILE_NAMES:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")
    file_stub = ALGORITHM_FILE_NAMES[algorithm]
    return os.path.join(models_dir, symbol, f"{file_stub}.pkl")


def load_q_table(models_dir: str, symbol: str, algorithm: str,
                  num_states: int, num_actions: int) -> np.ndarray:
    """
    Load `<models_dir>/<symbol>/<algorithm>.pkl` and return its raw
    (num_states, num_actions) Q-table, read-only. Shared by
    `load_greedy_policy` (below) and by `src/decision/engine.py`'s
    ensemble/voting layer, which additionally needs Q-VALUES (not just
    the argmax action) to break ties between algorithms.

    Raises ModelLoadError (never a bare FileNotFoundError/KeyError) if
    the model is missing, unreadable, or the wrong shape.
    """
    path = model_path(models_dir, symbol, algorithm)
    if not os.path.isfile(path):
        raise ModelLoadError(f"No trained model found for {symbol}/{algorithm} at {path}")

    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(f"Could not load model for {symbol}/{algorithm}: {exc}") from exc

    q_table = payload.get("q_table")
    if q_table is None:
        raise ModelLoadError(f"Model for {symbol}/{algorithm} has no 'q_table' entry")
    if q_table.shape != (num_states, num_actions):
        raise ModelLoadError(
            f"Model for {symbol}/{algorithm} has q_table shape {q_table.shape}, "
            f"expected {(num_states, num_actions)}"
        )
    q_table = np.array(q_table, copy=True)
    q_table.setflags(write=False)
    return q_table


def load_greedy_policy(models_dir: str, symbol: str, algorithm: str,
                        num_states: int, num_actions: int) -> np.ndarray:
    """
    Load `<models_dir>/<symbol>/<algorithm>.pkl` and return its GREEDY
    policy - a read-only (num_states,) int array of argmax_a q_table[s, a].

    Raises ModelLoadError (never a bare FileNotFoundError/KeyError) if
    the model is missing, unreadable, or the wrong shape - so callers
    (scripts/run_backtest.py) can log it to reports/backtest_errors.csv
    and continue with the remaining companies/algorithms (Sprint 5
    Section 33).
    """
    q_table = load_q_table(models_dir, symbol, algorithm, num_states, num_actions)
    policy = np.argmax(q_table, axis=1).astype(int)
    policy.setflags(write=False)  # backtesting must never mutate a trained policy
    return policy
