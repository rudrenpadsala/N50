"""
src/rl/model_io.py

Phase 1 - Sprint 3 - Steps 16-17, 20-21: Model storage, metadata, and
policy extraction, shared by all three algorithms so
scripts/train_model_free.py does not repeat this logic three times.

Layout on disk:

    models/
        <SYMBOL>/
            q_learning.pkl              (Q-table, pickled by the agent)
            q_learning_metadata.json    (hyperparameters / training info)
            sarsa.pkl
            sarsa_metadata.json
            monte_carlo.pkl
            monte_carlo_metadata.json

    reports/policies/
        <SYMBOL>_q_learning_policy.csv
        <SYMBOL>_sarsa_policy.csv
        <SYMBOL>_monte_carlo_policy.csv
"""

import json
import os

import pandas as pd

from src.environment.actions import ACTION_NAMES, HOLD, BUY, SELL, is_valid_action
from src.environment.state import StateEncoder

ALGORITHM_FILE_NAMES = {
    "Q-Learning": "q_learning",
    "SARSA": "sarsa",
    "Monte Carlo": "monte_carlo",
    # Sprint 4 - Value Iteration / Policy Iteration (model-based DP).
    # Added here (not a rewrite of the Sprint 3 entries above) so
    # `save_model` / `save_policy_csv` work unmodified for the two new
    # algorithms too.
    "Value Iteration": "value_iteration",
    "Policy Iteration": "policy_iteration",
    # Ensemble-accuracy addition: two more tabular, model-free algorithms
    # (see src/rl/expected_sarsa.py, src/rl/double_q_learning.py). Both
    # save/load a plain (num_states, num_actions) "q_table" just like the
    # three Sprint 3 algorithms above, so they need nothing beyond an
    # entry here to work with `save_model` / `save_policy_csv` and, on
    # the reading side, src/backtesting/model_loader.py's ensemble.
    "Expected SARSA": "expected_sarsa",
    "Double Q-Learning": "double_q_learning",
}


def company_model_dir(models_dir: str, symbol: str) -> str:
    path = os.path.join(models_dir, symbol)
    os.makedirs(path, exist_ok=True)
    return path


def save_model(models_dir: str, symbol: str, algorithm: str, agent, metadata: dict) -> dict:
    """
    Save an agent's Q-table (.pkl) and its metadata (.json) for one
    company + algorithm. Returns the paths written.
    """
    file_stub = ALGORITHM_FILE_NAMES[algorithm]
    company_dir = company_model_dir(models_dir, symbol)

    model_path = os.path.join(company_dir, f"{file_stub}.pkl")
    agent.save_q_table(model_path)

    metadata_path = os.path.join(company_dir, f"{file_stub}_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return {"model_path": model_path, "metadata_path": metadata_path}


def build_metadata(
    company: str,
    algorithm: str,
    agent,
    episodes: int,
    train_start: str,
    train_end: str,
    initial_capital: float,
    transaction_cost: float,
) -> dict:
    return {
        "company": company,
        "algorithm": algorithm,
        "states": agent.num_states,
        "actions": agent.num_actions,
        "alpha": agent.alpha,
        "gamma": agent.gamma,
        "initial_epsilon": 1.0,
        "minimum_epsilon": agent.epsilon_min,
        "epsilon_decay": agent.epsilon_decay,
        "episodes": episodes,
        "training_start": train_start,
        "training_end": train_end,
        "initial_capital": initial_capital,
        "transaction_cost": transaction_cost,
    }


def extract_policy_rows(symbol: str, algorithm: str, agent, encoder: StateEncoder) -> list:
    """
    Sprint 3 - Step 20/21: extract the greedy policy for every state and
    sanity-check that every extracted action is one of {0,1,2}.
    """
    rows = []
    policy = agent.get_policy()
    for state_id in range(agent.num_states):
        state = encoder.decode(state_id)
        best_action = int(policy[state_id])
        if not is_valid_action(best_action):
            raise ValueError(f"{symbol}/{algorithm}: invalid policy action {best_action} for state {state_id}")

        rows.append({
            "Company": symbol,
            "Algorithm": algorithm,
            "State_ID": state_id,
            "Trend": state.trend,
            "RSI": state.rsi_condition,
            "Volatility": state.volatility_condition,
            "Position": state.position,
            "Q_HOLD": agent.q_table[state_id, HOLD],
            "Q_BUY": agent.q_table[state_id, BUY],
            "Q_SELL": agent.q_table[state_id, SELL],
            "Best_Action": ACTION_NAMES[best_action],
        })
    return rows


def save_policy_csv(reports_dir: str, symbol: str, algorithm: str, rows: list) -> str:
    file_stub = ALGORITHM_FILE_NAMES[algorithm]
    policies_dir = os.path.join(reports_dir, "policies")
    os.makedirs(policies_dir, exist_ok=True)
    path = os.path.join(policies_dir, f"{symbol}_{file_stub}_policy.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Sprint 4 additions - Value Iteration / Policy Iteration only. These are
# NEW functions (the Sprint 3 functions above are untouched) because the
# two DP algorithms have different metadata (gamma/theta/convergence
# instead of alpha/epsilon) and a different policy-table shape (a single
# state Value V(s) instead of three per-action Q columns).
# ---------------------------------------------------------------------------

def build_dp_metadata(
    company: str,
    algorithm: str,
    num_states: int,
    num_actions: int,
    gamma: float,
    theta: float,
    train_start: str,
    train_end: str,
    initial_capital: float,
    transaction_cost: float,
    iterations: int,
    convergence_status: str,
) -> dict:
    """Sprint 4 Section 16 metadata for Value Iteration / Policy Iteration."""
    return {
        "company": company,
        "algorithm": algorithm,
        "states": num_states,
        "actions": num_actions,
        "gamma": gamma,
        "theta": theta,
        "training_start": train_start,
        "training_end": train_end,
        "initial_capital": initial_capital,
        "transaction_cost": transaction_cost,
        "training_iterations": iterations,
        "convergence_status": convergence_status,
    }


def extract_dp_policy_rows(symbol: str, algorithm: str, agent, encoder: StateEncoder) -> list:
    """
    Sprint 4 Section 18: policy table for Value Iteration / Policy
    Iteration - State ID, Trend, RSI Condition, Volatility Condition,
    Position, Value V(s), Optimal Action. `agent` must expose `.V`
    (np.ndarray of shape (num_states,)) and `.get_policy()`, which both
    ValueIterationAgent and PolicyIterationAgent do.
    """
    rows = []
    policy = agent.get_policy()
    for state_id in range(agent.num_states):
        state = encoder.decode(state_id)
        best_action = int(policy[state_id])
        if not is_valid_action(best_action):
            raise ValueError(f"{symbol}/{algorithm}: invalid policy action {best_action} for state {state_id}")

        rows.append({
            "Company": symbol,
            "Algorithm": algorithm,
            "State_ID": state_id,
            "Trend": state.trend,
            "RSI": state.rsi_condition,
            "Volatility": state.volatility_condition,
            "Position": state.position,
            "Value": float(agent.V[state_id]),
            "Best_Action": ACTION_NAMES[best_action],
        })
    return rows
