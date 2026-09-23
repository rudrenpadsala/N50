"""
src/rl/value_iteration.py

Phase 1 - Sprint 4 - Step 2: Classical Value Iteration (model-based
Dynamic Programming), run against the tabular MDP built by
src/rl/mdp.py from TRAINING-period data only.

Bellman optimality update, applied synchronously to every state on each
sweep (i.e. state s's update on sweep k uses ONLY sweep k-1's V, never
a value already updated earlier in the same sweep):

    V(s) <- max_a [ R(s,a) + gamma * sum_s' P(s'|s,a) * V(s') ]

Sweeps repeat until the maximum change in V across all 36 states in one
sweep falls below THETA, or MAX_ITERATIONS sweeps are reached. The
greedy policy is then read off directly from the converged V:

    pi(s) = argmax_a [ R(s,a) + gamma * sum_s' P(s'|s,a) * V(s') ]

This module contains no environment logic and no neural network - it
is plain Python + NumPy over the 36 x 3 tables, matching the Sprint 4
spec (Sections 2, 26, 27).
"""

import numpy as np

from src.environment.actions import NUM_ACTIONS
from src.rl.mdp import MDP

# Configurable defaults (Sprint 4 Section 2) - never hard-coded inline
# elsewhere; every caller can override these via config/config.yaml's
# `dp:` section (see scripts/train_dynamic_programming.py).
GAMMA = 0.95
THETA = 1e-6
MAX_ITERATIONS = 1000


class ValueIterationAgent:
    """Classical Value Iteration over a fixed, pre-built MDP."""

    def __init__(self, mdp: MDP, gamma: float = GAMMA, theta: float = THETA,
                 max_iterations: int = MAX_ITERATIONS):
        self.mdp = mdp
        self.num_states = mdp.num_states
        self.num_actions = mdp.num_actions
        self.gamma = float(gamma)
        self.theta = float(theta)
        self.max_iterations = int(max_iterations)

        # 1. Initialize V(s) - Sprint 4 Section 2, Step 1.
        self.V = np.zeros(self.num_states, dtype=float)
        self.policy = np.zeros(self.num_states, dtype=int)
        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=float)

        self.converged = False
        self.iterations_run = 0
        self.history = []  # [{"iteration": int, "max_delta": float}, ...]

    def action_values(self, state: int) -> np.ndarray:
        """
        2. Calculate action values for `state` - Sprint 4 Section 2, Step 2.
        Returns a (num_actions,) array: R(s,.) + gamma * P(s,.,:) @ V.
        """
        return self.mdp.rewards[state] + self.gamma * (self.mdp.transition_probs[state] @ self.V)

    def run(self) -> dict:
        """
        3. Update V(s). 4. Continue until convergence. 5. Extract the
        optimal policy. - Sprint 4 Section 2, Steps 3-5.
        """
        for it in range(1, self.max_iterations + 1):
            new_V = np.zeros_like(self.V)
            max_delta = 0.0
            for s in range(self.num_states):
                q_values = self.action_values(s)
                new_V[s] = float(np.max(q_values))
                max_delta = max(max_delta, abs(new_V[s] - self.V[s]))
            self.V = new_V
            self.iterations_run = it
            self.history.append({"iteration": it, "max_delta": max_delta})
            if max_delta < self.theta:
                self.converged = True
                break

        self._extract_policy()

        return {
            "converged": self.converged,
            "iterations": self.iterations_run,
            "final_max_delta": self.history[-1]["max_delta"] if self.history else None,
        }

    def _extract_policy(self) -> None:
        """5. Extract the optimal policy - argmax over the converged V."""
        for s in range(self.num_states):
            q_values = self.action_values(s)
            self.q_table[s] = q_values
            self.policy[s] = int(np.argmax(q_values))

    def get_policy(self) -> np.ndarray:
        """Greedy policy: policy[state] = argmax_a Q[state, a]. Shape (num_states,)."""
        return self.policy

    def get_q_table(self) -> np.ndarray:
        """
        Synthesize a (num_states, num_actions) Q-table from the
        converged V, R and P - Q(s,a) = R(s,a) + gamma * sum_s' P(s'|s,a) V(s').
        Lets this agent plug directly into src/rl/model_io.py's existing
        `extract_policy_rows` / `save_policy_csv` (Sprint 3), reusing
        that code rather than duplicating it for Sprint 4.
        """
        return self.q_table

    # ------------------------------------------------------------------
    # Persistence - deliberately mirrors BaseAgent.save_q_table's payload
    # shape (src/rl/base_agent.py) so src/rl/model_io.py's `save_model`
    # (Sprint 3) works unmodified for this Sprint 4 agent too.
    # ------------------------------------------------------------------

    def save_q_table(self, path: str) -> None:
        import pickle
        payload = {
            "q_table": self.q_table,
            "V": self.V,
            "policy": self.policy,
            "num_states": self.num_states,
            "num_actions": self.num_actions,
            "gamma": self.gamma,
            "theta": self.theta,
            "converged": self.converged,
            "iterations": self.iterations_run,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load_q_table(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if payload["q_table"].shape != (self.num_states, self.num_actions):
            raise ValueError(
                f"Q-table shape mismatch: file has {payload['q_table'].shape}, "
                f"expected {(self.num_states, self.num_actions)}"
            )
        self.q_table = payload["q_table"]
        self.V = payload.get("V", self.V)
        self.policy = payload.get("policy", self.policy)
        self.converged = payload.get("converged", self.converged)
        self.iterations_run = payload.get("iterations", self.iterations_run)
