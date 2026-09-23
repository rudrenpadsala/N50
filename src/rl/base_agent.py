"""
src/rl/base_agent.py

Phase 1 - Sprint 3 - Step 11: Common base class for the three tabular,
model-free RL algorithms (Q-Learning, SARSA, Monte Carlo Control).

This class holds only what is genuinely shared:
    - state/action counts and the (num_states, num_actions) Q-table
    - hyperparameters (alpha, gamma, epsilon schedule)
    - epsilon-greedy action selection with unbiased tie-breaking
    - epsilon decay
    - greedy policy extraction
    - Q-table save/load (pickle)

Each algorithm keeps its OWN update rule / training loop in its own file
(q_learning.py, sarsa.py, monte_carlo.py) so the three stay clearly
separate and easy to read for a college project - this base class does
NOT implement train_episode() itself.
"""

import pickle

import numpy as np

from src.environment.actions import NUM_ACTIONS
from src.environment.state import EXPECTED_STATE_SPACE_SIZE


class BaseAgent:
    def __init__(
        self,
        num_states: int = EXPECTED_STATE_SPACE_SIZE,
        num_actions: int = NUM_ACTIONS,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: int = None,
    ):
        self.num_states = num_states
        self.num_actions = num_actions

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        # (36, 3) tabular Q-table, optimistic-neutral zero initialization.
        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=float)

        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Epsilon-greedy action selection (shared by all 3 algorithms)
    # ------------------------------------------------------------------

    def choose_action(self, state: int) -> int:
        """
        With probability epsilon choose a uniformly random valid action,
        otherwise choose greedily from the Q-table. Ties among equally
        good actions are broken uniformly at random (never biased
        towards the lowest action index).
        """
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.num_actions))
        return self._greedy_action(state)

    def _greedy_action(self, state: int) -> int:
        q_values = self.q_table[state]
        max_q = q_values.max()
        best_actions = np.flatnonzero(np.isclose(q_values, max_q))
        return int(self._rng.choice(best_actions))

    def decay_epsilon(self) -> None:
        """Multiplicative epsilon decay, applied once per completed episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Policy extraction
    # ------------------------------------------------------------------

    def get_policy(self) -> np.ndarray:
        """Greedy policy: policy[state] = argmax_a Q[state, a]. Shape (num_states,)."""
        return np.argmax(self.q_table, axis=1)

    # ------------------------------------------------------------------
    # Persistence - Q-table only. Metadata is handled separately by
    # src/rl/model_io.py (kept apart so this class does not need to know
    # about company names, algorithm names, or training config).
    # ------------------------------------------------------------------

    def save_q_table(self, path: str) -> None:
        payload = {
            "q_table": self.q_table,
            "epsilon": self.epsilon,
            "num_states": self.num_states,
            "num_actions": self.num_actions,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load_q_table(self, path: str) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if payload["q_table"].shape != (self.num_states, self.num_actions):
            raise ValueError(
                f"Q-table shape mismatch: file has {payload['q_table'].shape}, "
                f"expected {(self.num_states, self.num_actions)}"
            )
        self.q_table = payload["q_table"]
        self.epsilon = payload.get("epsilon", self.epsilon)
