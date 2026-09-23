"""
src/rl/double_q_learning.py

Phase 1 - Sprint 3 addition: Tabular Double Q-Learning (van Hasselt, 2010).

Plain Q-Learning always bootstraps off max_a' Q(s', a') using the SAME
table that produced the action-values being maximised - early in
training, when Q-estimates are still noisy, that max operator
systematically OVER-estimates action values (maximisation bias), which
here means the agent can become over-confident about a BUY or SELL that
only looked good because of noise, not because it actually was good.

Double Q-Learning removes that bias by keeping TWO independent tables,
Q_A and Q_B, and on every step randomly updating only one of them -
using the OTHER table to evaluate the action the first table judges
best, decoupling "which action looks best" from "how good is that
action":

    with probability 0.5:
        a* = argmax_a Q_A(s', a)
        Q_A(s,a) <- Q_A(s,a) + alpha * [ r + gamma * Q_B(s', a*) - Q_A(s,a) ]
    else (swap A and B):
        b* = argmax_a Q_B(s', a)
        Q_B(s,a) <- Q_B(s,a) + alpha * [ r + gamma * Q_A(s', b*) - Q_B(s,a) ]

Behaviour (epsilon-greedy action selection) and the final saved/served
policy both use the AVERAGE table Q = (Q_A + Q_B) / 2, which is why
`self.q_table` is kept in sync with that average after every update -
this is what BaseAgent.choose_action()/get_policy() read from, and it
is also the single (num_states, num_actions) array
src/backtesting/model_loader.py and src/decision/engine.py expect on
disk, so Double Q-Learning is a drop-in 7th voter in the decision
ensemble with no changes needed anywhere else in the pipeline.
"""

import pickle

import numpy as np

from src.environment.actions import HOLD
from src.rl.base_agent import BaseAgent


class DoubleQLearningAgent(BaseAgent):
    """Off-policy TD control with two Q-tables to cancel maximisation bias."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Independent optimistic-neutral tables. self.q_table (from
        # BaseAgent) is kept as their running average - see module
        # docstring - so epsilon-greedy action selection, get_policy(),
        # and on-disk persistence all keep working unmodified.
        self.q_table_a = np.zeros((self.num_states, self.num_actions), dtype=float)
        self.q_table_b = np.zeros((self.num_states, self.num_actions), dtype=float)

    def _sync_average(self) -> None:
        self.q_table = (self.q_table_a + self.q_table_b) / 2.0

    def train_episode(self, env) -> dict:
        state = env.reset()
        total_reward = 0.0
        num_trades = 0

        while True:
            action = self.choose_action(state)
            next_state, reward, done, info = env.step(action)

            if self._rng.random() < 0.5:
                best_next_action = int(np.argmax(self.q_table_a[next_state]))
                td_target = reward + self.gamma * self.q_table_b[next_state, best_next_action]
                td_error = td_target - self.q_table_a[state, action]
                self.q_table_a[state, action] += self.alpha * td_error
            else:
                best_next_action = int(np.argmax(self.q_table_b[next_state]))
                td_target = reward + self.gamma * self.q_table_a[next_state, best_next_action]
                td_error = td_target - self.q_table_b[state, action]
                self.q_table_b[state, action] += self.alpha * td_error

            self._sync_average()

            if action != HOLD and not info["invalid_action"]:
                num_trades += 1
            total_reward += reward

            state = next_state
            if done:
                break

        self.decay_epsilon()

        return {
            "total_reward": total_reward,
            "final_portfolio_value": env.get_portfolio_value(),
            "num_trades": num_trades,
            "epsilon": self.epsilon,
        }

    # Persist both tables (plus the shared average) so training could, in
    # principle, be resumed without discarding either table's history -
    # same pattern MonteCarloAgent uses for its visit_counts.
    def save_q_table(self, path: str) -> None:
        payload = {
            "q_table": self.q_table,
            "q_table_a": self.q_table_a,
            "q_table_b": self.q_table_b,
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
        self.q_table_a = payload.get("q_table_a", self.q_table.copy())
        self.q_table_b = payload.get("q_table_b", self.q_table.copy())
        self.epsilon = payload.get("epsilon", self.epsilon)
