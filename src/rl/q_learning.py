"""
src/rl/q_learning.py

Phase 1 - Sprint 3 - Steps 4-6: Tabular Q-Learning (off-policy TD control).

Update rule (applied every step, using the SAME action-selection policy
- epsilon-greedy - for behaviour, but bootstrapping off the greedy
max over next-state Q-values regardless of what will actually be
chosen next):

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

Runs against the existing Sprint 2 TradingEnvironment - this file
contains no environment logic of its own.
"""

import numpy as np

from src.environment.actions import HOLD
from src.rl.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """Off-policy TD control: bootstraps off max_a' Q(s', a')."""

    def train_episode(self, env) -> dict:
        """
        Run one full training episode against `env` (already constructed
        for a single company's training-period data). Updates the
        Q-table in place, decays epsilon once at the end of the episode,
        and returns per-episode metrics.
        """
        state = env.reset()
        total_reward = 0.0
        num_trades = 0

        while True:
            action = self.choose_action(state)
            next_state, reward, done, info = env.step(action)

            best_next_q = np.max(self.q_table[next_state])
            td_target = reward + self.gamma * best_next_q
            td_error = td_target - self.q_table[state, action]
            self.q_table[state, action] += self.alpha * td_error

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
