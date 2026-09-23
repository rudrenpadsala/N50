"""
src/rl/expected_sarsa.py

Phase 1 - Sprint 3 addition: Tabular Expected SARSA (on-policy TD control).

Expected SARSA is a small but genuinely useful change on top of SARSA:
instead of bootstrapping off the Q-value of the ONE next action that was
actually (randomly) sampled by the epsilon-greedy policy - which injects
sampling noise into every single update - it bootstraps off the full
EXPECTATION of Q(s', a') under the epsilon-greedy policy itself:

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * E_{a'~pi}[Q(s',a')] - Q(s,a) ]

    where  E_{a'~pi}[Q(s',a')] = sum_a' pi(a'|s') * Q(s',a')

and pi is the SAME epsilon-greedy policy `choose_action()` samples from:
each of the (possibly several, tied) greedy actions shares probability
mass (1 - epsilon) / (#greedy actions), and every action gets epsilon /
num_actions on top of that.

Because it averages over every possible next action instead of a single
sampled one, Expected SARSA has strictly lower variance than SARSA while
remaining on-policy - in practice this tends to give steadier, more
accurate BUY/HOLD/SELL Q-estimates from the same amount of training data,
which is exactly why it is added here as a 6th voter in the decision
ensemble (see src/decision/engine.py) rather than a replacement for
SARSA - more low-correlation, low-variance voters improve the ensemble's
majority vote, not fewer.

Same hyperparameters, environment, reward and epsilon-greedy action
selection as Q-Learning/SARSA so all three are directly comparable -
only the bootstrap target differs.
"""

import numpy as np

from src.environment.actions import HOLD
from src.rl.base_agent import BaseAgent


class ExpectedSarsaAgent(BaseAgent):
    """On-policy TD control: bootstraps off E_{a'~pi}[Q(s',a')], not a single sampled a'."""

    def _expected_q(self, state: int) -> float:
        """
        E_{a'~pi}[Q(state, a')] under the SAME epsilon-greedy policy used
        by BaseAgent.choose_action() - including its unbiased tie-break
        (ties among equally good actions split the greedy probability
        mass evenly, never favouring the lowest action index).
        """
        q_values = self.q_table[state]
        max_q = q_values.max()
        greedy_mask = np.isclose(q_values, max_q)
        num_greedy = int(greedy_mask.sum())

        probs = np.full(self.num_actions, self.epsilon / self.num_actions)
        probs[greedy_mask] += (1.0 - self.epsilon) / num_greedy
        return float(np.dot(probs, q_values))

    def train_episode(self, env) -> dict:
        state = env.reset()
        total_reward = 0.0
        num_trades = 0

        while True:
            action = self.choose_action(state)
            next_state, reward, done, info = env.step(action)

            expected_next_q = self._expected_q(next_state)
            td_target = reward + self.gamma * expected_next_q
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
