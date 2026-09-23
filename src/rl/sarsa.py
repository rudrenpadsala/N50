"""
src/rl/sarsa.py

Phase 1 - Sprint 3 - Steps 7-8: Tabular SARSA (on-policy TD control).

Update rule - uses the ACTUAL next action chosen by the epsilon-greedy
policy (not the max over next-state Q-values, that would be Q-Learning):

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * Q(s',a') - Q(s,a) ]

Same hyperparameters, environment, reward and epsilon-greedy strategy
as Q-Learning so the two are directly comparable - only the update rule
differs.
"""

from src.environment.actions import HOLD
from src.rl.base_agent import BaseAgent


class SARSAAgent(BaseAgent):
    """On-policy TD control: bootstraps off Q(s', a') for the action actually taken next."""

    def train_episode(self, env) -> dict:
        state = env.reset()
        action = self.choose_action(state)
        total_reward = 0.0
        num_trades = 0

        while True:
            next_state, reward, done, info = env.step(action)
            # SARSA needs the NEXT action the policy would actually take,
            # not the greedy max - this is what makes it on-policy.
            next_action = self.choose_action(next_state)

            td_target = reward + self.gamma * self.q_table[next_state, next_action]
            td_error = td_target - self.q_table[state, action]
            self.q_table[state, action] += self.alpha * td_error

            if action != HOLD and not info["invalid_action"]:
                num_trades += 1
            total_reward += reward

            state, action = next_state, next_action
            if done:
                break

        self.decay_epsilon()

        return {
            "total_reward": total_reward,
            "final_portfolio_value": env.get_portfolio_value(),
            "num_trades": num_trades,
            "epsilon": self.epsilon,
        }
