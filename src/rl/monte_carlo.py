"""
src/rl/monte_carlo.py

Phase 1 - Sprint 3 - Steps 9-10: First-visit Monte Carlo Control.

Unlike Q-Learning/SARSA, Monte Carlo does not bootstrap and does not
update Q after every step. Instead it:

    1. Plays a complete episode using the current epsilon-greedy policy,
       storing (state, action, reward) for every step.
    2. Walks the episode BACKWARD, accumulating the discounted return
       G_t = r_(t+1) + gamma*r_(t+2) + gamma^2*r_(t+3) + ...
    3. For every state-action pair's FIRST occurrence in the episode
       (determined with a forward pass, then applied during the
       backward return calculation), updates Q(s,a) towards G using
       an incremental running average:

           N(s,a) <- N(s,a) + 1
           Q(s,a) <- Q(s,a) + (G - Q(s,a)) / N(s,a)

       which is algebraically identical to Q(s,a) = mean(all first-visit
       returns observed for (s,a) so far), the textbook first-visit MC
       control update - no separate alpha is used here, only counts.
    4. Improves the (epsilon-greedy) policy implicitly, since choose_action
       always reads from the just-updated Q-table.
"""

from src.environment.actions import HOLD
from src.rl.base_agent import BaseAgent

import numpy as np


class MonteCarloAgent(BaseAgent):
    """First-visit Monte Carlo control with incremental-mean Q updates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Visit counts N(s,a), used for the incremental-mean update.
        self.visit_counts = np.zeros((self.num_states, self.num_actions), dtype=np.int64)

    def train_episode(self, env) -> dict:
        state = env.reset()
        episode = []  # list of (state, action, reward)
        total_reward = 0.0
        num_trades = 0

        # ---- 1. Generate one complete episode -------------------------
        while True:
            action = self.choose_action(state)
            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward))

            if action != HOLD and not info["invalid_action"]:
                num_trades += 1
            total_reward += reward

            state = next_state
            if done:
                break

        # ---- 2. Forward pass: mark genuine first-visit steps ----------
        seen = set()
        is_first_visit = []
        for (s, a, _r) in episode:
            is_first_visit.append((s, a) not in seen)
            seen.add((s, a))

        # ---- 3. Backward pass: accumulate G, update Q at first visits -
        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = r + self.gamma * G
            if is_first_visit[t]:
                self.visit_counts[s, a] += 1
                n = self.visit_counts[s, a]
                self.q_table[s, a] += (G - self.q_table[s, a]) / n

        # ---- 4. Policy improvement is implicit: choose_action() always
        #         reads the just-updated Q-table with epsilon-greedy. --
        self.decay_epsilon()

        return {
            "total_reward": total_reward,
            "final_portfolio_value": env.get_portfolio_value(),
            "num_trades": num_trades,
            "epsilon": self.epsilon,
        }

    # Persist visit counts alongside the Q-table so training can, in
    # principle, be resumed without resetting the running averages.
    def save_q_table(self, path: str) -> None:
        import pickle
        payload = {
            "q_table": self.q_table,
            "visit_counts": self.visit_counts,
            "epsilon": self.epsilon,
            "num_states": self.num_states,
            "num_actions": self.num_actions,
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
        self.visit_counts = payload.get(
            "visit_counts", np.zeros_like(self.q_table, dtype=np.int64)
        )
        self.epsilon = payload.get("epsilon", self.epsilon)
