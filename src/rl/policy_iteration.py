"""
src/rl/policy_iteration.py

Phase 1 - Sprint 4 - Steps 5-8: Classical Policy Iteration (model-based
Dynamic Programming), run against the SAME tabular MDP (P(s'|s,a),
R(s,a)) as Value Iteration.

Policy Evaluation, for the CURRENT policy pi, applied synchronously
over all states each sweep:

    V(s) <- R(s, pi(s)) + gamma * sum_s' P(s'|s, pi(s)) * V(s')

repeated until max|V_new - V_old| < THETA or MAX_ITERATIONS evaluation
sweeps are reached (Sprint 4 Section 7).

Policy Improvement, for every state:

    pi'(s) = argmax_a [ R(s,a) + gamma * sum_s' P(s'|s,a) * V(s') ]

If pi'(s) differs from pi(s) for any state, the policy is updated and
evaluation runs again on the new policy. Evaluation and improvement
alternate until the policy no longer changes for any state
(`policy_stable = True`) - Sprint 4 Section 8.
"""

import numpy as np

from src.environment.actions import ACTIONS, NUM_ACTIONS
from src.rl.mdp import MDP

# Configurable defaults (Sprint 4 Section 7) - overridable via
# config/config.yaml's `dp:` section, never hard-coded elsewhere.
GAMMA = 0.95
THETA = 1e-6
MAX_ITERATIONS = 1000


class PolicyIterationAgent:
    """Classical Policy Iteration over a fixed, pre-built MDP."""

    def __init__(self, mdp: MDP, gamma: float = GAMMA, theta: float = THETA,
                 max_iterations: int = MAX_ITERATIONS, init: str = "HOLD"):
        """
        `init` controls policy initialization (Sprint 4 Section 6):
            "HOLD"   - HOLD (action 0) for every state (default)
            "RANDOM" - a fixed-seed random valid action per state

        Both options only ever use information available at construction
        time (action 0, or a fixed RNG seed) - never future information.
        """
        self.mdp = mdp
        self.num_states = mdp.num_states
        self.num_actions = mdp.num_actions
        self.gamma = float(gamma)
        self.theta = float(theta)
        self.max_iterations = int(max_iterations)

        self.V = np.zeros(self.num_states, dtype=float)
        self.policy = self._initial_policy(init)
        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=float)

        self.policy_stable = False
        self.policy_iterations_run = 0
        self.total_evaluation_iterations = 0
        self.history = []  # [{"policy_iteration", "evaluation_iterations", "policy_stable"}, ...]

    # ------------------------------------------------------------------
    # Step 6 - Policy initialization
    # ------------------------------------------------------------------

    def _initial_policy(self, init: str) -> np.ndarray:
        init_upper = init.upper()
        if init_upper == "HOLD":
            return np.zeros(self.num_states, dtype=int)
        if init_upper == "RANDOM":
            rng = np.random.default_rng(42)  # fixed seed -> reproducible, no future information
            return rng.integers(0, self.num_actions, size=self.num_states)
        raise ValueError(f"Unknown policy init strategy: {init!r} (expected 'HOLD' or 'RANDOM')")

    # ------------------------------------------------------------------
    # Step 7 - Policy evaluation
    # ------------------------------------------------------------------

    def _action_value(self, state: int, action: int) -> float:
        return float(
            self.mdp.rewards[state, action]
            + self.gamma * (self.mdp.transition_probs[state, action] @ self.V)
        )

    def evaluate_policy(self) -> int:
        """
        Iterative policy evaluation for the CURRENT self.policy.
        Returns the number of sweeps performed until
        max_change < theta (or max_iterations was reached).
        """
        for it in range(1, self.max_iterations + 1):
            new_V = np.zeros_like(self.V)
            max_delta = 0.0
            for s in range(self.num_states):
                new_V[s] = self._action_value(s, int(self.policy[s]))
                max_delta = max(max_delta, abs(new_V[s] - self.V[s]))
            self.V = new_V
            if max_delta < self.theta:
                return it
        return self.max_iterations

    # ------------------------------------------------------------------
    # Step 8 - Policy improvement
    # ------------------------------------------------------------------

    def improve_policy(self) -> bool:
        """
        Greedy policy improvement over every state. Returns True iff at
        least one state's action changed (i.e. the policy is NOT yet
        stable and another evaluation round is needed).
        """
        changed = False
        for s in range(self.num_states):
            q_values = np.array([self._action_value(s, a) for a in ACTIONS])
            self.q_table[s] = q_values
            best_action = int(np.argmax(q_values))
            if best_action != self.policy[s]:
                self.policy[s] = best_action
                changed = True
        return changed

    # ------------------------------------------------------------------
    # Full loop: Policy Evaluation -> Policy Improvement -> ... -> stable
    # ------------------------------------------------------------------

    def run(self) -> dict:
        for pi_iter in range(1, self.max_iterations + 1):
            eval_iters = self.evaluate_policy()
            self.total_evaluation_iterations += eval_iters
            changed = self.improve_policy()
            self.policy_iterations_run = pi_iter
            self.policy_stable = not changed

            self.history.append({
                "policy_iteration": pi_iter,
                "evaluation_iterations": eval_iters,
                "policy_stable": self.policy_stable,
            })
            if self.policy_stable:
                break

        return {
            "policy_stable": self.policy_stable,
            "policy_iterations": self.policy_iterations_run,
            "total_evaluation_iterations": self.total_evaluation_iterations,
        }

    def get_policy(self) -> np.ndarray:
        return self.policy

    def get_q_table(self) -> np.ndarray:
        """
        Q(s,a) = R(s,a) + gamma * sum_s' P(s'|s,a) V(s') for the
        converged V. Lets this agent plug directly into src/rl/
        model_io.py's existing `extract_policy_rows` / `save_policy_csv`
        (Sprint 3), reusing that code rather than duplicating it.
        """
        return self.q_table

    # ------------------------------------------------------------------
    # Persistence - mirrors BaseAgent.save_q_table's payload shape
    # (src/rl/base_agent.py) so src/rl/model_io.py's `save_model`
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
            "policy_stable": self.policy_stable,
            "policy_iterations": self.policy_iterations_run,
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
        self.policy_stable = payload.get("policy_stable", self.policy_stable)
        self.policy_iterations_run = payload.get("policy_iterations", self.policy_iterations_run)
