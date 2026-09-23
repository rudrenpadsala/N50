"""
src/environment/actions.py

Sprint 2 - Step 7: Action space.

Exactly 3 discrete actions. Using plain int constants (rather than a
Python Enum) keeps them directly usable as array/table indices for the
tabular RL algorithms in Sprint 3, while ACTIONS / ACTION_NAMES give a
single source of truth for validation and human-readable logging.
"""

HOLD = 0
BUY = 1
SELL = 2

ACTIONS = (HOLD, BUY, SELL)

ACTION_NAMES = {
    HOLD: "HOLD",
    BUY: "BUY",
    SELL: "SELL",
}

NUM_ACTIONS = len(ACTIONS)


def is_valid_action(action: int) -> bool:
    """The environment should never accept/produce an action outside {0,1,2}."""
    return action in ACTIONS


def action_name(action: int) -> str:
    if not is_valid_action(action):
        raise ValueError(f"Undefined action: {action!r}. Must be one of {ACTIONS}")
    return ACTION_NAMES[action]
