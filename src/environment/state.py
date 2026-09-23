"""
src/environment/state.py

Sprint 2 - Step 5, 6: Discrete RL state representation.

State = (Trend, RSI_Condition, Volatility_Condition, Position)

    Trend                : BULLISH | BEARISH | NEUTRAL      (3)
    RSI_Condition        : RSI_LOW | RSI_NORMAL | RSI_HIGH   (3)
    Volatility_Condition : LOW_VOLATILITY | HIGH_VOLATILITY  (2)
    Position              : NO_POSITION | HOLDING             (2)

    Total state space size = 3 x 3 x 2 x 2 = 36

The StateEncoder gives a deterministic, bijective mapping between a
state tuple and a unique integer ID in [0, 36) so tabular RL algorithms
(Q-tables etc.) can index directly by state.
"""

import itertools
from typing import NamedTuple, Tuple

from src.features.indicators import (
    BULLISH, BEARISH, NEUTRAL,
    RSI_LOW, RSI_NORMAL, RSI_HIGH,
    LOW_VOLATILITY, HIGH_VOLATILITY,
)

NO_POSITION = 0
HOLDING = 1

TREND_VALUES = (BULLISH, BEARISH, NEUTRAL)
RSI_VALUES = (RSI_LOW, RSI_NORMAL, RSI_HIGH)
VOLATILITY_VALUES = (LOW_VOLATILITY, HIGH_VOLATILITY)
POSITION_VALUES = (NO_POSITION, HOLDING)

EXPECTED_STATE_SPACE_SIZE = (
    len(TREND_VALUES) * len(RSI_VALUES) * len(VOLATILITY_VALUES) * len(POSITION_VALUES)
)


class State(NamedTuple):
    trend: str
    rsi_condition: str
    volatility_condition: str
    position: int


class StateEncoder:
    """
    Deterministic bijective encoder between a `State` tuple and an
    integer ID in [0, num_states).

    The ordering is fixed at construction time via itertools.product
    over (TREND_VALUES, RSI_VALUES, VOLATILITY_VALUES, POSITION_VALUES),
    so the same State always encodes to the same ID across runs/processes.
    """

    def __init__(self):
        self._all_states: Tuple[State, ...] = tuple(
            State(*combo)
            for combo in itertools.product(
                TREND_VALUES, RSI_VALUES, VOLATILITY_VALUES, POSITION_VALUES
            )
        )
        self._state_to_id = {state: i for i, state in enumerate(self._all_states)}
        self._id_to_state = {i: state for i, state in enumerate(self._all_states)}

        if len(self._all_states) != EXPECTED_STATE_SPACE_SIZE:
            raise AssertionError(
                f"State space size mismatch: got {len(self._all_states)}, "
                f"expected {EXPECTED_STATE_SPACE_SIZE}"
            )

    @property
    def num_states(self) -> int:
        return len(self._all_states)

    def encode(self, state) -> int:
        """Encode a State (or plain 4-tuple) into its unique integer ID."""
        state = State(*state)
        try:
            return self._state_to_id[state]
        except KeyError as exc:
            raise ValueError(f"Unrecognized state: {state}") from exc

    def decode(self, state_id: int) -> State:
        """Decode an integer ID back into its State tuple."""
        if state_id not in self._id_to_state:
            raise ValueError(f"Unrecognized state_id: {state_id}. Must be in [0, {self.num_states})")
        return self._id_to_state[state_id]

    def all_states(self) -> Tuple[State, ...]:
        return self._all_states
