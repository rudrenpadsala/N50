"""
scripts/test_environment.py

Sprint 2 - Step 26: Manual environment test/demo script.

Run with:
    python scripts/test_environment.py
"""

import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import load_config  # noqa: E402
from src.environment.trading_env import TradingEnvironment  # noqa: E402
from src.environment.actions import HOLD, BUY, SELL  # noqa: E402
from src.environment.state import StateEncoder  # noqa: E402

FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")


def load_features(symbol: str) -> pd.DataFrame:
    path = os.path.join(FEATURES_DIR, f"{symbol}_features.csv")
    return pd.read_csv(path, parse_dates=["Date"])


def demo_one_company(symbol: str, cfg: dict, encoder: StateEncoder, max_steps: int = 10):
    print(f"\n{'=' * 60}\nCompany: {symbol}\n{'=' * 60}")

    df = load_features(symbol)
    env_cfg = cfg["environment"]
    env = TradingEnvironment(
        df=df,
        initial_capital=env_cfg["initial_capital"],
        transaction_cost=env_cfg["transaction_cost"],
        invalid_action_penalty=env_cfg["invalid_action_penalty"],
        shares_per_trade=env_cfg["shares_per_trade"],
        state_encoder=encoder,
        symbol=symbol,
    )

    state = env.reset()
    print(f"Initial state ID: {state} -> {encoder.decode(state)}")
    print(f"Initial portfolio value: {env.get_portfolio_value():,.2f}")

    action_sequence = [BUY, HOLD, SELL] + [HOLD] * (max_steps - 3)
    for i, action in enumerate(action_sequence[:max_steps]):
        if env.is_done():
            break
        next_state, reward, done, info = env.step(action)
        print(
            f"step={i:2} action={info['action']:4} price={info['price']:9.2f} "
            f"cash={info['cash']:10.2f} shares={info['shares']:2} "
            f"portfolio={info['portfolio_value']:10.2f} reward={reward:+.6f} "
            f"invalid={info['invalid_action']} done={done}"
        )

    print(f"Final portfolio value after {i + 1} step(s): {env.get_portfolio_value():,.2f}")

    # Run to full termination to confirm the episode ends cleanly.
    steps_to_end = 0
    while not env.is_done():
        env.step(HOLD)
        steps_to_end += 1
    print(f"Ran {steps_to_end} more HOLD step(s) to reach termination. is_done={env.is_done()}")


def main():
    cfg = load_config()
    encoder = StateEncoder()
    print(f"State space size: {encoder.num_states} (expected 36)")

    companies = ["RELI", "INFY", "TCSc1_NS"]
    for symbol in companies:
        demo_one_company(symbol, cfg, encoder)


if __name__ == "__main__":
    main()
