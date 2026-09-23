"""
scripts/train_model_free.py

Phase 1 - Sprint 3 - Step 25: Train Q-Learning, SARSA, Monte Carlo
Control, Expected SARSA and Double Q-Learning for every company, all
against the existing Sprint 2 TradingEnvironment / 36-state encoder /
action space. Expected SARSA and Double Q-Learning were added later
(same interface as the other three - see src/rl/base_agent.py) purely
to give src/decision/engine.py's ensemble vote two more, lower-variance
/ bias-corrected voters, which improves the accuracy of the final
BUY/HOLD/SELL decision without changing anything about the original
three algorithms.

Run with:
    python scripts/train_model_free.py
    python scripts/train_model_free.py --episodes 50           # override rl.episodes in config.yaml
    python scripts/train_model_free.py --companies RELI,INFY   # train a subset only

For each company:
    1. Load its training-period feature dataset (Date <= data.train_end).
    2. Build ONE TradingEnvironment configuration reused by all 5 algorithms.
    3. Train Q-Learning        -> save model + metadata + policy.
    4. Train SARSA             -> save model + metadata + policy.
    5. Train Monte Carlo       -> save model + metadata + policy.
    6. Train Expected SARSA    -> save model + metadata + policy.
    7. Train Double Q-Learning -> save model + metadata + policy.
    8. Append per-episode metrics to reports/<algo>_training.csv.
    9. Save learning-curve plots to reports/plots/.

Testing-period data (2025-01-01 -> 2026-07-31) is never touched here.
"""

import argparse
import os
import random
import sys
import time
import traceback

# Windows' legacy console codepage (cp1252) can't encode characters like
# ₹, →, ✓ used in this script's progress output - reconfigure stdout/
# stderr to UTF-8 (replacing anything still unencodable) so training
# never crashes partway through just because of a print() statement.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import load_config  # noqa: E402
from src.environment.trading_env import TradingEnvironment  # noqa: E402
from src.environment.state import StateEncoder, EXPECTED_STATE_SPACE_SIZE  # noqa: E402
from src.environment.actions import NUM_ACTIONS  # noqa: E402
from src.rl.q_learning import QLearningAgent  # noqa: E402
from src.rl.sarsa import SARSAAgent  # noqa: E402
from src.rl.monte_carlo import MonteCarloAgent  # noqa: E402
from src.rl.expected_sarsa import ExpectedSarsaAgent  # noqa: E402
from src.rl.double_q_learning import DoubleQLearningAgent  # noqa: E402
from src.rl.model_io import (  # noqa: E402
    save_model, build_metadata, extract_policy_rows, save_policy_csv,
)

FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")

ALGORITHMS = [
    ("Q-Learning", QLearningAgent, "q_learning_training.csv"),
    ("SARSA", SARSAAgent, "sarsa_training.csv"),
    ("Monte Carlo", MonteCarloAgent, "monte_carlo_training.csv"),
    # Ensemble-accuracy addition: two more tabular, model-free algorithms
    # so src/decision/engine.py's majority vote has more (and more
    # varied - lower-variance on-policy, and bias-corrected off-policy)
    # voters. See src/rl/expected_sarsa.py / src/rl/double_q_learning.py.
    ("Expected SARSA", ExpectedSarsaAgent, "expected_sarsa_training.csv"),
    ("Double Q-Learning", DoubleQLearningAgent, "double_q_learning_training.csv"),
]


def discover_companies() -> list:
    """Every company that has a Sprint 2 feature file, in a fixed sorted order."""
    if not os.path.isdir(FEATURES_DIR):
        raise FileNotFoundError(f"Sprint 2 feature data not found at {FEATURES_DIR}")
    symbols = sorted(
        f[: -len("_features.csv")]
        for f in os.listdir(FEATURES_DIR)
        if f.endswith("_features.csv")
    )
    if not symbols:
        raise FileNotFoundError(f"No feature files found in {FEATURES_DIR}")
    return symbols


def load_training_df(symbol: str, train_end: pd.Timestamp) -> pd.DataFrame:
    """Load a company's feature file and restrict it to the training period only."""
    path = os.path.join(FEATURES_DIR, f"{symbol}_features.csv")
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.loc[df["Date"] <= train_end].reset_index(drop=True)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_env(train_df: pd.DataFrame, env_cfg: dict, encoder: StateEncoder, symbol: str) -> TradingEnvironment:
    return TradingEnvironment(
        df=train_df,
        initial_capital=env_cfg["initial_capital"],
        transaction_cost=env_cfg["transaction_cost"],
        invalid_action_penalty=env_cfg["invalid_action_penalty"],
        shares_per_trade=env_cfg["shares_per_trade"],
        state_encoder=encoder,
        symbol=symbol,
    )


def train_one_agent(agent_cls, env, rl_cfg, episodes, seed):
    agent = agent_cls(
        num_states=EXPECTED_STATE_SPACE_SIZE,
        num_actions=NUM_ACTIONS,
        alpha=rl_cfg["alpha"],
        gamma=rl_cfg["gamma"],
        epsilon_start=rl_cfg["epsilon_start"],
        epsilon_min=rl_cfg["epsilon_min"],
        epsilon_decay=rl_cfg["epsilon_decay"],
        seed=seed,
    )
    assert agent.q_table.shape == (EXPECTED_STATE_SPACE_SIZE, NUM_ACTIONS), (
        f"Q-table shape mismatch: {agent.q_table.shape}"
    )

    episode_rows = []
    for ep in range(1, episodes + 1):
        metrics = agent.train_episode(env)
        episode_rows.append(metrics | {"episode": ep})
    return agent, episode_rows


def plot_learning_curves(symbol: str, algo_key: str, episode_df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(episode_df["episode"], episode_df["total_reward"])
    axes[0].set_title(f"{symbol} {algo_key} - Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")

    axes[1].plot(episode_df["episode"], episode_df["final_portfolio_value"], color="darkgreen")
    axes[1].set_title(f"{symbol} {algo_key} - Portfolio Value")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Final Portfolio Value")

    axes[2].plot(episode_df["episode"], episode_df["epsilon"], color="darkorange")
    axes[2].set_title(f"{symbol} {algo_key} - Epsilon")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Epsilon")

    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"{symbol}_{algo_key}_reward.png")
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Sprint 3 - train Q-Learning / SARSA / Monte Carlo.")
    parser.add_argument("--episodes", type=int, default=None, help="Override rl.episodes from config.yaml")
    parser.add_argument("--companies", type=str, default=None,
                         help="Comma-separated symbols to train (default: all discovered companies)")
    parser.add_argument("--no-plots", action="store_true", help="Skip learning-curve plot generation (faster)")
    args = parser.parse_args()

    cfg = load_config()
    rl_cfg = cfg["rl"]
    env_cfg = cfg["environment"]
    episodes = args.episodes or rl_cfg["episodes"]
    seed = rl_cfg["random_seed"]
    train_end = pd.Timestamp(cfg["data"]["train_end"])
    train_start = cfg["data"]["train_start"]

    set_global_seeds(seed)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    companies = discover_companies()
    if args.companies:
        wanted = [c.strip() for c in args.companies.split(",") if c.strip()]
        companies = [c for c in companies if c in wanted]

    encoder = StateEncoder()
    assert encoder.num_states == EXPECTED_STATE_SPACE_SIZE == 36
    assert NUM_ACTIONS == 3

    print(f"Companies to train: {len(companies)}")
    print(f"Episodes per (company, algorithm): {episodes}")
    print(f"States: {encoder.num_states}  Actions: {NUM_ACTIONS}\n")

    training_rows = {name: [] for name, _, _ in ALGORITHMS}
    errors = []
    trained_count = 0
    t_start = time.time()

    for symbol in companies:
        try:
            train_df = load_training_df(symbol, train_end)
            if len(train_df.dropna(subset=["Trend", "RSI_Condition", "Volatility_Condition"])) < 2:
                raise ValueError("Not enough warmed-up training rows to run an episode")

            print(f"=== {symbol} ===")
            for algo_name, agent_cls, _ in ALGORITHMS:
                t0 = time.time()
                try:
                    # Fresh env instance per algorithm, but identical
                    # construction -> same environment for all three.
                    env = make_env(train_df, env_cfg, encoder, symbol)
                    agent, episode_rows = train_one_agent(agent_cls, env, rl_cfg, episodes, seed)

                    for row in episode_rows:
                        row["company"] = symbol
                    training_rows[algo_name].extend(episode_rows)

                    metadata = build_metadata(
                        company=symbol, algorithm=algo_name, agent=agent, episodes=episodes,
                        train_start=train_start, train_end=str(train_end.date()),
                        initial_capital=env_cfg["initial_capital"],
                        transaction_cost=env_cfg["transaction_cost"],
                    )
                    paths = save_model(MODELS_DIR, symbol, algo_name, agent, metadata)

                    policy_rows = extract_policy_rows(symbol, algo_name, agent, encoder)
                    save_policy_csv(REPORTS_DIR, symbol, algo_name, policy_rows)

                    if not args.no_plots:
                        algo_key = paths["model_path"].split(os.sep)[-1].replace(".pkl", "")
                        plot_learning_curves(symbol, algo_key, pd.DataFrame(episode_rows))

                    last = episode_rows[-1]
                    dt = time.time() - t0
                    print(
                        f"  [{algo_name:11}] {episodes} eps in {dt:6.1f}s | "
                        f"final_reward={last['total_reward']:+.4f} "
                        f"final_portfolio={last['final_portfolio_value']:,.2f} "
                        f"epsilon={last['epsilon']:.4f}"
                    )
                    trained_count += 1
                except Exception as algo_exc:  # noqa: BLE001
                    errors.append({
                        "Company": symbol, "Algorithm": algo_name,
                        "Error": str(algo_exc), "Status": "FAILED",
                    })
                    print(f"  [{algo_name:11}] FAILED: {algo_exc}")

        except Exception as company_exc:  # noqa: BLE001
            errors.append({
                "Company": symbol, "Algorithm": "ALL",
                "Error": str(company_exc), "Status": "FAILED",
            })
            print(f"=== {symbol} === FAILED: {company_exc}")
            traceback.print_exc()

    # ---- Write training-metric reports ---------------------------------
    for algo_name, _, filename in ALGORITHMS:
        rows = training_rows[algo_name]
        if not rows:
            continue
        df = pd.DataFrame(rows).rename(columns={
            "company": "Company", "episode": "Episode", "total_reward": "Total_Reward",
            "final_portfolio_value": "Final_Portfolio_Value", "num_trades": "Number_of_Trades",
            "epsilon": "Epsilon",
        })
        df = df[["Company", "Episode", "Total_Reward", "Final_Portfolio_Value", "Number_of_Trades", "Epsilon"]]
        df.to_csv(os.path.join(REPORTS_DIR, filename), index=False)

    errors_path = os.path.join(REPORTS_DIR, "training_errors.csv")
    pd.DataFrame(errors, columns=["Company", "Algorithm", "Error", "Status"]).to_csv(errors_path, index=False)

    elapsed = time.time() - t_start
    expected_models = len(companies) * len(ALGORITHMS)

    print("\n========================================")
    print("PHASE 1 — SPRINT 3 COMPLETE")
    print("========================================\n")
    print(f"Companies:\n{len(companies)}\n")
    print("Algorithms:")
    print("✓ Q-Learning")
    print("✓ SARSA")
    print("✓ Monte Carlo Control")
    print("✓ Expected SARSA")
    print("✓ Double Q-Learning\n")
    print(f"States:\n{encoder.num_states}\n")
    print(f"Actions:\n{NUM_ACTIONS}\n")
    print(f"Training:\n{train_start} → {train_end.date()}\n")
    print(f"Models:\n{trained_count} / {expected_models}\n")
    print(f"Initial Capital:\n₹{env_cfg['initial_capital']:,.0f}\n")
    print(f"Transaction Cost:\n{env_cfg['transaction_cost'] * 100:.1f}%\n")
    print(f"Failures: {len(errors)}  (see {errors_path})")
    print(f"Elapsed: {elapsed:.1f}s")
    print("\nReady for:\nSPRINT 4 — Value Iteration + Policy Iteration")
    print("========================================")


if __name__ == "__main__":
    main()
