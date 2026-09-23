"""
scripts/train_new_algorithms_only.py

One-off helper: trains ONLY Expected SARSA and Double Q-Learning for
every company (the three original model-free algorithms already have
trained models on disk and are left untouched). Otherwise identical to
scripts/train_model_free.py - same config, same env construction, same
seeding, same output locations - so its output is indistinguishable
from what train_model_free.py itself would have written for these two
algorithms.

Resumable: a (company, algorithm) pair is skipped if its .pkl already
exists on disk, so re-running this script after an interruption (or
after --companies) only trains what's still missing.

Parallel: pass --workers N (N>1) to train different COMPANIES in
separate processes at once via multiprocessing - each company is still
trained single-threaded internally (tabular RL doesn't parallelize
below that level), but on an 8-core machine --workers 8 is roughly an
8x wall-clock speedup over the default single-process run. Defaults to
1 (safe on any machine, including single-core sandboxes).

Run with:
    python scripts/train_new_algorithms_only.py                     # all companies, 1 process
    python scripts/train_new_algorithms_only.py --workers 8          # all companies, 8 parallel processes
    python scripts/train_new_algorithms_only.py --companies RELI,INFY --workers 2
"""
import argparse
import os, sys, random, time, traceback
from multiprocessing import Pool
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import load_config
from src.environment.trading_env import TradingEnvironment
from src.environment.state import StateEncoder, EXPECTED_STATE_SPACE_SIZE
from src.environment.actions import NUM_ACTIONS
from src.rl.expected_sarsa import ExpectedSarsaAgent
from src.rl.double_q_learning import DoubleQLearningAgent
from src.rl.model_io import save_model, build_metadata, extract_policy_rows, save_policy_csv

FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

ALGORITHMS = [
    ("Expected SARSA", ExpectedSarsaAgent, "expected_sarsa_training.csv"),
    ("Double Q-Learning", DoubleQLearningAgent, "double_q_learning_training.csv"),
]

def discover_companies():
    return sorted(f[:-len("_features.csv")] for f in os.listdir(FEATURES_DIR) if f.endswith("_features.csv"))


def model_exists(symbol: str, file_stub: str) -> bool:
    return os.path.isfile(os.path.join(MODELS_DIR, symbol, f"{file_stub}.pkl"))


def train_one_company(symbol: str) -> dict:
    """Trains whichever of the two new algorithms don't already have a
    saved model for this company. Returns {"symbol", "rows": {algo: [episode_row,...]}, "errors": [...]}.
    Safe to run inside a worker process (imports are already done at
    module level; each call reads its own copy of the training config)."""
    cfg = load_config()
    rl_cfg, env_cfg = cfg["rl"], cfg["environment"]
    episodes = rl_cfg["episodes"]
    seed = rl_cfg["random_seed"]
    train_end = pd.Timestamp(cfg["data"]["train_end"])
    train_start = cfg["data"]["train_start"]
    encoder = StateEncoder()

    out_rows = {name: [] for name, _, _ in ALGORITHMS}
    errors = []

    path = os.path.join(FEATURES_DIR, f"{symbol}_features.csv")
    df = pd.read_csv(path, parse_dates=["Date"])
    train_df = df.loc[df["Date"] <= train_end].reset_index(drop=True)
    if len(train_df.dropna(subset=["Trend", "RSI_Condition", "Volatility_Condition"])) < 2:
        errors.append({"Company": symbol, "Algorithm": "ALL", "Error": "not enough rows", "Status": "FAILED"})
        return {"symbol": symbol, "rows": out_rows, "errors": errors}

    for algo_name, agent_cls, _ in ALGORITHMS:
        from src.rl.model_io import ALGORITHM_FILE_NAMES
        file_stub = ALGORITHM_FILE_NAMES[algo_name]
        if model_exists(symbol, file_stub):
            print(f"  [{symbol}/{algo_name:17}] SKIP (already trained)", flush=True)
            continue
        t0 = time.time()
        try:
            env = TradingEnvironment(
                df=train_df, initial_capital=env_cfg["initial_capital"],
                transaction_cost=env_cfg["transaction_cost"],
                invalid_action_penalty=env_cfg["invalid_action_penalty"],
                shares_per_trade=env_cfg["shares_per_trade"],
                state_encoder=encoder, symbol=symbol,
            )
            agent = agent_cls(
                num_states=EXPECTED_STATE_SPACE_SIZE, num_actions=NUM_ACTIONS,
                alpha=rl_cfg["alpha"], gamma=rl_cfg["gamma"],
                epsilon_start=rl_cfg["epsilon_start"], epsilon_min=rl_cfg["epsilon_min"],
                epsilon_decay=rl_cfg["epsilon_decay"], seed=seed,
            )
            episode_rows = []
            for ep in range(1, episodes + 1):
                metrics = agent.train_episode(env)
                episode_rows.append(metrics | {"episode": ep})
            for row in episode_rows:
                row["company"] = symbol
            out_rows[algo_name].extend(episode_rows)

            metadata = build_metadata(
                company=symbol, algorithm=algo_name, agent=agent, episodes=episodes,
                train_start=train_start, train_end=str(train_end.date()),
                initial_capital=env_cfg["initial_capital"], transaction_cost=env_cfg["transaction_cost"],
            )
            save_model(MODELS_DIR, symbol, algo_name, agent, metadata)
            policy_rows = extract_policy_rows(symbol, algo_name, agent, encoder)
            save_policy_csv(REPORTS_DIR, symbol, algo_name, policy_rows)

            last = episode_rows[-1]
            dt = time.time() - t0
            print(f"  [{symbol}/{algo_name:17}] {episodes} eps in {dt:6.1f}s | final_reward={last['total_reward']:+.4f} "
                  f"final_portfolio={last['final_portfolio_value']:,.2f}", flush=True)
        except Exception as exc:
            errors.append({"Company": symbol, "Algorithm": algo_name, "Error": str(exc), "Status": "FAILED"})
            print(f"  [{symbol}/{algo_name:17}] FAILED: {exc}", flush=True)
            traceback.print_exc()

    return {"symbol": symbol, "rows": out_rows, "errors": errors}


def append_rows_to_csv(algo_name: str, filename: str, new_rows: list):
    """Append (not overwrite) so repeated/parallel runs accumulate rather than clobber each other."""
    if not new_rows:
        return
    d = pd.DataFrame(new_rows).rename(columns={
        "company": "Company", "episode": "Episode", "total_reward": "Total_Reward",
        "final_portfolio_value": "Final_Portfolio_Value", "num_trades": "Number_of_Trades",
        "epsilon": "Epsilon",
    })
    d = d[["Company", "Episode", "Total_Reward", "Final_Portfolio_Value", "Number_of_Trades", "Epsilon"]]
    out_path = os.path.join(REPORTS_DIR, filename)
    header = not os.path.isfile(out_path)
    d.to_csv(out_path, mode="a", header=header, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--companies", type=str, default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config()
    seed = cfg["rl"]["random_seed"]
    random.seed(seed); np.random.seed(seed)
    os.makedirs(MODELS_DIR, exist_ok=True)

    companies = discover_companies()
    if args.companies:
        wanted = [c.strip() for c in args.companies.split(",") if c.strip()]
        companies = [c for c in companies if c in wanted]

    print(f"Companies queued: {len(companies)}  Workers: {args.workers}\n", flush=True)
    t_start = time.time()
    all_errors = []

    if args.workers > 1:
        with Pool(args.workers) as pool:
            for result in pool.imap_unordered(train_one_company, companies):
                for algo_name, _, filename in ALGORITHMS:
                    append_rows_to_csv(algo_name, filename, result["rows"][algo_name])
                all_errors.extend(result["errors"])
    else:
        for symbol in companies:
            result = train_one_company(symbol)
            for algo_name, _, filename in ALGORITHMS:
                append_rows_to_csv(algo_name, filename, result["rows"][algo_name])
            all_errors.extend(result["errors"])

    if all_errors:
        err_path = os.path.join(REPORTS_DIR, "new_algorithms_training_errors.csv")
        pd.DataFrame(all_errors).to_csv(err_path, index=False)

    print(f"\nDONE in {time.time()-t_start:.1f}s. Errors: {len(all_errors)}", flush=True)


if __name__ == "__main__":
    main()
