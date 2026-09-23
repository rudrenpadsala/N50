"""
scripts/train_dynamic_programming.py

Phase 1 - Sprint 4 - Step 20: Build a tabular MDP from TRAINING-period
data and run Value Iteration + Policy Iteration for every company,
against the SAME 36-state / 3-action abstraction, reward definition,
transaction cost and initial capital as Sprint 3's Q-Learning / SARSA /
Monte Carlo Control (Sprint 4 Section 19).

Run with:
    python scripts/train_dynamic_programming.py
    python scripts/train_dynamic_programming.py --companies RELI,INFY   # train a subset only

For each company:
    1. Load its training-period feature dataset (Date <= data.train_end).
    2. Build ONE tabular MDP (src/rl/mdp.py) from that data only.
    3. Run Value Iteration -> extract optimal policy.
    4. Run Policy Iteration -> extract optimal policy.
    5. Save both policies (models/<SYMBOL>/value_iteration.pkl,
       policy_iteration.pkl) + JSON metadata.
    6. Save each algorithm's policy table (State ID, Trend, RSI,
       Volatility, Position, Value, Best Action) to reports/policies/.
    7. Append convergence rows to reports/value_iteration_training.csv
       and reports/policy_iteration_training.csv.

Testing-period data (2025-01-01 -> 2026-07-31) is never read here -
`load_training_df` filters every feature file to `data.train_end`
before anything else touches it (identical safeguard to
scripts/train_model_free.py).
"""

import argparse
import os
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

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import load_config  # noqa: E402
from src.environment.state import StateEncoder, EXPECTED_STATE_SPACE_SIZE  # noqa: E402
from src.environment.actions import NUM_ACTIONS  # noqa: E402
from src.rl.mdp import build_mdp  # noqa: E402
from src.rl.value_iteration import ValueIterationAgent  # noqa: E402
from src.rl.policy_iteration import PolicyIterationAgent  # noqa: E402
from src.rl.model_io import (  # noqa: E402
    save_model, build_dp_metadata, extract_dp_policy_rows, save_policy_csv,
)

FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


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


def main():
    parser = argparse.ArgumentParser(
        description="Sprint 4 - train Value Iteration / Policy Iteration."
    )
    parser.add_argument("--companies", type=str, default=None,
                         help="Comma-separated symbols to train (default: all discovered companies)")
    args = parser.parse_args()

    cfg = load_config()
    env_cfg = cfg["environment"]
    dp_cfg = cfg["dp"]
    train_end = pd.Timestamp(cfg["data"]["train_end"])
    train_start = cfg["data"]["train_start"]

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    companies = discover_companies()
    if args.companies:
        wanted = [c.strip() for c in args.companies.split(",") if c.strip()]
        companies = [c for c in companies if c in wanted]

    encoder = StateEncoder()
    assert encoder.num_states == EXPECTED_STATE_SPACE_SIZE == 36
    assert NUM_ACTIONS == 3

    print("========================================")
    print("SPRINT 4 TRAINING")
    print("========================================\n")
    print(f"Companies to process: {len(companies)}")
    print(f"States: {encoder.num_states}  Actions: {NUM_ACTIONS}")
    print(f"Gamma: {dp_cfg['gamma']}  Theta: {dp_cfg['theta']}  "
          f"Max iterations: {dp_cfg['max_iterations']}\n")

    vi_report_rows = []
    pi_report_rows = []
    errors = []
    vi_completed = 0
    pi_completed = 0
    t_start = time.time()

    for idx, symbol in enumerate(companies, start=1):
        print(f"Company {idx}/{len(companies)}: {symbol}")
        try:
            train_df = load_training_df(symbol, train_end)
            if len(train_df.dropna(subset=["Trend", "RSI_Condition", "Volatility_Condition"])) < 2:
                raise ValueError("Not enough warmed-up training rows to build an MDP")

            mdp = build_mdp(
                train_df=train_df,
                encoder=encoder,
                initial_capital=env_cfg["initial_capital"],
                transaction_cost=env_cfg["transaction_cost"],
                invalid_action_penalty=env_cfg["invalid_action_penalty"],
                shares_per_trade=env_cfg["shares_per_trade"],
            )

            # ---- Value Iteration -----------------------------------
            try:
                vi_agent = ValueIterationAgent(
                    mdp, gamma=dp_cfg["gamma"], theta=dp_cfg["theta"],
                    max_iterations=dp_cfg["max_iterations"],
                )
                vi_result = vi_agent.run()

                for h in vi_agent.history:
                    vi_report_rows.append({
                        "Company": symbol,
                        "Iteration": h["iteration"],
                        "Max_Value_Change": h["max_delta"],
                        "Converged": h["iteration"] == vi_agent.iterations_run and vi_agent.converged,
                    })

                vi_metadata = build_dp_metadata(
                    company=symbol, algorithm="Value Iteration",
                    num_states=vi_agent.num_states, num_actions=vi_agent.num_actions,
                    gamma=vi_agent.gamma, theta=vi_agent.theta,
                    train_start=train_start, train_end=str(train_end.date()),
                    initial_capital=env_cfg["initial_capital"],
                    transaction_cost=env_cfg["transaction_cost"],
                    iterations=vi_agent.iterations_run,
                    convergence_status="Converged" if vi_agent.converged else "Max iterations reached",
                )
                save_model(MODELS_DIR, symbol, "Value Iteration", vi_agent, vi_metadata)
                vi_rows = extract_dp_policy_rows(symbol, "Value Iteration", vi_agent, encoder)
                save_policy_csv(REPORTS_DIR, symbol, "Value Iteration", vi_rows)

                status = "Converged" if vi_result["converged"] else "NOT converged (max iterations reached)"
                print(f"Value Iteration: {status}")
                vi_completed += 1
            except Exception as vi_exc:  # noqa: BLE001
                errors.append({"Company": symbol, "Algorithm": "Value Iteration",
                                "Error": str(vi_exc), "Status": "FAILED"})
                print(f"Value Iteration: FAILED ({vi_exc})")

            # ---- Policy Iteration ------------------------------------
            try:
                pi_agent = PolicyIterationAgent(
                    mdp, gamma=dp_cfg["gamma"], theta=dp_cfg["theta"],
                    max_iterations=dp_cfg["max_iterations"], init=dp_cfg["policy_init"],
                )
                pi_result = pi_agent.run()

                for h in pi_agent.history:
                    pi_report_rows.append({
                        "Company": symbol,
                        "Policy_Iteration": h["policy_iteration"],
                        "Evaluation_Iterations": h["evaluation_iterations"],
                        "Policy_Stable": h["policy_stable"],
                    })

                pi_metadata = build_dp_metadata(
                    company=symbol, algorithm="Policy Iteration",
                    num_states=pi_agent.num_states, num_actions=pi_agent.num_actions,
                    gamma=pi_agent.gamma, theta=pi_agent.theta,
                    train_start=train_start, train_end=str(train_end.date()),
                    initial_capital=env_cfg["initial_capital"],
                    transaction_cost=env_cfg["transaction_cost"],
                    iterations=pi_agent.policy_iterations_run,
                    convergence_status="Converged" if pi_agent.policy_stable else "Max iterations reached",
                )
                save_model(MODELS_DIR, symbol, "Policy Iteration", pi_agent, pi_metadata)
                pi_rows = extract_dp_policy_rows(symbol, "Policy Iteration", pi_agent, encoder)
                save_policy_csv(REPORTS_DIR, symbol, "Policy Iteration", pi_rows)

                status = "Converged" if pi_result["policy_stable"] else "NOT converged (max iterations reached)"
                print(f"Policy Iteration: {status}")
                pi_completed += 1
            except Exception as pi_exc:  # noqa: BLE001
                errors.append({"Company": symbol, "Algorithm": "Policy Iteration",
                                "Error": str(pi_exc), "Status": "FAILED"})
                print(f"Policy Iteration: FAILED ({pi_exc})")

            print()

        except Exception as company_exc:  # noqa: BLE001
            errors.append({"Company": symbol, "Algorithm": "ALL",
                            "Error": str(company_exc), "Status": "FAILED"})
            print(f"FAILED: {company_exc}\n")
            traceback.print_exc()

    # ---- Write convergence reports (Sprint 4 Section 17) ---------------
    pd.DataFrame(vi_report_rows).to_csv(
        os.path.join(REPORTS_DIR, "value_iteration_training.csv"), index=False
    )
    pd.DataFrame(pi_report_rows).to_csv(
        os.path.join(REPORTS_DIR, "policy_iteration_training.csv"), index=False
    )
    errors_path = os.path.join(REPORTS_DIR, "dp_training_errors.csv")
    pd.DataFrame(errors, columns=["Company", "Algorithm", "Error", "Status"]).to_csv(errors_path, index=False)

    elapsed = time.time() - t_start

    print("========================================")
    print("SPRINT 4 COMPLETE")
    print("========================================\n")
    print(f"Companies processed: {len(companies)}\n")
    print("Value Iteration:")
    print(f"{vi_completed}/{len(companies)} completed\n")
    print("Policy Iteration:")
    print(f"{pi_completed}/{len(companies)} completed\n")
    print(f"States:\n{encoder.num_states}\n")
    print(f"Actions:\n{NUM_ACTIONS}\n")
    print(f"Failures: {len(errors)}  (see {errors_path})")
    print(f"Elapsed: {elapsed:.1f}s\n")
    print("Ready for Sprint 5:")
    print("Backtesting + Model Comparison + Dashboard")
    print("========================================")


if __name__ == "__main__":
    main()
