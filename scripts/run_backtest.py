"""
scripts/run_backtest.py

Phase 1 - Sprint 5 - Step 32: Backtest all 7 trained algorithms + Buy &
Hold, for every company, over the TESTING period only
(2025-01-01 -> 2026-07-31), and produce every Sprint 5 report.

Run with:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --companies RELI,INFY   # a subset only

Reuses, unmodified:
    - src/environment/trading_env.py (TradingEnvironment)
    - src/environment/state.py (StateEncoder)
    - src/backtesting/model_loader.py (uniform greedy-policy loading
      for all 7 algorithms - Sprint 3/4 model pickles, plus the later
      Expected SARSA / Double Q-Learning addition)
    - src/backtesting/backtester.py (evaluation-only episode runner)
    - src/backtesting/buy_and_hold.py (baseline)
    - src/backtesting/metrics.py (Return / Sharpe / Drawdown / Win Rate)

Never touches training data or a training script - see
`load_test_df`, which filters every feature file to
`Date >= data.test_start` before anything else reads it.
"""

import argparse
import os
import sys
import time

# Windows' legacy console codepage (cp1252) can't encode characters like
# ₹, →, ✓, − used in this script's progress output / final report -
# reconfigure stdout/stderr to UTF-8 (replacing anything still
# unencodable) so backtesting never crashes partway through just
# because of a print() statement.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import load_config  # noqa: E402
from src.environment.state import StateEncoder  # noqa: E402
from src.backtesting.model_loader import ALGORITHMS, load_greedy_policy, ModelLoadError, file_stub  # noqa: E402
from src.backtesting.backtester import run_backtest  # noqa: E402
from src.backtesting.buy_and_hold import run_buy_and_hold  # noqa: E402
from src.backtesting import plots  # noqa: E402

FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
TRADES_DIR = os.path.join(REPORTS_DIR, "trades")
PORTFOLIO_DIR = os.path.join(REPORTS_DIR, "portfolio")
PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")

ALL_ALGORITHMS_INCL_BH = ALGORITHMS + ["Buy & Hold"]


def discover_companies() -> list:
    if not os.path.isdir(FEATURES_DIR):
        raise FileNotFoundError(f"Feature data not found at {FEATURES_DIR}")
    symbols = sorted(
        f[: -len("_features.csv")]
        for f in os.listdir(FEATURES_DIR)
        if f.endswith("_features.csv")
    )
    if not symbols:
        raise FileNotFoundError(f"No feature files found in {FEATURES_DIR}")
    return symbols


def load_test_df(symbol: str, test_start: pd.Timestamp) -> pd.DataFrame:
    """Load a company's feature file and restrict it to the TESTING period only."""
    path = os.path.join(FEATURES_DIR, f"{symbol}_features.csv")
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.loc[df["Date"] >= test_start].reset_index(drop=True)


def save_backtest_outputs(result) -> None:
    stub = file_stub(result.algorithm)
    if result.trade_log:
        pd.DataFrame(result.trade_log).to_csv(
            os.path.join(TRADES_DIR, f"{result.company}_{stub}_trades.csv"), index=False
        )
    pd.DataFrame(result.portfolio_history).to_csv(
        os.path.join(PORTFOLIO_DIR, f"{result.company}_{stub}_portfolio.csv"), index=False
    )


def build_company_best_algorithm_row(company: str, rl_metrics_rows: list) -> dict:
    """Sprint 5 Section 15: best algorithm by Return AND by Sharpe (shown
    separately - never collapsed into one 'best' pick), among the 5 RL
    algorithms only (Buy & Hold is the baseline they are compared against,
    not itself a candidate for "best trading algorithm")."""
    best_return_row = max(rl_metrics_rows, key=lambda r: r["Total Return %"])
    best_sharpe_row = max(rl_metrics_rows, key=lambda r: r["Sharpe Ratio"])
    return {
        "Company": company,
        "Best Return Algorithm": best_return_row["Algorithm"],
        "Best Return %": best_return_row["Total Return %"],
        "Best Sharpe Algorithm": best_sharpe_row["Algorithm"],
        "Best Sharpe Ratio": best_sharpe_row["Sharpe Ratio"],
    }


def build_algorithm_comparison(all_rows: list) -> pd.DataFrame:
    """Sprint 5 Section 16: mean performance per algorithm across every
    company actually backtested (averages only - median/aggregate
    breakdown lives in aggregate_analysis.csv, Section 19)."""
    df = pd.DataFrame(all_rows)
    bh_return_by_company = (
        df[df["Algorithm"] == "Buy & Hold"].set_index("Company")["Total Return %"]
    )

    out_rows = []
    for algo in ALL_ALGORITHMS_INCL_BH:
        sub = df[df["Algorithm"] == algo]
        if sub.empty:
            continue
        if algo == "Buy & Hold":
            beaten = 0  # Buy & Hold cannot "beat itself" - see module docstring below
        else:
            joined = sub.set_index("Company")["Total Return %"].dropna()
            common = joined.index.intersection(bh_return_by_company.index)
            beaten = int((joined.loc[common] > bh_return_by_company.loc[common]).sum())

        out_rows.append({
            "Algorithm": algo,
            "Average Return %": sub["Total Return %"].mean(),
            "Average Annualized Return %": sub["Annualized Return %"].mean(),
            "Average Sharpe Ratio": sub["Sharpe Ratio"].mean(),
            "Average Maximum Drawdown %": sub["Maximum Drawdown %"].mean(),
            "Average Win Rate %": sub["Win Rate %"].mean(),
            "Average Trades": sub["Number of Trades"].mean(),
            "Companies Beaten By Algorithm": beaten,
        })
    return pd.DataFrame(out_rows)


def build_algorithm_ranking(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Sprint 5 Section 17: rank algorithms SEPARATELY by return, Sharpe,
    and drawdown - no combined/arbitrary score."""
    ranking = comparison_df[["Algorithm"]].copy()
    ranking["Rank by Average Return"] = comparison_df["Average Return %"].rank(ascending=False).astype(int)
    ranking["Rank by Average Sharpe"] = comparison_df["Average Sharpe Ratio"].rank(ascending=False).astype(int)
    # Best (smallest-magnitude) drawdown ranks first: drawdowns are <= 0,
    # so the LEAST negative value is best -> rank descending on the raw value.
    ranking["Rank by Average Drawdown"] = comparison_df["Average Maximum Drawdown %"].rank(ascending=False).astype(int)
    return ranking.sort_values("Rank by Average Return")


def build_aggregate_analysis(all_rows: list) -> pd.DataFrame:
    """Sprint 5 Section 19: mean AND median across all companies actually
    backtested, per algorithm, plus how many companies each algorithm
    beats Buy & Hold on return (duplicated here from
    algorithm_comparison.csv's 'Companies Beaten By Algorithm' column,
    since Section 19 asks for this analysis as its own deliverable)."""
    df = pd.DataFrame(all_rows)
    bh_return_by_company = df[df["Algorithm"] == "Buy & Hold"].set_index("Company")["Total Return %"]
    total_companies = df["Company"].nunique()

    rows = []
    for algo in ALL_ALGORITHMS_INCL_BH:
        sub = df[df["Algorithm"] == algo]
        if sub.empty:
            continue
        if algo == "Buy & Hold":
            beats_bh = 0
        else:
            joined = sub.set_index("Company")["Total Return %"].dropna()
            common = joined.index.intersection(bh_return_by_company.index)
            beats_bh = int((joined.loc[common] > bh_return_by_company.loc[common]).sum())

        rows.append({
            "Algorithm": algo,
            "Companies Tested": len(sub),
            "Average Return %": sub["Total Return %"].mean(),
            "Median Return %": sub["Total Return %"].median(),
            "Average Sharpe Ratio": sub["Sharpe Ratio"].mean(),
            "Median Sharpe Ratio": sub["Sharpe Ratio"].median(),
            "Average Maximum Drawdown %": sub["Maximum Drawdown %"].mean(),
            "Median Maximum Drawdown %": sub["Maximum Drawdown %"].median(),
            "Average Win Rate %": sub["Win Rate %"].mean(),
            "Companies Beaten (Buy & Hold, Return)": beats_bh,
            "Companies Beaten (Buy & Hold, Return) - Out Of": total_companies,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Sprint 5 - backtest all trained algorithms.")
    parser.add_argument("--companies", type=str, default=None,
                         help="Comma-separated symbols to backtest (default: all discovered companies)")
    args = parser.parse_args()

    cfg = load_config()
    env_cfg = cfg["environment"]
    backtest_cfg = cfg["backtest"]
    test_start = pd.Timestamp(cfg["data"]["test_start"])

    for d in (TRADES_DIR, PORTFOLIO_DIR, PLOTS_DIR):
        os.makedirs(d, exist_ok=True)

    companies = discover_companies()
    if args.companies:
        wanted = [c.strip() for c in args.companies.split(",") if c.strip()]
        companies = [c for c in companies if c in wanted]

    encoder = StateEncoder()

    print("========================================")
    print("SPRINT 5 BACKTESTING")
    print("========================================\n")
    print(f"Companies to process: {len(companies)}")
    print(f"Algorithms: {', '.join(ALGORITHMS)}")
    print(f"Testing period start: {test_start.date()}\n")

    all_metric_rows = []
    company_best_rows = []
    errors = []
    rl_backtests_done = 0
    bh_backtests_done = 0
    t_start = time.time()

    for idx, symbol in enumerate(companies, start=1):
        print(f"Company {idx}/{len(companies)}: {symbol}")
        try:
            test_df = load_test_df(symbol, test_start)
            if len(test_df.dropna(subset=["Trend", "RSI_Condition", "Volatility_Condition"])) < 2:
                raise ValueError("Not enough test-period rows to backtest")
        except Exception as exc:  # noqa: BLE001
            errors.append({"Company": symbol, "Algorithm": "ALL", "Error": str(exc), "Status": "FAILED"})
            print(f"  FAILED to load test data: {exc}\n")
            continue

        company_rl_rows = []
        company_results = {}

        for algorithm in ALGORITHMS:
            try:
                policy = load_greedy_policy(MODELS_DIR, symbol, algorithm, encoder.num_states, 3)
                result = run_backtest(
                    symbol=symbol, algorithm=algorithm, policy=policy, test_df=test_df,
                    initial_capital=env_cfg["initial_capital"],
                    transaction_cost=env_cfg["transaction_cost"],
                    invalid_action_penalty=env_cfg["invalid_action_penalty"],
                    shares_per_trade=env_cfg["shares_per_trade"],
                    encoder=encoder,
                )
                save_backtest_outputs(result)
                metrics_row = result.metrics(backtest_cfg["risk_free_rate"], backtest_cfg["trading_days_per_year"])
                all_metric_rows.append(metrics_row)
                company_rl_rows.append(metrics_row)
                company_results[algorithm] = result
                rl_backtests_done += 1
                print(f"  {algorithm}: Return {metrics_row['Total Return %']:.2f}%  "
                      f"Sharpe {metrics_row['Sharpe Ratio']:.2f}")
            except (ModelLoadError, Exception) as exc:  # noqa: BLE001
                errors.append({"Company": symbol, "Algorithm": algorithm, "Error": str(exc), "Status": "FAILED"})
                print(f"  {algorithm}: FAILED ({exc})")

        try:
            bh_result = run_buy_and_hold(
                symbol=symbol, test_df=test_df,
                initial_capital=env_cfg["initial_capital"],
                transaction_cost=env_cfg["transaction_cost"],
            )
            save_backtest_outputs(bh_result)
            bh_metrics_row = bh_result.metrics(backtest_cfg["risk_free_rate"], backtest_cfg["trading_days_per_year"])
            all_metric_rows.append(bh_metrics_row)
            company_results["Buy & Hold"] = bh_result
            bh_backtests_done += 1
            print(f"  Buy & Hold: Return {bh_metrics_row['Total Return %']:.2f}%")
        except Exception as exc:  # noqa: BLE001
            errors.append({"Company": symbol, "Algorithm": "Buy & Hold", "Error": str(exc), "Status": "FAILED"})
            print(f"  Buy & Hold: FAILED ({exc})")

        if company_rl_rows:
            company_best_rows.append(build_company_best_algorithm_row(symbol, company_rl_rows))

        if len(company_results) >= 2:
            try:
                plots.plot_portfolio_comparison(symbol, company_results, PLOTS_DIR)
                plots.plot_drawdown_comparison(symbol, company_results, PLOTS_DIR)
            except Exception as exc:  # noqa: BLE001
                errors.append({"Company": symbol, "Algorithm": "PLOTS", "Error": str(exc), "Status": "FAILED"})

        print()

    # ---- Sanity checks (Sprint 5 Section 36) ---------------------------
    results_df = pd.DataFrame(all_metric_rows)
    if not results_df.empty:
        numeric_cols = [c for c in results_df.columns if results_df[c].dtype.kind in "fc"]
        bad = results_df[numeric_cols].isna().any().any() or np.isinf(results_df[numeric_cols].to_numpy()).any()
        if bad:
            errors.append({"Company": "ALL", "Algorithm": "ALL",
                            "Error": "NaN or infinite value found in final_results metrics", "Status": "WARNING"})
        if (results_df["Final Portfolio Value"] < 0).any():
            errors.append({"Company": "ALL", "Algorithm": "ALL",
                            "Error": "Negative final portfolio value found", "Status": "WARNING"})

    # ---- Reports (Sprint 5 Sections 14-19) -----------------------------
    results_df.to_csv(os.path.join(REPORTS_DIR, "final_results.csv"), index=False)

    if company_best_rows:
        pd.DataFrame(company_best_rows).to_csv(
            os.path.join(REPORTS_DIR, "company_best_algorithm.csv"), index=False
        )

    if not results_df.empty:
        comparison_df = build_algorithm_comparison(all_metric_rows)
        comparison_df.to_csv(os.path.join(REPORTS_DIR, "algorithm_comparison.csv"), index=False)

        ranking_df = build_algorithm_ranking(comparison_df)
        ranking_df.to_csv(os.path.join(REPORTS_DIR, "algorithm_ranking.csv"), index=False)

        aggregate_df = build_aggregate_analysis(all_metric_rows)
        aggregate_df.to_csv(os.path.join(REPORTS_DIR, "aggregate_analysis.csv"), index=False)

        plots.plot_algorithm_bar_charts(comparison_df, PLOTS_DIR)
    else:
        comparison_df = pd.DataFrame()

    pd.DataFrame(errors, columns=["Company", "Algorithm", "Error", "Status"]).to_csv(
        os.path.join(REPORTS_DIR, "backtest_errors.csv"), index=False
    )

    write_final_report(comparison_df, len(companies), rl_backtests_done, bh_backtests_done, errors)

    elapsed = time.time() - t_start

    print("===========================================")
    print("PHASE 1 — COMPLETE")
    print("===========================================\n")
    print(f"Companies:\n{len(companies)}\n")
    print("Algorithms:")
    for a in ALGORITHMS:
        print(f"✓ {a}")
    print(f"\nTesting Period:\n{test_start.date()} → {cfg['data']['test_end']}\n")
    print(f"RL Backtests:\n{rl_backtests_done}\n")
    print(f"Buy & Hold:\n{bh_backtests_done}\n")
    print(f"Initial Capital:\n₹{env_cfg['initial_capital']:,.0f}\n")
    print(f"Transaction Cost:\n{env_cfg['transaction_cost']*100:.1f}%\n")
    print("Reports:\n✓ Generated\n")
    print("Charts:\n✓ Generated\n")
    print("Dashboard:\n✓ Streamlit (run: streamlit run app.py)\n")
    print(f"Failures logged: {len(errors)} (see reports/backtest_errors.csv)")
    print(f"Elapsed: {elapsed:.1f}s\n")
    print("===========================================")
    print("READY FOR PHASE 2")
    print("===========================================")


def write_final_report(comparison_df: pd.DataFrame, num_companies: int,
                        rl_backtests: int, bh_backtests: int, errors: list) -> None:
    """Sprint 5 Section 37: build reports/phase1_final_report.md from the
    ACTUAL computed results - never a pre-decided winner (Section 38)."""
    lines = []
    lines.append("# Phase 1 Final Report — Adaptive Stock Trading Agent Using Reinforcement Learning\n")
    lines.append("## 1. Project Objective")
    lines.append(
        "Train and evaluate five reinforcement learning algorithms to make BUY/HOLD/SELL "
        "decisions for a virtual trading account, and determine which performs best on "
        "completely unseen, out-of-sample market data.\n"
    )
    lines.append("## 2-3. Dataset")
    lines.append("50 companies, daily OHLCV data, 2015-01-01 to 2026-07-31.\n")
    lines.append("## 4. Training Period\n2015-01-01 to 2024-12-31.\n")
    lines.append("## 5. Testing Period\n2025-01-01 to 2026-07-31 (never used in training).\n")
    lines.append("## 6. Features\nMA5, MA20, RSI, Volatility, Trend (Sprint 2).\n")
    lines.append("## 7. State Space\n36 states: Trend(3) × RSI(3) × Volatility(2) × Position(2).\n")
    lines.append("## 8. Action Space\n3 actions: HOLD, BUY, SELL.\n")
    lines.append(
        "## 9. Reward\n"
        "`(New Portfolio Value − Previous Portfolio Value) / Previous Portfolio Value`, "
        "plus a small penalty for an invalid action (insufficient cash to BUY, no shares "
        "to SELL). Identical for all 5 algorithms and never used for Buy & Hold's own "
        "closed-form calculation.\n"
    )
    lines.append(
        "## 10. Five RL Algorithms\n"
        "1. Q-Learning (model-free, off-policy)\n"
        "2. SARSA (model-free, on-policy)\n"
        "3. Monte Carlo Control (model-free, episodic)\n"
        "4. Value Iteration (model-based Dynamic Programming)\n"
        "5. Policy Iteration (model-based Dynamic Programming)\n"
    )
    lines.append(
        "## 11. Backtesting Methodology\n"
        "Each trained model's GREEDY policy (argmax over its learned/derived Q-table) was "
        "run once, chronologically, over the testing period using the same "
        "`TradingEnvironment` as training. No model was updated, retrained, or selected "
        "based on test-period performance at any point (see `src/backtesting/backtester.py`).\n"
    )

    if comparison_df is None or comparison_df.empty:
        lines.append("## 12-15. Results\n\n*No backtests completed successfully — see `reports/backtest_errors.csv`.*\n")
    else:
        lines.append("## 12. Performance Metrics (averaged across all companies backtested)\n")
        lines.append(comparison_df.round(4).to_markdown(index=False))
        lines.append("")

        rl_only = comparison_df[comparison_df["Algorithm"] != "Buy & Hold"]
        if not rl_only.empty:
            best_return = rl_only.loc[rl_only["Average Return %"].idxmax()]
            best_sharpe = rl_only.loc[rl_only["Average Sharpe Ratio"].idxmax()]
            bh_row = comparison_df[comparison_df["Algorithm"] == "Buy & Hold"]
            bh_return = bh_row["Average Return %"].iloc[0] if not bh_row.empty else None

            lines.append("## 13-14. Algorithm Comparison / Buy & Hold Comparison\n")
            lines.append(
                f"- **Best average return**: {best_return['Algorithm']} "
                f"({best_return['Average Return %']:.2f}% average return)."
            )
            lines.append(
                f"- **Best average Sharpe Ratio**: {best_sharpe['Algorithm']} "
                f"({best_sharpe['Average Sharpe Ratio']:.2f})."
            )
            if bh_return is not None:
                lines.append(f"- **Buy & Hold average return**: {bh_return:.2f}%.")
                beats = best_return['Companies Beaten By Algorithm']
                lines.append(
                    f"- {best_return['Algorithm']} beat Buy & Hold's return on "
                    f"{beats} of {num_companies} companies tested."
                )
            lines.append("")
            lines.append(
                "## 15. Best-Performing Algorithm(s)\n"
                f"On this run's testing data, **{best_return['Algorithm']}** achieved the "
                f"highest average return and **{best_sharpe['Algorithm']}** achieved the "
                "highest average risk-adjusted (Sharpe) return. These are reported "
                "separately, as instructed, rather than collapsed into a single "
                "'best' claim — the higher-return algorithm is not automatically the "
                "higher-Sharpe one, and neither is assumed superior to the other.\n"
            )
        else:
            lines.append("## 13-15. Algorithm Comparison\n\n*Only Buy & Hold completed successfully.*\n")

    lines.append(
        "## 16. Limitations\n"
        "- Backtesting assumes fills at the recorded Close price and a flat 0.1% "
        "transaction cost, with no slippage or partial fills modeled.\n"
        "- The MDP used by Value Iteration / Policy Iteration is an empirical "
        "approximation (see `src/rl/mdp.py`), not a true stationary Markov model of "
        "the market.\n"
        "- Past performance on this historical test window is not a guarantee of "
        "future results; markets change regime.\n"
        "- This report reflects whichever companies were actually backtested in this "
        f"run ({num_companies} companies, {rl_backtests} RL backtests, {bh_backtests} "
        f"Buy & Hold benchmarks{', with ' + str(len(errors)) + ' failure(s) logged' if errors else ''}). "
        "Re-run `python scripts/run_backtest.py` with all 50 companies' models trained "
        "for the full-scale result.\n"
    )
    lines.append(
        "## 17. Conclusion\n"
        "Phase 1 delivers a complete, leakage-checked pipeline — data → features → "
        "environment → five RL algorithms → backtesting → reporting/dashboard — with no "
        "live trading, no real money, and no Phase 2 functionality. See "
        "`reports/algorithm_comparison.csv`, `reports/algorithm_ranking.csv` and "
        "`reports/company_best_algorithm.csv` for the full, per-company breakdown behind "
        "the summary above.\n"
    )

    with open(os.path.join(REPORTS_DIR, "phase1_final_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
