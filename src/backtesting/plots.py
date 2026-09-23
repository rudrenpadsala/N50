"""
src/backtesting/plots.py

Phase 1 - Sprint 5 - Step 18: Performance charts, generated from the
SAME `BacktestResult` objects / comparison dataframe the reports use -
no separate computation path, so a chart can never show a number that
disagrees with the CSV reports.

Per-company charts (portfolio value, drawdown) are produced for every
company actually backtested, overlaying all algorithms that
successfully produced a result for that company (Sprint 5 Section 18:
"Allow comparison of all 5 algorithms and Buy & Hold"). Aggregate bar
charts (return/Sharpe/drawdown/win-rate comparison) are produced once,
from `algorithm_comparison.csv`'s already-averaged numbers.
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless - this project never opens an interactive window
import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_portfolio_comparison(symbol: str, results_by_algorithm: dict, plots_dir: str) -> str:
    """Sprint 5 Section 18, item 1 & 7: Portfolio Value vs Date, RL vs Buy & Hold, all on one chart."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for algorithm, result in results_by_algorithm.items():
        hist = pd.DataFrame(result.portfolio_history)
        ax.plot(hist["Date"], hist["Portfolio_Value"], label=algorithm)
    ax.set_title(f"{symbol} — Portfolio Value vs Date (Testing Period)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (₹)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = os.path.join(plots_dir, f"{symbol}_portfolio_comparison.png")
    _save(fig, path)
    return path


def plot_drawdown_comparison(symbol: str, results_by_algorithm: dict, plots_dir: str) -> str:
    """Sprint 5 Section 18, item 2: Drawdown vs Date, all algorithms overlaid."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for algorithm, result in results_by_algorithm.items():
        hist = pd.DataFrame(result.portfolio_history)
        ax.plot(hist["Date"], hist["Drawdown"] * 100.0, label=algorithm)
    ax.set_title(f"{symbol} — Drawdown vs Date (Testing Period)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = os.path.join(plots_dir, f"{symbol}_drawdown_comparison.png")
    _save(fig, path)
    return path


def plot_algorithm_bar_charts(comparison_df: pd.DataFrame, plots_dir: str) -> list:
    """Sprint 5 Section 18, items 3-6: aggregate bar charts, straight from
    algorithm_comparison.csv's already-computed averages."""
    charts = [
        ("Average Return %", "algorithm_return_comparison.png", "Average Return by Algorithm (%)"),
        ("Average Sharpe Ratio", "algorithm_sharpe_comparison.png", "Average Sharpe Ratio by Algorithm"),
        ("Average Maximum Drawdown %", "algorithm_drawdown_comparison.png", "Average Maximum Drawdown by Algorithm (%)"),
        ("Average Win Rate %", "algorithm_win_rate_comparison.png", "Average Win Rate by Algorithm (%)"),
    ]
    paths = []
    for column, filename, title in charts:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(comparison_df["Algorithm"], comparison_df[column])
        ax.set_title(title)
        ax.set_ylabel(column)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
        path = os.path.join(plots_dir, filename)
        _save(fig, path)
        paths.append(path)
    return paths
