# Phase 1 Final Report — Adaptive Stock Trading Agent Using Reinforcement Learning

## 1. Project Objective
Train and evaluate five reinforcement learning algorithms to make BUY/HOLD/SELL decisions for a virtual trading account, and determine which performs best on completely unseen, out-of-sample market data.

## 2-3. Dataset
50 companies, daily OHLCV data, 2015-01-01 to 2026-07-31.

## 4. Training Period
2015-01-01 to 2024-12-31.

## 5. Testing Period
2025-01-01 to 2026-07-31 (never used in training).

## 6. Features
MA5, MA20, RSI, Volatility, Trend (Sprint 2).

## 7. State Space
36 states: Trend(3) × RSI(3) × Volatility(2) × Position(2).

## 8. Action Space
3 actions: HOLD, BUY, SELL.

## 9. Reward
`(New Portfolio Value − Previous Portfolio Value) / Previous Portfolio Value`, plus a small penalty for an invalid action (insufficient cash to BUY, no shares to SELL). Identical for all 5 algorithms and never used for Buy & Hold's own closed-form calculation.

## 10. Five RL Algorithms
1. Q-Learning (model-free, off-policy)
2. SARSA (model-free, on-policy)
3. Monte Carlo Control (model-free, episodic)
4. Value Iteration (model-based Dynamic Programming)
5. Policy Iteration (model-based Dynamic Programming)

## 11. Backtesting Methodology
Each trained model's GREEDY policy (argmax over its learned/derived Q-table) was run once, chronologically, over the testing period using the same `TradingEnvironment` as training. No model was updated, retrained, or selected based on test-period performance at any point (see `src/backtesting/backtester.py`).

## 12. Performance Metrics (averaged across all companies backtested)

| Algorithm         |   Average Return % |   Average Annualized Return % |   Average Sharpe Ratio |   Average Maximum Drawdown % |   Average Win Rate % |   Average Trades |   Companies Beaten By Algorithm |
|:------------------|-------------------:|------------------------------:|-----------------------:|-----------------------------:|---------------------:|-----------------:|--------------------------------:|
| Q-Learning        |             1.6153 |                        0.8407 |                -0.0282 |                      -9.2672 |              49.7814 |           183.3  |                              21 |
| SARSA             |            -0.9259 |                       -0.9986 |                 0.005  |                     -12.4014 |              51.3758 |           252.4  |                              16 |
| Monte Carlo       |             5.3046 |                        2.9494 |                 0.1787 |                     -17.1835 |              49.0093 |           204.44 |                              24 |
| Value Iteration   |             4.1212 |                        2.1563 |                 0.2151 |                     -14.3805 |              47.8095 |           130.88 |                              21 |
| Policy Iteration  |             4.0973 |                        2.1413 |                 0.2124 |                     -14.489  |              48.1681 |           128.6  |                              21 |
| Expected SARSA    |             1.2045 |                        0.732  |                -0.262  |                      -2.1998 |              44.3143 |           139.52 |                              18 |
| Double Q-Learning |            -0.0413 |                       -0.0284 |                -0.3515 |                      -0.4889 |              28.9759 |            62.18 |                              18 |
| Buy & Hold        |             6.4206 |                        2.6966 |                 0.2372 |                     -28.6677 |               0      |             1    |                               0 |

## 13-14. Algorithm Comparison / Buy & Hold Comparison

- **Best average return**: Monte Carlo (5.30% average return).
- **Best average Sharpe Ratio**: Value Iteration (0.22).
- **Buy & Hold average return**: 6.42%.
- Monte Carlo beat Buy & Hold's return on 24 of 50 companies tested.

## 15. Best-Performing Algorithm(s)
On this run's testing data, **Monte Carlo** achieved the highest average return and **Value Iteration** achieved the highest average risk-adjusted (Sharpe) return. These are reported separately, as instructed, rather than collapsed into a single 'best' claim — the higher-return algorithm is not automatically the higher-Sharpe one, and neither is assumed superior to the other.

## 16. Limitations
- Backtesting assumes fills at the recorded Close price and a flat 0.1% transaction cost, with no slippage or partial fills modeled.
- The MDP used by Value Iteration / Policy Iteration is an empirical approximation (see `src/rl/mdp.py`), not a true stationary Markov model of the market.
- Past performance on this historical test window is not a guarantee of future results; markets change regime.
- This report reflects whichever companies were actually backtested in this run (50 companies, 350 RL backtests, 50 Buy & Hold benchmarks). Re-run `python scripts/run_backtest.py` with all 50 companies' models trained for the full-scale result.

## 17. Conclusion
Phase 1 delivers a complete, leakage-checked pipeline — data → features → environment → five RL algorithms → backtesting → reporting/dashboard — with no live trading, no real money, and no Phase 2 functionality. See `reports/algorithm_comparison.csv`, `reports/algorithm_ranking.csv` and `reports/company_best_algorithm.csv` for the full, per-company breakdown behind the summary above.
