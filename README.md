# Phase 1 — Sprint 1: Data Foundation

## Project Purpose

This is the data foundation for an **Adaptive Stock Trading Agent Using
Reinforcement Learning**. The finished system (later sprints) will train
five RL algorithms — Q-Learning, SARSA, Monte Carlo Control, Value
Iteration, and Policy Iteration — to make BUY / HOLD / SELL decisions.

**Sprint 1 scope is data only.** No RL environment, rewards, features
(MA5/MA20/RSI/volatility), trading logic, live API/Upstox integration,
or dashboard is implemented here. Those belong to later sprints.

## 50 Companies

All 50 raw historical datasets provided by the user were found, inspected,
and processed. See `reports/data_quality_summary.csv` for the full list
with per-company statistics, and `src/data/company_map.py` for the
filename → company/symbol mapping.

**Note on company names:** raw filenames follow investing.com's export
naming. Codes ending in `c1_NS` are that source's naming for continuous
NSE *futures* series, not the plain equity ticker. A couple of codes
(`INGL`, `MAXE`) could not be identified with full confidence and are
flagged `UNVERIFIED` in the mapping — the original file code is always
used as the canonical `Symbol`, so no data is ever mis-attributed even
where the display name is a best guess.

## Dataset Period

Source data covers **01-01-2015 to 31-07-2026**. 45 of the 50 companies
have data across that entire window; 5 companies have a shorter history
because they were not listed/available that early (Eternal/Zomato from
2021, Jio Financial from 2023, plus two others starting 2020 and one
in late 2015) — this is expected and is reported, not treated as an error.

## Data Cleaning

Implemented in `src/data/cleaner.py`:

- Raw columns `Price, Vol.` are renamed to `Close, Volume`; `Change %`
  is dropped. Final schema: `Date, Open, High, Low, Close, Volume`.
- `Date` parsed from `DD-MM-YYYY` to datetime; unparseable dates are
  dropped and logged (none found in the 50 provided files).
- Data is sorted **ascending** by date (raw files are newest-first).
- Prices are parsed from strings with thousands separators (`"1,307.80"`);
  Volume is parsed from `K`/`M`/`B`-suffixed strings (`"8.62M"`).
- **Missing values — documented strategy, not a blanket fill:**
  - Every affected file had exactly one isolated missing `Volume`
    value with otherwise complete OHLC data for that day. These are
    forward-filled from the prior trading day's volume (never using
    future information), and every imputation is logged.
  - Rows with a missing/unparseable `Open/High/Low/Close` are **dropped**,
    never imputed, since fabricating a price is not defensible.
- Exact duplicate rows and duplicate dates (keeping the first
  chronological occurrence) are removed and logged.
- The raw files in `data/raw/` are never modified.

## Data Validation

Implemented in `src/data/validator.py`. For every processed dataset:

- `High >= Open`, `High >= Close`, `High >= Low`
- `Low <= Open`, `Low <= Close`
- `Open, High, Low, Close > 0`
- `Volume >= 0`

Violations are **reported, never silently fixed**. 9 of the 50 raw
datasets (all NSE-futures-style series) contain a small number of
genuine rows where `Close` marginally exceeds `High` in the source
data — these are flagged `FAIL` in `reports/data_quality_summary.csv`
with the exact row counts, for the user to decide how to handle before
Sprint 2.

## Training / Testing Split

Chronological split, implemented in `src/data/splitter.py`:

```
TRAINING: 01-01-2015 → 31-12-2024
TESTING:  01-01-2025 → 31-07-2026
```

**No random shuffling is used anywhere in this pipeline.**

## Data Leakage Prevention

- The split is a fixed date-boundary partition, never a random shuffle.
- Cleaning (missing-value forward-fill) only ever looks **backward** in
  time within a single company's series — never forward, and never
  across the train/test boundary.
- No statistic computed from the testing window (2025-01-01 onward) is
  used to clean, standardize, or otherwise transform the training window.
- No RL features, states, or rewards are computed in this sprint, so
  there is no feature-engineering leakage surface yet.
- Test data is written only to `data/test/` and is never read by any
  step that prepares or writes `data/train/`.

## Folder Structure

```
project/
│
├── data/
│   ├── raw/          # original files, untouched (50 files)
│   ├── processed/    # cleaned, standardized OHLCV per company
│   ├── train/         # 01-01-2015 -> 31-12-2024
│   └── test/           # 01-01-2025 -> 31-07-2026
│
├── src/
│   └── data/
│       ├── __init__.py
│       ├── company_map.py
│       ├── loader.py
│       ├── cleaner.py
│       ├── validator.py
│       ├── splitter.py
│       └── prepare.py
│
├── reports/
│   ├── data_quality_summary.csv
│   └── train_test_summary.csv
│
├── tests/
│   └── test_data_pipeline.py
│
├── requirements.txt
└── README.md
```

## How to Run the Pipeline

```bash
pip install -r requirements.txt
python -m src.data.prepare
```

This will (re)generate everything in `data/processed/`, `data/train/`,
`data/test/`, and `reports/`.

## How to Run Tests

```bash
pytest
```

32 tests currently pass, covering: dataset discovery/loading, column
standardization, price/volume parsing, date conversion and sorting,
duplicate detection and removal, the documented missing-value strategy,
OHLC/price/volume validation, and the chronological train/test split
(boundaries, no overlap, deterministic/non-random behavior, and
leakage prevention).

## Sprint 1 Status

All Sprint 1 acceptance criteria are met. 41/50 datasets are fully
clean (`PASS`); 9/50 have genuine OHLC anomalies in the *source* data
that are reported in `reports/data_quality_summary.csv` for review
before they're used downstream — they were intentionally **not**
silently altered.

---

# Phase 1 — Sprint 2: Feature Engineering + RL Trading Environment

Sprint 2 builds on Sprint 1's cleaned data (reused as-is, not
duplicated) to produce the market features, discrete RL state space,
action space, and trading environment that Sprint 3's RL algorithms
(Q-Learning, SARSA, Monte Carlo Control, Value Iteration, Policy
Iteration) will consume. **No RL algorithm is implemented in this
sprint.**

## Architecture

```
Historical Data (Sprint 1: data/processed/)
        |
        v
Features (MA5, MA20, RSI, Volatility -> Trend, RSI_Condition, Volatility_Condition)
        |
        v
RL State  (Trend x RSI_Condition x Volatility_Condition x Position -> 36 discrete states)
        |
        v
Trading Environment (reset / step / get_state / get_portfolio_value / is_done)
        |
        v
Action (HOLD / BUY / SELL)
        |
        v
Portfolio (Cash, Shares, Portfolio Value)
        |
        v
Reward ((New PV - Prev PV) / Prev PV, plus invalid-action penalty)
```

## 1. Feature Engineering (`src/features/indicators.py`, `build_features.py`)

Computed for all 50 companies, using each company's **full** Sprint 1
processed series (2015–2026) so that rolling windows are warmed up
*before* the testing period begins — never using test-period values to
compute anything, and never using future rows relative to any given row:

- **MA5** — 5-day simple moving average of Close (current day inclusive).
- **MA20** — 20-day simple moving average of Close.
- **RSI (14-period)** — standard formula, `RSI = 100 - 100/(1+RS)` where
  `RS = avg_gain(14) / avg_loss(14)` (simple, non-exponential rolling
  average). Always in `[0, 100]`. Classified as:
  - `RSI < 30` → `RSI_LOW`
  - `30 <= RSI <= 70` → `RSI_NORMAL`
  - `RSI > 70` → `RSI_HIGH`
- **Volatility** — rolling 20-day standard deviation of daily returns
  (`(Close_t - Close_(t-1)) / Close_(t-1)`). Classified as
  `LOW_VOLATILITY` / `HIGH_VOLATILITY` using a **per-company threshold
  fit only from that company's training-period (≤ 2024-12-31)
  volatility values** (median split), then applied unchanged to the
  rest of the series, including the testing period. This threshold is
  never refit on test data. See `reports/volatility_thresholds.csv`.
- **Trend** — `Close > MA20` → `BULLISH`, `Close < MA20` → `BEARISH`,
  else `NEUTRAL`.

### Feature warm-up

MA20/RSI/Volatility need history before they're defined. The first 19
rows of each company's series have `NaN` for these columns (and
downstream `Trend`/`RSI_Condition`/`Volatility_Condition`) — this is
never filled or faked. The trading environment automatically drops
these warm-up rows when it loads a feature file (see below), so
episodes only ever run over fully-defined states. Because features are
computed over the *full* series, the testing period (starting
2025-01-01, thousands of trading days after warm-up) is never affected
by warm-up gaps.

Feature files: `data/features/{SYMBOL}_features.csv` (Sprint 1's
`data/processed/` is never overwritten). Run:

```bash
python -m src.features.build_features
```

## 2. Position

The environment tracks `Position`: `0 = NO_POSITION`, `1 = HOLDING`.
It is part of the RL state.

## 3. RL State (`src/environment/state.py`)

```
State = (Trend, RSI_Condition, Volatility_Condition, Position)
```

`3 (Trend) × 3 (RSI) × 2 (Volatility) × 2 (Position) = 36` discrete
states — verified programmatically at `StateEncoder` construction time
(raises if the count ever drifts from 36). `StateEncoder.encode(state)`
/ `.decode(state_id)` give a deterministic, bijective mapping to
integer IDs `0..35` for tabular RL (e.g. Q-tables).

## 4. Action Space (`src/environment/actions.py`)

Exactly 3 actions: `0 = HOLD`, `1 = BUY`, `2 = SELL`. `is_valid_action()`
guards against anything else ever reaching the environment.

## 5. Trading Environment (`src/environment/trading_env.py`)

- **Initial capital**: ₹100,000 (`config/config.yaml: environment.initial_capital`, never hard-coded).
- **Transaction cost**: 0.1% (`config.yaml: environment.transaction_cost`), applied to BUY and SELL only, never HOLD.
- **BUY**: buys 1 virtual share if `cash >= price * (1 + transaction_cost)`; otherwise it's an invalid no-op (never allows negative cash).
- **SELL**: sells 1 virtual share if `shares > 0`; otherwise it's an invalid no-op (never allows negative shares).
- **HOLD**: cash/shares/position unchanged.
- **Invalid actions**: logged in `info["invalid_action"]` and penalized with a small, configurable `INVALID_ACTION_PENALTY = -0.001` added to the reward — small enough not to dominate normal trading rewards.
- **Portfolio value** = `Cash + Shares × Current Price`, recomputed every step.
- **Reward** = `(New Portfolio Value - Previous Portfolio Value) / Previous Portfolio Value`, using only information available after acting (the environment never looks ahead beyond the day it has just stepped into).
- **API**: `reset()`, `step(action) -> (next_state, reward, done, info)`, `get_state()`, `get_portfolio_value()`, `is_done()`.
- **Episode**: one chronological pass through whichever dataframe slice (train or test) it's constructed with — always forward, never shuffled, never backward. `done=True` when the last available observation is reached. The **same environment class** is used for both training and testing episodes; it never fits or refits feature thresholds itself — that only happens once, in `build_features.py`, using training data only.
- **Determinism**: given the same data, config, and action sequence, the environment always produces identical states/rewards/portfolio values (verified in tests) — no randomness lives in the environment; that belongs to Sprint 3's RL exploration policies.

Manual demo: `python scripts/test_environment.py` (runs BUY → HOLD →
SELL → full episode termination on 3 sample companies).

## 6. Data Leakage Prevention (Sprint 2 additions)

- Rolling features use only current + past rows (verified by a test
  that mutates a future price and asserts all earlier feature values
  are unchanged).
- The volatility LOW/HIGH threshold is fit **only** on training-period
  data per company, then reused unchanged for testing.
- The environment consumes pre-computed features and never fits
  anything itself, so pointing it at a test-period slice for Sprint 5
  evaluation cannot leak into feature engineering.
- Reward at step `t` depends only on prices at day `t` and day `t+1` —
  never anything further ahead.
- No shuffling anywhere: feature computation, chronological split
  (Sprint 1), and environment stepping all preserve date order.

## How to Run Sprint 2

```bash
pip install -r requirements.txt
python -m src.features.build_features   # builds data/features/*.csv for all 50 companies
python scripts/test_environment.py      # manual environment demo
pytest                                  # 78 tests (32 Sprint 1 + 46 Sprint 2), 0 failures
```

## Sprint 2 Status

```
========================================
PHASE 1 — SPRINT 2 COMPLETE
========================================

Companies processed: 50

Features:
✓ MA5
✓ MA20
✓ RSI
✓ Volatility
✓ Trend

RL State:
36 possible states

Actions:
✓ BUY
✓ HOLD
✓ SELL

Initial Capital:
₹100,000

Transaction Cost:
0.1%

Reward:
Portfolio-value based

Environment:
✓ Working

Tests:
✓ PASS (78/78)

Ready for:
SPRINT 3 — RL Algorithms
========================================
```

# Phase 1 — Sprint 3

Sprint 3 adds three classical, **tabular**, model-free reinforcement
learning algorithms on top of the Sprint 2 environment. No new
environment was created — `src/environment/trading_env.py`,
`src/environment/state.py` (36 states) and `src/environment/actions.py`
(3 actions) are reused unchanged by all three algorithms, so results are
directly comparable.

## Algorithms

### Q-Learning (`src/rl/q_learning.py`)
Off-policy temporal-difference control. Bootstraps off the **best**
possible next action regardless of what the exploring policy actually
does next:

```
Q(s,a) <- Q(s,a) + alpha [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
```

### SARSA (`src/rl/sarsa.py`)
On-policy temporal-difference control. Bootstraps off the **actual**
next action chosen by the (epsilon-greedy) policy:

```
Q(s,a) <- Q(s,a) + alpha [ r + gamma * Q(s',a') - Q(s,a) ]
```

### Monte Carlo Control (`src/rl/monte_carlo.py`)
Learns only from **complete episodes** and their observed returns — no
bootstrapping/no per-step update. Each episode is played out fully,
walked backward to compute the discounted return `G_t`, and every
state-action pair's **first** occurrence in the episode is updated
towards `G_t` using an incremental running average (equivalent to
averaging all first-visit returns seen so far):

```
G_t = r_(t+1) + gamma*r_(t+2) + gamma^2*r_(t+3) + ...
N(s,a) <- N(s,a) + 1
Q(s,a) <- Q(s,a) + (G_t - Q(s,a)) / N(s,a)
```

### Shared base class (`src/rl/base_agent.py`)
All three algorithms subclass `BaseAgent`, which holds only what is
genuinely common: the `(36, 3)` Q-table, hyperparameters, epsilon-greedy
`choose_action()` (with unbiased random tie-breaking), `decay_epsilon()`,
greedy `get_policy()`, and Q-table save/load. Each algorithm keeps its
own `train_episode()` with its own update rule, so the three stay
clearly separate.

## Configuration (`config/config.yaml` → `rl:`)

| Parameter | Default |
|---|---|
| Learning rate (alpha) | 0.1 |
| Discount factor (gamma) | 0.95 |
| Initial epsilon | 1.0 |
| Minimum epsilon | 0.01 |
| Epsilon decay (per episode) | 0.995 |
| Episodes per (company, algorithm) | 500 |
| Random seed | 42 |

Nothing is hard-coded — every training script reads these from
`config.yaml` (or a `--episodes` CLI override).

## Reward & environment consistency

All three algorithms train against the identical reward already defined
in the Sprint 2 environment:

```
reward = (new_portfolio_value - previous_portfolio_value) / previous_portfolio_value
```

and the identical `initial_capital` (₹100,000), `transaction_cost`
(0.1%), state space (36), action space (3), and training window
(2015-01-01 → 2024-12-31) — only the learning algorithm differs.
Testing-period data (2025-01-01 → 2026-07-31) is never read by Sprint 3.

## Training script

```bash
python scripts/train_model_free.py                       # all 50 companies, config.yaml episode count
python scripts/train_model_free.py --episodes 50          # override episode count
python scripts/train_model_free.py --companies RELI,INFY  # train a subset
python scripts/train_model_free.py --no-plots             # skip matplotlib plots (faster)
```

For each company this: loads its training-period feature slice, trains
Q-Learning → SARSA → Monte Carlo against fresh instances of the same
`TradingEnvironment` configuration, saves each Q-table + JSON metadata
under `models/<SYMBOL>/`, extracts and saves the greedy policy to
`reports/policies/`, appends per-episode metrics to
`reports/<algorithm>_training.csv`, and (unless `--no-plots`) saves
reward/portfolio/epsilon learning-curve plots to `reports/plots/`. A
company or algorithm that fails is logged to `reports/training_errors.csv`
and processing continues with the rest.

**A note on runtime:** each training episode steps through the full
~2,477-row training series one day at a time. At 500 episodes × 3
algorithms × 50 companies (= 75,000 episodes) this script is
compute-intensive (several hours on a single core). It has been
validated end-to-end with a reduced run
(`--episodes 25 --companies RELI,INFY,TCSc1_NS`) that produced correct
models, metadata, policies, reports and plots in under two minutes — the
full 50-company × 500-episode run is left for the user to kick off with
more time/compute via the plain `python scripts/train_model_free.py`
command above, or a reduced `--episodes` value.

Model storage layout:

```
models/
├── RELI/
│   ├── q_learning.pkl            + q_learning_metadata.json
│   ├── sarsa.pkl                 + sarsa_metadata.json
│   └── monte_carlo.pkl           + monte_carlo_metadata.json
├── INFY/
│   └── ...
└── ...
```

(Company folders use the project's existing `Symbol` convention — e.g.
`RELI`, `INFY`, `TCSc1_NS` — the same symbols already used by
`data/train/`, `data/test/` and `data/features/`.)

## How to run Sprint 3

```bash
pip install -r requirements.txt
python scripts/train_model_free.py --episodes 25 --companies RELI,INFY   # quick smoke test
pytest                                  # 127 tests (32 Sprint 1 + 46 Sprint 2 + 49 Sprint 3), 0 failures
```

## Sprint 3 acceptance criteria

```
[x] Q-Learning, SARSA, Monte Carlo Control implemented, each in its own file
[x] Q-table shape = (36, 3) for all three
[x] Epsilon-greedy action selection with unbiased tie-breaking
[x] Epsilon decay, tracked per episode
[x] Q-Learning / SARSA / Monte Carlo update rules verified independently
[x] Same reward, same environment, same 36 states / 3 actions for all three
[x] Training restricted to 2015-01-01 -> 2024-12-31 only (no test-set leakage)
[x] Models + JSON metadata saved per company per algorithm
[x] Per-episode training reports (reward, portfolio value, trades, epsilon)
[x] Learning-curve plots (reward / portfolio value / epsilon)
[x] Greedy policy extracted and sanity-checked (actions always in {0,1,2})
[x] No negative cash / no negative shares (environment-enforced, tested)
[x] Per-company/per-algorithm error handling -> reports/training_errors.csv
[x] 127/127 tests passing, Sprint 1 & 2 tests untouched
[x] No live API, no dashboard, no Value/Policy Iteration (Sprint 4 scope)
```

# Phase 1 — Sprint 4: Value Iteration + Policy Iteration

Sprint 4 adds two **model-based Dynamic Programming** algorithms —
Value Iteration and Policy Iteration — on top of the same 36-state /
3-action `TradingEnvironment` used by Sprint 3's Q-Learning, SARSA and
Monte Carlo Control. Unlike those three (which are **model-free** and
learn purely from environment interaction, see Sprint 3 above), Value
Iteration and Policy Iteration are **model-based**: they first build an
explicit MDP — a transition model `P(s'|s,a)` and reward model
`R(s,a)` — from TRAINING-period data, then solve that MDP directly with
the classical Bellman equations. Neither is a neural network; both are
plain tabular Dynamic Programming implemented with Python + NumPy only
(no TensorFlow/PyTorch/Stable-Baselines3).

> Value Iteration directly optimizes state values, while Policy
> Iteration alternates between evaluating and improving a policy.

## The MDP (`src/rl/mdp.py`)

`build_mdp()` estimates `P(s'|s,a)` and `R(s,a)` empirically, purely
from a company's TRAINING-period feature dataframe (2015-01-01 →
2024-12-31 — the same window Sprint 3 uses, and the testing period
2025-01-01 → 2026-07-31 is never read by this module). For every
training day *t* and both possible positions (`NO_POSITION`/`HOLDING`),
it applies each of the 3 actions using the identical invalid-action
rules as `TradingEnvironment._apply_action` (BUY never creates negative
cash, SELL never creates negative shares), marks the result to market
at day *t+1*'s price, and computes reward with the SAME formula as
Sprint 2:

```
Reward = (New Portfolio Value − Previous Portfolio Value) / Previous Portfolio Value
```

Because the 36-state abstraction discards information (exact date,
cash, share count beyond the binary Position flag), the same encoded
state can be followed by different outcomes on different historical
days — so `P` and `R` are estimated as empirical frequencies/averages
over every occurrence observed in the training data. This estimation
process itself never draws a random number, so it is fully
deterministic given the same training data. A `(state, action)` pair
never observed in training falls back to a self-loop with zero reward.
See the module docstring in `src/rl/mdp.py` for the full derivation.

## Value Iteration (`src/rl/value_iteration.py`)

Applies the Bellman optimality update synchronously to all 36 states
each sweep:

```
V(s) ← max_a [ R(s,a) + γ Σ P(s'|s,a) V(s') ]
```

until the maximum change across all states in one sweep drops below
`THETA`, or `MAX_ITERATIONS` sweeps are reached. The greedy policy is
then read off the converged `V`.

## Policy Iteration (`src/rl/policy_iteration.py`)

Alternates:

- **Policy Evaluation** — for the current policy π, iterate
  `V(s) ← R(s, π(s)) + γ Σ P(s'|s, π(s)) V(s')` until
  `max|ΔV| < THETA`.
- **Policy Improvement** — for every state, set
  `π(s) ← argmax_a [ R(s,a) + γ Σ P(s'|s,a) V(s') ]`.

Evaluation and improvement repeat until no state's action changes
(`policy_stable = True`). The initial policy is configurable
(`dp.policy_init` in `config.yaml`): `"HOLD"` (default — HOLD for
every state) or `"RANDOM"` (a fixed-seed random valid action per
state) — neither uses any future information.

## Configuration (`config/config.yaml` → `dp:`)

```yaml
dp:
  gamma: 0.95            # same value as rl.gamma, so all 5 algorithms are comparable
  theta: 0.000001        # convergence tolerance
  max_iterations: 1000   # safety cap
  policy_init: "HOLD"    # "HOLD" or "RANDOM"
```

## Algorithm consistency (all 5 algorithms)

Value Iteration and Policy Iteration reuse, unmodified:

- The same `TradingEnvironment`, `StateEncoder` (36 states) and action
  space (3 actions) as Sprint 2/3.
- The same reward definition, transaction cost (0.1%) and initial
  capital (₹100,000) as Sprint 2/3.
- The same TRAINING window (2015-01-01 → 2024-12-31) — the testing
  period is never touched.
- Sprint 3's `src/rl/model_io.py` for model saving/loading and policy
  CSV export (extended with two new algorithm entries and two new,
  additive helper functions — `build_dp_metadata`,
  `extract_dp_policy_rows` — rather than modifying the Sprint 3
  functions that Q-Learning/SARSA/Monte Carlo already rely on).

This is required so Sprint 5 can fairly compare all five algorithms —
Value Iteration and Policy Iteration are never given an easier
environment or a different reward to make them look better.

## Model / report storage

```
models/
├── RELIANCE/
│   ├── q_learning.pkl / .json          (Sprint 3)
│   ├── sarsa.pkl / .json               (Sprint 3)
│   ├── monte_carlo.pkl / .json         (Sprint 3)
│   ├── value_iteration.pkl / .json     (Sprint 4)
│   └── policy_iteration.pkl / .json    (Sprint 4)
└── ...

reports/
├── value_iteration_training.csv    # Company, Iteration, Max_Value_Change, Converged
├── policy_iteration_training.csv   # Company, Policy_Iteration, Evaluation_Iterations, Policy_Stable
├── dp_training_errors.csv
└── policies/
    ├── <SYMBOL>_value_iteration_policy.csv    # State_ID, Trend, RSI, Volatility, Position, Value, Best_Action
    └── <SYMBOL>_policy_iteration_policy.csv
```

Sprint 3's `models/<SYMBOL>/q_learning*`, `sarsa*` and `monte_carlo*`
files are never overwritten — Value Iteration and Policy Iteration
write new, separate files alongside them.

## Training script

```bash
python scripts/train_dynamic_programming.py                       # all 50 companies
python scripts/train_dynamic_programming.py --companies RELI,INFY  # train a subset
```

For each company this: loads its training-period feature slice,
builds one MDP, runs Value Iteration, runs Policy Iteration, saves
both models + JSON metadata under `models/<SYMBOL>/`, saves each
algorithm's policy table to `reports/policies/`, and appends
convergence rows to `reports/value_iteration_training.csv` /
`reports/policy_iteration_training.csv`. A company or algorithm that
fails is logged to `reports/dp_training_errors.csv` and processing
continues with the rest. Verified end-to-end on `--companies RELI,INFY`:
both algorithms converged in well under 2 seconds total (Value
Iteration and Policy Iteration reached identical optimal policies, as
expected for two algorithms solving the same MDP exactly) — the full
50-company run is left for the user to kick off with the command
above.

## How to run Sprint 4

```bash
pip install -r requirements.txt
python scripts/train_dynamic_programming.py --companies RELI,INFY   # quick smoke test
pytest                                  # 159 tests (32 Sprint 1 + 46 Sprint 2 + 49 Sprint 3 + 32 Sprint 4), 0 failures
```

## Sprint 4 acceptance criteria

```
[x] Sprint 1 still works
[x] Sprint 2 still works
[x] Sprint 3 still works
[x] MDP implemented (src/rl/mdp.py), training-data only, no future leakage
[x] 36 states confirmed
[x] 3 actions confirmed
[x] Value Iteration implemented (src/rl/value_iteration.py)
[x] Policy Iteration implemented (src/rl/policy_iteration.py)
[x] Value Iteration converges
[x] Policy Iteration converges
[x] Training uses only 2015–2024
[x] Testing data remains untouched
[x] Policies saved (per company; validated end-to-end on a subset — see above)
[x] Model metadata saved
[x] Convergence reports created
[x] Unit tests created (tests/test_sprint4.py)
[x] pytest passes (159/159)
[x] README updated
[x] No live API
[x] No real trading
[x] No dashboard
[x] Ready for Sprint 5
```

# Phase 1 — Complete

Phase 1 is now complete: data → features → environment → five RL
algorithms → backtesting → reporting → dashboard, entirely on
historical data and virtual money. No live market data, no real
trades, no Phase 2 functionality.

## Dataset

- **50 companies**, daily OHLCV data.
- **Historical period**: 01-01-2015 → 31-07-2026.
- **Training**: 01-01-2015 → 31-12-2024 (used by Sprints 3-4 only).
- **Testing**: 01-01-2025 → 31-07-2026 (used by Sprint 5 only — never
  seen during training, never used to retrain, update, or select a
  model).

## Algorithms

1. Q-Learning (model-free, off-policy)
2. SARSA (model-free, on-policy)
3. Monte Carlo Control (model-free, episodic)
4. Value Iteration (model-based Dynamic Programming)
5. Policy Iteration (model-based Dynamic Programming)

Actions: **BUY / HOLD / SELL** (3 actions), over a shared **36-state**
encoding (Trend × RSI × Volatility × Position). All five algorithms
train and are evaluated on the identical environment, reward,
transaction cost and initial capital, so their results are directly
comparable.

## Environment

₹100,000 initial virtual capital, 0.1% transaction cost per trade,
reward = portfolio return per step (plus a small penalty for an
invalid action). See the Sprint 2 section above for full details.

## Sprint 5 — Backtesting, comparison, dashboard

**`src/backtesting/`** (new in Sprint 5):
- `model_loader.py` — loads any of the 5 trained models uniformly (all
  five pickles share a `q_table` shape, so one `argmax` derives the
  greedy policy for all of them) and never mutates it.
- `backtester.py` — runs a frozen policy through the existing
  `TradingEnvironment` over the TEST period only. Evaluation only: no
  Q-table update, no policy update, no retraining is possible.
- `buy_and_hold.py` — the passive baseline: buy the maximum affordable
  whole shares on day 1 of the test period, then hold.
- `metrics.py` — Total Return, Annualized Return, Sharpe Ratio, Maximum
  Drawdown, Win Rate — pure functions, same formula for every algorithm
  and for Buy & Hold, documented divide-by-zero conventions so no
  metric is ever NaN or infinite.
- `plots.py` — per-company portfolio/drawdown charts (all 5 algorithms
  + Buy & Hold overlaid) and aggregate algorithm-comparison bar charts.

**Evaluation methodology**: for every company, every algorithm's
trained model is loaded, its policy is derived, and it is run once,
chronologically, over the test period. A **trade** is a valid,
executed BUY or SELL (HOLD never counts). A **completed trade** for
Win Rate purposes is a SELL matched FIFO against an earlier BUY —
a win if the SELL price exceeds the matched BUY price. Sharpe Ratio
uses daily returns, `risk_free_rate = 0.0` (see `config.yaml`'s
`backtest:` section — documented, not silently assumed), annualized
with `sqrt(252)`.

**Reports** (`scripts/run_backtest.py`):

```
reports/
├── final_results.csv            # every (company, algorithm) incl. Buy & Hold
├── company_best_algorithm.csv   # best-by-return AND best-by-Sharpe, per company
├── algorithm_comparison.csv     # average metrics per algorithm, across all companies tested
├── algorithm_ranking.csv        # separate rankings by return / Sharpe / drawdown - no combined score
├── aggregate_analysis.csv       # mean + median across companies, + "beats Buy & Hold" counts
├── backtest_errors.csv          # any company/algorithm that failed, with reason
├── phase1_final_report.md       # generated from the ACTUAL results of the run - no pre-decided winner
├── trades/<COMPANY>_<algo>_trades.csv
├── portfolio/<COMPANY>_<algo>_portfolio.csv
└── plots/
    ├── <COMPANY>_portfolio_comparison.png
    ├── <COMPANY>_drawdown_comparison.png
    └── algorithm_{return,sharpe,drawdown,win_rate}_comparison.png
```

**Dashboard** (`app.py`, Streamlit): a pure reader of the CSVs above —
it never re-runs a backtest or a model. Sidebar lets you pick a
Company and an Algorithm. Four pages: **Overview** (best-by-return /
best-by-Sharpe, shown separately, never combined into one "winner"),
**Stock Analysis** (KPIs, latest backtested action clearly labeled
"Backtest Decision" — never presented as a live signal — and a
portfolio-value chart), **Algorithm Comparison** (per-company table,
all-algorithms-vs-Buy&Hold chart, the 50-company comparison table),
and **Backtest Details** (full trade log + portfolio history). A
disclaimer banner ("uses historical backtesting and virtual money...")
is always visible.

## How to run

```bash
pip install -r requirements.txt

# 1. Train (Sprints 3-4) - full run trains all 50 companies; add
#    --companies X,Y to train a subset first for a quick check.
python scripts/train_model_free.py
python scripts/train_dynamic_programming.py

# 2. Backtest + generate all reports/charts (Sprint 5)
python scripts/run_backtest.py

# 3. Explore the results
streamlit run app.py
```

**Windows users**: run each command separately (not chained with `&&`
in classic PowerShell), and use `.venv\Scripts\Activate.ps1` to
activate the virtual environment. If `pip install` complains about
permissions or `python`/`pip` mismatch, use `python -m pip install -r
requirements.txt` instead. All console output and generated reports
are UTF-8-safe (the scripts reconfigure stdout to UTF-8 and write
files with explicit `encoding="utf-8"`), so the ₹/→/✓ characters in
progress output and `reports/phase1_final_report.md` won't crash the
run even on a legacy `cp1252` console.

**Verified end-to-end** on a subset of companies (models trained with
`--episodes 50`/`--companies RELI`, since a full 50-company ×
500-episode training run takes hours of CPU time): `run_backtest.py`
produced every report/chart above with 0 entries in
`backtest_errors.csv`, and `streamlit run app.py` served the dashboard
successfully (HTTP 200, all four pages). All 193 tests pass
(`pytest`) — 32 Sprint 1 + 46 Sprint 2 + 49 Sprint 3 + 32 Sprint 4 + 34
Sprint 5, 0 failures. Run the two training scripts on all 50 companies
(no `--companies` filter) for the full-scale, 250-backtest result the
spec describes.

## Limitations (see `reports/phase1_final_report.md` for the generated version)

- Fills assumed at recorded Close price; no slippage or partial fills.
- The MDP behind Value Iteration / Policy Iteration is an empirical
  approximation from training data (see `src/rl/mdp.py`), not a true
  stationary Markov model of the market.
- Historical backtest performance does not guarantee future results.

## Phase 1 acceptance criteria

```
[x] 50-company pipeline in place (backtest a subset end-to-end; run without
    --companies for the full 50 × 5 = 250 RL backtests + 50 Buy & Hold)
[x] 5 RL algorithms tested (Q-Learning, SARSA, Monte Carlo, Value Iteration, Policy Iteration)
[x] Buy & Hold benchmark completed
[x] Testing period = 01-01-2025 to 31-07-2026
[x] No model retraining during testing (backtester is read-only; policy arrays are write-protected)
[x] No data leakage (test data loaded independently from training; verified by tests)
[x] Return / Sharpe / Max Drawdown / Win Rate calculated (metrics.py, unit-tested)
[x] Trade logs generated (reports/trades/)
[x] Portfolio histories generated (reports/portfolio/)
[x] Comparison reports generated (final_results, algorithm_comparison, company_best_algorithm, aggregate_analysis)
[x] Algorithm ranking generated (separate rankings, no combined score)
[x] Charts generated (reports/plots/)
[x] Streamlit dashboard created (app.py) - simple, 4 pages, clearly labeled as backtesting
[x] Unit tests created (tests/test_sprint5.py, 34 tests)
[x] pytest passes (193/193)
[x] README updated
[x] Final Phase 1 report generator created (reports/phase1_final_report.md, built from real run results)
[x] No live API, no real trading, no Phase 2 functionality
```

Phase 2 (not implemented here) will add live market data, real-time
state, RL-driven decisions, and paper trading.

# Dashboard Redesign — User-Friendly AI Decision

The Streamlit dashboard (`app.py`) was redesigned so a normal user
never needs to understand RL to use it. The goal changed from "compare
5 algorithms" to "tell me BUY, HOLD, or SELL for one company."

## What changed

- **No algorithm selection.** The sidebar now only has a Company
  selector, three simple pages, and a hidden Developer Mode toggle.
- **One final decision, not five.** `src/decision/engine.py` (new)
  loads every trained model for the selected company, gets each one's
  own greedy action, and combines them by **majority vote** into a
  single BUY/HOLD/SELL — this is the only thing a normal user sees.
- **All 5 algorithms still run internally** — nothing from Sprints
  1–5 was retrained, removed, or had its train/test split changed.

## Ensemble / voting logic (`src/decision/engine.py`)

- **Vote**: each of the 5 algorithms proposes its own action for its
  own current state (see "Position assumption" below).
- **Tie-break** (only possible as a 2-2-1 split among 5 votes):
  - If HOLD is one of the tied top actions → **HOLD wins** (no
    transaction cost, no execution risk — the conservative default
    when the models disagree).
  - If the tie is BUY vs SELL only → the action with the **higher
    average Q-value** across the tied algorithms' own Q-tables wins
    (reuses the same Bellman values Sprints 3–4 already computed,
    instead of a coin flip).
- **Confidence** = (votes for the winning action / number of
  algorithms that voted) × 100 — reported at its true value even when
  a tie-break decided *which* action won.
  - 80–100% → **High**, 60–79% → **Medium**, below 60% → **Low**.
- **Position assumption**: each algorithm needs a Position bit
  (holding shares or not) to look up its state. This uses that
  algorithm's own **most recent simulated position from its Sprint 5
  backtest trade log** (i.e. "if this algorithm had been trading this
  stock through the test period, what position is it in today, and
  what would it do next?"), falling back to "not holding" only if no
  backtest exists yet for that company/algorithm.
- If fewer than 5 algorithms have a trained model for a company, the
  same rules apply over however many did vote (never silently padded
  to 5).

## New user-facing pages

1. **Stock Decision** (default) — latest available price/date, one
   large color-coded BUY/HOLD/SELL card with confidence, 3–5
   plain-language reasons, simple Price Trend / Momentum / Volatility
   / RSI indicators (RSI text uses the same 30/70 oversold/overbought
   thresholds as Sprint 2's own RSI classification), a price chart
   with selectable time ranges, and a decision summary.
2. **Stock History** — price chart, an illustrative day-by-day AI
   decision table (a simplified, position-agnostic recomputation —
   documented as different from the audited backtest), and a
   "How the AI Performed Historically" panel averaged across the 5
   algorithms' real Sprint 5 backtest results, clearly labeled
   "Historical simulation only."
3. **About** — plain-language explanation of the pipeline; states
   clearly that this uses virtual money and historical data only.

Historical data is always labeled **"Latest Available Price"** /
**"Latest Available Data Date"** — never "live" (Phase 2 territory).

## Developer Mode (hidden by default)

A single sidebar checkbox, unchecked by default, reveals a 4th page —
**Developer Analysis** — showing exactly what a normal user never
sees: each algorithm's name, its individual vote and full Q-values for
the current decision, the tie-break/confidence mechanics, and the full
Sprint 5 metrics (Sharpe Ratio, Drawdown, Win Rate, per-algorithm
comparison). This preserves the academic requirement that all 5 RL
algorithms are visibly implemented, without surfacing that complexity
to a normal user.

## Tests

`tests/test_decision_engine.py` (26 tests, new) covers vote counting,
both tie-break cases, confidence-label boundaries, the position
fallback, the simple-indicator classifiers, and `get_decision()`
end-to-end against a small fake model/feature setup (including the
case where only some algorithms are trained). All 219 tests pass
(193 from Sprints 1–5 + 26 new) — nothing in Sprints 1–5 was changed
except one additive helper (`load_q_table`) in
`src/backtesting/model_loader.py`, factored out of the existing
`load_greedy_policy` so the ensemble layer could reuse the exact same
loading/validation code instead of duplicating it.

**Verified end-to-end**: trained real models for 2 companies, ran the
full backtest pipeline, then called every dashboard page's render
function directly against real data (including a company with zero
trained models, to confirm graceful "not yet available" messaging
instead of a crash) — no errors in any case. `streamlit run app.py`
also verified serving successfully (HTTP 200).
#   R l  
 