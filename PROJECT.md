# Adaptive Statistical Arbitrage in Indian Equity Markets

## 1. Project Purpose

This is a personal quantitative-finance / financial-engineering portfolio project.

The purpose is to build a small and reliable statistical-arbitrage system for Indian equities and use it to understand:

- pairs trading
- correlation
- cointegration
- stationarity
- ADF testing
- mean reversion
- hedge ratios
- OLS regression
- rolling regression
- Kalman filtering
- z-score trading signals
- transaction costs
- backtesting
- risk-adjusted performance
- out-of-sample evaluation

The main comparison is:

**Static OLS vs Rolling OLS vs Kalman Filter hedge ratios**

The final project will provide a Streamlit interface for:

1. Pair Scanner
2. Pair Analyzer
3. Backtest & Performance

---

# 2. Important Project Boundary

This project is intentionally small.

It is NOT:

- an internship task
- a job task
- a production trading platform
- a live trading system
- a broker-connected system
- a commercial trading application
- a research publication
- a high-frequency trading system

The project should be completed in a few days to a few weeks.

Do not add complexity merely to make the project appear sophisticated.

If a feature:

- significantly increases implementation complexity,
- creates additional failure points,
- and does not provide meaningful quantitative-finance learning,

DO NOT implement it.

A smaller working project is preferred over a large unfinished project.

---

# 3. Final Technology Stack

## Programming

Python

## Data Source

Yahoo Finance using:

`yfinance`

## Data Storage

Parquet

CSV is not part of the planned data pipeline.

## Application

Streamlit

## Statistical / Numerical Libraries

Use only libraries that are genuinely required.

Expected libraries include:

- pandas
- numpy
- scipy
- statsmodels
- yfinance
- a suitable Kalman/state-space implementation
- matplotlib and/or plotly if required for visualization

## Testing

pytest

Do not add unnecessary dependencies.

---

# 4. Historical Data

Target historical dataset:

**2016 through September 2026**

Main modelling periods:

| Period | Purpose |
|---|---|
| 2016–2023 | Training / model estimation |
| 2024–2025 | Primary out-of-sample test |
| Jan–Sep 2026 | Current/demo period |

The primary performance results must come from the **2024–2025 out-of-sample period**.

2026 is only an additional current/demo period.

---

# 5. Stock Universe

The project uses exactly 50 stocks across six sectors.

## Banking & Financial Services

- HDFC Bank — HDFCBANK.NS
- ICICI Bank — ICICIBANK.NS
- State Bank of India — SBIN.NS
- Axis Bank — AXISBANK.NS
- Kotak Mahindra Bank — KOTAKBANK.NS
- IndusInd Bank — INDUSINDBK.NS
- Bank of Baroda — BANKBARODA.NS
- Punjab National Bank — PNB.NS
- Bajaj Finance — BAJFINANCE.NS

## IT

- TCS — TCS.NS
- Infosys — INFY.NS
- HCL Technologies — HCLTECH.NS
- Wipro — WIPRO.NS
- Tech Mahindra — TECHM.NS
- LTIMindtree — LTIM.NS
- Mphasis — MPHASIS.NS
- Persistent Systems — PERSISTENT.NS

## Automobile

- Maruti Suzuki — MARUTI.NS
- Mahindra & Mahindra — M&M.NS
- Tata Motors — TATAMOTORS.NS
- Bajaj Auto — BAJAJ-AUTO.NS
- Eicher Motors — EICHERMOT.NS
- Hero MotoCorp — HEROMOTOCO.NS
- TVS Motor — TVSMOTOR.NS
- Ashok Leyland — ASHOKLEY.NS

## FMCG / Consumer

- Hindustan Unilever — HINDUNILVR.NS
- ITC — ITC.NS
- Nestlé India — NESTLEIND.NS
- Britannia Industries — BRITANNIA.NS
- Tata Consumer Products — TATACONSUM.NS
- Dabur India — DABUR.NS
- Godrej Consumer Products — GODREJCP.NS
- Marico — MARICO.NS

## Energy & Industrials

- Reliance Industries — RELIANCE.NS
- ONGC — ONGC.NS
- Coal India — COALINDIA.NS
- NTPC — NTPC.NS
- Power Grid — POWERGRID.NS
- Bharat Petroleum — BPCL.NS
- Indian Oil — IOC.NS
- GAIL — GAIL.NS
- Larsen & Toubro — LT.NS

## Pharmaceuticals

- Sun Pharma — SUNPHARMA.NS
- Dr. Reddy's Laboratories — DRREDDY.NS
- Cipla — CIPLA.NS
- Divi's Laboratories — DIVISLAB.NS
- Lupin — LUPIN.NS
- Aurobindo Pharma — AUROPHARMA.NS
- Apollo Hospitals — APOLLOHOSP.NS
- Torrent Pharma — TORNTPHARM.NS

---

# 6. Data Validation Requirement

Before performing quantitative analysis, validate that each ticker provides sufficient historical data for the required period.

The system must check:

- ticker download success
- available date range
- number of observations
- missing values
- duplicate dates
- basic data integrity

Do not silently replace a stock if its history is insufficient.

If a stock fails validation, report the problem clearly.

---

# 7. Data Pipeline

The data flow must be:

Yahoo Finance
↓
`download_data.py`
↓
Local Parquet dataset
↓
`src/data/`
↓
Quantitative analysis
↓
Streamlit

Streamlit must NOT download data from Yahoo Finance whenever the application starts.

The application reads locally stored Parquet data.

---

# 8. Market Data

The project primarily requires:

- Date
- Adjusted price / adjusted close
- Volume

Other OHLC fields may be retained if returned by Yahoo Finance, but they are not central to the statistical-arbitrage calculations.

The implementation should use a consistent adjusted-price convention throughout the project.

---

# 9. GitHub Data Decision

The decision about committing the downloaded market-data file to GitHub is intentionally left open.

For now:

- keep the data under `data/`
- keep `.gitignore` configured appropriately
- maintain the download script
- do not depend on a remote database

The final decision can be made after inspecting the actual Parquet file size and practical redistribution considerations.

---

# 10. Pair Generation

Pairs are generated only within the same sector.

For example:

9 banking stocks

9 choose 2

= 36 possible pairs

The same logic applies to the other sectors.

No cross-sector pairs are required.

---

# 11. Pair Scanner

The Pair Scanner begins with a selected sector.

Workflow:

1. User selects sector.
2. System retrieves stocks belonging to that sector.
3. System generates every possible pair.
4. System validates that both stocks have usable overlapping data.
5. System calculates pair statistics.
6. System displays the candidate pairs.
7. User can select a pair for deeper analysis.

The scanner should calculate relevant statistics such as:

- correlation
- cointegration p-value
- ADF p-value
- half-life
- OLS hedge ratio
- Kalman hedge ratio
- current z-score
- number of observations

Do NOT create an artificial composite pair score.

The user should see the actual statistics and be able to sort/filter them.

---

# 12. Correlation

Correlation is used as a descriptive measure and an initial relationship filter.

Correlation alone does not establish that a pair is suitable for statistical arbitrage.

The implementation should distinguish:

**Correlation**

from

**Cointegration**

---

# 13. Cointegration

Use the Engle-Granger approach for the initial implementation.

The purpose is to identify pairs for which a linear combination of the two price series can exhibit stationary behaviour.

The implementation should calculate and expose the cointegration test result and p-value.

Do not implement Johansen cointegration in this project.

---

# 14. ADF Test

Use the Augmented Dickey-Fuller test to test the stationarity of the relevant spread/residual series.

The implementation should provide:

- ADF statistic
- p-value
- interpretation

The project should clearly explain why stationarity matters for a mean-reversion strategy.

No additional stationarity tests are required for the initial version.

---

# 15. Half-Life

Estimate the mean-reversion half-life of the spread.

Half-life is used to understand approximately how quickly deviations from the relationship tend to decay.

The implementation should be mathematically transparent.

If a half-life cannot be reliably estimated, return an informative result instead of allowing the application to fail.

---

# 16. Hedge-Ratio Models

Exactly three hedge-ratio methods are required.

## 16.1 Static OLS

Estimate:

Y_t = alpha + beta X_t + epsilon_t

where beta is the hedge ratio.

The hedge ratio is estimated from the training period and remains fixed for the corresponding backtest.

---

## 16.2 Rolling OLS

Estimate the hedge ratio repeatedly using a rolling historical window.

The hedge ratio therefore changes over time.

The rolling window must be configurable.

Use one sensible default.

Do not expose many unnecessary parameters.

---

## 16.3 Kalman Filter

Use a simple state-space model:

Y_t = alpha_t + beta_t X_t + epsilon_t

Both:

- alpha_t
- beta_t

are dynamic.

The Kalman filter should estimate the changing relationship through time.

Do not build a complicated custom filtering framework.

Use a reliable implementation/library.

---

# 17. Spread

For the Kalman model:

Spread_t = Y_t - alpha_t - beta_t X_t

For Static OLS and Rolling OLS, construct the corresponding spread using their estimated parameters.

The spread is used to construct the mean-reversion signal.

---

# 18. Z-Score

Use:

Z_t = (Spread_t - mean(Spread)) / std(Spread)

The exact rolling/static estimation method must be implemented consistently.

Most importantly:

**Do not use future information when generating historical trading signals.**

---

# 19. Trading Rules

The initial strategy uses only two thresholds.

## Entry

Enter when:

|Z-score| > 2

## Exit

Exit when:

|Z-score| < 0.5

## Stop Loss

No stop-loss is implemented.

Do not add a stop-loss in the initial version.

---

# 20. Position Logic

The position must be determined from the sign of the spread/z-score.

When the z-score is strongly positive:

- short the spread
- take the corresponding long/short position in the two securities

When the z-score is strongly negative:

- long the spread
- take the opposite long/short position

The exact implementation must remain consistent with the defined spread:

Spread = Y - beta X

Document the convention clearly in the code.

---

# 21. Position Sizing

Use fixed capital per pair.

Do not implement:

- volatility scaling
- Kelly criterion
- dynamic leverage
- portfolio optimization
- risk parity

The goal is transparent pairs-trading logic.

---

# 22. Transaction Costs

Use:

**10 basis points per side**

10 bps = 0.10%

The transaction cost must be configurable.

Apply transaction costs whenever the strategy executes the corresponding trade turnover.

This is a simplified assumption.

It does NOT attempt to model every:

- brokerage charge
- exchange fee
- STT
- GST
- slippage
- bid-ask spread

separately.

---

# 23. Out-of-Sample Backtesting

The primary backtest is:

Training:

2016–2023

Testing:

2024–2025

The test period must remain unseen when estimating the model parameters required for the strategy.

Avoid:

- look-ahead bias
- future-data leakage
- using the entire 2016–2025 dataset to calculate training parameters
- using future spread statistics for earlier signals

The 2024–2025 results are the main results reported by the project.

---

# 24. 2026 Demo Period

January–September 2026 may be displayed as a current/demo period.

It must not replace the primary 2024–2025 OOS evaluation.

The application should clearly distinguish:

- Training
- OOS Test
- 2026 Demo

---

# 25. Backtesting Metrics

Calculate:

## Returns

- cumulative return
- annualized return

## Risk

- volatility
- Sharpe ratio
- Sortino ratio
- maximum drawdown

## Trading

- number of trades
- average holding period
- turnover
- transaction costs

Do not add dozens of additional metrics.

---

# 26. Streamlit Application

The application has exactly three pages.

## Page 1 — Pair Scanner

Purpose:

Scan and inspect pairs within a selected sector.

User flow:

Select sector
↓
Generate pairs
↓
Calculate statistics
↓
Filter/sort
↓
Select pair

Display:

- pair
- correlation
- cointegration p-value
- ADF p-value
- half-life
- hedge ratios
- current z-score
- observations

---

## Page 2 — Pair Analyzer

Purpose:

Perform deeper analysis of one selected pair.

Display:

- Stock A
- Stock B
- correlation
- cointegration p-value
- ADF p-value
- half-life
- Static OLS hedge ratio
- Rolling OLS hedge ratio
- Kalman hedge ratio
- current spread
- current z-score

Charts:

1. Normalized prices
2. Spread
3. Z-score
4. Kalman hedge ratio

---

## Page 3 — Backtest & Performance

Purpose:

Run and compare the three approaches on the selected pair.

Allow selection of:

- pair
- hedge-ratio method

Display:

- equity curve
- cumulative return
- annualized return
- Sharpe
- Sortino
- volatility
- maximum drawdown
- number of trades
- average holding period
- turnover
- transaction costs

Clearly state that the main evaluation period is:

**2024–2025 Out-of-Sample**

---

# 27. Project Structure

The current project structure is:

adaptive-statistical-arbitrage/

├── app/
│   ├── pages/
│   │   ├── 1_Pair_Scanner.py
│   │   ├── 2_Pair_Analyzer.py
│   │   └── 3_Backtest_Performance.py
│   └── streamlit_app.py
│
├── config/
│   └── config.py
│
├── data/
│   └── .gitkeep
│
├── scripts/
│   └── download_data.py
│
├── src/
│   ├── backtesting/
│   ├── data/
│   ├── hedge_ratios/
│   ├── metrics/
│   ├── pairs/
│   ├── signals/
│   └── statistics/
│
├── tests/
│
├── .gitignore
├── PROJECT.md
├── README.md
└── requirements.txt

No notebooks directory is required.

---

# 28. Implementation Phases

IMPORTANT:

Do NOT implement the entire project at once.

Complete one phase, test it, verify it, and only then proceed to the next phase.

---

# PHASE 1 — Environment and Project Setup

## Objective

Make sure the project structure and Python environment are working.

Tasks:

1. Verify Python environment.
2. Install requirements.
3. Verify imports.
4. Verify project directories.
5. Verify configuration loading.
6. Verify Streamlit can start.
7. Verify pytest can run.

Expected result:

The project runs without import errors.

Do not implement quantitative logic yet.

---

# PHASE 2 — Configuration

## Objective

Centralize project constants.

Add configuration for:

- stock universe
- sector mapping
- training start date
- training end date
- test start date
- test end date
- demo start date
- demo end date
- entry z-score
- exit z-score
- transaction cost
- fixed capital
- rolling window

Current values:

Training:

2016-01-01 to 2023-12-31

Test:

2024-01-01 to 2025-12-31

Demo:

2026-01-01 to 2026-09-30

Entry:

2.0

Exit:

0.5

Transaction cost:

0.001 per side

The configuration should not contain unnecessary parameters.

---

# PHASE 3 — Historical Data Download

## Objective

Build the local Parquet dataset.

Implement:

1. Load the 50 tickers.
2. Download historical data using yfinance.
3. Request the required historical period.
4. Validate successful downloads.
5. Validate dates.
6. Handle missing values.
7. Check duplicate dates.
8. Save data as Parquet.

Do NOT connect Streamlit to yfinance.

Expected result:

A valid local Parquet dataset containing the required historical data.

---

# PHASE 4 — Data Loading and Validation

## Objective

Create reusable functions in `src/data/`.

Implement functions for:

- loading the Parquet dataset
- retrieving one stock
- retrieving multiple stocks
- selecting date ranges
- aligning two securities
- checking sufficient observations
- handling missing values

Test the functions.

Expected result:

All later modules can obtain clean aligned price data through `src/data/`.

---

# PHASE 5 — Statistical Functions

## Objective

Build and test the fundamental quantitative functions.

Implement:

1. correlation
2. OLS regression
3. Engle-Granger cointegration
4. ADF test
5. half-life
6. z-score

Each function should:

- accept clean input
- return clear output
- handle insufficient data
- avoid unnecessary side effects

Write unit tests.

Expected result:

The statistical building blocks work independently.

---

# PHASE 6 — Pair Generation

## Objective

Generate all possible pairs within a selected sector.

Example:

9 stocks → 36 pairs.

Implement:

- sector lookup
- pair generation
- duplicate prevention
- self-pair prevention

Test that:

- no stock is paired with itself
- A/B and B/A are not duplicated
- the expected number of pairs is generated

Expected result:

A clean list of sector-specific stock pairs.

---

# PHASE 7 — Pair Scanner

## Objective

Calculate statistics for every generated pair.

For each pair calculate:

- observations
- correlation
- cointegration p-value
- ADF p-value
- half-life
- OLS hedge ratio
- current z-score

Return the results in a pandas DataFrame.

Do not create a composite score.

Expected result:

A table of candidate pairs that can be sorted and filtered.

---

# PHASE 8 — Static OLS Hedge Ratio

## Objective

Implement the baseline hedge-ratio model.

Use:

Y_t = alpha + beta X_t + epsilon_t

Estimate alpha and beta using the training data.

Use beta as the hedge ratio.

Construct the spread.

Expected result:

Static OLS spread and hedge ratio for a selected pair.

---

# PHASE 9 — Rolling OLS Hedge Ratio

## Objective

Implement adaptive OLS estimation.

Use a configurable rolling window.

At each point:

- use only historical observations available up to that point
- estimate alpha and beta
- calculate the current hedge ratio
- calculate the corresponding spread

Avoid look-ahead bias.

Expected result:

A time-varying rolling OLS hedge ratio.

---

# PHASE 10 — Kalman Filter Hedge Ratio

## Objective

Implement the dynamic hedge-ratio model.

Use:

Y_t = alpha_t + beta_t X_t + epsilon_t

Estimate:

- dynamic alpha
- dynamic beta

The Kalman model must use sequential information and must not use future observations when generating historical estimates.

Expected result:

Time-varying Kalman alpha and beta.

---

# PHASE 11 — Trading Signals

## Objective

Convert the spread into trading signals.

Implement:

Entry:

|z| > 2

Exit:

|z| < 0.5

No stop-loss.

Generate:

- signal
- position state
- entry events
- exit events

Ensure positions do not repeatedly open on every observation while the strategy is already in a trade.

Expected result:

A clean sequence of trading positions.

---

# PHASE 12 — Position and P&L Engine

## Objective

Calculate strategy returns.

Implement:

- fixed capital per pair
- long/short position calculation
- daily P&L
- cumulative P&L
- transaction costs
- turnover

Transaction cost:

10 bps per side.

Make sure transaction costs are applied only when the relevant position changes.

Expected result:

Daily strategy returns and equity curve.

---

# PHASE 13 — Backtesting Engine

## Objective

Create one reusable backtesting function that can run:

1. Static OLS
2. Rolling OLS
3. Kalman Filter

using the same pair and trading rules.

The backtesting engine should return:

- daily returns
- equity curve
- trades
- positions
- turnover
- transaction costs

Expected result:

The same pair can be backtested using all three methods.

---

# PHASE 14 — Performance Metrics

## Objective

Create reusable performance functions.

Calculate:

- cumulative return
- annualized return
- volatility
- Sharpe ratio
- Sortino ratio
- maximum drawdown
- number of trades
- average holding period
- turnover
- total transaction costs

Expected result:

A clean performance summary for every strategy.

---

# PHASE 15 — Pair Analyzer

## Objective

Build Streamlit Page 2.

The page should allow the user to select a pair and display:

- pair statistics
- normalized prices
- spread
- z-score
- Static OLS hedge ratio
- Rolling OLS hedge ratio
- Kalman hedge ratio

The page should reuse functions already created in `src/`.

Do not duplicate quantitative logic inside Streamlit.

---

# PHASE 16 — Pair Scanner UI

## Objective

Build Streamlit Page 1.

User selects a sector.

The application:

1. generates pairs
2. calculates statistics
3. displays results
4. allows sorting/filtering
5. allows selecting a pair

Do not place statistical calculations directly inside the UI code if they already exist in `src/`.

---

# PHASE 17 — Backtest & Performance UI

## Objective

Build Streamlit Page 3.

User selects:

- pair
- hedge-ratio method

The page runs the reusable backtest and displays:

- equity curve
- cumulative return
- annualized return
- Sharpe
- Sortino
- volatility
- maximum drawdown
- trades
- holding period
- turnover
- transaction costs

Clearly display:

**Out-of-Sample Period: 2024–2025**

---

# PHASE 18 — Testing

## Objective

Verify the complete system.

Test:

- data loading
- pair generation
- correlation
- cointegration
- ADF
- half-life
- OLS
- rolling OLS
- Kalman output
- z-score
- entry/exit logic
- position direction
- P&L
- transaction costs
- drawdown
- performance metrics

Focus on important cases rather than creating excessive tests.

---

# PHASE 19 — End-to-End Validation

## Objective

Run the complete workflow.

Test:

Select sector
↓
Generate pairs
↓
Select pair
↓
Analyze pair
↓
Select methodology
↓
Run OOS backtest
↓
Display performance

Verify that:

- no imports fail
- no unnecessary Yahoo downloads occur
- no future-data leakage occurs
- charts display correctly
- metrics are internally consistent
- transaction costs are included
- the application does not crash on normal edge cases

---

# PHASE 20 — Cleanup and Documentation

## Objective

Prepare the final GitHub project.

Clean:

- unused code
- unused dependencies
- duplicate functions
- unnecessary files
- debug prints
- unnecessary UI elements

Update:

- README.md
- PROJECT.md
- comments/docstrings where useful

The README should eventually explain:

1. Project motivation
2. Quantitative methodology
3. Data
4. Pair selection
5. OLS / Rolling OLS / Kalman
6. Trading rules
7. Backtesting
8. Results
9. Limitations
10. How to run the application

Do not fabricate results.

Results should only be added after the actual backtests are completed.

---

# 29. Final Definition of Done

The project is complete when:

1. The 50-stock dataset is available locally.
2. Data is stored in Parquet.
3. Pair Scanner works by sector.
4. Pair statistics are calculated.
5. Pair Analyzer works for a selected pair.
6. Static OLS works.
7. Rolling OLS works.
8. Kalman dynamic hedge ratio works.
9. Z-score signals work.
10. Entry and exit rules work.
11. Fixed-capital sizing works.
12. 10-bps transaction costs are included.
13. OOS backtesting for 2024–2025 works.
14. Performance metrics are calculated.
15. All three Streamlit pages work.
16. Important core functions have tests.
17. No obvious look-ahead bias exists.
18. The project runs without major errors.

---

# 30. Explicit Non-Goals

Do not add:

- live trading
- broker integration
- intraday trading
- real-time feeds
- machine-learning price prediction
- LSTM
- Transformer
- reinforcement learning
- options
- GARCH
- Johansen
- portfolio optimization
- order books
- automated alerts
- cloud architecture
- database servers
- authentication
- multi-user functionality

The project should remain a focused statistical-arbitrage learning project.

---

# 31. Guiding Principle

The final project should answer one clear quantitative question well:

**How do Static OLS, Rolling OLS, and Kalman-filter hedge ratios behave when used in a simple pairs-trading strategy on Indian equities?**

Everything implemented in the project should support this objective.