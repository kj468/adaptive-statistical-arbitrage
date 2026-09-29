# Adaptive Statistical Arbitrage in Indian Equity Markets

A quantitative-finance project that studies **pairs trading and statistical arbitrage in Indian equities** by comparing three hedge-ratio estimation methods:

- **Static OLS**
- **Rolling OLS**
- **Kalman Filter**

The project uses historical NSE equity data to identify potentially mean-reverting stock pairs, generate z-score-based trading signals, and evaluate the strategies through out-of-sample backtesting.

## Project Workflow

```text
Indian Equity Data
       ↓
Sector-wise Pair Generation
       ↓
Correlation & Cointegration Analysis
       ↓
ADF & Mean-Reversion Analysis
       ↓
Hedge Ratio Estimation
 ┌─────┼──────────┐
 OLS  Rolling OLS  Kalman Filter
 └─────┼──────────┘
       ↓
Z-Score Trading Signals
       ↓
Out-of-Sample Backtesting
       ↓
Risk & Performance Analysis