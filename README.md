# Adaptive Statistical Arbitrage in Indian Equity Markets

A small quantitative-finance research project comparing **Static OLS, Rolling OLS, and Kalman-filter hedge ratios** in a rules-based pairs strategy on Indian equities. It is intended for learning and historical analysis, not live trading or investment advice.

## Motivation

The project studies how hedge-ratio estimation affects spread construction, mean-reversion signals, and historical out-of-sample performance. Its central question is: **How do Static OLS, Rolling OLS, and Kalman-filter hedge ratios behave when used in a simple pairs-trading strategy on Indian equities?**

## Data and universe

- Daily Yahoo Finance history is downloaded by [scripts/download_data.py](scripts/download_data.py).
- Adjusted close and volume are stored locally in `data/market_data.parquet`.
- Streamlit reads local Parquet; it does not download data at startup.
- The configured universe contains 50 securities in six fixed sectors. Pairs are generated only within a sector.
- Training/model estimation: 2016–2023. Primary OOS evaluation: 2024–2025. Jan–Sep 2026 is a separate demo period, not part of headline OOS results.

The downloader reports unavailable or invalid ticker histories without substituting other securities. It writes successful histories but returns an incomplete status if any configured ticker is missing. Review its report before using a partial dataset.

## Pair statistics and selection

The Pair Scanner reports daily-return correlation, Engle–Granger cointegration, ADF on the OLS residual spread, half-life, OLS hedge ratio, observation count, and a training-period z-score. Statistics use only 2016–2023 data. Pairs can be sorted and filtered; no composite pair score is created. Correlation describes co-movement but does not prove cointegration or mean reversion.

## Hedge-ratio models

For dependent price series $Y$ and explanatory price series $X$:

- **Static OLS:** estimates $Y_t=\alpha+\beta X_t+\varepsilon_t$ on training data and holds alpha/beta fixed.
- **Rolling OLS:** re-estimates alpha/beta from the configured trailing 60-observation window.
- **Kalman Filter:** estimates dynamic alpha and beta. Noise/model parameters are estimated on training data and held fixed in OOS; states update sequentially.

The spread is constructed from each model's parameters; for static OLS it is $Y_t-\alpha-\beta X_t$.

## Trading rules and execution

- Z-score mean and standard deviation use the previous 60 spread observations; the current spread is excluded.
- Enter short-spread when $z>2$, long-spread when $z<-2$; exit when $|z|<0.5$. There is no stop-loss.
- Under spread $Y-\beta X$, short-spread means short $Y$ and long $X$; long-spread means long $Y$ and short $X$.
- Each pair uses ₹100,000 fixed capital, split equally by notional between the two legs. Share quantities are price-derived and not rounded to whole shares.
- Signals use the adjusted close and execute at the **next available adjusted close**, the price convention supported by the stored dataset.
- Each leg's P&L is marked close-to-close for shares held over that interval. Transaction costs are 10 bps per side on traded notional at entry and exit, including period-end liquidation.

## Backtesting and results

The Backtest & Performance page runs one selected pair and one selected hedge method at a time over 2024–2025. Models may update during OOS only with data available by each date. The page reports cumulative/annualized return, volatility, Sharpe (0% risk-free rate), Sortino (0% target), maximum drawdown, trade count, average holding period, turnover, transaction costs, and an equity curve. Daily return metrics use 252 trading days per year.

No headline or aggregate performance claims are published here. Run and verify the selected pair/model backtest before recording any result; identify its pair, method, period, and assumptions. A single pair is not evidence of general strategy performance.

## Limitations

- Yahoo Finance history and symbol availability can change.
- The universe is fixed, not point-in-time; historical membership and survivorship/universe-selection effects are not modeled.
- The 10-bps cost is simplified; fees, taxes, slippage, and bid-ask spread are not separately modeled.
- Pairs are evaluated individually; no portfolio-level capital competition or cross-pair risk is modeled.
- This is not a broker-connected, live, intraday, or real-time trading system.
- Historical results are not forecasts or investment advice.

## Run locally

From the repository root, install dependencies and download the local dataset:

```powershell
python -m pip install -r requirements.txt
python scripts/download_data.py
```

Then start the app:

```powershell
streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`) in a browser. Keep the terminal running while using the app; press Ctrl+C there to stop it. Run tests from the project root with `python -m pytest -q`.

## Repository map

- `app/`: Streamlit entry point and three pages.
- `config/config.py`: universe, dates, and strategy parameters.
- `data/`: local Parquet data; ignored by Git by default.
- `scripts/`: data-download utility.
- `src/`: data, statistics, pairs, hedge ratios, signals, backtesting, and metrics.
- `tests/`: unit and end-to-end workflow tests.