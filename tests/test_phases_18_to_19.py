"""End-to-end workflow coverage for complete-system testing phases."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from config.config import STOCK_UNIVERSE
from src.backtesting.engine import run_backtest
from src.data.loader import get_prices, load_dataset
from src.hedge_ratios.kalman import kalman_hedge_ratios
from src.hedge_ratios.ols import apply_static_ols, fit_static_ols
from src.hedge_ratios.rolling_ols import rolling_ols
from src.metrics.performance import calculate_metrics
from src.pairs.generation import generate_pairs
from src.pairs.scanner import scan_sector
from src.statistics.core import adf_test, cointegration_test, correlation, half_life, rolling_zscore


def test_sector_to_oos_performance_workflow(tmp_path, monkeypatch) -> None:
	"""Exercise sector scan, selected-pair analysis, all models, and OOS metrics."""
	rng = np.random.default_rng(108)
	training_dates = pd.bdate_range("2016-01-04", periods=180)
	test_dates = pd.bdate_range("2024-01-02", periods=90)
	dates = training_dates.append(test_dates)
	base = 100 + np.cumsum(rng.normal(0, 0.35, len(dates)))
	tickers = tuple(STOCK_UNIVERSE["Pharmaceuticals"])
	rows = []
	for offset, ticker in enumerate(tickers):
		prices = (1 + offset * 0.02) * base + rng.normal(0, 0.15, len(dates))
		rows.extend(
			{"date": timestamp, "ticker": ticker, "adj_close": price, "volume": 1_000}
			for timestamp, price in zip(dates, prices)
		)
	dataset_path = tmp_path / "market_data.parquet"
	pd.DataFrame(rows).to_parquet(dataset_path, index=False)

	def forbidden_download(*args, **kwargs):
		pytest.fail("The local analysis workflow must not contact Yahoo Finance")

	monkeypatch.setattr("yfinance.download", forbidden_download)
	loaded = load_dataset(dataset_path)
	assert len(generate_pairs("Pharmaceuticals")) == 28
	scanned = scan_sector("Pharmaceuticals", dataset=loaded, window=20)
	assert len(scanned) == 28
	valid = scanned.loc[scanned["error"].isna()]
	assert not valid.empty
	pair = (valid.iloc[0]["ticker_y"], valid.iloc[0]["ticker_x"])
	prices = get_prices(loaded, pair)
	y, x = prices[pair[0]], prices[pair[1]]

	# Analyzer calculations: training statistics and all three spread models.
	training_y = y.loc[pd.Timestamp("2016-01-01") : pd.Timestamp("2023-12-31")]
	training_x = x.loc[pd.Timestamp("2016-01-01") : pd.Timestamp("2023-12-31")]
	static_fit = fit_static_ols(training_y, training_x)
	assert np.isfinite(correlation(training_y, training_x))
	assert 0 <= cointegration_test(training_y, training_x).pvalue <= 1
	assert 0 <= adf_test(static_fit.spread).pvalue <= 1
	assert half_life(static_fit.spread) is None or half_life(static_fit.spread) > 0

	full_y = pd.concat([training_y, y.loc[pd.Timestamp("2024-01-01") : pd.Timestamp("2024-12-31")]])
	full_x = pd.concat([training_x, x.loc[pd.Timestamp("2024-01-01") : pd.Timestamp("2024-12-31")]])
	static_spread = apply_static_ols(full_y, full_x, static_fit.alpha, static_fit.beta)
	rolling = rolling_ols(full_y, full_x, window=20)
	kalman = kalman_hedge_ratios(full_y, full_x)
	assert np.isfinite(rolling.loc[test_dates, "beta"].dropna()).all()
	assert np.isfinite(kalman.loc[test_dates, ["alpha", "beta", "spread"]].to_numpy()).all()
	assert rolling_zscore(static_spread, 20).loc[test_dates].notna().any()

	# Same pair and test dates flow into each method and then the metric summary.
	for method in ("static_ols", "rolling_ols", "kalman"):
		result = run_backtest(
			full_y,
			full_x,
			method=method,
			start_date=date(2024, 1, 1),
			end_date=date(2024, 12, 31),
			training_start=date(2016, 1, 1),
			training_end=date(2023, 12, 31),
			window=20,
		)
		assert result.daily.index.min().year == 2024
		assert result.daily.index.max().year == 2024
		assert result.daily["transaction_cost"].sum() >= 0
		assert np.allclose(
			result.daily["equity_curve"],
			result.capital_per_pair + result.daily["net_pnl"].cumsum(),
		)
		metrics = calculate_metrics(result)
		assert metrics["number_of_trades"] == len(result.trades)
		assert metrics["transaction_costs"] == pytest.approx(result.daily["transaction_cost"].sum())
		assert metrics["turnover"] == pytest.approx(result.daily["turnover"].sum())
		assert metrics["maximum_drawdown"] <= 0