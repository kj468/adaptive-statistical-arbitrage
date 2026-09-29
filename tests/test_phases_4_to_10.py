"""Focused unit tests for data, statistics, pairs, and hedge-ratio phases."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from config.config import STOCK_UNIVERSE
from src.data.loader import (
	align_prices,
	check_sufficient_observations,
	get_prices,
	load_dataset,
	select_date_range,
)
from src.hedge_ratios.kalman import kalman_hedge_ratios
from src.hedge_ratios.ols import fit_static_ols
from src.hedge_ratios.rolling_ols import rolling_ols
from src.pairs.generation import generate_pairs, get_sector
from src.pairs.scanner import scan_sector
from src.statistics.core import (
	adf_test,
	cointegration_test,
	correlation,
	half_life,
	ols_regression,
	rolling_zscore,
)


def _market_frame(dates: pd.DatetimeIndex, tickers: tuple[str, ...]) -> pd.DataFrame:
	rng = np.random.default_rng(31)
	base = 100 + np.cumsum(rng.normal(0, 0.6, len(dates)))
	rows = []
	for offset, ticker in enumerate(tickers):
		prices = base * (1 + offset * 0.015) + rng.normal(0, 0.3, len(dates))
		rows.extend(
			{"date": timestamp, "ticker": ticker, "adj_close": price, "volume": 1_000}
			for timestamp, price in zip(dates, prices)
		)
	return pd.DataFrame(rows)


def test_data_loader_validates_parquet_and_selects_ranges(tmp_path) -> None:
	path = tmp_path / "prices.parquet"
	data = pd.DataFrame(
		{
			"date": ["2024-01-01", "2024-01-02", "2024-01-03"],
			"ticker": ["AAA", "AAA", "AAA"],
			"adj_close": [10.0, np.nan, 12.0],
			"volume": [100, 200, 300],
		}
	)
	data.to_parquet(path, index=False)
	loaded = load_dataset(path)
	assert len(loaded) == 2
	assert len(select_date_range(loaded, date(2024, 1, 3), date(2024, 1, 3))) == 1
	assert list(get_prices(loaded, ["AAA"]).columns) == ["AAA"]
	assert check_sufficient_observations(loaded["adj_close"], 2)


def test_align_prices_inner_joins_and_rejects_insufficient_overlap() -> None:
	dates = pd.date_range("2024-01-01", periods=3)
	y = pd.Series([1.0, np.nan, 3.0], index=dates)
	x = pd.Series([4.0, 5.0, 6.0], index=dates)
	assert len(align_prices(y, x)) == 2
	with pytest.raises(ValueError, match="Insufficient overlapping"):
		align_prices(y, x, min_observations=3)


def test_statistics_return_clear_values_and_previous_window_zscores() -> None:
	x_values = pd.Series(np.arange(1.0, 21.0), index=pd.date_range("2020-01-01", periods=20))
	y_values = 2 * x_values
	fit = ols_regression(y_values, x_values)
	assert fit.alpha == pytest.approx(0.0)
	assert fit.beta == pytest.approx(2.0)
	assert np.allclose(fit.residuals, 0)
	assert correlation(y_values, x_values) == pytest.approx(1.0)

	rng = np.random.default_rng(9)
	innovations = rng.normal(size=500)
	rw = pd.Series(np.cumsum(innovations) + 200, index=pd.date_range("2016-01-01", periods=500))
	stationary_noise = pd.Series(rng.normal(0, 0.2, len(rw)), index=rw.index)
	y_pair = 5 + 1.4 * rw + stationary_noise
	assert 0 <= cointegration_test(y_pair, rw).pvalue <= 1
	assert adf_test(fit.residuals + pd.Series(rng.normal(0, 0.01, 20), index=x_values.index)).pvalue <= 1

	spread = pd.Series([1.0, 2.0, 3.0, 100.0])
	zscore = rolling_zscore(spread, window=3)
	assert zscore.iloc[:3].isna().all()
	assert zscore.iloc[3] == pytest.approx((100 - 2) / np.std([1, 2, 3], ddof=1))
	assert half_life(pd.Series(np.ones(5))) is None
	rng = np.random.default_rng(5)
	mean_reverting_values = np.zeros(1_000)
	for index in range(1, len(mean_reverting_values)):
		mean_reverting_values[index] = 0.8 * mean_reverting_values[index - 1] + rng.normal()
	mean_reverting = pd.Series(mean_reverting_values)
	assert half_life(mean_reverting) == pytest.approx(-np.log(2) / -0.2, rel=0.2)
	with pytest.raises(ValueError, match="window"):
		rolling_zscore(spread, window=0)


def test_pair_generation_is_unique_and_sector_limited() -> None:
	pairs = generate_pairs("Banking & Financial Services")
	assert len(pairs) == 36
	assert len(set(pairs)) == 36
	assert all(left != right for left, right in pairs)
	assert all(get_sector(left) == get_sector(right) for left, right in pairs)
	assert pairs[0] == ("HDFCBANK.NS", "ICICIBANK.NS")
	with pytest.raises(KeyError, match="Unknown sector"):
		generate_pairs("not a sector")


def test_pair_scanner_is_training_only_and_ols_only() -> None:
	sector = "Pharmaceuticals"
	dates = pd.bdate_range("2016-01-01", "2016-08-01")
	tickers = tuple(STOCK_UNIVERSE[sector])
	training = _market_frame(dates, tickers)
	future = _market_frame(pd.bdate_range("2024-01-01", periods=3), tickers)
	future["adj_close"] *= 100

	result_a = scan_sector(sector, pd.concat([training, future]), window=20)
	result_b = scan_sector(sector, training, window=20)
	assert len(result_a) == len(tickers) * (len(tickers) - 1) // 2
	assert result_a["ticker_y"].equals(result_b["ticker_y"])
	assert result_a["observations"].equals(result_b["observations"])
	assert np.allclose(result_a["ols_hedge_ratio"], result_b["ols_hedge_ratio"], equal_nan=True)
	assert "kalman_hedge_ratio" not in result_a.columns
	assert result_a["error"].isna().all()


def test_static_ols_is_training_only_and_rolling_ols_is_causal() -> None:
	dates = pd.bdate_range("2020-01-01", periods=100)
	x = pd.Series(np.linspace(50, 80, len(dates)), index=dates)
	y = 4 + 1.5 * x + np.sin(np.arange(len(dates)))
	training_end = dates[69].date()
	fit_a = fit_static_ols(y, x, date(2020, 1, 1), training_end)
	y_changed = y.copy()
	y_changed.iloc[70:] += 10_000
	fit_b = fit_static_ols(y_changed, x, date(2020, 1, 1), training_end)
	assert fit_a.alpha == pytest.approx(fit_b.alpha)
	assert fit_a.beta == pytest.approx(fit_b.beta)
	assert len(fit_a.spread) == 70

	rolling_a = rolling_ols(y, x, window=20)
	rolling_b = rolling_ols(y_changed, x, window=20)
	assert np.allclose(rolling_a.iloc[:70][["alpha", "beta", "spread"]], rolling_b.iloc[:70][["alpha", "beta", "spread"]], equal_nan=True)
	assert rolling_a["beta"].notna().sum() == len(dates) - 19


def test_kalman_parameters_train_then_filter_oos_sequentially() -> None:
	rng = np.random.default_rng(22)
	train_dates = pd.bdate_range("2023-08-01", periods=100)
	oos_dates = pd.bdate_range("2024-01-01", periods=8)
	dates = train_dates.append(oos_dates)
	x = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, len(dates))), index=dates)
	y = 4 + 1.25 * x + pd.Series(rng.normal(0, 0.15, len(dates)), index=dates)
	training_end = date(2023, 12, 31)

	result = kalman_hedge_ratios(y, x, date(2023, 8, 1), training_end)
	changed_y = y.copy()
	changed_y.loc[oos_dates[3]] += 500
	changed = kalman_hedge_ratios(changed_y, x, date(2023, 8, 1), training_end)

	assert result.loc[oos_dates[0], "alpha"] == pytest.approx(changed.loc[oos_dates[0], "alpha"])
	assert result.loc[oos_dates[0], "beta"] == pytest.approx(changed.loc[oos_dates[0], "beta"])
	assert result.loc[oos_dates[3], "beta"] == pytest.approx(changed.loc[oos_dates[3], "beta"])
	assert np.isfinite(result.loc[oos_dates, ["alpha", "beta", "spread"]].to_numpy()).all()
	assert not np.allclose(
		result.loc[oos_dates[4], ["alpha", "beta"]],
		changed.loc[oos_dates[4], ["alpha", "beta"]],
	)