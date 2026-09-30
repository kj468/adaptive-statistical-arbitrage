"""Focused tests for strategy signals, P&L, backtesting, and metrics."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import run_backtest
from src.metrics.performance import calculate_metrics
from src.signals.trading import generate_signals


def test_signals_follow_entry_exit_rules_without_reentry() -> None:
	dates = pd.date_range("2024-01-01", periods=7)
	spread = pd.Series([0.0, 1.0, 3.0, 4.0, 3.5, -1.0, -2.0], index=dates)
	signals = generate_signals(spread, window=2)

	assert signals["position"].tolist() == [0, 0, -1, -1, 0, 1, 1]
	assert signals["entry_event"].tolist() == [False, False, True, False, False, True, False]
	assert signals["exit_event"].tolist() == [False, False, False, False, True, False, False]
	assert signals["zscore"].iloc[:2].isna().all()


def test_static_backtest_accounts_for_pair_legs_costs_and_metrics() -> None:
	rng = np.random.default_rng(41)
	training_dates = pd.bdate_range("2016-01-04", periods=80)
	test_dates = pd.bdate_range("2024-01-02", periods=10)
	dates = training_dates.append(test_dates)
	x = pd.Series(80 + np.cumsum(rng.normal(0, 0.5, len(dates))), index=dates)
	residual = np.concatenate(
		[rng.normal(0, 0.5, len(training_dates)), [0, 1, 3, 4, 3.5, -1, -2, -3, -4, 0]]
	)
	y = pd.Series(30 + 1.4 * x.to_numpy() + residual, index=dates)

	result = run_backtest(
		y,
		x,
		method="static_ols",
		start_date=date(2024, 1, 1),
		end_date=date(2024, 1, 31),
		training_start=date(2016, 1, 1),
		training_end=date(2023, 12, 31),
		window=2,
		capital_per_pair=100_000,
	)
	assert result.method == "static_ols"
	assert len(result.daily) == len(test_dates)
	assert result.daily["transaction_cost"].sum() > 0
	assert result.daily["turnover"].sum() >= 100_000
	assert result.daily["equity_curve"].iloc[-1] == pytest.approx(
		100_000 + result.daily["net_pnl"].sum()
	)
	signal_entries = result.daily.loc[
		result.daily["signal"].diff().fillna(result.daily["signal"]).ne(0)
	]
	signal_entries = signal_entries.loc[signal_entries["signal"] != 0]
	for signal_date, signal_row in signal_entries.iterrows():
		fill_index = result.daily.index.get_loc(signal_date) + 1
		assert fill_index < len(result.daily)
		entry = result.daily.iloc[fill_index]
		assert result.daily.index[fill_index] > signal_date
		assert entry["position"] == signal_row["signal"]
		assert result.daily.loc[signal_date, "position"] == 0
		assert abs(entry["shares_y"] * entry["price_y"]) == pytest.approx(50_000)
		assert abs(entry["shares_x"] * entry["price_x"]) == pytest.approx(50_000)
		assert (entry["shares_y"] > 0) == (entry["position"] > 0)
		assert (entry["shares_x"] > 0) == (entry["position"] < 0)
		assert entry["transaction_cost"] == pytest.approx(100.0)
	assert not result.trades.empty
	for _, trade in result.trades.iterrows():
		trade_days = result.daily.loc[trade["entry_date"] : trade["exit_date"]]
		assert trade["gross_pnl"] == pytest.approx(trade_days["gross_pnl"].sum())
		assert trade["transaction_cost"] == pytest.approx(trade_days["transaction_cost"].sum())
		assert trade["net_pnl"] == pytest.approx(trade["gross_pnl"] - trade["transaction_cost"])
	assert result.daily.iloc[-1]["position"] == 0
	assert result.daily.iloc[-1]["shares_y"] == 0
	assert result.daily.iloc[-1]["shares_x"] == 0

	metrics = calculate_metrics(result)
	assert metrics["number_of_trades"] == len(result.trades)
	assert metrics["turnover"] == pytest.approx(result.daily["turnover"].sum())
	assert metrics["transaction_costs"] == pytest.approx(result.daily["transaction_cost"].sum())
	assert np.isfinite(metrics["cumulative_return"])
	assert metrics["maximum_drawdown"] <= 0
	equity = pd.concat(
		[pd.Series([result.capital_per_pair]), result.daily["equity_curve"].reset_index(drop=True)]
	)
	expected_drawdown = (equity / equity.cummax() - 1).min()
	assert metrics["maximum_drawdown"] == pytest.approx(expected_drawdown)


def test_backtest_rejects_invalid_method_and_capital() -> None:
    dates = pd.bdate_range("2024-01-01", periods=5)
    x = pd.Series([10, 11, 12, 13, 14], index=dates)
    y = pd.Series([20, 21, 22, 23, 24], index=dates)
    with pytest.raises(ValueError, match="method"):
        run_backtest(y, x, method="unknown")
    with pytest.raises(ValueError, match="capital_per_pair"):
        run_backtest(y, x, capital_per_pair=0)


@pytest.mark.parametrize("method", ["rolling_ols", "kalman"])
def test_backtest_runs_adaptive_hedge_ratio_methods(method: str) -> None:
	rng = np.random.default_rng(73)
	training_dates = pd.bdate_range("2016-01-04", periods=80)
	test_dates = pd.bdate_range("2024-01-02", periods=5)
	dates = training_dates.append(test_dates)
	x = pd.Series(80 + np.cumsum(rng.normal(0, 0.4, len(dates))), index=dates)
	y = pd.Series(20 + 1.2 * x + rng.normal(0, 0.2, len(dates)), index=dates)
	result = run_backtest(
		y,
		x,
		method=method,
		start_date=date(2024, 1, 1),
		end_date=date(2024, 1, 31),
		training_start=date(2016, 1, 1),
		training_end=training_dates[-1].date(),
		window=10,
	)
	assert result.method == method
	assert result.daily.index.equals(test_dates)
	assert result.daily["daily_return"].notna().all()