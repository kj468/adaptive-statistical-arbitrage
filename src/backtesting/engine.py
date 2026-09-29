"""Single-pair P&L and backtesting for the three specified hedge-ratio models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from config.config import (
	CAPITAL_PER_PAIR,
	ENTRY_ZSCORE,
	EXIT_ZSCORE,
	ROLLING_WINDOW,
	TEST_END_DATE,
	TEST_START_DATE,
	TRANSACTION_COST_PER_SIDE,
	TRAIN_END_DATE,
	TRAIN_START_DATE,
)
from src.data.loader import align_prices
from src.hedge_ratios.kalman import kalman_hedge_ratios
from src.hedge_ratios.ols import fit_static_ols
from src.hedge_ratios.rolling_ols import rolling_ols
from src.signals.trading import generate_signals


@dataclass(frozen=True)
class BacktestResult:
	"""Daily pair accounting and a trade ledger."""

	method: str
	capital_per_pair: float
	daily: pd.DataFrame
	trades: pd.DataFrame


TRADE_COLUMNS = [
	"direction",
	"entry_date",
	"exit_date",
	"entry_zscore",
	"exit_zscore",
	"holding_period",
	"exit_reason",
]


def _spread_for_method(
	prices: pd.DataFrame,
	method: str,
	training_start: date,
	training_end: date,
	rolling_window: int,
) -> pd.Series:
	if method == "static_ols":
		fit = fit_static_ols(
			prices["y"], prices["x"], training_start=training_start, training_end=training_end
		)
		return (prices["y"] - fit.alpha - fit.beta * prices["x"]).rename("spread")
	if method == "rolling_ols":
		return rolling_ols(prices["y"], prices["x"], rolling_window)["spread"]
	if method == "kalman":
		return kalman_hedge_ratios(
			prices["y"], prices["x"],
			training_start=training_start,
			training_end=training_end,
		)["spread"]
	raise ValueError("method must be 'static_ols', 'rolling_ols', or 'kalman'")


def run_backtest(
	y: pd.Series,
	x: pd.Series,
	method: str = "static_ols",
	*,
	start_date: date = TEST_START_DATE,
	end_date: date = TEST_END_DATE,
	training_start: date = TRAIN_START_DATE,
	training_end: date = TRAIN_END_DATE,
	window: int = ROLLING_WINDOW,
	entry_threshold: float = ENTRY_ZSCORE,
	exit_threshold: float = EXIT_ZSCORE,
	capital_per_pair: float = CAPITAL_PER_PAIR,
	transaction_cost_per_side: float = TRANSACTION_COST_PER_SIDE,
) -> BacktestResult:
	"""Backtest one pair, resetting flat at the requested evaluation start.

	A signal is evaluated at the close and transacted at that same close. P&L
	for a date is earned by the shares held across the preceding price interval;
	new positions therefore start earning from the following observation. Leg
	notionals are 50/50 at entry and share quantities remain fixed until exit.
	"""
	method_key = method.strip().lower().replace(" ", "_")
	if method_key in {"static", "ols", "static_ols"}:
		method_key = "static_ols"
	elif method_key in {"rolling", "rolling_ols"}:
		method_key = "rolling_ols"
	elif method_key in {"kalman", "kalman_filter"}:
		method_key = "kalman"
	else:
		raise ValueError("method must be Static OLS, Rolling OLS, or Kalman")
	if capital_per_pair <= 0:
		raise ValueError("capital_per_pair must be positive")
	if transaction_cost_per_side < 0:
		raise ValueError("transaction_cost_per_side cannot be negative")
	if window < 2:
		raise ValueError("window must be at least 2")

	prices = align_prices(y, x)
	if not isinstance(prices.index, pd.DatetimeIndex):
		raise ValueError("Backtesting requires a DatetimeIndex")
	if prices.empty:
		raise ValueError("No overlapping pair prices are available")
	spread = _spread_for_method(prices, method_key, training_start, training_end, window)
	signals = generate_signals(
		spread,
		window=window,
		entry_threshold=entry_threshold,
		exit_threshold=exit_threshold,
		start_date=start_date,
		end_date=end_date,
	)
	if signals.empty:
		raise ValueError("No observations are available in the requested backtest period")
	period = prices.loc[signals.index]
	rows: list[dict[str, float | int]] = []
	trades: list[dict[str, object]] = []
	shares_y = 0.0
	shares_x = 0.0
	active_trade: dict[str, object] | None = None
	previous_position = 0

	for timestamp, signal_row in signals.iterrows():
		price_y = float(period.at[timestamp, "y"])
		price_x = float(period.at[timestamp, "x"])
		all_position = prices.index.get_loc(timestamp)
		if all_position > 0:
			previous_prices = prices.iloc[all_position - 1]
			pnl_y = shares_y * (price_y - float(previous_prices["y"]))
			pnl_x = shares_x * (price_x - float(previous_prices["x"]))
		else:
			pnl_y = pnl_x = 0.0

		target_position = int(signal_row["position"])
		turnover = 0.0
		cost = 0.0
		if target_position != previous_position:
			turnover += abs(shares_y) * price_y + abs(shares_x) * price_x
			cost += turnover * transaction_cost_per_side
			if active_trade is not None:
				active_trade.update(
					{
						"exit_date": timestamp,
						"exit_zscore": float(signal_row["zscore"]),
						"holding_period": prices.index.get_loc(timestamp)
						- prices.index.get_loc(active_trade["entry_date"]),
						"exit_reason": "zscore_exit",
					}
				)
				trades.append(active_trade)
				active_trade = None
				shares_y = shares_x = 0.0
			if target_position != 0:
				half_capital = capital_per_pair / 2
				shares_y = target_position * half_capital / price_y
				shares_x = -target_position * half_capital / price_x
				active_trade = {
					"direction": "long_spread" if target_position > 0 else "short_spread",
					"entry_date": timestamp,
					"exit_date": pd.NaT,
					"entry_zscore": float(signal_row["zscore"]),
					"exit_zscore": np.nan,
					"holding_period": np.nan,
					"exit_reason": "open",
				}
				turnover += abs(shares_y) * price_y + abs(shares_x) * price_x
				cost += (abs(shares_y) * price_y + abs(shares_x) * price_x) * transaction_cost_per_side
		previous_position = target_position

		rows.append(
			{
				"price_y": price_y,
				"price_x": price_x,
				"spread": float(signal_row.get("spread", spread.loc[timestamp])),
				"zscore": float(signal_row["zscore"]),
				"signal": target_position,
				"position": target_position,
				"shares_y": shares_y,
				"shares_x": shares_x,
				"pnl_y": pnl_y,
				"pnl_x": pnl_x,
				"gross_pnl": pnl_y + pnl_x,
				"turnover": turnover,
				"transaction_cost": cost,
				"net_pnl": pnl_y + pnl_x - cost,
			}
		)

	# Liquidate any remaining position at the final evaluation close.
	if active_trade is not None:
		last_date = signals.index[-1]
		last_prices = period.loc[last_date]
		liquidation_turnover = abs(shares_y) * float(last_prices["y"]) + abs(shares_x) * float(last_prices["x"])
		liquidation_cost = liquidation_turnover * transaction_cost_per_side
		rows[-1]["turnover"] += liquidation_turnover
		rows[-1]["transaction_cost"] += liquidation_cost
		rows[-1]["net_pnl"] -= liquidation_cost
		rows[-1]["position"] = 0
		active_trade.update(
			{
				"exit_date": last_date,
				"exit_zscore": float(signals.iloc[-1]["zscore"]),
				"holding_period": prices.index.get_loc(last_date)
				- prices.index.get_loc(active_trade["entry_date"]),
				"exit_reason": "period_end",
			}
		)
		trades.append(active_trade)

	daily = pd.DataFrame(rows, index=signals.index)
	daily.index.name = "date"
	daily["daily_return"] = daily["net_pnl"] / capital_per_pair
	daily["cumulative_pnl"] = daily["net_pnl"].cumsum()
	daily["equity_curve"] = capital_per_pair + daily["cumulative_pnl"]
	return BacktestResult(
		method=method_key,
		capital_per_pair=capital_per_pair,
		daily=daily,
		trades=pd.DataFrame(trades, columns=TRADE_COLUMNS),
	)