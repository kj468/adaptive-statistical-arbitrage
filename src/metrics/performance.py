"""Compact performance summary for the pair-level backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.config import (
	SHARPE_RISK_FREE_RATE,
	SORTINO_TARGET_RETURN,
	TRADING_DAYS_PER_YEAR,
)
from src.backtesting.engine import BacktestResult


def calculate_metrics(
	result: BacktestResult,
	*,
	annualization: int = TRADING_DAYS_PER_YEAR,
	risk_free_rate: float = SHARPE_RISK_FREE_RATE,
	target_return: float = SORTINO_TARGET_RETURN,
) -> dict[str, float | int]:
	"""Return requested return, risk, and trading metrics for a backtest.

	Annualized return uses the fixed-capital cumulative return; volatility,
	Sharpe, and Sortino use daily returns and the selected annualization factor.
	Turnover is gross rupee notional traded. Holding period is in trading
	observations and averages completed trades only.
	"""
	if annualization < 1:
		raise ValueError("annualization must be at least 1")
	daily = result.daily
	if "daily_return" not in daily:
		raise ValueError("Backtest daily data must contain 'daily_return'")
	returns = pd.to_numeric(daily["daily_return"], errors="coerce").replace(
		[np.inf, -np.inf], np.nan
	).dropna()
	if returns.empty:
		return {
			"cumulative_return": np.nan,
			"annualized_return": np.nan,
			"volatility": np.nan,
			"sharpe_ratio": np.nan,
			"sortino_ratio": np.nan,
			"maximum_drawdown": np.nan,
			"number_of_trades": 0,
			"average_holding_period": np.nan,
			"turnover": 0.0,
			"transaction_costs": 0.0,
		}

	cumulative_return = float(daily["net_pnl"].sum() / result.capital_per_pair)
	annualized_return = (
		float((1 + cumulative_return) ** (annualization / len(returns)) - 1)
		if cumulative_return > -1
		else np.nan
	)
	volatility = float(returns.std(ddof=1) * np.sqrt(annualization)) if len(returns) > 1 else np.nan
	excess = returns - risk_free_rate / annualization
	sharpe_denominator = returns.std(ddof=1)
	sharpe = (
		float(excess.mean() / sharpe_denominator * np.sqrt(annualization))
		if len(returns) > 1 and sharpe_denominator > 0
		else np.nan
	)
	downside = np.minimum(returns - target_return / annualization, 0.0)
	downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
	sortino = (
		float((returns.mean() - target_return / annualization) / downside_deviation * np.sqrt(annualization))
		if downside_deviation > 0
		else np.nan
	)
	initial_equity = result.capital_per_pair
	equity = pd.concat(
		[pd.Series([initial_equity]), pd.to_numeric(daily["equity_curve"], errors="coerce").reset_index(drop=True)]
	)
	drawdown = equity / equity.cummax() - 1
	trades = result.trades
	closed = trades.loc[pd.to_numeric(trades["holding_period"], errors="coerce").notna()]
	average_holding = (
		float(pd.to_numeric(closed["holding_period"]).mean()) if not closed.empty else np.nan
	)
	return {
		"cumulative_return": cumulative_return,
		"annualized_return": annualized_return,
		"volatility": volatility,
		"sharpe_ratio": sharpe,
		"sortino_ratio": sortino,
		"maximum_drawdown": float(drawdown.min()),
		"number_of_trades": int(len(trades)),
		"average_holding_period": average_holding,
		"turnover": float(daily["turnover"].sum()),
		"transaction_costs": float(daily["transaction_cost"].sum()),
	}