"""Training-period-only statistics for every pair in a selected sector."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import ROLLING_WINDOW, STOCK_UNIVERSE, TRAIN_END_DATE, TRAIN_START_DATE
from src.data.loader import (
	DEFAULT_DATASET_PATH,
	align_prices,
	get_prices,
	load_dataset,
	select_date_range,
)
from src.pairs.generation import generate_pairs
from src.statistics.core import (
	adf_test,
	cointegration_test,
	correlation,
	half_life,
	ols_regression,
	rolling_zscore,
)


def scan_sector(
	sector: str,
	dataset: pd.DataFrame | None = None,
	*,
	data_path: str | Path = DEFAULT_DATASET_PATH,
	start_date: date = TRAIN_START_DATE,
	end_date: date = TRAIN_END_DATE,
	window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
	"""Calculate training-only pair statistics without a composite score."""
	if dataset is None:
		dataset = load_dataset(data_path)
	training_data = select_date_range(dataset, start_date, end_date)
	prices = get_prices(training_data, STOCK_UNIVERSE[sector])
	rows: list[dict[str, object]] = []

	for ticker_y, ticker_x in generate_pairs(sector):
		row: dict[str, object] = {
			"sector": sector,
			"ticker_y": ticker_y,
			"company_y": STOCK_UNIVERSE[sector][ticker_y],
			"ticker_x": ticker_x,
			"company_x": STOCK_UNIVERSE[sector][ticker_x],
			"observations": 0,
			"correlation": np.nan,
			"cointegration_statistic": np.nan,
			"cointegration_pvalue": np.nan,
			"adf_statistic": np.nan,
			"adf_pvalue": np.nan,
			"half_life": np.nan,
			"ols_alpha": np.nan,
			"ols_hedge_ratio": np.nan,
			"current_zscore": np.nan,
			"error": None,
		}
		try:
			if ticker_y not in prices or ticker_x not in prices:
				raise ValueError("One or both tickers have no training-period observations")
			aligned = align_prices(prices[ticker_y], prices[ticker_x])
			row["observations"] = len(aligned)
			fit = ols_regression(aligned["y"], aligned["x"])
			coint_result = cointegration_test(aligned["y"], aligned["x"])
			adf_result = adf_test(fit.residuals)
			zscore = rolling_zscore(fit.residuals, window)
			row.update(
				{
					"correlation": correlation(aligned["y"], aligned["x"]),
					"cointegration_statistic": coint_result.statistic,
					"cointegration_pvalue": coint_result.pvalue,
					"adf_statistic": adf_result.statistic,
					"adf_pvalue": adf_result.pvalue,
					"half_life": half_life(fit.residuals),
					"ols_alpha": fit.alpha,
					"ols_hedge_ratio": fit.beta,
					"current_zscore": zscore.iloc[-1] if len(zscore) else np.nan,
				}
			)
		except (ValueError, KeyError, np.linalg.LinAlgError) as error:
			row["error"] = str(error)
		rows.append(row)
	return pd.DataFrame(rows)