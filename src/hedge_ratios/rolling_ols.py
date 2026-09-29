"""Causal rolling OLS hedge ratios using only data through each row."""

from __future__ import annotations

import pandas as pd

from config.config import ROLLING_WINDOW
from src.data.loader import align_prices
from src.statistics.core import ols_regression


def rolling_ols(
	y: pd.Series,
	x: pd.Series,
	window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
	"""Fit each rolling window through its current observation; return alpha, beta, spread."""
	if window < 2:
		raise ValueError("Rolling OLS window must be at least 2")
	prices = align_prices(y, x)
	result = pd.DataFrame(
		index=prices.index,
		columns=["alpha", "beta", "spread", "observations"],
		dtype=float,
	)
	for end in range(window - 1, len(prices)):
		segment = prices.iloc[end - window + 1 : end + 1]
		try:
			fit = ols_regression(segment["y"], segment["x"])
		except ValueError:
			continue
		result.iloc[end] = [
			fit.alpha,
			fit.beta,
			float(segment["y"].iloc[-1] - fit.alpha - fit.beta * segment["x"].iloc[-1]),
			window,
		]
	result["observations"] = result["observations"].astype("Int64")
	return result