"""Static training-period OLS hedge ratio and spread."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from config.config import TRAIN_END_DATE, TRAIN_START_DATE
from src.data.loader import align_prices, select_date_range
from src.statistics.core import OLSResult, ols_regression


@dataclass(frozen=True)
class StaticOLSFit:
	"""Static OLS parameters and the training-period residual spread."""

	alpha: float
	beta: float
	spread: pd.Series
	observations: int


def fit_static_ols(
	y: pd.Series,
	x: pd.Series,
	training_start: date = TRAIN_START_DATE,
	training_end: date = TRAIN_END_DATE,
) -> StaticOLSFit:
	"""Estimate alpha/beta only from the configured training date range."""
	prices = align_prices(y, x)
	if not isinstance(prices.index, pd.DatetimeIndex):
		raise ValueError("Static OLS requires a DatetimeIndex to enforce training dates")
	prices = select_date_range(prices, training_start, training_end)
	fit: OLSResult = ols_regression(prices["y"], prices["x"])
	return StaticOLSFit(fit.alpha, fit.beta, fit.residuals, len(fit.residuals))


def apply_static_ols(y: pd.Series, x: pd.Series, alpha: float, beta: float) -> pd.Series:
	"""Construct a static spread using fixed, already-estimated parameters."""
	prices = align_prices(y, x)
	return (prices["y"] - alpha - beta * prices["x"]).rename("spread")