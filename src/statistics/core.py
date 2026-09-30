"""Statistical functions used by the pair scanner and hedge-ratio models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from src.data.loader import align_prices


@dataclass(frozen=True)
class OLSResult:
	"""Intercept, slope, and in-sample residuals for y on x."""

	alpha: float
	beta: float
	residuals: pd.Series


@dataclass(frozen=True)
class CointegrationResult:
	"""Engle-Granger test output."""

	statistic: float
	pvalue: float
	critical_values: tuple[float, float, float]
	observations: int


@dataclass(frozen=True)
class ADFResult:
	"""Augmented Dickey-Fuller test output."""

	statistic: float
	pvalue: float
	used_lag: int
	observations: int
	critical_values: dict[str, float]


def _clean_series(values: pd.Series, name: str) -> pd.Series:
	series = pd.to_numeric(values, errors="coerce")
	series = series.replace([np.inf, -np.inf], np.nan).dropna()
	if series.empty:
		raise ValueError(f"{name} contains no finite observations")
	return series


def _align_values(y: pd.Series, x: pd.Series) -> pd.DataFrame:
	"""Inner-align finite regression observations without price-only constraints."""
	if y.index.has_duplicates or x.index.has_duplicates:
		raise ValueError("Regression series contain duplicate observations")
	data = pd.concat([y.rename("y"), x.rename("x")], axis=1, join="inner").sort_index()
	for column in ("y", "x"):
		data[column] = pd.to_numeric(data[column], errors="coerce")
	valid = np.isfinite(data["y"]) & np.isfinite(data["x"])
	return data.loc[valid]


def correlation(price_y: pd.Series, price_x: pd.Series) -> float:
	"""Pearson correlation of aligned daily simple returns, not price levels."""
	prices = align_prices(price_y, price_x)
	returns = prices.pct_change(fill_method=None).dropna()
	if len(returns) < 2:
		raise ValueError("At least three overlapping prices are needed for return correlation")
	if returns["y"].nunique() < 2 or returns["x"].nunique() < 2:
		raise ValueError("Correlation is undefined for a constant return series")
	return float(returns["y"].corr(returns["x"]))


def ols_regression(y: pd.Series, x: pd.Series) -> OLSResult:
	"""Fit y = alpha + beta*x by least squares and return indexed residuals."""
	data = _align_values(y, x)
	if len(data) < 2:
		raise ValueError("OLS regression requires at least two aligned observations")
	if data["x"].nunique() < 2:
		raise ValueError("OLS hedge ratio is undefined for a constant X series")
	design = np.column_stack((np.ones(len(data)), data["x"].to_numpy(dtype=float)))
	coefficients, _, rank, _ = np.linalg.lstsq(
		design, data["y"].to_numpy(dtype=float), rcond=None
	)
	if rank < 2:
		raise ValueError("OLS design matrix is rank deficient")
	residuals = data["y"] - (coefficients[0] + coefficients[1] * data["x"])
	return OLSResult(float(coefficients[0]), float(coefficients[1]), residuals.rename("spread"))


def cointegration_test(y: pd.Series, x: pd.Series) -> CointegrationResult:
	"""Run the Engle-Granger two-step cointegration test on price levels."""
	data = align_prices(y, x)
	if len(data) < 3:
		raise ValueError("Cointegration test requires at least three aligned observations")
	statistic, pvalue, critical = coint(
		data["y"], data["x"], trend="c", method="aeg", autolag="aic"
	)
	return CointegrationResult(
		float(statistic), float(pvalue), tuple(float(value) for value in critical), len(data)
	)


def adf_test(spread: pd.Series) -> ADFResult:
	"""Test a spread for a unit root using the ADF test with AIC lag selection."""
	values = _clean_series(spread, "Spread")
	if len(values) < 4:
		raise ValueError("ADF test requires at least four finite observations")
	statistic, pvalue, used_lag, observations, critical_values, _ = adfuller(
		values.to_numpy(dtype=float),
		regression="c",
		autolag="AIC",
	)
	return ADFResult(
		float(statistic),
		float(pvalue),
		int(used_lag),
		int(observations),
		{key: float(value) for key, value in critical_values.items()},
	)


def half_life(spread: pd.Series) -> float | None:
	"""Estimate AR(1) mean-reversion half-life in observations, or None if undefined."""
	values = _clean_series(spread, "Spread")
	pairs = pd.concat(
		[values.diff().rename("delta"), values.shift(1).rename("lagged")], axis=1
	).dropna()
	if len(pairs) < 3 or pairs["lagged"].nunique() < 2:
		return None
	try:
		coefficient = ols_regression(pairs["delta"], pairs["lagged"]).beta
		result = -np.log(2.0) / coefficient if coefficient < 0 else np.nan
		return float(result) if np.isfinite(result) else None
	except (ValueError, np.linalg.LinAlgError):
		return None


def rolling_zscore(spread: pd.Series, window: int) -> pd.Series:
	"""Standardize each value using only the previous `window` observations."""
	if window < 1:
		raise ValueError("window must be at least 1")
	values = pd.to_numeric(spread, errors="coerce").replace([np.inf, -np.inf], np.nan)
	history = values.shift(1)
	mean = history.rolling(window=window, min_periods=window).mean()
	std = history.rolling(window=window, min_periods=window).std()
	return ((values - mean) / std.where(std > 0)).rename("zscore")