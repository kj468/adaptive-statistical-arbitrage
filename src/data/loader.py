"""Load, validate, and align the local long-format Parquet price dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "market_data.parquet"
REQUIRED_COLUMNS = {"date", "ticker", "adj_close"}


def load_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
	"""Read validated long-format market data; remove rows without usable prices."""
	file_path = Path(path)
	if not file_path.is_file():
		raise FileNotFoundError(f"Market-data Parquet file not found: {file_path}")
	data = pd.read_parquet(file_path)
	missing = REQUIRED_COLUMNS.difference(data.columns)
	if missing:
		raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

	data = data.copy()
	data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
	data["ticker"] = data["ticker"].astype("string").str.strip()
	data["adj_close"] = pd.to_numeric(data["adj_close"], errors="coerce")
	data = data.dropna(subset=["date", "ticker", "adj_close"])
	data = data.loc[np.isfinite(data["adj_close"]) & (data["adj_close"] > 0)]

	duplicate = data.duplicated(["date", "ticker"], keep=False)
	if duplicate.any():
		raise ValueError(f"Dataset contains {int(duplicate.sum())} duplicate date/ticker rows")
	return data.sort_values(["date", "ticker"], ignore_index=True)


def select_date_range(
	data: pd.DataFrame,
	start_date: str | pd.Timestamp | None = None,
	end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
	"""Select an inclusive date range from long data or a date-indexed frame."""
	if "date" in data.columns:
		dates = pd.to_datetime(data["date"], errors="coerce")
		mask = dates.notna()
		if start_date is not None:
			mask &= dates >= pd.Timestamp(start_date)
		if end_date is not None:
			mask &= dates < pd.Timestamp(end_date) + pd.Timedelta(days=1)
		return data.loc[mask].copy()
	if not isinstance(data.index, pd.DatetimeIndex):
		raise ValueError("Data must have a 'date' column or a DatetimeIndex")
	mask = pd.Series(True, index=data.index)
	if start_date is not None:
		mask &= data.index >= pd.Timestamp(start_date)
	if end_date is not None:
		mask &= data.index < pd.Timestamp(end_date) + pd.Timedelta(days=1)
	return data.loc[mask.to_numpy()].copy()


def get_multiple_stocks(data: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
	"""Return long-format rows for the requested tickers."""
	if "ticker" not in data.columns:
		raise ValueError("Data must contain a 'ticker' column")
	return data.loc[data["ticker"].isin(tuple(dict.fromkeys(tickers)))].copy()


def get_stock_data(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
	"""Return one ticker's records indexed by date."""
	stock = get_multiple_stocks(data, [ticker])
	if stock.empty:
		raise KeyError(f"Ticker {ticker!r} is not present in the dataset")
	return stock.set_index("date").sort_index()


def get_prices(data: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
	"""Return adjusted close prices as date-indexed columns, preserving missing values."""
	selected = get_multiple_stocks(data, tickers)
	if selected.empty:
		return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
	return selected.pivot(index="date", columns="ticker", values="adj_close").sort_index()


def align_prices(
	price_a: pd.Series,
	price_b: pd.Series,
	min_observations: int | None = None,
) -> pd.DataFrame:
	"""Inner-align two prices and remove missing, non-finite, or non-positive rows."""
	if min_observations is not None and min_observations < 1:
		raise ValueError("min_observations must be at least 1")
	if price_a.index.has_duplicates or price_b.index.has_duplicates:
		raise ValueError("Price series contain duplicate dates")
	aligned = pd.concat(
		[price_a.rename("y"), price_b.rename("x")], axis=1, join="inner"
	).sort_index()
	for column in ("y", "x"):
		aligned[column] = pd.to_numeric(aligned[column], errors="coerce")
	valid = np.isfinite(aligned["y"]) & np.isfinite(aligned["x"])
	valid &= (aligned["y"] > 0) & (aligned["x"] > 0)
	aligned = aligned.loc[valid]
	if min_observations is not None and len(aligned) < min_observations:
		raise ValueError(
			f"Insufficient overlapping observations: {len(aligned)} available, "
			f"{min_observations} required"
		)
	return aligned


def check_sufficient_observations(data: pd.Series | pd.DataFrame, minimum: int) -> bool:
	"""Report whether a series/frame contains at least the requested row count."""
	if minimum < 1:
		raise ValueError("minimum must be at least 1")
	return len(data.dropna()) >= minimum