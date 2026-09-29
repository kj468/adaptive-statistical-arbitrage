"""Download and validate adjusted daily prices into a local Parquet dataset."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import pandas as pd
import yfinance as yf

# Support both `python scripts/download_data.py` and module execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    DATA_END_DATE,
    DATA_END_DATE_EXCLUSIVE,
    DATA_START_DATE,
    STOCK_UNIVERSE,
    TICKER_TO_SECTOR,
    TICKERS,
)


LOGGER = logging.getLogger(__name__)
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "market_data.parquet"
PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")
REQUIRED_COLUMNS = ("adj_close", "volume")
COLUMN_NAMES = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "adj_close",
    "adj_close": "adj_close",
    "volume": "volume",
}


def _standardize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance's flat or MultiIndex columns to lowercase fields."""
    standardized = pd.DataFrame(index=raw.index)
    for column in raw.columns:
        parts = column if isinstance(column, tuple) else (column,)
        for part in parts:
            field = COLUMN_NAMES.get(str(part).strip().lower())
            if field and field not in standardized:
                standardized[field] = raw[column]
                break
    return standardized


def _download_ticker(ticker: str) -> tuple[pd.DataFrame | None, str | None]:
    """Fetch and validate one ticker; return cleaned rows or an error message."""
    try:
        raw = yf.download(
            ticker,
            start=DATA_START_DATE.isoformat(),
            end=DATA_END_DATE_EXCLUSIVE.isoformat(),
            auto_adjust=False,
            actions=False,
            progress=False,
        )
    except Exception as error:  # Network/provider errors vary by ticker.
        return None, f"download failed: {error}"

    if raw.empty:
        return None, "download returned no rows"

    prices = _standardize_columns(raw)
    missing_columns = set(REQUIRED_COLUMNS).difference(prices.columns)
    if missing_columns:
        return None, f"required columns missing: {', '.join(sorted(missing_columns))}"

    dates = pd.to_datetime(prices.index, errors="coerce")
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    prices.index = dates.normalize()

    invalid_dates = int(prices.index.isna().sum())
    prices = prices.loc[~prices.index.isna()]
    in_range = (prices.index >= pd.Timestamp(DATA_START_DATE)) & (
        prices.index <= pd.Timestamp(DATA_END_DATE)
    )
    removed_out_of_range = int((~in_range).sum())
    prices = prices.loc[in_range]
    if prices.empty:
        return None, "no rows within the requested date range"

    for column in prices.columns:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")

    missing_core = prices.loc[:, REQUIRED_COLUMNS].isna().any(axis=1)
    invalid_values = (prices["adj_close"] <= 0) | (prices["volume"] < 0)
    duplicate_dates = prices.index.duplicated(keep=False)
    if duplicate_dates.any():
        duplicate_count = int(duplicate_dates.sum())
        return None, f"{duplicate_count} rows have duplicate dates"

    removed_missing = int(missing_core.sum())
    removed_invalid = int((invalid_values & ~missing_core).sum())
    prices = prices.loc[~(missing_core | invalid_values)].copy()
    if prices.empty:
        return None, "no valid rows remain after missing-value and integrity checks"

    prices.index.name = "date"
    prices = prices.reset_index()
    prices.insert(1, "ticker", ticker)
    prices.insert(
        2,
        "company",
        STOCK_UNIVERSE[TICKER_TO_SECTOR[ticker]][ticker],
    )
    prices.insert(3, "sector", TICKER_TO_SECTOR[ticker])

    LOGGER.info(
        "%s: %d observations (%s to %s); dropped %d missing, %d invalid, %d out-of-range rows; %d invalid dates",
        ticker,
        len(prices),
        prices["date"].min().date(),
        prices["date"].max().date(),
        removed_missing,
        removed_invalid,
        removed_out_of_range,
        invalid_dates,
    )
    return prices, None


def download_dataset(output_path: Path = OUTPUT_PATH) -> bool:
    """Download configured tickers and save successful histories to Parquet.

    Returns True only when every configured ticker is available. A partial file
    is still written for usable tickers; missing symbols are reported and cause
    a False return so they are never mistaken for a complete universe.
    """
    datasets: list[pd.DataFrame] = []
    failures: dict[str, str] = {}

    for ticker in TICKERS:
        data, error = _download_ticker(ticker)
        if error:
            failures[ticker] = error
            LOGGER.error("%s: %s", ticker, error)
        elif data is not None:
            datasets.append(data)

    if not datasets:
        LOGGER.error("Dataset not written: no ticker histories passed validation.")
        return False

    result = pd.concat(datasets, ignore_index=True)
    result = result.sort_values(["date", "ticker"], ignore_index=True)
    downloaded_tickers = set(result["ticker"].unique())
    missing_tickers = sorted(set(TICKERS).difference(downloaded_tickers))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    LOGGER.info(
        "Saved %d observations for %d of %d tickers to %s (requested %s through %s).",
        len(result),
        len(downloaded_tickers),
        len(TICKERS),
        output_path,
        DATA_START_DATE,
        DATA_END_DATE,
    )
    if missing_tickers:
        LOGGER.warning(
            "Partial dataset saved. Missing ticker histories (not substituted): %s",
            ", ".join(missing_tickers),
        )
        return False
    return True


def main() -> int:
    """Run the historical download and return a process status code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return 0 if download_dataset() else 1


if __name__ == "__main__":
    raise SystemExit(main())
