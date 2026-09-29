"""Smoke tests for project setup and historical-data input normalization."""

from collections import Counter

import pandas as pd

from config.config import TICKER_TO_SECTOR, TICKERS
from scripts import download_data
from scripts.download_data import _standardize_columns


def test_config_contains_the_fixed_50_ticker_universe() -> None:
    assert len(TICKERS) == 50
    assert len(set(TICKERS)) == 50
    assert set(TICKERS) == set(TICKER_TO_SECTOR)
    assert Counter(TICKER_TO_SECTOR.values()) == Counter(
        {
            "Banking & Financial Services": 9,
            "IT": 8,
            "Automobile": 8,
            "FMCG / Consumer": 8,
            "Energy & Industrials": 9,
            "Pharmaceuticals": 8,
        }
    )


def test_yfinance_multiindex_columns_are_standardized() -> None:
    columns = pd.MultiIndex.from_tuples(
        [("Adj Close", "HDFCBANK.NS"), ("Volume", "HDFCBANK.NS")]
    )
    raw = pd.DataFrame([[100.0, 1_000]], columns=columns)

    standardized = _standardize_columns(raw)

    assert list(standardized.columns) == ["adj_close", "volume"]
    assert standardized.iloc[0].tolist() == [100.0, 1_000]


def test_download_dataset_writes_parquet_without_network(
    monkeypatch, tmp_path
) -> None:
    observation_date = pd.Timestamp("2016-01-04")

    def fake_download(ticker: str):
        sector = TICKER_TO_SECTOR[ticker]
        company = download_data.STOCK_UNIVERSE[sector][ticker]
        return (
            pd.DataFrame(
                {
                    "date": [observation_date],
                    "ticker": [ticker],
                    "company": [company],
                    "sector": [sector],
                    "adj_close": [100.0],
                    "volume": [1_000],
                }
            ),
            None,
        )

    output_path = tmp_path / "market_data.parquet"
    monkeypatch.setattr(download_data, "_download_ticker", fake_download)

    assert download_data.download_dataset(output_path)
    saved = pd.read_parquet(output_path)
    assert saved["ticker"].nunique() == 50
    assert len(saved) == 50