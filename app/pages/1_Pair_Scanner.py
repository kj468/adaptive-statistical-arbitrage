"""Training-period scanner for same-sector candidate pairs."""

import pandas as pd
import streamlit as st

from config.config import STOCK_UNIVERSE, TICKERS
from src.data.loader import load_dataset
from src.pairs.scanner import scan_sector


@st.cache_data(show_spinner=False)
def _load_data() -> pd.DataFrame:
	return load_dataset()


@st.cache_data(show_spinner="Calculating training-period pair statistics...")
def _scan(sector: str, dataset: pd.DataFrame) -> pd.DataFrame:
	return scan_sector(sector, dataset=dataset)


st.title("Pair Scanner")
st.caption("All statistics use 2016–2023 training data. Correlation is computed from daily returns.")

try:
	dataset = _load_data()
except (FileNotFoundError, ValueError, OSError) as error:
	st.warning(f"Local market data is unavailable: {error}")
	st.info("Download and validate the Parquet dataset with the project data script first.")
	st.stop()

missing_tickers = sorted(set(TICKERS).difference(dataset["ticker"].unique()))
if missing_tickers:
	st.warning(
		"Partial dataset: no Yahoo Finance history is available for "
		+ ", ".join(missing_tickers)
		+ ". Pairs containing those tickers will be reported as unavailable."
	)

sector = st.selectbox("Sector", list(STOCK_UNIVERSE))
if st.button("Generate pairs", type="primary"):
	try:
		st.session_state["pair_scan_results"] = _scan(sector, dataset)
		st.session_state["pair_scan_sector"] = sector
	except (ValueError, KeyError, OSError) as error:
		st.error(f"Could not scan this sector: {error}")

results = st.session_state.get("pair_scan_results")
if results is None or st.session_state.get("pair_scan_sector") != sector:
	st.info("Choose a sector and generate its pair statistics.")
	st.stop()

sort_columns = [
	"cointegration_pvalue",
	"adf_pvalue",
	"correlation",
	"half_life",
	"ols_hedge_ratio",
	"observations",
]
controls = st.columns(2)
sort_by = controls[0].selectbox("Sort by", sort_columns)
ascending = controls[1].checkbox("Ascending", value=True)
max_pvalue = st.slider("Maximum cointegration p-value", 0.0, 1.0, 1.0, 0.01)
filtered = results.loc[
	results["cointegration_pvalue"].le(max_pvalue) | results["cointegration_pvalue"].isna()
].sort_values(sort_by, ascending=ascending, na_position="last")

failed = filtered.loc[filtered["error"].notna()]
if not failed.empty:
	with st.expander(f"Pairs with unavailable statistics ({len(failed)})"):
		st.dataframe(failed[["ticker_y", "ticker_x", "observations", "error"]], hide_index=True)

available = filtered.loc[filtered["error"].isna()]
display_columns = [
	"ticker_y",
	"ticker_x",
	"observations",
	"correlation",
	"cointegration_pvalue",
	"adf_pvalue",
	"half_life",
	"ols_hedge_ratio",
	"current_zscore",
]
st.dataframe(available[display_columns], width="stretch", hide_index=True)

if available.empty:
	st.info("No valid pairs match this filter.")
	st.stop()

pair_options = list(zip(available["ticker_y"], available["ticker_x"]))
def _pair_label(pair: tuple[str, str]) -> str:
	return f"{pair[0]} / {pair[1]}"


selected_pair = st.selectbox("Select a pair for deeper analysis", pair_options, format_func=_pair_label)
if st.button("Open Pair Analyzer"):
	st.session_state["selected_pair"] = selected_pair
	st.switch_page("pages/2_Pair_Analyzer.py")
