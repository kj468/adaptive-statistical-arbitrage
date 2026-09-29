"""Detailed analysis of a selected same-sector stock pair."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.config import (
	DEMO_END_DATE,
	ROLLING_WINDOW,
	STOCK_UNIVERSE,
	TICKERS,
	TRAIN_END_DATE,
	TRAIN_START_DATE,
)
from src.data.loader import align_prices, get_prices, load_dataset, select_date_range
from src.hedge_ratios.kalman import kalman_hedge_ratios
from src.hedge_ratios.ols import apply_static_ols, fit_static_ols
from src.hedge_ratios.rolling_ols import rolling_ols
from src.pairs.generation import generate_pairs
from src.statistics.core import adf_test, cointegration_test, correlation, half_life, rolling_zscore


@st.cache_data(show_spinner=False)
def _load_data() -> pd.DataFrame:
	return load_dataset()


def _line_chart(title: str, series: dict[str, pd.Series], y_title: str) -> go.Figure:
	figure = go.Figure()
	for label, values in series.items():
		values = values.dropna()
		if not values.empty:
			figure.add_trace(go.Scatter(x=values.index, y=values, mode="lines", name=label))
	figure.update_layout(title=title, xaxis_title="Date", yaxis_title=y_title, hovermode="x unified")
	return figure


st.title("Pair Analyzer")
try:
	dataset = _load_data()
except (FileNotFoundError, ValueError, OSError) as error:
	st.warning(f"Local market data is unavailable: {error}")
	st.info("Download and validate the Parquet dataset with the project data script first.")
	st.stop()

missing_tickers = sorted(set(TICKERS).difference(dataset["ticker"].unique()))
if missing_tickers:
	st.warning("Partial dataset. Missing tickers: " + ", ".join(missing_tickers))

selected_pair = st.session_state.get("selected_pair")
sectors = list(STOCK_UNIVERSE)
default_sector = next(
	(sector for sector, stocks in STOCK_UNIVERSE.items() if selected_pair and all(t in stocks for t in selected_pair)),
	sectors[0],
)
sector = st.selectbox("Sector", sectors, index=sectors.index(default_sector))
pairs = generate_pairs(sector)
default_pair_index = pairs.index(selected_pair) if selected_pair in pairs else 0

def _pair_label(pair: tuple[str, str]) -> str:
	return f"{STOCK_UNIVERSE[sector][pair[0]]} ({pair[0]}) / {STOCK_UNIVERSE[sector][pair[1]]} ({pair[1]})"


pair = st.selectbox("Pair", pairs, index=default_pair_index, format_func=_pair_label)
price_table = get_prices(dataset, pair)
if any(ticker not in price_table for ticker in pair):
	st.error("One or both selected securities are missing from the local dataset.")
	st.stop()

prices = align_prices(price_table[pair[0]], price_table[pair[1]])
if prices.empty:
	st.error("The selected securities have no overlapping valid adjusted-price observations.")
	st.stop()

available_start = prices.index.min().date()
available_end = prices.index.max().date()
default_start = max(available_start, TRAIN_START_DATE)
default_end = min(available_end, DEMO_END_DATE)
if default_start > default_end:
	default_start, default_end = available_start, available_end
selected_dates = st.date_input(
	"Chart period",
	value=(default_start, default_end),
	min_value=available_start,
	max_value=available_end,
)
if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
	chart_start, chart_end = selected_dates
else:
	st.info("Select both a start and end date to display the analysis charts.")
	st.stop()
if chart_start > chart_end:
	st.error("Chart start date must be on or before the end date.")
	st.stop()

try:
	static_fit = fit_static_ols(
		prices["y"], prices["x"], TRAIN_START_DATE, TRAIN_END_DATE
	)
	static_spread = apply_static_ols(prices["y"], prices["x"], static_fit.alpha, static_fit.beta)
	rolling_frame = rolling_ols(prices["y"], prices["x"], ROLLING_WINDOW)
	kalman_frame = kalman_hedge_ratios(
		prices["y"], prices["x"], TRAIN_START_DATE, TRAIN_END_DATE
	)
	training_prices = select_date_range(prices, TRAIN_START_DATE, TRAIN_END_DATE)
	training_spread = static_fit.spread
	return_correlation = correlation(training_prices["y"], training_prices["x"])
	coint = cointegration_test(training_prices["y"], training_prices["x"])
	adf = adf_test(training_spread)
	spread_half_life = half_life(training_spread)
except (KeyError, ValueError, np.linalg.LinAlgError) as error:
	st.error(f"Pair analysis could not be calculated: {error}")
	st.stop()

training_metrics = st.columns(5)
training_metrics[0].metric("Training correlation · returns", f"{return_correlation:.3f}")
training_metrics[1].metric("Cointegration p-value", f"{coint.pvalue:.4g}")
training_metrics[2].metric("ADF p-value · OLS spread", f"{adf.pvalue:.4g}")
training_metrics[3].metric(
	"Half-life · observations", f"{spread_half_life:.2f}" if spread_half_life is not None else "Unavailable"
)
training_metrics[4].metric("Training observations", f"{len(training_prices):,}")

st.subheader("Hedge ratios")
ratio_columns = st.columns(3)
ratio_columns[0].metric("Static OLS · β", f"{static_fit.beta:.5f}")
chart_start_ts = pd.Timestamp(chart_start)
chart_end_ts = pd.Timestamp(chart_end)
rolling_beta = rolling_frame.loc[chart_start_ts:chart_end_ts, "beta"].dropna()
kalman_beta = kalman_frame.loc[chart_start_ts:chart_end_ts, "beta"].dropna()
ratio_columns[1].metric("Rolling OLS · latest β", f"{rolling_beta.iloc[-1]:.5f}" if not rolling_beta.empty else "Unavailable")
ratio_columns[2].metric("Kalman · latest β", f"{kalman_beta.iloc[-1]:.5f}" if not kalman_beta.empty else "Unavailable")

model_spreads = {
	"Static OLS": static_spread,
	"Rolling OLS": rolling_frame["spread"],
	"Kalman Filter": kalman_frame["spread"],
}
model_name = st.selectbox("Spread and z-score model", list(model_spreads))
full_zscore = rolling_zscore(model_spreads[model_name], ROLLING_WINDOW)
chart_prices = select_date_range(prices, chart_start, chart_end)
chart_spread = select_date_range(model_spreads[model_name].to_frame("spread"), chart_start, chart_end)["spread"]
chart_zscore = select_date_range(full_zscore.to_frame("zscore"), chart_start, chart_end)["zscore"]
chart_prices = chart_prices.dropna()
if chart_prices.empty:
	st.warning("No overlapping prices are available in the selected chart period.")
	st.stop()

normalized = chart_prices.div(chart_prices.iloc[0]).mul(100)
normalized.columns = ["Stock A", "Stock B"]
st.plotly_chart(_line_chart("Normalized adjusted prices (base 100)", {name: normalized[name] for name in normalized}, "Index"), width="stretch")
st.plotly_chart(_line_chart(f"{model_name} spread", {model_name: chart_spread}, "Spread"), width="stretch")
st.plotly_chart(_line_chart(f"{model_name} z-score", {model_name: chart_zscore}, "Z-score"), width="stretch")
st.plotly_chart(_line_chart("Kalman hedge ratio", {"β": kalman_frame["beta"].loc[chart_start_ts:chart_end_ts]}, "β"), width="stretch")

current = pd.concat(
	[chart_spread.rename("spread"), chart_zscore.rename("zscore")], axis=1
).dropna()
if not current.empty:
	st.caption(f"Latest values in selected range · {current.index[-1].date()}")
	current_columns = st.columns(2)
	current_columns[0].metric("Current spread", f"{current['spread'].iloc[-1]:.4f}")
	current_columns[1].metric("Current z-score", f"{current['zscore'].iloc[-1]:.3f}")
