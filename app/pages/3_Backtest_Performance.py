"""Out-of-sample pair backtest and performance view."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.config import STOCK_UNIVERSE, TEST_END_DATE, TEST_START_DATE, TICKERS
from src.backtesting.engine import BacktestResult, run_backtest
from src.data.loader import get_prices, load_dataset
from src.metrics.performance import calculate_metrics
from src.pairs.generation import generate_pairs


@st.cache_data(show_spinner=False)
def _load_data() -> pd.DataFrame:
	return load_dataset()


METHODS = {
	"Static OLS": "static_ols",
	"Rolling OLS": "rolling_ols",
	"Kalman Filter": "kalman",
}


def _pair_label(pair: tuple[str, str], sector: str) -> str:
	return f"{STOCK_UNIVERSE[sector][pair[0]]} ({pair[0]}) / {STOCK_UNIVERSE[sector][pair[1]]} ({pair[1]})"


st.title("Backtest Performance")
st.caption(f"Primary evaluation: Out-of-Sample · {TEST_START_DATE:%Y}–{TEST_END_DATE:%Y}")

try:
	dataset = _load_data()
except (FileNotFoundError, ValueError, OSError) as error:
	st.warning(f"Local market data is unavailable: {error}")
	st.info("Download and validate the Parquet dataset with the project data script first.")
	st.stop()

missing_tickers = sorted(set(TICKERS).difference(dataset["ticker"].unique()))
if missing_tickers:
	st.warning("Partial dataset. Missing tickers: " + ", ".join(missing_tickers))

sector_names = list(STOCK_UNIVERSE)
sector = st.selectbox("Sector", sector_names)
pairs = generate_pairs(sector)
scanner_pair = st.session_state.get("selected_pair")
default_pair_index = pairs.index(scanner_pair) if scanner_pair in pairs else 0
pair = st.selectbox(
	"Pair",
	pairs,
	index=default_pair_index,
	format_func=lambda value: _pair_label(value, sector),
)
method_label = st.selectbox("Hedge-ratio method", list(METHODS))

if st.button("Run OOS backtest", type="primary"):
	try:
		price_table = get_prices(dataset, pair)
		if any(ticker not in price_table for ticker in pair):
			raise ValueError("One or both tickers are missing from the local market dataset")
		result = run_backtest(
			price_table[pair[0]],
			price_table[pair[1]],
			method=METHODS[method_label],
		)
		st.session_state["backtest_result"] = {
			"pair": pair,
			"method": method_label,
			"result": result,
		}
	except (KeyError, ValueError, OSError, np.linalg.LinAlgError) as error:
		st.session_state.pop("backtest_result", None)
		st.error(f"Backtest could not be run: {error}")

saved = st.session_state.get("backtest_result")
if not saved or saved["pair"] != pair or saved["method"] != method_label:
	st.info("Select a pair and model, then run the out-of-sample backtest.")
	st.stop()

result: BacktestResult = saved["result"]
metrics = calculate_metrics(result)
st.subheader(f"{method_label} · {pair[0]} / {pair[1]}")

def _format_percent(value: float | int) -> str:
	return f"{value:.2%}" if pd.notna(value) else "Unavailable"


metric_rows = [
	(
		("Cumulative return", _format_percent(metrics["cumulative_return"])),
		("Annualized return", _format_percent(metrics["annualized_return"])),
		("Sharpe ratio", f"{metrics['sharpe_ratio']:.2f}" if pd.notna(metrics["sharpe_ratio"]) else "Unavailable"),
		("Sortino ratio", f"{metrics['sortino_ratio']:.2f}" if pd.notna(metrics["sortino_ratio"]) else "Unavailable"),
		("Volatility", _format_percent(metrics["volatility"])),
	),
	(
		("Maximum drawdown", _format_percent(metrics["maximum_drawdown"])),
		("Trades", str(metrics["number_of_trades"])),
		("Average holding period", f"{metrics['average_holding_period']:.1f} trading days" if pd.notna(metrics["average_holding_period"]) else "Unavailable"),
		("Turnover", f"₹{metrics['turnover']:,.0f}"),
		("Transaction costs", f"₹{metrics['transaction_costs']:,.2f}"),
	),
]
for values in metric_rows:
	columns = st.columns(len(values))
	for column, (label, value) in zip(columns, values):
		column.metric(label, value)

figure = go.Figure()
figure.add_trace(
	go.Scatter(
		x=result.daily.index,
		y=result.daily["equity_curve"],
		mode="lines",
		name="Equity",
	)
)
figure.update_layout(
	title="Out-of-Sample Equity Curve",
	xaxis_title="Date",
	yaxis_title="Equity (₹)",
	hovermode="x unified",
)
st.plotly_chart(figure, use_container_width=True)

st.subheader("Trades")
if result.trades.empty:
	st.info("No trades were generated in the selected test period.")
else:
	st.dataframe(result.trades, use_container_width=True, hide_index=True)

with st.expander("Daily positions and P&L"):
	st.dataframe(result.daily, use_container_width=True)
