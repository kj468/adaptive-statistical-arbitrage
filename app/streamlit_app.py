"""Entry point for the Streamlit dashboard."""

import streamlit as st

st.set_page_config(page_title="Adaptive Statistical Arbitrage", layout="wide")
st.title("Adaptive Statistical Arbitrage")
st.write("Research pairs within a sector, analyze hedge-ratio models, and compare out-of-sample backtests.")
st.caption("Use the sidebar to open Pair Scanner, Pair Analyzer, or Backtest & Performance.")
st.info("The dashboard reads the local Parquet dataset in data/ and does not download market data at startup.")
