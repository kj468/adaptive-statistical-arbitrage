"""Causal spread z-score signals and stateful entry/exit rules."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from config.config import ENTRY_ZSCORE, EXIT_ZSCORE, ROLLING_WINDOW
from src.statistics.core import rolling_zscore


def generate_signals(
	spread: pd.Series,
	window: int = ROLLING_WINDOW,
	entry_threshold: float = ENTRY_ZSCORE,
	exit_threshold: float = EXIT_ZSCORE,
	start_date: date | None = None,
	end_date: date | None = None,
) -> pd.DataFrame:
	"""Return causal z-scores and long/short/flat spread states.

	Positive z enters a short spread (-1); negative z enters a long spread (+1).
	An active trade exits only when abs(z) is below the exit threshold; an
	opposite extreme does not reverse an open trade. Signals are initialized
	flat at the requested start date.
	"""
	if entry_threshold <= 0 or exit_threshold < 0 or exit_threshold >= entry_threshold:
		raise ValueError("Require entry_threshold > exit_threshold >= 0")
	zscore = rolling_zscore(spread, window)
	if start_date is not None:
		zscore = zscore.loc[zscore.index >= pd.Timestamp(start_date)]
	if end_date is not None:
		zscore = zscore.loc[zscore.index < pd.Timestamp(end_date) + pd.Timedelta(days=1)]

	state = 0
	positions: list[int] = []
	entries: list[bool] = []
	exits: list[bool] = []
	for value in zscore:
		entry = False
		exit_trade = False
		if np.isfinite(value):
			if state == 0:
				if value > entry_threshold:
					state = -1
					entry = True
				elif value < -entry_threshold:
					state = 1
					entry = True
			elif abs(value) < exit_threshold:
				state = 0
				exit_trade = True
		positions.append(state)
		entries.append(entry)
		exits.append(exit_trade)

	return pd.DataFrame(
		{
			"zscore": zscore,
			"signal": positions,
			"position": positions,
			"entry_event": entries,
			"exit_event": exits,
		},
		index=zscore.index,
	)