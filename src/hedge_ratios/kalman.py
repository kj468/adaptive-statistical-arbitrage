"""Training-calibrated, sequential Kalman hedge ratios using pykalman."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from pykalman import KalmanFilter

from config.config import TRAIN_END_DATE, TRAIN_START_DATE
from src.data.loader import align_prices, select_date_range


def _design_matrices(x: pd.Series) -> np.ndarray:
	"""Build time-varying observation matrices for [alpha_t, beta_t]."""
	return np.column_stack((np.ones(len(x)), x.to_numpy(dtype=float)))[:, None, :]


def kalman_hedge_ratios(
	y: pd.Series,
	x: pd.Series,
	training_start: date = TRAIN_START_DATE,
	training_end: date = TRAIN_END_DATE,
) -> pd.DataFrame:
	"""Fit noise parameters on training data and filter observations sequentially.

	Each row's alpha/beta are the state available before observing that row's Y;
	its spread is the one-step innovation. The training-fitted model parameters
	remain fixed while state updates continue through the OOS period.
	"""
	prices = align_prices(y, x)
	if not isinstance(prices.index, pd.DatetimeIndex):
		raise ValueError("Kalman hedge ratios require a DatetimeIndex")
	training = select_date_range(prices, training_start, training_end)
	if len(training) < 4:
		raise ValueError("Kalman parameter estimation requires at least four training observations")

	training_observations = training["y"].to_numpy(dtype=float)[:, None]
	model = KalmanFilter(
		transition_matrices=np.eye(2),
		observation_matrices=_design_matrices(training["x"]),
		initial_state_mean=np.array([training["y"].iloc[0], 0.0]),
	)
	model = model.em(
		training_observations,
		n_iter=10,
		em_vars=[
			"transition_covariance",
			"observation_covariance",
			"initial_state_covariance",
		],
	)

	training_end_position = prices.index.get_loc(training.index[-1])
	training_means, training_covariances = model.filter(training_observations)
	means = np.full((len(prices), 2), np.nan, dtype=float)
	spreads = np.full(len(prices), np.nan, dtype=float)
	transition = np.asarray(model.transition_matrices, dtype=float)

	for position, (_, row) in enumerate(training.iterrows()):
		global_position = prices.index.get_loc(row.name)
		state_before_observation = (
			np.asarray(model.initial_state_mean, dtype=float)
			if position == 0
			else transition @ training_means[position - 1]
		)
		observation_matrix = np.array([[1.0, float(row["x"])]])
		means[global_position] = state_before_observation
		spreads[global_position] = float(
			row["y"] - (observation_matrix @ state_before_observation).item()
		)

	state_mean = training_means[-1]
	state_covariance = training_covariances[-1]
	for position in range(training_end_position + 1, len(prices)):
		row = prices.iloc[position]
		observation_matrix = np.array([[1.0, float(row["x"])]] )
		state_before_observation = transition @ state_mean
		means[position] = state_before_observation
		spreads[position] = float(
			row["y"] - (observation_matrix @ state_before_observation).item()
		)
		state_mean, state_covariance = model.filter_update(
			state_mean,
			state_covariance,
			observation=np.array([row["y"]], dtype=float),
			observation_matrix=observation_matrix,
			transition_matrix=transition,
		)

	result = pd.DataFrame(
		{"alpha": means[:, 0], "beta": means[:, 1], "spread": spreads},
		index=prices.index,
	)
	return result.loc[training.index[0] :]