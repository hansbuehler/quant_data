"""Causal rolling PCA factors for SANOS-derived daily states."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
FactorMethod = Literal["ordinary", "spot_orthogonal"]


@dataclass(frozen=True, kw_only=True)
class SanosFactorWindow:
    """The last 252 available exact one-session transitions through an origin."""

    origin_dates: tuple[date, ...]
    endpoint_dates: tuple[date, ...]
    state_changes: FloatArray
    spot_returns: FloatArray


@dataclass(frozen=True, kw_only=True)
class RollingSanosFactors:
    """One-standard-deviation state loadings and standardized factor scores."""

    method: FactorMethod
    window: SanosFactorWindow
    state_loadings: FloatArray
    factor_scores: FloatArray
    spot_correlated_state_loading: FloatArray
    explained_variance_fraction: FloatArray
    spot_factor_covariance: FloatArray

    @property
    def rank(self) -> int:
        return self.state_loadings.shape[1]


def select_sanos_factor_window(
    *,
    dates: tuple[date, ...],
    raw_states: NDArray[np.floating],
    spot_returns_into_date: NDArray[np.floating],
    sessions_from_previous_date: NDArray[np.integer],
    as_of_date: date,
    window_sessions: int = 252,
) -> SanosFactorWindow:
    """Select trailing exact-session pairs without using post-origin states."""
    states = np.asarray(raw_states, dtype=np.float64)
    spot_returns = np.asarray(spot_returns_into_date, dtype=np.float64)
    sessions = np.asarray(sessions_from_previous_date, dtype=np.int64)
    if window_sessions < 3 or states.ndim != 2 or len(dates) != len(states):
        raise ValueError("Invalid rolling SANOS state history")
    if spot_returns.shape != (len(dates),) or sessions.shape != (len(dates),):
        raise ValueError("Spot returns and session counts must align with dates")
    if any(left >= right for left, right in zip(dates, dates[1:])):
        raise ValueError("Dates must be strictly increasing")
    if not np.all(np.isfinite(states)):
        raise ValueError("SANOS raw states must be finite")

    indices = [
        index
        for index in range(1, len(dates))
        if dates[index] <= as_of_date and sessions[index] == 1
    ][-window_sessions:]
    if len(indices) != window_sessions:
        raise ValueError("Insufficient exact one-session SANOS transitions")
    selected_returns = spot_returns[indices]
    if not np.all(np.isfinite(selected_returns)):
        raise ValueError("Selected spot returns must be finite")
    return SanosFactorWindow(
        origin_dates=tuple(dates[index - 1] for index in indices),
        endpoint_dates=tuple(dates[index] for index in indices),
        state_changes=states[indices] - states[np.asarray(indices) - 1],
        spot_returns=selected_returns,
    )


def fit_rolling_sanos_factors(
    window: SanosFactorWindow,
    *,
    method: FactorMethod,
    rank: int = 8,
) -> RollingSanosFactors:
    """Fit ordinary state PCA or PCA after removing the spot projection."""
    changes = np.asarray(window.state_changes, dtype=np.float64)
    spot_returns = np.asarray(window.spot_returns, dtype=np.float64)
    sample_count, state_size = changes.shape
    if (
        sample_count < 3
        or rank >= sample_count
        or state_size < rank
        or rank < 1
        or spot_returns.shape != (sample_count,)
        or len(window.origin_dates) != sample_count
        or len(window.endpoint_dates) != sample_count
        or not np.all(np.isfinite(changes))
        or not np.all(np.isfinite(spot_returns))
    ):
        raise ValueError("Invalid rolling SANOS factor window or rank")
    if method not in ("ordinary", "spot_orthogonal"):
        raise ValueError(f"Unknown SANOS factor method: {method}")

    centered_changes = changes - changes.mean(axis=0)
    centered_spot = spot_returns - spot_returns.mean()
    spot_variance_sum = float(centered_spot @ centered_spot)
    if spot_variance_sum <= 0.0:
        raise ValueError("Spot returns have zero sample variance")
    spot_loading = np.zeros(state_size, dtype=np.float64)
    if method == "spot_orthogonal":
        spot_loading = centered_spot @ centered_changes / spot_variance_sum
        centered_changes = centered_changes - np.outer(centered_spot, spot_loading)

    gram = centered_changes @ centered_changes.T / (sample_count - 1)
    eigenvalues, left_vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:rank]
    selected_values = eigenvalues[order]
    if np.any(selected_values <= 0.0):
        raise ValueError("The selected SANOS PCA factors have zero sample variance")
    loadings = centered_changes.T @ left_vectors[:, order] / np.sqrt(sample_count - 1)
    pivot_indices = np.argmax(np.abs(loadings), axis=0)
    signs = np.sign(loadings[pivot_indices, np.arange(rank)])
    loadings *= signs
    scores = centered_changes @ loadings / selected_values
    total_variance = float(np.trace(gram))
    if total_variance <= 0.0:
        raise ValueError("SANOS state changes have zero sample variance")
    shock_matrix = np.column_stack((spot_returns, scores))
    return RollingSanosFactors(
        method=method,
        window=window,
        state_loadings=loadings,
        factor_scores=scores,
        spot_correlated_state_loading=spot_loading,
        explained_variance_fraction=selected_values / total_variance,
        spot_factor_covariance=np.cov(shock_matrix, rowvar=False, ddof=1),
    )
