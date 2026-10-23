"""Common observation-space diagnostics for DYSANOS 1.1 model comparisons.

Spot returns are forward-adjusted log returns and ATM-IV changes are changes
in log Black implied volatility at a fixed rolling business-day tenor.  A
transition is assigned to the sample containing its *endpoint* date, so the
first 2025 observation may use the final 2024 surface without look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AUDIT_BUSINESS_DTES = (1, 5, 10, 20, 40, 70)
AUTOCORRELATION_LAGS = (1, 5, 20)


@dataclass(frozen=True)
class LeverageStatistics:
    """Spot/log-ATM-IV leverage with time-block bootstrap uncertainty."""

    observations: int
    correlation: np.ndarray
    covariance: np.ndarray
    slope: np.ndarray
    correlation_interval: np.ndarray
    covariance_interval: np.ndarray
    slope_interval: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "observations": self.observations,
            "correlation": self.correlation.tolist(),
            "covariance": self.covariance.tolist(),
            "slope": self.slope.tolist(),
            "correlation_interval_95": self.correlation_interval.tolist(),
            "covariance_interval_95": self.covariance_interval.tolist(),
            "slope_interval_95": self.slope_interval.tolist(),
        }


def transition_sample_indices(
    dates: np.ndarray, *, evaluation_start: str = "2025-01-01"
) -> tuple[np.ndarray, np.ndarray]:
    """Return pre-evaluation and forward endpoint indices into a level series."""
    values = np.asarray(dates).astype("U10")
    if (
        values.ndim != 1
        or len(values) < 3
        or not np.all(values[1:] > values[:-1])
    ):
        raise ValueError("dates must be a strictly increasing one-dimensional series")
    endpoint = np.arange(1, len(values))
    training = endpoint[values[endpoint] < evaluation_start]
    forward = endpoint[values[endpoint] >= evaluation_start]
    if len(training) < 2 or len(forward) < 2:
        raise ValueError("both pre-evaluation and forward samples need transitions")
    return training, forward


def _leverage_values(
    spot_log_returns: np.ndarray, log_iv_changes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = spot_log_returns - np.mean(spot_log_returns, axis=-1, keepdims=True)
    y = log_iv_changes - np.mean(log_iv_changes, axis=-2, keepdims=True)
    covariance = np.mean(x[..., None] * y, axis=-2)
    spot_variance = np.mean(x * x, axis=-1)
    iv_variance = np.mean(y * y, axis=-2)
    if np.any(spot_variance <= 0.0) or np.any(iv_variance <= 0.0):
        raise ValueError("spot and each ATM-IV tenor must have positive variance")
    correlation = covariance / np.sqrt(spot_variance[..., None] * iv_variance)
    slope = covariance / spot_variance[..., None]
    return correlation, covariance, slope


def block_bootstrap_leverage(
    spot_log_returns: np.ndarray,
    log_iv_changes: np.ndarray,
    *,
    block_length: int = 10,
    replicates: int = 1_000,
    seed: int = 20261018,
) -> LeverageStatistics:
    """Estimate leverage and 95% circular moving-block percentile intervals.

    Covariance uses population normalization ``1/n``.  Resampling is over
    contiguous transition blocks, not independent daily observations.
    """
    spot = np.asarray(spot_log_returns, dtype=np.float64)
    changes = np.asarray(log_iv_changes, dtype=np.float64)
    if (
        spot.ndim != 1
        or changes.ndim != 2
        or changes.shape[0] != len(spot)
        or len(spot) <= block_length
        or block_length < 2
        or replicates < 100
        or np.any(~np.isfinite(spot))
        or np.any(~np.isfinite(changes))
    ):
        raise ValueError("invalid aligned leverage observations or bootstrap setup")
    estimate = _leverage_values(spot, changes)
    generator = np.random.default_rng(seed)
    blocks = (len(spot) + block_length - 1) // block_length
    starts = generator.integers(0, len(spot), size=(replicates, blocks))
    indices = (
        starts[:, :, None] + np.arange(block_length)[None, None, :]
    ) % len(spot)
    indices = indices.reshape(replicates, -1)[:, : len(spot)]
    bootstrapped = _leverage_values(spot[indices], changes[indices])
    intervals = tuple(
        np.quantile(values, (0.025, 0.975), axis=0).T
        for values in bootstrapped
    )
    return LeverageStatistics(
        observations=len(spot),
        correlation=estimate[0],
        covariance=estimate[1],
        slope=estimate[2],
        correlation_interval=intervals[0],
        covariance_interval=intervals[1],
        slope_interval=intervals[2],
    )


def lag_correlations(
    values: np.ndarray, *, lags: tuple[int, ...] = AUTOCORRELATION_LAGS
) -> dict[int, list[float]]:
    """Population-centered lag correlations, with columns kept separate."""
    series = np.asarray(values, dtype=np.float64)
    if series.ndim == 1:
        series = series[:, None]
    if series.ndim != 2 or np.any(~np.isfinite(series)):
        raise ValueError("values must be a finite one- or two-dimensional series")
    result: dict[int, list[float]] = {}
    for lag in lags:
        if lag <= 0 or lag >= len(series) - 1:
            raise ValueError(f"lag {lag} leaves too few observations")
        left = series[:-lag] - np.mean(series[:-lag], axis=0)
        right = series[lag:] - np.mean(series[lag:], axis=0)
        denominator = np.sqrt(np.sum(left * left, axis=0) * np.sum(right * right, axis=0))
        result[lag] = np.divide(
            np.sum(left * right, axis=0),
            denominator,
            out=np.full(series.shape[1], np.nan),
            where=denominator > 0.0,
        ).tolist()
    return result


def downside_leverage(
    spot_log_returns: np.ndarray,
    log_iv_changes: np.ndarray,
    *,
    spot_down_threshold: float,
    iv_up_thresholds: np.ndarray,
) -> dict[str, object]:
    """Separate down/up slopes and spot-down/IV-up co-exceedance."""
    spot = np.asarray(spot_log_returns, dtype=np.float64)
    changes = np.asarray(log_iv_changes, dtype=np.float64)
    up_thresholds = np.asarray(iv_up_thresholds, dtype=np.float64)
    if (
        spot.ndim != 1
        or changes.ndim != 2
        or changes.shape[0] != len(spot)
        or up_thresholds.shape != (changes.shape[1],)
        or np.any(~np.isfinite(spot))
        or np.any(~np.isfinite(changes))
        or np.any(~np.isfinite(up_thresholds))
    ):
        raise ValueError("invalid downside-leverage observations or thresholds")

    def conditional_slope(mask: np.ndarray) -> list[float]:
        if int(np.sum(mask)) < 3:
            return [float("nan")] * changes.shape[1]
        x = spot[mask] - np.mean(spot[mask])
        y = changes[mask] - np.mean(changes[mask], axis=0)
        variance = float(np.mean(x * x))
        if variance <= 0.0:
            return [float("nan")] * changes.shape[1]
        return (np.mean(x[:, None] * y, axis=0) / variance).tolist()

    down = spot <= spot_down_threshold
    up = changes >= up_thresholds[None, :]
    return {
        "negative_spot_slope": conditional_slope(spot < 0.0),
        "nonnegative_spot_slope": conditional_slope(spot >= 0.0),
        "joint_down_up_rate": np.mean(down[:, None] & up, axis=0).tolist(),
        "spot_down_rate": float(np.mean(down)),
        "iv_up_rate": np.mean(up, axis=0).tolist(),
        "spot_down_threshold": float(spot_down_threshold),
        "iv_up_thresholds": up_thresholds.tolist(),
    }


def historical_observation_panels(
    dates: np.ndarray,
    business_dtes: np.ndarray,
    atm_iv_levels: np.ndarray,
    forward_adjusted_simple_spot_returns: np.ndarray,
    *,
    evaluation_start: str = "2025-01-01",
    bootstrap_replicates: int = 1_000,
    bootstrap_block_length: int = 10,
    seed: int = 20261018,
) -> dict[str, object]:
    """Construct identical pre-2025 and forward-2025 observation panels.

    The spot and IV-up co-exceedance thresholds are estimated only from the
    pre-evaluation sample, then held fixed in the forward evaluation.
    """
    dates = np.asarray(dates).astype("U10")
    dtes = np.asarray(business_dtes, dtype=np.float64)
    iv = np.asarray(atm_iv_levels, dtype=np.float64)
    simple_spot = np.asarray(forward_adjusted_simple_spot_returns, dtype=np.float64)
    if (
        iv.shape != (len(dates), len(dtes))
        or simple_spot.shape != (len(dates) - 1,)
        or np.any(~np.isfinite(iv))
        or np.any(iv <= 0.0)
        or np.any(~np.isfinite(simple_spot))
        or np.any(simple_spot <= -1.0)
    ):
        raise ValueError("historical dates, ATM IV and spot returns do not align")
    tenor_indices = []
    for tenor in AUDIT_BUSINESS_DTES:
        matches = np.flatnonzero(np.isclose(dtes, tenor, rtol=0.0, atol=1.0e-10))
        if len(matches) != 1:
            raise ValueError(f"historical ATM panel lacks unique {tenor}D tenor")
        tenor_indices.append(int(matches[0]))
    selected_iv = iv[:, tenor_indices]
    spot_log = np.log1p(simple_spot)
    iv_changes = np.diff(np.log(selected_iv), axis=0)
    training, forward = transition_sample_indices(
        dates, evaluation_start=evaluation_start
    )
    training_spot = spot_log[training - 1]
    training_changes = iv_changes[training - 1]
    down_threshold = float(np.quantile(training_spot, 0.05))
    up_thresholds = np.quantile(training_changes, 0.95, axis=0)

    def panel(endpoints: np.ndarray, *, panel_seed: int) -> dict[str, object]:
        spot = spot_log[endpoints - 1]
        changes = iv_changes[endpoints - 1]
        levels = np.log(selected_iv[endpoints])
        leverage = block_bootstrap_leverage(
            spot,
            changes,
            block_length=bootstrap_block_length,
            replicates=bootstrap_replicates,
            seed=panel_seed,
        )
        return {
            "first_endpoint_date": str(dates[endpoints[0]]),
            "last_endpoint_date": str(dates[endpoints[-1]]),
            "observations": len(endpoints),
            "leverage": leverage.to_dict(),
            "autocorrelation": {
                "atm_log_iv_changes": lag_correlations(changes),
                "atm_log_iv_levels": lag_correlations(levels),
                "absolute_spot_log_returns": lag_correlations(np.abs(spot)),
                "absolute_atm_log_iv_changes": lag_correlations(np.abs(changes)),
            },
            "downside_leverage": downside_leverage(
                spot,
                changes,
                spot_down_threshold=down_threshold,
                iv_up_thresholds=up_thresholds,
            ),
            "spot_log_return_quantiles": {
                str(probability): float(np.quantile(spot, probability))
                for probability in (0.01, 0.05, 0.50, 0.95, 0.99)
            },
            "atm_log_iv_change_quantiles": {
                str(probability): np.quantile(changes, probability, axis=0).tolist()
                for probability in (0.01, 0.05, 0.50, 0.95, 0.99)
            },
        }

    return {
        "schema": "dysanos-v11-historical-observation-panels-v1",
        "definition": {
            "spot_return": "log(1 + forward-adjusted simple spot return)",
            "atm_iv_change": "one-session change in log rolling-tenor forward-ATM Black IV",
            "transition_assignment": "endpoint date; the first 2025 change uses the final 2024 surface",
            "covariance_denominator": "population 1/n within each sample",
            "bootstrap": "circular moving blocks of transition observations, percentile interval",
        },
        "business_dtes": list(AUDIT_BUSINESS_DTES),
        "downside_threshold_source": "pre-evaluation sample only",
        "pre_evaluation": panel(training, panel_seed=seed),
        "forward_evaluation": panel(forward, panel_seed=seed + 1),
    }
