"""Static-fit, forecast, and no-arbitrage metrics for Tangent Lévy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from cdxcore.err import verify_inp

from .coordinates import implied_volatility, normalized_call_intrinsic


@dataclass(frozen=True)
class PriceErrorMetrics:
    """Outright errors against a smoothed normalized call target."""

    observations: int
    weighted_rmse: float
    rmse: float
    mae: float
    median_absolute_error: float
    p95_absolute_error: float
    p99_absolute_error: float
    maximum_absolute_error: float
    mean_signed_error: float


@dataclass(frozen=True)
class IntervalErrorMetrics:
    """Half-spread-normalized executable-interval diagnostics."""

    observations: int
    fraction_inside: float
    fraction_outside: float
    mean_all: float
    median_all: float
    mean_positive: float
    median_positive: float
    p95: float
    p99: float


@dataclass(frozen=True)
class NoArbitrageMetrics:
    """Maximum finite-grid static-arbitrage violations."""

    lower_bound: float
    upper_bound: float
    strike_monotonicity: float
    strike_convexity: float
    calendar_monotonicity: float

    @property
    def maximum(self) -> float:
        """Largest reported violation."""

        return max(
            self.lower_bound,
            self.upper_bound,
            self.strike_monotonicity,
            self.strike_convexity,
            self.calendar_monotonicity,
        )


@dataclass(frozen=True)
class StatisticalFidelityMetrics:
    """Centered-innovation moment and factor-structure comparison."""

    reference_observations: int
    model_observations: int
    dimensions: int
    mean_change_error_norm: float
    relative_covariance_error: float
    correlation_error: float
    median_variance_ratio: float
    eigenvalue_spectrum_error: float
    maximum_principal_angle_degrees: float
    reference_covariance_condition: float
    model_covariance_condition: float


def price_error_metrics(
    model: np.ndarray,
    target: np.ndarray,
    *,
    half_spread: np.ndarray | None = None,
    residual_multiplier: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> PriceErrorMetrics:
    """Compute original-strike or dense-grid outright price errors."""

    model = np.asarray(model, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    verify_inp(
        model.shape == target.shape,
        lambda: f"model and target shapes differ: {model.shape}, {target.shape}",
    )
    verify_inp(
        half_spread is None or residual_multiplier is None,
        "provide either half spreads or residual multipliers, not both",
    )
    active = (
        np.ones(model.shape, dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    verify_inp(
        active.shape == model.shape,
        lambda: f"mask shape {active.shape} does not match prices {model.shape}",
    )
    active &= np.isfinite(model) & np.isfinite(target)
    verify_inp(np.any(active), "no finite active observations for price metrics")
    errors = model[active] - target[active]
    absolute = np.abs(errors)
    if residual_multiplier is not None:
        multiplier = np.asarray(residual_multiplier, dtype=np.float64)
        verify_inp(
            multiplier.shape == model.shape
            and np.all(np.isfinite(multiplier[active]))
            and np.all(multiplier[active] > 0.0),
            lambda: "residual multipliers must be finite, positive, and match prices",
        )
        weighted_rmse = float(np.sqrt(np.mean(np.square(multiplier[active] * errors))))
    elif half_spread is None:
        weighted_rmse = float(np.sqrt(np.mean(np.square(errors))))
    else:
        spread = np.asarray(half_spread, dtype=np.float64)
        verify_inp(
            spread.shape == model.shape
            and np.all(np.isfinite(spread[active]))
            and np.all(spread[active] > 0.0),
            lambda: "half spreads must be finite, positive, and match prices",
        )
        inverse_variance = 1.0 / np.square(spread[active])
        weighted_rmse = float(
            np.sqrt(
                np.sum(inverse_variance * np.square(errors)) / np.sum(inverse_variance)
            )
        )
    return PriceErrorMetrics(
        observations=int(errors.size),
        weighted_rmse=weighted_rmse,
        rmse=float(np.sqrt(np.mean(np.square(errors)))),
        mae=float(np.mean(absolute)),
        median_absolute_error=float(np.median(absolute)),
        p95_absolute_error=float(np.quantile(absolute, 0.95)),
        p99_absolute_error=float(np.quantile(absolute, 0.99)),
        maximum_absolute_error=float(np.max(absolute)),
        mean_signed_error=float(np.mean(errors)),
    )


def interval_error_metrics(
    model: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    *,
    half_spread_floor: float = 1.0e-4,
    mask: np.ndarray | None = None,
) -> IntervalErrorMetrics:
    """Implement the repository half-spread interval-error convention."""

    model = np.asarray(model, dtype=np.float64)
    bids = np.asarray(bids, dtype=np.float64)
    asks = np.asarray(asks, dtype=np.float64)
    verify_inp(
        model.shape == bids.shape == asks.shape,
        lambda: "model, bid, and ask arrays must have the same shape",
    )
    verify_inp(
        math.isfinite(half_spread_floor) and half_spread_floor > 0.0,
        lambda: f"invalid half-spread floor {half_spread_floor!r}",
    )
    active = (
        np.ones(model.shape, dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    verify_inp(active.shape == model.shape, "mask shape does not match interval arrays")
    active &= np.isfinite(model) & np.isfinite(bids) & np.isfinite(asks) & (asks > bids)
    verify_inp(np.any(active), "no finite active observations for interval metrics")
    model_active = model[active]
    bids_active = bids[active]
    asks_active = asks[active]
    half_spread = np.maximum(
        0.5 * (asks_active - bids_active),
        half_spread_floor,
    )
    errors = (
        np.maximum(0.0, model_active - asks_active)
        + np.maximum(0.0, bids_active - model_active)
    ) / half_spread
    positive = errors[errors > 0.0]
    return IntervalErrorMetrics(
        observations=int(errors.size),
        fraction_inside=float(np.mean(errors == 0.0)),
        fraction_outside=float(np.mean(errors > 0.0)),
        mean_all=float(np.mean(errors)),
        median_all=float(np.median(errors)),
        mean_positive=float(np.mean(positive)) if positive.size else 0.0,
        median_positive=float(np.median(positive)) if positive.size else 0.0,
        p95=float(np.quantile(errors, 0.95)),
        p99=float(np.quantile(errors, 0.99)),
    )


def implied_volatility_error(
    *,
    tau: np.ndarray,
    strikes: np.ndarray,
    model_calls: np.ndarray,
    target_calls: np.ndarray,
    mask: np.ndarray | None = None,
) -> PriceErrorMetrics:
    """Return implied-volatility error summaries in volatility points."""

    tau_array, strike_array, model_array, target_array = np.broadcast_arrays(
        np.asarray(tau, dtype=np.float64),
        np.asarray(strikes, dtype=np.float64),
        np.asarray(model_calls, dtype=np.float64),
        np.asarray(target_calls, dtype=np.float64),
    )
    active = (
        np.ones(model_array.shape, dtype=bool)
        if mask is None
        else np.broadcast_to(np.asarray(mask, dtype=bool), model_array.shape).copy()
    )
    lower = normalized_call_intrinsic(strike_array)
    active &= (
        np.isfinite(tau_array)
        & (tau_array > 0.0)
        & np.isfinite(strike_array)
        & (strike_array > 0.0)
        & np.isfinite(model_array)
        & np.isfinite(target_array)
        & (model_array >= lower - 1.0e-10)
        & (model_array <= 1.0 + 1.0e-10)
        & (target_array >= lower - 1.0e-10)
        & (target_array <= 1.0 + 1.0e-10)
    )
    verify_inp(np.any(active), "no active calls satisfy implied-volatility bounds")
    model_vol = implied_volatility(
        tau=tau_array[active],
        strike=strike_array[active],
        call_price=model_array[active],
    )
    target_vol = implied_volatility(
        tau=tau_array[active],
        strike=strike_array[active],
        call_price=target_array[active],
    )
    return price_error_metrics(model_vol, target_vol)


def no_arbitrage_metrics(
    calls: np.ndarray,
    *,
    tau: np.ndarray,
    strikes: np.ndarray,
) -> NoArbitrageMetrics:
    """Measure normalized call bounds and finite-grid shape restrictions."""

    calls = np.asarray(calls, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    strikes = np.asarray(strikes, dtype=np.float64)
    verify_inp(
        calls.ndim == 2
        and tau.shape == (calls.shape[0],)
        and strikes.shape in {(calls.shape[1],), calls.shape}
        and np.all(np.isfinite(calls)),
        lambda: (
            "calls must have shape (T,N), tau shape (T,), and strikes shape "
            f"(N,) or (T,N); got {calls.shape}, {tau.shape}, {strikes.shape}"
        ),
    )
    strike_grid = (
        np.broadcast_to(strikes, calls.shape) if strikes.ndim == 1 else strikes
    )
    verify_inp(
        calls.shape[1] >= 3
        and np.all(np.diff(strike_grid, axis=1) > 0.0)
        and np.allclose(strike_grid, strike_grid[:1]),
        lambda: (
            "no-arbitrage surface metrics require at least three common, "
            "strictly increasing strikes across maturities"
        ),
    )
    intrinsic = normalized_call_intrinsic(strike_grid)
    lower = float(np.max(np.maximum(intrinsic - calls, 0.0)))
    upper = float(np.max(np.maximum(calls - 1.0, 0.0)))
    monotonic = float(np.max(np.maximum(np.diff(calls, axis=1), 0.0)))
    slopes = np.diff(calls, axis=1) / np.diff(strike_grid, axis=1)
    convexity = float(np.max(np.maximum(-np.diff(slopes, axis=1), 0.0)))
    calendar = float(np.max(np.maximum(-np.diff(calls, axis=0), 0.0)))
    return NoArbitrageMetrics(
        lower_bound=lower,
        upper_bound=upper,
        strike_monotonicity=monotonic,
        strike_convexity=convexity,
        calendar_monotonicity=calendar,
    )


def statistical_fidelity_metrics(
    reference_changes: np.ndarray,
    model_changes: np.ndarray,
    *,
    num_factors: int = 5,
) -> StatisticalFidelityMetrics:
    """Compare two change samples without assigning a physical-measure claim."""

    reference = np.asarray(reference_changes, dtype=np.float64)
    model = np.asarray(model_changes, dtype=np.float64)
    verify_inp(
        reference.ndim == 2
        and model.ndim == 2
        and reference.shape[1] == model.shape[1]
        and reference.shape[0] >= 2
        and model.shape[0] >= 2
        and np.all(np.isfinite(reference))
        and np.all(np.isfinite(model)),
        lambda: (
            "statistical fidelity requires finite samples (N,D) with a common "
            f"dimension, got {reference.shape} and {model.shape}"
        ),
    )
    verify_inp(
        isinstance(num_factors, int) and num_factors > 0,
        lambda: f"'num_factors' must be positive, got {num_factors!r}",
    )
    reference_covariance = np.atleast_2d(np.cov(reference, rowvar=False, ddof=1))
    model_covariance = np.atleast_2d(np.cov(model, rowvar=False, ddof=1))
    covariance_scale = max(float(np.linalg.norm(reference_covariance)), 1.0e-15)
    covariance_error = float(
        np.linalg.norm(model_covariance - reference_covariance) / covariance_scale
    )
    reference_variance = np.diag(reference_covariance)
    model_variance = np.diag(model_covariance)
    active = (reference_variance > 1.0e-15) & (model_variance > 1.0e-15)
    verify_inp(
        np.count_nonzero(active) >= 2,
        lambda: "fewer than two common nondegenerate dimensions",
    )
    reference_scale = np.sqrt(reference_variance[active])
    model_scale = np.sqrt(model_variance[active])
    reference_correlation = reference_covariance[np.ix_(active, active)] / (
        reference_scale[:, None] * reference_scale[None, :]
    )
    model_correlation = model_covariance[np.ix_(active, active)] / (
        model_scale[:, None] * model_scale[None, :]
    )
    correlation_error = float(
        np.linalg.norm(model_correlation - reference_correlation)
        / max(float(np.linalg.norm(reference_correlation)), 1.0e-15)
    )
    reference_values, reference_vectors = np.linalg.eigh(reference_covariance)
    model_values, model_vectors = np.linalg.eigh(model_covariance)
    reference_order = np.argsort(reference_values)[::-1]
    model_order = np.argsort(model_values)[::-1]
    reference_values = np.maximum(reference_values[reference_order], 0.0)
    model_values = np.maximum(model_values[model_order], 0.0)
    eigenvalue_error = float(
        np.linalg.norm(model_values - reference_values)
        / max(float(np.linalg.norm(reference_values)), 1.0e-15)
    )
    factors = min(
        num_factors,
        int(np.count_nonzero(reference_values > 1.0e-15)),
        int(np.count_nonzero(model_values > 1.0e-15)),
    )
    if factors == 0:
        maximum_angle = 90.0
    else:
        reference_basis = reference_vectors[:, reference_order[:factors]]
        model_basis = model_vectors[:, model_order[:factors]]
        singular_values = np.linalg.svd(
            reference_basis.T @ model_basis,
            compute_uv=False,
        )
        angles = np.arccos(np.clip(singular_values, 0.0, 1.0))
        maximum_angle = float(np.degrees(np.max(angles)))

    def condition(values: np.ndarray) -> float:
        positive = values[values > 1.0e-15]
        return float(positive[0] / positive[-1]) if positive.size > 0 else math.inf

    return StatisticalFidelityMetrics(
        reference_observations=reference.shape[0],
        model_observations=model.shape[0],
        dimensions=reference.shape[1],
        mean_change_error_norm=float(
            np.linalg.norm(np.mean(model, axis=0) - np.mean(reference, axis=0))
        ),
        relative_covariance_error=covariance_error,
        correlation_error=correlation_error,
        median_variance_ratio=float(
            np.median(model_variance[active] / reference_variance[active])
        ),
        eigenvalue_spectrum_error=eigenvalue_error,
        maximum_principal_angle_degrees=maximum_angle,
        reference_covariance_condition=condition(reference_values),
        model_covariance_condition=condition(model_values),
    )


def energy_score(
    ensemble: np.ndarray,
    observation: np.ndarray,
    *,
    component_weights: np.ndarray | None = None,
) -> float:
    """Compute the unbiased finite-ensemble multivariate energy score."""

    ensemble = np.asarray(ensemble, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    verify_inp(
        ensemble.ndim == 2
        and ensemble.shape[0] >= 2
        and observation.shape == (ensemble.shape[1],)
        and np.all(np.isfinite(ensemble))
        and np.all(np.isfinite(observation)),
        lambda: "energy score requires finite ensemble (M,D) and observation (D,)",
    )
    weights = (
        np.ones(ensemble.shape[1], dtype=np.float64)
        if component_weights is None
        else np.asarray(component_weights, dtype=np.float64)
    )
    verify_inp(
        weights.shape == (ensemble.shape[1],)
        and np.all(np.isfinite(weights))
        and np.all(weights >= 0.0)
        and np.any(weights > 0.0),
        lambda: "component weights must be finite, nonnegative, and nonzero",
    )

    def norms(values: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum(weights * np.square(values), axis=-1))

    first = float(np.mean(norms(ensemble - observation)))
    scaled = ensemble * np.sqrt(weights)[None, :]
    squared_norms = np.sum(np.square(scaled), axis=1)
    pair_sum = 0.0
    block_size = min(512, ensemble.shape[0])
    # Block the O(M^2) Gram calculation to keep memory bounded at 5,000 paths.
    for left_start in range(0, ensemble.shape[0], block_size):
        left_stop = min(left_start + block_size, ensemble.shape[0])
        left = scaled[left_start:left_stop]
        for right_start in range(left_start, ensemble.shape[0], block_size):
            right_stop = min(right_start + block_size, ensemble.shape[0])
            right = scaled[right_start:right_stop]
            distance_squared = (
                squared_norms[left_start:left_stop, None]
                + squared_norms[None, right_start:right_stop]
                - 2.0 * left @ right.T
            )
            distances = np.sqrt(np.maximum(distance_squared, 0.0))
            if right_start == left_start:
                pair_sum += float(np.sum(np.triu(distances, k=1)))
            else:
                pair_sum += float(np.sum(distances))
    second = pair_sum / (ensemble.shape[0] * (ensemble.shape[0] - 1))
    return first - second


def variogram_score(
    ensemble: np.ndarray,
    observation: np.ndarray,
    *,
    pairs: np.ndarray,
    pair_weights: np.ndarray | None = None,
    order: float = 0.5,
) -> float:
    """Compute a finite-ensemble variogram score on prespecified pairs."""

    ensemble = np.asarray(ensemble, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64)
    verify_inp(
        ensemble.ndim == 2
        and observation.shape == (ensemble.shape[1],)
        and pairs.ndim == 2
        and pairs.shape[1] == 2
        and pairs.size > 0
        and np.all(pairs >= 0)
        and np.all(pairs < ensemble.shape[1]),
        lambda: "invalid ensemble, observation, or variogram pairs",
    )
    verify_inp(
        math.isfinite(order) and 0.0 < order < 2.0,
        lambda: f"variogram order must lie in (0,2), got {order!r}",
    )
    weights = (
        np.ones(pairs.shape[0], dtype=np.float64)
        if pair_weights is None
        else np.asarray(pair_weights, dtype=np.float64)
    )
    verify_inp(
        weights.shape == (pairs.shape[0],)
        and np.all(np.isfinite(weights))
        and np.all(weights >= 0.0),
        lambda: "invalid variogram pair weights",
    )
    left = pairs[:, 0]
    right = pairs[:, 1]
    observed = np.abs(observation[left] - observation[right]) ** order
    predicted = np.mean(
        np.abs(ensemble[:, left] - ensemble[:, right]) ** order,
        axis=0,
    )
    return float(np.sum(weights * np.square(observed - predicted)))


def paired_block_bootstrap_interval(
    score_difference: np.ndarray,
    *,
    block_length: int = 20,
    confidence: float = 0.95,
    samples: int = 2_000,
    seed: int = 15_040_033,
) -> tuple[float, float, float]:
    """Return mean and percentile interval for paired rolling-origin scores."""

    values = np.asarray(score_difference, dtype=np.float64)
    verify_inp(
        values.ndim == 1 and values.size >= 2 and np.all(np.isfinite(values)),
        lambda: "score differences must be a finite vector with at least two values",
    )
    verify_inp(
        isinstance(block_length, int) and 0 < block_length <= values.size,
        lambda: f"invalid block length {block_length!r} for {values.size} scores",
    )
    verify_inp(
        0.0 < confidence < 1.0 and isinstance(samples, int) and samples > 0,
        lambda: "confidence and bootstrap sample count are invalid",
    )
    rng = np.random.default_rng(seed)
    starts = np.arange(values.size - block_length + 1)
    blocks_needed = math.ceil(values.size / block_length)
    bootstrap = np.empty(samples, dtype=np.float64)
    offsets = np.arange(block_length)
    for sample in range(samples):
        selected_starts = rng.choice(starts, size=blocks_needed, replace=True)
        index = (selected_starts[:, None] + offsets[None, :]).ravel()[: values.size]
        bootstrap[sample] = np.mean(values[index])
    tail = 0.5 * (1.0 - confidence)
    return (
        float(np.mean(values)),
        float(np.quantile(bootstrap, tail)),
        float(np.quantile(bootstrap, 1.0 - tail)),
    )
