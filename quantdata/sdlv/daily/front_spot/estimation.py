"""Estimate locally fixed-contract daily DLV and front-spot innovations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.special import ndtr, ndtri

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.rolling_hedging import one_session_pairs
from quantdata.sdlv.daily import DATASET_PATH, load_raw_grids

MIN_DLV_SQ = 0.01**2
MAX_DLV_SQ = 3.0**2
BACKGROUND_ANNUAL_VOL = 0.05


@dataclass(frozen=True)
class AlignedDlvHistory:
    """One-session fixed-expiry and fixed-cash-strike DLV observations."""

    dates: tuple[str, ...]
    origin_indices: NDArray[np.int64]
    endpoint_indices: NDArray[np.int64]
    origin_log_dlvsq: NDArray[np.float64]
    endpoint_log_dlvsq: NDArray[np.float64]
    spot_gaussian_scores: NDArray[np.float64]
    front_cdf_values: NDArray[np.float64]

    @property
    def increments(self) -> NDArray[np.float64]:
        """Return aligned one-session log-DLV-squared increments."""
        return self.endpoint_log_dlvsq - self.origin_log_dlvsq


@dataclass(frozen=True)
class JointInnovationFit:
    """Physical AR drift and joint spot/surface diffusion estimate."""

    conditional_intercept: NDArray[np.float64]
    conditional_slope: NDArray[np.float64]
    centered_residuals: NDArray[np.float64]
    spot_score_mean: float
    spot_score_std: float
    annualized_loadings: NDArray[np.float64]
    daily_eigenvalues: NDArray[np.float64]
    explained_variance_fraction: float
    factor_scores: NDArray[np.float64]

    @property
    def factor_rank(self) -> int:
        """Number of retained Brownian factors, including the spot factor."""
        return self.annualized_loadings.shape[-1]


@dataclass(frozen=True)
class PhysicalMeasureChangeFit:
    """Projection of empirical physical drift onto an equivalent measure change."""

    empirical_log_drift: NDArray[np.float64]
    risk_neutral_log_drift: NDArray[np.float64]
    achievable_log_drift: NDArray[np.float64]
    market_price_of_risk: NDArray[np.float64]
    constant_achievable_log_drift: NDArray[np.float64]
    constant_market_price_of_risk: NDArray[np.float64]
    coordinate_weights: NDArray[np.float64]
    weighted_target_rms: float
    weighted_residual_rms: float
    achievable_fraction: float
    constant_weighted_residual_rms: float
    constant_achievable_fraction: float

    @property
    def factor_rank(self) -> int:
        """Number of estimated Brownian risk prices."""
        return self.market_price_of_risk.shape[-1]


def _density0_from_support(support: NDArray[np.float64]) -> NDArray[np.float64]:
    calls = np.maximum(1.0 - support, 0.0)
    slopes = np.diff(calls, prepend=1.0) / np.diff(support, prepend=0.0)
    density = np.diff(slopes, append=0.0)
    if np.min(density) < -1.0e-12:
        raise FloatingPointError("Initial lattice density is negative")
    density = np.maximum(density, 0.0)
    return density / np.sum(density)


def _solve_sanos_step(
    support: NDArray[np.float64],
    previous: NDArray[np.float64],
    dlvsq_dt: NDArray[np.float64],
) -> NDArray[np.float64]:
    box = support[1:-1] ** 2 / (support[2:] - support[:-2])
    w_plus = dlvsq_dt * box / (support[2:] - support[1:-1])
    w_minus = dlvsq_dt * box / (support[1:-1] - support[:-2])
    lower = np.concatenate((np.zeros(2), -w_plus))
    diagonal = np.concatenate(
        ((1.0,), 1.0 + w_minus + w_plus, (1.0,))
    )
    upper = np.concatenate((-w_minus, np.zeros(2)))
    size = len(previous)
    modified_upper = np.empty(size, dtype=np.float64)
    modified_rhs = np.empty(size, dtype=np.float64)
    modified_upper[0] = upper[0] / diagonal[0]
    modified_rhs[0] = previous[0] / diagonal[0]
    for index in range(1, size):
        denominator = diagonal[index] - lower[index] * modified_upper[index - 1]
        modified_upper[index] = upper[index] / denominator
        modified_rhs[index] = (
            previous[index] - lower[index] * modified_rhs[index - 1]
        ) / denominator
    result = np.empty(size, dtype=np.float64)
    result[-1] = modified_rhs[-1]
    for index in range(size - 2, -1, -1):
        result[index] = modified_rhs[index] - modified_upper[index] * result[index + 1]
    return result


def _front_density(
    support: NDArray[np.float64],
    dlvsq: NDArray[np.float64],
) -> NDArray[np.float64]:
    density = _solve_sanos_step(
        support,
        _density0_from_support(support),
        dlvsq / DAYS_PER_YEAR,
    )
    if np.min(density) < -1.0e-11:
        raise FloatingPointError("Decoded front density is negative")
    density = np.maximum(density, 0.0)
    return density / np.sum(density)


def _front_spot_cdf(
    gross_spot_return: float,
    support: NDArray[np.float64],
    density: NDArray[np.float64],
) -> float:
    variance = BACKGROUND_ANNUAL_VOL**2 / DAYS_PER_YEAR
    standardized = (
        np.log(gross_spot_return / support) + 0.5 * variance
    ) / np.sqrt(variance)
    return float(density @ ndtr(standardized))


def decode_dlvsq(raw_grids: NDArray[np.float32]) -> NDArray[np.float64]:
    """Decode fitted raw coordinates into bounded DLV-squared values."""
    maximum_raw = np.log(MAX_DLV_SQ - MIN_DLV_SQ)
    return MIN_DLV_SQ + np.exp(
        np.minimum(np.asarray(raw_grids, dtype=np.float64), maximum_raw)
    )


def align_fixed_contract_history(
    *,
    fit_label: str = "full",
    training_end: date = date(2024, 12, 31),
) -> AlignedDlvHistory:
    """Build locally fixed-expiry/fixed-strike one-session observations.

    Each pair uses the origin day's floating relative-strike grid. The cash
    strike is held fixed inside that pair, so the endpoint surface is queried
    at the origin relative strike divided by the realized gross spot return.
    Across pairs the cash grid is allowed to float with the origin spot and
    fitted support.
    """
    with np.load(DATASET_PATH, allow_pickle=False) as dataset:
        dates = tuple(str(value) for value in dataset["dates"])
        supports = np.asarray(dataset["supports"], dtype=np.float64)
        spots = np.asarray(dataset["spot_ref"], dtype=np.float64)
    if spots.shape != (len(dates),) or np.any(~np.isfinite(spots) | (spots <= 0.0)):
        raise ValueError("Dataset reference spots are missing or invalid")
    parsed_dates = np.asarray([date.fromisoformat(value) for value in dates], dtype=object)
    origin, endpoint = one_session_pairs(list(dates))
    selected = parsed_dates[endpoint] <= training_end
    origin = np.asarray(origin[selected], dtype=np.int64)
    endpoint = np.asarray(endpoint[selected], dtype=np.int64)
    if len(origin) < 100:
        raise ValueError("Too few exact one-session training pairs")

    dlvsq = decode_dlvsq(load_raw_grids(fit_label))
    log_dlvsq = np.log(dlvsq)
    pair_count = len(origin)
    expiry_count = dlvsq.shape[1] - 1
    strike_count = dlvsq.shape[2]
    origin_values = np.asarray(log_dlvsq[origin, 1:, :], dtype=np.float64)
    endpoint_values = np.empty(
        (pair_count, expiry_count, strike_count), dtype=np.float64
    )
    front_cdf = np.empty(pair_count, dtype=np.float64)

    for pair_index, (origin_index, endpoint_index) in enumerate(
        zip(origin, endpoint, strict=True)
    ):
        gross_return = spots[endpoint_index] / spots[origin_index]
        origin_strikes = supports[origin_index, 1:-1]
        endpoint_query = origin_strikes / gross_return
        endpoint_strikes = supports[endpoint_index, 1:-1]
        for expiry_index in range(expiry_count):
            endpoint_values[pair_index, expiry_index] = np.interp(
                endpoint_query,
                endpoint_strikes,
                log_dlvsq[endpoint_index, expiry_index],
            )
        density = _front_density(
            supports[origin_index],
            dlvsq[origin_index, 0],
        )
        front_cdf[pair_index] = _front_spot_cdf(
            gross_return,
            supports[origin_index],
            density,
        )

    probability_floor = np.finfo(np.float64).eps
    spot_scores = ndtri(np.clip(front_cdf, probability_floor, 1.0 - probability_floor))
    if np.any(~np.isfinite(endpoint_values)) or np.any(~np.isfinite(spot_scores)):
        raise FloatingPointError("Aligned DLV history contains non-finite values")
    return AlignedDlvHistory(
        dates=dates,
        origin_indices=origin,
        endpoint_indices=endpoint,
        origin_log_dlvsq=origin_values,
        endpoint_log_dlvsq=endpoint_values,
        spot_gaussian_scores=spot_scores,
        front_cdf_values=front_cdf,
    )


def _low_rank_loadings(
    residuals: NDArray[np.float64],
    rank: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    sample_count = residuals.shape[0]
    if not 0 < rank < sample_count:
        raise ValueError(f"rank must lie in [1,{sample_count - 1}], found {rank}")
    gram = residuals @ residuals.T / sample_count
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    retained = eigenvectors[:, :rank]
    loadings = residuals.T @ retained / np.sqrt(sample_count)
    scores = np.sqrt(sample_count) * retained
    return loadings, eigenvalues, scores


def _noise_corrected_low_rank_loadings(
    residuals: NDArray[np.float64],
    linked_pairs: NDArray[np.bool_],
    rank: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Estimate latent one-day covariance after removing fit observation noise.

    For consecutive observations ``r_t = u_t + e_{t+1} - e_t`` with serially
    independent latent innovations and fit errors, the latent covariance is
    ``Cov(r_t) + Cov(r_t,r_{t+1}) + Cov(r_{t+1},r_t)``.  Only pairs sharing the
    intermediate market date enter the lag-one term.
    """
    sample_count = residuals.shape[0]
    if not 0 < rank < sample_count:
        raise ValueError(f"rank must lie in [1,{sample_count - 1}], found {rank}")
    if linked_pairs.shape != (sample_count - 1,):
        raise ValueError(
            f"linked-pair shape {linked_pairs.shape!r} does not match {sample_count}"
        )
    link_indices = np.flatnonzero(linked_pairs)
    if len(link_indices) < sample_count // 2:
        raise ValueError("Too few consecutive residual pairs for noise correction")

    left_vectors, singular_values, right_vectors = np.linalg.svd(
        residuals,
        full_matrices=False,
    )
    time_operator = np.eye(sample_count, dtype=np.float64) / sample_count
    lag_weight = 1.0 / len(link_indices)
    time_operator[link_indices, link_indices + 1] += lag_weight
    time_operator[link_indices + 1, link_indices] += lag_weight
    reduced_covariance = (
        singular_values[:, None]
        * (left_vectors.T @ time_operator @ left_vectors)
        * singular_values[None, :]
    )
    eigenvalues, reduced_eigenvectors = np.linalg.eigh(
        0.5 * (reduced_covariance + reduced_covariance.T)
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    positive_count = int(np.sum(eigenvalues > np.finfo(np.float64).eps))
    if positive_count < rank:
        raise FloatingPointError(
            f"Noise-corrected covariance has only {positive_count} positive factors"
        )
    retained_values = eigenvalues[:rank]
    spatial_eigenvectors = right_vectors.T @ reduced_eigenvectors[:, order[:rank]]
    loadings = spatial_eigenvectors * np.sqrt(retained_values)[None, :]
    scores = residuals @ spatial_eigenvectors / np.sqrt(retained_values)[None, :]
    return loadings, np.maximum(eigenvalues, 0.0), scores


def fit_joint_innovations(
    history: AlignedDlvHistory,
    *,
    factor_rank: int = 8,
    correct_lag1_fit_noise: bool = False,
    smoothing_sigma: tuple[float, float] | None = None,
) -> JointInnovationFit:
    """Fit physical AR drift and spot-oriented low-rank innovations."""
    if factor_rank < 2:
        raise ValueError("factor_rank must include spot and at least one surface factor")
    origin = history.origin_log_dlvsq
    endpoint = history.endpoint_log_dlvsq
    origin_mean = np.mean(origin, axis=0)
    endpoint_mean = np.mean(endpoint, axis=0)
    centered_origin = origin - origin_mean
    centered_endpoint = endpoint - endpoint_mean
    denominator = np.sum(centered_origin**2, axis=0)
    scale = np.maximum(np.sum(origin**2, axis=0), 1.0)
    variable = denominator > np.finfo(np.float64).eps * scale * len(origin)
    slope = np.zeros_like(denominator)
    slope[variable] = (
        np.sum(centered_origin * centered_endpoint, axis=0)[variable]
        / denominator[variable]
    )
    intercept = endpoint_mean - slope * origin_mean
    residual = endpoint - (intercept + slope * origin)
    residual -= np.mean(residual, axis=0, keepdims=True)
    residual_vectors = residual.reshape(len(residual), -1)
    if smoothing_sigma is None:
        diffusion_residual = residual
    else:
        if len(smoothing_sigma) != 2 or any(
            not np.isfinite(value) or value < 0.0 for value in smoothing_sigma
        ):
            raise ValueError(
                f"smoothing_sigma must contain two non-negative values, found "
                f"{smoothing_sigma!r}"
            )
        diffusion_residual = gaussian_filter(
            residual,
            sigma=(0.0, *smoothing_sigma),
            mode="nearest",
        )
        diffusion_residual -= np.mean(
            diffusion_residual,
            axis=0,
            keepdims=True,
        )
    diffusion_vectors = diffusion_residual.reshape(len(residual), -1)

    raw_spot_score = np.asarray(history.spot_gaussian_scores, dtype=np.float64)
    spot_mean = float(np.mean(raw_spot_score))
    spot_std = float(np.std(raw_spot_score, ddof=0))
    if not np.isfinite(spot_std) or spot_std <= np.finfo(np.float64).tiny:
        raise FloatingPointError("Front Gaussian scores have zero variance")
    spot_score = (raw_spot_score - spot_mean) / spot_std
    spot_loading = np.mean(diffusion_vectors * spot_score[:, None], axis=0)
    orthogonal_residual = diffusion_vectors - spot_score[:, None] * spot_loading
    if correct_lag1_fit_noise:
        linked_pairs = history.endpoint_indices[:-1] == history.origin_indices[1:]
        orthogonal_loadings, eigenvalues, orthogonal_scores = (
            _noise_corrected_low_rank_loadings(
                orthogonal_residual,
                linked_pairs,
                factor_rank - 1,
            )
        )
    else:
        orthogonal_loadings, eigenvalues, orthogonal_scores = _low_rank_loadings(
            orthogonal_residual,
            factor_rank - 1,
        )
    daily_loadings = np.column_stack((spot_loading, orthogonal_loadings))
    annualized_loadings = daily_loadings.reshape(
        *origin.shape[1:], factor_rank
    ) / np.sqrt(1.0 / DAYS_PER_YEAR)
    total_variance = float(np.sum(diffusion_vectors**2) / len(diffusion_vectors))
    retained_variance = float(np.sum(daily_loadings**2))
    factor_scores = np.column_stack((spot_score, orthogonal_scores))
    return JointInnovationFit(
        conditional_intercept=intercept,
        conditional_slope=slope,
        centered_residuals=residual_vectors,
        spot_score_mean=spot_mean,
        spot_score_std=spot_std,
        annualized_loadings=annualized_loadings,
        daily_eigenvalues=eigenvalues,
        explained_variance_fraction=retained_variance / total_variance,
        factor_scores=factor_scores,
    )


def fit_achievable_physical_drift(
    history: AlignedDlvHistory,
    innovations: JointInnovationFit,
    risk_neutral_log_drift: NDArray[np.float64],
    coordinate_weights: NDArray[np.float64],
    *,
    ridge: float = 1.0e-10,
) -> PhysicalMeasureChangeFit:
    """Fit the physical drift achievable through a Brownian measure change.

    The conditional AR fit supplies an empirical annual log-DLV drift at every
    historical state.  Under the convention

    ``dW_Q = dW_P + lambda dt``, its attainable component is
    ``m_P = m_Q + xi lambda``.  The market price ``lambda`` is fitted at each
    state by density-weighted least squares.  Weights should give equal total
    importance to each expiry and downweight economically irrelevant tail
    nodes where relative density volatilities can be very large.
    """
    if ridge < 0.0 or not np.isfinite(ridge):
        raise ValueError(f"ridge must be finite and non-negative, found {ridge}")
    shape = history.origin_log_dlvsq.shape
    q_drift = np.asarray(risk_neutral_log_drift, dtype=np.float64)
    weights = np.asarray(coordinate_weights, dtype=np.float64)
    if q_drift.shape != shape:
        raise ValueError(
            f"risk-neutral drift shape {q_drift.shape!r} does not match {shape!r}"
        )
    if weights.shape != shape:
        raise ValueError(f"weight shape {weights.shape!r} does not match {shape!r}")
    if np.any(~np.isfinite(q_drift)):
        raise FloatingPointError("Risk-neutral log drift contains non-finite values")
    if np.any(~np.isfinite(weights) | (weights < 0.0)):
        raise ValueError("Coordinate weights must be finite and non-negative")

    empirical_daily_increment = (
        innovations.conditional_intercept
        + innovations.conditional_slope * history.origin_log_dlvsq
        - history.origin_log_dlvsq
    )
    empirical_drift = DAYS_PER_YEAR * empirical_daily_increment
    loading_matrix = innovations.annualized_loadings.reshape(-1, innovations.factor_rank)
    target = (empirical_drift - q_drift).reshape(len(history.origin_indices), -1)
    flat_weights = weights.reshape(len(history.origin_indices), -1)
    weight_sums = np.sum(flat_weights, axis=1)
    if np.any(weight_sums <= np.finfo(np.float64).tiny):
        raise ValueError("Every historical state must have positive projection weight")
    flat_weights = flat_weights * (flat_weights.shape[1] / weight_sums[:, None])

    risk_prices = np.empty(
        (len(history.origin_indices), innovations.factor_rank), dtype=np.float64
    )
    projected = np.empty_like(target)
    identity = np.eye(innovations.factor_rank, dtype=np.float64)
    global_gram = np.zeros(
        (innovations.factor_rank, innovations.factor_rank), dtype=np.float64
    )
    global_right_hand_side = np.zeros(innovations.factor_rank, dtype=np.float64)
    for sample_index, (sample_target, sample_weight) in enumerate(
        zip(target, flat_weights, strict=True)
    ):
        weighted_loadings = sample_weight[:, None] * loading_matrix
        gram = loading_matrix.T @ weighted_loadings
        ridge_scale = ridge * max(float(np.trace(gram)) / innovations.factor_rank, 1.0)
        right_hand_side = loading_matrix.T @ (sample_weight * sample_target)
        global_gram += gram
        global_right_hand_side += right_hand_side
        risk_prices[sample_index] = np.linalg.solve(
            gram + ridge_scale * identity,
            right_hand_side,
        )
        projected[sample_index] = loading_matrix @ risk_prices[sample_index]

    residual = target - projected
    weighted_target_square = float(np.sum(flat_weights * target**2))
    weighted_residual_square = float(np.sum(flat_weights * residual**2))
    weight_total = float(np.sum(flat_weights))
    target_rms = np.sqrt(weighted_target_square / weight_total)
    residual_rms = np.sqrt(weighted_residual_square / weight_total)
    achievable_fraction = 1.0
    if weighted_target_square > np.finfo(np.float64).tiny:
        achievable_fraction -= weighted_residual_square / weighted_target_square
    achievable = q_drift + projected.reshape(shape)
    global_ridge_scale = ridge * max(
        float(np.trace(global_gram)) / innovations.factor_rank,
        1.0,
    )
    constant_risk_price = np.linalg.solve(
        global_gram + global_ridge_scale * identity,
        global_right_hand_side,
    )
    constant_projected = loading_matrix @ constant_risk_price
    constant_residual = target - constant_projected[None, :]
    constant_residual_square = float(
        np.sum(flat_weights * constant_residual**2)
    )
    constant_residual_rms = np.sqrt(constant_residual_square / weight_total)
    constant_achievable_fraction = 1.0
    if weighted_target_square > np.finfo(np.float64).tiny:
        constant_achievable_fraction -= (
            constant_residual_square / weighted_target_square
        )
    return PhysicalMeasureChangeFit(
        empirical_log_drift=empirical_drift,
        risk_neutral_log_drift=q_drift,
        achievable_log_drift=achievable,
        market_price_of_risk=risk_prices,
        constant_achievable_log_drift=(
            q_drift + constant_projected.reshape((1,) + shape[1:])
        ),
        constant_market_price_of_risk=constant_risk_price,
        coordinate_weights=flat_weights.reshape(shape),
        weighted_target_rms=float(target_rms),
        weighted_residual_rms=float(residual_rms),
        achievable_fraction=float(achievable_fraction),
        constant_weighted_residual_rms=float(constant_residual_rms),
        constant_achievable_fraction=float(constant_achievable_fraction),
    )
