"""Rolling fixed-contract DLV innovation factors for the hedging benchmark."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version
from numpy.typing import NDArray
from scipy.linalg import solve_banded
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import logsumexp, ndtr, ndtri

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.hedging.data.normalization import (
    interpolate_positive_curve_at_business_dte,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    OneDayTransition,
    PublicHedgingPanel,
    PublicMarketDay,
)
from quantdata.sdlv.daily.front_spot.hedging.data.public_only import (
    require_public_hedging_panel,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@version("1.2")
@dataclass(frozen=True, kw_only=True)
class RollingDlvFactorConfig:
    """Controls for one-day fixed-contract DLV factor estimation."""

    window_sessions: int = 252
    factor_rank: int = 8
    smoothing_expiry_sigma: float = 1.0
    smoothing_strike_sigma: float = 2.0

    def __post_init__(self) -> None:
        verify_inp(
            self.window_sessions >= 3,
            lambda: "window_sessions must be at least three",
        )
        verify_inp(
            2 <= self.factor_rank < self.window_sessions,
            lambda: "factor_rank must include spot and be smaller than the window",
        )
        verify_inp(
            np.isfinite(self.smoothing_expiry_sigma)
            and self.smoothing_expiry_sigma >= 0.0,
            lambda: "smoothing_expiry_sigma must be finite and non-negative",
        )
        verify_inp(
            np.isfinite(self.smoothing_strike_sigma)
            and self.smoothing_strike_sigma >= 0.0,
            lambda: "smoothing_strike_sigma must be finite and non-negative",
        )


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class RollingFactorAlignmentConfig:
    """Controls for aligning consecutive rolling DLV factor bases.

    A PCA direction is not identified inside an eigenspace whose eigenvalues
    cannot be distinguished at the rolling-window sampling precision.  The
    default treats adjacent eigenvalues as near-degenerate when their log gap
    is below two asymptotic relative-eigenvalue standard errors,
    ``2 sqrt(2 / n)``.
    """

    degenerate_standard_errors: float = 2.0
    minimum_loading_norm: float = 1.0e-14

    def __post_init__(self) -> None:
        verify_inp(
            np.isfinite(self.degenerate_standard_errors)
            and self.degenerate_standard_errors >= 0.0,
            lambda: (
                "degenerate_standard_errors must be finite and non-negative"
            ),
        )
        verify_inp(
            np.isfinite(self.minimum_loading_norm)
            and self.minimum_loading_norm > 0.0,
            lambda: "minimum_loading_norm must be finite and positive",
        )


@dataclass(frozen=True, kw_only=True)
class FixedContractDlvPair:
    """One exact-session DLV transition in origin fixed-contract coordinates."""

    origin_date: date
    endpoint_date: date
    origin_log_dlvsq: FloatArray
    endpoint_log_dlvsq: FloatArray
    front_cdf: float
    spot_gaussian_score: float
    front_support: FloatArray
    front_density: FloatArray
    terminal_normalized_spot: float
    background_annual_volatility: float
    interpolation_clip_fraction: float
    forward_tail_extrapolation_fraction: float

    def __post_init__(self) -> None:
        if self.endpoint_date <= self.origin_date:
            raise ValueError("DLV pair endpoint must follow its origin")
        if (
            self.origin_log_dlvsq.ndim != 2
            or self.endpoint_log_dlvsq.shape != self.origin_log_dlvsq.shape
            or not np.all(np.isfinite(self.origin_log_dlvsq))
            or not np.all(np.isfinite(self.endpoint_log_dlvsq))
        ):
            raise ValueError("DLV pair states must be aligned finite matrices")
        if not 0.0 < self.front_cdf < 1.0:
            raise ValueError("front_cdf must lie strictly between zero and one")
        if not np.isfinite(self.spot_gaussian_score):
            raise ValueError("spot_gaussian_score must be finite")
        if (
            self.front_support.ndim != 1
            or self.front_density.shape != self.front_support.shape
            or np.any(~np.isfinite(self.front_support))
            or np.any(np.diff(self.front_support) <= 0.0)
            or np.any(~np.isfinite(self.front_density))
            or np.any(self.front_density < 0.0)
            or not np.isclose(np.sum(self.front_density), 1.0)
            or not np.isfinite(self.terminal_normalized_spot)
            or self.terminal_normalized_spot <= 0.0
            or not np.isfinite(self.background_annual_volatility)
            or self.background_annual_volatility <= 0.0
        ):
            raise ValueError("front-mixture observation is invalid")
        if not 0.0 <= self.interpolation_clip_fraction <= 1.0:
            raise ValueError("interpolation_clip_fraction must lie in [0,1]")
        if not 0.0 <= self.forward_tail_extrapolation_fraction <= 1.0:
            raise ValueError("forward_tail_extrapolation_fraction must lie in [0,1]")


@dataclass(frozen=True, kw_only=True)
class FixedContractDlvHistory:
    """Rolling collection of fixed-expiry, fixed-cash-strike DLV pairs."""

    origin_dates: tuple[date, ...]
    endpoint_dates: tuple[date, ...]
    origin_log_dlvsq: FloatArray
    endpoint_log_dlvsq: FloatArray
    front_cdf_values: FloatArray
    spot_gaussian_scores: FloatArray
    interpolation_clip_fractions: FloatArray
    forward_tail_extrapolation_fractions: FloatArray
    front_supports: FloatArray | None = None
    front_densities: FloatArray | None = None
    terminal_normalized_spots: FloatArray | None = None
    background_annual_volatilities: FloatArray | None = None

    def __post_init__(self) -> None:
        sample_count = len(self.origin_dates)
        if sample_count == 0 or len(self.endpoint_dates) != sample_count:
            raise ValueError("DLV history dates must be non-empty and aligned")
        if any(
            left >= right
            for left, right in zip(self.origin_dates, self.endpoint_dates, strict=True)
        ):
            raise ValueError("every DLV history endpoint must follow its origin")
        if any(
            self.origin_dates[index] >= self.origin_dates[index + 1]
            for index in range(sample_count - 1)
        ):
            raise ValueError("DLV history origins must be strictly increasing")
        expected_prefix = (sample_count,)
        if (
            self.origin_log_dlvsq.ndim != 3
            or self.origin_log_dlvsq.shape[:1] != expected_prefix
            or self.endpoint_log_dlvsq.shape != self.origin_log_dlvsq.shape
        ):
            raise ValueError("DLV history state arrays do not align")
        vectors = (
            self.front_cdf_values,
            self.spot_gaussian_scores,
            self.interpolation_clip_fractions,
            self.forward_tail_extrapolation_fractions,
        )
        if any(values.shape != expected_prefix for values in vectors):
            raise ValueError("DLV history diagnostic vectors do not align")
        if any(not np.all(np.isfinite(values)) for values in vectors) or not np.all(
            np.isfinite(self.origin_log_dlvsq)
            & np.isfinite(self.endpoint_log_dlvsq)
        ):
            raise ValueError("DLV history must be finite")
        mixture_arrays = (
            self.front_supports,
            self.front_densities,
            self.terminal_normalized_spots,
            self.background_annual_volatilities,
        )
        if any(values is None for values in mixture_arrays):
            if not all(values is None for values in mixture_arrays):
                raise ValueError("front-mixture history arrays must be supplied together")
        else:
            assert self.front_supports is not None
            assert self.front_densities is not None
            assert self.terminal_normalized_spots is not None
            assert self.background_annual_volatilities is not None
            if (
                self.front_supports.ndim != 2
                or self.front_supports.shape != self.front_densities.shape
                or self.front_supports.shape[0] != sample_count
                or self.terminal_normalized_spots.shape != expected_prefix
                or self.background_annual_volatilities.shape != expected_prefix
                or np.any(~np.isfinite(self.front_supports))
                or np.any(np.diff(self.front_supports, axis=1) <= 0.0)
                or np.any(~np.isfinite(self.front_densities))
                or np.any(self.front_densities < 0.0)
                or not np.allclose(np.sum(self.front_densities, axis=1), 1.0)
                or np.any(~np.isfinite(self.terminal_normalized_spots))
                or np.any(self.terminal_normalized_spots <= 0.0)
                or np.any(~np.isfinite(self.background_annual_volatilities))
                or np.any(self.background_annual_volatilities <= 0.0)
            ):
                raise ValueError("front-mixture history arrays are invalid")

    @property
    def increments(self) -> FloatArray:
        """Return one-session fixed-contract log-DLV-squared increments."""
        return self.endpoint_log_dlvsq - self.origin_log_dlvsq

    @property
    def has_front_mixture_observations(self) -> bool:
        """Whether exact DLV-front/background likelihood inputs are present."""
        return self.front_supports is not None


@dataclass(frozen=True, kw_only=True)
class RollingDlvFactorFit:
    """Physical AR fit and spot-oriented low-rank daily DLV innovations."""

    as_of_date: date
    training_origin_dates: tuple[date, ...]
    training_endpoint_dates: tuple[date, ...]
    conditional_intercept: FloatArray
    conditional_slope: FloatArray
    centered_residuals: FloatArray
    spot_score_mean: float
    spot_score_std: float
    daily_loadings: FloatArray
    daily_factor_variances: FloatArray
    factor_scores: FloatArray
    explained_variance_fraction: float
    interpolation_clip_fraction: float
    forward_tail_extrapolation_fraction: float
    front_spot_market_price_of_risk: float = 0.0
    front_spot_market_price_of_risk_standard_error: float = 1.0

    def __post_init__(self) -> None:
        state_shape = self.conditional_intercept.shape
        sample_count = len(self.training_origin_dates)
        if self.conditional_slope.shape != state_shape or len(state_shape) != 2:
            raise ValueError("conditional DLV coefficient shapes do not align")
        if self.centered_residuals.shape != (sample_count, *state_shape):
            raise ValueError("centered DLV residual shape does not align")
        if self.daily_loadings.shape[:2] != state_shape:
            raise ValueError("daily DLV loading coordinates do not align")
        factor_rank = self.daily_loadings.shape[-1]
        if (
            self.daily_factor_variances.shape != (factor_rank,)
            or self.factor_scores.shape != (sample_count, factor_rank)
        ):
            raise ValueError("DLV factor arrays do not align")
        arrays = (
            self.conditional_intercept,
            self.conditional_slope,
            self.centered_residuals,
            self.daily_loadings,
            self.daily_factor_variances,
            self.factor_scores,
        )
        if (
            not np.isfinite(self.front_spot_market_price_of_risk)
            or not np.isfinite(self.front_spot_market_price_of_risk_standard_error)
            or self.front_spot_market_price_of_risk_standard_error <= 0.0
        ):
            raise ValueError("front-spot risk-price estimate is invalid")
        if any(not np.all(np.isfinite(values)) for values in arrays):
            raise ValueError("DLV factor fit must be finite")
        if not (
            np.isfinite(self.spot_score_mean)
            and np.isfinite(self.spot_score_std)
            and self.spot_score_std > 0.0
            and 0.0 <= self.explained_variance_fraction <= 1.0 + 1.0e-12
            and 0.0 <= self.interpolation_clip_fraction <= 1.0
            and 0.0 <= self.forward_tail_extrapolation_fraction <= 1.0
        ):
            raise ValueError("DLV factor diagnostics are invalid")
        if any(endpoint > self.as_of_date for endpoint in self.training_endpoint_dates):
            raise ValueError("DLV factor fit contains an endpoint after its as-of date")

    @property
    def factor_rank(self) -> int:
        """Return the retained Brownian factor count, including spot."""
        return self.daily_loadings.shape[-1]

    @property
    def annualized_loadings(self) -> FloatArray:
        """Return log-DLV-squared diffusion loadings per square-root year."""
        return self.daily_loadings * np.sqrt(DAYS_PER_YEAR)


@dataclass(frozen=True, kw_only=True)
class RollingFactorAlignment:
    """An aligned rolling fit and diagnostics relative to its predecessor."""

    fit: RollingDlvFactorFit
    surface_factor_permutation: tuple[int, ...]
    surface_factor_signs: tuple[int, ...]
    procrustes_blocks: tuple[tuple[int, ...], ...]
    pre_procrustes_cosines: FloatArray
    post_procrustes_cosines: FloatArray
    log_eigenvalue_gap_tolerance: float

    def __post_init__(self) -> None:
        surface_rank = self.fit.factor_rank - 1
        if (
            len(self.surface_factor_permutation) != surface_rank
            or len(self.surface_factor_signs) != surface_rank
            or sorted(self.surface_factor_permutation)
            != list(range(1, self.fit.factor_rank))
        ):
            raise ValueError("surface-factor alignment metadata is invalid")
        if any(sign not in (-1, 1) for sign in self.surface_factor_signs):
            raise ValueError("surface-factor alignment signs must be +/-1")
        if (
            self.pre_procrustes_cosines.shape != (surface_rank,)
            or self.post_procrustes_cosines.shape != (surface_rank,)
            or not np.all(np.isfinite(self.pre_procrustes_cosines))
            or not np.all(np.isfinite(self.post_procrustes_cosines))
        ):
            raise ValueError("surface-factor cosine diagnostics are invalid")
        if not np.isfinite(self.log_eigenvalue_gap_tolerance):
            raise ValueError("alignment eigenvalue tolerance must be finite")


def align_rolling_dlv_factors(
    reference: RollingDlvFactorFit,
    current: RollingDlvFactorFit,
    *,
    config: RollingFactorAlignmentConfig = RollingFactorAlignmentConfig(),
) -> RollingFactorAlignment:
    """Align one rolling factor basis to the preceding fitted basis.

    Factor zero remains fixed because it is identified by the front-spot
    Gaussian score.  The remaining factors are first matched by maximum
    absolute loading cosine, then sign-aligned.  Contiguous factors whose
    eigenvalues are statistically indistinguishable in both windows receive
    an orthogonal Procrustes rotation.  Applying the same orthogonal transform
    to loadings and scores preserves the fitted innovation reconstruction and
    covariance exactly up to floating-point error.
    """
    if reference.as_of_date >= current.as_of_date:
        raise ValueError(
            "reference factor fit must strictly precede the current as-of date"
        )
    if reference.factor_rank != current.factor_rank:
        raise ValueError("rolling factor ranks must match before alignment")
    if reference.daily_loadings.shape != current.daily_loadings.shape:
        raise ValueError("rolling factor loading grids must match before alignment")
    if current.factor_rank < 2:
        raise ValueError("rolling factor alignment requires a surface factor")

    reference_flat = reference.daily_loadings.reshape(-1, reference.factor_rank)
    current_flat = current.daily_loadings.reshape(-1, current.factor_rank)
    reference_surface = reference_flat[:, 1:]
    current_surface = current_flat[:, 1:]
    reference_norms = np.linalg.norm(reference_surface, axis=0)
    current_norms = np.linalg.norm(current_surface, axis=0)
    if (
        np.any(reference_norms <= config.minimum_loading_norm)
        or np.any(current_norms <= config.minimum_loading_norm)
    ):
        raise FloatingPointError("cannot align a numerically zero DLV factor")

    cosine_matrix = (
        reference_surface.T @ current_surface
        / (reference_norms[:, None] * current_norms[None, :])
    )
    reference_indices, current_indices = linear_sum_assignment(
        -np.abs(cosine_matrix)
    )
    order = np.argsort(reference_indices)
    surface_permutation_zero_based = current_indices[order]
    matched = current_surface[:, surface_permutation_zero_based].copy()
    matched_scores = current.factor_scores[:, 1:][
        :, surface_permutation_zero_based
    ].copy()
    signs = np.where(
        np.sum(reference_surface * matched, axis=0) < 0.0,
        -1,
        1,
    )
    matched *= signs[None, :]
    matched_scores *= signs[None, :]
    pre_cosines = _column_cosines(reference_surface, matched)

    reference_variances = np.sum(reference_surface**2, axis=0)
    matched_variances = np.sum(matched**2, axis=0)
    sample_count = min(
        len(reference.training_origin_dates),
        len(current.training_origin_dates),
    )
    if sample_count < 2:
        raise ValueError("factor alignment requires at least two observations")
    log_gap_tolerance = (
        config.degenerate_standard_errors * np.sqrt(2.0 / sample_count)
    )
    blocks = _near_degenerate_blocks(
        reference_variances,
        matched_variances,
        log_gap_tolerance=log_gap_tolerance,
    )
    for block in blocks:
        block_indices = np.asarray(block, dtype=np.int64)
        cross_product = matched[:, block_indices].T @ reference_surface[:, block_indices]
        left, _, right_transpose = np.linalg.svd(cross_product, full_matrices=False)
        rotation = left @ right_transpose
        matched[:, block_indices] = matched[:, block_indices] @ rotation
        matched_scores[:, block_indices] = matched_scores[:, block_indices] @ rotation

    aligned_flat = np.column_stack((current_flat[:, 0], matched))
    aligned_scores = np.column_stack((current.factor_scores[:, 0], matched_scores))
    aligned_loadings = aligned_flat.reshape(current.daily_loadings.shape)
    aligned_variances = np.sum(aligned_flat**2, axis=0)
    aligned_fit = replace(
        current,
        daily_loadings=aligned_loadings,
        daily_factor_variances=aligned_variances,
        factor_scores=aligned_scores,
    )
    return RollingFactorAlignment(
        fit=aligned_fit,
        surface_factor_permutation=tuple(
            int(index + 1) for index in surface_permutation_zero_based
        ),
        surface_factor_signs=tuple(int(sign) for sign in signs),
        procrustes_blocks=tuple(
            tuple(int(index + 1) for index in block) for block in blocks
        ),
        pre_procrustes_cosines=pre_cosines,
        post_procrustes_cosines=_column_cosines(reference_surface, matched),
        log_eigenvalue_gap_tolerance=float(log_gap_tolerance),
    )


def _column_cosines(left: FloatArray, right: FloatArray) -> FloatArray:
    left_norms = np.linalg.norm(left, axis=0)
    right_norms = np.linalg.norm(right, axis=0)
    return np.sum(left * right, axis=0) / (left_norms * right_norms)


def _near_degenerate_blocks(
    reference_variances: FloatArray,
    current_variances: FloatArray,
    *,
    log_gap_tolerance: float,
) -> tuple[tuple[int, ...], ...]:
    if reference_variances.shape != current_variances.shape:
        raise ValueError("factor variance vectors must have equal shape")
    if np.any(reference_variances <= 0.0) or np.any(current_variances <= 0.0):
        raise FloatingPointError("factor variances must be strictly positive")
    linked = (
        np.abs(np.diff(np.log(reference_variances))) <= log_gap_tolerance
    ) & (np.abs(np.diff(np.log(current_variances))) <= log_gap_tolerance)
    blocks: list[tuple[int, ...]] = []
    start = 0
    for index, is_linked in enumerate(linked):
        if is_linked:
            continue
        if index + 1 - start >= 2:
            blocks.append(tuple(range(start, index + 1)))
        start = index + 1
    if len(reference_variances) - start >= 2:
        blocks.append(tuple(range(start, len(reference_variances))))
    return tuple(blocks)


def align_fixed_contract_pair(
    transition: OneDayTransition,
    endpoint: PublicMarketDay,
) -> FixedContractDlvPair:
    """Align one exact-session transition at fixed cash strike and expiry.

    Origin DLV slab ``h+1`` and endpoint slab ``h`` represent the same calendar
    expiry. For every origin ``K/F_t(T)`` coordinate, the endpoint query is

    ``K/F_{t+1}(T) = [K/F_t(T)] [F_t(T)/S_t] / [S_{t+1}/S_t] / [F_{t+1}(T)/S_{t+1}]``.
    """
    origin = transition.origin
    if endpoint.trading_date != transition.next_date:
        raise ValueError("endpoint market day does not match the transition")
    origin_dtes = _integer_model_dtes(origin)
    endpoint_dtes = _integer_model_dtes(endpoint)
    if len(origin_dtes) < 2 or len(endpoint_dtes) != len(origin_dtes):
        raise ValueError("DLV factor alignment needs equal multi-expiry grids")
    if not np.array_equal(origin_dtes[1:] - 1, endpoint_dtes[:-1]):
        raise ValueError("origin and endpoint DLV slabs do not share calendar expiries")
    if origin.dlvsq.shape != endpoint.dlvsq.shape:
        raise ValueError("origin and endpoint DLV states have different shapes")

    origin_interior = origin.dlv_support_strikes[1:-1]
    endpoint_interior = endpoint.dlv_support_strikes[1:-1]
    origin_values = np.log(origin.dlvsq[1:])
    endpoint_log_dlvsq = np.log(endpoint.dlvsq[:-1])
    aligned_endpoint = np.empty_like(origin_values)
    clipped = 0
    total = aligned_endpoint.size
    forward_tail_extrapolations = 0
    forward_queries = 2 * len(aligned_endpoint)
    gross_spot_return = (
        transition.forward_over_spot * transition.terminal_normalized_spot
    )

    for row, (origin_dte, endpoint_dte) in enumerate(
        zip(origin_dtes[1:], endpoint_dtes[:-1], strict=True)
    ):
        origin_forward, origin_extrapolated = _model_forward_over_spot(
            origin,
            target_business_dte=int(origin_dte),
        )
        endpoint_forward, endpoint_extrapolated = _model_forward_over_spot(
            endpoint,
            target_business_dte=int(endpoint_dte),
        )
        forward_tail_extrapolations += int(origin_extrapolated)
        forward_tail_extrapolations += int(endpoint_extrapolated)
        endpoint_query = (
            origin_interior
            * origin_forward
            / (gross_spot_return * endpoint_forward)
        )
        clipped += int(
            np.count_nonzero(
                (endpoint_query < endpoint_interior[0])
                | (endpoint_query > endpoint_interior[-1])
            )
        )
        aligned_endpoint[row] = np.interp(
            endpoint_query,
            endpoint_interior,
            endpoint_log_dlvsq[row],
        )

    front_density = _decode_front_density(origin)
    front_cdf = _front_spot_cdf(
        terminal_normalized_spot=transition.terminal_normalized_spot,
        support=origin.dlv_support_strikes,
        density=front_density,
        background_annual_volatility=origin.backbone_annual_volatility,
    )
    probability_floor = np.finfo(np.float64).eps
    clipped_cdf = float(np.clip(front_cdf, probability_floor, 1.0 - probability_floor))
    return FixedContractDlvPair(
        origin_date=origin.trading_date,
        endpoint_date=endpoint.trading_date,
        origin_log_dlvsq=origin_values,
        endpoint_log_dlvsq=aligned_endpoint,
        front_cdf=clipped_cdf,
        spot_gaussian_score=float(ndtri(clipped_cdf)),
        front_support=np.asarray(origin.dlv_support_strikes, dtype=np.float64),
        front_density=front_density,
        terminal_normalized_spot=transition.terminal_normalized_spot,
        background_annual_volatility=origin.backbone_annual_volatility,
        interpolation_clip_fraction=clipped / total,
        forward_tail_extrapolation_fraction=(
            forward_tail_extrapolations / forward_queries
        ),
    )


def build_fixed_contract_dlv_history(
    panel: PublicHedgingPanel,
    *,
    origin_dates: tuple[date, ...],
) -> FixedContractDlvHistory:
    """Construct a strictly ordered fixed-contract history from public rows."""
    if not origin_dates:
        raise ValueError("origin_dates must be non-empty")
    if any(
        origin_dates[index] >= origin_dates[index + 1]
        for index in range(len(origin_dates) - 1)
    ):
        raise ValueError("origin_dates must be strictly increasing")
    pairs = tuple(
        align_fixed_contract_pair(
            transition,
            panel.market_day(transition.next_date),
        )
        for transition in (
            panel.one_day_transition(origin_date) for origin_date in origin_dates
        )
    )
    return FixedContractDlvHistory(
        origin_dates=tuple(pair.origin_date for pair in pairs),
        endpoint_dates=tuple(pair.endpoint_date for pair in pairs),
        origin_log_dlvsq=np.stack([pair.origin_log_dlvsq for pair in pairs]),
        endpoint_log_dlvsq=np.stack([pair.endpoint_log_dlvsq for pair in pairs]),
        front_cdf_values=np.asarray([pair.front_cdf for pair in pairs]),
        spot_gaussian_scores=np.asarray(
            [pair.spot_gaussian_score for pair in pairs]
        ),
        interpolation_clip_fractions=np.asarray(
            [pair.interpolation_clip_fraction for pair in pairs]
        ),
        forward_tail_extrapolation_fractions=np.asarray(
            [pair.forward_tail_extrapolation_fraction for pair in pairs]
        ),
        front_supports=np.stack([pair.front_support for pair in pairs]),
        front_densities=np.stack([pair.front_density for pair in pairs]),
        terminal_normalized_spots=np.asarray(
            [pair.terminal_normalized_spot for pair in pairs]
        ),
        background_annual_volatilities=np.asarray(
            [pair.background_annual_volatility for pair in pairs]
        ),
    )


def fit_dlv_factors(
    history: FixedContractDlvHistory,
    *,
    as_of_date: date,
    config: RollingDlvFactorConfig = RollingDlvFactorConfig(),
    active_coordinate_mask: BoolArray | None = None,
) -> RollingDlvFactorFit:
    """Fit coordinatewise physical AR dynamics and joint spot/DLV factors."""
    sample_count = len(history.origin_dates)
    if sample_count != config.window_sessions:
        raise ValueError(
            f"history has {sample_count} pairs; expected {config.window_sessions}"
        )
    if any(endpoint > as_of_date for endpoint in history.endpoint_dates):
        raise ValueError("history contains an endpoint after the factor as-of date")

    origin = history.origin_log_dlvsq
    endpoint = history.endpoint_log_dlvsq
    origin_mean = np.mean(origin, axis=0)
    endpoint_mean = np.mean(endpoint, axis=0)
    centered_origin = origin - origin_mean
    centered_endpoint = endpoint - endpoint_mean
    denominator = np.sum(centered_origin**2, axis=0)
    scale = np.maximum(np.sum(origin**2, axis=0), 1.0)
    variable = denominator > np.finfo(np.float64).eps * scale * sample_count
    slope = np.zeros_like(denominator)
    slope[variable] = (
        np.sum(centered_origin * centered_endpoint, axis=0)[variable]
        / denominator[variable]
    )
    intercept = endpoint_mean - slope * origin_mean
    residual = endpoint - (intercept + slope * origin)
    residual -= np.mean(residual, axis=0, keepdims=True)
    smoothed = gaussian_filter(
        residual,
        sigma=(
            0.0,
            config.smoothing_expiry_sigma,
            config.smoothing_strike_sigma,
        ),
        mode="nearest",
    )
    smoothed -= np.mean(smoothed, axis=0, keepdims=True)
    if active_coordinate_mask is not None:
        active = np.asarray(active_coordinate_mask, dtype=bool)
        if active.shape != origin.shape[1:] or not np.any(active):
            raise ValueError("active coordinate mask does not align with DLV states")
        smoothed[..., ~active] = 0.0
    diffusion_vectors = smoothed.reshape(sample_count, -1)

    if history.has_front_mixture_observations:
        front_risk_price, front_risk_price_se = (
            estimate_front_spot_market_price_of_risk(history)
        )
        raw_spot_score = filtered_front_spot_innovations(
            history,
            market_price_of_risk=front_risk_price,
        )
        spot_mean = float(
            np.mean(raw_spot_score)
            + front_risk_price / np.sqrt(DAYS_PER_YEAR)
        )
        centered_spot_score = raw_spot_score - np.mean(raw_spot_score)
        spot_variance = float(np.mean(centered_spot_score**2))
        spot_std = float(np.sqrt(spot_variance))
        if not np.isfinite(spot_std) or spot_std <= np.finfo(np.float64).tiny:
            raise FloatingPointError("filtered front innovations have zero variance")
        spot_loading = np.mean(
            diffusion_vectors * centered_spot_score[:, None], axis=0
        ) / spot_variance
        spot_score = centered_spot_score
    else:
        raw_spot_score = history.spot_gaussian_scores
        spot_mean = float(np.mean(raw_spot_score))
        spot_std = float(np.std(raw_spot_score, ddof=0))
        if not np.isfinite(spot_std) or spot_std <= np.finfo(np.float64).tiny:
            raise FloatingPointError("front Gaussian scores have zero variance")
        spot_score = (raw_spot_score - spot_mean) / spot_std
        spot_loading = np.mean(diffusion_vectors * spot_score[:, None], axis=0)
        front_risk_price = spot_mean * np.sqrt(DAYS_PER_YEAR)
        front_risk_price_se = np.sqrt(DAYS_PER_YEAR / sample_count)
    if history.has_front_mixture_observations:
        orthogonal_loadings, orthogonal_variances, orthogonal_scores = (
            _partial_observation_low_rank_loadings(
                diffusion_vectors,
                observed_loading=spot_loading,
                observed_scores=spot_score,
                rank=config.factor_rank - 1,
            )
        )
    else:
        orthogonal_residual = (
            diffusion_vectors - spot_score[:, None] * spot_loading
        )
        orthogonal_loadings, orthogonal_variances, orthogonal_scores = (
            _low_rank_loadings(orthogonal_residual, config.factor_rank - 1)
        )
    flat_loadings = np.column_stack((spot_loading, orthogonal_loadings))
    daily_loadings = flat_loadings.reshape(
        *origin.shape[1:],
        config.factor_rank,
    )
    factor_variances = np.concatenate(
        ([float(spot_loading @ spot_loading)], orthogonal_variances)
    )
    total_variance = float(np.mean(np.sum(diffusion_vectors**2, axis=1)))
    retained_variance = float(np.sum(factor_variances))
    if total_variance <= np.finfo(np.float64).tiny:
        raise FloatingPointError("smoothed DLV residuals have zero variance")
    return RollingDlvFactorFit(
        as_of_date=as_of_date,
        training_origin_dates=history.origin_dates,
        training_endpoint_dates=history.endpoint_dates,
        conditional_intercept=intercept,
        conditional_slope=slope,
        centered_residuals=residual,
        spot_score_mean=spot_mean,
        spot_score_std=spot_std,
        daily_loadings=daily_loadings,
        daily_factor_variances=factor_variances,
        factor_scores=np.column_stack((spot_score, orthogonal_scores)),
        explained_variance_fraction=min(retained_variance / total_variance, 1.0),
        interpolation_clip_fraction=float(
            np.mean(history.interpolation_clip_fractions)
        ),
        forward_tail_extrapolation_fraction=float(
            np.mean(history.forward_tail_extrapolation_fractions)
        ),
        front_spot_market_price_of_risk=float(front_risk_price),
        front_spot_market_price_of_risk_standard_error=float(front_risk_price_se),
    )


def fit_rolling_dlv_factors(
    panel: PublicHedgingPanel,
    *,
    as_of_date: date,
    config: RollingDlvFactorConfig = RollingDlvFactorConfig(),
) -> RollingDlvFactorFit:
    """Fit the trailing public DLV window using information through ``as_of``."""
    require_public_hedging_panel(panel)
    origin_dates = panel.prior_one_day_origins(
        as_of_date,
        count=config.window_sessions,
    )
    history = build_fixed_contract_dlv_history(panel, origin_dates=origin_dates)
    return fit_dlv_factors(history, as_of_date=as_of_date, config=config)


def raw_dlv_state_with_factor_bump(
    day: PublicMarketDay,
    fit: RollingDlvFactorFit,
    *,
    factor_index: int,
    standard_deviations: float,
) -> FloatArray:
    """Apply a central-difference factor bump in log-DLV-squared coordinates."""
    if not 0 <= factor_index < fit.factor_rank:
        raise ValueError(f"factor_index {factor_index} is out of range")
    if not np.isfinite(standard_deviations):
        raise ValueError("standard_deviations must be finite")
    expected_shape = day.dlvsq[1:].shape
    if fit.daily_loadings.shape[:2] != expected_shape:
        raise ValueError(
            f"factor coordinates {fit.daily_loadings.shape[:2]} do not match "
            f"current DLV state {expected_shape}"
        )
    log_dlvsq = np.log(day.dlvsq)
    log_dlvsq[1:] += (
        standard_deviations * fit.daily_loadings[..., factor_index]
    )
    lower = np.nextafter(day.minimum_dlvsq, np.inf)
    bumped_dlvsq = np.clip(np.exp(log_dlvsq), lower, day.maximum_dlvsq)
    return np.log(bumped_dlvsq - day.minimum_dlvsq)


def _integer_model_dtes(day: PublicMarketDay) -> IntArray:
    raw = np.asarray(day.dlv_model_expiries_y) * DAYS_PER_YEAR
    rounded = np.rint(raw).astype(np.int64)
    if (
        np.any(rounded <= 0)
        or np.any(np.diff(rounded) <= 0)
        or not np.allclose(raw, rounded, rtol=0.0, atol=2.0e-6)
    ):
        raise ValueError("public DLV model expiries must be increasing business days")
    return rounded


def _model_forward_over_spot(
    day: PublicMarketDay,
    *,
    target_business_dte: int,
) -> tuple[float, bool]:
    """Interpolate ``F/S`` and hold its terminal value past listed expiries.

    The fitted DLV grid has synthetic daily nodes through 25 business days,
    while the last listed expiry in the released book can be earlier. The
    normalized-forward convention is therefore log-linear inside the released
    curve and flat in ``F/S`` beyond its terminal node.
    """
    if target_business_dte <= day.business_dtes[-1] + 1.0e-8:
        return (
            interpolate_positive_curve_at_business_dte(
                business_dtes=day.business_dtes,
                values=day.forward_over_spot,
                target_business_dte=target_business_dte,
                name="forward-over-spot",
            ),
            False,
        )
    return float(day.forward_over_spot[-1]), True


def estimate_front_spot_market_price_of_risk(
    history: FixedContractDlvHistory,
) -> tuple[float, float]:
    """Estimate the front Brownian risk price under the simulated spot law.

    The observed normalized spot is the product of a discrete DLV-front node
    and the independent lognormal backbone.  Under ``P`` the Gaussian copula
    coordinate selecting the node is shifted by ``lambda / sqrt(252)``.  This
    routine maximizes the exact resulting mixture likelihood, rather than
    treating a PIT of the convolved spot distribution as the latent Brownian
    coordinate.
    """
    arrays = _front_mixture_arrays(history)
    initial = float(np.mean(history.spot_gaussian_scores) * np.sqrt(DAYS_PER_YEAR))

    def objective(value: FloatArray) -> float:
        return -_front_mixture_log_likelihood(
            arrays,
            market_price_of_risk=float(value[0]),
        )

    result = minimize(
        objective,
        np.asarray([initial], dtype=np.float64),
        method="BFGS",
        options={"gtol": 1.0e-9, "maxiter": 200},
    )
    estimate = float(result.x[0])
    if not np.isfinite(estimate) or not np.isfinite(result.fun):
        raise FloatingPointError("front-mixture likelihood optimization failed")
    step = max(1.0e-3, 1.0e-4 * max(abs(estimate), 1.0))
    center = objective(np.asarray([estimate]))
    curvature = (
        objective(np.asarray([estimate + step]))
        - 2.0 * center
        + objective(np.asarray([estimate - step]))
    ) / step**2
    if not np.isfinite(curvature) or curvature <= 0.0:
        raise FloatingPointError("front-mixture likelihood has no positive curvature")
    return estimate, float(1.0 / np.sqrt(curvature))


def filtered_front_spot_innovations(
    history: FixedContractDlvHistory,
    *,
    market_price_of_risk: float,
) -> FloatArray:
    """Return posterior means of the physical front Brownian innovations."""
    arrays = _front_mixture_arrays(history)
    mean_shift = market_price_of_risk / np.sqrt(DAYS_PER_YEAR)
    node_probabilities, lower, upper = _shifted_front_probabilities(
        arrays[1],
        mean_shift=mean_shift,
    )
    log_background = _front_background_log_likelihood(*arrays)
    log_joint = np.log(np.maximum(node_probabilities, np.finfo(np.float64).tiny))
    log_joint += log_background
    posterior = np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))
    lower_epsilon = lower - mean_shift
    upper_epsilon = upper - mean_shift
    normalizer = np.maximum(node_probabilities, np.finfo(np.float64).tiny)
    conditional_mean = (
        _standard_normal_density(lower_epsilon)
        - _standard_normal_density(upper_epsilon)
    ) / normalizer
    conditional_mean = np.where(node_probabilities > 0.0, conditional_mean, 0.0)
    result = np.sum(posterior * conditional_mean, axis=1)
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("filtered front Brownian innovations are not finite")
    return result


def _front_mixture_arrays(
    history: FixedContractDlvHistory,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    if not history.has_front_mixture_observations:
        raise ValueError("front-mixture observations are required")
    assert history.front_supports is not None
    assert history.front_densities is not None
    assert history.terminal_normalized_spots is not None
    assert history.background_annual_volatilities is not None
    return (
        history.front_supports,
        history.front_densities,
        history.terminal_normalized_spots,
        history.background_annual_volatilities,
    )


def _front_mixture_log_likelihood(
    arrays: tuple[FloatArray, FloatArray, FloatArray, FloatArray],
    *,
    market_price_of_risk: float,
) -> float:
    _, densities, _, _ = arrays
    mean_shift = market_price_of_risk / np.sqrt(DAYS_PER_YEAR)
    node_probabilities, _, _ = _shifted_front_probabilities(
        densities,
        mean_shift=mean_shift,
    )
    log_joint = np.log(np.maximum(node_probabilities, np.finfo(np.float64).tiny))
    log_joint += _front_background_log_likelihood(*arrays)
    return float(np.sum(logsumexp(log_joint, axis=1)))


def _shifted_front_probabilities(
    densities: FloatArray,
    *,
    mean_shift: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    cumulative = np.cumsum(densities, axis=1)[:, :-1]
    probability_floor = np.finfo(np.float64).eps
    thresholds = ndtri(np.clip(cumulative, probability_floor, 1.0 - probability_floor))
    lower = np.concatenate(
        (np.full((len(densities), 1), -np.inf), thresholds), axis=1
    )
    upper = np.concatenate(
        (thresholds, np.full((len(densities), 1), np.inf)), axis=1
    )
    probabilities = ndtr(upper - mean_shift) - ndtr(lower - mean_shift)
    probabilities = np.maximum(probabilities, 0.0)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return probabilities, lower, upper


def _front_background_log_likelihood(
    supports: FloatArray,
    densities: FloatArray,
    terminal_spots: FloatArray,
    annual_volatilities: FloatArray,
) -> FloatArray:
    del densities
    dt = 1.0 / DAYS_PER_YEAR
    standard_deviation = annual_volatilities[:, None] * np.sqrt(dt)
    variance = standard_deviation**2
    positive_support = supports > 0.0
    safe_support = np.where(positive_support, supports, 1.0)
    log_ratio = np.log(terminal_spots[:, None] / safe_support)
    standardized = (log_ratio + 0.5 * variance) / standard_deviation
    log_density = (
        -0.5 * standardized**2
        - np.log(terminal_spots[:, None])
        - np.log(standard_deviation)
        - 0.5 * np.log(2.0 * np.pi)
    )
    return np.where(positive_support, log_density, -np.inf)


def _standard_normal_density(values: FloatArray) -> FloatArray:
    result = np.exp(-0.5 * values**2) / np.sqrt(2.0 * np.pi)
    return np.where(np.isfinite(values), result, 0.0)


def _decode_front_density(day: PublicMarketDay) -> FloatArray:
    support = np.asarray(day.dlv_support_strikes, dtype=np.float64)
    previous = _density0_from_support(support)
    dlvsq_dt = np.asarray(day.dlvsq[0], dtype=np.float64) / DAYS_PER_YEAR
    box = support[1:-1] ** 2 / (support[2:] - support[:-2])
    upper_weight = dlvsq_dt * box / (support[2:] - support[1:-1])
    lower_weight = dlvsq_dt * box / (support[1:-1] - support[:-2])
    lower = np.concatenate((np.zeros(2), -upper_weight))
    diagonal = np.concatenate(
        ((1.0,), 1.0 + lower_weight + upper_weight, (1.0,))
    )
    upper = np.concatenate((-lower_weight, np.zeros(2)))
    banded = np.zeros((3, len(support)), dtype=np.float64)
    banded[0, 1:] = upper[:-1]
    banded[1] = diagonal
    banded[2, :-1] = lower[1:]
    density = solve_banded((1, 1), banded, previous)
    if np.min(density) < -1.0e-11:
        raise FloatingPointError("decoded DLV front density is negative")
    density = np.maximum(density, 0.0)
    return density / np.sum(density)


def _density0_from_support(support: FloatArray) -> FloatArray:
    calls = np.maximum(1.0 - support, 0.0)
    slopes = np.diff(calls, prepend=1.0) / np.diff(support, prepend=0.0)
    density = np.diff(slopes, append=0.0)
    if np.min(density) < -1.0e-12:
        raise FloatingPointError("initial lattice density is negative")
    density = np.maximum(density, 0.0)
    return density / np.sum(density)


def _front_spot_cdf(
    *,
    terminal_normalized_spot: float,
    support: FloatArray,
    density: FloatArray,
    background_annual_volatility: float,
) -> float:
    variance = background_annual_volatility**2 / DAYS_PER_YEAR
    positive = support > 0.0
    standardized = np.full_like(support, np.inf)
    standardized[positive] = (
        np.log(terminal_normalized_spot / support[positive]) + 0.5 * variance
    ) / np.sqrt(variance)
    return float(density @ ndtr(standardized))


def _low_rank_loadings(
    residuals: FloatArray,
    rank: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    sample_count = residuals.shape[0]
    if not 0 < rank < sample_count:
        raise ValueError(f"rank must lie in [1,{sample_count - 1}], found {rank}")
    gram = residuals @ residuals.T / sample_count
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    retained_values = eigenvalues[:rank]
    tolerance = np.finfo(np.float64).eps * max(float(eigenvalues[0]), 1.0)
    if np.any(retained_values <= tolerance):
        positive = int(np.count_nonzero(eigenvalues > tolerance))
        raise FloatingPointError(
            f"DLV residual covariance has only {positive} positive factors"
        )
    retained_vectors = eigenvectors[:, order[:rank]]
    loadings = residuals.T @ retained_vectors / np.sqrt(sample_count)
    scores = np.sqrt(sample_count) * retained_vectors
    return loadings, retained_values, scores


def _partial_observation_low_rank_loadings(
    values: FloatArray,
    *,
    observed_loading: FloatArray,
    observed_scores: FloatArray,
    rank: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Extract latent factors without duplicating a partially observed factor.

    ``observed_scores`` are posterior means of a unit-variance latent shock.
    Regressing on those means identifies its loading, but PCA of the regression
    residual would count the shock's posterior uncertainty a second time.  We
    instead eigendecompose the positive-semidefinite projection of
    ``Cov(values) - b b'`` in the sample covariance span.
    """
    sample_count, coordinate_count = values.shape
    if not 0 < rank < sample_count:
        raise ValueError(f"rank must lie in [1,{sample_count - 1}], found {rank}")
    loading = np.asarray(observed_loading, dtype=np.float64)
    scores = np.asarray(observed_scores, dtype=np.float64)
    if loading.shape != (coordinate_count,) or scores.shape != (sample_count,):
        raise ValueError("partial-observation factor inputs do not align")
    gram = values @ values.T / sample_count
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
    order = np.argsort(eigenvalues)[::-1]
    tolerance = np.finfo(np.float64).eps * max(float(eigenvalues[order[0]]), 1.0)
    positive = eigenvalues[order] > tolerance
    retained_sample_values = eigenvalues[order][positive]
    retained_sample_vectors = eigenvectors[:, order][:, positive]
    coordinate_basis = (
        values.T
        @ retained_sample_vectors
        / np.sqrt(sample_count * retained_sample_values)[None, :]
    )
    loading_coordinates = coordinate_basis.T @ loading
    unexplained = np.diag(retained_sample_values) - np.outer(
        loading_coordinates,
        loading_coordinates,
    )
    residual_values, residual_vectors = np.linalg.eigh(
        0.5 * (unexplained + unexplained.T)
    )
    residual_order = np.argsort(residual_values)[::-1]
    residual_values = np.maximum(residual_values[residual_order], 0.0)
    retained_values = residual_values[:rank]
    residual_tolerance = np.finfo(np.float64).eps * max(
        float(residual_values[0]),
        1.0,
    )
    if np.any(retained_values <= residual_tolerance):
        positive_count = int(np.count_nonzero(residual_values > residual_tolerance))
        raise FloatingPointError(
            "partial-observation DLV covariance has only "
            f"{positive_count} positive latent factors"
        )
    retained_vectors = residual_vectors[:, residual_order[:rank]]
    loadings = (
        coordinate_basis
        @ retained_vectors
        * np.sqrt(retained_values)[None, :]
    )
    conditional_residual = values - scores[:, None] * loading
    factor_scores = conditional_residual @ np.linalg.pinv(loadings).T
    return loadings, retained_values, factor_scores
