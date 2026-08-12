"""Daily double-exponential calibration and forward-density reconstruction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from cdxcore.err import verify, verify_inp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import OptimizeResult, isotonic_regression, minimize

from .config import StaticFitConfig
from .coordinates import log_moneyness_from_strike, normalized_call_intrinsic
from .double_exponential import (
    DoubleExponentialParameters,
    characteristic_function,
    constrained_parameters,
    continuous_negative_rate,
    density,
    symmetry_index,
)
from .markets import TangentLevyMarketDay
from .pricing import PreparedFourierPricer


@dataclass(frozen=True)
class SymmetryCurve:
    """Deterministic linearly interpolated symmetry index ``Xi(tau)``."""

    maturities: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        maturities = np.asarray(self.maturities, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)
        verify_inp(
            maturities.ndim == 1
            and values.shape == maturities.shape
            and maturities.size > 0
            and np.all(np.isfinite(maturities))
            and np.all(np.isfinite(values))
            and np.all(maturities > 0.0)
            and np.all(np.diff(maturities) > 0.0),
            lambda: "symmetry-curve maturities must be finite, positive, and increasing",
        )

    def __call__(self, tau: float | np.ndarray) -> np.ndarray:
        """Evaluate within the calibrated maturity domain without extrapolation."""

        value = np.asarray(tau, dtype=np.float64)
        verify_inp(
            np.all(np.isfinite(value))
            and np.all(value >= self.maturities[0])
            and np.all(value <= self.maturities[-1]),
            lambda: (
                "requested Xi outside calibrated range "
                f"[{self.maturities[0]}, {self.maturities[-1]}]"
            ),
        )
        return np.interp(value, self.maturities, self.values)


@dataclass(frozen=True)
class MaturityFit:
    """Optimization result for one market maturity."""

    tau: float
    parameters: DoubleExponentialParameters
    objective: float
    success: bool
    iterations: int
    selected_start: int
    start_objectives: np.ndarray
    num_quotes: int
    weighted_rmse: float
    maximum_absolute_error: float
    message: str


@dataclass(frozen=True)
class DayFit:
    """All calibrated maturities for one market date."""

    market_date: object
    fits: tuple[MaturityFit, ...]
    initial_symmetry_fit: bool

    @property
    def maturities(self) -> np.ndarray:
        """Calibrated maturities."""

        return np.asarray([fit.tau for fit in self.fits], dtype=np.float64)

    @property
    def parameters(self) -> tuple[DoubleExponentialParameters, ...]:
        """Calibrated parameter objects in maturity order."""

        return tuple(fit.parameters for fit in self.fits)

    @property
    def successful(self) -> bool:
        """Whether every maturity optimizer reported success."""

        return len(self.fits) > 0 and all(fit.success for fit in self.fits)

    def symmetry_curve(self) -> SymmetryCurve:
        """Build the initial-day deterministic symmetry curve."""

        verify_inp(
            self.initial_symmetry_fit,
            "a symmetry curve can only be created from an initial three-variable fit",
        )
        return SymmetryCurve(
            maturities=self.maturities,
            values=np.asarray(
                [symmetry_index(parameter) for parameter in self.parameters],
                dtype=np.float64,
            ),
        )


@dataclass(frozen=True)
class ForwardDensityReconstruction:
    """Result of constrained cumulative-intensity interpolation."""

    tau: np.ndarray
    jump_x: np.ndarray
    average_density: np.ndarray
    projected_fitted_average_density: np.ndarray
    forward_density: np.ndarray
    maximum_relative_projection: float
    relative_l1_projection: float
    maximum_absolute_projection: float
    passed_tolerance: bool


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_negative = math.exp(-value)
        return 1.0 / (1.0 + exp_negative)
    exp_positive = math.exp(value)
    return exp_positive / (1.0 + exp_positive)


def _initial_parameters(raw: np.ndarray) -> DoubleExponentialParameters:
    intensity = math.exp(float(raw[0]))
    positive_rate = 1.0 + math.exp(float(raw[1]))
    probability = _sigmoid(float(raw[2]))
    return DoubleExponentialParameters(
        intensity=intensity,
        positive_rate=positive_rate,
        negative_rate=continuous_negative_rate(
            positive_rate=positive_rate,
            probability=probability,
        ),
        positive_probability=probability,
    )


def _constrained_parameters(
    raw: np.ndarray,
    *,
    target_symmetry_index: float,
) -> DoubleExponentialParameters:
    return constrained_parameters(
        intensity=math.exp(float(raw[0])),
        positive_rate=1.0 + math.exp(float(raw[1])),
        target_symmetry_index=target_symmetry_index,
    )


def _optimization_starts(
    *,
    config: StaticFitConfig,
    initial: bool,
    warm_start: DoubleExponentialParameters | None,
) -> list[np.ndarray]:
    intensity_bounds = np.log(np.asarray(config.intensity_bounds, dtype=np.float64))
    tail_bounds = np.log(
        np.asarray(config.positive_tail_bounds, dtype=np.float64) - 1.0
    )
    rng = np.random.default_rng(config.optimizer_seed)
    starts: list[np.ndarray] = []
    if warm_start is not None:
        value = [
            math.log(warm_start.intensity),
            math.log(warm_start.positive_rate - 1.0),
        ]
        if initial:
            value.append(_logit(warm_start.positive_probability))
        starts.append(np.asarray(value, dtype=np.float64))
    center = [float(np.mean(intensity_bounds)), float(np.mean(tail_bounds))]
    if initial:
        center.append(0.0)
    starts.append(np.asarray(center, dtype=np.float64))
    while len(starts) < config.num_starts:
        value = [
            rng.uniform(*intensity_bounds),
            rng.uniform(*tail_bounds),
        ]
        if initial:
            probability = rng.uniform(
                config.probability_floor,
                1.0 - config.probability_floor,
            )
            value.append(_logit(float(probability)))
        starts.append(np.asarray(value, dtype=np.float64))
    return starts[: config.num_starts]


def fit_maturity(
    *,
    tau: float,
    strikes: np.ndarray,
    target_calls: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    mask: np.ndarray,
    config: StaticFitConfig = StaticFitConfig(),
    target_symmetry_index: float | None = None,
    warm_start: DoubleExponentialParameters | None = None,
) -> MaturityFit:
    """Fit one maturity using equations (2.14), (2.20), or (2.21)."""

    strikes = np.asarray(strikes, dtype=np.float64)
    target_calls = np.asarray(target_calls, dtype=np.float64)
    bids = np.asarray(bids, dtype=np.float64)
    asks = np.asarray(asks, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    verify_inp(
        math.isfinite(tau) and tau > 0.0,
        lambda: f"'tau' must be finite and positive, got {tau!r}",
    )
    verify_inp(
        strikes.shape
        == target_calls.shape
        == bids.shape
        == asks.shape
        == mask.shape,
        lambda: "strike, price, interval, and mask arrays must have the same shape",
    )
    finite = (
        np.isfinite(strikes)
        & np.isfinite(target_calls)
        & np.isfinite(bids)
        & np.isfinite(asks)
    )
    active = mask & finite & (strikes > 0.0) & (asks > bids)
    verify_inp(
        np.count_nonzero(active) >= 5,
        lambda: (
            "at least five active quotes are required for a maturity fit, got "
            f"{np.count_nonzero(active)}"
        ),
    )
    strikes_fit = strikes[active]
    x_fit = log_moneyness_from_strike(strikes_fit)
    target_fit = target_calls[active]
    intrinsic = normalized_call_intrinsic(strikes_fit)
    target_time_value = target_fit - intrinsic
    half_spread = np.maximum(
        0.5 * (asks[active] - bids[active]),
        config.half_spread_floor,
    )
    prepared = PreparedFourierPricer.create(x_fit, config=config.pricing)
    initial = target_symmetry_index is None

    def to_parameters(raw: np.ndarray) -> DoubleExponentialParameters:
        if initial:
            return _initial_parameters(raw)
        return _constrained_parameters(
            raw,
            target_symmetry_index=float(target_symmetry_index),
        )

    def objective(raw: np.ndarray) -> float:
        try:
            parameters = to_parameters(raw)
            model_calls = prepared.price(
                lambda z: characteristic_function(z, tau=tau, parameters=parameters)
            )
            model_time_value = model_calls - intrinsic
            residual = (model_time_value - target_time_value) / half_spread
            value = float(np.mean(np.square(residual)))
            return value if math.isfinite(value) else 1.0e30
        except (FloatingPointError, OverflowError, ValueError):
            return 1.0e30

    intensity_bounds = tuple(np.log(np.asarray(config.intensity_bounds)))
    tail_bounds = tuple(
        np.log(np.asarray(config.positive_tail_bounds, dtype=np.float64) - 1.0)
    )
    bounds = [intensity_bounds, tail_bounds]
    if initial:
        bounds.append(
            (
                _logit(config.probability_floor),
                _logit(1.0 - config.probability_floor),
            )
        )
    starts = _optimization_starts(
        config=config,
        initial=initial,
        warm_start=warm_start,
    )
    results: list[OptimizeResult] = []
    start_objectives = np.empty(len(starts), dtype=np.float64)
    for index, start in enumerate(starts):
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": config.max_iterations,
                "maxls": 100,
                "ftol": 1.0e-12,
            },
        )
        results.append(result)
        start_objectives[index] = float(result.fun)
    selected = int(np.argmin(start_objectives))
    best = results[selected]
    verify(
        np.isfinite(best.fun) and float(best.fun) < 1.0e29,
        lambda: f"all maturity calibration starts failed: {start_objectives}",
    )
    parameters = to_parameters(np.asarray(best.x, dtype=np.float64))
    model_calls = prepared.price(
        lambda z: characteristic_function(z, tau=tau, parameters=parameters)
    )
    errors = model_calls - target_fit
    return MaturityFit(
        tau=float(tau),
        parameters=parameters,
        objective=float(best.fun),
        success=bool(best.success),
        iterations=int(best.nit),
        selected_start=selected,
        start_objectives=start_objectives,
        num_quotes=int(strikes_fit.size),
        weighted_rmse=float(math.sqrt(best.fun)),
        maximum_absolute_error=float(np.max(np.abs(errors))),
        message=str(best.message),
    )


def fit_market_day(
    market: TangentLevyMarketDay,
    *,
    config: StaticFitConfig = StaticFitConfig(),
    symmetry_curve: SymmetryCurve | None = None,
    warm_start: DayFit | None = None,
) -> DayFit:
    """Fit every eligible maturity in one smoothed market."""

    warm_maturities = (
        np.asarray([], dtype=np.float64)
        if warm_start is None
        else warm_start.maturities
    )
    fits: list[MaturityFit] = []
    for index, tau in enumerate(market.expiries):
        tau_value = float(tau)
        if symmetry_curve is not None and not (
            symmetry_curve.maturities[0]
            <= tau_value
            <= symmetry_curve.maturities[-1]
        ):
            continue
        target_xi = (
            None
            if symmetry_curve is None
            else float(np.asarray(symmetry_curve(tau_value)).item())
        )
        warm = None
        if warm_start is not None:
            nearest = int(np.argmin(np.abs(warm_maturities - tau_value)))
            warm = warm_start.fits[nearest].parameters
        fits.append(
            fit_maturity(
                tau=tau_value,
                strikes=market.strikes[index],
                target_calls=market.sanos_calls[index],
                bids=market.pure_bids[index],
                asks=market.pure_asks[index],
                mask=market.model_masks[index],
                config=config,
                target_symmetry_index=target_xi,
                warm_start=warm,
            )
        )
    verify(
        len(fits) >= 2,
        lambda: f"fewer than two maturities calibrated on {market.market_date}",
    )
    return DayFit(
        market_date=market.market_date,
        fits=tuple(fits),
        initial_symmetry_fit=symmetry_curve is None,
    )


def fitted_average_density(day_fit: DayFit, jump_x: np.ndarray) -> np.ndarray:
    """Evaluate calibrated maturity-average densities on a jump grid."""

    jump_x = np.asarray(jump_x, dtype=np.float64)
    verify_inp(
        jump_x.ndim == 1 and np.all(np.isfinite(jump_x)),
        lambda: "'jump_x' must be a finite vector",
    )
    return np.stack(
        [density(jump_x, parameters) for parameters in day_fit.parameters],
        axis=0,
    )


def reconstruct_forward_density(
    *,
    fitted_tau: np.ndarray,
    fitted_average_density: np.ndarray,
    target_tau: np.ndarray,
    jump_x: np.ndarray,
    relative_tolerance: float = 0.25,
    forward_density_floor: float = 0.0,
) -> ForwardDensityReconstruction:
    """Recover nonnegative ``kappa = d(tau*eta)/d tau`` with monotone PCHIP.

    Per jump size, weighted isotonic regression projects cumulative intensity
    ``A=tau*eta`` onto the nondecreasing cone. A shape-preserving cubic then
    provides both ``A`` and its nonnegative maturity derivative.
    """

    fitted_tau = np.asarray(fitted_tau, dtype=np.float64)
    eta_fit = np.asarray(fitted_average_density, dtype=np.float64)
    target_tau = np.asarray(target_tau, dtype=np.float64)
    jump_x = np.asarray(jump_x, dtype=np.float64)
    verify_inp(
        fitted_tau.ndim == 1
        and fitted_tau.size >= 2
        and np.all(fitted_tau > 0.0)
        and np.all(np.diff(fitted_tau) > 0.0),
        lambda: "'fitted_tau' must be positive and strictly increasing",
    )
    verify_inp(
        eta_fit.shape == (fitted_tau.size, jump_x.size)
        and np.all(np.isfinite(eta_fit))
        and np.all(eta_fit >= 0.0),
        lambda: (
            "fitted density must be finite and nonnegative with shape "
            f"{(fitted_tau.size, jump_x.size)}, got {eta_fit.shape}"
        ),
    )
    verify_inp(
        target_tau.ndim == 1
        and target_tau.size >= 2
        and target_tau[0] == 0.0
        and np.all(np.diff(target_tau) > 0.0)
        and target_tau[-1] <= fitted_tau[-1],
        lambda: (
            "'target_tau' must start at zero, increase, and remain inside the "
            f"fitted upper maturity {fitted_tau[-1]}"
        ),
    )
    verify_inp(
        math.isfinite(relative_tolerance) and relative_tolerance >= 0.0,
        lambda: f"'relative_tolerance' must be nonnegative, got {relative_tolerance}",
    )
    verify_inp(
        math.isfinite(forward_density_floor) and forward_density_floor >= 0.0,
        lambda: (
            "'forward_density_floor' must be finite and nonnegative, got "
            f"{forward_density_floor}"
        ),
    )
    source_tau = np.concatenate(([0.0], fitted_tau))
    source_a = np.concatenate(
        [np.zeros((1, jump_x.size)), fitted_tau[:, None] * eta_fit],
        axis=0,
    )
    projected_shifted_a = np.empty_like(source_a)
    for jump_index in range(jump_x.size):
        shifted = (
            source_a[:, jump_index]
            - forward_density_floor * source_tau
        )
        result = isotonic_regression(shifted, increasing=True)
        # The PAVA solution is unconstrained in level. Tiny tail intensities
        # below the configured forward floor can otherwise pool A(0, x) below
        # zero. Clipping the increasing shifted solution at zero preserves
        # monotonicity while imposing A(0, x)=0 and dA/dtau >= floor.
        projected_shifted_a[:, jump_index] = np.maximum(
            np.asarray(result.x, dtype=np.float64),
            0.0,
        )
    projected_a = (
        projected_shifted_a
        + forward_density_floor * source_tau[:, None]
    )
    interpolator = PchipInterpolator(
        source_tau,
        projected_shifted_a,
        axis=0,
        extrapolate=False,
    )
    target_a = (
        np.asarray(interpolator(target_tau), dtype=np.float64)
        + forward_density_floor * target_tau[:, None]
    )
    kappa = (
        np.asarray(interpolator.derivative()(target_tau), dtype=np.float64)
        + forward_density_floor
    )
    numerical_floor = 1.0e-12 * max(1.0, float(np.max(kappa)))
    verify(
        np.min(kappa) >= forward_density_floor - numerical_floor,
        lambda: f"monotone cumulative interpolation produced kappa min {np.min(kappa)}",
    )
    kappa = np.maximum(kappa, forward_density_floor)
    eta = np.empty_like(target_a)
    eta[0] = kappa[0]
    eta[1:] = target_a[1:] / target_tau[1:, None]
    absolute_projection = np.abs(projected_a[1:] - source_a[1:])
    material_floor = 1.0e-6 * max(1.0, float(np.max(np.abs(source_a[1:]))))
    relative_projection = absolute_projection / np.maximum(
        np.abs(source_a[1:]),
        material_floor,
    )
    maximum_relative = float(np.max(relative_projection))
    relative_l1 = float(
        np.sum(absolute_projection)
        / max(float(np.sum(np.abs(source_a[1:]))), 1.0e-15)
    )
    return ForwardDensityReconstruction(
        tau=target_tau,
        jump_x=jump_x,
        average_density=eta,
        projected_fitted_average_density=(
            projected_a[1:] / fitted_tau[:, None]
        ),
        forward_density=kappa,
        maximum_relative_projection=maximum_relative,
        relative_l1_projection=relative_l1,
        maximum_absolute_projection=float(np.max(absolute_projection)),
        passed_tolerance=relative_l1 <= relative_tolerance,
    )
