"""No-replay stochastic-volatility diffusion for affine DYSANOS states.

The affine conditional mean and innovation covariance retain their fitted
values.  A scalar latent log variance follows the exact transition of a
stationary Ornstein--Uhlenbeck diffusion and multiplies every coordinate of
the Gaussian state innovation.  The multiplier is normalized to have unit
stationary mean, so ``unconditional_scale`` is the sole covariance-scale
parameter.

For a ``dimension``-variate standardized innovation, the observed normalized
squared radius is

``radius / dimension = rho * exp(g - Var(g)/2) * chi2_d / d``.

We fit this model through the logarithm of the radius.  The log-chi-square
measurement mean and variance are exact; its Gaussian likelihood
approximation is accurate in the 200-dimensional application and is recorded
explicitly in the resulting diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version
from scipy.optimize import OptimizeResult, minimize
from scipy.special import digamma, polygamma


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class LogVarianceDiffusionConfig:
    """Exact-step parameters of the stationary log-variance OU diffusion."""

    persistence: float
    stationary_log_variance: float
    unconditional_scale: float = 1.0

    def __post_init__(self) -> None:
        verify_inp(
            math.isfinite(self.persistence) and 0.0 <= self.persistence < 1.0,
            lambda: f"persistence must lie in [0,1), found {self.persistence}",
        )
        verify_inp(
            math.isfinite(self.stationary_log_variance)
            and self.stationary_log_variance >= 0.0,
            lambda: (
                "stationary_log_variance must be finite and non-negative, found "
                f"{self.stationary_log_variance}"
            ),
        )
        verify_inp(
            math.isfinite(self.unconditional_scale)
            and self.unconditional_scale > 0.0,
            lambda: (
                "unconditional_scale must be finite and positive, found "
                f"{self.unconditional_scale}"
            ),
        )

    @property
    def log_variance_innovation_standard_deviation(self) -> float:
        """Standard deviation of one exact-step OU innovation."""
        return math.sqrt(
            self.stationary_log_variance * (1.0 - self.persistence**2)
        )

    def continuous_ou_parameters(
        self, *, step_sessions: int
    ) -> dict[str, float | None]:
        """Return continuous-time OU parameters in trading-session units."""
        if step_sessions <= 0:
            raise ValueError("step_sessions must be positive")
        if self.persistence == 0.0:
            return {
                "mean_reversion_per_session": None,
                "log_variance_diffusion_per_sqrt_session": None,
            }
        mean_reversion = -math.log(self.persistence) / step_sessions
        diffusion = math.sqrt(
            2.0 * mean_reversion * self.stationary_log_variance
        )
        return {
            "mean_reversion_per_session": mean_reversion,
            "log_variance_diffusion_per_sqrt_session": diffusion,
        }


@dataclass(frozen=True)
class LogVarianceDiffusionFilter:
    """Kalman approximation for the latent log-variance state."""

    predicted_means: np.ndarray
    predicted_variances: np.ndarray
    filtered_means: np.ndarray
    filtered_variances: np.ndarray
    next_predicted_means: np.ndarray
    next_predicted_variances: np.ndarray
    log_likelihood: np.ndarray
    conditional_mean_scales: np.ndarray
    measurement_log_mean: float
    measurement_log_variance: float


@dataclass(frozen=True)
class LogVarianceDiffusionFit:
    """Maximum approximate-likelihood fit of the latent diffusion."""

    config: LogVarianceDiffusionConfig
    filter: LogVarianceDiffusionFilter
    success: bool
    message: str
    iterations: int
    evaluations: int
    negative_log_likelihood: float
    gaussian_negative_log_likelihood: float
    scaled_gaussian_negative_log_likelihood: float
    scaled_gaussian_unconditional_scale: float

    def to_dict(self, *, step_sessions: int) -> dict[str, object]:
        """Return a compact JSON-compatible fit report."""
        return {
            "config": asdict(self.config),
            "continuous_ou": self.config.continuous_ou_parameters(
                step_sessions=step_sessions
            ),
            "success": self.success,
            "message": self.message,
            "iterations": self.iterations,
            "evaluations": self.evaluations,
            "negative_log_likelihood": self.negative_log_likelihood,
            "gaussian_negative_log_likelihood": (
                self.gaussian_negative_log_likelihood
            ),
            "scaled_gaussian_negative_log_likelihood": (
                self.scaled_gaussian_negative_log_likelihood
            ),
            "scaled_gaussian_unconditional_scale": (
                self.scaled_gaussian_unconditional_scale
            ),
            "measurement_approximation": {
                "law": "Gaussian approximation to log(chi-square_d / d)",
                "exact_log_mean": self.filter.measurement_log_mean,
                "exact_log_variance": self.filter.measurement_log_variance,
            },
            "conditional_mean_scale": _summary(
                self.filter.conditional_mean_scales
            ),
        }


def log_chi_square_moments(dimension: int) -> tuple[float, float]:
    """Return exact mean and variance of ``log(chi-square_d / d)``."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    mean = float(digamma(0.5 * dimension) + math.log(2.0) - math.log(dimension))
    variance = float(polygamma(1, 0.5 * dimension))
    return mean, variance


def filter_log_variance_diffusion(
    normalized_squared_radii: np.ndarray,
    segment_starts: np.ndarray,
    *,
    dimension: int,
    config: LogVarianceDiffusionConfig,
) -> LogVarianceDiffusionFilter:
    """Filter the latent log variance from normalized squared radii.

    ``normalized_squared_radii`` contains ``r/d`` rather than ``r``.  A new
    filtering segment starts from the exact stationary latent distribution.
    """
    radii = np.asarray(normalized_squared_radii, dtype=np.float64)
    starts = np.asarray(segment_starts, dtype=np.bool_)
    if (
        radii.ndim != 1
        or starts.shape != radii.shape
        or len(radii) == 0
        or np.any(~np.isfinite(radii))
        or np.any(radii <= 0.0)
    ):
        raise ValueError("radii and segment_starts must be aligned positive vectors")
    if not starts[0]:
        raise ValueError("the first observation must start a filtering segment")

    measurement_mean, measurement_variance = log_chi_square_moments(dimension)
    stationary_variance = config.stationary_log_variance
    state_constant = (
        math.log(config.unconditional_scale) - 0.5 * stationary_variance
    )
    observations = np.log(radii)
    predicted_means = np.empty_like(observations)
    predicted_variances = np.empty_like(observations)
    filtered_means = np.empty_like(observations)
    filtered_variances = np.empty_like(observations)
    next_means = np.empty_like(observations)
    next_variances = np.empty_like(observations)
    log_likelihood = np.empty_like(observations)
    conditional_scales = np.empty_like(observations)

    predicted_mean = 0.0
    predicted_variance = stationary_variance
    process_variance = stationary_variance * (1.0 - config.persistence**2)
    for index, observation in enumerate(observations):
        if starts[index]:
            predicted_mean = 0.0
            predicted_variance = stationary_variance
        innovation = observation - (
            state_constant + measurement_mean + predicted_mean
        )
        innovation_variance = predicted_variance + measurement_variance
        if innovation_variance <= 0.0 or not math.isfinite(innovation_variance):
            raise ValueError("log-radius innovation variance is not positive")
        gain = predicted_variance / innovation_variance
        filtered_mean = predicted_mean + gain * innovation
        filtered_variance = max(
            predicted_variance * (1.0 - gain),
            0.0,
        )
        next_mean = config.persistence * filtered_mean
        next_variance = (
            config.persistence**2 * filtered_variance + process_variance
        )

        predicted_means[index] = predicted_mean
        predicted_variances[index] = predicted_variance
        filtered_means[index] = filtered_mean
        filtered_variances[index] = filtered_variance
        next_means[index] = next_mean
        next_variances[index] = next_variance
        conditional_scales[index] = config.unconditional_scale * math.exp(
            predicted_mean
            + 0.5 * predicted_variance
            - 0.5 * stationary_variance
        )
        log_likelihood[index] = -0.5 * (
            math.log(2.0 * math.pi * innovation_variance)
            + innovation * innovation / innovation_variance
        )
        predicted_mean = next_mean
        predicted_variance = next_variance

    return LogVarianceDiffusionFilter(
        predicted_means=predicted_means,
        predicted_variances=predicted_variances,
        filtered_means=filtered_means,
        filtered_variances=filtered_variances,
        next_predicted_means=next_means,
        next_predicted_variances=next_variances,
        log_likelihood=log_likelihood,
        conditional_mean_scales=conditional_scales,
        measurement_log_mean=measurement_mean,
        measurement_log_variance=measurement_variance,
    )


def fit_log_variance_diffusion(
    normalized_squared_radii: np.ndarray,
    segment_starts: np.ndarray,
    *,
    dimension: int,
) -> LogVarianceDiffusionFit:
    """Fit stationary OU log variance by approximate maximum likelihood."""
    radii = np.asarray(normalized_squared_radii, dtype=np.float64)
    starts = np.asarray(segment_starts, dtype=np.bool_)
    if len(radii) < 50:
        raise ValueError("at least 50 innovations are required for fitting")

    def objective(values: np.ndarray) -> float:
        try:
            config = LogVarianceDiffusionConfig(
                persistence=float(values[0]),
                stationary_log_variance=float(values[1]),
                unconditional_scale=float(values[2]),
            )
            filtered = filter_log_variance_diffusion(
                radii,
                starts,
                dimension=dimension,
                config=config,
            )
        except (ValueError, OverflowError):
            return 1.0e12
        value = -float(np.mean(filtered.log_likelihood))
        return value if math.isfinite(value) else 1.0e12

    empirical_scale = float(np.mean(radii))
    measurement_log_mean = log_chi_square_moments(dimension)[0]
    log_radius_scale = math.exp(float(np.mean(np.log(radii))) - measurement_log_mean)
    initial_log_variance = max(
        float(np.var(np.log(radii), ddof=1))
        - log_chi_square_moments(dimension)[1],
        1.0e-4,
    )
    starts_to_try = (
        np.asarray((0.20, initial_log_variance, empirical_scale)),
        np.asarray((0.60, initial_log_variance, empirical_scale)),
        np.asarray((0.90, initial_log_variance, empirical_scale)),
        np.asarray((0.98, 0.5 * initial_log_variance, empirical_scale)),
    )
    bounds = ((0.0, 0.999), (1.0e-8, 10.0), (0.05, 20.0))
    results: list[OptimizeResult] = []
    for initial in starts_to_try:
        results.append(
            minimize(
                objective,
                initial,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 4_000, "ftol": 1.0e-14, "gtol": 1.0e-10},
            )
        )
    result = min(results, key=lambda item: float(item.fun))
    config = LogVarianceDiffusionConfig(
        persistence=float(result.x[0]),
        stationary_log_variance=float(result.x[1]),
        unconditional_scale=float(result.x[2]),
    )
    filtered = filter_log_variance_diffusion(
        radii,
        starts,
        dimension=dimension,
        config=config,
    )
    gaussian_config = LogVarianceDiffusionConfig(
        persistence=0.0,
        stationary_log_variance=0.0,
        unconditional_scale=1.0,
    )
    scaled_gaussian_config = LogVarianceDiffusionConfig(
        persistence=0.0,
        stationary_log_variance=0.0,
        unconditional_scale=log_radius_scale,
    )
    gaussian = filter_log_variance_diffusion(
        radii,
        starts,
        dimension=dimension,
        config=gaussian_config,
    )
    scaled_gaussian = filter_log_variance_diffusion(
        radii,
        starts,
        dimension=dimension,
        config=scaled_gaussian_config,
    )
    return LogVarianceDiffusionFit(
        config=config,
        filter=filtered,
        success=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit),
        evaluations=int(result.nfev),
        negative_log_likelihood=-float(np.mean(filtered.log_likelihood)),
        gaussian_negative_log_likelihood=-float(
            np.mean(gaussian.log_likelihood)
        ),
        scaled_gaussian_negative_log_likelihood=-float(
            np.mean(scaled_gaussian.log_likelihood)
        ),
        scaled_gaussian_unconditional_scale=log_radius_scale,
    )


def _summary(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    if finite.ndim != 1 or not np.all(np.isfinite(finite)):
        raise ValueError("summary values must be a finite vector")
    quantiles = np.quantile(finite, (0.01, 0.05, 0.50, 0.95, 0.99))
    return {
        "mean": float(np.mean(finite)),
        "standard_deviation": float(np.std(finite)),
        "minimum": float(np.min(finite)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q95": float(quantiles[3]),
        "q99": float(quantiles[4]),
        "maximum": float(np.max(finite)),
    }


__all__ = [
    "LogVarianceDiffusionConfig",
    "LogVarianceDiffusionFilter",
    "LogVarianceDiffusionFit",
    "filter_log_variance_diffusion",
    "fit_log_variance_diffusion",
    "log_chi_square_moments",
]
