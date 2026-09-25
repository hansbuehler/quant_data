"""Parametric heavy-tailed innovations for affine DYSANOS dynamics.

The affine surface decoder and its stable conditional mean remain unchanged.
Only the innovation law is replaced.  A variance-normalized multivariate
Student-t shock is multiplied by a persistent score-driven scalar variance.
The dynamic multiplier is analytically normalized to unit mean; one fitted
global scale then makes the unconditional innovation covariance ``rho*Q``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version
from scipy.optimize import OptimizeResult, minimize
from scipy.special import gammaln, hyp1f1


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class ScoreDrivenStudentTConfig:
    """Identifiable parameters of the normalized dynamic-scale innovation.

    ``unconditional_scale`` is the scalar ``rho`` multiplying the affine
    Gaussian model's innovation covariance.
    """

    degrees_of_freedom: float
    score_loading: float
    persistence: float
    unconditional_scale: float = 1.0

    def __post_init__(self) -> None:
        verify_inp(
            math.isfinite(self.degrees_of_freedom)
            and self.degrees_of_freedom > 2.0,
            lambda: (
                "degrees_of_freedom must be finite and exceed two, found "
                f"{self.degrees_of_freedom}"
            ),
        )
        verify_inp(
            math.isfinite(self.score_loading) and self.score_loading >= 0.0,
            lambda: (
                "score_loading must be finite and non-negative, found "
                f"{self.score_loading}"
            ),
        )
        verify_inp(
            math.isfinite(self.persistence) and 0.0 <= self.persistence < 1.0,
            lambda: (
                "persistence must lie in [0,1), found "
                f"{self.persistence}"
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


@dataclass(frozen=True)
class ScoreDrivenFilter:
    """Conditional scales, scores, and likelihood from one filtered history."""

    scales: np.ndarray
    scores: np.ndarray
    log_likelihood: np.ndarray
    log_mean_normalizer: float


@dataclass(frozen=True)
class ScoreDrivenStudentTFit:
    """Maximum-likelihood result for the parametric innovation law."""

    config: ScoreDrivenStudentTConfig
    filter: ScoreDrivenFilter
    success: bool
    message: str
    iterations: int
    evaluations: int
    negative_log_likelihood_per_dimension: float
    iid_gaussian_negative_log_likelihood_per_dimension: float
    scaled_gaussian_negative_log_likelihood_per_dimension: float
    scaled_gaussian_unconditional_scale: float
    iid_student_t_negative_log_likelihood_per_dimension: float
    iid_student_t_degrees_of_freedom: float
    iid_student_t_unconditional_scale: float

    def to_dict(self) -> dict[str, object]:
        """Return the compact JSON-compatible fit summary."""
        return {
            "config": asdict(self.config),
            "success": self.success,
            "message": self.message,
            "iterations": self.iterations,
            "evaluations": self.evaluations,
            "negative_log_likelihood_per_dimension": (
                self.negative_log_likelihood_per_dimension
            ),
            "iid_gaussian_negative_log_likelihood_per_dimension": (
                self.iid_gaussian_negative_log_likelihood_per_dimension
            ),
            "scaled_gaussian_negative_log_likelihood_per_dimension": (
                self.scaled_gaussian_negative_log_likelihood_per_dimension
            ),
            "scaled_gaussian_unconditional_scale": (
                self.scaled_gaussian_unconditional_scale
            ),
            "iid_student_t_negative_log_likelihood_per_dimension": (
                self.iid_student_t_negative_log_likelihood_per_dimension
            ),
            "iid_student_t_degrees_of_freedom": (
                self.iid_student_t_degrees_of_freedom
            ),
            "iid_student_t_unconditional_scale": (
                self.iid_student_t_unconditional_scale
            ),
            "log_mean_normalizer": self.filter.log_mean_normalizer,
            "scale_summary": _summary(self.filter.scales),
            "score_summary": _summary(self.filter.scores),
        }


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


def student_t_scale_score(
    standardized_squared_radius: np.ndarray | float,
    *,
    dimension: int,
    degrees_of_freedom: float,
) -> np.ndarray:
    """Return the log-scale score for a variance-normalized Student-t law."""
    radius = np.asarray(standardized_squared_radius, dtype=np.float64)
    if dimension <= 0:
        raise ValueError(f"dimension must be positive, found {dimension}")
    if degrees_of_freedom <= 2.0:
        raise ValueError("degrees_of_freedom must exceed two")
    if np.any(~np.isfinite(radius)) or np.any(radius < 0.0):
        raise ValueError("standardized squared radii must be finite and non-negative")
    weight = radius / (degrees_of_freedom - 2.0 + radius)
    return 0.5 * (
        (degrees_of_freedom + dimension) * weight - dimension
    )


def score_log_mgf(
    argument: float,
    *,
    dimension: int,
    degrees_of_freedom: float,
) -> float:
    """Return the exact log-MGF of the Student-t log-scale score.

    Under the model, ``r / (nu - 2 + r)`` is
    ``Beta(dimension / 2, nu / 2)``.  Its confluent-hypergeometric MGF gives
    the normalizer without simulation or replay.
    """
    if argument == 0.0:
        return 0.0
    if dimension <= 0 or degrees_of_freedom <= 2.0:
        raise ValueError("invalid score distribution parameters")
    if not math.isfinite(argument):
        raise ValueError("MGF argument must be finite")
    score_variance = (
        dimension
        * degrees_of_freedom
        / (2.0 * (dimension + degrees_of_freedom + 2.0))
    )
    if abs(argument) < 1.0e-7:
        return 0.5 * score_variance * argument * argument
    beta_shape = 0.5 * dimension
    total_shape = 0.5 * (dimension + degrees_of_freedom)
    beta_argument = total_shape * argument
    hypergeometric = float(hyp1f1(beta_shape, total_shape, beta_argument))
    if not math.isfinite(hypergeometric) or hypergeometric <= 0.0:
        return math.inf
    return -0.5 * dimension * argument + math.log(hypergeometric)


def stationary_log_mean_normalizer(
    *,
    dimension: int,
    config: ScoreDrivenStudentTConfig,
    tolerance: float = 1.0e-12,
    maximum_terms: int = 100_000,
) -> float:
    """Return ``log E[exp(g)]`` for ``g'=beta*g+alpha*score``.

    The stationary log scale is an infinite weighted sum of independent
    scores.  Independence turns its MGF into this convergent sum of exact
    one-score log-MGFs.  Dividing ``exp(g)`` by the resulting expectation
    preserves the covariance multiplier's mean exactly in population.  The
    remaining unconditional scale is ``config.unconditional_scale``.
    """
    if config.score_loading == 0.0:
        return 0.0
    total = 0.0
    argument = config.score_loading
    for _ in range(maximum_terms):
        increment = score_log_mgf(
            argument,
            dimension=dimension,
            degrees_of_freedom=config.degrees_of_freedom,
        )
        if not math.isfinite(increment):
            return math.inf
        total += increment
        argument *= config.persistence
        if abs(increment) <= tolerance and abs(argument) <= math.sqrt(tolerance):
            return total
    raise RuntimeError("stationary scale normalizer did not converge")


def filter_score_driven_student_t(
    squared_radii: np.ndarray,
    segment_starts: np.ndarray,
    *,
    dimension: int,
    config: ScoreDrivenStudentTConfig,
) -> ScoreDrivenFilter:
    """Filter conditional scales and likelihood through observed radii."""
    radii = np.asarray(squared_radii, dtype=np.float64)
    starts = np.asarray(segment_starts, dtype=np.bool_)
    if (
        radii.ndim != 1
        or starts.shape != radii.shape
        or len(radii) == 0
        or np.any(~np.isfinite(radii))
        or np.any(radii < 0.0)
    ):
        raise ValueError("radii and segment_starts must be aligned finite vectors")
    if not starts[0]:
        raise ValueError("the first observation must start a filtering segment")
    log_normalizer = stationary_log_mean_normalizer(
        dimension=dimension,
        config=config,
    )
    if not math.isfinite(log_normalizer):
        raise ValueError("the stationary scale normalizer is not finite")

    nu = config.degrees_of_freedom
    constant = (
        gammaln(0.5 * (nu + dimension))
        - gammaln(0.5 * nu)
        - 0.5 * dimension * math.log((nu - 2.0) * math.pi)
    )
    scales = np.empty_like(radii)
    scores = np.empty_like(radii)
    log_likelihood = np.empty_like(radii)
    log_scale_state = 0.0
    for index, radius in enumerate(radii):
        if starts[index]:
            log_scale_state = 0.0
        log_scale = (
            math.log(config.unconditional_scale)
            + log_scale_state
            - log_normalizer
        )
        if log_scale < -700.0 or log_scale > 700.0:
            raise ValueError("filtered conditional scale overflowed")
        scale = math.exp(log_scale)
        standardized = radius / scale
        score = float(
            student_t_scale_score(
                standardized,
                dimension=dimension,
                degrees_of_freedom=nu,
            )
        )
        scales[index] = scale
        scores[index] = score
        log_likelihood[index] = (
            constant
            - 0.5 * dimension * log_scale
            - 0.5
            * (nu + dimension)
            * math.log1p(standardized / (nu - 2.0))
        )
        log_scale_state = (
            config.persistence * log_scale_state
            + config.score_loading * score
        )
    return ScoreDrivenFilter(
        scales=scales,
        scores=scores,
        log_likelihood=log_likelihood,
        log_mean_normalizer=log_normalizer,
    )


def _gaussian_log_likelihood(
    squared_radii: np.ndarray,
    *,
    dimension: int,
) -> np.ndarray:
    return -0.5 * (
        dimension * math.log(2.0 * math.pi) + squared_radii
    )


def fit_score_driven_student_t(
    squared_radii: np.ndarray,
    segment_starts: np.ndarray,
    *,
    dimension: int,
) -> ScoreDrivenStudentTFit:
    """Fit ``(nu, alpha, beta)`` by conditional maximum likelihood."""
    radii = np.asarray(squared_radii, dtype=np.float64)
    starts = np.asarray(segment_starts, dtype=np.bool_)
    if len(radii) < 50:
        raise ValueError("at least 50 innovations are required for fitting")

    def objective(values: np.ndarray) -> float:
        try:
            config = ScoreDrivenStudentTConfig(
                degrees_of_freedom=float(values[0]),
                score_loading=float(values[1]),
                persistence=float(values[2]),
                unconditional_scale=float(values[3]),
            )
            filtered = filter_score_driven_student_t(
                radii,
                starts,
                dimension=dimension,
                config=config,
            )
        except (ValueError, RuntimeError, OverflowError):
            return 1.0e12
        value = -float(np.mean(filtered.log_likelihood)) / dimension
        return value if math.isfinite(value) else 1.0e12

    initial_scale = float(np.mean(radii) / dimension)
    starts_to_try = (
        np.asarray((5.0, 0.005, 0.90, initial_scale)),
        np.asarray((8.0, 0.010, 0.97, initial_scale)),
        np.asarray((15.0, 0.005, 0.80, initial_scale)),
        np.asarray((40.0, 0.001, 0.50, initial_scale)),
        np.asarray((200.0, 0.0, 0.50, initial_scale)),
    )
    results: list[OptimizeResult] = []
    bounds = (
        (2.05, 500.0),
        (0.0, 0.20),
        (0.0, 0.999),
        (0.05, 20.0),
    )
    for initial in starts_to_try:
        results.append(
            minimize(
                objective,
                initial,
                method="Nelder-Mead",
                bounds=bounds,
                options={
                    "maxiter": 4_000,
                    "xatol": 1.0e-7,
                    "fatol": 1.0e-10,
                },
            )
        )
    result = min(results, key=lambda item: float(item.fun))
    config = ScoreDrivenStudentTConfig(
        degrees_of_freedom=float(result.x[0]),
        score_loading=float(result.x[1]),
        persistence=float(result.x[2]),
        unconditional_scale=float(result.x[3]),
    )
    filtered = filter_score_driven_student_t(
        radii,
        starts,
        dimension=dimension,
        config=config,
    )

    iid_result = minimize(
        lambda value: objective(
            np.asarray((float(value[0]), 0.0, 0.0, float(value[1])))
        ),
        np.asarray((10.0, initial_scale)),
        method="Nelder-Mead",
        bounds=((2.05, 500.0), (0.05, 20.0)),
        options={"maxiter": 2_000, "xatol": 1.0e-7, "fatol": 1.0e-10},
    )
    iid_nu = float(iid_result.x[0])
    iid_scale = float(iid_result.x[1])
    iid_filter = filter_score_driven_student_t(
        radii,
        starts,
        dimension=dimension,
        config=ScoreDrivenStudentTConfig(
            degrees_of_freedom=iid_nu,
            score_loading=0.0,
            persistence=0.0,
            unconditional_scale=iid_scale,
        ),
    )
    gaussian_nll = -float(
        np.mean(_gaussian_log_likelihood(radii, dimension=dimension))
    ) / dimension
    gaussian_scale = initial_scale
    scaled_gaussian_nll = -float(
        np.mean(
            _gaussian_log_likelihood(
                radii / gaussian_scale,
                dimension=dimension,
            )
            - 0.5 * dimension * math.log(gaussian_scale)
        )
    ) / dimension
    return ScoreDrivenStudentTFit(
        config=config,
        filter=filtered,
        success=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit),
        evaluations=int(result.nfev),
        negative_log_likelihood_per_dimension=float(result.fun),
        iid_gaussian_negative_log_likelihood_per_dimension=gaussian_nll,
        scaled_gaussian_negative_log_likelihood_per_dimension=scaled_gaussian_nll,
        scaled_gaussian_unconditional_scale=gaussian_scale,
        iid_student_t_negative_log_likelihood_per_dimension=(
            -float(np.mean(iid_filter.log_likelihood)) / dimension
        ),
        iid_student_t_degrees_of_freedom=iid_nu,
        iid_student_t_unconditional_scale=iid_scale,
    )


__all__ = [
    "ScoreDrivenFilter",
    "ScoreDrivenStudentTConfig",
    "ScoreDrivenStudentTFit",
    "filter_score_driven_student_t",
    "fit_score_driven_student_t",
    "score_log_mgf",
    "stationary_log_mean_normalizer",
    "student_t_scale_score",
]
