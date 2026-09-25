"""Correlated front-expiry spot sampling for Direct-200 surface dynamics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtr

from quantdata.dysanos.flex.model import JointDirectDynamicsModel, bound_native_raw
from quantdata.dysanos.flex.rolling_hedging import one_session_pairs
from quantdata.dysanos.flex.weekly_simulation import front_week_sanos_gaussian_score


@dataclass(frozen=True)
class FrontSpotCopulaFit:
    """Regularized Gaussian copula between state shocks and front-expiry spot."""

    state_loading: np.ndarray
    orthogonal_std: float
    score_mean: float
    score_std: float
    ridge_penalty: float
    observations: int
    fit_end: str
    in_sample_correlation: float
    explained_fraction: float

    def __post_init__(self) -> None:
        if (
            self.state_loading.ndim != 1
            or np.any(~np.isfinite(self.state_loading))
            or not math.isfinite(self.orthogonal_std)
            or self.orthogonal_std <= 0.0
            or not math.isfinite(self.score_mean)
            or not math.isfinite(self.score_std)
            or self.score_std <= 0.0
            or not math.isfinite(self.ridge_penalty)
            or self.ridge_penalty <= 0.0
            or self.observations <= 0
        ):
            raise ValueError("invalid front-spot copula fit")

    @property
    def total_std(self) -> float:
        """Return the pre-normalization standard deviation of the spot score."""
        loading_variance = float(self.state_loading @ self.state_loading)
        return math.sqrt(loading_variance + self.orthogonal_std**2)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible fit record."""
        result = asdict(self)
        result["state_loading"] = self.state_loading.tolist()
        result["state_loading_norm"] = float(np.linalg.norm(self.state_loading))
        result["total_std"] = self.total_std
        return result


def _decode_surface(model: JointDirectDynamicsModel, states: torch.Tensor):
    raw = model.raw_states_from_states(states)
    parameters = bound_native_raw(raw, geometry=model.config.geometry)
    return model.sanos_decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )


def fit_gaussian_front_spot_copula(
    state_normals: np.ndarray,
    spot_scores: np.ndarray,
    *,
    fit_end: str,
    ridge_multipliers: np.ndarray | None = None,
) -> FrontSpotCopulaFit:
    """Fit a time-ordered ridge regression for the front Gaussian score."""
    states = np.asarray(state_normals, dtype=np.float64)
    scores = np.asarray(spot_scores, dtype=np.float64)
    if (
        states.ndim != 2
        or scores.shape != (len(states),)
        or len(states) < max(states.shape[1] + 20, 100)
        or np.any(~np.isfinite(states))
        or np.any(~np.isfinite(scores))
    ):
        raise ValueError("copula fit requires aligned finite observations with headroom")
    score_mean = float(np.mean(scores))
    score_std = float(np.std(scores, ddof=1))
    if score_std <= np.finfo(np.float64).tiny:
        raise ValueError("front Gaussian scores have zero variance")
    x = states - np.mean(states, axis=0, keepdims=True)
    y = (scores - score_mean) / score_std
    multipliers = (
        np.logspace(-4.0, 4.0, 25)
        if ridge_multipliers is None
        else np.asarray(ridge_multipliers, dtype=np.float64)
    )
    if (
        multipliers.ndim != 1
        or len(multipliers) == 0
        or np.any(~np.isfinite(multipliers))
        or np.any(multipliers <= 0.0)
    ):
        raise ValueError("ridge multipliers must be positive and finite")

    first_validation = max(states.shape[1] + 10, len(states) // 2)
    boundaries = np.unique(
        np.linspace(first_validation, len(states), num=6, dtype=np.int64)
    )
    validation_error = np.zeros(len(multipliers), dtype=np.float64)
    validation_count = np.zeros(len(multipliers), dtype=np.int64)
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        train_x = x[:start]
        train_y = y[:start]
        validation_x = x[start:stop]
        validation_y = y[start:stop]
        if len(validation_y) == 0:
            continue
        u, singular, vt = np.linalg.svd(train_x, full_matrices=False)
        projected = u.T @ train_y
        penalties = start * multipliers
        scaled = (
            singular[:, None]
            * projected[:, None]
            / (singular[:, None] ** 2 + penalties[None, :])
        )
        coefficients = vt.T @ scaled
        predictions = validation_x @ coefficients
        validation_error += np.sum(
            np.square(predictions - validation_y[:, None]), axis=0
        )
        validation_count += len(validation_y)
    if np.any(validation_count == 0):
        raise RuntimeError("time-ordered ridge validation produced an empty fold")
    selected = int(np.argmin(validation_error / validation_count))
    penalty = float(len(states) * multipliers[selected])
    loading = np.linalg.solve(
        x.T @ x + penalty * np.eye(states.shape[1], dtype=np.float64),
        x.T @ y,
    )
    prediction = x @ loading
    residual = y - prediction
    orthogonal_std = float(np.sqrt(np.mean(np.square(residual))))
    loading_norm_sq = float(loading @ loading)
    total_variance = loading_norm_sq + orthogonal_std**2
    correlation = (
        float(np.corrcoef(y, prediction)[0, 1])
        if np.std(prediction) > np.finfo(np.float64).tiny
        else 0.0
    )
    return FrontSpotCopulaFit(
        state_loading=loading,
        orthogonal_std=orthogonal_std,
        score_mean=score_mean,
        score_std=score_std,
        ridge_penalty=penalty,
        observations=len(states),
        fit_end=fit_end,
        in_sample_correlation=correlation,
        explained_fraction=loading_norm_sq / total_variance,
    )


def fit_front_spot_copula_from_history(
    *,
    model: JointDirectDynamicsModel,
    states: np.ndarray,
    spots: np.ndarray,
    dates: list[str],
    daily_transition: np.ndarray,
    daily_factor: np.ndarray,
    fit_before: date,
    batch_size: int = 256,
) -> tuple[FrontSpotCopulaFit, dict[str, float | int | str]]:
    """Fit the front-score copula on transitions preceding the evaluation."""
    if states.shape != (len(dates), model.config.state_size):
        raise ValueError("state history does not match Direct-200 geometry")
    if spots.shape != (len(dates),) or np.any(~np.isfinite(spots)) or np.any(spots <= 0.0):
        raise ValueError("spot history must be positive, finite, and aligned")
    origin, endpoint = one_session_pairs(dates)
    parsed = np.asarray([date.fromisoformat(value) for value in dates], dtype=object)
    training = parsed[endpoint] < fit_before
    evaluation = parsed[origin] >= fit_before
    if int(np.sum(training)) < max(model.config.state_size + 20, 100):
        raise ValueError("insufficient pre-evaluation observations for spot copula")
    mean = model.equilibrium_mean.detach().cpu().double().numpy()
    expected = mean + (states[origin] - mean) @ daily_transition
    residual = states[endpoint] - expected
    state_normals = np.linalg.solve(daily_factor, residual.T).T
    gross_return = spots[endpoint] / spots[origin]
    score = np.empty(len(origin), dtype=np.float64)
    with torch.inference_mode():
        for first in range(0, len(origin), batch_size):
            last = min(first + batch_size, len(origin))
            state_batch = torch.as_tensor(
                states[origin[first:last]], dtype=model.fdtype, device=model.device
            )
            gross_batch = torch.as_tensor(
                gross_return[first:last], dtype=model.fdtype, device=model.device
            )
            score[first:last] = (
                front_week_sanos_gaussian_score(
                    _decode_surface(model, state_batch), gross_batch
                )
                .cpu()
                .double()
                .numpy()
            )
    fit = fit_gaussian_front_spot_copula(
        state_normals[training],
        score[training],
        fit_end=dates[int(endpoint[training][-1])],
    )
    standardized_score = (score - fit.score_mean) / fit.score_std
    modeled_score = state_normals @ fit.state_loading
    training_correlation = (
        float(np.corrcoef(standardized_score[training], modeled_score[training])[0, 1])
        if np.std(modeled_score[training]) > np.finfo(np.float64).tiny
        else 0.0
    )
    evaluation_correlation = (
        float(np.corrcoef(standardized_score[evaluation], modeled_score[evaluation])[0, 1])
        if int(np.sum(evaluation)) > 1
        and np.std(modeled_score[evaluation]) > np.finfo(np.float64).tiny
        else 0.0
    )
    diagnostics: dict[str, float | int | str] = {
        "fit_before": fit_before.isoformat(),
        "first_training_transition": dates[int(origin[training][0])],
        "last_training_transition": dates[int(endpoint[training][-1])],
        "training_observations": int(np.sum(training)),
        "training_raw_score_mean": float(np.mean(score[training])),
        "training_raw_score_standard_deviation": float(
            np.std(score[training], ddof=1)
        ),
        "training_state_score_correlation": training_correlation,
        "evaluation_observations": int(np.sum(evaluation)),
        "evaluation_state_score_correlation": evaluation_correlation,
    }
    return fit, diagnostics


def antithetic_normals(
    *,
    count: int,
    dimension: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return antithetic state and idiosyncratic spot normals."""
    if count <= 0 or count % 2 != 0 or dimension <= 0:
        raise ValueError("antithetic sampling requires positive even count and dimension")
    generator = np.random.default_rng(seed)
    half_state = generator.standard_normal((count // 2, dimension))
    half_spot = generator.standard_normal(count // 2)
    return (
        np.concatenate((half_state, -half_state), axis=0),
        np.concatenate((half_spot, -half_spot), axis=0),
    )


def _front_quantile_spline(
    *,
    support: np.ndarray,
    density: np.ndarray,
    sqrt_variance: float,
    gaussian_score: np.ndarray,
    grid_size: int = 4_097,
) -> np.ndarray:
    """Invert one front CDF with a monotone PCHIP quantile interpolant."""
    support_values = np.asarray(support, dtype=np.float64)
    density_values = np.maximum(np.asarray(density, dtype=np.float64), 0.0)
    if (
        support_values.ndim != 1
        or density_values.shape != support_values.shape
        or np.any(~np.isfinite(support_values))
        or np.any(support_values <= 0.0)
        or np.any(~np.isfinite(density_values))
        or np.sum(density_values) <= 0.0
        or not math.isfinite(sqrt_variance)
        or sqrt_variance <= 0.0
        or grid_size < 257
    ):
        raise ValueError("invalid front mixture for monotone quantile interpolation")
    density_values /= np.sum(density_values)
    support_values /= float(density_values @ support_values)
    lower = math.log(float(support_values[0])) - 10.0 * sqrt_variance
    upper = math.log(float(support_values[-1])) + 10.0 * sqrt_variance
    log_grid = np.linspace(lower, upper, grid_size)
    z = (
        log_grid[:, None]
        - np.log(support_values)[None, :]
        + 0.5 * sqrt_variance**2
    ) / sqrt_variance
    cdf = np.maximum.accumulate(ndtr(z) @ density_values)
    keep = np.concatenate(([True], np.diff(cdf) > 16.0 * np.finfo(np.float64).eps))
    cdf = cdf[keep]
    log_grid = log_grid[keep]
    if len(cdf) < 2:
        raise RuntimeError("front CDF grid collapsed during spline construction")
    probability = np.clip(ndtr(gaussian_score), cdf[0], cdf[-1])
    quantile = PchipInterpolator(cdf, log_grid, extrapolate=False)(probability)
    gross_return = np.exp(quantile)
    if np.any(~np.isfinite(gross_return)) or np.any(gross_return <= 0.0):
        raise RuntimeError("front quantile spline produced invalid spot returns")
    return gross_return


def sample_correlated_front_scenarios(
    *,
    model: JointDirectDynamicsModel,
    current_state: np.ndarray,
    daily_transition: np.ndarray,
    daily_factor: np.ndarray,
    fit: FrontSpotCopulaFit,
    state_normals: np.ndarray,
    independent_spot_normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample next states and front-expiry spot from one fitted Gaussian copula."""
    state = np.asarray(current_state, dtype=np.float64)
    normals = np.asarray(state_normals, dtype=np.float64)
    independent = np.asarray(independent_spot_normals, dtype=np.float64)
    if (
        state.shape != (model.config.state_size,)
        or normals.ndim != 2
        or normals.shape[1] != model.config.state_size
        or independent.shape != (len(normals),)
        or fit.state_loading.shape != (model.config.state_size,)
    ):
        raise ValueError("joint front scenario geometry is inconsistent")
    mean = model.equilibrium_mean.detach().cpu().double().numpy()
    expected = mean + (state - mean) @ daily_transition
    next_states = expected[None, :] + normals @ daily_factor.T
    spot_score = (
        normals @ fit.state_loading + fit.orthogonal_std * independent
    ) / fit.total_std
    with torch.inference_mode():
        surface = _decode_surface(
            model,
            torch.as_tensor(state[None, :], dtype=model.fdtype, device=model.device),
        )
        support = surface.support_strikes[0, 0].cpu().double().numpy()
        density = surface.densities[0, 0].cpu().double().numpy()
        sqrt_variance = float(surface.backbone_total_vols[0, 0].cpu())
    gross_return = _front_quantile_spline(
        support=support,
        density=density,
        sqrt_variance=sqrt_variance,
        gaussian_score=spot_score,
    )
    return next_states, gross_return


__all__ = [
    "FrontSpotCopulaFit",
    "antithetic_normals",
    "fit_front_spot_copula_from_history",
    "fit_gaussian_front_spot_copula",
    "sample_correlated_front_scenarios",
]
