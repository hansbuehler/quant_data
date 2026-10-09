"""Short-end spot process for the genuine Tangent Levy v2 model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cdxcore.err import verify_inp
from scipy.stats import poisson

from .coordinates import trapezoid_weights


@dataclass(frozen=True)
class SpotStep:
    """One simulated exponential-Levy spot transition."""

    gross_return: np.ndarray
    log_return: np.ndarray
    jump_count: np.ndarray
    compensator: np.ndarray


def short_end_characteristics(
    short_end_density: np.ndarray,
    *,
    jump_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return intensity, exponential compensator, and jump probabilities.

    The finite-grid short-end codebook is interpreted as a compound-Poisson
    Levy measure.  The deterministic log drift is minus the exponential
    compensator, so ``E_Q[S_{t+dt}/S_t | F_t] = 1``.
    """

    density = np.asarray(short_end_density, dtype=np.float64)
    jump = np.asarray(jump_x, dtype=np.float64)
    verify_inp(
        density.ndim in {1, 2}
        and density.shape[-1] == jump.size
        and np.all(np.isfinite(density))
        and np.all(density >= 0.0),
        lambda: (
            "short-end density must be finite, non-negative, and have trailing "
            f"size {jump.size}; found {density.shape}"
        ),
    )
    weights = trapezoid_weights(jump)
    weighted = density * weights
    intensity = np.sum(weighted, axis=-1)
    verify_inp(
        np.all(intensity > 0.0),
        lambda: "every simulated short-end Levy measure must have positive mass",
    )
    compensator = weighted @ np.expm1(jump)
    probabilities = weighted / intensity[..., None]
    return intensity, compensator, probabilities


def _paired_uniforms(pair_count: int, rng: np.random.Generator) -> np.ndarray:
    base = rng.random(pair_count)
    return np.column_stack((base, 1.0 - base)).reshape(-1)


def sample_short_end_spot_step(
    short_end_density: np.ndarray,
    *,
    jump_x: np.ndarray,
    dt: float,
    rng: np.random.Generator,
    antithetic_pairs: bool = True,
) -> SpotStep:
    """Sample one spot transition from the Tangent Levy short-end compensator.

    Consecutive paths form antithetic pairs when ``antithetic_pairs`` is true.
    Both the Poisson count uniform and every categorical jump uniform are
    reflected within the pair.  This preserves each marginal exactly; it does
    not assume that paired observations are independent.
    """

    verify_inp(
        np.isfinite(dt) and dt > 0.0,
        lambda: f"dt must be finite-positive, found {dt!r}",
    )
    density = np.asarray(short_end_density, dtype=np.float64)
    scalar_state = density.ndim == 1
    if scalar_state:
        density = density[None, :]
    path_count = density.shape[0]
    verify_inp(
        not antithetic_pairs or (path_count >= 2 and path_count % 2 == 0),
        lambda: "antithetic simulation requires a positive even path count",
    )
    intensity, compensator, probabilities = short_end_characteristics(
        density,
        jump_x=jump_x,
    )
    count_uniform = (
        _paired_uniforms(path_count // 2, rng)
        if antithetic_pairs
        else rng.random(path_count)
    )
    # scipy's inverse is deterministic and makes the antithetic definition
    # explicit.  Avoid the endpoint one, where the generalized inverse is inf.
    count_uniform = np.minimum(count_uniform, np.nextafter(1.0, 0.0))
    jump_count = poisson.ppf(count_uniform, intensity * dt).astype(np.int32)
    maximum_count = int(np.max(jump_count, initial=0))
    jump_sum = np.zeros(path_count, dtype=np.float64)
    cumulative = np.cumsum(probabilities, axis=-1)
    cumulative[..., -1] = 1.0
    for jump_number in range(maximum_count):
        active = jump_count > jump_number
        if not np.any(active):
            break
        jump_uniform = (
            _paired_uniforms(path_count // 2, rng)
            if antithetic_pairs
            else rng.random(path_count)
        )
        selected = np.sum(
            jump_uniform[:, None] > cumulative,
            axis=-1,
            dtype=np.int64,
        )
        jump_sum[active] += np.asarray(jump_x, dtype=np.float64)[selected[active]]
    log_return = jump_sum - compensator * dt
    gross_return = np.exp(log_return)
    verify_inp(
        np.all(np.isfinite(gross_return)) and np.all(gross_return > 0.0),
        lambda: "spot simulation produced a non-finite or non-positive return",
    )
    if scalar_state:
        return SpotStep(
            gross_return=gross_return[:1],
            log_return=log_return[:1],
            jump_count=jump_count[:1],
            compensator=compensator[:1],
        )
    return SpotStep(
        gross_return=gross_return,
        log_return=log_return,
        jump_count=jump_count,
        compensator=compensator,
    )
