"""Batched statistical martingale audits for the daily front-spot model."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.sdlv.daily.front_spot.calibration import FrontSpotCalibration
from quantdata.sdlv.daily.front_spot.jax_model import (
    RiskNeutralState,
    black_mixture_call_exercise_probabilities,
    black_mixture_call_prices,
    black_mixture_call_second_moments,
    one_day_risk_neutral_substepped,
    substep_brownian_weights,
    terminal_front_step,
)
from quantdata.sdlv.daily.front_spot.simulation import load_initial_model_state


@dataclass(frozen=True)
class CoordinateDiagnostic:
    """Mean, standard error and z-score for one tested coordinate."""

    index: tuple[int, ...]
    mean: float
    standard_error: float
    z_score: float


@dataclass(frozen=True)
class TestSummary:
    """Multiple-coordinate summary of zero-mean Monte Carlo tests."""

    tested_count: int
    total_count: int
    max_abs_z: float
    p99_abs_z: float
    rms_z: float
    count_above_five: int
    max_abs_mean: float
    worst_index: tuple[int, ...]
    worst_mean: float
    worst_standard_error: float
    top_abs_z: tuple[CoordinateDiagnostic, ...]


@dataclass(frozen=True)
class MartingaleAuditReport:
    """Q-martingale audit covering densities, all model calls and spot."""

    initial_date: str
    path_count: int
    batch_size: int
    factor_rank: int
    dlv_substeps: int
    time_power: float
    daily_density: TestSummary
    daily_latent_options: TestSummary
    daily_smoothed_options: TestSummary
    expiry_density: TestSummary
    expiry_options: TestSummary
    daily_spot: TestSummary
    nonfinite_count: int
    max_mass_error: float
    max_forward_error: float


def _accumulate(
    sums: NDArray[np.float64],
    squares: NDArray[np.float64],
    values: NDArray[np.float64],
    index: tuple[object, ...],
) -> None:
    sums[index] += np.sum(values, axis=0)
    squares[index] += np.sum(values**2, axis=0)


def _summarize(
    sums: NDArray[np.float64],
    squares: NDArray[np.float64],
    path_count: int,
    structural_mask: NDArray[np.bool_],
    *,
    scale_floor: float = 1.0e-12,
    standard_error_override: NDArray[np.float64] | None = None,
    identifiability_mask: NDArray[np.bool_] | None = None,
) -> TestSummary:
    mean = sums / path_count
    centered_square = np.maximum(
        squares - sums**2 / path_count,
        0.0,
    )
    variance = centered_square / max(path_count - 1, 1)
    empirical_standard_error = np.sqrt(variance / path_count)
    identifiable = (
        np.ones_like(structural_mask)
        if identifiability_mask is None
        else np.asarray(identifiability_mask, dtype=np.bool_)
    )
    if identifiable.shape != structural_mask.shape:
        raise ValueError(
            f"Identifiability shape {identifiable.shape!r} does not match "
            f"structural-mask shape {structural_mask.shape!r}"
        )
    if standard_error_override is None:
        standard_error = empirical_standard_error
        active = structural_mask & identifiable & (np.sqrt(variance) > scale_floor)
    else:
        override = np.asarray(standard_error_override, dtype=np.float64)
        if override.shape != mean.shape:
            raise ValueError(
                f"Standard-error shape {override.shape!r} does not match "
                f"mean shape {mean.shape!r}"
            )
        standard_error = np.where(
            np.isfinite(override),
            np.maximum(override, empirical_standard_error),
            empirical_standard_error,
        )
        active = (
            structural_mask
            & identifiable
            & (standard_error > scale_floor / np.sqrt(path_count))
        )
    z_score = np.zeros_like(mean)
    np.divide(mean, standard_error, out=z_score, where=active)
    active_z = np.abs(z_score[active])
    if active_z.size:
        flat_active_indices = np.flatnonzero(active)
        active_worst = int(np.argmax(active_z))
        worst_flat_index = int(flat_active_indices[active_worst])
        worst_index = tuple(int(value) for value in np.unravel_index(worst_flat_index, mean.shape))
        max_abs_z = float(active_z[active_worst])
        p99_abs_z = float(np.quantile(active_z, 0.99))
        rms_z = float(np.sqrt(np.mean(active_z**2)))
        count_above_five = int(np.sum(active_z > 5.0))
        worst_mean = float(mean.ravel()[worst_flat_index])
        worst_standard_error = float(standard_error.ravel()[worst_flat_index])
        top_order = np.argsort(active_z)[-10:][::-1]
        top_abs_z = tuple(
            CoordinateDiagnostic(
                index=tuple(
                    int(value)
                    for value in np.unravel_index(
                        int(flat_active_indices[active_index]),
                        mean.shape,
                    )
                ),
                mean=float(mean.ravel()[flat_active_indices[active_index]]),
                standard_error=float(
                    standard_error.ravel()[flat_active_indices[active_index]]
                ),
                z_score=float(z_score.ravel()[flat_active_indices[active_index]]),
            )
            for active_index in top_order
        )
    else:
        worst_index = tuple(0 for _ in mean.shape)
        max_abs_z = 0.0
        p99_abs_z = 0.0
        rms_z = 0.0
        count_above_five = 0
        worst_mean = 0.0
        worst_standard_error = 0.0
        top_abs_z = ()
    structural_means = np.abs(mean[structural_mask])
    return TestSummary(
        tested_count=int(np.sum(active)),
        total_count=int(np.sum(structural_mask)),
        max_abs_z=max_abs_z,
        p99_abs_z=p99_abs_z,
        rms_z=rms_z,
        count_above_five=count_above_five,
        max_abs_mean=(
            float(np.max(structural_means)) if structural_means.size else 0.0
        ),
        worst_index=worst_index,
        worst_mean=worst_mean,
        worst_standard_error=worst_standard_error,
        top_abs_z=top_abs_z,
    )


def _latent_call_prices(
    densities: NDArray[np.float64],
    support: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Value all interior-strike calls by reverse cumulative sums."""
    reverse_mass = np.cumsum(densities[..., ::-1], axis=-1)[..., ::-1]
    reverse_first_moment = np.cumsum(
        (densities * support)[..., ::-1],
        axis=-1,
    )[..., ::-1]
    return (
        reverse_first_moment[..., 2:]
        - support[1:-1] * reverse_mass[..., 2:]
    )


def run_risk_neutral_martingale_audit(
    calibration: FrontSpotCalibration,
    *,
    fit_label: str = "full",
    date_index: int = -1,
    path_count: int = 1_000,
    batch_size: int = 32,
    dlv_substeps: int = 8,
    time_power: float = 1.0,
    seed: int = 20261004,
    dt: float = 1.0 / DAYS_PER_YEAR,
) -> MartingaleAuditReport:
    """Audit all daily model densities and calls through expiry under Q.

    Daily density martingality tests every linear claim on the latent discrete
    process.  Daily smoothed-call tests value every live interior-strike call at
    every remaining model expiry, including the independent 5% Black background.
    The option-to-expiry test separately evaluates the same all-strike payoff
    family at settlement.
    """
    if not jax.config.x64_enabled:
        raise RuntimeError("Martingale auditing requires JAX 64-bit mode")
    if path_count < 2:
        raise ValueError(f"path_count must be at least two, found {path_count}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, found {batch_size}")
    if dlv_substeps < 1:
        raise ValueError(f"dlv_substeps must be positive, found {dlv_substeps}")
    if time_power < 1.0 or not np.isfinite(time_power):
        raise ValueError(f"time_power must be finite and at least one, found {time_power}")

    initial = load_initial_model_state(
        fit_label=fit_label,
        date_index=date_index,
        dt=dt,
    )
    expiry_count, node_count = initial.density_tower.shape
    strike_count = node_count - 2
    factor_rank = calibration.innovations.factor_rank
    support = jnp.asarray(initial.support, dtype=jnp.float64)
    strikes = jnp.asarray(initial.support[1:-1], dtype=jnp.float64)
    loadings = jnp.asarray(
        calibration.innovations.annualized_loadings,
        dtype=jnp.float64,
    )
    initial_tower = jnp.asarray(initial.density_tower, dtype=jnp.float64)
    latent_payoffs = np.maximum(
        initial.support[:, None] - initial.support[None, 1:-1],
        0.0,
    )
    initial_calls = np.asarray(
        black_mixture_call_prices(
            support,
            initial_tower,
            strikes,
            jnp.asarray(1.0, dtype=jnp.float64),
            jnp.arange(1, expiry_count + 1, dtype=jnp.float64),
        )
    )
    initial_call_second_moments = np.asarray(
        black_mixture_call_second_moments(
            support,
            initial_tower,
            strikes,
            jnp.asarray(1.0, dtype=jnp.float64),
            jnp.arange(1, expiry_count + 1, dtype=jnp.float64),
        )
    )
    initial_call_exercise_probabilities = np.asarray(
        black_mixture_call_exercise_probabilities(
            support,
            initial_tower,
            strikes,
            jnp.asarray(1.0, dtype=jnp.float64),
            jnp.arange(1, expiry_count + 1, dtype=jnp.float64),
        )
    )
    if initial_calls.shape != (expiry_count, strike_count):
        raise RuntimeError(f"Unexpected initial call shape {initial_calls.shape!r}")

    daily_density_sum = np.zeros(
        (expiry_count, expiry_count, node_count), dtype=np.float64
    )
    daily_density_square = np.zeros_like(daily_density_sum)
    daily_front_predictable_variance = np.zeros(
        (expiry_count, node_count),
        dtype=np.float64,
    )
    daily_latent_option_sum = np.zeros(
        (expiry_count, expiry_count, strike_count), dtype=np.float64
    )
    daily_latent_option_square = np.zeros_like(daily_latent_option_sum)
    daily_smoothed_option_sum = np.zeros_like(daily_latent_option_sum)
    daily_smoothed_option_square = np.zeros_like(daily_latent_option_sum)
    daily_front_option_predictable_variance = np.zeros(
        (expiry_count, strike_count),
        dtype=np.float64,
    )
    expiry_density_sum = np.zeros((expiry_count, node_count), dtype=np.float64)
    expiry_density_square = np.zeros_like(expiry_density_sum)
    expiry_option_sum = np.zeros((expiry_count, strike_count), dtype=np.float64)
    expiry_option_square = np.zeros_like(expiry_option_sum)
    daily_spot_sum = np.zeros(expiry_count, dtype=np.float64)
    daily_spot_square = np.zeros_like(daily_spot_sum)
    nonfinite_count = 0
    max_mass_error = 0.0
    max_forward_error = 0.0
    generator = np.random.default_rng(seed)
    daily_step = jax.jit(
        one_day_risk_neutral_substepped,
        static_argnames=("expiry_dt", "time_power"),
    )
    final_step = jax.jit(
        terminal_front_step,
        static_argnames=("dt",),
    )
    smoothed_call_pricer = jax.jit(black_mixture_call_prices)

    completed = 0
    while completed < path_count:
        valid_count = min(batch_size, path_count - completed)
        factor_normals = generator.standard_normal(
            (expiry_count, dlv_substeps, batch_size, factor_rank)
        )
        background_normals = generator.standard_normal((expiry_count, batch_size))
        q_state = RiskNeutralState(
            initial_tower[0],
            jnp.asarray(initial.dlvsq[1:], dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
        )
        current_tower = np.broadcast_to(
            initial.density_tower,
            (batch_size,) + initial.density_tower.shape,
        ).copy()
        current_spot = np.ones(batch_size, dtype=np.float64)

        for day_index in range(expiry_count - 1):
            remaining_count = expiry_count - day_index
            current_smoothed_calls = np.asarray(
                smoothed_call_pricer(
                    support,
                    jnp.asarray(current_tower, dtype=jnp.float64),
                    strikes,
                    q_state.background_level,
                    jnp.arange(1, remaining_count + 1, dtype=jnp.float64),
                )
            )
            future_count = q_state.future_dlvsq.shape[-2]
            result = daily_step(
                support,
                q_state,
                loadings[:future_count],
                jnp.asarray(factor_normals[day_index], dtype=jnp.float64),
                jnp.asarray(background_normals[day_index], dtype=jnp.float64),
                expiry_dt=dt,
                time_power=time_power,
            )
            sampled_density = np.asarray(
                jax.nn.one_hot(
                    result.sampled_front_index,
                    node_count,
                    dtype=jnp.float64,
                )
            )
            future_tower = np.asarray(result.density_tower)
            after_tower = np.concatenate(
                (sampled_density[:, None, :], future_tower),
                axis=1,
            )
            daily_increment = (
                after_tower[:valid_count] - current_tower[:valid_count]
            )
            current_front = current_tower[:valid_count, 0]
            daily_front_predictable_variance[day_index] += np.sum(
                current_front * (1.0 - current_front),
                axis=0,
            )
            _accumulate(
                daily_density_sum,
                daily_density_square,
                daily_increment,
                (day_index, slice(0, remaining_count), slice(None)),
            )
            current_latent_calls = _latent_call_prices(
                current_tower[:valid_count],
                initial.support,
            )
            current_front_second_moment = (
                current_front @ (latent_payoffs**2)
            )
            daily_front_option_predictable_variance[day_index] += np.sum(
                np.maximum(
                    current_front_second_moment - current_latent_calls[:, 0] ** 2,
                    0.0,
                ),
                axis=0,
            )
            after_latent_calls = _latent_call_prices(
                after_tower[:valid_count],
                initial.support,
            )
            _accumulate(
                daily_latent_option_sum,
                daily_latent_option_square,
                after_latent_calls - current_latent_calls,
                (day_index, slice(0, remaining_count), slice(None)),
            )
            after_smoothed_calls = np.empty_like(current_smoothed_calls)
            after_smoothed_calls[:, 0] = np.maximum(
                np.asarray(result.spot)[:, None] - initial.support[None, 1:-1],
                0.0,
            )
            after_smoothed_calls[:, 1:] = np.asarray(
                smoothed_call_pricer(
                    support,
                    result.density_tower,
                    strikes,
                    result.state.background_level,
                    jnp.arange(1, remaining_count, dtype=jnp.float64),
                )
            )
            _accumulate(
                daily_smoothed_option_sum,
                daily_smoothed_option_square,
                (
                    after_smoothed_calls[:valid_count]
                    - current_smoothed_calls[:valid_count]
                ),
                (day_index, slice(day_index, expiry_count), slice(None)),
            )
            terminal_density_error = (
                sampled_density[:valid_count] - initial.density_tower[day_index]
            )
            _accumulate(
                expiry_density_sum,
                expiry_density_square,
                terminal_density_error,
                (day_index, slice(None)),
            )
            spot = np.asarray(result.spot)
            payoff_error = np.maximum(
                spot[:valid_count, None] - initial.support[None, 1:-1],
                0.0,
            ) - initial_calls[day_index]
            _accumulate(
                expiry_option_sum,
                expiry_option_square,
                payoff_error,
                (day_index, slice(None)),
            )
            spot_increment = spot[:valid_count] - current_spot[:valid_count]
            _accumulate(
                daily_spot_sum,
                daily_spot_square,
                spot_increment,
                (day_index,),
            )
            mass_error = np.max(np.abs(np.sum(future_tower, axis=-1) - 1.0))
            forward_error = np.max(
                np.abs(
                    future_tower @ initial.support
                    - np.asarray(result.latent_spot)[..., None]
                )
            )
            max_mass_error = max(max_mass_error, float(mass_error))
            max_forward_error = max(max_forward_error, float(forward_error))
            diagnostic_values = (future_tower, np.asarray(result.state.future_dlvsq), spot)
            nonfinite_count += sum(
                int(np.sum(~np.isfinite(value[:valid_count])))
                for value in diagnostic_values
            )
            q_state = result.state
            current_tower = future_tower
            current_spot = spot

        daily_terminal_normal = np.einsum(
            "s,sb->b",
            np.asarray(
                substep_brownian_weights(
                    dlv_substeps,
                    time_power=time_power,
                    dtype=jnp.float64,
                )
            ),
            factor_normals[-1, :, :, 0],
        )
        terminal = final_step(
            support,
            q_state.front_density,
            q_state.background_level,
            jnp.asarray(daily_terminal_normal, dtype=jnp.float64),
            jnp.asarray(background_normals[-1], dtype=jnp.float64),
            dt=dt,
        )
        sampled_density = np.asarray(
            jax.nn.one_hot(
                terminal.sampled_front_index,
                node_count,
                dtype=jnp.float64,
            )
        )
        terminal_current_smoothed_calls = np.asarray(
            smoothed_call_pricer(
                support,
                jnp.asarray(current_tower, dtype=jnp.float64),
                strikes,
                q_state.background_level,
                jnp.asarray([1.0], dtype=jnp.float64),
            )
        )
        terminal_smoothed_calls = np.maximum(
            np.asarray(terminal.spot)[:, None] - initial.support[None, 1:-1],
            0.0,
        )
        _accumulate(
            daily_smoothed_option_sum,
            daily_smoothed_option_square,
            terminal_smoothed_calls[:valid_count, None, :]
            - terminal_current_smoothed_calls[:valid_count],
            (expiry_count - 1, slice(expiry_count - 1, expiry_count), slice(None)),
        )
        daily_increment = sampled_density[:valid_count] - current_tower[:valid_count, 0]
        current_front = current_tower[:valid_count, 0]
        daily_front_predictable_variance[-1] += np.sum(
            current_front * (1.0 - current_front),
            axis=0,
        )
        _accumulate(
            daily_density_sum,
            daily_density_square,
            daily_increment,
            (expiry_count - 1, 0, slice(None)),
        )
        terminal_current_calls = _latent_call_prices(
            current_tower[:valid_count],
            initial.support,
        )[:, 0]
        terminal_front_second_moment = current_front @ (latent_payoffs**2)
        daily_front_option_predictable_variance[-1] += np.sum(
            np.maximum(
                terminal_front_second_moment - terminal_current_calls**2,
                0.0,
            ),
            axis=0,
        )
        terminal_payoffs = np.maximum(
            np.asarray(terminal.latent_spot)[:valid_count, None]
            - initial.support[None, 1:-1],
            0.0,
        )
        _accumulate(
            daily_latent_option_sum,
            daily_latent_option_square,
            terminal_payoffs - terminal_current_calls,
            (expiry_count - 1, 0, slice(None)),
        )
        terminal_density_error = sampled_density[:valid_count] - initial.density_tower[-1]
        _accumulate(
            expiry_density_sum,
            expiry_density_square,
            terminal_density_error,
            (expiry_count - 1, slice(None)),
        )
        spot = np.asarray(terminal.spot)
        payoff_error = np.maximum(
            spot[:valid_count, None] - initial.support[None, 1:-1],
            0.0,
        ) - initial_calls[-1]
        _accumulate(
            expiry_option_sum,
            expiry_option_square,
            payoff_error,
            (expiry_count - 1, slice(None)),
        )
        spot_increment = spot[:valid_count] - current_spot[:valid_count]
        _accumulate(
            daily_spot_sum,
            daily_spot_square,
            spot_increment,
            (expiry_count - 1,),
        )
        nonfinite_count += int(np.sum(~np.isfinite(spot[:valid_count])))
        completed += valid_count

    daily_mask = np.zeros_like(daily_density_sum, dtype=np.bool_)
    daily_density_identifiable = np.zeros_like(daily_mask)
    daily_latent_option_mask = np.zeros_like(
        daily_latent_option_sum,
        dtype=np.bool_,
    )
    daily_latent_option_identifiable = np.zeros_like(
        daily_latent_option_mask,
    )
    daily_smoothed_option_mask = np.zeros_like(
        daily_latent_option_mask,
    )
    daily_smoothed_option_identifiable = np.zeros_like(
        daily_latent_option_mask,
    )
    initial_latent_exercise_probability = np.cumsum(
        initial.density_tower[..., ::-1],
        axis=-1,
    )[..., ::-1][..., 2:]
    for day_index in range(expiry_count):
        remaining_count = expiry_count - day_index
        daily_mask[day_index, :remaining_count] = True
        daily_latent_option_mask[day_index, :remaining_count] = True
        probability = initial.density_tower[day_index:]
        daily_density_identifiable[day_index, :remaining_count] = (
            (probability * path_count >= 20.0)
            & ((1.0 - probability) * path_count >= 20.0)
        )
        daily_latent_option_identifiable[day_index, :remaining_count] = (
            initial_latent_exercise_probability[day_index:] * path_count >= 20.0
        )
        daily_smoothed_option_mask[day_index, day_index:] = True
        daily_smoothed_option_identifiable[day_index, day_index:] = (
            initial_call_exercise_probabilities[day_index:] * path_count >= 20.0
        )
    daily_density_standard_error = np.full_like(
        daily_density_sum,
        np.nan,
    )
    daily_density_standard_error[:, 0, :] = (
        np.sqrt(daily_front_predictable_variance) / path_count
    )
    daily_latent_option_standard_error = np.full_like(
        daily_latent_option_sum,
        np.nan,
    )
    daily_latent_option_standard_error[:, 0, :] = (
        np.sqrt(daily_front_option_predictable_variance) / path_count
    )
    expiry_probability = initial.density_tower
    expiry_density_mask = np.ones_like(expiry_probability, dtype=np.bool_)
    expiry_density_identifiable = (
        (expiry_probability * path_count >= 20.0)
        & ((1.0 - expiry_probability) * path_count >= 20.0)
    )
    expiry_density_standard_error = np.sqrt(
        expiry_probability * (1.0 - expiry_probability) / path_count
    )
    expiry_option_variance = np.maximum(
        initial_call_second_moments - initial_calls**2,
        0.0,
    )
    expiry_option_standard_error = np.sqrt(expiry_option_variance / path_count)
    expiry_option_mask = np.ones_like(initial_calls, dtype=np.bool_)
    expiry_option_identifiable = (
        initial_call_exercise_probabilities * path_count >= 20.0
    )
    return MartingaleAuditReport(
        initial_date=initial.date,
        path_count=path_count,
        batch_size=batch_size,
        factor_rank=factor_rank,
        dlv_substeps=dlv_substeps,
        time_power=time_power,
        daily_density=_summarize(
            daily_density_sum,
            daily_density_square,
            path_count,
            daily_mask,
            scale_floor=1.0e-8,
            standard_error_override=daily_density_standard_error,
            identifiability_mask=daily_density_identifiable,
        ),
        daily_latent_options=_summarize(
            daily_latent_option_sum,
            daily_latent_option_square,
            path_count,
            daily_latent_option_mask,
            scale_floor=1.0e-7,
            standard_error_override=daily_latent_option_standard_error,
            identifiability_mask=daily_latent_option_identifiable,
        ),
        daily_smoothed_options=_summarize(
            daily_smoothed_option_sum,
            daily_smoothed_option_square,
            path_count,
            daily_smoothed_option_mask,
            scale_floor=1.0e-8,
            identifiability_mask=daily_smoothed_option_identifiable,
        ),
        expiry_density=_summarize(
            expiry_density_sum,
            expiry_density_square,
            path_count,
            expiry_density_mask,
            standard_error_override=expiry_density_standard_error,
            identifiability_mask=expiry_density_identifiable,
        ),
        expiry_options=_summarize(
            expiry_option_sum,
            expiry_option_square,
            path_count,
            expiry_option_mask,
            scale_floor=1.0e-7,
            standard_error_override=expiry_option_standard_error,
            identifiability_mask=expiry_option_identifiable,
        ),
        daily_spot=_summarize(
            daily_spot_sum,
            daily_spot_square,
            path_count,
            np.ones(expiry_count, dtype=np.bool_),
        ),
        nonfinite_count=nonfinite_count,
        max_mass_error=max_mass_error,
        max_forward_error=max_forward_error,
    )
