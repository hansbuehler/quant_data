"""Joint codebook/spot simulation for the Tangent Levy v2 audits."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from cdxcore.err import verify, verify_inp
from scipy.special import ndtri

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS

from .constraints import hjm_drift
from .dynamics import DynamicsFit
from .physical_measure import PhysicalMeasureFit
from .pricing import average_density_from_forward, density_call_prices
from .v2_audit import RestrictedPopulation
from .v2_config import TangentLevyV2AuditConfig
from .v2_spot import sample_short_end_spot_step

NORMALIZED_MONEYNESS = np.asarray(
    (-1.5, -1.3, -1.1, -0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5),
    dtype=np.float64,
)


@dataclass(frozen=True)
class RestrictedUniverse:
    """The 44 fixed-cash-strike contracts used by the paper's audit."""

    expiry_days: np.ndarray
    strikes: np.ndarray
    is_put: np.ndarray
    labels: tuple[str, ...]


@dataclass(frozen=True)
class PopulationDiagnostics:
    """Boundary-scaling evidence that must accompany a valid population."""

    measure: str
    path_count: int
    minimum_gamma: float
    boundary_limited_substeps: int
    frozen_paths: int


def _normalized_calls(
    codebooks: np.ndarray,
    *,
    maturity_days: int,
    relative_strikes: np.ndarray,
    dynamics: DynamicsFit,
) -> np.ndarray:
    """Price one maturity for a path batch in spot-normalized units."""

    if dynamics.num_factors == 0:
        eta = average_density_from_forward(
            tau=dynamics.tau, forward_density=codebooks[:1]
        )[:, maturity_days]
        return density_call_prices(
            tau=maturity_days / DAYS_PER_YEAR,
            log_strikes=np.log(relative_strikes),
            jump_x=dynamics.jump_x,
            average_density=eta[0],
        )
    eta = average_density_from_forward(
        tau=dynamics.tau, forward_density=codebooks
    )[:, maturity_days]
    result = np.empty(
        (codebooks.shape[0], relative_strikes.shape[1]), dtype=np.float64
    )
    tau = maturity_days / DAYS_PER_YEAR
    for path_index in range(codebooks.shape[0]):
        result[path_index] = density_call_prices(
            tau=tau,
            log_strikes=np.log(relative_strikes[path_index]),
            jump_x=dynamics.jump_x,
            average_density=eta[path_index],
        )
    return result


def restricted_universe(
    initial_forward_density: np.ndarray,
    dynamics: DynamicsFit,
    *,
    config: TangentLevyV2AuditConfig,
) -> RestrictedUniverse:
    """Construct eleven OTM puts/calls at each fixed initial expiry."""

    codebook = np.asarray(initial_forward_density, dtype=np.float64)[None]
    expiry_days = np.asarray(config.option_expiry_days, dtype=np.int64)
    atm_calls = np.asarray(
        [
            _normalized_calls(
                codebook,
                maturity_days=int(expiry),
                relative_strikes=np.ones((1, 1), dtype=np.float64),
                dynamics=dynamics,
            )[0, 0]
            for expiry in expiry_days
        ]
    )
    probabilities = np.clip(0.5 * (atm_calls + 1.0), 0.5, 1.0 - 1.0e-14)
    total_volatility = 2.0 * ndtri(probabilities)
    epsilon_volatility = total_volatility[0] / math.sqrt(
        expiry_days[0] / DAYS_PER_YEAR
    )
    nms_scale = np.sqrt(
        (
            epsilon_volatility**2 * float(NMS.EPS)
            + np.square(total_volatility)
        )
        / (1.0 + float(NMS.EPS))
    )
    strikes = np.exp(nms_scale[:, None] * NORMALIZED_MONEYNESS[None])
    # OTM means puts below spot and calls above spot.  The near-ATM negative
    # node is a put and the positive nodes are calls; cash strikes never move.
    is_put = np.broadcast_to(
        NORMALIZED_MONEYNESS[None] < 0.0, strikes.shape
    ).copy()
    labels = tuple(
        f"{expiry}D NMS {score:g} {'put' if score < 0.0 else 'call'}"
        for expiry in expiry_days
        for score in NORMALIZED_MONEYNESS
    )
    return RestrictedUniverse(
        expiry_days=expiry_days,
        strikes=strikes,
        is_put=is_put,
        labels=("spot", *labels),
    )


def _option_values(
    codebooks: np.ndarray,
    spots: np.ndarray,
    universe: RestrictedUniverse,
    *,
    day_index: int,
    dynamics: DynamicsFit,
) -> np.ndarray:
    path_count = len(spots)
    result = np.zeros(
        (path_count, len(universe.expiry_days), universe.strikes.shape[1]),
        dtype=np.float64,
    )
    for expiry_index, expiry_day in enumerate(universe.expiry_days):
        remaining = int(expiry_day) - day_index
        strikes = universe.strikes[expiry_index]
        if remaining < 0:
            continue
        if remaining == 0:
            calls = np.maximum(spots[:, None] - strikes[None], 0.0)
            puts = np.maximum(strikes[None] - spots[:, None], 0.0)
        else:
            relative = strikes[None] / spots[:, None]
            normalized_calls = _normalized_calls(
                codebooks,
                maturity_days=remaining,
                relative_strikes=relative,
                dynamics=dynamics,
            )
            calls = spots[:, None] * normalized_calls
            puts = calls - spots[:, None] + strikes[None]
        result[:, expiry_index] = np.where(
            universe.is_put[expiry_index][None], puts, calls
        )
    verify(
        np.all(np.isfinite(result)) and np.min(result) >= -2.0e-10,
        lambda: (
            "option repricing produced a negative or non-finite value: "
            f"minimum={np.nanmin(result):.6g}"
        ),
    )
    return np.maximum(result, 0.0)


def _active_mask(config: TangentLevyV2AuditConfig) -> np.ndarray:
    mask = np.zeros(
        (config.horizon_days, config.instrument_count), dtype=bool
    )
    mask[:, 0] = True
    for index, expiry in enumerate(config.option_expiry_days):
        start = 1 + index * config.options_per_expiry
        mask[:expiry, start : start + config.options_per_expiry] = True
    return mask


def simulate_restricted_population(
    initial_forward_density: np.ndarray,
    dynamics: DynamicsFit,
    *,
    config: TangentLevyV2AuditConfig,
    measure: str = "Q",
    physical_fit: PhysicalMeasureFit | None = None,
    path_count: int | None = None,
    batch_size: int = 128,
) -> tuple[RestrictedPopulation, RestrictedUniverse, PopulationDiagnostics]:
    """Generate the restricted audit population without discarding failures."""

    verify_inp(measure in {"P", "Q"}, lambda: f"unknown measure {measure!r}")
    verify_inp(
        (measure == "Q" and physical_fit is None)
        or (measure == "P" and physical_fit is not None),
        lambda: "P requires, and Q forbids, a physical market-price-of-risk fit",
    )
    paths = config.path_count if path_count is None else path_count
    verify_inp(
        paths >= 4 and paths % 4 == 0 and batch_size >= 4 and batch_size % 2 == 0,
        lambda: "path count must be divisible by four and batches must preserve pairs",
    )
    verify_inp(paths % batch_size == 0, lambda: "batch size must divide path count")
    universe = restricted_universe(initial_forward_density, dynamics, config=config)
    feature_count = (
        dynamics.num_factors
        + 1
        + config.instrument_count
        - 1
        + 1
        + len(config.option_expiry_days)
    )
    features = np.empty(
        (paths, config.horizon_days, feature_count), dtype=np.float32
    )
    returns = np.empty(
        (paths, config.horizon_days, config.instrument_count), dtype=np.float64
    )
    active_mask = _active_mask(config)
    active_density_mask = (
        np.ones_like(initial_forward_density, dtype=bool)
        if dynamics.active_mask is None
        else np.asarray(dynamics.active_mask, dtype=bool)
    )
    alpha = hjm_drift(
        tau=dynamics.tau,
        jump_x=dynamics.jump_x,
        beta=dynamics.beta,
        active_mask=active_density_mask,
    )
    physical_adjustment = (
        np.zeros_like(initial_forward_density)
        if physical_fit is None
        else np.einsum("f,ftx->tx", physical_fit.theta, dynamics.beta)
    )
    generator = np.random.default_rng(config.seed)
    substep_dt = 1.0 / (DAYS_PER_YEAR * config.steps_per_day)
    sqrt_substep_dt = math.sqrt(substep_dt)
    started = time.perf_counter()
    minimum_gamma = 1.0
    boundary_limited_substeps = 0
    frozen_path_flags = np.zeros(paths, dtype=bool)
    for completed in range(0, paths, batch_size):
        pair_count = batch_size // 2
        codebooks = (
            initial_forward_density[None].copy()
            if dynamics.num_factors == 0
            else np.broadcast_to(
                initial_forward_density,
                (batch_size,) + initial_forward_density.shape,
            ).copy()
        )
        spots = np.ones(batch_size, dtype=np.float64)
        options = _option_values(
            codebooks, spots, universe, day_index=0, dynamics=dynamics
        )
        factor_history = np.zeros(
            (batch_size, dynamics.num_factors), dtype=np.float64
        )
        independent = generator.standard_normal(
            (
                config.horizon_days,
                config.steps_per_day,
                pair_count,
                dynamics.num_factors,
            )
        )
        factor_normals = np.stack((independent, -independent), axis=3).reshape(
            config.horizon_days,
            config.steps_per_day,
            batch_size,
            dynamics.num_factors,
        )
        for day_index in range(config.horizon_days):
            live_expiry = universe.expiry_days > day_index
            normalized_options = options / spots[:, None, None]
            log_options = np.where(
                live_expiry[None, :, None],
                np.log(np.maximum(normalized_options, 1.0e-12)),
                0.0,
            )
            time_to_expiry = np.maximum(
                universe.expiry_days - day_index, 0
            ) / config.horizon_days
            features[completed : completed + batch_size, day_index] = np.concatenate(
                (
                    factor_history,
                    np.log(spots)[:, None],
                    log_options.reshape(batch_size, -1),
                    np.full((batch_size, 1), day_index / config.horizon_days),
                    np.broadcast_to(
                        time_to_expiry, (batch_size, len(time_to_expiry))
                    ),
                ),
                axis=1,
            )
            previous_spots = spots.copy()
            previous_options = options.copy()
            day_start_codebooks = codebooks.copy()
            for substep in range(config.steps_per_day):
                if dynamics.num_factors == 0:
                    midpoint = (substep + 0.5) / config.steps_per_day
                    short_end_density = (
                        (1.0 - midpoint) * day_start_codebooks[:, 0]
                        + midpoint * day_start_codebooks[:, 1]
                    )
                else:
                    short_end_density = codebooks[:, 0]
                spot_step = sample_short_end_spot_step(
                    np.broadcast_to(
                        short_end_density, (batch_size, codebooks.shape[-1])
                    ),
                    jump_x=dynamics.jump_x,
                    dt=substep_dt,
                    rng=generator,
                )
                spots *= spot_step.gross_return
                # Split the transport from the HJM source term.  On the daily
                # uniform maturity grid this linear semi-Lagrangian shift is
                # positivity preserving, unlike an explicit derivative Euler
                # step at sharp reconstructed maturity profiles.
                shift_weight = substep_dt / float(
                    dynamics.tau[1] - dynamics.tau[0]
                )
                if dynamics.num_factors == 0:
                    fraction = (substep + 1) / config.steps_per_day
                    shifted = (
                        (1.0 - fraction) * day_start_codebooks
                        + fraction
                        * np.concatenate(
                            (day_start_codebooks[:, 1:], day_start_codebooks[:, -1:]),
                            axis=1,
                        )
                    )
                else:
                    shifted = (1.0 - shift_weight) * codebooks + shift_weight * np.concatenate(
                        (codebooks[:, 1:], codebooks[:, -1:]), axis=1
                    )
                drift = alpha + physical_adjustment
                drift[..., ~active_density_mask] = 0.0
                shock = (
                    0.0
                    if dynamics.num_factors == 0
                    else sqrt_substep_dt
                    * np.einsum(
                        "pf,ftx->ptx",
                        factor_normals[day_index, substep],
                        dynamics.beta,
                    )
                )
                minimum = np.min(codebooks, axis=(1, 2))
                gamma = np.minimum(1.0, minimum / 1.0e-6)
                minimum_gamma = min(minimum_gamma, float(np.min(gamma)))
                boundary_limited_substeps += int(
                    np.count_nonzero(gamma < 1.0 - 1.0e-12)
                )
                if np.any(gamma == 0.0):
                    frozen_path_flags[completed : completed + batch_size] = True
                candidate = (
                    shifted
                    + np.square(gamma)[:, None, None] * drift * substep_dt
                    + gamma[:, None, None] * shock
                )
                verify(
                    np.all(np.isfinite(candidate)) and np.min(candidate) >= 0.0,
                    lambda: (
                        "a Tangent Levy codebook path failed positivity; the entire "
                        "audit population is invalid"
                    ),
                )
                codebooks = candidate
                factor_history += sqrt_substep_dt * factor_normals[day_index, substep]
            options = _option_values(
                codebooks,
                spots,
                universe,
                day_index=day_index + 1,
                dynamics=dynamics,
            )
            option_returns = (options - previous_options) / previous_spots[:, None, None]
            returns[completed : completed + batch_size, day_index, 0] = (
                spots - previous_spots
            ) / previous_spots
            returns[completed : completed + batch_size, day_index, 1:] = (
                option_returns.reshape(batch_size, -1)
            )
        elapsed = time.perf_counter() - started
        finished = completed + batch_size
        eta = elapsed * (paths - finished) / finished
        print(
            f"tangent-levy-v2 {measure} paths={finished}/{paths} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )
    verify(np.all(np.isfinite(features)) and np.all(np.isfinite(returns)), "non-finite population")
    frozen_paths = int(np.count_nonzero(frozen_path_flags))
    diagnostics = PopulationDiagnostics(
        measure=measure,
        path_count=paths,
        minimum_gamma=minimum_gamma,
        boundary_limited_substeps=boundary_limited_substeps,
        frozen_paths=frozen_paths,
    )
    verify(
        frozen_paths == 0,
        lambda: "one or more paths froze at the positivity boundary; audit invalid",
    )
    pair_ids = np.repeat(np.arange(paths // 2, dtype=np.int64), 2)
    return (
        RestrictedPopulation(features, returns, active_mask, pair_ids),
        universe,
        diagnostics,
    )
