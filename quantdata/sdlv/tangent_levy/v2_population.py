"""Joint codebook/spot simulation for the Tangent Levy v2 audits."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from cdxcore.err import verify, verify_inp
from scipy.special import ndtri

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS

from .constraints import (
    hjm_cell_drift,
    hjm_integrated_characteristic_residual,
)
from .dynamics import DynamicsFit
from .physical_measure import PhysicalMeasureFit
from .pricing import (
    average_density_from_forward,
    density_call_prices,
    lattice_density_call_prices_batched,
)
from .v2_audit import RestrictedPopulation
from .v2_config import TangentLevyV2AuditConfig
from .v2_spot import sample_short_end_spot_step, short_end_characteristics

NORMALIZED_MONEYNESS = np.asarray(
    (-1.5, -1.3, -1.1, -0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5),
    dtype=np.float64,
)
BOUNDARY_STANDARD_DEVIATIONS = 12.0


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
    mean_gamma: float
    boundary_limited_substeps: int
    gamma_below_0_1: int
    gamma_below_0_01: int
    gamma_below_0_0001: int
    frozen_paths: int


@dataclass(frozen=True)
class DeterministicQAudit:
    """Exact martingale identities for the reduced zero-factor Q model."""

    tested_coordinates: int
    maturity_decompositions: int
    maximum_absolute_density_error: float
    maximum_relative_density_error: float
    maximum_spot_moment_error: float
    tolerance: float
    passed: bool


def _predictable_boundary_scale(
    shifted_density: np.ndarray,
    *,
    factor_scale: np.ndarray,
    q_drift: np.ndarray,
    physical_drift: np.ndarray,
    active_mask: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Return a pre-shock scalar time change with positivity headroom.

    The factor loading is multiplied by ``gamma`` and the quadratic HJM drift
    by ``gamma**2``.  The bound is computed before observing the Gaussian
    increment and reserves ``BOUNDARY_STANDARD_DEVIATIONS`` conditional
    standard deviations at every active density coordinate.
    """

    shifted = np.asarray(shifted_density, dtype=np.float64)
    scale = np.asarray(factor_scale, dtype=np.float64)
    alpha = np.asarray(q_drift, dtype=np.float64)
    physical = np.asarray(physical_drift, dtype=np.float64)
    active = np.asarray(active_mask, dtype=bool)
    verify_inp(
        shifted.ndim == 3
        and scale.shape == alpha.shape == physical.shape == active.shape
        and shifted.shape[1:] == scale.shape,
        lambda: "boundary-scale arrays have incompatible density geometry",
    )
    quadratic = dt * np.maximum(-alpha, 0.0)
    linear = (
        BOUNDARY_STANDARD_DEVIATIONS * math.sqrt(dt) * scale
        + dt * np.maximum(-physical, 0.0)
    )
    affected = active | (np.abs(alpha) > 0.0) | (np.abs(physical) > 0.0)
    constrained = affected & ((quadratic > 0.0) | (linear > 0.0))
    if not np.any(constrained):
        return np.ones(shifted.shape[0], dtype=np.float64)
    available = shifted[:, constrained]
    a = quadratic[constrained][None]
    b = linear[constrained][None]
    # Solve a*gamma^2 + b*gamma <= available with the stable positive-root
    # formula.  This avoids the severe over-suppression caused by replacing
    # gamma^2 with gamma in alpha-only cells.
    roots = 2.0 * available / (
        b + np.sqrt(np.square(b) + 4.0 * a * available)
    )
    return np.minimum(1.0, np.min(roots, axis=1))


def analytic_deterministic_q_audit(
    initial_forward_density: np.ndarray,
    dynamics: DynamicsFit,
    *,
    config: TangentLevyV2AuditConfig,
    tolerance: float = 1.0e-12,
) -> DeterministicQAudit:
    """Verify the reduced Q martingale restrictions without Monte Carlo.

    For a deterministic Musiela codebook, today's integrated Levy measure for
    any remaining maturity must equal the one-day spot Levy measure plus
    tomorrow's shifted integrated measure.  Equality of those measures gives
    equality of the complete characteristic exponents, so it establishes the
    one-step martingale identity for every European strike, including the 44
    fixed-cash-strike contracts in the audit.  The spot moment is checked
    separately at every simulated midpoint substep.
    """

    density = np.asarray(initial_forward_density, dtype=np.float64)
    tau = np.asarray(dynamics.tau, dtype=np.float64)
    verify_inp(
        dynamics.num_factors == 0,
        lambda: "the analytic reduced-Q audit requires a zero-factor codebook",
    )
    verify_inp(
        density.shape == (tau.size, dynamics.jump_x.size)
        and np.all(np.isfinite(density))
        and np.all(density >= 0.0),
        lambda: (
            "initial forward density must be finite, non-negative, and match "
            f"the dynamics grid; found {density.shape}"
        ),
    )
    maturity_step = 1.0 / DAYS_PER_YEAR
    verify_inp(
        np.allclose(np.diff(tau), maturity_step, rtol=0.0, atol=1.0e-14),
        lambda: "the deterministic audit requires the daily Musiela grid",
    )
    verify_inp(
        config.horizon_days < tau.size
        and max(config.option_expiry_days) < tau.size,
        lambda: "the audit horizon or option maturities exceed the density grid",
    )
    verify_inp(
        np.isfinite(tolerance) and tolerance > 0.0,
        lambda: f"tolerance must be finite-positive, found {tolerance!r}",
    )

    maximum_absolute = 0.0
    maximum_scale = 1.0
    maximum_spot_moment_error = 0.0
    maturity_decompositions = 0
    substep_dt = maturity_step / config.steps_per_day
    for day_index in range(config.horizon_days):
        current = np.concatenate(
            (
                density[day_index:],
                np.broadcast_to(
                    density[-1], (day_index, density.shape[1])
                ),
            ),
            axis=0,
        )
        tomorrow = np.concatenate((current[1:], current[-1:]), axis=0)
        one_day_density = 0.5 * (current[0] + current[1]) * maturity_step

        # The simulator samples the same trapezoidal one-day measure through
        # midpoint substeps.  Each exponential compensator must give E[R]=1.
        for substep in range(config.steps_per_day):
            midpoint = (substep + 0.5) / config.steps_per_day
            short_end = (1.0 - midpoint) * current[0] + midpoint * current[1]
            intensity, compensator, probabilities = short_end_characteristics(
                short_end,
                jump_x=dynamics.jump_x,
            )
            expected_jump_exponential = probabilities @ np.exp(dynamics.jump_x)
            expected_gross = np.exp(
                -compensator * substep_dt
                + intensity
                * substep_dt
                * (expected_jump_exponential - 1.0)
            )
            maximum_spot_moment_error = max(
                maximum_spot_moment_error,
                abs(float(expected_gross) - 1.0),
            )

        for expiry_day in config.option_expiry_days:
            remaining = int(expiry_day) - day_index
            if remaining <= 0:
                continue
            current_integrated = maturity_step * (
                0.5 * current[0]
                + np.sum(current[1:remaining], axis=0)
                + 0.5 * current[remaining]
            )
            if remaining == 1:
                tomorrow_integrated = np.zeros_like(one_day_density)
            else:
                tomorrow_integrated = maturity_step * (
                    0.5 * tomorrow[0]
                    + np.sum(tomorrow[1 : remaining - 1], axis=0)
                    + 0.5 * tomorrow[remaining - 1]
                )
            error = current_integrated - (one_day_density + tomorrow_integrated)
            maximum_absolute = max(maximum_absolute, float(np.max(np.abs(error))))
            maximum_scale = max(
                maximum_scale,
                float(np.max(np.abs(current_integrated))),
            )
            maturity_decompositions += 1

    maximum_relative = maximum_absolute / maximum_scale
    active_option_days = sum(
        min(config.horizon_days, expiry) for expiry in config.option_expiry_days
    )
    tested_coordinates = (
        config.horizon_days + config.options_per_expiry * active_option_days
    )
    passed = (
        maximum_relative <= tolerance
        and maximum_spot_moment_error <= tolerance
        and maturity_decompositions
        == active_option_days
    )
    return DeterministicQAudit(
        tested_coordinates=tested_coordinates,
        maturity_decompositions=maturity_decompositions,
        maximum_absolute_density_error=maximum_absolute,
        maximum_relative_density_error=maximum_relative,
        maximum_spot_moment_error=maximum_spot_moment_error,
        tolerance=tolerance,
        passed=passed,
    )


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
    tau = maturity_days / DAYS_PER_YEAR
    return lattice_density_call_prices_batched(
        tau=tau,
        log_strikes=np.log(relative_strikes),
        jump_x=dynamics.jump_x,
        average_density=eta,
    )


def _cell_integrals(values: np.ndarray, *, steps_per_day: int) -> np.ndarray:
    """Integrate a piecewise-linear daily maturity curve on substep cells."""

    values = np.asarray(values, dtype=np.float64)
    verify_inp(
        values.ndim >= 2 and values.shape[-2] >= 2 and steps_per_day > 0,
        lambda: "cell integration requires a maturity axis and positive steps",
    )
    dt = 1.0 / (DAYS_PER_YEAR * steps_per_day)
    cells = []
    for substep in range(steps_per_day):
        midpoint = (substep + 0.5) / steps_per_day
        cells.append(
            dt
            * (
                (1.0 - midpoint) * values[..., :-1, :]
                + midpoint * values[..., 1:, :]
            )
        )
    stacked = np.stack(cells, axis=-2)
    cell_count = (values.shape[-2] - 1) * steps_per_day
    shape = (*values.shape[:-2], cell_count, values.shape[-1])
    return stacked.reshape(shape)


def _normalized_calls_from_cells(
    cell_measures: np.ndarray,
    *,
    maturity_cells: int,
    relative_strikes: np.ndarray,
    jump_x: np.ndarray,
    substep_dt: float,
) -> np.ndarray:
    """Price normalized calls from integrated maturity-cell Levy measures."""

    integrated = np.sum(cell_measures[..., :maturity_cells, :], axis=-2)
    average_density = integrated / (maturity_cells * substep_dt)
    relative_strikes = np.asarray(relative_strikes, dtype=np.float64)
    if average_density.shape[0] == 1 and relative_strikes.shape[0] > 1:
        average_density = np.broadcast_to(
            average_density,
            (relative_strikes.shape[0], average_density.shape[1]),
        )
    return lattice_density_call_prices_batched(
        tau=maturity_cells * substep_dt,
        log_strikes=np.log(relative_strikes),
        jump_x=jump_x,
        average_density=average_density,
    )


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


def _restricted_universe_from_cells(
    initial_cells: np.ndarray,
    dynamics: DynamicsFit,
    *,
    config: TangentLevyV2AuditConfig,
    substep_dt: float,
) -> RestrictedUniverse:
    expiry_days = np.asarray(config.option_expiry_days, dtype=np.int64)
    atm_calls = np.asarray(
        [
            _normalized_calls_from_cells(
                initial_cells[None],
                maturity_cells=int(expiry) * config.steps_per_day,
                relative_strikes=np.ones((1, 1), dtype=np.float64),
                jump_x=dynamics.jump_x,
                substep_dt=substep_dt,
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
    is_put = np.broadcast_to(
        NORMALIZED_MONEYNESS[None] < 0.0,
        strikes.shape,
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


def _option_values_from_cells(
    cell_measures: np.ndarray,
    spots: np.ndarray,
    universe: RestrictedUniverse,
    *,
    day_index: int,
    dynamics: DynamicsFit,
    config: TangentLevyV2AuditConfig,
    substep_dt: float,
) -> np.ndarray:
    result = np.zeros(
        (len(spots), len(universe.expiry_days), universe.strikes.shape[1]),
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
            normalized_calls = _normalized_calls_from_cells(
                cell_measures,
                maturity_cells=remaining * config.steps_per_day,
                relative_strikes=relative,
                jump_x=dynamics.jump_x,
                substep_dt=substep_dt,
            )
            calls = spots[:, None] * normalized_calls
            puts = calls - spots[:, None] + strikes[None]
        result[:, expiry_index] = np.where(
            universe.is_put[expiry_index][None],
            puts,
            calls,
        )
    verify(
        np.all(np.isfinite(result)) and np.min(result) >= -2.0e-10,
        lambda: "cell-grid option repricing produced invalid values",
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
    substep_dt = 1.0 / (DAYS_PER_YEAR * config.steps_per_day)
    expected_tau = np.arange(dynamics.tau.size, dtype=np.float64) / DAYS_PER_YEAR
    verify_inp(
        np.allclose(dynamics.tau, expected_tau, rtol=0.0, atol=1.0e-14),
        lambda: "the maturity-cell audit requires daily maturities j/252",
    )
    initial_cells = _cell_integrals(
        initial_forward_density,
        steps_per_day=config.steps_per_day,
    )
    required_cells = max(config.option_expiry_days) * config.steps_per_day
    verify_inp(
        required_cells <= initial_cells.shape[-2],
        lambda: (
            "the longest option expiry exceeds the maturity-cell horizon: "
            f"requires {required_cells} cells, found {initial_cells.shape[-2]}"
        ),
    )
    # The restricted audit prices no claim beyond its longest fixed expiry.
    # Retaining unused maturity cells would let an irrelevant far-boundary
    # spline coefficient govern the global positivity time change.
    initial_cells = initial_cells[..., :required_cells, :]
    beta_cells = _cell_integrals(
        dynamics.beta,
        steps_per_day=config.steps_per_day,
    )[..., :required_cells, :]
    zero_jump = int(np.argmin(np.abs(dynamics.jump_x)))
    verify_inp(
        dynamics.jump_x[zero_jump] == 0.0,
        lambda: "the symmetric jump lattice must contain the zero-jump node",
    )
    # A point mass at jump size zero is a gauge coordinate: it is annihilated
    # by both the compensated Levy generator and every option payoff.  Keeping
    # its arbitrary fitted intensity in the positivity screen can freeze all
    # economically active factors, so remove it from the numerical measure.
    initial_cells = initial_cells.copy()
    beta_cells = beta_cells.copy()
    initial_cells[..., zero_jump] = 0.0
    beta_cells[..., zero_jump] = 0.0
    alpha_cells = hjm_cell_drift(beta_cells, jump_x=dynamics.jump_x)
    alpha_cells[..., zero_jump] = 0.0
    integrated_residual = hjm_integrated_characteristic_residual(
        beta_cells=beta_cells,
        alpha_cells=alpha_cells,
        jump_x=dynamics.jump_x,
        frequencies=np.linspace(-40.0, 40.0, 161),
    )
    verify(
        np.max(np.abs(integrated_residual), initial=0.0) <= 1.0e-10,
        lambda: (
            "the maturity-cell factors do not satisfy the integrated HJM "
            f"identity: maximum residual={np.max(np.abs(integrated_residual)):.6g}"
        ),
    )
    universe = _restricted_universe_from_cells(
        initial_cells,
        dynamics,
        config=config,
        substep_dt=substep_dt,
    )
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
    physical_adjustment = (
        np.zeros_like(initial_cells)
        if physical_fit is None
        else np.einsum("f,fcx->cx", physical_fit.theta, beta_cells)
    )
    factor_scale = np.sqrt(np.sum(np.square(beta_cells), axis=0))
    affected_cells = (factor_scale > 0.0) | (np.abs(alpha_cells) > 0.0)
    stochastic_dynamics = bool(np.any(dynamics.beta != 0.0))
    generator = np.random.default_rng(config.seed)
    sqrt_substep_dt = math.sqrt(substep_dt)
    started = time.perf_counter()
    minimum_gamma = 1.0
    sum_gamma = 0.0
    gamma_observations = 0
    boundary_limited_substeps = 0
    gamma_below_0_1 = 0
    gamma_below_0_01 = 0
    gamma_below_0_0001 = 0
    frozen_path_flags = np.zeros(paths, dtype=bool)
    for completed in range(0, paths, batch_size):
        pair_count = batch_size // 2
        codebooks = (
            initial_cells[None].copy()
            if dynamics.num_factors == 0
            else np.broadcast_to(
                initial_cells,
                (batch_size,) + initial_cells.shape,
            ).copy()
        )
        spots = np.ones(batch_size, dtype=np.float64)
        options = _option_values_from_cells(
            codebooks,
            spots,
            universe,
            day_index=0,
            dynamics=dynamics,
            config=config,
            substep_dt=substep_dt,
        )
        factor_history = np.zeros(
            (batch_size, dynamics.num_factors), dtype=np.float64
        )
        independent = (
            generator.standard_normal(
                (
                    config.horizon_days,
                    config.steps_per_day,
                    pair_count,
                    dynamics.num_factors,
                )
            )
            if stochastic_dynamics
            else np.zeros(
                (
                    config.horizon_days,
                    config.steps_per_day,
                    pair_count,
                    dynamics.num_factors,
                ),
                dtype=np.float64,
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
            for substep in range(config.steps_per_day):
                short_end_density = codebooks[:, 0] / substep_dt
                spot_step = sample_short_end_spot_step(
                    np.broadcast_to(
                        short_end_density, (batch_size, codebooks.shape[-1])
                    ),
                    jump_x=dynamics.jump_x,
                    dt=substep_dt,
                    rng=generator,
                )
                spots *= spot_step.gross_return
                shifted = np.concatenate(
                    (codebooks[:, 1:], codebooks[:, -1:]),
                    axis=1,
                )
                shock = (
                    0.0
                    if not stochastic_dynamics
                    else sqrt_substep_dt
                    * np.einsum(
                        "pf,fcx->pcx",
                        factor_normals[day_index, substep],
                        beta_cells,
                    )
                )
                gamma = _predictable_boundary_scale(
                    shifted,
                    factor_scale=factor_scale,
                    q_drift=alpha_cells,
                    physical_drift=physical_adjustment,
                    active_mask=affected_cells,
                    dt=substep_dt,
                )
                minimum_gamma = min(minimum_gamma, float(np.min(gamma)))
                sum_gamma += float(np.sum(gamma))
                gamma_observations += len(gamma)
                boundary_limited_substeps += int(
                    np.count_nonzero(gamma < 1.0 - 1.0e-12)
                )
                gamma_below_0_1 += int(np.count_nonzero(gamma < 0.1))
                gamma_below_0_01 += int(np.count_nonzero(gamma < 0.01))
                gamma_below_0_0001 += int(np.count_nonzero(gamma < 0.0001))
                frozen_path_flags[completed : completed + batch_size] |= gamma == 0.0
                candidate = (
                    shifted
                    + np.square(gamma)[:, None, None] * alpha_cells * substep_dt
                    + gamma[:, None, None] * physical_adjustment * substep_dt
                    + gamma[:, None, None] * shock
                )
                verify(
                    np.all(np.isfinite(candidate)) and np.min(candidate) >= 0.0,
                    lambda: (
                        "a Tangent Levy codebook path failed positivity; the entire "
                        "audit population is invalid: "
                        f"day={day_index + 1}, substep={substep + 1}, "
                        f"minimum={np.nanmin(candidate):.6g}"
                    ),
                )
                codebooks = candidate
                factor_history += (
                    gamma[:, None]
                    * sqrt_substep_dt
                    * factor_normals[day_index, substep]
                )
            options = _option_values_from_cells(
                codebooks,
                spots,
                universe,
                day_index=day_index + 1,
                dynamics=dynamics,
                config=config,
                substep_dt=substep_dt,
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
        mean_gamma=sum_gamma / gamma_observations,
        boundary_limited_substeps=boundary_limited_substeps,
        gamma_below_0_1=gamma_below_0_1,
        gamma_below_0_01=gamma_below_0_01,
        gamma_below_0_0001=gamma_below_0_0001,
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
