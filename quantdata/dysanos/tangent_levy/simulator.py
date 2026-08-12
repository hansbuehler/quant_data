"""Seeded risk-neutral simulation of Tangent Lévy code-books."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cdxcore.err import verify, verify_inp

from .config import PricingConfig, SimulationConfig
from .constraints import hjm_drift, musiela_drift
from .dynamics import DynamicsFit
from .pricing import average_density_from_forward, density_call_prices


@dataclass(frozen=True)
class SimulationResult:
    """Recorded code-book states and explicit path-failure diagnostics."""

    record_steps: np.ndarray
    forward_density: np.ndarray
    alive: np.ndarray
    failure_step: np.ndarray
    minimum_gamma: np.ndarray
    boundary_limited_steps: np.ndarray
    seed: int

    @property
    def failed_path_fraction(self) -> float:
        """Fraction of paths rejected for a nonpositive or nonfinite state."""

        return float(np.mean(~self.alive))

    @property
    def frozen_path_fraction(self) -> float:
        """Fraction of paths for which the paper's global scale reached zero."""

        return float(np.mean(self.minimum_gamma == 0.0))


@dataclass(frozen=True)
class PricedSimulation:
    """Normalized calls derived from one recorded code-book ensemble."""

    tau: np.ndarray
    log_moneyness: np.ndarray
    calls: np.ndarray
    valid_paths: np.ndarray


def simulate_codebooks(
    initial_forward_density: np.ndarray,
    dynamics: DynamicsFit,
    *,
    config: SimulationConfig = SimulationConfig(),
    factor_risk_premium: np.ndarray | None = None,
) -> SimulationResult:
    """Advance the Musiela code-book with the paper's scaled Euler scheme.

    A candidate path that becomes negative or nonfinite is rejected and remains
    ``NaN``. No density clipping or shock resampling is performed. Passing a
    factor risk premium changes the drift to a separately estimated physical
    measure; omitting it is the risk-neutral implementation.
    """

    initial = np.asarray(initial_forward_density, dtype=np.float64)
    beta = np.asarray(dynamics.beta, dtype=np.float64)
    tau = np.asarray(dynamics.tau, dtype=np.float64)
    jump_x = np.asarray(dynamics.jump_x, dtype=np.float64)
    verify_inp(
        initial.shape == (tau.size, jump_x.size)
        and np.all(np.isfinite(initial))
        and np.all(initial >= 0.0),
        lambda: (
            "initial density must be finite and nonnegative with shape "
            f"{(tau.size, jump_x.size)}, got {initial.shape}"
        ),
    )
    verify_inp(
        abs(dynamics.dt * 252.0 - 1.0) < 1.0e-12,
        lambda: f"simulator currently requires one-session dt=1/252, got {dynamics.dt}",
    )
    physical_adjustment = np.zeros_like(initial)
    if factor_risk_premium is not None:
        theta = np.asarray(factor_risk_premium, dtype=np.float64)
        verify_inp(
            theta.shape == (dynamics.num_factors,) and np.all(np.isfinite(theta)),
            lambda: (
                "factor risk premium must be finite with shape "
                f"{(dynamics.num_factors,)}, got {theta.shape}"
            ),
        )
        physical_adjustment = np.einsum("f,ftx->tx", theta, beta)
    active_mask = (
        np.ones(initial.shape, dtype=bool)
        if dynamics.active_mask is None
        else np.asarray(dynamics.active_mask, dtype=bool)
    )
    verify_inp(
        active_mask.shape == initial.shape,
        lambda: f"dynamics active mask has invalid shape {active_mask.shape}",
    )
    alpha = hjm_drift(
        tau=tau,
        jump_x=jump_x,
        beta=beta,
        active_mask=active_mask,
    )
    rng = np.random.default_rng(config.seed)
    current = np.broadcast_to(initial, (config.num_paths,) + initial.shape).copy()
    alive = np.ones(config.num_paths, dtype=bool)
    failure_step = np.full(config.num_paths, -1, dtype=np.int32)
    minimum_gamma = np.ones(config.num_paths, dtype=np.float64)
    boundary_limited_steps = np.zeros(config.num_paths, dtype=np.int32)
    records: list[np.ndarray] = []
    record_set = set(config.record_steps)
    sqrt_dt = np.sqrt(dynamics.dt)

    for step in range(1, config.num_steps + 1):
        active_index = np.flatnonzero(alive)
        if active_index.size > 0:
            active = current[active_index]
            state_minimum = np.min(active, axis=(1, 2))
            gamma = np.minimum(
                1.0,
                state_minimum / config.boundary_epsilon,
            )
            minimum_gamma[active_index] = np.minimum(
                minimum_gamma[active_index],
                gamma,
            )
            boundary_limited_steps[active_index] += gamma < 1.0
            drift = musiela_drift(
                tau=tau,
                forward_density=active,
                hjm_alpha=alpha,
            )
            drift += physical_adjustment
            drift[..., ~active_mask] = 0.0
            normal = rng.standard_normal((active_index.size, dynamics.num_factors))
            shock = sqrt_dt * np.einsum("pf,ftx->ptx", normal, beta)
            candidate = (
                active
                + np.square(gamma)[:, None, None] * drift * dynamics.dt
                + gamma[:, None, None] * shock
            )
            valid = np.all(np.isfinite(candidate), axis=(1, 2)) & (
                np.min(candidate, axis=(1, 2)) >= 0.0
            )
            current[active_index[valid]] = candidate[valid]
            failed_index = active_index[~valid]
            if failed_index.size > 0:
                alive[failed_index] = False
                failure_step[failed_index] = step
                current[failed_index] = np.nan
        if step in record_set:
            records.append(current.copy())

    verify(
        len(records) == len(config.record_steps),
        lambda: f"recorded {len(records)} states for {config.record_steps}",
    )
    return SimulationResult(
        record_steps=np.asarray(config.record_steps, dtype=np.int32),
        forward_density=np.stack(records, axis=0),
        alive=alive,
        failure_step=failure_step,
        minimum_gamma=minimum_gamma,
        boundary_limited_steps=boundary_limited_steps,
        seed=config.seed,
    )


def price_simulated_state(
    forward_density: np.ndarray,
    *,
    tau: np.ndarray,
    jump_x: np.ndarray,
    log_moneyness: np.ndarray,
    pricing_config: PricingConfig = PricingConfig(),
) -> PricedSimulation:
    """Price one ``(P,T,X)`` ensemble with the generic equation (2.36)."""

    kappa = np.asarray(forward_density, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    jump_x = np.asarray(jump_x, dtype=np.float64)
    scoring_x = np.asarray(log_moneyness, dtype=np.float64)
    verify_inp(
        kappa.ndim == 3 and kappa.shape[1:] == (tau.size, jump_x.size),
        lambda: (
            "forward-density ensemble must have shape (P,T,X), got "
            f"{kappa.shape}"
        ),
    )
    finite = np.all(np.isfinite(kappa), axis=(1, 2))
    finite_minimum = np.min(np.where(np.isfinite(kappa), kappa, -np.inf), axis=(1, 2))
    valid = finite & (finite_minimum >= 0.0)
    eta = average_density_from_forward(
        tau=tau,
        forward_density=kappa[valid],
    )
    positive_tau = tau[1:]
    calls = np.full(
        (kappa.shape[0], positive_tau.size, scoring_x.size),
        np.nan,
        dtype=np.float64,
    )
    for path_index, original_index in enumerate(np.flatnonzero(valid)):
        for maturity_index, maturity in enumerate(positive_tau, start=1):
            calls[original_index, maturity_index - 1] = density_call_prices(
                tau=float(maturity),
                log_strikes=scoring_x,
                jump_x=jump_x,
                average_density=eta[path_index, maturity_index],
                config=pricing_config,
            )
    return PricedSimulation(
        tau=positive_tau,
        log_moneyness=scoring_x,
        calls=calls,
        valid_paths=valid,
    )
