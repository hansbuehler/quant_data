r"""Section 2: the discrete-time pricing identity, formalised and tested.

Market and identity (as specified)
-----------------------------------
Zero carry, numeraire = spot. Normalised decoder ``c(h; tau, k)`` prices a
unit-notional call at moneyness ``k = K/S`` and remaining maturity ``tau``,
given surface state ``h``. For a fixed cash contract ``(T,K)`` followed
through time,

    H(t, S_t, h_t; T, K) = S_t * c(h_t; T-t, K/S_t).

FLEX already normalises exactly this way: ``_option_values`` divides the
strike by spot before calling the decoder and multiplies the decoded price
back up by spot. The discrete-time martingale condition to impose and test
is

    E_Q[ H(t+Delta, Z_next; T, K) | Z_t ] = H(t, Z_t; T, K),

with the correct settlement payoff once expiry is crossed. Dividing both
sides by S_t and writing R = S_{t+Delta}/S_t (gross return) gives, in
normalised units,

    E_Q[ R * c(h_next; T-t-Delta, K/(S_t*R)) ] = c(h_t; T-t, K/S_t).

This is *exactly* what ``WeeklySanosSimulator.hjm_roll_residuals`` already
computes pathwise (verified here by direct derivation, not by trusting its
docstring): for a non-front model expiry slab m>=2, the tower rolls one
week, so the (t,K) contract at model-expiry index m becomes the
(t+Delta,K) contract at index m-1 next week, giving
``R * c_next[m-1](k/R) - c_current[m](k)``; for the front expiry (m=1) the
correct terminal payoff ``relu(R-k)`` replaces ``c_next[0]``.

A continuous-time drift formula would not certify this discrete weekly
step even if one existed for FLEX's decoder; FLEX is a discrete-time
(weekly) simulator by construction, so the object actually tested below
*is* the operative pricing identity, not a discretisation of a separate
continuous-time one.

Scope of the guarantee tested
------------------------------
Only a small, explicitly declared set of (expiry-index, strike) pairs is
tested below -- a declared finite protected book, per Section 2(a) of the
assignment. No claim is made, or should be inferred, about the entire
smooth surface (2(b)): the decoder produces a continuous strike curve, but
martingality has only been tested at the strikes listed in
``TEST_LOG_MONEYNESS`` below.

Data honesty
------------
The simulator built here uses the SAME small, explicitly synthetic
construction the existing test suite uses for its own wiring checks
(``tests/test_flex_weekly_phase3.py::_simulator``/``_history``) -- a
random-but-structurally-valid decoder and a synthetic AR(1) history. This
environment's local ``QUANTDATA_TEST_MODE=1`` fixture data covers only a
handful of scattered single trading days (not a continuous multi-year
weekly series), so a genuine historically-fitted P law cannot be built
here; that requires the production ``QUANTDATA_ROOT``/WRDS-backed
history. Every number below is therefore a *structural/machinery*
validation, not a claim about real SPX dynamics -- stated explicitly at
every place a number is reported, not left implicit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch

from quantdata.dysanos.flex.config import (
    ForwardMuFlexGeometryConfig,
    TriangularPcaDecoderConfig,
)
from quantdata.dysanos.flex.model import LatentStateSanosModel
from quantdata.dysanos.flex.weekly_dynamics import (
    TorchWeeklyDynamics,
    WeeklyDynamicsConfig,
    fit_weekly_hierarchical_dynamics,
)
from quantdata.dysanos.flex.weekly_measure import (
    NormalWeeklyMeasureKernel,
    WeeklyMeasureConfig,
)
from quantdata.dysanos.flex.weekly_simulation import (
    WeeklySanosSimulator,
    hjm_roll_moment_loss,
)
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig

STATE_SIZES: tuple[int, ...] = (3, 3, 2)
TEST_STRIKES = torch.tensor([0.85, 0.95, 1.0, 1.05, 1.15], dtype=torch.float64)


def synthetic_history(
    sizes: tuple[int, ...] = STATE_SIZES, *, observations: int = 400, seed: int = 4471,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic-but-structurally-valid daily history, same generator shape
    as the existing test suite's own ``_history`` fixture (own-lag + prior-
    expiry-lag hierarchical AR(1) plus a common factor), NOT real market
    data."""
    rng = np.random.default_rng(seed)
    blocks = [np.zeros((observations, width)) for width in sizes]
    log_spot = np.zeros(observations)
    for t in range(1, observations):
        common = rng.normal()
        for i, block in enumerate(blocks):
            innovation = 0.02 * rng.normal(size=block.shape[1])
            innovation[0] += 0.03 * common
            own = 0.75 + 0.03 * i
            value = own * block[t - 1] + innovation
            if i > 0:
                lag_driver = np.resize(blocks[i - 1][t - 1], block.shape[1])
                value += 0.10 * lag_driver
            block[t] = value
        log_spot[t] = log_spot[t - 1] + 0.0006 + 0.010 * common + 0.004 * rng.normal()
    return np.concatenate(blocks, axis=1), log_spot


def build_simulator(
    env: BaseEnvironment, sizes: tuple[int, ...] = STATE_SIZES,
) -> WeeklySanosSimulator:
    states, log_spot = synthetic_history(sizes)
    dyn_config = WeeklyDynamicsConfig(
        label="crc-ext-section2", state_sizes_by_expiry=sizes,
        weekly_session_stride=5, pca_mode="full", ridge_penalty=1.0e-5,
    )
    fit = fit_weekly_hierarchical_dynamics(states, log_spot, config=dyn_config)
    dynamics = TorchWeeklyDynamics(env, fit=fit)
    geometry = ForwardMuFlexGeometryConfig(num_inner_strikes=25, num_model_expiries=len(sizes))
    calibrated = LatentStateSanosModel(
        env, num_days=2,
        config=TriangularPcaDecoderConfig(
            label="crc-ext-section2-decoder", state_sizes_by_expiry=sizes,
            state_size=sum(sizes), geometry=geometry,
        ),
    )
    return WeeklySanosSimulator(env, dynamics=dynamics, calibrated_model=calibrated)


@dataclass(frozen=True)
class MartingaleTestResult:
    label: str
    num_paths: int
    mean_abs_residual_by_expiry: list[float]
    std_error_by_expiry: list[float]
    max_abs_t_stat: float
    reliable_expiries: list[int]

    def to_dict(self) -> dict:
        return dict(
            label=self.label, num_paths=self.num_paths,
            mean_abs_residual_by_expiry=self.mean_abs_residual_by_expiry,
            std_error_by_expiry=self.std_error_by_expiry,
            max_abs_t_stat=self.max_abs_t_stat,
            reliable_expiries=self.reliable_expiries,
        )


def martingale_test(
    simulator: WeeklySanosSimulator, *, label: str, num_paths: int = 20_000,
    measure_kernel: NormalWeeklyMeasureKernel | None = None, seed: int = 17,
) -> MartingaleTestResult:
    """The actual E_Q[H(t+Delta,Z_next;T,K)|Z_t] = H(t,Z_t;T,K) test: one
    weekly step from a representative starting state, pathwise residuals via
    hjm_roll_residuals, aggregated into a genuine Monte Carlo conditional-
    mean estimate with its own standard error (not merely a single-path
    check)."""
    torch.manual_seed(seed)
    initial_state = torch.zeros(simulator.dynamics.reduced_state_size, dtype=torch.float64)
    initial_spot = torch.tensor(1.0, dtype=torch.float64)
    states, log_spots = simulator._simulate_internal(
        initial_state, initial_spot, num_paths=num_paths, internal_steps=1,
        spot_mode="implied", standard_normals=None, generator=None,
        measure_kernel=measure_kernel,
    )
    current_state, next_state = states[:, 0], states[:, 1]
    gross = torch.exp(log_spots[:, 1] - log_spots[:, 0])
    residuals = simulator.hjm_roll_residuals(
        current_state, next_state, gross, TEST_STRIKES,
    )  # (paths, expiries, strikes)
    means_by_expiry, ses_by_expiry, t_stats = [], [], []
    for m in range(residuals.shape[1]):
        block = residuals[:, m, :].detach().numpy()  # (paths, strikes)
        mean_abs = float(np.mean(np.abs(block.mean(axis=0))))
        se = block.std(axis=0, ddof=1) / np.sqrt(block.shape[0])
        t_stat = block.mean(axis=0) / np.clip(se, 1.0e-12, None)
        means_by_expiry.append(mean_abs)
        ses_by_expiry.append(float(np.mean(se)))
        t_stats.append(float(np.max(np.abs(t_stat))))
    return MartingaleTestResult(
        label=label, num_paths=num_paths,
        mean_abs_residual_by_expiry=means_by_expiry, std_error_by_expiry=ses_by_expiry,
        max_abs_t_stat=max(t_stats),
        reliable_expiries=list(range(residuals.shape[1])),
    )


def train_target_measure_kernel(
    simulator: WeeklySanosSimulator, *, epochs: int = 400, paths_per_epoch: int = 512,
    seed: int = 314,
) -> NormalWeeklyMeasureKernel:
    """Fits a NormalWeeklyMeasureKernel by minimising the *instrumented*
    conditional HJM roll moment (not just the unconditional mean): the
    instrument set is [1, current_state components], directly operationalising
    'impose and test E_Q[...|Z_current]', not merely an unconditional
    average across all starting states pooled together."""
    torch.manual_seed(seed)
    dynamics = simulator.dynamics
    kernel = NormalWeeklyMeasureKernel(
        simulator.env, dynamics=dynamics,
        config=WeeklyMeasureConfig(
            label="crc-ext-target-kernel", state_sizes_by_expiry=dynamics.reduced_sizes_by_expiry,
            hierarchical=True,
            context_network=FeedForwardConfig(activation="SiLU", num_depth=2, num_width=32),
        ),
    )
    optimizer = torch.optim.Adam(kernel.parameters(), lr=5.0e-4)
    state_dim = dynamics.reduced_state_size
    for epoch in range(epochs):
        current_state = 0.3 * torch.randn((paths_per_epoch, state_dim), dtype=torch.float64)
        initial_spot = torch.ones(paths_per_epoch, dtype=torch.float64)
        states, log_spots = simulator._simulate_internal(
            current_state, initial_spot, num_paths=paths_per_epoch, internal_steps=1,
            spot_mode="implied", standard_normals=None, generator=None,
            measure_kernel=kernel,
        )
        next_state = states[:, 1]
        gross = torch.exp(log_spots[:, 1] - log_spots[:, 0])
        residuals = simulator.hjm_roll_residuals(current_state, next_state, gross, TEST_STRIKES)
        instruments = torch.cat(
            [torch.ones((paths_per_epoch, 1), dtype=torch.float64), current_state], dim=-1,
        )
        loss = hjm_roll_moment_loss(residuals, instruments)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return kernel


def run() -> dict:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    simulator = build_simulator(env)

    p_result = martingale_test(simulator, label="raw_P_AR1_negative_control")
    kernel = train_target_measure_kernel(simulator)
    q_result = martingale_test(simulator, label="trained_target_measure_kernel", measure_kernel=kernel)

    return dict(
        scope_note=(
            "Synthetic structural-validation data (same generator shape as the existing "
            "test suite's own _history/_simulator fixtures), NOT real market dynamics. "
            f"Finite protected book tested: strikes={TEST_STRIKES.tolist()}, "
            f"{len(STATE_SIZES)} model expiries."
        ),
        p_law=p_result.to_dict(),
        trained_q_candidate=q_result.to_dict(),
    )


if __name__ == "__main__":
    result = run()
    with open("section2_pricing_identity_report.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


# ===========================================================================
# Section 3: drift feasibility equations -- rank, conditioning, residual.
#
# Linearised, discrete-time analogue of the continuous-time tangent-space
# feasibility test used throughout this research programme (cf. the
# heston_bates_oracle Task 5/8 compensator-range analysis): does an
# additive mean-shift mu (dimension = reduced_state_size) to the P-law's
# next-period state exist that cancels the current mean roll residual
# across ALL tested (expiry,strike) pairs SIMULTANEOUSLY, to first order?
# This is the finite-market analogue of solving D_mu H . mu = -R for a
# drift adjustment mu, exactly Section 3's "derive the drift feasibility
# equations... report their rank, conditioning and residual."
# ===========================================================================


@dataclass(frozen=True)
class FeasibilityReport:
    residual_dim: int
    mu_dim: int
    jacobian_rank: int
    jacobian_condition_number: float
    r0_norm: float
    residual_norm_after_linearised_correction: float
    reduction_fraction: float
    mu_star: list[float]

    def to_dict(self) -> dict:
        return dict(
            residual_dim=self.residual_dim, mu_dim=self.mu_dim,
            jacobian_rank=self.jacobian_rank, jacobian_condition_number=self.jacobian_condition_number,
            r0_norm=self.r0_norm,
            residual_norm_after_linearised_correction=self.residual_norm_after_linearised_correction,
            reduction_fraction=self.reduction_fraction, mu_star=self.mu_star,
        )


def drift_feasibility_analysis(
    simulator: WeeklySanosSimulator, *, num_paths: int = 60_000, seed: int = 71,
) -> FeasibilityReport:
    torch.manual_seed(seed)
    state_dim = simulator.dynamics.reduced_state_size
    initial_state = torch.zeros(state_dim, dtype=torch.float64)
    initial_spot = torch.tensor(1.0, dtype=torch.float64)
    states, log_spots = simulator._simulate_internal(
        initial_state, initial_spot, num_paths=num_paths, internal_steps=1,
        spot_mode="implied", standard_normals=None, generator=None, measure_kernel=None,
    )
    current_state = states[:, 0]
    next_state_p = states[:, 1].detach()
    gross = torch.exp(log_spots[:, 1] - log_spots[:, 0]).detach()

    mu = torch.zeros(state_dim, dtype=torch.float64, requires_grad=True)

    def mean_residual(mu_: torch.Tensor) -> torch.Tensor:
        shifted_next = next_state_p + mu_.unsqueeze(0)
        residuals = simulator.hjm_roll_residuals(current_state, shifted_next, gross, TEST_STRIKES)
        return residuals.mean(dim=0).reshape(-1)  # (expiries*strikes,)

    r0 = mean_residual(mu).detach()
    jac_rows = []
    for i in range(r0.numel()):
        mu.grad = None
        r = mean_residual(mu)
        r[i].backward(retain_graph=True)
        jac_rows.append(mu.grad.detach().clone())
    J = torch.stack(jac_rows, dim=0).numpy()  # (residual_dim, mu_dim)
    r0_np = r0.numpy()

    rank = int(np.linalg.matrix_rank(J))
    singular_values = np.linalg.svd(J, compute_uv=False)
    condition_number = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 1.0e-14 else float("inf")

    mu_star, *_ = np.linalg.lstsq(J, -r0_np, rcond=None)
    residual_after = J @ mu_star + r0_np
    reduction = float(1.0 - np.linalg.norm(residual_after) / np.linalg.norm(r0_np))

    return FeasibilityReport(
        residual_dim=r0_np.size, mu_dim=state_dim, jacobian_rank=rank,
        jacobian_condition_number=condition_number, r0_norm=float(np.linalg.norm(r0_np)),
        residual_norm_after_linearised_correction=float(np.linalg.norm(residual_after)),
        reduction_fraction=reduction, mu_star=mu_star.tolist(),
    )
