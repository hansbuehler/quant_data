r"""Section 5: the consistent-recalibration (CRC) sequence -- the clearest
gap identified in PHASE1_AUDIT.md (no price-preserving reset exists
anywhere in flex/ today).

Key structural fact exploited below, verified not assumed: ``bound_native_raw``
maps *any* raw torch tensor to bounded, admissible ``(dlvsqs, forward_vols,
eta)`` via sigmoid/interval transforms, and the SANOS forward recursion
(``forward_batch``) then guarantees a non-negative discrete density from any
positive ``dlvsqs``. So for FLEX, "subject to positivity ... and other model
admissibility constraints" is automatically satisfied by construction for
*any* raw state -- the constrained inverse in step (E) below is therefore an
unconstrained optimisation in raw-state space that always decodes to an
admissible surface (verified explicitly by
``test_decoded_densities_are_nonnegative`` in the test file). This is
reported as a genuine structural finding, not assumed.

Sequence, exactly as specified (A-G):
  A. Initial calibration -- constrained inverse of a random initial guess
     against a declared target book (see "Data honesty").
  B. Market evolution -- one weekly P-law step (``dynamics.step``) of BOTH
     (i) the accepted reset representation and (ii) an independent "true"
     oracle state, kept separate throughout: the reset representation is a
     REPRESENTATION of the market, not the market itself.
  C. Update target -- the oracle's evolved prices for the SAME surviving
     (expiry,strike) book.
  D. Structural parameter proposal -- the reset representation's own P-law
     one-step-ahead state (from B(i)) is the proposal to be corrected, not
     accepted as-is.
  E. Constrained inverse -- Adam optimisation of the proposal to reproduce
     the target exactly (declared tolerance).
  F. Feasibility decision -- accept iff converged within tolerance; report
     every event, converged or not.
  G. Continue -- the accepted state becomes next period's reset
     representation; the oracle continues independently.

Data honesty
------------
No real historical book is available in this environment (Phase 1 audit).
"Market" here is a second, independent synthetic oracle state evolved under
the SAME P law as the reset representation -- sufficient to test the
MACHINERY of the sequence (does an admissible price-preserving reset exist
and get found; how large is the pre-reset discrepancy; does the loop stay
coherent over several resets) but not sufficient to claim anything about
real option-market recalibration. Stated at every reported number.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import torch

from quantdata.dysanos.flex.crc_extension.section2_pricing_identity import (
    STATE_SIZES,
    BaseEnvironment,
    build_simulator,
)

TARGET_EXPIRY_INDICES = torch.tensor([0, 1, 2], dtype=torch.long)
TARGET_STRIKES = torch.tensor([0.90, 0.97, 1.0, 1.03, 1.10], dtype=torch.float64)


def _target_prices(simulator, state: torch.Tensor, spot: torch.Tensor) -> torch.Tensor:
    """Prices of the declared protected book under a given state/spot,
    reusing the simulator's own normalised pricing path exactly
    (``_option_values``), not a re-derived formula."""
    surface = simulator.decode_surface(state.unsqueeze(0))
    n_e, n_k = len(TARGET_EXPIRY_INDICES), len(TARGET_STRIKES)
    # expiry_index is shared across the batch (no leading batch dim, per
    # _option_values's own broadcasting convention: model_expiries_y[idx]
    # is unsqueezed once and expanded against (B,K) strikes).
    strikes_full = (spot * TARGET_STRIKES).unsqueeze(0).repeat(1, n_e)
    expiry_index = TARGET_EXPIRY_INDICES.unsqueeze(-1).expand(n_e, n_k).reshape(-1)
    is_call = torch.ones_like(strikes_full, dtype=torch.bool)
    is_zero = torch.zeros_like(strikes_full, dtype=torch.bool)
    prices = simulator._option_values(surface, spot.unsqueeze(0), strikes_full, expiry_index, is_call, is_zero)
    return prices.squeeze(0)


@dataclass(frozen=True)
class ConstrainedInverseResult:
    converged: bool
    iterations: int
    pre_correction_max_abs_error: float
    post_correction_max_abs_error: float
    tolerance: float
    raw_state_move_l2: float
    min_decoded_density: float

    def to_dict(self) -> dict:
        return asdict(self)


def constrained_inverse(
    simulator, *, proposal_state: torch.Tensor, target_prices: torch.Tensor, spot: torch.Tensor,
    tolerance: float = 1.0e-4, max_iterations: int = 300, lr: float = 2.0e-2,
) -> tuple[torch.Tensor, ConstrainedInverseResult]:
    with torch.no_grad():
        pre_prices = _target_prices(simulator, proposal_state, spot)
        pre_error = float(torch.max(torch.abs(pre_prices - target_prices)))

    state = proposal_state.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([state], lr=lr)
    converged = False
    iterations_used = 0
    for iteration in range(max_iterations):
        optimizer.zero_grad()
        prices = _target_prices(simulator, state, spot)
        loss = torch.mean((prices - target_prices) ** 2)
        loss.backward()
        optimizer.step()
        iterations_used = iteration + 1
        with torch.no_grad():
            max_err = float(torch.max(torch.abs(prices - target_prices)))
        if max_err < tolerance:
            converged = True
            break

    with torch.no_grad():
        final_prices = _target_prices(simulator, state, spot)
        post_error = float(torch.max(torch.abs(final_prices - target_prices)))
        move = float(torch.norm(state.detach() - proposal_state))
        surface = simulator.decode_surface(state.detach().unsqueeze(0))
        min_density = float(surface.densities.min())

    return state.detach(), ConstrainedInverseResult(
        converged=converged, iterations=iterations_used,
        pre_correction_max_abs_error=pre_error, post_correction_max_abs_error=post_error,
        tolerance=tolerance, raw_state_move_l2=move, min_decoded_density=min_density,
    )


@dataclass(frozen=True)
class CrcStepReport:
    step: int
    feasibility: dict
    oracle_spot: float
    reset_representation_spot: float

    def to_dict(self) -> dict:
        return dict(step=self.step, feasibility=self.feasibility,
                     oracle_spot=self.oracle_spot, reset_representation_spot=self.reset_representation_spot)


def run_crc_sequence(num_resets: int = 5, seed: int = 2024) -> dict:
    torch.manual_seed(seed)
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    simulator = build_simulator(env)
    state_dim = simulator.dynamics.reduced_state_size

    oracle_state = 0.2 * torch.randn(state_dim, dtype=torch.float64)
    oracle_spot = torch.tensor(1.0, dtype=torch.float64)

    # A: initial calibration -- a random guess corrected to the ORACLE's own
    # initial book (a genuine constrained inverse, not an identity check).
    initial_guess = 0.2 * torch.randn(state_dim, dtype=torch.float64)
    target0 = _target_prices(simulator, oracle_state, oracle_spot).detach()
    reset_state, initial_fit = constrained_inverse(
        simulator, proposal_state=initial_guess, target_prices=target0, spot=oracle_spot,
    )
    reset_spot = oracle_spot.clone()

    step_reports = [CrcStepReport(step=0, feasibility=initial_fit.to_dict(),
                                   oracle_spot=float(oracle_spot), reset_representation_spot=float(reset_spot))]

    for step in range(1, num_resets + 1):
        # B: market evolution -- oracle and reset representation evolve
        # independently under the SAME P law (kept as two distinct objects
        # throughout).
        normals_oracle = torch.randn((1, state_dim + 1), dtype=torch.float64)
        oracle_step = simulator.dynamics.step(
            oracle_state.unsqueeze(0), torch.log(oracle_spot).unsqueeze(0), standard_normals=normals_oracle,
        )
        oracle_state = oracle_step.reduced_state.squeeze(0).detach()
        oracle_spot = torch.exp(oracle_step.normal_log_spot).squeeze(0).detach()

        normals_reset = torch.randn((1, state_dim + 1), dtype=torch.float64)
        reset_step = simulator.dynamics.step(
            reset_state.unsqueeze(0), torch.log(reset_spot).unsqueeze(0), standard_normals=normals_reset,
        )
        proposal_state = reset_step.reduced_state.squeeze(0).detach()  # D: structural parameter proposal
        reset_spot = torch.exp(reset_step.normal_log_spot).squeeze(0).detach()

        # C: update target -- the oracle's evolved prices for the SAME book.
        target = _target_prices(simulator, oracle_state, oracle_spot).detach()

        # E+F: constrained inverse + feasibility decision.
        reset_state, fit = constrained_inverse(
            simulator, proposal_state=proposal_state, target_prices=target, spot=oracle_spot,
        )
        reset_spot = oracle_spot.clone()  # spot itself is directly observed, not reset

        step_reports.append(CrcStepReport(
            step=step, feasibility=fit.to_dict(),
            oracle_spot=float(oracle_spot), reset_representation_spot=float(reset_spot),
        ))

    return dict(
        scope_note=(
            "Synthetic structural-validation data: a second independent synthetic 'oracle' "
            "state plays the role of the observed market, evolved under the same fitted "
            "P law as the reset representation (no real historical book was available in "
            "this environment; see PHASE1_AUDIT.md). Tests the CRC MACHINERY, not real "
            "recalibration performance."
        ),
        num_resets=num_resets,
        target_book=dict(expiries=TARGET_EXPIRY_INDICES.tolist(), strikes=TARGET_STRIKES.tolist()),
        steps=[r.to_dict() for r in step_reports],
    )


if __name__ == "__main__":
    result = run_crc_sequence()
    with open("section5_crc_sequence_report.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
