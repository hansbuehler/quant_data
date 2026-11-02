"""Read-only full-policy proposals from the original positive raw history."""

from __future__ import annotations

import math

import torch

from quantdata.gromm2.conditional_audit import AffineDeepAction
from quantdata.gromm2.ggn_audit_continuation import action_point_sha256, optimizer_state_sha256
from quantdata.gromm2.gradient_reliable_ggn import verify_actual_metric
from quantdata.gromm2.lbfgs_audit_preconditioner import FrozenLBFGSPreconditioner
from quantdata.gromm2.metric_audit_optimization import MetricLBFGS


@torch.no_grad()
def propose_positive_history_step(*, policy: AffineDeepAction, optimizer: MetricLBFGS,
                                  gradient: torch.Tensor, point_sha256: str) -> tuple[torch.Tensor, dict]:
    """Propose -Pg without curvature passes, optimizer updates or a certificate.

    P is the frozen original positive inverse-BFGS operator. Every candidate
    still needs a complete canonical objective/gradient and exact-Hessian
    nonlinear-remainder check; positive descent alone is not acceptance or
    evidence that either original stationarity gate has been met.
    """
    verify_actual_metric(optimizer.metric, policy)
    first = next(policy.parameters())
    if (policy.affine_only or action_point_sha256(policy) != point_sha256
        or gradient.shape != (optimizer.metric.size,) or gradient.dtype != torch.float64
        or gradient.device != first.device or first.dtype != torch.float64
        or not bool(torch.isfinite(gradient).all()) or float(torch.linalg.vector_norm(gradient)) == 0.):
        raise ValueError('history proposal requires the exact full-policy point and finite nonzero canonical gradient')
    frozen = FrozenLBFGSPreconditioner(optimizer, policy)
    image, energy = frozen.apply_and_energy(gradient)
    direction = -image
    slope, stable_energy = float(torch.dot(gradient, direction)), float(energy)
    if (not bool(torch.isfinite(direction).all()) or not math.isfinite(slope) or slope >= 0.
        or not math.isfinite(stable_energy) or stable_energy <= 0.
        or not math.isclose(-slope, stable_energy, rel_tol=1.e-7, abs_tol=0.)):
        raise FloatingPointError('positive-history direction is not numerically reliable strict descent')
    frozen.check_fixed()
    if (optimizer_state_sha256(optimizer) != frozen.source_sha256
        or action_point_sha256(policy) != point_sha256):
        raise ValueError('history proposal changed its accepted native point or raw optimizer')
    verify_actual_metric(optimizer.metric, policy)
    return direction, {'phase': 'full-policy-frozen-positive-history-proposal',
        'source_point_sha256': point_sha256, 'source_optimizer_sha256': frozen.source_sha256,
        'retained_positive_secants': len(frozen.history), 'slope': slope,
        'stable_positive_directional_energy': stable_energy,
        'full_curvature_passes': 0, 'inner_linear_solve_performed': False,
        'optimizer_updates': 0, 'original_objective_unchanged': True,
        'outer_gradient_convergence_certified': False}
