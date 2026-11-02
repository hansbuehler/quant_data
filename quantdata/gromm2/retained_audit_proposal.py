"""Source-bound reuse of an actual rounded shadow increment, not nominal CG."""

from __future__ import annotations

import hashlib

import torch

from quantdata.gromm2.conditional_audit import AffineDeepAction
from quantdata.gromm2.ggn_audit_continuation import action_point_sha256
from quantdata.gromm2.ggn_audit_optimization import _flatten


@torch.no_grad()
def retained_actual_step(*, policy: AffineDeepAction, candidate: dict,
                         source_point_sha256: str, candidate_point_sha256: str) -> tuple[torch.Tensor, dict]:
    """Validate and clone a prior source-to-candidate parameter displacement.

    The returned vector is the saved ACTUAL rounded increment. It is not
    divided by its previous line-search rate or relabelled as the exact
    original nominal GGN direction. The caller still performs the original
    full-data canonical Armijo/exact-H acceptance at the new declared rate.
    """
    if policy.affine_only or action_point_sha256(policy) != source_point_sha256:
        raise ValueError('retained displacement requires the exact original full-policy source point')
    first = next(policy.parameters())
    if any(value.dtype != torch.float64 for value in policy.parameters()):
        raise ValueError('retained displacement requires original float64 native parameters')
    source = policy.state_dict()
    saved = candidate['model']
    if (candidate['interpretation'] != 'shadow candidate, not a native trained endpoint'
        or set(saved) != set(source)
        or any(not isinstance(value, torch.Tensor) or value.shape != source[name].shape
            or value.dtype != source[name].dtype or not bool(torch.isfinite(value).all())
            for name, value in saved.items())):
        raise ValueError('retained candidate model geometry, precision or interpretation changed')
    point = hashlib.sha256()
    for name, value in (*policy.named_parameters(), *policy.named_buffers()):
        point.update(name.encode())
        point.update(saved[name].detach().cpu().contiguous().numpy().tobytes())
    if point.hexdigest() != candidate_point_sha256 or candidate['report']['point_sha256'] != candidate_point_sha256:
        raise ValueError('retained candidate point differs from the original shadow report')
    difference = _flatten({name: saved[name].detach().cpu()-value.detach().cpu()
        for name, value in policy.named_parameters()})
    increment = candidate['actual_increment']
    if (not isinstance(increment, torch.Tensor) or increment.shape != difference.shape
        or increment.dtype != torch.float64 or not bool(torch.isfinite(increment).all())
        or not torch.equal(increment.detach().cpu(), difference) or not bool(increment.ne(0.).any())):
        raise ValueError('retained increment must equal the nonzero actual rounded candidate-minus-source displacement')
    return increment.detach().to(first.device).clone(), {
        'phase': 'retained-actual-shadow-increment', 'source_point_sha256': source_point_sha256,
        'candidate_point_sha256': candidate_point_sha256, 'exact_original_nominal_direction_recovered': False,
        'full_curvature_passes': 0, 'optimizer_updates': 0,
        'original_objective_unchanged': True, 'outer_gradient_convergence_certified': False}
