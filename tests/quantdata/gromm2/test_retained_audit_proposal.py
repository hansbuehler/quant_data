from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from quantdata.gromm2.conditional_audit import (
    ActionNetworkConfig,
    AffineDeepAction,
    LocalActionConfig,
)
from quantdata.gromm2.ggn_audit_continuation import action_point_sha256
from quantdata.gromm2.ggn_audit_optimization import _flatten
from quantdata.gromm2.retained_audit_proposal import retained_actual_step
from quantdata.torch import BaseEnvironment


def prepared():
    with torch.random.fork_rng():
        torch.manual_seed(5)
        policy = AffineDeepAction(BaseEnvironment(device='cpu', precision=64),
            config=LocalActionConfig(label='retained-step', feature_count=2, instrument_count=2, action_scale=.2,
                network=ActionNetworkConfig(num_depth=1, num_width=2)))
        shadow = deepcopy(policy)
        with torch.no_grad():
            for value in shadow.parameters():
                value.add_(.01)
    candidate = {'interpretation': 'shadow candidate, not a native trained endpoint',
        'model': deepcopy(shadow.state_dict()), 'report': {'point_sha256': action_point_sha256(shadow)},
        'actual_increment': _flatten({name: value.detach()-dict(policy.named_parameters())[name].detach()
            for name, value in shadow.named_parameters()})}
    return policy, candidate


def test_retained_step_is_exact_actual_increment_without_nominal_recovery_or_mutation():
    policy, candidate = prepared()
    original_point, rng = action_point_sha256(policy), torch.get_rng_state().clone()
    direction, report = retained_actual_step(policy=policy, candidate=candidate,
        source_point_sha256=original_point, candidate_point_sha256=candidate['report']['point_sha256'])
    assert torch.equal(direction, candidate['actual_increment'])
    assert direction.data_ptr() != candidate['actual_increment'].data_ptr()
    assert not report['exact_original_nominal_direction_recovered']
    assert report['full_curvature_passes'] == report['optimizer_updates'] == 0
    assert action_point_sha256(policy) == original_point and torch.equal(torch.get_rng_state(), rng)
    assert all(value.grad is None for value in policy.parameters())


@pytest.mark.parametrize('case', ['source', 'candidate', 'increment', 'nonfinite', 'precision', 'interpretation', 'affine_only'])
def test_retained_step_rejects_unbound_or_changed_evidence(case):
    policy, candidate = prepared()
    source = action_point_sha256(policy)
    point = candidate['report']['point_sha256']
    if case == 'source':
        source = '0'*64
    elif case == 'candidate':
        point = '0'*64
    elif case == 'increment':
        candidate['actual_increment'][0] += .01
    elif case == 'nonfinite':
        next(iter(candidate['model'].values()))[0] = float('nan')
    elif case == 'precision':
        candidate['actual_increment'] = candidate['actual_increment'].float()
    elif case == 'interpretation':
        candidate['interpretation'] = 'native trained checkpoint'
    else:
        policy.affine_only = True
    with pytest.raises(ValueError):
        retained_actual_step(policy=policy, candidate=candidate,
            source_point_sha256=source, candidate_point_sha256=point)
