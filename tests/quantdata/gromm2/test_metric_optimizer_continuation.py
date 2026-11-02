from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from quantdata.gromm2 import metric_optimizer_continuation
from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem, AffineSolveConfig
from quantdata.gromm2.conditional_audit import (
    ActionNetworkConfig,
    AffineDeepAction,
    LocalActionConfig,
)
from quantdata.gromm2.metric_audit_optimization import (
    AffineBlockMetric,
    MetricLBFGS,
    MetricSolveConfig,
    fit_metric_action,
)
from quantdata.gromm2.metric_optimizer_continuation import prepare_deep_metric_continuation
from quantdata.gromm2.neural_audit_optimization import NeuralSolveConfig
from quantdata.torch import BaseEnvironment


@pytest.fixture
def checkpoint(tmp_path):
    generator = torch.Generator().manual_seed(61)
    states = torch.randn((64, 3), generator=generator, dtype=torch.float64)
    problem = AffineAuditProblem(states=states, returns=.2*states[:, :2]+.1*states[:, :2].square(),
        eligible=torch.tensor([[True, True], [True, False]], dtype=torch.bool), rows=torch.arange(48),
        action_scale=.2, config=AffineSolveConfig(batch_size=13))
    torch.manual_seed(71)
    policy = AffineDeepAction(BaseEnvironment(device='cpu', precision=64),
        config=LocalActionConfig(label='metric-transition', feature_count=3, instrument_count=2,
            action_scale=.2, network=ActionNetworkConfig(num_depth=1, num_width=6)))
    source = tmp_path/'source.pt'

    def save(model, optimizer, state):
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'state': state,
            'identity': {'source': 'synthetic'}, 'initializer_provenance': {'original': 'retained'},
            'code_version': LocalActionConfig.version.unique_id48, 'user_version': policy.config.user_version,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': None}, source)

    config = MetricSolveConfig(neural=NeuralSolveConfig(maximum_iterations=4,
        gradient_relative_tolerance=1.e-12, gradient_absolute_tolerance=1.e-15))
    fit_metric_action(policy=policy, problem=problem, diagnostic='squared', data_identity='synthetic-source',
        config=config, checkpoint=save)
    return source, problem, policy, config


def test_explicit_metric_transition_preserves_model_raw_history_rng_threshold_and_native_resume(checkpoint, tmp_path):
    source, problem, policy, config = checkpoint
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    original = torch.load(source, weights_only=False)
    rng = torch.get_rng_state().clone()
    output = tmp_path/'prepared.pt'
    manifest = prepare_deep_metric_continuation(source=source, output=output, source_sha256=digest)
    derived = torch.load(output, weights_only=False)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    assert torch.equal(torch.get_rng_state(), rng)
    assert manifest['derived_checkpoint_sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    for field in ('model',):
        for name, expected in original[field].items():
            torch.testing.assert_close(derived[field][name], expected, atol=0., rtol=0.)
    assert derived['state']['history'] == original['state']['history']
    assert derived['state']['initial_gradient_norm'] == original['state']['initial_gradient_norm']
    assert derived['state']['full_objective_evaluations'] == original['state']['full_objective_evaluations']
    assert derived['identity'] == original['identity']
    assert torch.equal(derived['torch_rng'], original['torch_rng'])
    assert derived['initializer_provenance']['previous_initializer_provenance'] == original['initializer_provenance']
    for actual, expected in zip(derived['optimizer']['history'], original['optimizer']['history'], strict=True):
        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_tensor, expected_tensor, atol=0., rtol=0.)
    scale = manifest['transition']['derived_metric_identity']['deep_scale']
    assert scale > 0.
    changed = replace(config, deep_metric_scale=scale, neural=replace(config.neural, maximum_iterations=5))
    policy.load_state_dict(derived['model'], strict=True)
    optimizer = MetricLBFGS(AffineBlockMetric(policy, problem, deep_scale=scale), history_size=30, curvature_floor=1.e-12)
    optimizer.load_state_dict(derived['optimizer'])
    report = fit_metric_action(policy=policy, problem=problem, diagnostic='squared', data_identity='synthetic-source',
        config=changed, optimizer=optimizer, resume_state=derived['state'])
    assert report['history'][4]['training_loss'] == original['state']['history'][-1]['training_loss']
    assert report['history'][4]['gradient_threshold'] == original['state']['history'][-1]['gradient_threshold']
    assert report['history'][-1]['training_loss'] < report['history'][4]['training_loss']


def test_retained_raw_secants_with_changed_initial_metric_equal_dense_inverse_bfgs(checkpoint, tmp_path):
    source, problem, policy, _ = checkpoint
    output = tmp_path/'prepared.pt'
    manifest = prepare_deep_metric_continuation(source=source, output=output,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    saved = torch.load(output, weights_only=False)
    metric = AffineBlockMetric(policy, problem, deep_scale=manifest['transition']['derived_metric_identity']['deep_scale'])
    optimizer = MetricLBFGS(metric, history_size=30, curvature_floor=1.e-12)
    optimizer.load_state_dict(saved['optimizer'])
    basis = torch.eye(metric.size, dtype=torch.float64)
    inverse = torch.stack([metric(vector) for vector in basis], dim=1)
    for step, change, reciprocal in optimizer.history:
        transform = basis-reciprocal*torch.outer(step, change)
        inverse = transform@inverse@transform.T+reciprocal*torch.outer(step, step)
    gradient = torch.randn(metric.size, generator=torch.Generator().manual_seed(18), dtype=torch.float64)
    torch.testing.assert_close(optimizer.direction(gradient), -inverse@gradient, rtol=1.e-9, atol=1.e-10)
    assert torch.dot(gradient, optimizer.direction(gradient)) < 0.


def test_metric_transition_fails_closed_for_changed_source_invalid_raw_history_and_overwrite(checkpoint, tmp_path):
    source, _, _, _ = checkpoint
    output = tmp_path/'prepared.pt'
    with pytest.raises(ValueError, match='source changed'):
        prepare_deep_metric_continuation(source=source, output=output, source_sha256='0'*64)
    original = torch.load(source, weights_only=False)
    modified = deepcopy(original)
    modified['optimizer']['history'][0][2].mul_(-1.)
    invalid = tmp_path/'invalid.pt'
    torch.save(modified, invalid)
    with pytest.raises(ValueError, match='curvature or reciprocal'):
        prepare_deep_metric_continuation(source=invalid, output=output,
            source_sha256=hashlib.sha256(invalid.read_bytes()).hexdigest())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    prepare_deep_metric_continuation(source=source, output=output, source_sha256=digest)
    with pytest.raises(ValueError, match='already exists'):
        prepare_deep_metric_continuation(source=source, output=output, source_sha256=digest)


def test_metric_transition_rejects_manifest_alias_and_nonfinite_median(checkpoint, tmp_path, monkeypatch):
    source, _, _, _ = checkpoint
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='distinct paths'):
        prepare_deep_metric_continuation(source=source, output=tmp_path/'derived.json', source_sha256=digest)
    monkeypatch.setattr(metric_optimizer_continuation.statistics, 'median', lambda values: float('inf'))
    with pytest.raises(ValueError, match='median metric scale'):
        prepare_deep_metric_continuation(source=source, output=tmp_path/'prepared.pt', source_sha256=digest)
    assert not (tmp_path/'prepared.pt').exists()
