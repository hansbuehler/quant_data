from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

import h5py
import numpy as np
import pytest
import torch

from quantdata.gromm2 import metric_gradient_audit
from quantdata.gromm2.affine_audit_optimization import AffineSolveConfig
from quantdata.gromm2.conditional_audit import ActionNetworkConfig
from quantdata.gromm2.full_gradient_audit import fit_full_gradient_fold
from quantdata.gromm2.local_audit_training import LocalAuditProtocol
from quantdata.gromm2.metric_audit_optimization import MetricSolveConfig
from quantdata.gromm2.metric_gradient_audit import (
    fit_metric_gradient_fold,
    freeze_metric_actions,
    validate_migration_resolution,
)
from quantdata.gromm2.neural_audit_optimization import NeuralSolveConfig
from quantdata.gromm2.path_audit import PathAuditConfig
from quantdata.gromm2.terminal_audit import terminal_semantic_hashes
from quantdata.gromm2.terminal_inference import load_terminal_actions


@pytest.fixture
def population(tmp_path, monkeypatch):
    monkeypatch.setattr(metric_gradient_audit, 'check_resource_headroom', lambda: None)
    monkeypatch.setattr('quantdata.gromm2.full_gradient_audit.check_resource_headroom', lambda: None)
    path = tmp_path/'population.h5'
    config = PathAuditConfig(paths=32, batch_size=8, days=5, substeps=8, seed=71)
    states = np.random.default_rng(19).normal(size=(32, 5, 3)).astype('f4')
    with h5py.File(path, 'w') as saved:
        saved.attrs.update(identity=json.dumps({'config': asdict(config)}, sort_keys=True), completed_paths=32)
        saved['features'] = states
        saved['returns'] = np.broadcast_to(np.array([.02, -.01, .03], dtype='f4'), states.shape)
        saved['path_validity'] = np.ones(32, dtype=bool)
        saved['batch_diagnostic_validity'] = np.ones(4, dtype=bool)
        saved['expiry_days'] = [5]
        saved['strikes'] = [[.9, 1.1]]
    controls = {'protocol': LocalAuditProtocol(network=ActionNetworkConfig(num_depth=1, num_width=4)),
        'affine_config': AffineSolveConfig(batch_size=16),
        'metric_config': MetricSolveConfig(neural=NeuralSolveConfig(maximum_iterations=3))}
    return path, controls


def test_metric_complete_fold_and_exact_native_restart_avoid_opposite_arrays(population, tmp_path):
    path, controls = population
    with h5py.File(path, 'r+') as saved:
        saved['features'][16:] = float('nan')
        saved['returns'][16:] = float('nan')
    output = tmp_path/'fold'
    report = fit_metric_gradient_fold(path=path, output=output, fold=0, device='cpu', **controls)
    assert report['outer_holdout_read'] is False
    for name in ('squared', 'entropy'):
        assert report['diagnostics'][name]['neural']['converged']
    original = {name: torch.load(output/f'{name}-weights.pt', weights_only=False)['state_dict']
        for name in ('squared', 'entropy')}
    fit_metric_gradient_fold(path=path, output=output, fold=0, device='cpu', **controls)
    for name, state in original.items():
        saved = torch.load(output/f'{name}-weights.pt', weights_only=False)
        for parameter, expected in state.items():
            torch.testing.assert_close(saved['state_dict'][parameter], expected, atol=0., rtol=0.)


def test_metric_freeze_loads_declared_full_actions_and_rejects_changed_controls(population, tmp_path):
    path, controls = population
    folds = (tmp_path/'fold-0', tmp_path/'fold-1')
    for fold, output in enumerate(folds):
        fit_metric_gradient_fold(path=path, output=output, fold=fold, device='cpu', **controls)
    protocol = tmp_path/'PROTOCOL.md'
    protocol.write_text('Fixed SPD metric; original objectives and model class.\n')
    plan = tmp_path/'evaluation-plan.json'
    settings = {'schema': 'gromm2-terminal-confirmation-plan-v1', 'protocol': str(protocol),
        'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
        'semantic_sha256': terminal_semantic_hashes(), 'familywise_endpoint_count': 12,
        'metric_controls': asdict(controls['metric_config']),
        'full_gradient_controls': {'affine': asdict(controls['affine_config']),
            'neural': asdict(controls['metric_config'].neural), 'protocol': asdict(controls['protocol'])},
        'variants': {'eight_step': {'paths': 32, 'batch_size': 8, 'substeps': 8, 'seed': 72},
            'one_step': {'paths': 32, 'batch_size': 8, 'substeps': 1, 'seed': 73}}}
    plan.write_text(json.dumps(settings))
    frozen = freeze_metric_actions(folds=folds, output=tmp_path/'frozen', evaluation_plan=plan, variant='eight_step')
    assert frozen['outer_or_fresh_evaluation_arrays_read'] is False
    _, actions = load_terminal_actions(tmp_path/'frozen/terminal-policies.json', device='cpu')
    assert len(actions) == 4
    settings['metric_controls']['deep_metric_scale'] = 2.
    plan.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match='controls differ'):
        freeze_metric_actions(folds=folds, output=tmp_path/'changed', evaluation_plan=plan, variant='eight_step')


def test_raw_optimizer_migration_preserves_actual_weights_objective_threshold_and_prior_work(population, tmp_path):
    path, controls = population
    raw_output = tmp_path/'raw'
    fit_full_gradient_fold(path=path, output=raw_output, fold=0, device='cpu', protocol=controls['protocol'],
        affine_config=controls['affine_config'], neural_config=controls['metric_config'].neural)
    checkpoint = raw_output/'squared-neural.pt'
    previous = torch.load(checkpoint, weights_only=False)
    specification = {'checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'diagnostic': 'squared'}
    report = fit_metric_gradient_fold(path=path, output=tmp_path/'migrated', fold=0,
        migration_specs={'squared': specification}, device='cpu', **controls)
    result = report['diagnostics']['squared']['neural']
    assert result['optimizer_migration']['prior_accepted_updates'] == previous['state']['history'][-1]['iteration']
    assert result['history'][0]['training_loss'] == pytest.approx(previous['state']['history'][-1]['training_loss'])
    assert result['history'][0]['gradient_threshold'] <= previous['state']['gradient_threshold']
    changed = replace(controls['metric_config'], neural=replace(controls['metric_config'].neural,
        gradient_relative_tolerance=.1, gradient_absolute_tolerance=.1))
    with pytest.raises(ValueError, match='cannot relax'):
        fit_metric_gradient_fold(path=path, output=tmp_path/'relaxed', fold=0,
            migration_specs={'squared': specification}, device='cpu', **{**controls, 'metric_config': changed})
    with pytest.raises(ValueError, match='checkpoint changed'):
        fit_metric_gradient_fold(path=path, output=tmp_path/'tampered', fold=0,
            migration_specs={'squared': {**specification, 'checkpoint_sha256': '0'*64}}, device='cpu', **controls)
    with pytest.raises(ValueError, match='preserve each named diagnostic'):
        fit_metric_gradient_fold(path=path, output=tmp_path/'swapped', fold=0,
            migration_specs={'entropy': specification}, device='cpu', **controls)


def test_resolution_gate_checks_hashes_actual_guard_and_accepted_state_on_every_restart(tmp_path):
    previous = tmp_path/'previous'
    directory = previous/'eight_step/fold-0'
    directory.mkdir(parents=True)
    source_plan = previous/'run-plan.json'
    source_plan.write_text(json.dumps({'controls': {'neural': {'maximum_iterations': 300}}}))
    report_path, checkpoint = directory/'training-report.json', directory/'squared-neural.pt'
    identity = {'source_population_identity_sha256': 'original-population'}
    history = [{'iteration': 300, 'closure_evaluations': 1201}]
    report = {'identity': identity, 'diagnostics': {'squared': {'neural': {'converged': False,
        'stop_reason': 'work_limit_not_converged', 'history': history}}}}
    report_path.write_text(json.dumps(report))
    torch.save({'identity': identity, 'state': {'history': history, 'closure_evaluations': 1201,
        'empirical_local_stationarity': False}}, checkpoint)
    plan = {'preceding_plan': str(source_plan), 'preceding_plan_sha256': hashlib.sha256(source_plan.read_bytes()).hexdigest(),
        'source_population_identity_sha256': 'original-population'}
    resolution = {'preceding_plan_sha256': plan['preceding_plan_sha256'], 'preceding_report': str(report_path),
        'preceding_report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'specifications': {'squared': {'checkpoint': str(checkpoint), 'diagnostic': 'squared',
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}}}
    validate_migration_resolution(resolution=resolution, plan=plan)
    report['diagnostics']['squared']['neural']['converged'] = True
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='report or diagnostics changed'):
        validate_migration_resolution(resolution=resolution, plan=plan)
    resolution['preceding_report_sha256'] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='actual work-guard failure'):
        validate_migration_resolution(resolution=resolution, plan=plan)
