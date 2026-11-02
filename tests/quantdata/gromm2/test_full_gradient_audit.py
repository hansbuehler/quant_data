from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

import h5py
import numpy as np
import pytest
import torch

from quantdata.gromm2 import full_gradient_audit
from quantdata.gromm2.affine_audit_optimization import AffineAuditProblem, AffineSolveConfig
from quantdata.gromm2.conditional_audit import (
    ActionNetworkConfig,
    AffineDeepAction,
    LocalActionConfig,
)
from quantdata.gromm2.full_gradient_audit import (
    fit_full_gradient_fold,
    freeze_full_gradient_actions,
)
from quantdata.gromm2.local_audit_training import LocalAuditProtocol
from quantdata.gromm2.neural_audit_optimization import NeuralSolveConfig
from quantdata.gromm2.path_audit import PathAuditConfig
from quantdata.gromm2.terminal_audit import terminal_semantic_hashes
from quantdata.gromm2.terminal_inference import load_terminal_actions
from quantdata.torch import BaseEnvironment


@pytest.fixture
def population(tmp_path, monkeypatch):
    monkeypatch.setattr(full_gradient_audit, 'check_resource_headroom', lambda: None)
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
        'neural_config': NeuralSolveConfig(maximum_iterations=3)}
    return path, controls


def test_complete_fold_fits_actual_full_class_without_reading_opposite_arrays(population, tmp_path):
    path, controls = population
    with h5py.File(path, 'r+') as saved:
        saved['features'][16:] = float('nan')
        saved['returns'][16:] = float('nan')
    report = fit_full_gradient_fold(path=path, output=tmp_path/'fold-0', fold=0, device='cpu', **controls)
    assert report['status'] == 'both independent full-gradient policies fitted; inference pending'
    assert report['outer_training_paths'] == [0, 16]
    assert report['outer_holdout_read'] is False
    for name, fitted in report['diagnostics'].items():
        assert fitted['affine']['converged']
        assert fitted['neural']['converged']
        assert fitted['neural']['history'][-1]['training_loss'] < 0.
        assert fitted['weights_sha256'] == hashlib.sha256((tmp_path/f'fold-0/{name}-weights.pt').read_bytes()).hexdigest()


def test_two_completed_folds_freeze_for_exact_declared_confirmation(population, tmp_path):
    path, controls = population
    folds = (tmp_path/'fold-0', tmp_path/'fold-1')
    for fold, output in enumerate(folds):
        fit_full_gradient_fold(path=path, output=output, fold=fold, device='cpu', **controls)
    protocol = tmp_path/'PROTOCOL.md'
    protocol.write_text('Synthetic full-gradient confirmation, before evaluation.\n')
    plan = tmp_path/'evaluation-plan.json'
    plan.write_text(json.dumps({'schema': 'gromm2-terminal-confirmation-plan-v1',
        'protocol': str(protocol), 'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
        'semantic_sha256': terminal_semantic_hashes(), 'familywise_endpoint_count': 12,
        'full_gradient_controls': {'affine': asdict(controls['affine_config']),
            'neural': asdict(controls['neural_config']), 'protocol': asdict(controls['protocol'])},
        'variants': {'eight_step': {'paths': 32, 'batch_size': 8, 'substeps': 8, 'seed': 72},
            'one_step': {'paths': 32, 'batch_size': 8, 'substeps': 1, 'seed': 73}}}))
    destination = tmp_path/'frozen'
    frozen = freeze_full_gradient_actions(folds=folds, output=destination, evaluation_plan=plan, variant='eight_step')
    assert frozen['outer_or_fresh_evaluation_arrays_read'] is False
    assert frozen['exact_global_optimum_certified'] is False
    assert freeze_full_gradient_actions(folds=folds, output=destination, evaluation_plan=plan, variant='eight_step') == frozen
    _, actions = load_terminal_actions(destination/'terminal-policies.json', device='cpu')
    assert len(actions) == 4
    assert all(next(action.model.parameters()).dtype == torch.float64 for action in actions.values())
    assert all(action.mean.dtype == torch.float32 for action in actions.values())
    weights = folds[0]/'entropy-weights.pt'
    weights.write_bytes(weights.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='weights changed'):
        freeze_full_gradient_actions(folds=folds, output=tmp_path/'tampered', evaluation_plan=plan, variant='eight_step')


def test_resume_rejects_changed_lbfgs_curvature_history_size(population, tmp_path):
    path, controls = population
    fit_full_gradient_fold(path=path, output=tmp_path/'fold', fold=0, device='cpu', **controls)
    controls['neural_config'] = replace(controls['neural_config'], history_size=1)
    with pytest.raises(ValueError, match='curvature history size'):
        fit_full_gradient_fold(path=path, output=tmp_path/'fold', fold=0, device='cpu', **controls)


@pytest.mark.parametrize('squared_lambda', [1, 2])
def test_saved_normalization_identity_compares_numeric_values_not_json_spelling(population, tmp_path, squared_lambda):
    path, controls = population
    output = tmp_path/'fold'
    output.mkdir()
    source, _ = full_gradient_audit._path_identity(path)
    protocol = controls['protocol']
    identity = json.dumps({'path_identity': source, 'protocol': asdict(protocol), 'fold': 0,
        'protocol_version': LocalAuditProtocol.version.unique_id48}, sort_keys=True)
    mask = full_gradient_audit.internal_training_mask(16, protocol=protocol, fold=0)
    full_gradient_audit._normalization(path, outer=slice(0, 16), mask=mask, protocol=protocol,
        output=output/'normalization.npz', identity=identity, upload_paths=8)
    original = (output/'normalization.npz').read_bytes()
    controls['protocol'] = replace(protocol, squared_lambda=squared_lambda)
    if squared_lambda == 2:
        with pytest.raises(ValueError, match='normalization has a different identity'):
            fit_full_gradient_fold(path=path, output=output, fold=0, device='cpu', **controls)
    else:
        fitted = fit_full_gradient_fold(path=path, output=output, fold=0, device='cpu', **controls)
        assert fitted['status'] == 'both independent full-gradient policies fitted; inference pending'
    assert (output/'normalization.npz').read_bytes() == original


def test_external_initializer_is_bound_to_checkpoint_and_actual_training_objective(tmp_path):
    problem = AffineAuditProblem(states=torch.zeros((8, 1), dtype=torch.float32),
        returns=torch.full((8, 1), .1, dtype=torch.float32), eligible=torch.ones((1, 1), dtype=torch.bool),
        rows=torch.arange(8), action_scale=.2)
    weights, report = problem.solve(diagnostic='squared')
    report.update(source_population_identity_sha256='population', outer_holdout_read=False)
    config = LocalActionConfig(label='adoption-test', feature_count=1, instrument_count=1, action_scale=.2,
        network=ActionNetworkConfig(num_depth=1, num_width=4))
    model = AffineDeepAction(BaseEnvironment(device='cpu', precision=64), config=config)
    with torch.no_grad():
        model.affine.I0.weight.copy_(weights[:-1].T)
        model.affine.I0.bias.copy_(weights[-1])
    source = tmp_path/'implementation.py'
    source.write_text('synthetic source identity\n')
    plan = {'normalization_sha256': 'normalization',
        'implementation_sha256': {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}}
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    (tmp_path/'squared-report.json').write_text(json.dumps(report))
    torch.save({'plan': plan, 'weights': weights, 'state': {'gradient_converged': True,
        'diagnostic': 'squared', 'history': report['history']}}, tmp_path/'squared-checkpoint.pt')
    export = {'plan': plan, 'model_config': asdict(config), 'state_dict': model.state_dict(),
        'code_version': LocalActionConfig.version.unique_id48, 'precision': 64}
    torch.save(export, tmp_path/'squared-affine-initializer.pt')
    identity = {'source_population_identity_sha256': 'population', 'normalization_sha256': 'normalization'}
    _, provenance = full_gradient_audit._adopt_affine_initializer(root=tmp_path, diagnostic='squared',
        policy=model, problem=problem, identity=identity)
    assert provenance['checkpoint_export_equal']
    assert len(provenance['artifacts_sha256']) == 4
    export['state_dict']['affine.I0.bias'] += 1.
    torch.save(export, tmp_path/'squared-affine-initializer.pt')
    with pytest.raises(ValueError, match='accepted checkpoint'):
        full_gradient_audit._adopt_affine_initializer(root=tmp_path, diagnostic='squared',
            policy=model, problem=problem, identity=identity)
