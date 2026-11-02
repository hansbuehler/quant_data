from __future__ import annotations

import hashlib
import json
import os
import signal
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from quantdata.gromm2 import selected_audit_inference, validation_selected_audit
from quantdata.gromm2.affine_audit_optimization import AffineSolveConfig
from quantdata.gromm2.conditional_audit import ActionNetworkConfig
from quantdata.gromm2.local_audit_training import LocalAuditProtocol
from quantdata.gromm2.metric_audit_optimization import MetricSolveConfig
from quantdata.gromm2.neural_audit_optimization import NeuralSolveConfig
from quantdata.gromm2.path_audit import PathAuditConfig
from quantdata.gromm2.selected_audit_artifacts import freeze_selected_actions, load_selected_actions
from quantdata.gromm2.selected_audit_inference import evaluate_fresh_selected
from quantdata.gromm2.terminal_audit import terminal_semantic_hashes
from quantdata.gromm2.terminal_inference import (
    ENDPOINTS,
    load_terminal_actions,
    score_terminal_batch,
    summarize_terminal_scores,
)
from quantdata.gromm2.validation_action_optimization import ValidationSelectionConfig
from quantdata.gromm2.validation_selected_audit import (
    fit_validation_selected_fold,
    selection_source_hashes,
)


@pytest.fixture
def population(tmp_path, monkeypatch):
    monkeypatch.setattr(validation_selected_audit, 'check_resource_headroom', lambda: None)
    path = tmp_path/'population.h5'
    config = PathAuditConfig(paths=32, batch_size=8, days=5, substeps=8, seed=71,
        retain_nonfinite=True, nms=(-.1, .1))
    states = np.random.default_rng(19).normal(size=(32, 5, 3)).astype('f4')
    returns = .01*(states**2+.2*states)+np.array([.02, -.01, .03], dtype='f4')
    identity = json.dumps({'config': asdict(config)}, sort_keys=True)
    with h5py.File(path, 'w') as saved:
        saved.attrs.update(identity=identity, completed_paths=32)
        saved['features'], saved['returns'] = states, returns
        saved['path_validity'] = np.ones(32, dtype=bool)
        saved['batch_diagnostic_validity'] = np.ones(4, dtype=bool)
        saved['expiry_days'], saved['strikes'] = [5], [[.9, 1.1]]
    controls = {'protocol': LocalAuditProtocol(network=ActionNetworkConfig(num_depth=1, num_width=4)),
        'affine_config': AffineSolveConfig(batch_size=16),
        'optimizer_config': MetricSolveConfig(neural=NeuralSolveConfig(maximum_iterations=4,
            gradient_relative_tolerance=1.e-12, gradient_absolute_tolerance=1.e-15)),
        'selection_config': ValidationSelectionConfig(interval=1, patience=2)}
    declaration = tmp_path/'declaration.json'
    declaration.write_text(json.dumps({'schema': 'gromm2-validation-selection-declaration-v1',
        'controls': {'protocol': asdict(controls['protocol']), 'affine': asdict(controls['affine_config']),
            'optimizer': asdict(controls['optimizer_config']), 'selection': asdict(controls['selection_config'])},
        'implementation_sha256': selection_source_hashes(), 'independent_fresh_confirmation_only': True,
        'source_population_identity_sha256': hashlib.sha256(identity.encode()).hexdigest()}))
    return path, declaration, controls


@pytest.fixture
def fitted_pair(population, tmp_path):
    path, declaration, controls = population
    calibration = tmp_path/'calibration'
    own = calibration/'own_dates/2025-08-29'
    own.mkdir(parents=True)
    reference = np.array([.04])
    np.savez(own/'inputs.npz', projected_total_variances=reference)
    metadata = {'calibration_root': str(calibration), 'origin': '2025-08-29', 'background_volatility': .05}
    with h5py.File(path, 'r+') as saved:
        source = json.loads(saved.attrs['identity'])
        source.update(model=metadata, code_version=PathAuditConfig.version.unique_id48,
            implementation_sha256={}, reference_variances_sha256=hashlib.sha256(reference.tobytes()).hexdigest())
        identity = json.dumps(source, sort_keys=True)
        saved.attrs.modify('identity', identity)
    declared = json.loads(declaration.read_text())
    declared['source_population_identity_sha256'] = hashlib.sha256(identity.encode()).hexdigest()
    declaration.write_text(json.dumps(declared))
    folds = (tmp_path/'fold-0', tmp_path/'fold-1')
    for fold, directory in enumerate(folds):
        fit_validation_selected_fold(path=path, output=directory, fold=fold,
            declaration=declaration, device='cpu', **controls)
    protocol = tmp_path/'PROTOCOL.md'
    protocol.write_text('Synthetic validation selection; independent fresh confirmation only.\n')
    plan = tmp_path/'evaluation-plan.json'
    plan.write_text(json.dumps({'schema': 'gromm2-terminal-confirmation-plan-v1',
        'protocol': str(protocol), 'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
        'semantic_sha256': terminal_semantic_hashes(), 'familywise_endpoint_count': 12,
        'selection_declaration': str(declaration),
        'selection_declaration_sha256': hashlib.sha256(declaration.read_bytes()).hexdigest(),
        'variants': {'eight_step': {'paths': 32, 'batch_size': 8, 'substeps': 8, 'seed': 72},
            'one_step': {'paths': 32, 'batch_size': 8, 'substeps': 1, 'seed': 73}}}))
    return folds, plan


@pytest.fixture
def frozen_pair(fitted_pair, tmp_path):
    folds, plan = fitted_pair
    output = tmp_path/'frozen'
    freeze_selected_actions(folds=folds, output=output, evaluation_plan=plan, variant='eight_step')
    return output/'selected-policies.json'


@pytest.fixture
def streaming_pair(frozen_pair, monkeypatch):
    monkeypatch.setattr(selected_audit_inference, 'check_resource_headroom', lambda: None)
    metadata = json.loads(frozen_pair.read_text())['source_population_identity']['model']
    arrays = {'loadings': np.zeros((1, 1, 3)), 'expiries_y': np.array([5/252])}
    monkeypatch.setattr(selected_audit_inference, 'load_model', lambda *args: (arrays, metadata))

    def simulator(*, invalid=False):
        def simulate(normals, background):
            gains = .01*np.asarray(normals)[:, 0, :, :3].transpose(1, 0, 2)
            count, days, _ = gains.shape
            features = np.zeros((count, days, 3))
            features[:, 1:, 1] = gains[:, :-1, 0]
            health = np.zeros((count, days), dtype=np.uint8)
            if invalid:
                health[0, 1] = 1
            return features, gains, np.ones((count, days+1)), gains.sum(axis=1), np.zeros((days, 7)), health
        return simulate

    monkeypatch.setattr(selected_audit_inference, 'path_function', lambda **kwargs: simulator())
    return frozen_pair, simulator


def test_both_full_objectives_select_and_restart_without_complementary_population(population, tmp_path):
    path, declaration, controls = population
    with h5py.File(path, 'r+') as saved:
        saved['features'][16:], saved['returns'][16:] = float('nan'), float('nan')
    output = tmp_path/'fold-0'
    result = fit_validation_selected_fold(path=path, output=output, fold=0,
        declaration=declaration, device='cpu', **controls)
    assert result['outer_holdout_read'] is False
    assert result['inference_performed'] is False
    assert result['inner_training_paths']+result['inner_validation_paths'] == 16
    assert result['status'] == 'both independent full-class selections complete; fresh confirmation pending'
    assert set(result['diagnostics']) == {'squared', 'entropy'}
    with np.load(output/'normalization.npz', allow_pickle=False) as saved:
        assert np.array_equal(saved['inner_training_mask'][::2], saved['inner_training_mask'][1::2])
    first = {}
    for name, fitted in result['diagnostics'].items():
        assert fitted['status'] == 'selection_complete'
        first[name] = torch.load(fitted['result']['selected_weights'], weights_only=False)['state_dict']
    repeated = fit_validation_selected_fold(path=path, output=output, fold=0,
        declaration=declaration, device='cpu', **controls)
    for name, fitted in repeated['diagnostics'].items():
        weights = torch.load(fitted['result']['selected_weights'], weights_only=False)['state_dict']
        for parameter, expected in first[name].items():
            torch.testing.assert_close(weights[parameter], expected, rtol=0., atol=0.)


def test_changed_selection_declaration_rejected_before_population_upload(population, tmp_path, monkeypatch):
    path, declaration, controls = population
    settings = json.loads(declaration.read_text())
    settings['controls']['selection']['interval'] += 1
    declaration.write_text(json.dumps(settings))

    def reject_upload(*args, **kwargs):
        pytest.fail('invalid selection declaration reached population upload')

    monkeypatch.setattr(validation_selected_audit, '_resident_fold', reject_upload)
    with pytest.raises(ValueError, match='declaration controls'):
        fit_validation_selected_fold(path=path, output=tmp_path/'invalid', fold=0,
            declaration=declaration, device='cpu', **controls)


def test_initializer_provenance_edit_is_rejected_by_native_identity(population, tmp_path):
    path, declaration, controls = population
    output = tmp_path/'fold-0'
    fit_validation_selected_fold(path=path, output=output, fold=0,
        declaration=declaration, device='cpu', **controls)
    report_path = output/'training-report.json'
    report = json.loads(report_path.read_text())
    report['diagnostics']['squared']['initializer']['provenance'] = {'modified': True}
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='identity or declared controls'):
        fit_validation_selected_fold(path=path, output=output, fold=0,
            declaration=declaration, device='cpu', **controls)


def test_interrupted_resume_cannot_leave_a_complete_fold_status(population, tmp_path, monkeypatch):
    path, declaration, controls = population
    output = tmp_path/'fold-0'
    fit_validation_selected_fold(path=path, output=output, fold=0,
        declaration=declaration, device='cpu', **controls)

    def interrupt(**kwargs):
        raise InterruptedError('stopped after producer committed in-progress status')

    monkeypatch.setattr(validation_selected_audit, 'fit_validation_selected_action', interrupt)
    with pytest.raises(InterruptedError):
        fit_validation_selected_fold(path=path, output=output, fold=0,
            declaration=declaration, device='cpu', **controls)
    report = json.loads((output/'training-report.json').read_text())
    assert report['status'] == 'in_progress; no frozen actions or inference'
    assert report['diagnostics']['squared']['status'] == 'fitting'


def test_freeze_requires_all_four_and_loads_exact_selected_not_terminal_states(fitted_pair, tmp_path, monkeypatch):
    folds, plan = fitted_pair
    monkeypatch.setattr(h5py, 'File', lambda *args, **kwargs: pytest.fail('freezing must not read path arrays'))
    output = tmp_path/'frozen'
    frozen = freeze_selected_actions(folds=folds, output=output, evaluation_plan=plan, variant='eight_step')
    repeated = freeze_selected_actions(folds=folds, output=output, evaluation_plan=plan, variant='eight_step')
    assert repeated == frozen
    assert frozen['outer_or_fresh_evaluation_arrays_read'] is False
    manifest = output/'selected-policies.json'
    loaded, actions = load_selected_actions(manifest, device='cpu')
    assert set(actions) == set(ENDPOINTS[:4])
    for fold in frozen['folds']:
        for name, policy in fold['policies'].items():
            saved = torch.load(policy['weights'], weights_only=False)
            action = actions[f'fold-{fold["fold"]}/{name}']
            assert action.mean.dtype == torch.float32
            assert action.model.fdtype == torch.float64
            assert saved['selected'] == policy['selected']
            for parameter, value in action.model.state_dict().items():
                torch.testing.assert_close(value, saved['state_dict'][parameter], rtol=0., atol=0.)
    with pytest.raises(ValueError, match='unknown frozen terminal-action schema'):
        load_terminal_actions(manifest, device='cpu')


def test_incomplete_selection_cannot_freeze(fitted_pair, tmp_path):
    folds, plan = fitted_pair
    path = folds[1]/'training-report.json'
    report = json.loads(path.read_text())
    report['diagnostics']['entropy']['status'] = 'fitting'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='incomplete full-class selection'):
        freeze_selected_actions(folds=folds, output=tmp_path/'incomplete', evaluation_plan=plan, variant='eight_step')
    assert not (tmp_path/'incomplete/selected-policies.json').exists()


def test_inaccurate_terminal_label_is_rejected_even_with_updated_artifact_hash(frozen_pair):
    frozen = json.loads(frozen_pair.read_text())
    policy = frozen['folds'][0]['policies']['squared']
    path = Path(policy['weights'])
    saved = torch.load(path, weights_only=False)
    saved['policy_state_source'] = 'terminal trained model, not best_state'
    torch.save(saved, path)
    policy['weights_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    frozen_pair.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match='interpretation changed'):
        load_selected_actions(frozen_pair, device='cpu')


def test_duplicate_policy_entries_cannot_masquerade_as_two_fitted_halves(frozen_pair):
    frozen = json.loads(frozen_pair.read_text())
    frozen['folds'][1]['policies'] = deepcopy(frozen['folds'][0]['policies'])
    frozen_pair.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match='normalization differs|different fitted fold'):
        load_selected_actions(frozen_pair, device='cpu')


def test_frozen_simulator_cannot_change_while_keeping_declaration_digest(frozen_pair):
    frozen = json.loads(frozen_pair.read_text())
    frozen['source_population_identity']['config']['days'] += 1
    frozen_pair.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match='fitted development half'):
        load_selected_actions(frozen_pair, device='cpu')


def test_swapping_entire_fold_record_cannot_bypass_endpoint_binding(frozen_pair):
    frozen = json.loads(frozen_pair.read_text())
    frozen['folds'][1] = deepcopy(frozen['folds'][0])
    frozen['folds'][1]['fold'] = 1
    frozen_pair.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match='fitted development half'):
        load_selected_actions(frozen_pair, device='cpu')


def test_selected_stream_reuses_original_unit_score_and_pair_statistic_functions(streaming_pair, tmp_path):
    manifest, simulator = streaming_pair
    output = tmp_path/'evaluation'
    result = evaluate_fresh_selected(manifest=manifest, output=output, seed=72, paths=32, backend='cpu')
    assert result['status'] == 'complete'
    assert result['paths_dropped'] == 0
    with h5py.File(output/'path_scores.h5') as saved:
        scores = saved['scores'][:]
        assert int(saved.attrs['completed_paths']) == 32
    assert result['diagnostics'] == summarize_terminal_scores(scores)
    repeated = evaluate_fresh_selected(manifest=manifest, output=output, seed=72, paths=32, backend='cpu')
    assert repeated == result
    _, actions = load_selected_actions(manifest, device='cpu')
    config, arrays, eligible, simulate = selected_audit_inference._selected_population(
        json.loads(manifest.read_text()), seed=72, paths=32)
    normals, background = selected_audit_inference.normal_block(config=config, start=0, count=8, factors=3)
    features, gains = simulate(normals, background)[:2]
    expected = score_terminal_batch(features=features, gains=gains, eligible=eligible, actions=actions)
    np.testing.assert_allclose(scores[:8], expected, rtol=2.e-6, atol=1.e-8)


def test_selected_stream_invalid_paths_are_not_censored(streaming_pair, tmp_path, monkeypatch):
    manifest, simulator = streaming_pair
    monkeypatch.setattr(selected_audit_inference, 'path_function', lambda **kwargs: simulator(invalid=True))
    result = evaluate_fresh_selected(manifest=manifest, output=tmp_path/'invalid', seed=72, paths=32, backend='cpu')
    assert result['status'].startswith('unavailable')
    assert result['invalid_paths'] == 4
    assert result['paths_dropped'] == 0
    assert 'diagnostics' not in result


def test_selected_stream_interrupt_resumes_exact_committed_pairs(streaming_pair, tmp_path, monkeypatch):
    manifest, simulator = streaming_pair
    original = selected_audit_inference.score_terminal_batch

    def interrupt(**kwargs):
        result = original(**kwargs)
        os.kill(os.getpid(), signal.SIGINT)
        return result

    output = tmp_path/'resumed'
    monkeypatch.setattr(selected_audit_inference, 'score_terminal_batch', interrupt)
    with pytest.raises(InterruptedError):
        evaluate_fresh_selected(manifest=manifest, output=output, seed=72, paths=32, backend='cpu')
    with h5py.File(output/'path_scores.h5') as saved:
        assert int(saved.attrs['completed_paths']) == 8
    monkeypatch.setattr(selected_audit_inference, 'score_terminal_batch', original)
    resumed = evaluate_fresh_selected(manifest=manifest, output=output, seed=72, paths=32, backend='cpu')
    baseline = evaluate_fresh_selected(manifest=manifest, output=tmp_path/'baseline', seed=72, paths=32, backend='cpu')
    assert resumed == baseline
    with h5py.File(output/'path_scores.h5') as left, h5py.File(tmp_path/'baseline/path_scores.h5') as right:
        np.testing.assert_array_equal(left['scores'][:], right['scores'][:])


@pytest.mark.parametrize('field,value', [('seed', 71), ('seed', 73), ('paths', 16)])
def test_selected_stream_rejects_undeclared_confirmation_parameters(streaming_pair, tmp_path, field, value):
    manifest, simulator = streaming_pair
    parameters = dict(manifest=manifest, output=tmp_path/'invalid', seed=72, paths=32, backend='cpu')
    parameters[field] = value
    with pytest.raises(ValueError, match='differs from its declaration'):
        evaluate_fresh_selected(**parameters)
    assert not (tmp_path/'invalid').exists()
