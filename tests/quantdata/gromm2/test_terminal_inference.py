from __future__ import annotations

import hashlib
import json
import os
import signal
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from scipy.special import logsumexp
from scipy.stats import norm

from quantdata.gromm2 import terminal_audit, terminal_inference
from quantdata.gromm2.conditional_audit import (
    ActionNetworkConfig,
    AffineDeepAction,
    LocalActionConfig,
)
from quantdata.gromm2.path_audit import PathAuditConfig
from quantdata.gromm2.terminal_inference import (
    ENDPOINTS,
    evaluate_fresh_terminal,
    load_terminal_actions,
    score_terminal_batch,
    summarize_terminal_scores,
)
from quantdata.torch import BaseEnvironment


@pytest.fixture
def frozen_actions(tmp_path):
    root = tmp_path / 'frozen'
    root.mkdir()
    config = LocalActionConfig(label='test', feature_count=3, instrument_count=3, action_scale=.2,
        network=ActionNetworkConfig(num_width=4, num_depth=1))
    folds = []
    for fold in range(2):
        policies = {}
        for index, name in enumerate(('entropy', 'squared')):
            model = AffineDeepAction(BaseEnvironment(device='cpu', precision=64), config=config)
            with torch.no_grad():
                model.affine.I0.weight[0, 1] = .4 + index
                model.affine.I0.bias[:] = torch.tensor([.3, -.2, .1]) * (1. if fold == 0 else -.5)
            weights = root / f'fold-{fold}-{name}.pt'
            torch.save({'model_config': asdict(config), 'code_version': LocalActionConfig.version.unique_id48,
                'user_version': config.user_version, 'diagnostic': name, 'state_dict': model.state_dict(),
                'policy_state_source': 'terminal trained model, not best_state'}, weights)
            normalization = root / f'fold-{fold}-{name}.npz'
            np.savez(normalization, mean=[-.1, 0., .2], scale=[1., 2., .5], action_scale=config.action_scale)
            policies[name] = {'weights': str(weights), 'normalization': str(normalization),
                'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
                'normalization_sha256': hashlib.sha256(normalization.read_bytes()).hexdigest()}
        folds.append({'fold': fold, 'policies': policies})
    protocol = root / 'protocol.md'
    protocol.write_text('Synthetic confirmation declaration before scoring.\n')
    semantic = terminal_audit.terminal_semantic_hashes()
    protocol_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()
    variants = {'one_step': {'substeps': 1, 'seed': 72, 'paths': 8, 'batch_size': 4},
                'eight_step': {'substeps': 8, 'seed': 73, 'paths': 8, 'batch_size': 4}}
    plan = root / 'evaluation-plan.json'
    plan.write_text(json.dumps({'schema': 'gromm2-terminal-confirmation-plan-v1',
        'protocol': str(protocol), 'protocol_sha256': protocol_hash, 'semantic_sha256': semantic,
        'variants': variants, 'familywise_endpoint_count': 12}))
    manifest = root / 'terminal-policies.json'
    manifest.write_text(json.dumps({'schema': 'gromm2-terminal-frozen-actions-v2', 'folds': folds,
        'protocol': str(protocol), 'protocol_sha256': protocol_hash,
        'evaluation_plan': str(plan), 'evaluation_plan_sha256': hashlib.sha256(plan.read_bytes()).hexdigest(),
        'evaluation_variant': 'one_step', 'evaluation_settings': variants['one_step'], 'semantic_sha256': semantic}))
    return manifest


def test_batch_scores_match_original_unit_actions_and_ensemble_nonlinearity(frozen_actions):
    frozen, actions = load_terminal_actions(frozen_actions, device='cpu')
    features = np.array([[[1., 2., 3.], [.5, 1., -1.]]])
    gains = np.array([[[.01, .02, -.01], [.03, -.01, .02]]])
    eligible = np.array([[True, True, True], [True, False, True]])
    actual = score_terminal_batch(features=features, gains=gains, eligible=eligible, actions=actions)
    holdings = {}
    for name, action in actions.items():
        observed = torch.tensor(features.astype('f4'), dtype=torch.float64)
        holdings[name] = (.2 * action.model((observed - action.mean) / action.scale, torch.tensor(eligible))).detach().numpy()
    for name in ('entropy', 'squared'):
        holdings[f'ensemble/{name}'] = .5 * (holdings[f'fold-0/{name}'] + holdings[f'fold-1/{name}'])
    for index, name in enumerate(ENDPOINTS):
        gain = np.sum(holdings[name] * gains.astype('f4'), axis=-1)
        squared = np.sum(holdings[name]**2, axis=-1)
        expected = [gain.mean(), squared.mean(), float(logsumexp(-gain + .5 * squared) - np.log(2)),
                    float((gain - .5 * squared - .5 * gain**2).mean())]
        np.testing.assert_allclose(actual[0, index], expected, rtol=1.e-12, atol=1.e-15)
    assert actual[0, 4, 1] != pytest.approx(.5 * (actual[0, 0, 1] + actual[0, 2, 1]))


def test_score_summary_uses_complete_antithetic_pairs_and_standard_error():
    scores = np.zeros((8, 6, 4))
    scores[:, :, 0] = np.arange(8)[:, None]
    summary = summarize_terminal_scores(scores)
    pairs = np.array([.5, 2.5, 4.5, 6.5])
    error = pairs.std(ddof=1) / 2
    for value in summary.values():
        assert value['mean'] == 3.5
        assert value['standard_error'] == pytest.approx(error)
        assert value['action_conditional_statistic'] == pytest.approx(3.5 / error)
        assert value['familywise_pvalue'] == min(1., 12 * 2 * norm.sf(3.5 / error))
        assert value['familywise_99_threshold'] == norm.ppf(1. - .01 / 24.)
    zero = summarize_terminal_scores(np.zeros_like(scores))
    assert zero['fold-0/entropy']['test_status'] == 'degenerate_zero_score'
    assert zero['fold-0/entropy']['action_conditional_statistic'] is None


def test_f64_policy_reuses_declared_f32_training_normalization(frozen_actions):
    manifest = json.loads(frozen_actions.read_text())
    for fold in manifest['folds']:
        for selected in fold['policies'].values():
            path = Path(selected['weights'])
            saved = torch.load(path, weights_only=False)
            saved['input_normalization_dtype'] = 'float32'
            torch.save(saved, path)
            selected['weights_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            normalization = Path(selected['normalization'])
            np.savez(normalization, mean=[0., 100000001., 0.], scale=[1., 3., 1.], action_scale=.2)
            selected['normalization_sha256'] = hashlib.sha256(normalization.read_bytes()).hexdigest()
    frozen_actions.write_text(json.dumps(manifest))
    _, actions = load_terminal_actions(frozen_actions, device='cpu')
    features = np.array([[[0., 100000008., 0.]]], dtype=np.float32)
    gains = np.array([[[1., 0., 0.]]], dtype=np.float32)
    eligible = np.ones((1, 3), dtype=bool)
    result = score_terminal_batch(features=features, gains=gains, eligible=eligible, actions=actions)
    for index, name in enumerate(ENDPOINTS[:4]):
        action = actions[name]
        assert action.mean.dtype == torch.float32
        normalized = ((torch.as_tensor(features)-action.mean)/action.scale).double()
        expected = .2*action.model(normalized, torch.as_tensor(eligible))[0, 0, 0]
        assert result[0, index, 0] == pytest.approx(float(expected.detach()), rel=1.e-13)
        changed = ((torch.as_tensor(features).double()-torch.tensor([0., 100000001., 0.], dtype=torch.float64))
            /torch.tensor([1., 3., 1.], dtype=torch.float64))
        assert not torch.equal(normalized, changed)


def test_loading_rejects_tampered_weights_and_unallocated_cuda(frozen_actions, monkeypatch):
    manifest = json.loads(frozen_actions.read_text())
    filename = manifest['folds'][0]['policies']['entropy']['weights']
    with open(filename, 'ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='differs from its manifest'):
        load_terminal_actions(frozen_actions, device='cpu')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(ValueError, match='managed CUDA'):
        load_terminal_actions(frozen_actions, device='cuda')


@pytest.fixture
def streaming_fixture(frozen_actions, tmp_path, monkeypatch):
    monkeypatch.setattr(terminal_inference, 'check_resource_headroom', lambda: None)
    config = PathAuditConfig(paths=8, batch_size=4, days=5, substeps=1, seed=71,
        retain_nonfinite=True, nms=(-.1, .1))
    calibration = tmp_path / 'calibration'
    own = calibration / 'own_dates/2025-08-29'
    own.mkdir(parents=True)
    reference = np.array([.04])
    np.savez(own / 'inputs.npz', projected_total_variances=reference)
    metadata = {'calibration_root': str(calibration), 'origin': '2025-08-29', 'background_volatility': .05}
    arrays = {'loadings': np.zeros((1, 1, 16)), 'expiries_y': np.array([5 / 252])}
    frozen = json.loads(frozen_actions.read_text())
    frozen['source_population_identity'] = {'config': asdict(config), 'model': metadata,
        'code_version': PathAuditConfig.version.unique_id48, 'implementation_sha256': {},
        'reference_variances_sha256': hashlib.sha256(reference.tobytes()).hexdigest()}
    frozen_actions.write_text(json.dumps(frozen))
    monkeypatch.setattr(terminal_inference, 'load_model', lambda *args: (arrays, metadata))

    def simulator(*, invalid=False):
        def simulate(normals, background):
            returns = np.asarray(normals)[:, 0, :, :3].transpose(1, 0, 2) * .01
            count, days, instruments = returns.shape
            features = np.zeros((count, days, 3))
            features[:, 1:, 1] = returns[:, :-1, 0]
            spots = np.ones((count, days + 1))
            health = np.zeros((count, days), dtype=np.uint8)
            if invalid:
                health[0, 1] = 1
            return features, returns, spots, returns.sum(axis=1), np.zeros((days, 7)), health
        return simulate

    monkeypatch.setattr(terminal_inference, 'path_function', lambda **kwargs: simulator())
    return frozen_actions, simulator


def test_streamed_evaluation_finishes_and_same_identity_can_resume(streaming_fixture, tmp_path):
    manifest, simulator = streaming_fixture
    output = tmp_path / 'evaluation'
    result = evaluate_fresh_terminal(manifest=manifest, output=output, seed=72, paths=8, backend='cpu')
    assert result['status'] == 'complete'
    assert result['paths_dropped'] == 0
    assert result['invalid_paths'] == 0
    assert all(value['test_status'] == 'ok' for value in result['diagnostics'].values())
    repeated = evaluate_fresh_terminal(manifest=manifest, output=output, seed=72, paths=8, backend='cpu')
    assert repeated == result
    with pytest.raises(ValueError, match='separately seeded'):
        evaluate_fresh_terminal(manifest=manifest, output=tmp_path / 'bad-seed', seed=71, paths=8, backend='cpu')


def test_invalid_population_is_not_replaced_by_valid_path_subset(streaming_fixture, tmp_path, monkeypatch):
    manifest, simulator = streaming_fixture
    monkeypatch.setattr(terminal_inference, 'path_function', lambda **kwargs: simulator(invalid=True))
    output = tmp_path / 'invalid'
    result = evaluate_fresh_terminal(manifest=manifest, output=output, seed=72, paths=8, backend='cpu')
    assert result['status'].startswith('unavailable')
    assert result['invalid_paths'] == 2
    assert result['paths_dropped'] == 0
    assert 'diagnostics' not in result
    with h5py.File(output / 'path_scores.h5') as saved:
        assert saved.attrs['completed_paths'] == 8
        assert saved['path_validity'][:].sum() == 6


def test_interrupt_resumes_from_committed_batch_without_redrawing(streaming_fixture, tmp_path, monkeypatch):
    manifest, simulator = streaming_fixture
    original = terminal_inference.score_terminal_batch
    calls = 0

    def interrupt(**kwargs):
        nonlocal calls
        result = original(**kwargs)
        calls += 1
        if calls == 1:
            os.kill(os.getpid(), signal.SIGINT)
        return result

    monkeypatch.setattr(terminal_inference, 'score_terminal_batch', interrupt)
    output = tmp_path / 'resumed'
    with pytest.raises(InterruptedError):
        evaluate_fresh_terminal(manifest=manifest, output=output, seed=72, paths=8, backend='cpu')
    with h5py.File(output / 'path_scores.h5') as saved:
        assert saved.attrs['completed_paths'] == 4
    monkeypatch.setattr(terminal_inference, 'score_terminal_batch', original)
    resumed = evaluate_fresh_terminal(manifest=manifest, output=output, seed=72, paths=8, backend='cpu')
    baseline_output = tmp_path / 'baseline'
    baseline = evaluate_fresh_terminal(manifest=manifest, output=baseline_output, seed=72, paths=8, backend='cpu')
    assert resumed == baseline
    with h5py.File(output / 'path_scores.h5') as left, h5py.File(baseline_output / 'path_scores.h5') as right:
        np.testing.assert_array_equal(left['scores'][:], right['scores'][:])


@pytest.mark.parametrize('field,value', [('seed', 74), ('paths', 12)])
def test_confirmation_parameters_cannot_be_changed(streaming_fixture, tmp_path, field, value):
    manifest, _ = streaming_fixture
    parameters = {'manifest': manifest, 'output': tmp_path / 'undeclared', 'seed': 72, 'paths': 8, 'backend': 'cpu'}
    parameters[field] = value
    with pytest.raises(ValueError, match='frozen declaration'):
        evaluate_fresh_terminal(**parameters)
    assert not (tmp_path / 'undeclared').exists()


def test_policy_semantics_changed_after_checkpoint_cannot_resume(streaming_fixture, tmp_path, monkeypatch):
    manifest, _ = streaming_fixture
    output = tmp_path / 'changed-semantics'
    original = terminal_inference.score_terminal_batch

    def interrupt(**kwargs):
        result = original(**kwargs)
        os.kill(os.getpid(), signal.SIGINT)
        return result

    monkeypatch.setattr(terminal_inference, 'score_terminal_batch', interrupt)
    with pytest.raises(InterruptedError):
        evaluate_fresh_terminal(manifest=manifest, output=output, seed=72, paths=8, backend='cpu')
    semantic = terminal_audit.terminal_semantic_hashes()
    semantic[str(Path(terminal_inference.__file__).with_name('conditional_audit.py'))] = 'changed-policy-forward'
    monkeypatch.setattr(terminal_audit, 'terminal_semantic_hashes', lambda: semantic)
    with pytest.raises(ValueError, match='semantic dependencies changed'):
        evaluate_fresh_terminal(manifest=manifest, output=output, seed=72, paths=8, backend='cpu')
    with h5py.File(output / 'path_scores.h5') as saved:
        assert saved.attrs['completed_paths'] == 4


def test_partial_initialization_is_unpublished_and_retryable(streaming_fixture, tmp_path, monkeypatch):
    manifest, _ = streaming_fixture
    output = tmp_path / 'partial-initialization'
    create = h5py.Group.create_dataset

    def fail_policy_dataset(self, name, *args, **kwargs):
        if name == 'policy_validity':
            raise OSError('injected dataset creation failure')
        return create(self, name, *args, **kwargs)

    monkeypatch.setattr(h5py.Group, 'create_dataset', fail_policy_dataset)
    with pytest.raises(OSError, match='injected'):
        evaluate_fresh_terminal(manifest=manifest, output=output, seed=72, paths=8, backend='cpu')
    assert not (output / 'path_scores.h5').exists()
    monkeypatch.setattr(h5py.Group, 'create_dataset', create)
    result = evaluate_fresh_terminal(manifest=manifest, output=output, seed=72, paths=8, backend='cpu')
    assert result['status'] == 'complete'


def test_protocol_cannot_change_after_freezing(frozen_actions):
    frozen = json.loads(frozen_actions.read_text())
    Path(frozen['protocol']).write_text('changed protocol')
    with pytest.raises(ValueError, match='protocol changed'):
        load_terminal_actions(frozen_actions, device='cpu')
