from __future__ import annotations

import json
import os
import signal
from dataclasses import asdict, replace

import h5py
import numpy as np
import pytest
import torch

from quantdata.gromm2.conditional_audit import (
    ActionNetworkConfig,
    IndependentLocalActions,
    LocalActionConfig,
    LocalActionStep,
)
from quantdata.gromm2.local_audit_training import (
    LocalAuditProtocol,
    LocalAuditTrainConfig,
    _normalization,
    _path_identity,
    _restore_controls,
    effective_batch_size,
    internal_training_mask,
    main,
    run_conditional_audit,
    train_fold,
)
from quantdata.torch import BaseEnvironment


def path_population(path, *, paths: int = 32) -> None:
    generator = np.random.default_rng(1)
    features = generator.normal(size=(paths, 5, 3))
    returns = generator.normal(0., .01, size=(paths, 5, 2))
    returns[..., 0] += .003 * features[..., 0]
    with h5py.File(path, 'w') as stream:
        stream.attrs['identity'] = json.dumps({'schema': 'test', 'seed': 1})
        stream.attrs['completed_paths'] = paths
        stream.create_dataset('features', data=features.astype('f4'))
        stream.create_dataset('returns', data=returns.astype('f4'))
        stream.create_dataset('expiry_days', data=[5])
        stream.create_dataset('strikes', data=[[1.]])


def protocol() -> LocalAuditProtocol:
    return LocalAuditProtocol(network=ActionNetworkConfig(num_depth=1, num_width=4))


def training() -> LocalAuditTrainConfig:
    return LocalAuditTrainConfig(device='cpu', precision=64, batch_size=7, evaluation_batch_size=16,
        upload_paths=4, affine_epochs=1, minimum_full_epochs=2, maximum_full_epochs=4,
        convergence_patience=1, absolute_improvement=1.e9, minimum_learning_rate=2.5e-4,
        compile=False, use_cuda_graphs=False)


def test_internal_selection_keeps_pairs_and_has_no_outer_holdout_access(tmp_path) -> None:
    path = tmp_path / 'paths.h5'
    path_population(path)
    config = protocol()
    mask = internal_training_mask(16, protocol=config, fold=0)
    np.testing.assert_array_equal(mask[::2], mask[1::2])
    assert np.sum(mask) >= 4 and np.sum(~mask) >= 4
    first = _normalization(path, outer=slice(0, 16), mask=mask, protocol=config,
        output=tmp_path / 'first.npz', identity='test', upload_paths=4)
    with h5py.File(path, 'r+') as stream:
        stream['features'][16:] = 1.e8
        stream['returns'][16:] = 1.e8
        expected = np.asarray(stream['features'][:16])[mask].reshape(-1, 3).mean(axis=0)
    second = _normalization(path, outer=slice(0, 16), mask=mask, protocol=config,
        output=tmp_path / 'second.npz', identity='test', upload_paths=4)
    np.testing.assert_allclose(first['mean'], expected, atol=1.e-8)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


@pytest.mark.parametrize('field', ['path_validity', 'batch_diagnostic_validity'])
def test_conditional_entrypoint_rejects_invalid_population_metadata(tmp_path, field: str) -> None:
    path = tmp_path / 'paths.h5'
    path_population(path)
    with h5py.File(path, 'r+') as stream:
        stream.create_dataset(field, data=[True, False])
    with pytest.raises(ValueError, match='numerically invalid complete population'):
        _path_identity(path)


def test_production_batch_can_avoid_partial_graph_recaptures() -> None:
    assert effective_batch_size(5_625_000, 16_384) == 15_625
    assert effective_batch_size(625_000, 32_768) == 31_250
    assert effective_batch_size(17, 6) == 6
    assert effective_batch_size(4, 20) == 4


def test_cli_forwards_explicit_batch_and_schedule_controls(monkeypatch) -> None:
    received = {}
    monkeypatch.setattr('sys.argv', ['local-audit', '--paths', 'paths.h5', '--output-directory', 'output',
        '--device', 'cpu', '--batch-size', '1024', '--learning-rate', '1e-5',
        '--minimum-learning-rate', '1e-8', '--learning-rate-patience', '20',
        '--convergence-patience', '30', '--affine-epochs', '50',
        '--minimum-full-epochs', '100', '--maximum-full-epochs', '500'])
    monkeypatch.setattr('quantdata.gromm2.local_audit_training.run_conditional_audit',
        lambda **values: received.update(values))
    main()
    settings = received['train_config']
    assert settings.batch_size == 1024
    assert settings.learning_rate == 1.e-5
    assert settings.minimum_learning_rate == 1.e-8
    assert settings.learning_rate_patience == 20
    assert settings.convergence_patience == 30
    assert settings.affine_epochs == 50
    assert settings.minimum_full_epochs == 100
    assert settings.maximum_full_epochs == 500


def test_two_fold_audit_finishes_and_load_only_does_not_acquire_a_training_lock(tmp_path, monkeypatch) -> None:
    path = tmp_path / 'paths.h5'
    path_population(path)
    monkeypatch.setattr('quantdata.gromm2.local_audit_training.check_resource_headroom', lambda: None)
    output = tmp_path / 'audit'
    result = run_conditional_audit(path=path, output=output, protocol=protocol(), train_config=training())
    assert set(result['diagnostics']) == {'entropy', 'squared'}
    assert (output / 'conditional_report.json').exists()
    for fold in result['folds']:
        assert fold['selection']['epoch'] == 3
        assert fold['selection']['outer_holdout_used_for_selection'] is False
        assert fold['selection']['stop_reason'] == 'internal_validation_plateau'
        for entry in fold['diagnostics'].values():
            assert entry['holdout']['frozen_action_return']['independent_pairs'] == 8
            assert entry['holdout']['mean_squared_action'] == 0.
    monkeypatch.setattr('quantdata.gromm2.local_audit_training.FileLock',
        lambda *args, **kwargs: pytest.fail('read-only loading must not acquire a training lock'))
    loaded = train_fold(path=path, output=output / 'fold-0', fold=0, protocol=protocol(), train_config=training(), load_only=True)
    assert loaded['epoch'] == 3


def test_gpu_fold_handoff_releases_cached_allocations_before_loading_next_fold(tmp_path, monkeypatch) -> None:
    events = []

    def fitted_fold(*, fold, **kwargs):
        events.append(('train', fold))
        return {'fold': fold, 'stop_reason': 'internal_validation_plateau'}

    def evaluated_fold(*, report, **kwargs):
        events.append(('evaluate', report['fold']))
        statistic = {'paths': 16, 'mean': 0., 'standard_error': 1., 'action_conditional_statistic': 0.}
        return {'fold': report['fold'], 'selection': report, 'diagnostics': {
            name: {'holdout': {'frozen_action_return': statistic}} for name in ('entropy', 'squared')}}

    monkeypatch.setattr('quantdata.gromm2.local_audit_training.train_fold', fitted_fold)
    monkeypatch.setattr('quantdata.gromm2.local_audit_training.evaluate_fold', evaluated_fold)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: events.append('release-unused-VRAM'))
    result = run_conditional_audit(path=tmp_path / 'not-read.h5', output=tmp_path / 'audit',
        protocol=protocol(), train_config=replace(training(), device='cuda'))
    assert set(result['diagnostics']) == {'entropy', 'squared'}
    assert events == [('train', 0), ('evaluate', 0), 'release-unused-VRAM',
                      ('train', 1), ('evaluate', 1), 'release-unused-VRAM']


def test_mid_epoch_checkpoint_restores_optimizers_rng_and_observation_order(tmp_path, monkeypatch) -> None:
    path = tmp_path / 'paths.h5'
    path_population(path)
    monkeypatch.setattr('quantdata.gromm2.local_audit_training.check_resource_headroom', lambda: None)
    baseline = train_fold(path=path, output=tmp_path / 'baseline', fold=0, protocol=protocol(), train_config=training())
    original_step = LocalActionStep.__call__
    completed = 0

    def interrupt_after_second_update(self, *args, **kwargs):
        nonlocal completed
        result = original_step(self, *args, **kwargs)
        completed += 1
        if completed == 2:
            os.kill(os.getpid(), signal.SIGINT)
        return result

    monkeypatch.setattr(LocalActionStep, '__call__', interrupt_after_second_update)
    with pytest.raises(InterruptedError):
        train_fold(path=path, output=tmp_path / 'resumed', fold=0, protocol=protocol(), train_config=training())
    interrupted = train_fold(path=path, output=tmp_path / 'resumed', fold=0, protocol=protocol(), train_config=training(), load_only=True)
    partial = torch.load(interrupted['checkpoint'], map_location='cpu', weights_only=False)
    assert partial['state']['epoch'] == 0 and partial['state']['next_batch'] > 0
    assert partial['state']['order'] is not None
    monkeypatch.setattr(LocalActionStep, '__call__', original_step)
    resumed = train_fold(path=path, output=tmp_path / 'resumed', fold=0, protocol=protocol(), train_config=replace(training(), upload_paths=8))
    first = torch.load(baseline['checkpoint'], map_location='cpu', weights_only=False)
    second = torch.load(resumed['checkpoint'], map_location='cpu', weights_only=False)
    assert first['state']['history'] == second['state']['history']
    for name in first['model']:
        torch.testing.assert_close(first['model'][name], second['model'][name], rtol=0., atol=0.)
    for name in first['optimizers']:
        for parameter, values in first['optimizers'][name]['state'].items():
            for key, value in values.items():
                torch.testing.assert_close(value, second['optimizers'][name]['state'][parameter][key], rtol=0., atol=0.)
    torch.testing.assert_close(first['generator_rng'], second['generator_rng'], rtol=0., atol=0.)


def test_actions_detect_conditional_drift_even_with_zero_unconditional_means(tmp_path, monkeypatch) -> None:
    path = tmp_path / 'conditional-drift.h5'
    path_population(path, paths=128)
    generator = np.random.default_rng(32)
    half = generator.normal(size=(64, 5, 3))
    features = np.stack((half, -half), axis=1).reshape(128, 5, 3)
    returns = np.stack((.1 * features[..., 0], .02 * features[..., 1]), axis=-1)
    np.testing.assert_allclose(returns.mean(axis=(0, 1)), 0., atol=1.e-16)
    with h5py.File(path, 'r+') as stream:
        stream['features'][:] = features
        stream['returns'][:] = returns
    monkeypatch.setattr('quantdata.gromm2.local_audit_training.check_resource_headroom', lambda: None)
    settings = replace(training(), batch_size=256, evaluation_batch_size=256, affine_epochs=10,
        minimum_full_epochs=10, maximum_full_epochs=100, learning_rate=.02,
        minimum_learning_rate=.02, absolute_improvement=.005, convergence_patience=5)
    report = run_conditional_audit(path=path, output=tmp_path / 'audit', protocol=protocol(), train_config=settings)
    for fold in report['folds']:
        for diagnostic in fold['diagnostics'].values():
            score = diagnostic['holdout']['frozen_action_return']
            assert score['mean'] > 0.
            assert score['action_conditional_statistic'] > 3.
            assert diagnostic['holdout']['mean_squared_action'] > 0.


def test_changed_resume_controls_are_applied_and_reported(tmp_path, monkeypatch) -> None:
    path = tmp_path / 'paths.h5'
    path_population(path)
    monkeypatch.setattr('quantdata.gromm2.local_audit_training.check_resource_headroom', lambda: None)
    original = train_fold(path=path, output=tmp_path / 'audit', fold=0, protocol=protocol(), train_config=training())
    changed = replace(training(), learning_rate=.002, minimum_learning_rate=.002,
        learning_rate_patience=2, batch_size=4, minimum_full_epochs=4, maximum_full_epochs=8)
    resumed = train_fold(path=path, output=tmp_path / 'audit', fold=0, protocol=protocol(), train_config=changed)
    assert resumed['epoch'] > original['epoch']
    assert resumed['train_config']['batch_size'] == 4
    for controls in resumed['effective_optimizer_controls'].values():
        assert controls['learning_rates'] == [.002]
        assert controls['minimum_learning_rates'] == [.002]
        assert controls['scheduler_patience'] == 2


def test_stopping_requires_training_at_learning_rate_floor(tmp_path, monkeypatch) -> None:
    path = tmp_path / 'paths.h5'
    path_population(path)
    monkeypatch.setattr('quantdata.gromm2.local_audit_training.check_resource_headroom', lambda: None)
    settings = replace(training(), learning_rate=.001, minimum_learning_rate=.000125,
        learning_rate_patience=1, maximum_full_epochs=20, convergence_patience=2)
    report = train_fold(path=path, output=tmp_path / 'audit', fold=0, protocol=protocol(), train_config=settings)
    assert report['stop_reason'] == 'internal_validation_plateau'
    assert report['epoch'] > settings.affine_epochs + settings.minimum_full_epochs
    checkpoint = torch.load(report['checkpoint'], map_location='cpu', weights_only=False)
    for name in ('entropy', 'squared'):
        floor_epoch = checkpoint['state']['floor_since'][name]
        assert floor_epoch is not None
        assert report['epoch'] - floor_epoch >= settings.convergence_patience
        assert report['effective_optimizer_controls'][name]['learning_rates'] == [.000125]


def control_fixture(settings: LocalAuditTrainConfig, *, epoch: int = 4, next_batch: int = 0):
    models = {name: torch.nn.Linear(1, 1) for name in ('entropy', 'squared')}
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=.0001) for name, model in models.items()}
    schedulers = {name: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        min_lr=settings.minimum_learning_rate, patience=settings.learning_rate_patience)
        for name, optimizer in optimizers.items()}
    state = {'epoch': epoch, 'next_batch': next_batch,
        **{key: {name: value for name in models} for key, value in (
            ('best_loss', 0.), ('stopped', True), ('last_improvement', 0), ('floor_since', None))}}
    return optimizers, schedulers, state


def test_unchanged_resume_preserves_a_decayed_learning_rate() -> None:
    settings = replace(training(), learning_rate=.001, minimum_learning_rate=.00001)
    optimizers, schedulers, state = control_fixture(settings)
    assert not _restore_controls(optimizers=optimizers, schedulers=schedulers, state=state,
        previous=asdict(settings), requested=settings)
    assert all(optimizer.param_groups[0]['lr'] == .0001 for optimizer in optimizers.values())


def test_raised_resume_minimum_is_enforced() -> None:
    settings = replace(training(), learning_rate=.001, minimum_learning_rate=.00001)
    optimizers, schedulers, state = control_fixture(settings)
    assert _restore_controls(optimizers=optimizers, schedulers=schedulers, state=state,
        previous=asdict(settings), requested=replace(settings, minimum_learning_rate=.0005))
    assert all(optimizer.param_groups[0]['lr'] == .0005 for optimizer in optimizers.values())


def test_interrupted_first_deep_epoch_cannot_revert_to_affine() -> None:
    settings = training()
    optimizers, schedulers, state = control_fixture(settings, epoch=settings.affine_epochs, next_batch=1)
    with pytest.raises(ValueError, match='trained deep phase'):
        _restore_controls(optimizers=optimizers, schedulers=schedulers, state=state,
            previous=asdict(settings), requested=replace(settings, affine_epochs=3))


def test_operational_resume_does_not_train_an_extra_epoch(tmp_path, monkeypatch) -> None:
    path = tmp_path / 'paths.h5'
    path_population(path)
    monkeypatch.setattr('quantdata.gromm2.local_audit_training.check_resource_headroom', lambda: None)
    original = train_fold(path=path, output=tmp_path / 'audit', fold=0, protocol=protocol(), train_config=training())
    resumed = train_fold(path=path, output=tmp_path / 'audit', fold=0, protocol=protocol(),
        train_config=replace(training(), upload_paths=8))
    assert original == resumed


@pytest.mark.parametrize('first_mean,status,expected_statistic', [
    (0., 'ok', 2.), (1., 'degenerate_nonzero_fold', None)])
def test_mixed_degenerate_folds_preserve_weights_and_validity(tmp_path, monkeypatch,
        first_mean: float, status: str, expected_statistic: float | None) -> None:
    def fitted_fold(*, fold, **kwargs):
        return {'fold': fold, 'stop_reason': 'internal_validation_plateau'}

    def evaluated_fold(*, report, **kwargs):
        statistic = {'paths': 16, 'mean': first_mean if report['fold'] == 0 else .2,
            'standard_error': 0. if report['fold'] == 0 else .1,
            'action_conditional_statistic': None if report['fold'] == 0 else 2.}
        return {'fold': report['fold'], 'selection': report, 'diagnostics': {
            name: {'holdout': {'frozen_action_return': statistic}} for name in ('entropy', 'squared')}}

    monkeypatch.setattr('quantdata.gromm2.local_audit_training.train_fold', fitted_fold)
    monkeypatch.setattr('quantdata.gromm2.local_audit_training.evaluate_fold', evaluated_fold)
    result = run_conditional_audit(path=tmp_path / 'not-read.h5', output=tmp_path / 'audit',
        protocol=protocol(), train_config=training())
    for diagnostic in result['diagnostics'].values():
        assert diagnostic['test_status'] == status
        assert diagnostic['cross_fitted_action_statistic'] == expected_statistic
        assert diagnostic['cross_fitted_mean'] == pytest.approx(.5 * first_mean + .1)
        assert diagnostic['cross_fitted_standard_error_upper_bound_estimate'] == .05


@pytest.mark.skipif(os.environ.get('CUDA_VISIBLE_DEVICES', '') in {'', '-1'}, reason='requires a gpuctl-allocated GPU')
def test_captured_local_objectives_match_eager_across_both_training_phases() -> None:
    env = BaseEnvironment(device='cuda', precision=64)
    config = LocalActionConfig(label='captured-local-test', feature_count=3, instrument_count=2,
        action_scale=.01, network=ActionNetworkConfig(num_depth=1, num_width=4))
    torch.cuda.manual_seed(19)
    captured, eager = IndependentLocalActions(env, config=config), IndependentLocalActions(env, config=config)
    eager.load_state_dict(captured.state_dict())
    optimizers = [{name: torch.optim.Adam(policy.parameters(), lr=.001, capturable=True)
        for name, policy in model.policies.items()} for model in (captured, eager)]
    generator = torch.Generator(device='cuda').manual_seed(22)
    for full in (False, True):
        for model in (captured, eager):
            for policy in model.policies.values():
                policy.deep.requires_grad_(full)
                policy.affine_only = not full
        steps = [LocalActionStep(model, optimizer, gradient_clip=5., compile=False, use_cuda_graphs=graph)
            for model, optimizer, graph in zip((captured, eager), optimizers, (True, False), strict=True)]
        for observations in (8, 8, 5):
            features = torch.randn((observations, 3), generator=generator, device='cuda', dtype=torch.float64)
            returns = torch.randn((observations, 2), generator=generator, device='cuda', dtype=torch.float64)
            eligible = torch.ones((observations, 2), device='cuda', dtype=torch.bool)
            eligible[0, 1] = False
            for step in steps:
                step(features, returns, eligible, ('entropy', 'squared'))
        for name, value in captured.state_dict().items():
            torch.testing.assert_close(value, eager.state_dict()[name], rtol=2.e-8, atol=2.e-10)
