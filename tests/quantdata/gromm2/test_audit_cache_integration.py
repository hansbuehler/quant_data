from __future__ import annotations

import os
from dataclasses import replace

import h5py
import numpy as np
import pytest
import torch

from quantdata.gromm2.audit_host_cache import cache_audit_arrays
from quantdata.gromm2.conditional_audit import ActionNetworkConfig
from quantdata.gromm2.local_audit_training import (
    LocalAuditProtocol,
    LocalAuditTrainConfig,
    _cpu_tensor_block,
    _normalization,
    _resident_fold,
    internal_training_mask,
    run_conditional_audit,
)
from quantdata.torch import BaseEnvironment


def population(path) -> None:
    generator = np.random.default_rng(32)
    with h5py.File(path, 'w') as stream:
        stream.attrs['identity'] = 'raw-cache-integration-v1'
        stream.attrs['completed_paths'] = 32
        stream.create_dataset('features', data=generator.normal(size=(32, 5, 3)).astype('f4'), chunks=(4, 5, 3))
        stream.create_dataset('returns', data=generator.normal(size=(32, 5, 2)).astype('f4') * .01, chunks=(4, 5, 2))
        stream.create_dataset('expiry_days', data=[5])
        stream.create_dataset('strikes', data=[[1.]])


@pytest.fixture(autouse=True)
def resources(monkeypatch):
    monkeypatch.setattr('quantdata.gromm2.local_audit_training.check_resource_headroom', lambda: None)
    monkeypatch.setattr('quantdata.gromm2.audit_host_cache._host_cache_resources',
        lambda needed: ({'blocked': False, 'warning': False, 'report': {'checks': []}}, True))


def protocol() -> LocalAuditProtocol:
    return LocalAuditProtocol(network=ActionNetworkConfig(num_depth=1, num_width=4))


def config() -> LocalAuditTrainConfig:
    return LocalAuditTrainConfig(device='cpu', precision=64, batch_size=7, evaluation_batch_size=16,
        upload_paths=4, affine_epochs=1, minimum_full_epochs=2, maximum_full_epochs=4,
        convergence_patience=1, absolute_improvement=1.e9, compile=False, use_cuda_graphs=False)


def test_cached_normalization_and_resident_inputs_are_exactly_equal_and_do_not_mutate_raw_arrays(tmp_path):
    path = tmp_path / 'paths.h5'
    population(path)
    with h5py.File(path, 'r') as stream:
        cached = cache_audit_arrays(stream, chunk_paths=4)
    mask = internal_training_mask(16, protocol=protocol(), fold=0)
    normalized = []
    try:
        for label, cache in (('direct', None), ('cached', cached)):
            values = _normalization(path, outer=slice(0, 16), mask=mask, protocol=protocol(),
                output=tmp_path / f'{label}.npz', identity='same', upload_paths=4, cache=cache)
            normalized.append(values)
        for key in normalized[0]:
            np.testing.assert_array_equal(normalized[0][key], normalized[1][key])
        observed = []
        for cache in (None, cached):
            observed.append(_resident_fold(path, outer=slice(0, 16), normalization=normalized[0],
                env=BaseEnvironment(device='cpu', precision=64), train_config=config(), progress={}, cache=cache))
        for direct, cached_value in zip(*observed, strict=True):
            torch.testing.assert_close(direct, cached_value, rtol=0., atol=0.)
        with h5py.File(path, 'r') as stream:
            for key in ('features', 'returns'):
                np.testing.assert_array_equal(cached.block(stream, key, slice(None)), stream[key][:])
    finally:
        cached.release()


def test_full_cached_and_uncached_audits_preserve_losses_optimizer_rng_scores_and_release(tmp_path, monkeypatch):
    path = tmp_path / 'paths.h5'
    population(path)
    with h5py.File(path, 'r+') as stream:
        stream['returns'][:] = .05 * stream['features'][:, :, :2]
    training = replace(config(), affine_epochs=10, minimum_full_epochs=10, maximum_full_epochs=60,
        convergence_patience=5, learning_rate=.02, minimum_learning_rate=.02, absolute_improvement=.0005)
    caches = []
    original = cache_audit_arrays

    def record_cache(*args, **kwargs):
        result = original(*args, **kwargs)
        caches.append(result)
        return result

    monkeypatch.setattr('quantdata.gromm2.local_audit_training.cache_audit_arrays', record_cache)
    direct = run_conditional_audit(path=path, output=tmp_path / 'direct', protocol=protocol(), train_config=training)
    cached = run_conditional_audit(path=path, output=tmp_path / 'cached', protocol=protocol(),
                                 train_config=replace(training, host_cache=True))
    assert direct['diagnostics'] == cached['diagnostics']
    assert len(caches) == 1 and caches[0].bytes_cached == 0
    for first, second in zip(direct['folds'], cached['folds'], strict=True):
        assert first['selection']['history'] == second['selection']['history']
        assert first['diagnostics'] == second['diagnostics']
        assert all(entry['holdout']['mean_squared_action'] > 0. for entry in first['diagnostics'].values())
        checkpoints = [torch.load(value['selection']['checkpoint'], map_location='cpu', weights_only=False)
                       for value in (first, second)]
        torch.testing.assert_close(checkpoints[0]['generator_rng'], checkpoints[1]['generator_rng'], rtol=0., atol=0.)
        torch.testing.assert_close(checkpoints[0]['torch_rng'], checkpoints[1]['torch_rng'], rtol=0., atol=0.)
        for key, value in checkpoints[0]['model'].items():
            torch.testing.assert_close(value, checkpoints[1]['model'][key], rtol=0., atol=0.)
        for name, optimizer in checkpoints[0]['optimizers'].items():
            for parameter, state in optimizer['state'].items():
                for key, value in state.items():
                    torch.testing.assert_close(value, checkpoints[1]['optimizers'][name]['state'][parameter][key], rtol=0., atol=0.)


def test_cached_run_releases_owned_arrays_when_training_fails(tmp_path, monkeypatch):
    path = tmp_path / 'paths.h5'
    population(path)
    retained = []

    def fail(*, cache, **kwargs):
        retained.append(cache)
        assert cache.bytes_cached > 0
        raise RuntimeError('training stopped')

    monkeypatch.setattr('quantdata.gromm2.local_audit_training.train_fold', fail)
    with pytest.raises(RuntimeError, match='training stopped'):
        run_conditional_audit(path=path, output=tmp_path / 'failed', protocol=protocol(),
                              train_config=replace(config(), host_cache=True))
    assert retained[0].bytes_cached == 0


def test_stop_during_resident_upload_never_advances_an_optimizer(tmp_path):
    path = tmp_path / 'paths.h5'
    population(path)
    mask = internal_training_mask(16, protocol=protocol(), fold=0)
    values = _normalization(path, outer=slice(0, 16), mask=mask, protocol=protocol(),
        output=tmp_path / 'normalization.npz', identity='same', upload_paths=4)
    checks = iter((False, True))
    with pytest.raises(InterruptedError, match='no optimizer update committed'):
        _resident_fold(path, outer=slice(0, 16), normalization=values,
            env=BaseEnvironment(device='cpu', precision=64), train_config=config(), progress={}, stop_requested=lambda: next(checks))


def test_already_requested_stop_does_not_allocate_resident_tensors(tmp_path, monkeypatch):
    environment = BaseEnvironment(device='cpu', precision=64)
    training = config()
    monkeypatch.setattr(torch, 'empty', lambda *args, **kwargs: pytest.fail('stopped fold must not allocate tensors'))
    with pytest.raises(InterruptedError, match='stopped before fold allocation'):
        _resident_fold(tmp_path / 'not-needed.h5', outer=slice(0, 16), normalization={},
            env=environment, train_config=training, progress={}, stop_requested=lambda: True)


def test_readonly_pinned_staging_uses_empty_and_copy_not_numpy_tensor_constructor(monkeypatch):
    values = np.asarray([[1., -0.], [3., 4.]], dtype=np.float32)
    values.flags.writeable = False
    original = torch.empty
    requested = []

    def empty(*args, **kwargs):
        requested.append(kwargs.get('pin_memory'))
        kwargs['pin_memory'] = False
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, 'empty', empty)
    monkeypatch.setattr(torch, 'tensor', lambda *args, **kwargs: pytest.fail('NumPy tensor constructor cannot pin storage'))
    result = _cpu_tensor_block(values, pin_memory=True)
    assert requested == [True]
    np.testing.assert_array_equal(result.numpy().view(np.uint32), values.view(np.uint32))
    result[0, 0] = 9.
    assert values[0, 0] == 1.


@pytest.mark.skipif(not os.environ.get('CUDA_VISIBLE_DEVICES'), reason='requires an allocated GPU')
def test_cuda_cached_resident_inputs_exactly_match_direct_upload(tmp_path):
    path = tmp_path / 'paths.h5'
    population(path)
    with h5py.File(path, 'r') as stream:
        cached = cache_audit_arrays(stream, chunk_paths=4)
    mask = internal_training_mask(16, protocol=protocol(), fold=0)
    values = _normalization(path, outer=slice(0, 16), mask=mask, protocol=protocol(),
        output=tmp_path / 'normalization.npz', identity='same', upload_paths=4)
    training = replace(config(), device='cuda', precision=32)
    try:
        actual = []
        for cache in (None, cached):
            actual.append(_resident_fold(path, outer=slice(0, 16), normalization=values,
                env=BaseEnvironment(device='cuda', precision=32), train_config=training, progress={}, cache=cache))
        for direct, copied in zip(*actual, strict=True):
            torch.testing.assert_close(direct, copied, rtol=0., atol=0.)
        with h5py.File(path, 'r') as stream:
            host = cached.block(stream, 'features', slice(0, 4))
        pinned = _cpu_tensor_block(host, pin_memory=True)
        assert pinned.is_pinned()
        np.testing.assert_array_equal(pinned.numpy(), host)
    finally:
        cached.release()
