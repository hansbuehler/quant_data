from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import pytest

from quantdata.gromm2 import audit_host_cache
from quantdata.gromm2.audit_host_cache import cache_audit_arrays, read_audit_block


def population(path):
    features = np.arange(8 * 3 * 4, dtype=np.float32).reshape(8, 3, 4)
    returns = np.arange(8 * 3 * 2, dtype=np.float64).reshape(8, 3, 2) / 17.
    with h5py.File(path, 'w') as stream:
        stream.attrs.update(identity='test-population', completed_paths=8)
        stream.create_dataset('features', data=features, chunks=(2, 3, 4), compression='lzf')
        stream.create_dataset('returns', data=returns, chunks=(2, 3, 2), compression='lzf')
        stream.create_dataset('expiry_days', data=[2, 3])
    return features, returns


def resources(*, health='OK'):
    return dict(blocked=health in {'STOP', 'ERROR', 'UNAVAILABLE'}, warning=health != 'OK')


@pytest.fixture(autouse=True)
def healthy(monkeypatch):
    monkeypatch.setattr(audit_host_cache, '_host_cache_resources', lambda remaining: (resources(), True))


def test_cache_preserves_bits_native_dtypes_slices_and_metadata(tmp_path):
    path = tmp_path / 'paths.h5'
    expected = population(path)
    with h5py.File(path, 'r') as stream:
        cache = cache_audit_arrays(stream, chunk_paths=3)
        assert cache.bytes_cached == sum(value.nbytes for value in expected)
        assert cache.array_names == ('features', 'returns')
        for name, values in zip(cache.array_names, expected, strict=True):
            for rows in (slice(None), slice(2, 7), slice(-3, None), slice(5, 2)):
                actual = read_audit_block(stream, name, rows, cache=cache)
                assert actual.dtype == values.dtype
                assert actual.tobytes() == values[rows].tobytes()
                assert not actual.flags.writeable
        np.testing.assert_array_equal(cache.block(stream, 'expiry_days', slice(None)), [2, 3])


def test_fold_coverage_and_partial_fields_fall_back_without_reordering(tmp_path, monkeypatch):
    path = tmp_path / 'paths.h5'
    expected = population(path)
    capacity = expected[0][2:6].nbytes
    probes = []

    def admit(remaining):
        probes.append(remaining)
        return resources(), remaining <= capacity

    monkeypatch.setattr(audit_host_cache, '_host_cache_resources', admit)
    with h5py.File(path, 'r') as stream:
        cache = cache_audit_arrays(stream, rows=slice(2, 6), chunk_paths=3)
        assert cache.path_range == (2, 6)
        assert cache.array_names == ('features',)
        assert probes[0] == sum(value[2:6].nbytes for value in expected)
        for name, values in zip(('features', 'returns'), expected, strict=True):
            for rows in (slice(2, 5), slice(0, 4), slice(5, None)):
                actual = cache.block(stream, name, rows)
                assert actual.tobytes() == values[rows].tobytes()
        assert cache.check_headroom()


@pytest.mark.parametrize('health', ['WARN', 'STOP', 'ERROR', 'UNAVAILABLE'])
def test_resource_rejection_before_allocation(tmp_path, monkeypatch, health):
    path = tmp_path / 'paths.h5'
    population(path)
    monkeypatch.setattr(audit_host_cache, '_host_cache_resources', lambda remaining: (resources(health=health), False))
    monkeypatch.setattr(audit_host_cache.np, 'empty', lambda *args, **kwargs: pytest.fail('must not allocate'))
    with h5py.File(path, 'r') as stream:
        if health == 'WARN':
            assert cache_audit_arrays(stream) is None
        else:
            with pytest.raises(RuntimeError, match='resource manager blocked'):
                cache_audit_arrays(stream)


@pytest.mark.parametrize('blocked', [False, True])
def test_release_on_pressure_keeps_inflight_view_valid_and_future_reads_hdf5(tmp_path, monkeypatch, blocked):
    path = tmp_path / 'paths.h5'
    features, _ = population(path)
    with h5py.File(path, 'r') as stream:
        cache = cache_audit_arrays(stream)
        view = cache.block(stream, 'features', slice(1, 3))
        monkeypatch.setattr(audit_host_cache, '_host_cache_resources',
            lambda remaining: (dict(blocked=blocked, warning=True, probe_error='unreadable' if not blocked else None), False))
        with ThreadPoolExecutor(max_workers=1) as workers:
            future = workers.submit(cache.check_headroom)
            if blocked:
                with pytest.raises(RuntimeError, match='released or declined'):
                    future.result()
            else:
                assert not future.result()
        assert cache.bytes_cached == 0 and cache.array_names == ()
        assert view.tobytes() == features[1:3].tobytes()
        later = cache.block(stream, 'features', slice(1, 3))
        assert later.flags.writeable and not np.shares_memory(view, later)
        cache.release()
        assert not cache.check_headroom()


@pytest.mark.parametrize('health', ['WARN', 'STOP'])
def test_midbuild_pressure_aborts_without_publication(tmp_path, monkeypatch, capsys, health):
    path = tmp_path / 'paths.h5'
    population(path)
    calls = 0

    def admit(remaining):
        nonlocal calls
        calls += 1
        return (resources(), True) if calls == 1 else (resources(health=health), False)

    ticks = iter(range(100))
    monkeypatch.setattr(audit_host_cache, '_host_cache_resources', admit)
    monkeypatch.setattr(audit_host_cache.time, 'monotonic', lambda: next(ticks))
    with h5py.File(path, 'r') as stream:
        if health == 'WARN':
            assert cache_audit_arrays(stream, chunk_paths=2, check_interval_seconds=.5) is None
        else:
            with pytest.raises(RuntimeError, match='resource manager blocked'):
                cache_audit_arrays(stream, chunk_paths=2, check_interval_seconds=.5)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    progress = next(event for event in events if event['event'] == 'audit-host-cache-progress')
    assert 0 < progress['completed_bytes'] < progress['total_bytes']
    assert progress['elapsed_seconds'] > 0 and progress['remaining_seconds'] > 0
    assert all(event['event'] != 'audit-host-cache-ready' for event in events)


def test_interruption_no_cache_and_optional_direct_reads(tmp_path):
    path = tmp_path / 'paths.h5'
    features, _ = population(path)
    with h5py.File(path, 'r') as stream:
        with pytest.raises(InterruptedError, match='no cache published'):
            cache_audit_arrays(stream, stop_requested=lambda: True)
        assert read_audit_block(stream, 'features', slice(1, 4)).tobytes() == features[1:4].tobytes()


def test_cache_rejects_different_source_and_invalid_selection(tmp_path):
    first, second = tmp_path / 'first.h5', tmp_path / 'second.h5'
    population(first)
    population(second)
    with h5py.File(first, 'r') as stream, h5py.File(second, 'r') as other:
        cache = cache_audit_arrays(stream)
        with pytest.raises(ValueError, match='different source'):
            cache.block(other, 'features', slice(None))
        with pytest.raises(ValueError, match='contiguous'):
            cache.block(stream, 'features', slice(None, None, 2))
        with pytest.raises(ValueError, match='raw features'):
            cache_audit_arrays(stream, fields=('expiry_days',))


def test_cache_preserves_nonfinite_payloads_signed_zero_and_bounded_reads(tmp_path, monkeypatch):
    path = tmp_path / 'paths.h5'
    features, _ = population(path)
    features.view(np.uint32).reshape(-1)[:4] = [0x7fc00013, 0xff800000, 0x80000000, 0x7f800000]
    with h5py.File(path, 'r+') as stream:
        stream['features'][:] = features
    original = h5py.Dataset.read_direct
    reads = []

    def read(dataset, destination, source_sel=None, dest_sel=None):
        reads.append((dataset.name, source_sel.stop - source_sel.start))
        return original(dataset, destination, source_sel=source_sel, dest_sel=dest_sel)

    monkeypatch.setattr(h5py.Dataset, 'read_direct', read)
    with h5py.File(path, 'r') as stream:
        cache = cache_audit_arrays(stream, chunk_paths=3)
        assert cache.block(stream, 'features', slice(None)).tobytes() == features.tobytes()
    assert max(count for _, count in reads) == 3
    assert sum(count for _, count in reads) == 16


def test_declines_without_partial_admission_or_reading_data(tmp_path, monkeypatch):
    path = tmp_path / 'paths.h5'
    population(path)
    probes = []

    def admit(remaining):
        probes.append(remaining)
        return resources(), False

    monkeypatch.setattr(audit_host_cache, '_host_cache_resources', admit)
    monkeypatch.setattr(h5py.Dataset, 'read_direct', lambda *args, **kwargs: pytest.fail('declined cache must not read'))
    with h5py.File(path, 'r') as stream:
        assert cache_audit_arrays(stream, allow_partial=False) is None
    assert len(probes) == 1


def test_partial_admission_rechecks_before_allocating(tmp_path, monkeypatch):
    path = tmp_path / 'paths.h5'
    population(path)
    calls = 0

    def admit(remaining):
        nonlocal calls
        calls += 1
        return resources(), calls == 2

    monkeypatch.setattr(audit_host_cache, '_host_cache_resources', admit)
    monkeypatch.setattr(h5py.Dataset, 'read_direct', lambda *args, **kwargs: pytest.fail('stale admission must not read'))
    with h5py.File(path, 'r') as stream:
        assert cache_audit_arrays(stream) is None
    assert calls == 4


def test_final_resource_recheck_does_not_publish_newly_unsafe_cache(tmp_path, monkeypatch):
    path = tmp_path / 'paths.h5'
    population(path)
    probes = iter([(resources(), True), (resources(health='WARN'), False)])
    monkeypatch.setattr(audit_host_cache, '_host_cache_resources', lambda remaining: next(probes))
    with h5py.File(path, 'r') as stream:
        assert cache_audit_arrays(stream) is None


def test_readonly_completed_population_is_required(tmp_path):
    path = tmp_path / 'paths.h5'
    population(path)
    with h5py.File(path, 'r+') as stream:
        with pytest.raises(ValueError, match='read-only'):
            cache_audit_arrays(stream)
        stream.attrs['completed_paths'] = 7
    with h5py.File(path, 'r') as stream:
        with pytest.raises(ValueError, match='completed paths'):
            cache_audit_arrays(stream)


def test_cache_can_reuse_new_readonly_handle_without_owning_original_stream(tmp_path):
    path = tmp_path / 'paths.h5'
    features, _ = population(path)
    with h5py.File(path, 'r') as stream:
        cache = cache_audit_arrays(stream)
    with h5py.File(path, 'r') as reopened:
        actual = cache.block(reopened, 'features', slice(None))
        assert not actual.flags.writeable and actual.tobytes() == features.tobytes()
