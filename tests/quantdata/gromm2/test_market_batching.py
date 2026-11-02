from __future__ import annotations

import os
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quantdata.gromm2 import market_batching
from quantdata.gromm2.hedging_policy import ScenarioData, _batch
from quantdata.gromm2.market_batching import (
    GPU_CAPTURE_LOCK,
    MarketBatchIterator,
    assemble_market_batch,
    cache_market_training_arrays,
)
from quantdata.gromm2.market_books import primary_contracts
from quantdata.gromm2.market_hedging import MarketView
from quantdata.torch import BaseEnvironment


def market_data(*, kind="cohorts", paths=17):
    generator = np.random.default_rng(572)
    book = primary_contracts()
    arrays = dict(spots=np.exp(generator.normal(scale=0.1, size=(paths, 21))),
        backgrounds=np.exp(generator.normal(scale=0.1, size=(paths, 21))),
        core_features=generator.normal(size=(paths, 20, 5)).astype("f4"),
        option_prices=generator.normal(size=(paths, 21, 84)),
        factor_prices=generator.normal(size=(paths, 21, 2)),
        daily_calls=generator.normal(size=(paths, 20)))
    return ScenarioData(MarketView(arrays, book, kind=kind, features=True),
        MarketView(arrays, book, kind=kind, features=False), generator.normal(size=paths),
        tuple(map(int, book["expiries"])), {}, "test")


def cache_resource_report(*, free_gib=90.0, health="OK", limits=None):
    return dict(blocked=health in {"STOP", "UNAVAILABLE", "ERROR"}, warning=health != "OK",
        report=dict(checks=[dict(type="ram", health=health,
            current=dict(free_gib=free_gib, total_gib=100.0, free_pct=free_gib, used_pct=100 - free_gib),
            limits=limits if limits is not None else [dict(level="WARN", metric="free_pct", op="<=", value=15.0),
                dict(level="STOP", metric="free_pct", op="<=", value=10.0)])]))


def mock_cache_resources(monkeypatch, reports):
    from quantdata.gromm2 import gpu_forward_queue

    remaining = iter(reports)
    monkeypatch.setattr(gpu_forward_queue, "resource_report", lambda: next(remaining))


@pytest.mark.parametrize("kind", ["cohorts", "factors", "combined"])
def test_host_cache_preserves_exact_arrays_shared_views_and_release(monkeypatch, kind):
    data = market_data(kind="cohorts" if kind == "combined" else kind)
    if kind == "combined":
        from quantdata.gromm2.combined_hedging import CombinedMarketView

        arrays, book = data.features.arrays, data.features.book
        data.features = CombinedMarketView(arrays, book, features=True)
        data.prices = CombinedMarketView(arrays, book, features=False)
    data.identity = {"split": "train", "model": "unchanged"}
    originals = dict(data.features.arrays)
    expected = assemble_market_batch(data, np.array([16, 2, 2, 0, 11]))
    identity, payoff, digest = dict(data.identity), data.payoff, data.sha256
    mock_cache_resources(monkeypatch, [cache_resource_report()] * 3)
    cache = cache_market_training_arrays(data, chunk_bytes=113)
    assert cache is not None
    assert cache.bytes_cached == sum(source.nbytes for source in originals.values())
    assert data.features.arrays is data.prices.arrays
    assert data.identity == identity and data.sha256 == digest and data.payoff is payoff
    for name, source in originals.items():
        copied = data.features.arrays[name]
        assert type(copied) is np.ndarray
        assert copied.dtype == source.dtype and not copied.flags.writeable
        assert not np.shares_memory(copied, source)
        assert copied.tobytes() == source.tobytes()
    for original, prepared in zip(expected, assemble_market_batch(data, np.array([16, 2, 2, 0, 11])), strict=True):
        assert original.tobytes() == prepared.tobytes()
    assert cache.check_headroom()
    retained = data.features.arrays["core_features"]
    cache.release()
    assert cache.bytes_cached == 0
    assert all(data.features.arrays[name] is source for name, source in originals.items())
    np.testing.assert_array_equal(retained, originals["core_features"])
    cache.release()
    assert not cache.check_headroom()


def test_partial_host_cache_admits_only_raw_fields_that_fit_manager_limits(monkeypatch):
    data = market_data()
    data.identity["split"] = "train"
    originals = dict(data.features.arrays)
    largest = max(originals, key=lambda name: originals[name].nbytes)
    capacity = originals[largest].nbytes
    probes = []

    def resources(remaining_bytes):
        probes.append(remaining_bytes)
        return cache_resource_report(), remaining_bytes <= capacity

    monkeypatch.setattr(market_batching, "_host_cache_resources", resources)
    rows = np.array([16, 2, 2, 0, 11])
    expected = assemble_market_batch(data, rows)
    cache = cache_market_training_arrays(data, allow_partial=True)
    assert cache is not None and cache.bytes_cached == capacity
    assert probes[0] == sum(value.nbytes for value in originals.values())
    assert set(cache.array_names) == {largest}
    assert not np.shares_memory(data.features.arrays[largest], originals[largest])
    assert all(data.features.arrays[name] is source for name, source in originals.items() if name != largest)
    for original, prepared in zip(expected, assemble_market_batch(data, rows), strict=True):
        assert original.tobytes() == prepared.tobytes()
    cache.release()
    assert cache.array_names == ()
    assert all(data.features.arrays[name] is source for name, source in originals.items())


def test_partial_cache_interruption_rolls_back_without_publishing_any_fields(monkeypatch):
    data = market_data()
    data.identity["split"] = "train"
    originals = dict(data.features.arrays)
    capacity = max(source.nbytes for source in originals.values())
    monkeypatch.setattr(market_batching, "_host_cache_resources",
        lambda remaining: (cache_resource_report(), remaining <= capacity))
    checks = iter([False, False, True])
    with pytest.raises(InterruptedError, match="mapped views unchanged"):
        cache_market_training_arrays(data, allow_partial=True, chunk_bytes=1, stop_requested=lambda: next(checks))
    assert all(data.features.arrays[name] is source for name, source in originals.items())


def test_partial_cache_rechecks_total_when_headroom_shrinks_during_selection(monkeypatch):
    data = market_data()
    data.identity["split"] = "train"
    originals = dict(data.features.arrays)
    probes = []

    def resources(remaining_bytes):
        probes.append(remaining_bytes)
        return cache_resource_report(), len(probes) == 2

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("a stale partial admission must not allocate any copies")

    monkeypatch.setattr(market_batching, "_host_cache_resources", resources)
    monkeypatch.setattr(np, "empty_like", unexpected_allocation)
    assert cache_market_training_arrays(data, allow_partial=True) is None
    assert len(probes) == len(originals) + 2
    assert probes[-1] == max(source.nbytes for source in originals.values())
    assert all(data.features.arrays[name] is source for name, source in originals.items())


@pytest.mark.parametrize("health", ["WARN", "STOP"])
def test_partial_admission_never_overrides_new_resource_pressure(monkeypatch, health):
    data = market_data()
    data.identity["split"] = "train"
    originals = dict(data.features.arrays)
    reports = iter([(cache_resource_report(), False), (cache_resource_report(health=health), False)])
    monkeypatch.setattr(market_batching, "_host_cache_resources", lambda _: next(reports))
    if health == "STOP":
        with pytest.raises(RuntimeError, match="resource manager blocked"):
            cache_market_training_arrays(data, allow_partial=True)
    else:
        assert cache_market_training_arrays(data, allow_partial=True) is None
    assert all(data.features.arrays[name] is source for name, source in originals.items())


@pytest.mark.parametrize("limits", [
    [dict(level="WARN", metric="free_pct", op="<=", value=15.0)],
    [dict(level="WARN", metric="used_pct", op=">=", value=85.0)],
    [dict(level="WARN", metric="free_gib", op="<=", value=15.0)],
    [dict(level="STOP", metric="used_gib", op=">=", value=85.0)],
])
def test_host_cache_projects_complete_allocation_against_reported_limits(monkeypatch, limits):
    mock_cache_resources(monkeypatch, [cache_resource_report(limits=limits)] * 2)
    assert market_batching._host_cache_resources(74 * 2**30)[1]
    assert not market_batching._host_cache_resources(75 * 2**30)[1]


@pytest.mark.parametrize("resources", [
    cache_resource_report(health="WARN"),
    dict(blocked=False, warning=True, message="gpuctl resources could not be read"),
    cache_resource_report(limits=[]),
    cache_resource_report(limits=[dict(level="WARN", metric="unrecognized", op="<=", value=15.0)]),
    dict(blocked=False, warning=False, report=dict(checks=[])),
])
def test_host_cache_declines_without_reliable_headroom(monkeypatch, resources):
    data = market_data()
    data.identity["split"] = "train"
    originals = dict(data.features.arrays)
    mock_cache_resources(monkeypatch, [resources])
    assert cache_market_training_arrays(data) is None
    assert all(data.features.arrays[name] is source for name, source in originals.items())


@pytest.mark.parametrize("health", ["WARN", "STOP", "UNAVAILABLE", "ERROR"])
@pytest.mark.parametrize("stage", ["admission", "copy", "publish", "active"])
def test_host_cache_releases_or_aborts_on_resource_pressure(monkeypatch, health, stage):
    data = market_data()
    data.identity["split"] = "train"
    originals = dict(data.features.arrays)
    healthy, pressure = cache_resource_report(), cache_resource_report(health=health)
    preceding = {"admission": 0, "copy": 1, "publish": 1, "active": 2}[stage]
    mock_cache_resources(monkeypatch, [healthy] * preceding + [pressure])
    cache = None

    def run():
        nonlocal cache
        cache = cache_market_training_arrays(data, check_interval_seconds=1e-12 if stage == "copy" else 60.0)
        if stage == "active":
            assert cache is not None
            assert not cache.check_headroom()
        else:
            assert cache is None

    if health == "WARN":
        run()
    else:
        with pytest.raises(RuntimeError, match="resource manager blocked"):
            run()
    assert all(data.features.arrays[name] is source for name, source in originals.items())
    if cache is not None:
        assert cache.bytes_cached == 0


def test_host_cache_interruption_never_publishes_partial_copies(monkeypatch):
    data = market_data()
    data.identity["split"] = "train"
    originals = dict(data.features.arrays)
    mock_cache_resources(monkeypatch, [cache_resource_report()])
    checks = iter([False, False, True])
    with pytest.raises(InterruptedError, match="mapped views unchanged"):
        cache_market_training_arrays(data, chunk_bytes=1, stop_requested=lambda: next(checks))
    assert all(data.features.arrays[name] is source for name, source in originals.items())


def test_host_cache_failed_copy_frees_arrays_even_with_retained_traceback(monkeypatch):
    data = market_data()
    data.identity["split"] = "train"
    allocated = []
    empty_like = np.empty_like

    def track_allocation(*args, **kwargs):
        values = empty_like(*args, **kwargs)
        allocated.append(weakref.ref(values))
        return values

    monkeypatch.setattr(np, "empty_like", track_allocation)
    mock_cache_resources(monkeypatch, [cache_resource_report(), cache_resource_report(health="STOP")])
    with pytest.raises(RuntimeError, match="resource manager blocked") as caught:
        cache_market_training_arrays(data, check_interval_seconds=1e-12)
    assert caught.value.__traceback__ is not None
    assert allocated and all(reference() is None for reference in allocated)


def test_host_cache_copies_readonly_memmaps_without_modifying_files(monkeypatch, tmp_path):
    data = market_data()
    data.identity["split"] = "train"
    source_bytes = {}
    for name, values in data.features.arrays.items():
        path = tmp_path / f"{name}.raw"
        source_bytes[path] = values.tobytes()
        path.write_bytes(source_bytes[path])
        data.features.arrays[name] = np.memmap(path, mode="r", dtype=values.dtype, shape=values.shape)
    originals = dict(data.features.arrays)
    mock_cache_resources(monkeypatch, [cache_resource_report()] * 2)
    cache = cache_market_training_arrays(data, chunk_bytes=113)
    assert cache is not None
    assert all(type(values) is np.ndarray and not values.flags.writeable for values in data.features.arrays.values())
    cache.release()
    assert all(data.features.arrays[name] is values for name, values in originals.items())
    assert all(path.read_bytes() == contents for path, contents in source_bytes.items())


def test_host_cache_release_serializes_heartbeat_and_main_cleanup(monkeypatch):
    entered, proceed, second_started, second_done = Event(), Event(), Event(), Event()

    class PausedRestore(dict):
        pause = True

        def __setitem__(self, name, value):
            if self.pause:
                self.pause = False
                entered.set()
                assert proceed.wait(timeout=5)
            super().__setitem__(name, value)

    data = market_data()
    data.identity["split"] = "train"
    originals = dict(data.features.arrays)
    data.features.arrays = data.prices.arrays = PausedRestore(originals)
    mock_cache_resources(monkeypatch, [cache_resource_report()] * 2)
    cache = cache_market_training_arrays(data)
    assert cache is not None

    def second_release():
        second_started.set()
        cache.release()
        second_done.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(cache.release)
        assert entered.wait(timeout=5)
        second = executor.submit(second_release)
        try:
            assert second_started.wait(timeout=5)
            assert not second_done.wait(timeout=0.05)
        finally:
            proceed.set()
        first.result(timeout=5)
        second.result(timeout=5)
    assert cache.bytes_cached == 0
    assert all(data.features.arrays[name] is source for name, source in originals.items())


def test_host_cache_slow_probe_never_blocks_main_release(monkeypatch):
    entered, proceed = Event(), Event()
    data = market_data()
    data.identity["split"] = "train"
    mock_cache_resources(monkeypatch, [cache_resource_report()] * 2)
    cache = cache_market_training_arrays(data)
    assert cache is not None

    def slow_probe(remaining_bytes):
        entered.set()
        assert proceed.wait(timeout=5)
        return cache_resource_report(health="WARN"), False

    monkeypatch.setattr(market_batching, "_host_cache_resources", slow_probe)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(cache.check_headroom)
        try:
            assert entered.wait(timeout=5)
            cache.release()
            assert cache.bytes_cached == 0
        finally:
            proceed.set()
        assert future.result(timeout=5) is False


def test_host_cache_rejects_validation_and_never_accesses_cuda(monkeypatch):
    def unexpected_cuda(*args, **kwargs):
        raise AssertionError("raw host cache must not inspect CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_cuda)
    monkeypatch.setattr(torch.cuda, "current_stream", unexpected_cuda)
    data = market_data()
    data.identity["split"] = "validation"
    with pytest.raises(ValueError, match="training split"):
        cache_market_training_arrays(data)
    data.identity["split"] = "train"
    mock_cache_resources(monkeypatch, [cache_resource_report()] * 2)
    cache = cache_market_training_arrays(data)
    assert cache is not None
    cache.release()


@pytest.mark.parametrize("kind", ["cohorts", "factors"])
@pytest.mark.parametrize("rows", [slice(None), slice(1, 8, 2), slice(16, 17), slice(0, 0),
    np.array([16, 2, 2, 0, 11]), np.array([], dtype=np.int64)])
def test_shared_assembly_is_bitwise_identical(kind, rows):
    data = market_data(kind=kind)
    expected = data.features[rows], data.prices[rows], data.payoff[rows]
    actual = assemble_market_batch(data, rows)
    for original, prepared in zip(expected, actual, strict=True):
        assert original.dtype == prepared.dtype
        assert original.shape == prepared.shape
        assert original.tobytes() == prepared.tobytes()
    buffers = tuple(np.empty_like(values) for values in expected)
    assert assemble_market_batch(data, rows, out=buffers) is buffers
    for original, prepared in zip(expected, buffers, strict=True):
        assert original.tobytes() == prepared.tobytes()


def test_shared_assembly_gathers_each_raw_array_once():
    class CountedArray:
        def __init__(self, values):
            self.values = values
            self.count = 0
            self.shape = values.shape

        def __getitem__(self, rows):
            self.count += 1
            return self.values[rows]

    data = market_data()
    arrays = data.features.arrays
    for name, values in arrays.items():
        arrays[name] = CountedArray(values)
    assemble_market_batch(data, np.array([4, 1, 4]))
    assert all(values.count == 1 for values in arrays.values())


@pytest.mark.parametrize("precision", [32, 64])
@pytest.mark.parametrize("ordered", [False, True])
def test_cpu_iterator_matches_existing_batches_and_retains_outputs(precision, ordered):
    data = market_data()
    order = np.random.default_rng(102).permutation(len(data.payoff)) if ordered else None
    before = None if order is None else order.copy()
    env = BaseEnvironment(device="cpu", precision=precision)
    rng_state = torch.get_rng_state().clone()
    with MarketBatchIterator(data, env=env, batch_size=4, order=order) as iterator:
        actual = list(iterator)
    assert iterator.closed
    assert [len(batch[0]) for batch in actual] == [4, 4, 4, 4, 1]
    for start, batch in zip(range(0, len(data.payoff), 4), actual, strict=True):
        rows = slice(start, start + 4) if order is None else order[start:start + 4]
        expected = _batch(data, rows, env=env)
        for original, prepared in zip(expected, batch, strict=True):
            torch.testing.assert_close(original, prepared, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng_state)
    if order is not None:
        np.testing.assert_array_equal(order, before)


def test_resume_offset_and_early_close_preserve_remaining_order():
    data = market_data()
    order = np.array([5, 2, 2, 7, 1, 9, 0, 15, 8, 4, 11])
    env = BaseEnvironment(device="cpu", precision=32)
    with MarketBatchIterator(data, env=env, batch_size=4, order=order) as iterator:
        first = next(iterator)
    assert iterator.closed
    with pytest.raises(StopIteration):
        next(iterator)
    with MarketBatchIterator(data, env=env, batch_size=4, order=order, start=4) as iterator:
        remainder = list(iterator)
    actual = tuple(torch.cat(values) for values in zip(first, *remainder, strict=True))
    expected = _batch(data, order, env=env)
    for original, prepared in zip(expected, actual, strict=True):
        torch.testing.assert_close(original, prepared, rtol=0, atol=0)


def test_generic_readonly_scenario_falls_back_without_cuda(monkeypatch):
    data = market_data()
    data = replace(data, features=data.features[:], prices=data.prices[:])
    for values in (data.features, data.prices, data.payoff):
        values.flags.writeable = False

    def unexpected_cuda(*args, **kwargs):
        raise AssertionError("CPU batching must not inspect CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_cuda)
    monkeypatch.setattr(torch.cuda, "current_stream", unexpected_cuda)
    env = BaseEnvironment(device="cpu", precision=64)
    with MarketBatchIterator(data, env=env, batch_size=6, start=6) as iterator:
        actual = list(iterator)
    expected = _batch(data, slice(6, None), env=env)
    for original, prepared in zip(expected, (torch.cat(values) for values in zip(*actual, strict=True)), strict=True):
        torch.testing.assert_close(original, prepared, rtol=0, atol=0)


def test_invalid_output_geometry_and_iterator_configuration():
    data = market_data()
    env = BaseEnvironment(device="cpu", precision=32)
    with pytest.raises(ValueError, match="batch_size"):
        MarketBatchIterator(data, env=env, batch_size=0)
    with pytest.raises(ValueError, match="prefetch_workers"):
        MarketBatchIterator(data, env=env, batch_size=4, prefetch_workers=3)
    with pytest.raises(ValueError, match="start"):
        MarketBatchIterator(data, env=env, batch_size=4, start=18)
    with pytest.raises(ValueError, match="order"):
        MarketBatchIterator(data, env=env, batch_size=4, order=np.array([0.0]))
    wrong = tuple(np.empty_like(values) for values in assemble_market_batch(data, slice(0, 2)))
    with pytest.raises(ValueError, match="shapes and dtypes"):
        assemble_market_batch(data, slice(0, 3), out=wrong)
    with MarketBatchIterator(data, env=env, batch_size=4, start=len(data.payoff)) as iterator:
        assert list(iterator) == []


def allocated_cuda_available():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return visible.lower() not in {"", "-1", "none", "void"} and torch.cuda.is_available()


def test_capture_guard_excludes_producer_cuda_event_wait(monkeypatch):
    started, synchronized = Event(), Event()
    data = market_data()
    iterator = MarketBatchIterator(data, env=BaseEnvironment(device="cpu", precision=32), batch_size=4)
    slot = SimpleNamespace(copied=SimpleNamespace(synchronize=synchronized.set),
        tensors=(torch.empty((4, 20, data.features.shape[-1])),
            torch.empty((4, 21, data.prices.shape[-1]), dtype=torch.float64), torch.empty(4, dtype=torch.float64)))

    def stop_after_event(*args, **kwargs):
        raise RuntimeError("assembly reached")

    def prepare():
        started.set()
        return iterator._prepare(slot, slice(0, 4))

    monkeypatch.setattr(market_batching, "assemble_market_batch", stop_after_event)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with GPU_CAPTURE_LOCK:
            future = executor.submit(prepare)
            assert started.wait(timeout=5)
            assert not synchronized.wait(timeout=0.05)
        with pytest.raises(RuntimeError, match="assembly reached"):
            future.result(timeout=5)
    assert synchronized.is_set()
    iterator.close()


@pytest.mark.skipif(not allocated_cuda_available(), reason="requires a caller-owned gpuctl allocation")
@pytest.mark.parametrize("precision", [32, 64])
@pytest.mark.parametrize("prefetch_workers", [1, 2])
def test_cuda_pinned_slots_preserve_retained_batches_and_consumer_stream(precision, prefetch_workers):
    data = market_data(paths=37)
    order = np.random.default_rng(103).permutation(len(data.payoff))
    env = BaseEnvironment(device=torch.device("cuda", 0), precision=precision)
    consumer = torch.cuda.Stream(device=env.device)
    retained, sums = [], []
    with MarketBatchIterator(data, env=env, batch_size=4, order=order, start=4, prefetch_workers=prefetch_workers) as iterator:
        assert len(iterator.slots) == 2
        assert all(tensor.is_pinned() for slot in iterator.slots for tensor in slot.tensors)
        with torch.cuda.stream(consumer):
            for batch in iterator:
                retained.append(batch)
                sums.append(tuple(tensor.square().sum() for tensor in batch))
    consumer.synchronize()
    for start, batch, totals in zip(range(4, len(order), 4), retained, sums, strict=True):
        expected = _batch(data, order[start:start + 4], env=env)
        for original, prepared, total in zip(expected, batch, totals, strict=True):
            torch.testing.assert_close(original, prepared, rtol=0, atol=0)
            torch.testing.assert_close(original.square().sum(), total, rtol=0, atol=0)
    assert iterator.closed and not iterator.pending and not iterator.slots


@pytest.mark.skipif(not allocated_cuda_available(), reason="requires a caller-owned gpuctl allocation")
def test_cuda_early_close_leaves_yielded_batch_valid():
    data = market_data(paths=37)
    env = BaseEnvironment(device=torch.device("cuda", 0), precision=32)
    with MarketBatchIterator(data, env=env, batch_size=4) as iterator:
        first = next(iterator)
    expected = _batch(data, slice(0, 4), env=env)
    for original, prepared in zip(expected, first, strict=True):
        torch.testing.assert_close(original, prepared, rtol=0, atol=0)
    iterator.close()


@pytest.mark.skipif(not allocated_cuda_available(), reason="requires a caller-owned gpuctl allocation")
def test_cuda_capture_guard_excludes_prefetch_and_allows_later_batches():
    data = market_data(paths=13)
    env = BaseEnvironment(device=torch.device("cuda", 0), precision=32)
    probe = torch.ones(4, device=env.device)
    graph = torch.cuda.CUDAGraph()
    with GPU_CAPTURE_LOCK:
        iterator = MarketBatchIterator(data, env=env, batch_size=4)
        with torch.cuda.graph(graph):
            result = probe * 2
        assert all(not future.done() for _, future in iterator.pending)
    with iterator:
        batches = list(iterator)
    graph.replay()
    torch.testing.assert_close(result, torch.full_like(probe, 2))
    expected = _batch(data, slice(None), env=env)
    for original, prepared in zip(expected, (torch.cat(values) for values in zip(*batches, strict=True)), strict=True):
        torch.testing.assert_close(original, prepared, rtol=0, atol=0)
