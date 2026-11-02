"""Shared market assembly and bounded, pinned lookahead without changing samples."""

from __future__ import annotations

import json
import math
import operator
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING

import numpy as np
import torch
from cdxcore.err import verify_inp

if TYPE_CHECKING:
    from quantdata.gromm2.hedging_policy import ScenarioData
    from quantdata.torch import BaseEnvironment

ArrayBatch = tuple[np.ndarray, np.ndarray, np.ndarray]
TensorBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
GPU_CAPTURE_LOCK = RLock()


def _host_cache_resources(remaining_bytes: int) -> tuple[dict, bool]:
    from quantdata.gromm2.gpu_forward_queue import resource_report

    resources = resource_report()
    if resources["blocked"] or resources["warning"]:
        return resources, False
    ram_checks = [check for check in resources["report"]["checks"] if check.get("type") == "ram"]
    if not ram_checks:
        return resources, False
    comparisons = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
                   "==": operator.eq, "!=": operator.ne}
    for check in ram_checks:
        current = check.get("current") or {}
        total = current.get("total_gib", 0)
        free = current.get("free_gib", float("nan")) - remaining_bytes / 2**30
        if not math.isfinite(total) or total <= 0 or not math.isfinite(free) or free <= 0:
            return resources, False
        projected = dict(free_gib=free, total_gib=total, used_gib=total - free,
                         free_pct=100 * free / total, used_pct=100 * (total - free) / total)
        limits = [limit for limit in check.get("limits", []) if limit.get("level") in {"WARN", "STOP"}]
        if not limits:
            return resources, False
        for limit in limits:
            metric, comparison, value = limit.get("metric"), comparisons.get(limit.get("op")), limit.get("value")
            if (metric not in projected or comparison is None or not isinstance(value, (int, float))
                    or not math.isfinite(value) or comparison(projected[metric], value)):
                return resources, False
    return resources, True


@dataclass
class MarketHostCache:
    """Own exact raw training copies; release restores the original mapped views.

    Call ``check_headroom`` periodically while using this cache. Releasing never
    invalidates arrays already held by a producer, and does not access CUDA.
    """

    _arrays: dict[str, np.ndarray]
    _originals: dict[str, np.ndarray]
    _cached: dict[str, np.ndarray]
    bytes_cached: int
    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)

    @property
    def array_names(self) -> tuple[str, ...]:
        """Return the raw fields currently owned by this exact host cache."""
        with self._lock:
            return tuple(self._cached)

    def release(self) -> None:
        """Restore only entries still owned by this cache, without touching sources."""
        with self._lock:
            for name, cached in self._cached.items():
                if self._arrays.get(name) is cached:
                    self._arrays[name] = self._originals[name]
            self._cached.clear()
            self._originals.clear()
            self.bytes_cached = 0

    def check_headroom(self) -> bool:
        """Release on WARN or unreadable probes; release and raise on STOP/errors."""
        with self._lock:
            if not self._cached:
                return False
        resources, allowed = _host_cache_resources(0)
        with self._lock:
            if not self._cached:
                return False
            if not allowed:
                released_bytes = self.bytes_cached
                self.release()
        if not allowed:
            print(json.dumps(dict(event="market-host-cache-released", released_bytes=released_bytes,
                resources=resources)), flush=True)
            if resources["blocked"]:
                raise RuntimeError("resource manager blocked work; raw training host cache released")
        return allowed


def cache_market_training_arrays(data: ScenarioData, *, chunk_bytes: int = 64 * 2**20,
                                 check_interval_seconds: float = 30.0,
                                 stop_requested: Callable[[], bool] | None = None,
                                 allow_partial: bool = False) -> MarketHostCache | None:
    """Cache compact training inputs, preserving dtypes, sample order and identity.

    Admission projects the complete allocation against the manager's reported
    RAM limits. Copies are pageable, not pinned; only the iterator's existing
    bounded staging slots need pinned memory. Validation must remain separate.
    A declined or interrupted build leaves both scenario views unchanged.
    When requested, partial admission prioritizes the largest raw arrays and
    leaves the remaining fields mapped. Every admission uses the same manager
    thresholds; no partial cache is built during a current resource warning.
    """
    verify_inp(isinstance(chunk_bytes, int) and chunk_bytes > 0, lambda: "chunk_bytes must be positive")
    verify_inp(isinstance(allow_partial, bool), lambda: "allow_partial must be a boolean")
    verify_inp(math.isfinite(check_interval_seconds) and 0 < check_interval_seconds <= 60,
        lambda: "host-cache resource and progress interval must lie in (0, 60] seconds")
    arrays = getattr(data.features, "arrays", None)
    if (data.identity.get("split") != "train" or not isinstance(arrays, dict) or not arrays
            or arrays is not getattr(data.prices, "arrays", None)):
        raise ValueError("host caching requires shared raw market arrays from the training split")
    if any(not isinstance(source, np.ndarray) or source.ndim == 0 or source.dtype.hasobject
           for source in arrays.values()):
        raise ValueError("host-cache sources must be numeric path arrays")
    originals = dict(arrays)
    total_bytes = sum(source.nbytes for source in originals.values())
    resources, allowed = _host_cache_resources(total_bytes)
    if not allowed and allow_partial and not resources["blocked"] and not resources["warning"]:
        selected = {}
        selected_bytes = 0
        for name, source in sorted(originals.items(), key=lambda item: item[1].nbytes, reverse=True):
            resources, admitted = _host_cache_resources(selected_bytes + source.nbytes)
            if resources["blocked"] or resources["warning"]:
                break
            if admitted:
                selected[name] = source
                selected_bytes += source.nbytes
        if selected and not resources["blocked"] and not resources["warning"]:
            originals = selected
            total_bytes = selected_bytes
            resources, allowed = _host_cache_resources(selected_bytes)
    if not allowed:
        print(json.dumps(dict(event="market-host-cache-declined", required_bytes=total_bytes,
            resources=resources)), flush=True)
        if resources["blocked"]:
            raise RuntimeError("resource manager blocked raw training host cache allocation")
        return None
    cached = {}
    result = None
    completed = 0
    started = checked = time.monotonic()
    print(json.dumps(dict(event="market-host-cache-start", completed_bytes=0, total_bytes=total_bytes,
        array_names=list(originals), partial_cache=len(originals) != len(arrays),
        elapsed_seconds=0.0, remaining_seconds=None)), flush=True)
    try:
        for name, source in originals.items():
            target = np.empty_like(source, order="C", subok=False)
            cached[name] = target
            row_bytes = math.prod(source.shape[1:]) * source.dtype.itemsize
            rows_per_chunk = max(1, chunk_bytes // max(1, row_bytes))
            for start in range(0, len(source), rows_per_chunk):
                if stop_requested is not None and stop_requested():
                    raise InterruptedError("stopped while caching raw training arrays; mapped views unchanged")
                stop = min(start + rows_per_chunk, len(source))
                np.copyto(target[start:stop], source[start:stop], casting="no")
                completed += (stop - start) * row_bytes
                now = time.monotonic()
                if now - checked >= check_interval_seconds:
                    resources, allowed = _host_cache_resources(total_bytes - completed)
                    elapsed = now - started
                    print(json.dumps(dict(event="market-host-cache-progress", completed_bytes=completed,
                        total_bytes=total_bytes, elapsed_seconds=elapsed,
                        remaining_seconds=(total_bytes - completed) * elapsed / completed if completed else None)), flush=True)
                    checked = time.monotonic()
                    if not allowed:
                        print(json.dumps(dict(event="market-host-cache-aborted", resources=resources)), flush=True)
                        if resources["blocked"]:
                            raise RuntimeError("resource manager blocked work while caching raw training arrays")
                        return None
            target.flags.writeable = False
        resources, allowed = _host_cache_resources(0)
        if not allowed:
            print(json.dumps(dict(event="market-host-cache-aborted", resources=resources)), flush=True)
            if resources["blocked"]:
                raise RuntimeError("resource manager blocked work before publishing raw training host cache")
            return None
        if stop_requested is not None and stop_requested():
            raise InterruptedError("stopped before publishing raw training host cache; mapped views unchanged")
        result = MarketHostCache(arrays, originals, dict(cached), total_bytes)
        arrays.update(cached)
        print(json.dumps(dict(event="market-host-cache-ready", completed_bytes=completed, total_bytes=total_bytes,
            array_names=list(originals), partial_cache=len(originals) != len(arrays),
            elapsed_seconds=time.monotonic() - started, remaining_seconds=0.0)), flush=True)
        return result
    except BaseException:
        if result is not None:
            result.release()
        raise
    finally:
        cached.clear()
        target = None


def _row_count(rows: slice | np.ndarray, paths: int) -> int:
    return len(range(*rows.indices(paths))) if isinstance(rows, slice) else len(rows)


def assemble_market_batch(data: ScenarioData, rows: slice | np.ndarray, *, out: ArrayBatch | None = None) -> ArrayBatch:
    """Gather shared raw inputs once, retaining the existing feature-rounding order.

    Optional outputs must have the exact selected batch shapes and native feature,
    float64 price and float64 payoff dtypes. Generic scenario views use their
    existing indexing implementation instead of the market-specific assembly.
    """
    count = _row_count(rows, len(data.payoff))
    shapes = ((count, *data.features.shape[1:]), (count, *data.prices.shape[1:]), (count,))
    dtypes = (np.dtype(data.features.dtype), np.dtype("f8"), np.dtype("f8"))
    if out is None:
        out = tuple(np.empty(shape, dtype=dtype) for shape, dtype in zip(shapes, dtypes, strict=True))
    if any(values.shape != shape or values.dtype != dtype or not values.flags.writeable
           for values, shape, dtype in zip(out, shapes, dtypes, strict=True)):
        raise ValueError("batch outputs must have matching writable shapes and dtypes")
    features, prices, payoff = out
    feature_view, price_view = data.features, data.prices
    shared_market = (getattr(feature_view, "features", False) is True
        and getattr(price_view, "features", None) is False
        and getattr(feature_view, "arrays", None) is getattr(price_view, "arrays", object())
        and getattr(feature_view, "book", None) is getattr(price_view, "book", object())
        and getattr(feature_view, "kind", None) == getattr(price_view, "kind", None))
    if shared_market:
        values = {name: np.asarray(source[rows]) for name, source in feature_view.arrays.items()}
        book = feature_view.book
        starts = book["starts"]
        days = np.arange(21)
        strikes = values["spots"][:, starts] * book["levels"][None]
        known = np.where(days[None, :, None] >= starts[None, None], strikes[:, None], 0.0)
        spot, background = values["spots"][:, :20], values["backgrounds"][:, :20]
        fixing = np.where(np.arange(20)[None] >= 10, np.log(values["spots"][:, 10:11]), 0.0)
        alive = (days[:20, None] >= starts) & (days[:20, None] < book["expiries"])
        np.concatenate((values["core_features"], np.log(spot)[..., None], np.log(background)[..., None],
            fixing[..., None], values["option_prices"][:, :20], known[:, :20],
            np.broadcast_to(alive, values["option_prices"][:, :20].shape),
            values["factor_prices"][:, :20], values["daily_calls"][..., None]),
            axis=-1, out=features, casting="unsafe")
        if feature_view.kind == "cohorts":
            if getattr(price_view, "with_spot", False):
                np.concatenate((values["spots"][..., None], values["option_prices"],
                    values["spots"][..., None], known, np.zeros_like(values["spots"])[..., None]), axis=-1, out=prices)
            else:
                np.concatenate((values["spots"][..., None], values["option_prices"], known), axis=-1, out=prices)
        elif feature_view.kind == "combined":
            calls = np.pad(values["daily_calls"], ((0, 0), (0, 1)))
            np.concatenate((values["spots"][..., None], values["option_prices"], values["factor_prices"],
                values["spots"][..., None], calls[..., None], known), axis=-1, out=prices)
        else:
            calls = np.pad(values["daily_calls"], ((0, 0), (0, 1)))
            np.concatenate((values["spots"][..., None], values["factor_prices"], calls[..., None]), axis=-1, out=prices)
    else:
        np.copyto(features, data.features[rows])
        np.copyto(prices, data.prices[rows], casting="unsafe")
    np.copyto(payoff, data.payoff[rows], casting="unsafe")
    return out


@dataclass
class _PinnedSlot:
    tensors: TensorBatch
    copied: torch.cuda.Event | None = None


class MarketBatchIterator:
    """Yield unchanged batches with two pinned slots and bounded producer threads.

    Use as a context manager so early exits join the producer and finish pending
    transfers. ``order`` is read-only caller-owned sample order; ``start`` is the
    saved offset into that order. No generator or training state is advanced.
    CUDA results own separate device allocations, so retaining an earlier batch
    cannot expose it to later slot reuse. CPU execution is synchronous and does
    not inspect or initialize CUDA.

    CUDA graph capture, including any compilation warmup that can capture, must
    hold ``GPU_CAPTURE_LOCK`` to exclude producer-thread CUDA runtime calls.
    """

    def __init__(self, data: ScenarioData, *, env: BaseEnvironment, batch_size: int,
                 order: np.ndarray | None = None, start: int = 0, prefetch_workers: int = 1) -> None:
        verify_inp(isinstance(batch_size, int) and batch_size > 0, lambda: "batch_size must be a positive integer")
        verify_inp(isinstance(prefetch_workers, int) and 1 <= prefetch_workers <= 2,
            lambda: "prefetch_workers must be one or two for the two pinned slots")
        verify_inp(order is None or (isinstance(order, np.ndarray) and order.ndim == 1
            and np.issubdtype(order.dtype, np.integer)), lambda: "order must be a one-dimensional integer array")
        self.total = len(data.payoff) if order is None else len(order)
        verify_inp(isinstance(start, int) and 0 <= start <= self.total, lambda: "start must be a valid saved batch offset")
        verify_inp(env.device.type in {"cpu", "cuda"}, lambda: "batch device must be CPU or allocated CUDA")
        self.data, self.env, self.batch_size, self.order = data, env, batch_size, order
        self.next_start = start
        self.closed = False
        self.executor: ThreadPoolExecutor | None = None
        self.pending: deque[tuple[_PinnedSlot, Future]] = deque()
        self.slots: list[_PinnedSlot] = []
        self.copy_stream: torch.cuda.Stream | None = None
        if env.device.type == "cuda" and start < self.total:
            from quantdata.gromm2.hedging_policy import _require_cuda_visibility

            _require_cuda_visibility()
            try:
                with torch.cuda.device(env.device):
                    self.copy_stream = torch.cuda.Stream(device=env.device)
                    feature_dtype = torch.from_numpy(np.empty(0, dtype=data.features.dtype)).dtype
                    shapes = ((batch_size, *data.features.shape[1:]), (batch_size, *data.prices.shape[1:]), (batch_size,))
                    dtypes = (feature_dtype, torch.float64, torch.float64)
                    for _ in range(2):
                        self.slots.append(_PinnedSlot(tuple(torch.empty(shape, dtype=dtype, pin_memory=True)
                            for shape, dtype in zip(shapes, dtypes, strict=True))))
                self.executor = ThreadPoolExecutor(max_workers=prefetch_workers, thread_name_prefix="gromm2-market-batches")
                for slot in self.slots:
                    self._submit(slot)
            except BaseException:
                self.close()
                raise

    def __enter__(self) -> MarketBatchIterator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __iter__(self) -> MarketBatchIterator:
        return self

    def _rows(self) -> slice | np.ndarray:
        start = self.next_start
        self.next_start = min(start + self.batch_size, self.total)
        return slice(start, self.next_start) if self.order is None else self.order[start:self.next_start]

    def _submit(self, slot: _PinnedSlot) -> None:
        if self.next_start < self.total:
            assert self.executor is not None
            self.pending.append((slot, self.executor.submit(self._prepare, slot, self._rows())))

    def _prepare(self, slot: _PinnedSlot, rows: slice | np.ndarray) -> tuple[TensorBatch, torch.cuda.Event]:
        with GPU_CAPTURE_LOCK:
            if slot.copied is not None:
                slot.copied.synchronize()
        count = _row_count(rows, len(self.data.payoff))
        host = tuple(tensor[:count] for tensor in slot.tensors)
        assemble_market_batch(self.data, rows, out=tuple(tensor.numpy() for tensor in host))
        assert self.copy_stream is not None
        with GPU_CAPTURE_LOCK, torch.cuda.device(self.env.device), torch.cuda.stream(self.copy_stream):
            device = tuple(tensor.to(device=self.env.device, dtype=dtype, non_blocking=True)
                for tensor, dtype in zip(host, (self.env.fdtype, torch.float64, torch.float64), strict=True))
            copied = torch.cuda.Event()
            copied.record(self.copy_stream)
            slot.copied = copied
        return device, copied

    def __next__(self) -> TensorBatch:
        if self.closed:
            raise StopIteration
        if self.env.device.type == "cpu":
            if self.next_start >= self.total:
                self.close()
                raise StopIteration
            arrays = assemble_market_batch(self.data, self._rows())
            return tuple(torch.as_tensor(values, dtype=dtype, device=self.env.device)
                for values, dtype in zip(arrays, (self.env.fdtype, torch.float64, torch.float64), strict=True))
        if not self.pending:
            self.close()
            raise StopIteration
        slot, future = self.pending.popleft()
        try:
            tensors, copied = future.result()
            consumer = torch.cuda.current_stream(self.env.device)
            consumer.wait_event(copied)
            for tensor in tensors:
                tensor.record_stream(consumer)
            self._submit(slot)
            return tensors
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Join bounded preparation and protect pinned buffers until DMA finishes."""
        if self.closed:
            return
        self.closed = True
        try:
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
            if self.copy_stream is not None:
                self.copy_stream.synchronize()
        finally:
            self.pending.clear()
            self.slots.clear()
