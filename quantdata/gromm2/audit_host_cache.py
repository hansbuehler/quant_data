"""Optional exact raw host copies for completed conditional-audit populations."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import RLock

import h5py
import numpy as np
from cdxcore.err import verify_inp

from quantdata.gromm2.market_batching import _host_cache_resources


def _source_key(stream: h5py.File) -> tuple[str, str]:
    return os.path.abspath(stream.filename), str(stream.attrs.get('identity', ''))


def _path_range(rows: slice, count: int) -> tuple[int, int]:
    verify_inp(isinstance(rows, slice) and rows.step in {None, 1},
        lambda: 'audit cache reads require a contiguous path slice')
    start, stop, _ = rows.indices(count)
    return start, max(start, stop)


def _declined(resources: dict, *, event: str, **values) -> None:
    print(json.dumps(dict(event=event, resources=resources, **values)), flush=True)
    if resources['blocked']:
        raise RuntimeError('resource manager blocked work; raw audit host cache released or declined')


@dataclass
class AuditHostCache:
    """Own pageable raw arrays without changing HDF5, precision or normalization.

    Call ``check_headroom`` periodically, including during resident GPU epochs,
    and ``release`` in the run's finally block. Returned views may temporarily
    retain their backing array after release; consumers must discard each block
    after use rather than retain a population of views. Sources must remain
    immutable completed populations for the lifetime of the cache.
    """

    _source: tuple[str, str]
    _rows: tuple[int, int]
    _geometry: dict[str, tuple[tuple[int, ...], np.dtype]]
    _cached: dict[str, np.ndarray]
    bytes_cached: int
    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)

    @property
    def array_names(self) -> tuple[str, ...]:
        """Return the fields still retained in owned host memory."""
        with self._lock:
            return tuple(self._cached)

    @property
    def path_range(self) -> tuple[int, int]:
        """Return the source path interval covered by cached fields."""
        return self._rows

    def block(self, stream: h5py.File, field: str, rows: slice) -> np.ndarray:
        """Read an exact raw slice, falling back to HDF5 outside cache coverage."""
        if _source_key(stream) != self._source:
            raise ValueError('audit host cache belongs to a different source population')
        dataset = stream[field]
        start, stop = _path_range(rows, dataset.shape[0])
        if field in self._geometry and (dataset.shape, dataset.dtype) != self._geometry[field]:
            raise ValueError('audit source geometry or dtype changed while cached')
        with self._lock:
            cached = self._cached.get(field)
            if cached is not None and self._rows[0] <= start <= stop <= self._rows[1]:
                return cached[start - self._rows[0]:stop - self._rows[0]]
        return np.asarray(dataset[start:stop])

    def release(self) -> None:
        """Release owned references; subsequent block lookups use HDF5."""
        with self._lock:
            self._cached.clear()
            self.bytes_cached = 0

    def check_headroom(self) -> bool:
        """Release on WARN/unreadable probes, and release then raise on STOP."""
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
            _declined(resources, event='audit-host-cache-released', released_bytes=released_bytes)
        return allowed


def read_audit_block(stream: h5py.File, field: str, rows: slice, *,
                     cache: AuditHostCache | None = None) -> np.ndarray:
    """Read unchanged values with or without an optional raw host cache."""
    if cache is not None:
        return cache.block(stream, field, rows)
    start, stop = _path_range(rows, stream[field].shape[0])
    return np.asarray(stream[field][start:stop])


def cache_audit_arrays(stream: h5py.File, *, rows: slice = slice(None),
                       fields: tuple[str, ...] = ('features', 'returns'),
                       chunk_paths: int = 128, check_interval_seconds: float = 30.,
                       allow_partial: bool = True,
                       stop_requested: Callable[[], bool] | None = None) -> AuditHostCache | None:
    """Cache a requested full population or fold using resource-manager admission.

    If the requested fields do not all fit, optional partial admission selects
    the largest fields that fit. A caller can request a fold via ``rows``;
    uncached fields and other folds transparently retain HDF5 reads. No training
    mask, mean, scale, return or measure is calculated or changed here.
    """
    verify_inp(isinstance(chunk_paths, int) and chunk_paths > 0, lambda: 'chunk_paths must be positive')
    verify_inp(math.isfinite(check_interval_seconds) and 0. < check_interval_seconds <= 60.,
        lambda: 'cache resource/progress interval must lie in (0, 60] seconds')
    verify_inp(isinstance(allow_partial, bool), lambda: 'allow_partial must be boolean')
    verify_inp(bool(fields) and len(set(fields)) == len(fields) and set(fields) <= {'features', 'returns'},
        lambda: 'cache fields must be distinct raw features and/or returns')
    if stream.mode != 'r':
        raise ValueError('audit host caching requires a read-only completed population')
    count = stream[fields[0]].shape[0]
    if stream.attrs.get('completed_paths') != count:
        raise ValueError('audit host caching requires completed paths')
    start, stop = _path_range(rows, count)
    if start == stop:
        return None
    geometry = {name: (stream[name].shape, stream[name].dtype) for name in fields}
    if any(len(shape) < 2 or shape[0] != count or not np.issubdtype(dtype, np.number)
           for shape, dtype in geometry.values()):
        raise ValueError('audit cache fields must be numeric arrays with matching path axes')
    sizes = {name: (stop - start) * math.prod(shape[1:]) * dtype.itemsize
             for name, (shape, dtype) in geometry.items()}
    total = sum(sizes.values())
    resources, allowed = _host_cache_resources(total)
    if not allowed and allow_partial and not resources['blocked'] and not resources['warning']:
        selected = {}
        for name, size in sorted(sizes.items(), key=lambda item: item[1], reverse=True):
            resources, admitted = _host_cache_resources(sum(selected.values()) + size)
            if resources['blocked'] or resources['warning']:
                break
            if admitted:
                selected[name] = size
        if selected and not resources['blocked'] and not resources['warning']:
            sizes = selected
            total = sum(sizes.values())
            resources, allowed = _host_cache_resources(total)
    if not allowed:
        _declined(resources, event='audit-host-cache-declined', required_bytes=total)
        return None
    cached = {}
    target = None
    completed = 0
    started = checked = time.monotonic()
    print(json.dumps(dict(event='audit-host-cache-start', array_names=list(sizes), path_range=[start, stop],
        completed_bytes=0, total_bytes=total, elapsed_seconds=0., remaining_seconds=None)), flush=True)
    try:
        for name in sizes:
            shape, dtype = geometry[name]
            target = np.empty((stop - start, *shape[1:]), dtype=dtype)
            cached[name] = target
            row_bytes = math.prod(shape[1:]) * dtype.itemsize
            for offset in range(start, stop, chunk_paths):
                if stop_requested is not None and stop_requested():
                    raise InterruptedError('stopped while caching raw audit arrays; no cache published')
                end = min(offset + chunk_paths, stop)
                stream[name].read_direct(target, source_sel=np.s_[offset:end],
                    dest_sel=np.s_[offset - start:end - start])
                completed += (end - offset) * row_bytes
                now = time.monotonic()
                if now - checked >= check_interval_seconds:
                    resources, allowed = _host_cache_resources(total - completed)
                    elapsed = now - started
                    print(json.dumps(dict(event='audit-host-cache-progress', completed_bytes=completed,
                        total_bytes=total, elapsed_seconds=elapsed,
                        remaining_seconds=(total - completed) * elapsed / completed)), flush=True)
                    checked = time.monotonic()
                    if not allowed:
                        _declined(resources, event='audit-host-cache-aborted')
                        return None
            target.flags.writeable = False
        resources, allowed = _host_cache_resources(0)
        if not allowed:
            _declined(resources, event='audit-host-cache-aborted')
            return None
        if stop_requested is not None and stop_requested():
            raise InterruptedError('stopped before publishing raw audit host cache')
        result = AuditHostCache(_source_key(stream), (start, stop), geometry, dict(cached), total)
        print(json.dumps(dict(event='audit-host-cache-ready', array_names=list(sizes), path_range=[start, stop],
            completed_bytes=completed, total_bytes=total, elapsed_seconds=time.monotonic() - started,
            remaining_seconds=0.)), flush=True)
        return result
    finally:
        cached.clear()
        target = None
