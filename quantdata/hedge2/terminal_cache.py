"""Immutable remaining-horizon forecasts shared without sharing fitted decisions."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import asdict, fields
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from filelock import FileLock, Timeout

from quantdata.hedge2.terminal_market import TerminalMarketPaths


class ForecastCheckpointStop(InterruptedError):
    """A stop request arrived while waiting for another forecast writer."""


def _outside_git(directory: Path) -> None:
    resolved = directory.resolve()
    if any((parent / ".git").is_file() or (parent / ".git" / "HEAD").is_file()
           for parent in (resolved, *resolved.parents)):
        raise ValueError("Terminal forecast artifacts must be outside Git working trees")


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_json(path: Path, values: dict[str, object]) -> None:
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(values, indent=2) + "\n")
    temporary.replace(path)


@contextmanager
def _writer_lock(directory: Path, *, progress: Callable[[str], None] | None,
                 stop_requested: Callable[[], bool] | None) -> Iterator[None]:
    lock = FileLock(str(directory / "forecast.lock"))
    started, last_progress = time.monotonic(), 0.
    try:
        while True:
            if stop_requested is not None and stop_requested():
                raise ForecastCheckpointStop("Checkpoint requested while awaiting a shared terminal forecast")
            try:
                lock.acquire(timeout=0)
                break
            except Timeout:
                now = time.monotonic()
                if progress is not None and now - last_progress >= 30.:
                    progress(f"terminal forecast waiting completed=0/1 elapsed_s={now-started:.1f} remaining_s=unknown")
                    last_progress = now
                time.sleep(.1)
        yield
    finally:
        if lock.is_locked:
            lock.release()


def _verified_record(directory: Path, identity: dict[str, object]) -> dict[str, object]:
    recorded = json.loads((directory / "forecast.json").read_text())
    if recorded["identity"] != identity:
        raise ValueError("Saved terminal forecast inputs changed; do not mix simulations on resume")
    if _digest(directory / "forecast.npz") != recorded["sha256"]:
        raise ValueError("A saved terminal forecast changed after completion")
    return recorded


def _load(directory: Path) -> TerminalMarketPaths:
    with np.load(directory / "forecast.npz", allow_pickle=False) as saved:
        if set(saved.files) != {field.name for field in fields(TerminalMarketPaths)}:
            raise ValueError("Saved terminal forecast schema is incompatible")
        values = {name: saved[name].copy() for name in saved.files}
    for name in ("model_date", "seed", "elapsed_day", "future_spread_convention", "alpha_convention"):
        values[name] = values[name].item()
    values["claim_labels"] = tuple(values["claim_labels"].tolist())
    return TerminalMarketPaths(**values)


def _stored(directory: Path, *, identity: dict[str, object], generate: Callable[[], TerminalMarketPaths],
            progress: Callable[[str], None] | None,
            stop_requested: Callable[[], bool] | None) -> tuple[TerminalMarketPaths, dict[str, object]]:
    directory.mkdir(parents=True, exist_ok=True)
    with _writer_lock(directory, progress=progress, stop_requested=stop_requested):
        if (directory / "forecast.json").exists():
            recorded = _verified_record(directory, identity)
            return _load(directory), recorded
        result = generate()
        temporary = directory / "forecast.partial.npz"
        np.savez_compressed(temporary, **asdict(result))
        temporary.replace(directory / "forecast.npz")
        recorded = {"identity": identity, "sha256": _digest(directory / "forecast.npz")}
        _atomic_json(directory / "forecast.json", recorded)
        return result, recorded


def saved_terminal_forecast(directory: Path, *, identity: dict[str, object],
                            generate: Callable[[], TerminalMarketPaths], cache_directory: Path | None = None,
                            progress: Callable[[str], None] | None = None,
                            stop_requested: Callable[[], bool] | None = None) -> TerminalMarketPaths:
    """Share exogenous draws, never policies, actual inventory or fitted normalization.

    The caller's identity must contain the dated state/law, frozen contracts,
    claim observation state, original numeraire, current costs, carry, source
    hashes, seed and simulation configuration. Objective, aversion, claim
    side and policy inventory do not affect the exogenous forecast law.
    Distinct claim payoffs remain distinct identities.

    Completed arrays are hash checked and never regenerated or overwritten.
    Episode-local paths are hard links on one filesystem; an explicit copy
    fallback supports external caches on a different filesystem. Existing
    episode files remain readable if the shared cache is later removed.
    """
    _outside_git(directory)
    normalized = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
    if (directory / "forecast.json").exists():
        with _writer_lock(directory, progress=progress, stop_requested=stop_requested):
            _verified_record(directory, normalized)
            return _load(directory)
    if cache_directory is None:
        return _stored(directory, identity=normalized, generate=generate,
            progress=progress, stop_requested=stop_requested)[0]
    _outside_git(cache_directory)
    cache_key = hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    shared = cache_directory / cache_key
    result, recorded = _stored(shared, identity=normalized, generate=generate,
        progress=progress, stop_requested=stop_requested)
    directory.mkdir(parents=True, exist_ok=True)
    with _writer_lock(directory, progress=progress, stop_requested=stop_requested):
        if (directory / "forecast.json").exists():
            local = _verified_record(directory, normalized)
            if local["sha256"] != recorded["sha256"]:
                raise ValueError("Identical forecast inputs produced different completed arrays")
            return result
        temporary = directory / "forecast.partial.npz"
        temporary.unlink(missing_ok=True)
        storage_mode = "hardlink"
        try:
            os.link(shared / "forecast.npz", temporary)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copyfile(shared / "forecast.npz", temporary)
            storage_mode = "copy_cross_filesystem"
        temporary.replace(directory / "forecast.npz")
        _atomic_json(directory / "forecast.json", {**recorded,
            "shared_cache_entry": str(shared.resolve()), "storage_mode": storage_mode})
    return result
