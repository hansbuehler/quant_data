"""Read complete contiguous HDF5 scenarios without loading million-path tensors."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np

from quantdata.gromm2.hedging_policy import ScenarioData, _write_json


def scenario_digest(path: Path) -> str:
    """Cache a streamed file digest only while its file identity remains unchanged."""
    stat = path.stat()
    identity = {"size": stat.st_size, "modified_ns": stat.st_mtime_ns, "inode": stat.st_ino}
    target = path.with_suffix(".sha256.json")
    if target.exists():
        report = json.loads(target.read_text())
        if report["identity"] == identity:
            return str(report["sha256"])
    digest = hashlib.sha256()
    started = last_report = time.monotonic()
    completed = 0
    print(json.dumps({"event": "scenario_digest_started", "path": str(path),
        "completed_bytes": 0, "total_bytes": stat.st_size, "elapsed_seconds": 0.0}), flush=True)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            completed += len(block)
            now = time.monotonic()
            if now - last_report >= 60:
                elapsed = now - started
                print(json.dumps({"event": "scenario_digest_progress", "path": str(path),
                    "completed_bytes": completed, "total_bytes": stat.st_size, "elapsed_seconds": elapsed,
                    "remaining_seconds": elapsed * (stat.st_size - completed) / max(completed, 1)}), flush=True)
                last_report = now
    if path.stat().st_mtime_ns != stat.st_mtime_ns:
        raise RuntimeError("scenario file changed while calculating its digest")
    value = digest.hexdigest()
    _write_json(target, {"identity": identity, "sha256": value})
    print(json.dumps({"event": "scenario_digest_complete", "path": str(path),
        "completed_bytes": completed, "total_bytes": stat.st_size,
        "elapsed_seconds": time.monotonic() - started}), flush=True)
    return value


def map_dataset(path: Path, dataset: h5py.Dataset) -> np.memmap:
    """Expose a completed, uncompressed contiguous dataset as read-only storage."""
    offset = dataset.id.get_offset()
    if dataset.chunks is not None or dataset.compression is not None or offset is None:
        raise ValueError("million-path training requires fully allocated contiguous, uncompressed datasets")
    return np.memmap(path, dtype=dataset.dtype, mode="r", offset=offset, shape=dataset.shape, order="C")


class ScenarioView:
    """Materialize only selected rows; information is matched across trading universes."""

    def __init__(self, base: np.ndarray, fixing_prices: np.ndarray, *, features: bool,
                 fixing_day: int, terminal_day: int) -> None:
        self.base, self.fixing_prices, self.features = base, fixing_prices, features
        self.fixing_day, self.terminal_day = fixing_day, terminal_day
        extra = fixing_prices.shape[-1] * (2 if features else 1)
        self.shape = (*base.shape[:-1], base.shape[-1] + extra)
        self.ndim, self.dtype = base.ndim, base.dtype

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, rows: slice | np.ndarray) -> np.ndarray:
        base = np.asarray(self.base[rows])
        extra = np.asarray(self.fixing_prices[rows])
        if not self.features:
            return np.concatenate((base, extra), axis=-1)
        extra = extra[:, :-1].astype(base.dtype)
        days = np.arange(base.shape[1])
        alive = (days >= self.fixing_day) & (days < self.terminal_day)
        mask = np.broadcast_to(alive[None, :, None], extra.shape).astype(base.dtype)
        return np.concatenate((base, extra, mask), axis=-1)


def load_mapped_scenarios(path: Path, *, expected_split: str, with_fixing_options: bool) -> ScenarioData:
    """Both universes observe the same causal quotes but have different action books."""
    with h5py.File(path, "r") as stream:
        identity = json.loads(str(stream.attrs["identity"]))
        config = identity["config"]
        expected = int(config["paths"])
        if identity["split"] != expected_split or int(stream.attrs.get("completed_paths", 0)) != expected:
            raise ValueError("scenario sets must be complete and have the requested independent split")
        if expected < 4 or expected % 2:
            raise ValueError("complete antithetic pairs are required")
        base_features = map_dataset(path, stream["features"])
        base_prices = map_dataset(path, stream["prices"])
        payoff = map_dataset(path, stream["payoff"])
        fixing_prices = map_dataset(path, stream["fixing_option_prices"])
        fixing_day, terminal_day = int(config["fixing_day"]), int(config["terminal_day"])
        levels = tuple(map(float, config["fixing_hedge_moneyness"]))
        expiries = tuple(map(int, stream["option_expiry_days"][:]))
        strikes = tuple(map(float, stream["option_strikes"][:]))
        calls = tuple(map(bool, stream["option_is_call"][:]))
        if (base_features.shape[0] != expected or base_prices.shape != (expected, terminal_day + 1, len(expiries) + 1)
                or fixing_prices.shape != (expected, terminal_day + 1, len(levels)) or payoff.shape != (expected,)):
            raise ValueError("incompatible memory-mapped scenario geometry")
        features = ScenarioView(base_features, fixing_prices, features=True,
            fixing_day=fixing_day, terminal_day=terminal_day)
        prices = base_prices
        if with_fixing_options:
            prices = ScenarioView(base_prices, fixing_prices, features=False,
                fixing_day=fixing_day, terminal_day=terminal_day)
            expiries += (terminal_day,) * len(levels)
            strikes += levels
            calls += tuple(level >= 1.0 for level in levels)
    return ScenarioData(features, prices, payoff, expiries, identity, scenario_digest(path), strikes, calls)
