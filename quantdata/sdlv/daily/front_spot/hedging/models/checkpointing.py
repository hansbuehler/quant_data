"""Shared checkpoint identity and audit helpers for hedging models."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Mapping

import numpy as np
from filelock import FileLock
from numpy.typing import NDArray

CHECKPOINT_FORMAT_VERSION = 2
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def training_data_fingerprint(
    *,
    arrays: Mapping[str, NDArray[np.generic]],
    metadata: Mapping[str, object],
) -> str:
    """Return a stable SHA-256 identity for exact training inputs."""
    digest = hashlib.sha256()
    digest.update(_canonical_json(metadata).encode("utf-8"))
    for name in sorted(arrays):
        values = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(_canonical_json(values.shape).encode("ascii"))
        _update_array_bytes(digest, values)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Hash one file without retaining it in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def remove_completed_checkpoint(path: Path) -> None:
    """Remove one reproducible fit checkpoint under its training lock."""
    path = Path(path)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with FileLock(lock_path):
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)


def train_config_snapshot(config: object) -> dict[str, object]:
    """Serialize a dataclass training configuration for checkpoint audit."""
    if not is_dataclass(config) or isinstance(config, type):
        raise TypeError("training configuration must be a dataclass instance")
    snapshot = _jsonable(asdict(config))
    if not isinstance(snapshot, dict):
        raise TypeError("training configuration did not serialize to a mapping")
    return snapshot


def validate_train_config_change(
    saved: Mapping[str, object] | None,
    current: Mapping[str, object],
    *,
    ignored_fields: frozenset[str],
    immutable_fields: frozenset[str],
    allow_warm_start: bool,
    model_name: str,
) -> tuple[str, ...]:
    """Validate an explicitly requested warm start under changed controls."""
    if saved is None:
        raise RuntimeError(
            f"{model_name} checkpoint has no training-config snapshot"
        )
        return ()
    changed = tuple(
        sorted(
            key
            for key in set(saved) | set(current)
            if key not in ignored_fields and saved.get(key) != current.get(key)
        )
    )
    if not changed:
        return changed
    immutable_changed = tuple(sorted(set(changed) & immutable_fields))
    if immutable_changed:
        raise ValueError(
            f"{model_name} checkpoint is incompatible with changed training fields: "
            + ", ".join(immutable_changed)
        )
    if not allow_warm_start:
        raise ValueError(
            f"{model_name} checkpoint has changed training fields: "
            + ", ".join(changed)
            + "; set allow_changed_config_warm_start=True to continue training"
        )
    warnings.warn(
        f"Warm-starting {model_name} with changed training fields: "
        + ", ".join(changed),
        stacklevel=2,
    )
    return changed


def _update_array_bytes(digest: object, values: NDArray[np.generic]) -> None:
    if values.flags.c_contiguous:
        byte_view = memoryview(values).cast("B")
        for start in range(0, len(byte_view), _HASH_CHUNK_BYTES):
            digest.update(byte_view[start : start + _HASH_CHUNK_BYTES])
        return
    iterator = np.nditer(
        values,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="C",
        buffersize=max(1, _HASH_CHUNK_BYTES // max(values.dtype.itemsize, 1)),
    )
    for chunk in iterator:
        digest.update(memoryview(np.ascontiguousarray(chunk)).cast("B"))


def _canonical_json(value: object) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported checkpoint metadata type {type(value).__name__}")
