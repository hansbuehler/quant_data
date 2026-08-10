"""Versioned block cache for path-matched expiry and incremental markets."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from cdxcore.err import verify_inp

_SCHEMA_VERSION = "1.3"
_COMMON_FIELDS = (
    "C",
    "spot",
    "delta",
    "expiries",
    "log_strikes",
    "xstate",
    "log_path_weight",
    "importance_component",
)
_EXPIRY_FIELDS = ("dC", "expiry_dS", "option_expired")
_INCREMENTAL_FIELDS = (
    "dC",
    "dS",
    "option_dS",
    "option_settlement_dS",
    "option_expired",
)


@dataclass(frozen=True, kw_only=True)
class MatchedCachedBlock:
    """Two return horizons reconstructed from one cached path block."""

    start: int
    stop: int
    expiry: dict[str, np.ndarray]
    incremental: dict[str, np.ndarray]


class MatchedUniverseBlockCache:
    """Persist shared primitives once and horizon-specific returns separately."""

    def __init__(
        self,
        root: str | Path,
        *,
        cache_uid: str,
        configuration: Mapping[str, Any],
        num_samples: int,
    ) -> None:
        verify_inp(cache_uid != "", "matched cache UID cannot be empty")
        verify_inp(num_samples > 1, "matched cache requires at least two paths")
        self.cache_uid = cache_uid
        self.configuration = json.loads(json.dumps(configuration))
        self.num_samples = num_samples
        self.directory = Path(root) / cache_uid
        self.manifest_path = self.directory / "manifest.json"
        self._blocks: list[dict[str, Any]] = []

    @property
    def is_complete(self) -> bool:
        if not self.manifest_path.exists():
            return False
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        blocks = manifest.get("blocks", [])
        return (
            manifest.get("status") == "complete"
            and manifest.get("schema_version") == _SCHEMA_VERSION
            and manifest.get("cache_uid") == self.cache_uid
            and manifest.get("configuration") == self.configuration
            and manifest.get("num_samples") == self.num_samples
            and bool(blocks)
            and blocks[0]["start"] == 0
            and blocks[-1]["stop"] == self.num_samples
            and all((self.directory / item["file"]).exists() for item in blocks)
        )

    def begin(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._blocks = []
        self._write_manifest(status="writing")

    @staticmethod
    def _payload(block) -> dict[str, np.ndarray]:
        expiry = block.expiry
        incremental = block.incremental
        verify_inp(
            expiry["C"].shape[1] == 10 and incremental["C"].shape[1] == 9,
            "matched cache requires ten expiry cohorts and nine intervals",
        )
        payload: dict[str, np.ndarray] = {}
        for name in _COMMON_FIELDS:
            if name not in expiry:
                continue
            value = np.asarray(expiry[name])
            if name not in {"expiries", "log_path_weight", "importance_component"}:
                verify_inp(
                    np.array_equal(incremental[name], value[:, :9]),
                    lambda name=name: (
                        f"incremental common field {name!r} is not an exact "
                        "slice of the expiry path block"
                    ),
                )
            elif name in incremental:
                verify_inp(
                    np.array_equal(incremental[name], value),
                    lambda name=name: f"shared field {name!r} differs by horizon",
                )
            payload[f"common__{name}"] = value
        for name in _EXPIRY_FIELDS:
            payload[f"expiry__{name}"] = np.asarray(expiry[name])
        for name in _INCREMENTAL_FIELDS:
            payload[f"incremental__{name}"] = np.asarray(incremental[name])
        for name, value in payload.items():
            if not np.all(np.isfinite(value)):
                raise FloatingPointError(
                    f"matched cache field {name!r} contains NaN or infinity"
                )
        return payload

    def write(self, block) -> None:
        verify_inp(
            0 <= block.start < block.stop <= self.num_samples,
            lambda: f"invalid matched block {block.start}:{block.stop}",
        )
        expected = self._blocks[-1]["stop"] if self._blocks else 0
        verify_inp(
            block.start == expected,
            lambda: f"matched block starts at {block.start}, expected {expected}",
        )
        payload = self._payload(block)
        filename = f"block-{block.start:09d}-{block.stop:09d}.npz"
        destination = self.directory / filename
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez(handle, **payload)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        self._blocks.append(
            {
                "start": int(block.start),
                "stop": int(block.stop),
                "file": filename,
                "fields": sorted(payload),
            }
        )
        self._write_manifest(status="writing")

    def complete(self, *, diagnostics: Mapping[str, Any]) -> None:
        verify_inp(
            bool(self._blocks) and self._blocks[-1]["stop"] == self.num_samples,
            "cannot complete an incomplete matched universe cache",
        )
        self._write_manifest(status="complete", diagnostics=diagnostics)

    def _write_manifest(
        self,
        *,
        status: str,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        value: dict[str, Any] = {
            "status": status,
            "schema_version": _SCHEMA_VERSION,
            "cache_uid": self.cache_uid,
            "configuration": self.configuration,
            "num_samples": self.num_samples,
            "storage_semantics": (
                "one coherent path block; common prices, states, deltas, "
                "weights and contract descriptors stored once; expiry and "
                "incremental returns stored separately"
            ),
            "blocks": self._blocks,
        }
        if diagnostics is not None:
            value["diagnostics"] = diagnostics
        temporary = self.manifest_path.with_name(
            f".{self.manifest_path.name}.{os.getpid()}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
            )
            os.replace(temporary, self.manifest_path)
        finally:
            temporary.unlink(missing_ok=True)

    def manifest(self) -> dict[str, Any]:
        verify_inp(self.is_complete, "matched universe cache is incomplete")
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def iter_blocks(self) -> Iterator[MatchedCachedBlock]:
        """Read one path block and reconstruct exact shared horizon views."""

        manifest = self.manifest()
        for item in manifest["blocks"]:
            with np.load(self.directory / item["file"], allow_pickle=False) as data:
                values = {name: data[name] for name in data.files}
            common = {
                name.removeprefix("common__"): value
                for name, value in values.items()
                if name.startswith("common__")
            }
            expiry = {
                name.removeprefix("expiry__"): value
                for name, value in values.items()
                if name.startswith("expiry__")
            }
            incremental = {
                name.removeprefix("incremental__"): value
                for name, value in values.items()
                if name.startswith("incremental__")
            }
            expiry.update(common)
            for name, value in common.items():
                incremental[name] = (
                    value
                    if name in {"expiries", "log_path_weight", "importance_component"}
                    else value[:, :9]
                )
            yield MatchedCachedBlock(
                start=int(item["start"]),
                stop=int(item["stop"]),
                expiry=expiry,
                incremental=incremental,
            )


def migrate_legacy_option_settlement_cache(
    *,
    root: str | Path,
    source_uid: str,
    destination_uid: str,
    destination_configuration: Mapping[str, Any],
    num_samples: int,
    source_schema_version: str = "1.0",
    progress: Callable[[int, int], None] | None = None,
) -> MatchedUniverseBlockCache:
    """Upgrade a matched cache to close an option hedge at option expiry.

    In schema 1.0, incremental ``option_dS`` contains the option settlement
    endpoint.  Schema 1.2 retains that endpoint separately in
    ``option_settlement_dS``.  The current schema uses the settlement endpoint
    for the option's delta hedge as well.  No simulated state, option return,
    or likelihood weight is changed.
    """

    verify_inp(
        source_schema_version in {"1.0", "1.2"},
        lambda: f"unsupported source schema {source_schema_version!r}",
    )

    root = Path(root)
    source_directory = root / source_uid
    source_manifest_path = source_directory / "manifest.json"
    verify_inp(
        source_manifest_path.exists(),
        lambda: f"legacy matched-cache manifest does not exist: {source_manifest_path}",
    )
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    verify_inp(
        source.get("status") == "complete"
        and source.get("schema_version") == source_schema_version
        and source.get("cache_uid") == source_uid
        and source.get("num_samples") == num_samples,
        "legacy matched-cache manifest is incomplete or incompatible",
    )
    source_configuration = dict(source.get("configuration", {}))
    expected_source_configuration = json.loads(
        json.dumps(destination_configuration)
    )
    expected_source_configuration["schema_version"] = source_schema_version
    verify_inp(
        source_configuration == expected_source_configuration,
        "legacy and destination matched-cache configurations differ beyond schema",
    )
    blocks = source.get("blocks", [])
    verify_inp(
        bool(blocks)
        and blocks[0]["start"] == 0
        and blocks[-1]["stop"] == num_samples
        and all(
            left["stop"] == right["start"]
            for left, right in zip(blocks, blocks[1:])
        ),
        "legacy matched-cache blocks are not complete and contiguous",
    )

    destination = MatchedUniverseBlockCache(
        root,
        cache_uid=destination_uid,
        configuration=destination_configuration,
        num_samples=num_samples,
    )
    destination.begin()
    for block_index, item in enumerate(blocks, start=1):
        source_file = source_directory / item["file"]
        verify_inp(
            source_file.exists(),
            lambda source_file=source_file: (
                f"legacy matched-cache block does not exist: {source_file}"
            ),
        )
        with np.load(source_file, allow_pickle=False) as data:
            values = {name: data[name] for name in data.files}
        verify_inp(
            "incremental__option_dS" in values and "incremental__dS" in values,
            "source block does not contain incremental option endpoints",
        )
        common = {
            name.removeprefix("common__"): value
            for name, value in values.items()
            if name.startswith("common__")
        }
        expiry = {
            name.removeprefix("expiry__"): value
            for name, value in values.items()
            if name.startswith("expiry__")
        }
        incremental = {
            name.removeprefix("incremental__"): value
            for name, value in values.items()
            if name.startswith("incremental__")
        }
        settlement_dS = incremental.get(
            "option_settlement_dS", incremental["option_dS"]
        )
        common_dS = incremental["dS"]
        verify_inp(
            settlement_dS.ndim == 3
            and common_dS.shape == settlement_dS.shape[:2],
            "legacy option and common spot endpoints have inconsistent geometry",
        )
        incremental["option_settlement_dS"] = settlement_dS
        incremental["option_dS"] = settlement_dS.copy()
        expiry.update(common)
        for name, value in common.items():
            incremental[name] = (
                value
                if name in {"expiries", "log_path_weight", "importance_component"}
                else value[:, :9]
            )
        destination.write(
            MatchedCachedBlock(
                start=int(item["start"]),
                stop=int(item["stop"]),
                expiry=expiry,
                incremental=incremental,
            )
        )
        if progress is not None:
            progress(block_index, len(blocks))
    destination.complete(diagnostics=source["diagnostics"])
    verify_inp(destination.is_complete, "migrated matched cache is incomplete")
    return destination


__all__ = [
    "MatchedCachedBlock",
    "MatchedUniverseBlockCache",
    "migrate_legacy_option_settlement_cache",
]
