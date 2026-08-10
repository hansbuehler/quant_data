"""Chunked, version-keyed cache for million-path PCA universes."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from cdxcore.err import verify_inp

from .experiment import ImpliedPCAExperiment
from .universe import PCAUniverseBlock

_UNIVERSE_CACHE_SCHEMA_VERSION = "0.2"


class PCAUniverseBlockCache:
    """Persist compact universe arrays one simulation block at a time.

    The directory name is derived from the complete versioned experiment, not
    from path count or a caller-selected filename. Blocks are never combined
    into a second in-memory universe.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        experiment: ImpliedPCAExperiment,
    ) -> None:
        self.experiment = experiment
        self.directory = Path(root) / experiment.filename
        self.manifest_path = self.directory / "manifest.json"
        self._blocks: list[dict[str, Any]] = []

    @property
    def _configuration(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self.experiment)))

    @property
    def is_complete(self) -> bool:
        if not self.manifest_path.exists():
            return False
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if (
            manifest.get("status") != "complete"
            or manifest.get("cache_uid") != self.experiment.filename
            or manifest.get("schema_version") != _UNIVERSE_CACHE_SCHEMA_VERSION
            or manifest.get("experiment_code_version")
            != ImpliedPCAExperiment.version.unique_id48
            or manifest.get("configuration") != self._configuration
        ):
            return False
        blocks = manifest.get("blocks", [])
        return (
            bool(blocks)
            and blocks[0]["start"] == 0
            and blocks[-1]["stop"] == self.experiment.num_samples
            and all((self.directory / block["file"]).exists() for block in blocks)
        )

    def begin(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._blocks = []
        self._write_manifest(status="writing")

    def write(self, block: PCAUniverseBlock) -> None:
        verify_inp(
            0 <= block.start < block.stop <= self.experiment.num_samples,
            lambda: f"invalid cached block interval {block.start}:{block.stop}",
        )
        expected_start = self._blocks[-1]["stop"] if self._blocks else 0
        verify_inp(
            block.start == expected_start,
            lambda: (
                f"cached blocks must be contiguous; expected {expected_start}, "
                f"got {block.start}"
            ),
        )
        rows = block.stop - block.start
        for name, value in block.universe.items():
            verify_inp(
                isinstance(value, np.ndarray)
                and np.issubdtype(value.dtype, np.number)
                and np.all(np.isfinite(value)),
                lambda name=name: (
                    f"cached universe field {name!r} must be a finite numeric array"
                ),
            )
            verify_inp(
                name == "expiries" or value.shape[0] == rows,
                lambda name=name, value=value: (
                    f"cached field {name!r} has {value.shape[0]} rows, expected {rows}"
                ),
            )
        filename = f"block-{block.start:09d}-{block.stop:09d}.npz"
        path = self.directory / filename
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **block.universe)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        self._blocks.append(
            {
                "start": block.start,
                "stop": block.stop,
                "file": filename,
                "fields": sorted(block.universe),
            }
        )
        self._write_manifest(status="writing")

    def complete(self, *, diagnostics: Mapping[str, Any]) -> None:
        verify_inp(
            bool(self._blocks)
            and self._blocks[-1]["stop"] == self.experiment.num_samples,
            "cannot complete an incomplete PCA universe cache",
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
            "schema_version": _UNIVERSE_CACHE_SCHEMA_VERSION,
            "cache_uid": self.experiment.filename,
            "experiment_code_version": (
                ImpliedPCAExperiment.version.unique_id48
            ),
            "configuration": self._configuration,
            "blocks": self._blocks,
            "market_scope": {
                "ticker": "SPX",
                "start": "2020-01-01",
                "end": "2025-08-30",
                "PCA_input": "mid_vols",
            },
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
        verify_inp(self.is_complete, "PCA universe cache is not complete")
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def update_diagnostics(self, diagnostics: Mapping[str, Any]) -> None:
        """Atomically attach completed downstream diagnostics to this cache."""

        manifest = self.manifest()
        manifest["diagnostics"] = diagnostics
        temporary = self.manifest_path.with_name(
            f".{self.manifest_path.name}.{os.getpid()}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
            )
            os.replace(temporary, self.manifest_path)
        finally:
            temporary.unlink(missing_ok=True)

    def iter_blocks(self) -> Iterator[PCAUniverseBlock]:
        """Load one compact cached block at a time with pickle disabled."""

        manifest = self.manifest()
        for block in manifest["blocks"]:
            with np.load(
                self.directory / block["file"], allow_pickle=False
            ) as state:
                universe = {name: state[name] for name in state.files}
            yield PCAUniverseBlock(
                start=int(block["start"]),
                stop=int(block["stop"]),
                universe=universe,
            )


__all__ = ["PCAUniverseBlockCache"]
