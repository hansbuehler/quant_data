"""Temporary disk-backed arrays for one derived required-run case."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from cdxcore.err import verify_inp

from .required_runs import DerivedCaseArrays, RequiredCase, derive_case_arrays

_SCHEMA_VERSION = "1.0"


class DerivedCaseStore:
    """Materialize one case without retaining the full universe in memory."""

    def __init__(
        self,
        root: str | Path,
        *,
        case: RequiredCase,
        source_cache_uid: str,
    ) -> None:
        self.case = case
        verify_inp(source_cache_uid != "", "source cache UID cannot be empty")
        self.source_cache_uid = source_cache_uid
        label = (
            f"{case.generator}-{case.trading_frequency}-{case.path_count}-"
            f"{case.return_horizon}-{case.hedge}-{case.cost_rate:.0e}"
        )
        self.directory = Path(root) / source_cache_uid / label
        self.manifest_path = self.directory / "manifest.json"

    @property
    def is_complete(self) -> bool:
        if not self.manifest_path.exists():
            return False
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            value.get("status") == "complete"
            and value.get("schema_version") == _SCHEMA_VERSION
            and value.get("source_cache_uid") == self.source_cache_uid
            and all((self.directory / filename).exists() for filename in value["files"].values())
        )

    def build(self, blocks: Iterable) -> None:
        """Stream matched base blocks into one derived-case memmap."""

        self.directory.mkdir(parents=True, exist_ok=True)
        files = {
            "state": "state.npy",
            "returns": "returns.npy",
            "costs": "costs.npy",
            "numerical_band": "numerical-band.npy",
            "log_path_weight": "log-path-weight.npy",
            "importance_component": "importance-component.npy",
        }
        arrays: dict[str, np.memmap] | None = None
        descriptors: dict[str, object] | None = None
        last_stop = 0
        for block in blocks:
            universe = getattr(block, self.case.return_horizon)
            derived = derive_case_arrays(
                universe,
                hedge=self.case.hedge,
                cost_rate=self.case.cost_rate,
            )
            state = np.asarray(universe["xstate"], dtype=np.float32)
            rows = block.stop - block.start
            verify_inp(
                block.start == last_stop and state.shape[0] == rows,
                "derived case blocks must be contiguous",
            )
            if arrays is None:
                shapes = {
                    "state": (self.case.path_count, *state.shape[1:]),
                    "returns": (
                        self.case.path_count,
                        *derived.returns.shape[1:],
                    ),
                    "costs": (
                        self.case.path_count,
                        *derived.costs.shape[1:],
                    ),
                    "numerical_band": (
                        self.case.path_count,
                        *derived.numerical_band.shape[1:],
                    ),
                    "log_path_weight": (self.case.path_count,),
                    "importance_component": (self.case.path_count,),
                }
                dtypes = {
                    "state": np.float32,
                    "returns": np.float32,
                    "costs": np.float32,
                    "numerical_band": np.float32,
                    "log_path_weight": np.float64,
                    "importance_component": np.int16,
                }
                arrays = {
                    name: np.lib.format.open_memmap(
                        self.directory / files[name],
                        mode="w+",
                        dtype=dtypes[name],
                        shape=shape,
                    )
                    for name, shape in shapes.items()
                }
                descriptors = {
                    "instrument_kind": list(derived.instrument_kind),
                    "expiry_index": derived.expiry_index.astype(int).tolist(),
                    "strike_index": derived.strike_index.astype(int).tolist(),
                }
            assert arrays is not None and descriptors is not None
            interval = slice(block.start, block.stop)
            arrays["state"][interval] = state
            arrays["returns"][interval] = derived.returns
            arrays["costs"][interval] = derived.costs
            arrays["numerical_band"][interval] = derived.numerical_band
            arrays["log_path_weight"][interval] = np.asarray(
                universe.get("log_path_weight", np.zeros(rows)), dtype=np.float64
            )
            arrays["importance_component"][interval] = np.asarray(
                universe.get("importance_component", np.zeros(rows)),
                dtype=np.int16,
            )
            last_stop = block.stop
        verify_inp(
            arrays is not None and last_stop == self.case.path_count,
            "derived case store did not receive the complete path sample",
        )
        for value in arrays.values():
            value.flush()
        manifest = {
            "status": "complete",
            "schema_version": _SCHEMA_VERSION,
            "case": self.case.__dict__,
            "source_cache_uid": self.source_cache_uid,
            "files": files,
            "descriptors": descriptors,
        }
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

    def arrays(self) -> tuple[np.ndarray, DerivedCaseArrays, np.ndarray, np.ndarray]:
        """Open state, case, likelihood, and component memmaps read-only."""

        verify_inp(self.is_complete, "derived case store is incomplete")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        values = {
            name: np.load(self.directory / filename, mmap_mode="r")
            for name, filename in manifest["files"].items()
        }
        descriptors = manifest["descriptors"]
        derived = DerivedCaseArrays(
            returns=values["returns"],
            costs=values["costs"],
            numerical_band=values["numerical_band"],
            instrument_kind=tuple(descriptors["instrument_kind"]),
            expiry_index=np.asarray(descriptors["expiry_index"], dtype=np.int32),
            strike_index=np.asarray(descriptors["strike_index"], dtype=np.int32),
            return_horizon=self.case.return_horizon,
            hedge=self.case.hedge,
            cost_rate=self.case.cost_rate,
        )
        return (
            values["state"],
            derived,
            values["log_path_weight"],
            values["importance_component"],
        )

    def cleanup(self) -> None:
        """Remove only the known temporary files owned by this case store.

        Cleanup is idempotent because an interrupted or duplicate worker may
        have already removed the shared temporary store after all arrays were
        opened.  A present but incomplete store remains protected.
        """

        if not self.directory.exists():
            return

        verify_inp(self.is_complete, "refusing to clean an incomplete case store")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        for filename in manifest["files"].values():
            (self.directory / filename).unlink()
        self.manifest_path.unlink()
        self.directory.rmdir()
        try:
            self.directory.parent.rmdir()
        except OSError:
            pass
        try:
            self.directory.parent.parent.rmdir()
        except OSError:
            pass


__all__ = ["DerivedCaseStore"]
