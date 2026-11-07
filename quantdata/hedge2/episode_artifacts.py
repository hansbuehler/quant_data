"""Publish complete episode evidence, binding cash records to inputs and checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantdata.hedge2.episodes import EpisodeResult

REQUIRED_ARTIFACTS = ("summary.parquet", "trades.parquet", "initial_book.parquet",
                      "positions.npz", "diagnostics.json", "input_hashes.json",
                      "steps/episode_checkpoint.npz")


def publish_episode(*, directory: Path, result: EpisodeResult, origin: str,
                    family: str, configuration_sha256: str) -> None:
    """Publish the completion marker last, only for an internally consistent final checkpoint."""
    manifest_path = directory / "completed.json"
    if manifest_path.exists():
        raise ValueError("A completed episode is immutable; verify it instead of overwriting it")
    horizons = {"digital-1D": 1, "digital-3D": 3, "digital-5D": 5, "ko": 5, "ratio_straddle": 5}
    if family not in horizons or not result.input_hashes or not {
            "frozen_book", "observed_marks_spreads_and_carry"}.issubset(result.input_hashes):
        raise ValueError("Episode completion requires a supported family and original input hashes")
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in (*result.input_hashes.values(), configuration_sha256)):
        raise ValueError("Episode input and configuration hashes must be SHA-256 digests")
    if (result.summary.empty or not result.summary.origin.eq(origin).all()
            or len(result.positions) != horizons[family]
            or len(result.active_indices) != horizons[family]):
        raise ValueError("Episode results must cover the original origin and complete contractual horizon")
    book_hash = hashlib.sha256(pd.util.hash_pandas_object(result.initial_book, index=True).to_numpy().tobytes()).hexdigest()
    if result.input_hashes["frozen_book"] != book_hash:
        raise ValueError("The final episode book differs from the original input identity")
    with np.load(directory / "steps/episode_checkpoint.npz", allow_pickle=False) as saved:
        identity = json.loads(str(saved["identity_json"]))
        if (identity["origin"] != origin or identity["family"] != family
                or identity["input_hashes"] != result.input_hashes
                or int(saved["next_step"]) != horizons[family]):
            raise ValueError("The final checkpoint does not match the complete episode inputs and horizon")
        for step, (positions, indices) in enumerate(zip(result.positions, result.active_indices, strict=True)):
            if (not np.isfinite(positions).all()
                    or not np.array_equal(positions, saved[f"positions_{step}"])
                    or not np.array_equal(indices, saved[f"indices_{step}"])):
                raise ValueError("The final checkpoint and recorded positions or active instruments disagree")
    result.summary.to_parquet(directory / "summary.parquet", index=False)
    result.trades.to_parquet(directory / "trades.parquet", index=False)
    result.initial_book.to_parquet(directory / "initial_book.parquet", index=False)
    np.savez_compressed(directory / "positions.npz",
        **{f"step_{step}": values for step, values in enumerate(result.positions)},
        **{f"active_{step}": values for step, values in enumerate(result.active_indices)})
    (directory / "diagnostics.json").write_text(json.dumps(result.diagnostics, indent=2) + "\n")
    (directory / "input_hashes.json").write_text(json.dumps(result.input_hashes, indent=2) + "\n")
    artifacts = REQUIRED_ARTIFACTS[:-1] + tuple(str(path.relative_to(directory))
        for path in sorted((directory / "steps").rglob("*")) if path.is_file())
    if any(not (directory / name).resolve().is_relative_to(directory.resolve()) for name in artifacts):
        raise ValueError("Episode evidence must not refer outside its owned directory")
    manifest = {"schema_version": 1, "origin": origin, "family": family,
        "configuration_sha256": configuration_sha256,
        "artifacts_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in artifacts}}
    temporary = manifest_path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)


def verify_episode_artifacts(*, directory: Path, configuration_sha256: str) -> dict[str, object]:
    """Reject unsealed legacy inputs, substitutions and paths outside the episode."""
    manifest = json.loads((directory / "completed.json").read_text())
    if (manifest.get("schema_version") != 1
            or not set(REQUIRED_ARTIFACTS).issubset(manifest.get("artifacts_sha256", {}))):
        raise ValueError("Episode provenance is incomplete: rerun legacy references with sealed inputs and checkpoints")
    if (manifest.get("configuration_sha256") != configuration_sha256
            or manifest.get("origin") != directory.parent.name or manifest.get("family") != directory.name):
        raise ValueError("Episode completion belongs to another configuration, origin or family")
    for name, digest in manifest["artifacts_sha256"].items():
        path = directory / name
        if (not path.resolve().is_relative_to(directory.resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            raise ValueError(f"Completed episode artifact changed or escaped its directory: {path}")
    return manifest
