"""Fit nested affine DLV-only states with an exact JAX Gram eigensolve."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from cdxcore.util import is_filename


def _atomic_write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_npz(payload: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_raw_dlvs(
    artifact_directory: Path,
    *,
    min_dlvsq: float,
    dlvsq_distance_floor: float,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    paths = tuple(sorted(artifact_directory.glob("*.npz")))
    if not paths:
        raise ValueError(f"No LP artifacts found in {artifact_directory}")
    values: list[np.ndarray] = []
    dates: list[str] = []
    expected_shape: tuple[int, int] | None = None
    for path in paths:
        with np.load(path, allow_pickle=False) as artifact:
            dlvsqs = np.asarray(artifact["dlvsqs"], dtype=np.float64)
        if expected_shape is None:
            expected_shape = dlvsqs.shape
        if dlvsqs.shape != expected_shape:
            raise ValueError(
                f"Inconsistent DLV shape in {path}: "
                f"{dlvsqs.shape} versus {expected_shape}"
            )
        if not np.all(np.isfinite(dlvsqs) & (dlvsqs >= 0.0)):
            raise ValueError(f"Invalid DLV values in {path}")
        distance = dlvsqs - min_dlvsq
        values.append(
            np.log(np.maximum(distance, dlvsq_distance_floor)).reshape(-1)
        )
        dates.append(path.stem)
    assert expected_shape is not None
    return np.stack(values), np.asarray(expected_shape), tuple(dates)


def fit_affine_pca_jax(
    raw: np.ndarray,
    *,
    maximum_rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return mean, loadings, scores, and singular values from exact PCA."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or min(raw.shape) < 2:
        raise ValueError(f"Expected a non-trivial matrix, found {raw.shape}")
    if not 0 < maximum_rank < min(raw.shape):
        raise ValueError(
            f"maximum_rank must lie in [1,{min(raw.shape) - 1}], "
            f"found {maximum_rank}"
        )
    mean = np.mean(raw, axis=0)
    centered = raw - mean
    centered_jax = jax.device_put(jnp.asarray(centered))

    def gram_eigh(value: jax.Array) -> tuple[jax.Array, jax.Array]:
        gram = value @ value.T
        return jnp.linalg.eigh(gram)

    started = perf_counter()
    compiled = jax.jit(gram_eigh).lower(centered_jax).compile()
    eigenvalues_jax, eigenvectors_jax = compiled(centered_jax)
    eigenvalues_jax.block_until_ready()
    elapsed_seconds = perf_counter() - started
    eigenvalues = np.asarray(eigenvalues_jax)[::-1]
    eigenvectors = np.asarray(eigenvectors_jax)[:, ::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    singular_values = np.sqrt(eigenvalues)
    tolerance = np.finfo(np.float64).eps * max(raw.shape) * singular_values[0]
    available_rank = int(np.sum(singular_values > tolerance))
    retained_rank = min(maximum_rank, available_rank)
    if retained_rank < maximum_rank:
        raise RuntimeError(
            f"DLV matrix rank {available_rank} is below requested rank "
            f"{maximum_rank}"
        )
    left = eigenvectors[:, :maximum_rank]
    retained_singular_values = singular_values[:maximum_rank]
    scores = left * retained_singular_values[None, :]
    loadings = centered.T @ left / retained_singular_values[None, :]

    anchors = np.argmax(np.abs(loadings), axis=0)
    signs = np.where(
        loadings[anchors, np.arange(maximum_rank)] < 0.0,
        -1.0,
        1.0,
    )
    loadings *= signs[None, :]
    scores *= signs[None, :]
    return mean, loadings, scores, singular_values, elapsed_seconds


def _rank_statistics(
    raw: np.ndarray,
    mean: np.ndarray,
    loadings: np.ndarray,
    scores: np.ndarray,
    singular_values: np.ndarray,
    *,
    rank: int,
    min_dlvsq: float,
) -> dict[str, float | int]:
    reconstruction = mean + scores[:, :rank] @ loadings[:, :rank].T
    residual = reconstruction - raw
    target_dlvsq = min_dlvsq + np.exp(raw)
    reconstructed_dlvsq = min_dlvsq + np.exp(
        np.minimum(reconstruction, np.log(np.finfo(np.float64).max) - 2.0)
    )
    dlvsq_error = reconstructed_dlvsq - target_dlvsq
    relative = np.abs(dlvsq_error) / np.maximum(
        target_dlvsq,
        np.finfo(np.float64).tiny,
    )
    total_energy = float(np.sum(singular_values**2))
    retained_energy = float(np.sum(singular_values[:rank] ** 2))
    return {
        "rank": rank,
        "explained_variance_fraction": (
            retained_energy / total_energy if total_energy > 0.0 else 1.0
        ),
        "raw_mse": float(np.mean(residual**2)),
        "raw_mae": float(np.mean(np.abs(residual))),
        "dlvsq_mae": float(np.mean(np.abs(dlvsq_error))),
        "dlvsq_relative_mae": float(np.mean(relative)),
        "dlvsq_relative_median": float(np.median(relative)),
        "dlvsq_relative_p90": float(np.quantile(relative, 0.9)),
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--label", default="daily25-dlv-only-affine-jax-v1")
    parser.add_argument("--ranks", nargs="+", type=int, default=(100, 200, 400))
    parser.add_argument("--min-dlvsq", type=float, default=1.0e-4)
    parser.add_argument("--dlvsq-distance-floor", type=float, default=9.0e-6)
    return parser.parse_args()


def main() -> None:
    """Fit one nested affine basis and write all requested rank audits."""
    arguments = _parse_arguments()
    if not is_filename(arguments.label):
        raise ValueError(f"Invalid label {arguments.label!r}")
    ranks = tuple(sorted(set(arguments.ranks)))
    if not ranks or ranks[0] <= 0:
        raise ValueError("Every requested rank must be positive")
    if not np.isfinite(arguments.dlvsq_distance_floor) or not (
        arguments.dlvsq_distance_floor > 0.0
    ):
        raise ValueError("'dlvsq-distance-floor' must be finite and positive")
    jax.config.update("jax_enable_x64", True)
    raw, dlv_shape, dates = _load_raw_dlvs(
        arguments.artifact_directory,
        min_dlvsq=arguments.min_dlvsq,
        dlvsq_distance_floor=arguments.dlvsq_distance_floor,
    )
    print(
        f"loaded {raw.shape[0]} surfaces with {raw.shape[1]} DLV nodes; "
        f"JAX backend={jax.default_backend()} devices={jax.devices()}",
        flush=True,
    )
    mean, loadings, scores, singular_values, elapsed_seconds = (
        fit_affine_pca_jax(raw, maximum_rank=ranks[-1])
    )
    statistics = [
        _rank_statistics(
            raw,
            mean,
            loadings,
            scores,
            singular_values,
            rank=rank,
            min_dlvsq=arguments.min_dlvsq,
        )
        for rank in ranks
    ]
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arguments.output_directory / f"{arguments.label}.npz"
    _atomic_write_npz(
        {
            "schema": np.asarray("flex-daily25-dlv-affine-jax-v1"),
            "dates": np.asarray(dates),
            "dlv_shape": dlv_shape,
            "min_dlvsq": np.asarray(arguments.min_dlvsq),
            "mean": mean,
            "loadings": loadings,
            "scores": scores,
            "singular_values": singular_values,
        },
        checkpoint_path,
    )
    report = {
        "schema": "flex-daily25-dlv-affine-jax-v1",
        "status": "complete",
        "label": arguments.label,
        "artifact_directory": str(arguments.artifact_directory),
        "checkpoint": str(checkpoint_path),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "num_dates": int(raw.shape[0]),
        "num_dlvs_per_date": int(raw.shape[1]),
        "dlv_shape": [int(value) for value in dlv_shape],
        "dlvsq_distance_floor": arguments.dlvsq_distance_floor,
        "ranks": list(ranks),
        "compile_and_solve_seconds": elapsed_seconds,
        "statistics": statistics,
    }
    report_path = arguments.output_directory / f"{arguments.label}.json"
    _atomic_write_json(report, report_path)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
