"""Minimum-entropy complete-path weights for unconditional option gain constraints."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version
from filelock import FileLock
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import logsumexp

from quantdata.gromm2.rolling_audit import check_resource_headroom


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class MeasureConfig:
    """Conditioning and solver tolerances, without economic mean corrections."""

    standard_deviation_floor: float = 1.0e-8
    relative_eigenvalue_floor: float = 1.0e-12
    scaled_mean_tolerance: float = 1.0e-7
    maximum_iterations: int = 3000

    def __post_init__(self) -> None:
        verify_inp(self.standard_deviation_floor > 0.0 and 0.0 < self.relative_eigenvalue_floor < 1.0,
                   lambda: "invalid measure conditioning floors")
        verify_inp(self.scaled_mean_tolerance > 0.0 and self.maximum_iterations > 0,
                   lambda: "invalid measure solver controls")


def source_sha256(path: Path) -> str:
    """Hash scenario contents, following links without changing their source."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def option_gain_constraints(prices: np.ndarray, expiries: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Return pair-average raw and spot-scaled gains for live option/day cells."""
    prices = np.asarray(prices, dtype=np.float64)
    expiries = np.asarray(expiries)
    if (prices.ndim != 3 or prices.shape[0] < 4 or prices.shape[0] % 2
            or prices.shape[2] != len(expiries) + 1 or prices.shape[1] < 2
            or np.any(~np.isfinite(prices)) or np.any(prices[..., 0] <= 0.0)
            or np.any(expiries <= 0)):
        raise ValueError("constraints require finite complete antithetic paths and valid option metadata")
    days = np.arange(prices.shape[1] - 1)
    live = days[:, None] < expiries[None, :]
    gains = np.diff(prices[..., 1:], axis=1)
    coordinates = np.argwhere(live)
    arrays, labels = [], []
    for name, values in (("raw", gains), ("spot_scaled", gains / prices[:, :-1, :1])):
        flat = values[:, live]
        arrays.append(0.5 * (flat[0::2] + flat[1::2]))
        labels.extend({"scaling": name, "day": int(day), "option": int(option)} for day, option in coordinates)
    return np.concatenate(arrays, axis=1), labels


def entropy_path_weights(pair_gains: np.ndarray, *, config: MeasureConfig = MeasureConfig()) -> tuple[np.ndarray, dict[str, object]]:
    """Fit one strictly positive weight per complete antithetic pair.

    Covariance whitening only conditions the dual. Every original constraint
    is checked afterwards, including any numerically redundant directions.
    No return is recentered, clipped or discarded for having a tiny variance.
    """
    values = np.asarray(pair_gains, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("pair gains must be a nonempty finite matrix")
    check_resource_headroom()
    started = time.monotonic()
    scale = np.maximum(values.std(axis=0), config.standard_deviation_floor)
    scaled = values / scale
    mean = scaled.mean(axis=0)
    centered = scaled - mean
    covariance = centered.T @ centered / len(scaled)
    eigenvalues, eigenvectors = eigh(covariance, check_finite=False)
    threshold = max(float(eigenvalues[-1]), 1.0) * config.relative_eigenvalue_floor
    retained = eigenvalues > threshold
    rotation = eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])
    whitened = scaled @ rotation
    last_report = time.monotonic()
    iteration = 0

    def objective(parameter: np.ndarray) -> tuple[float, np.ndarray]:
        scores = -(whitened @ parameter)
        normalizer = logsumexp(scores)
        weights = np.exp(scores - normalizer)
        return float(normalizer - np.log(len(values))), -(whitened.T @ weights)

    def progress(parameter: np.ndarray) -> None:
        nonlocal last_report, iteration
        iteration += 1
        if time.monotonic() - last_report >= 30.0 or iteration == 1:
            check_resource_headroom()
            loss, gradient = objective(parameter)
            print(json.dumps({"event": "entropy_measure_progress", "iteration": iteration,
                "total_iteration_safety_limit": config.maximum_iterations,
                "elapsed_seconds": time.monotonic() - started, "dual_objective": loss,
                "maximum_whitened_residual": float(np.max(np.abs(gradient), initial=0.0))}), flush=True)
            last_report = time.monotonic()

    if whitened.shape[1]:
        result = minimize(objective, np.zeros(whitened.shape[1]), jac=True, method="L-BFGS-B", callback=progress,
            options={"maxiter": config.maximum_iterations, "ftol": 1.0e-15, "gtol": 1.0e-10, "maxls": 50, "maxcor": 30})
        scores = -(whitened @ result.x)
        weights = np.exp(scores - logsumexp(scores))
        solver_message = str(result.message)
    else:
        weights = np.full(len(values), 1.0 / len(values))
        solver_message = "No nonconstant constraint directions"
    residual = weights @ values
    scaled_residual = residual / scale
    maximum_residual = float(np.max(np.abs(scaled_residual), initial=0.0))
    positive = bool(np.all(weights > 0.0) and np.all(np.isfinite(weights)))
    report = {"config": asdict(config), "version": config.version.unique_id48,
        "independent_pairs": len(values), "paths": 2 * len(values), "constraint_count": values.shape[1],
        "retained_dual_directions": int(retained.sum()), "solver_message": solver_message,
        "iterations": iteration, "elapsed_seconds": time.monotonic() - started,
        "maximum_absolute_mean_before": float(np.max(np.abs(values.mean(axis=0)), initial=0.0)),
        "maximum_absolute_mean_after": float(np.max(np.abs(residual), initial=0.0)),
        "maximum_scaled_mean_after": maximum_residual,
        "effective_pairs": float(1.0 / np.sum(weights**2)),
        "maximum_pair_weight_multiple": float(len(weights) * weights.max()),
        "minimum_pair_weight": float(weights.min()),
        "relative_entropy": float(np.sum(weights * np.log(np.maximum(weights * len(weights), np.finfo(float).tiny)))),
        "all_weights_strictly_positive": positive,
        "passed": positive and maximum_residual <= config.scaled_mean_tolerance,
        "interpretation": "Unconditional zero means are imposed, not tested. No conditional martingale claim follows."}
    if not report["passed"]:
        raise ValueError("Entropy measure failed strict positivity or original-constraint residual checks: " + json.dumps(report))
    return np.repeat(weights / 2.0, 2), report


def precondition_scenarios(path: Path, *, split: str, directory: Path,
                          config: MeasureConfig = MeasureConfig()) -> dict[str, object]:
    """Save independently fitted training/validation measures outside the source tree."""
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / f"{split}.lock"), timeout=0):
        digest = source_sha256(path)
        output = directory / f"{split}.npz"
        report_path = directory / f"{split}.json"
        identity = {"scenario_sha256": digest, "config": asdict(config), "version": config.version.unique_id48}
        if output.exists() and report_path.exists():
            saved = json.loads(report_path.read_text())
            if saved["identity"] != identity:
                raise ValueError("preconditioning identity changed; use a new run directory")
            return saved
        with h5py.File(path, "r") as stream:
            source_identity = json.loads(str(stream.attrs["identity"]))
            if source_identity["split"] != split or int(stream.attrs["completed_paths"]) != int(source_identity["config"]["paths"]):
                raise ValueError("preconditioning requires the matching complete scenario set")
            prices = np.asarray(stream["prices"][:], dtype=np.float64)
            gains, coordinates = option_gain_constraints(prices, stream["option_expiry_days"][:])
        weights, report = entropy_path_weights(gains, config=config)
        report.update(identity=identity, source=str(path), coordinates=coordinates,
                      original_mean_after=(weights[0::2] * 2.0 @ gains).tolist())
        temporary = output.with_suffix(".partial.npz")
        np.savez_compressed(temporary, path_weights=weights, scenario_sha256=np.asarray(digest))
        temporary.replace(output)
        temporary_report = report_path.with_suffix(".partial.json")
        temporary_report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary_report.replace(report_path)
        print(json.dumps({"event": "entropy_measure_complete", "split": split,
            **{key: report[key] for key in ("effective_pairs", "maximum_pair_weight_multiple", "maximum_scaled_mean_after", "passed")}}), flush=True)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    arguments = parser.parse_args()
    for split in ("train", "validation"):
        precondition_scenarios(arguments.run_directory / f"{split}.h5", split=split,
                              directory=arguments.run_directory / "measures")


if __name__ == "__main__":
    main()
