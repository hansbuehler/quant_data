"""Resumable synthetic surfaces for the Algorithm-1 learned inverse map.

The paper's online CRC loop needs two learned maps: parameters-to-IVS (NN1)
and an implicitly regularized inverse from ``theta, xi, rho, IVS`` to the ten
jump parameters (NN2).  This module supplies deterministic, independently
sharded CPU data for that training and an exact-pricer gate that every reset
must pass on contracts excluded from inverse fitting.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr

from quantdata.dysanos.bates_crc.gambara_teichmann_crc import (
    FiveBucketJumps,
    GeneralizedBatesState,
    paired_maturity_boundaries,
    price_calls_generalized_bates,
)
from quantdata.dysanos.bates_crc.heston_riccati import HestonParams

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class InverseDatasetConfig:
    """Logical geometry and sampling law for independently reproducible shards."""

    seed: int = 4_271
    samples_per_shard: int = 1_000
    heldout_shard_modulus: int = 10
    n_quad: int = 240
    u_max: float = 600.0
    kappa: float = 4.0
    intensity: float = 1.0
    maximum_price_bound_error: float = 5.0e-4
    maximum_candidate_attempt_multiplier: int = 20
    rho_lower: float = -0.995
    jump_std_lower: float = 0.005
    maturities: tuple[float, ...] = (
        7 / 365,
        14 / 365,
        30 / 365,
        45 / 365,
        60 / 365,
        90 / 365,
        120 / 365,
        180 / 365,
        270 / 365,
        440 / 365,
    )
    moneyness: tuple[float, ...] = tuple(np.linspace(0.8, 1.2, 13))

    def __post_init__(self) -> None:
        if self.samples_per_shard <= 0 or self.heldout_shard_modulus < 2:
            raise ValueError("samples_per_shard must be positive and heldout modulus at least two")
        paired_maturity_boundaries(np.asarray(self.maturities))
        if len(self.moneyness) != 13 or any(value <= 0.0 for value in self.moneyness):
            raise ValueError("the paper grid requires thirteen positive moneyness nodes")
        if self.maximum_price_bound_error <= 0.0:
            raise ValueError("maximum_price_bound_error must be positive")
        if self.maximum_candidate_attempt_multiplier < 1:
            raise ValueError("maximum_candidate_attempt_multiplier must be positive")
        if not -1.0 < self.rho_lower < 0.50:
            raise ValueError("rho_lower must lie strictly between -1 and 0.50")
        if not 0.0 < self.jump_std_lower < 0.30:
            raise ValueError("jump_std_lower must lie strictly between zero and 0.30")


def shard_split(shard_index: int, modulus: int) -> str:
    """Assign whole shards to held-out data, preventing sample-level leakage."""

    if shard_index < 0 or modulus < 2:
        raise ValueError("shard_index must be non-negative and modulus at least two")
    return "heldout" if shard_index % modulus == modulus - 1 else "train"


def _black_calls(strikes: FloatArray, expiries: FloatArray, volatility: FloatArray) -> FloatArray:
    total = volatility * np.sqrt(expiries)
    d1 = -np.log(strikes) / total + 0.5 * total
    return ndtr(d1) - strikes * ndtr(d1 - total)


def implied_volatilities(
    prices: FloatArray,
    strikes: FloatArray,
    expiries: FloatArray,
    *,
    maximum_bound_error: float,
) -> tuple[FloatArray, int]:
    """Vectorized Black implied vols with a bounded Fourier-tail correction.

    Carr--Madan truncation can put a very short, far-OTM call a few basis
    points below zero.  Such a node is clipped only when the violation is
    below the configured absolute tolerance; larger errors reject the sampled
    surface instead of silently contaminating inverse-map training.
    """

    prices = np.asarray(prices, dtype=np.float64)
    strikes = np.asarray(strikes, dtype=np.float64)
    expiries = np.asarray(expiries, dtype=np.float64)
    intrinsic = np.maximum(1.0 - strikes, 0.0)
    if np.any(~np.isfinite(prices)):
        raise RuntimeError("generated call price is non-finite")
    lower_error = intrinsic - prices
    upper_error = prices - 1.0
    maximum_error = float(max(np.max(lower_error), np.max(upper_error)))
    if maximum_error > maximum_bound_error:
        raise RuntimeError(
            "generated call price exceeds the permitted Fourier bound error: "
            f"{maximum_error:.6g} > {maximum_bound_error:.6g}"
        )
    invalid = (prices <= intrinsic) | (prices >= 1.0)
    prices = np.clip(prices, intrinsic + 1.0e-10, 1.0 - 1.0e-10)
    lower = np.full_like(prices, 1.0e-5)
    upper = np.full_like(prices, 5.0)
    for _ in range(64):
        middle = 0.5 * (lower + upper)
        below = _black_calls(strikes, expiries, middle) < prices
        lower = np.where(below, middle, lower)
        upper = np.where(below, upper, middle)
    return 0.5 * (lower + upper), int(np.count_nonzero(invalid))


def generate_surface_shard(config: InverseDatasetConfig, shard_index: int) -> dict[str, np.ndarray]:
    """Generate one deterministic CPU shard for NN1 and inverse-map training."""

    rng = np.random.default_rng(np.random.SeedSequence((config.seed, shard_index)))
    maturities = np.asarray(config.maturities, dtype=np.float64)
    moneyness = np.asarray(config.moneyness, dtype=np.float64)
    expiries = np.repeat(maturities, moneyness.size)
    strikes = np.tile(moneyness, maturities.size)
    boundaries = paired_maturity_boundaries(maturities)
    count = config.samples_per_shard
    parameter_rows = np.empty((count, 14), dtype=np.float64)
    surfaces = np.empty((count, strikes.size), dtype=np.float64)

    sample = 0
    attempts = 0
    clipped_nodes = 0
    maximum_attempts = count * config.maximum_candidate_attempt_multiplier
    while sample < count:
        attempts += 1
        if attempts > maximum_attempts:
            raise RuntimeError(
                f"accepted only {sample} of {count} surfaces after {maximum_attempts} candidates"
            )
        variance = float(rng.uniform(0.005, 0.20))
        theta = float(rng.uniform(0.01, 0.30))
        xi_limit = min(0.50, math.sqrt(2.0 * config.kappa * theta) * 0.98)
        xi = float(rng.uniform(0.01, xi_limit))
        rho = float(rng.uniform(config.rho_lower, 0.50))
        means = rng.uniform(-0.30, 0.15, size=5)
        stds = rng.uniform(config.jump_std_lower, 0.30, size=5)
        state = GeneralizedBatesState(
            log_spot=0.0,
            variance=variance,
            heston=HestonParams(kappa=config.kappa, theta=theta, xi=xi, rho=rho),
            jumps=FiveBucketJumps(
                intensity=config.intensity,
                boundaries=boundaries,
                means=tuple(float(value) for value in means),
                stds=tuple(float(value) for value in stds),
            ),
        )
        try:
            calls = price_calls_generalized_bates(
                state,
                strikes,
                expiries,
                n_quad=config.n_quad,
                u_max=config.u_max,
            )
            volatility, clipped = implied_volatilities(
                calls,
                strikes,
                expiries,
                maximum_bound_error=config.maximum_price_bound_error,
            )
        except RuntimeError:
            continue
        surfaces[sample] = volatility
        parameter_rows[sample] = (variance, theta, xi, rho, *means, *stds)
        clipped_nodes += clipped
        sample += 1
    return {
        "parameters": parameter_rows,
        "implied_volatilities": surfaces,
        "maturities": maturities,
        "moneyness": moneyness,
        "kappa": np.asarray(config.kappa, dtype=np.float64),
        "intensity": np.asarray(config.intensity, dtype=np.float64),
        "quadrature_nodes": np.asarray(config.n_quad, dtype=np.int64),
        "quadrature_u_max": np.asarray(config.u_max, dtype=np.float64),
        "candidate_attempt_count": np.asarray(attempts, dtype=np.int64),
        "clipped_price_node_count": np.asarray(clipped_nodes, dtype=np.int64),
    }


def write_surface_shard(
    config: InverseDatasetConfig,
    shard_index: int,
    run_directory: Path,
) -> Path:
    """Atomically write one resumable shard and its provenance manifest entry."""

    run_directory.mkdir(parents=True, exist_ok=True)
    split = shard_split(shard_index, config.heldout_shard_modulus)
    destination = run_directory / f"{split}-{shard_index:06d}.npz"
    if destination.exists():
        return destination
    started = time.monotonic()
    arrays = generate_surface_shard(config, shard_index)
    temporary = destination.with_name(f".{destination.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, destination)
    record = {
        "schema": "gambara-teichmann-inverse-surface-shard-v3",
        "shard_index": shard_index,
        "split": split,
        "samples": config.samples_per_shard,
        "surface_nodes": 130,
        "candidate_attempts": int(arrays["candidate_attempt_count"]),
        "clipped_price_nodes": int(arrays["clipped_price_node_count"]),
        "elapsed_seconds": time.monotonic() - started,
        "config": asdict(config),
        "path": str(destination),
    }
    with (run_directory / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
    return destination


def write_surface_shards(
    config: InverseDatasetConfig,
    *,
    first_shard: int,
    shard_count: int,
    workers: int,
    run_directory: Path,
) -> None:
    """Generate a resumable shard range with persistent aggregate progress."""

    if first_shard < 0 or shard_count <= 0 or workers <= 0:
        raise ValueError("first_shard must be non-negative; shard_count/workers must be positive")
    run_directory.mkdir(parents=True, exist_ok=True)
    indices = range(first_shard, first_shard + shard_count)
    started = time.monotonic()
    completed = 0
    progress_path = run_directory / "aggregate-progress.jsonl"

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        pending = {
            executor.submit(write_surface_shard, config, index, run_directory): index
            for index in indices
        }
        for future in concurrent.futures.as_completed(pending):
            index = pending[future]
            future.result()
            completed += 1
            elapsed = time.monotonic() - started
            remaining = elapsed * (shard_count - completed) / completed
            record = {
                "stage": "inverse_surface_generation",
                "elapsed_seconds": elapsed,
                "completed_work_units": completed,
                "total_work_units": shard_count,
                "estimated_remaining_seconds": remaining,
                "last_completed_shard": index,
            }
            line = json.dumps(record, sort_keys=True)
            print(line, flush=True)
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()


def exact_reset_price_gate(
    before: GeneralizedBatesState,
    after: GeneralizedBatesState,
    *,
    calibration_strikes: FloatArray,
    calibration_expiries: FloatArray,
    heldout_strikes: FloatArray,
    heldout_expiries: FloatArray,
    tolerance: float,
    n_quad: int = 240,
    u_max: float = 600.0,
) -> dict[str, float | bool]:
    """Gate a proposed learned reset with independent exact Fourier repricing."""

    if before.heston.kappa != after.heston.kappa or before.jumps.intensity != after.jumps.intensity:
        raise ValueError("CRC reset must keep kappa and jump intensity fixed")
    errors: list[FloatArray] = []
    maxima: list[float] = []
    for strikes, expiries in (
        (calibration_strikes, calibration_expiries),
        (heldout_strikes, heldout_expiries),
    ):
        old_prices = price_calls_generalized_bates(before, strikes, expiries, n_quad=n_quad, u_max=u_max)
        new_prices = price_calls_generalized_bates(after, strikes, expiries, n_quad=n_quad, u_max=u_max)
        error = new_prices - old_prices
        errors.append(error)
        maxima.append(float(np.max(np.abs(error))))
    maximum = max(maxima)
    return {
        "accepted": bool(maximum <= tolerance),
        "calibration_max_abs_price_error": maxima[0],
        "heldout_max_abs_price_error": maxima[1],
        "combined_rms_price_error": float(np.sqrt(np.mean(np.concatenate(errors) ** 2))),
        "tolerance": tolerance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--shard-index", type=int)
    selection.add_argument("--shard-count", type=int)
    parser.add_argument("--first-shard", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--samples-per-shard", type=int, default=1_000)
    parser.add_argument("--target-fit-report", type=Path)
    parser.add_argument("--kappa", type=float)
    parser.add_argument("--intensity", type=float)
    arguments = parser.parse_args()
    target_values: dict[str, object] = {}
    if arguments.target_fit_report is not None:
        report = json.loads(arguments.target_fit_report.read_text(encoding="utf-8"))
        if report.get("target_date") != "2025-08-29":
            raise RuntimeError("inverse data received the wrong target-date fit")
        fit = report["target_fit"]
        target_values = {
            "kappa": float(fit["kappa_fixed_from_history"]),
            "intensity": float(fit["intensity_fixed_from_history"]),
            "maturities": tuple(float(value) for value in fit["maturity_nodes"]),
        }
    if arguments.kappa is not None:
        target_values["kappa"] = arguments.kappa
    if arguments.intensity is not None:
        target_values["intensity"] = arguments.intensity
    config = InverseDatasetConfig(
        samples_per_shard=arguments.samples_per_shard,
        **target_values,
    )
    if arguments.shard_index is not None:
        print(write_surface_shard(config, arguments.shard_index, arguments.run_directory))
    else:
        write_surface_shards(
            config,
            first_shard=arguments.first_shard,
            shard_count=arguments.shard_count,
            workers=arguments.workers,
            run_directory=arguments.run_directory,
        )


if __name__ == "__main__":
    main()
