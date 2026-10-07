"""Construct a normalized empirical density from a fitted OCE policy PnL."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from quantdata.dysanos.flex.market_encoder import write_json_atomic


@dataclass(frozen=True, kw_only=True)
class OceDensityTilt:
    """A marginal-utility density relative to an empirical base measure."""

    weights: np.ndarray
    log_weights: np.ndarray
    log_normalizer: float
    effective_sample_size: float


def fit_oce_density_tilt(
    base_weights: np.ndarray,
    pnl: np.ndarray,
) -> OceDensityTilt:
    """Return ``base_weights * exp(-pnl)`` with sample mean one."""

    base_weights = np.asarray(base_weights, dtype=np.float64)
    pnl = np.asarray(pnl, dtype=np.float64)
    if base_weights.ndim != 1 or pnl.shape != base_weights.shape:
        raise ValueError("base weights and PnL must be matching vectors")
    if not len(base_weights):
        raise ValueError("base weights and PnL cannot be empty")
    if np.any(~np.isfinite(base_weights) | (base_weights <= 0.0)):
        raise ValueError("base weights must be finite-positive")
    if np.any(~np.isfinite(pnl)):
        raise ValueError("PnL must be finite")

    log_raw = np.log(base_weights) - pnl
    log_normalizer = float(logsumexp(log_raw) - math.log(len(log_raw)))
    log_weights = log_raw - log_normalizer

    # Remove the final floating-point normalization error before exponentiating.
    log_mass = float(logsumexp(log_weights) - math.log(len(log_weights)))
    log_weights -= log_mass
    weights = np.exp(log_weights)
    if np.any(~np.isfinite(weights) | (weights <= 0.0)):
        raise FloatingPointError("the OCE density underflowed or became non-finite")
    effective_sample_size = float(
        np.sum(weights, dtype=np.float64) ** 2
        / np.sum(np.square(weights), dtype=np.float64)
    )
    return OceDensityTilt(
        weights=weights,
        log_weights=log_weights,
        log_normalizer=log_normalizer + log_mass,
        effective_sample_size=effective_sample_size,
    )


def _save_npy_atomic(values: np.ndarray, path: Path) -> None:
    if path.suffix.lower() != ".npy":
        raise ValueError(f"array output must use the .npy suffix, found {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-weights", type=Path, required=True)
    parser.add_argument("--pnl", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    base_weights = np.load(arguments.base_weights, mmap_mode="r", allow_pickle=False)
    pnl = np.load(arguments.pnl, mmap_mode="r", allow_pickle=False)
    fit = fit_oce_density_tilt(base_weights, pnl)

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    _save_npy_atomic(fit.weights, arguments.output_directory / "weights.npy")
    _save_npy_atomic(fit.log_weights, arguments.output_directory / "log-weights.npy")
    report = {
        "schema": "dysanos-oce-marginal-utility-tilt-v1",
        "status": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "definition": (
            "w_next = w_base * exp(-net_pnl) / mean(w_base * exp(-net_pnl))"
        ),
        "base_weights": str(arguments.base_weights),
        "pnl": str(arguments.pnl),
        "paths": len(fit.weights),
        "log_normalizer": fit.log_normalizer,
        "oce_value": -fit.log_normalizer,
        "effective_sample_size": fit.effective_sample_size,
        "effective_sample_fraction": fit.effective_sample_size / len(fit.weights),
        "density_mean": float(np.mean(fit.weights)),
        "density_minimum": float(np.min(fit.weights)),
        "density_maximum": float(np.max(fit.weights)),
    }
    write_json_atomic(report, arguments.output_directory / "tilt-report.json")
    print(
        f"OCE={-fit.log_normalizer:.9g} "
        f"ESS={fit.effective_sample_size:.1f} "
        f"density_mean={np.mean(fit.weights):.17g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
