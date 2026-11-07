"""Backward implicit Dupire on a daily homogeneous strike grid; no optimizer."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.hedge2.source import SourceArchive, normalize_surface_moments, surface_time_values
from quantdata.sanos.tridiag import solve_sanos_tridiagonal


@version("0.4")
@dataclass(frozen=True, kw_only=True)
class DlvConfig:
    business_dtes: tuple[float, ...] = (1., 2., 3., 4., 5., 10., 15., 20., 25.)
    num_inner_strikes: int = 200
    nms_range: tuple[float, float] = (-4., 3.)
    minimum_spacing_fraction: float = 0.05
    min_local_vol: float = 0.001
    max_local_vol: float = 1000.0
    extreme_local_vol: float = 10.0
    density_floor: float = 1e-12
    density_tolerance: float = 1e-10
    price_tolerance: float = 1e-8
    source_calendar_tolerance: float = 1e-5
    reconstruction_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        verify_inp(self.num_inner_strikes >= 10 and self.num_inner_strikes % 2 == 0,
                   lambda: "The shared grid needs an even strike count of at least ten")
        verify_inp(self.nms_range[0] < 0 < self.nms_range[1],
                   lambda: "NMS endpoints must straddle zero")
        verify_inp(0 < self.min_local_vol < self.max_local_vol < np.inf,
                   lambda: "Local-volatility guards must be finite, positive and ordered")
        verify_inp(0 < self.minimum_spacing_fraction <= 1,
                   lambda: "Minimum spacing fraction must be in (0,1]")
        verify_inp(0 < self.density_floor < 1 and self.price_tolerance > 0
                   and self.reconstruction_tolerance > 0,
                   lambda: "Density and price tolerances must be positive")
        times = np.asarray(self.business_dtes)
        verify_inp(np.isfinite(times).all() and times[0] > 0 and np.all(np.diff(times) > 0),
                   lambda: "Business DTEs must be positive and increasing")

    @property
    def expiries_y(self) -> np.ndarray:
        return np.asarray(self.business_dtes, dtype=np.float64) / DAYS_PER_YEAR


def shared_inner_strikes(forward_vols: np.ndarray, *, config: DlvConfig) -> np.ndarray:
    """Merge front/final DYSANOS-style NMS grids, keeping one exact ATM node."""
    times = config.expiries_y
    increments = np.diff(times, prepend=0.)
    total_vols = np.cumsum(forward_vols * np.sqrt(increments))
    scales = np.sqrt((forward_vols[0] ** 2 * NMS.EPS + total_vols**2) / (1. + NMS.EPS))
    half = config.num_inner_strikes // 2
    lower, upper = config.nms_range
    left_count = min(max(2, int(half * abs(lower) // (upper - lower))), half - 3)
    front_nodes = np.r_[np.linspace(lower, 0., left_count, endpoint=False),
                        np.linspace(0., upper, half - left_count)]
    final_nodes = np.linspace(lower, upper, half)
    if np.any(np.isclose(final_nodes, 0., atol=1e-14, rtol=0.)):
        edges = np.linspace(lower, upper, half + 1)
        final_nodes = 0.5 * (edges[1:] + edges[:-1])
    ordered = np.sort(np.r_[scales[0] * front_nodes, scales[-1] * final_nodes])
    atm = np.flatnonzero(ordered == 0.)
    if len(atm) != 1:
        raise ValueError("The shared grid must have one exact ATM node")
    atm_index = int(atm[0])
    gap = config.minimum_spacing_fraction * (ordered[-1] - ordered[0]) / (len(ordered) - 1)
    left = -np.cumsum(np.maximum(np.diff(ordered[:atm_index + 1]), gap)[::-1])[::-1]
    right = np.cumsum(np.maximum(np.diff(ordered[atm_index:]), gap))
    return np.exp(np.r_[left, 0., right])


def add_boundaries(inner: np.ndarray, calls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Add intrinsic boundary calls far enough away to retain discrete convexity."""
    first_slope = (calls[:, 1] - calls[:, 0]) / (inner[1] - inner[0])
    last_slope = (calls[:, -1] - calls[:, -2]) / (inner[-1] - inner[-2])
    first_put = np.maximum(calls[:, 0] - 1. + inner[0], 0.)
    lower_limit = inner[0] - first_put / np.maximum(1. + first_slope, 1e-14)
    upper_limit = inner[-1] + calls[:, -1] / np.maximum(-last_slope, 1e-14)
    lower = min(0.5 * inner[0], 0.9 * float(np.min(lower_limit)))
    upper = max(1.5 * inner[-1], 1.1 * float(np.max(upper_limit)))
    if not (np.isfinite(lower) and np.isfinite(upper) and 0 < lower < inner[0] < inner[-1] < upper):
        raise ValueError("No finite positive convex boundary extension exists")
    support = np.r_[lower, inner, upper]
    full_calls = np.c_[np.full(len(calls), 1. - lower), calls, np.zeros(len(calls))]
    return support, full_calls


def densities_from_calls(support: np.ndarray, calls: np.ndarray) -> np.ndarray:
    slopes = np.diff(calls, axis=-1) / np.diff(support)
    return np.diff(slopes, axis=-1, prepend=-np.ones((*slopes.shape[:-1], 1)),
                   append=np.zeros((*slopes.shape[:-1], 1)))


def decode_densities(support: np.ndarray, dlvsqs: np.ndarray, expiries_y: np.ndarray) -> np.ndarray:
    """Apply the implicit forward operators with absorbing boundary nodes."""
    density = np.zeros_like(support)
    atm = np.flatnonzero(support == 1.)
    if len(atm) != 1:
        raise ValueError("The initial Dirac mass requires an exact ATM node")
    density[int(atm[0])] = 1.
    output = []
    for local_variance, step in zip(dlvsqs, np.diff(expiries_y, prepend=0.), strict=True):
        box = local_variance * support[1:-1] ** 2 * step / (support[2:] - support[:-2])
        density = solve_sanos_tridiagonal(
            w_minus=box / (support[1:-1] - support[:-2]),
            w_plus=box / (support[2:] - support[1:-1]), rhs=density,
        )
        output.append(density)
    return np.asarray(output)


def positive_reference_density(support: np.ndarray) -> np.ndarray:
    """A positive reference law on every node, with exact unit mass and mean."""
    density = np.full(len(support), 1. / len(support))
    mean = float(density @ support)
    if mean > 1.:
        weight = (mean - 1.) / (mean - support[0])
        density *= 1. - weight
        density[0] += weight
    elif mean < 1.:
        weight = (1. - mean) / (support[-1] - mean)
        density *= 1. - weight
        density[-1] += weight
    return density


def recover_dlvs(support: np.ndarray, calls: np.ndarray, *, forward_vols: np.ndarray,
                 config: DlvConfig, time_values: np.ndarray | None = None) -> dict[str, np.ndarray | float | int]:
    """Recover equation (20)'s backward local variance and audit every guard."""
    if (support.shape != (config.num_inner_strikes + 2,)
            or calls.shape != (len(config.business_dtes), len(support))
            or forward_vols.shape != (len(config.business_dtes),)):
        raise ValueError("Calls, strikes and forward vols do not match the configured grid")
    if not (np.isfinite(calls).all() and np.isfinite(support).all()
            and np.isfinite(forward_vols).all() and np.all(forward_vols > 0)
            and np.all(support > 0) and np.all(np.diff(support) > 0)):
        raise ValueError("Calls and strike support must be finite and ordered")
    if time_values is None:
        time_values = calls - np.maximum(1. - support[None, :], 0.)
    if time_values.shape != calls.shape or not np.isfinite(time_values).all():
        raise ValueError("Time values must be finite and match the call grid")
    time_slopes = np.diff(time_values, axis=-1) / np.diff(support)
    density = np.diff(time_slopes, axis=-1, prepend=np.zeros((len(calls), 1)),
                      append=np.zeros((len(calls), 1)))
    density[:, np.flatnonzero(support == 1.)] += 1.
    theta = np.diff(time_values[:, 1:-1], axis=0, prepend=np.zeros((1, len(support)-2)))
    if density.min() < -config.density_tolerance or theta.min() < -config.price_tolerance:
        raise ValueError(f"Source grid violates no-arbitrage: min density={density.min():.3g}, "
                         f"min backward theta={theta.min():.3g}")
    box = support[1:-1] ** 2 * np.diff(config.expiries_y, prepend=0.)[:, None]
    box = box / (support[2:] - support[:-2])
    denominator = box * density[:, 1:-1]
    positive_mass = density[:, 1:-1] > 0.
    incompatible = (~positive_mass) & (theta > config.price_tolerance)
    if incompatible.any():
        raise ValueError(f"Positive backward theta with numerically zero gamma at {incompatible.sum()} nodes")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        raw = np.divide(theta, denominator, out=np.full_like(theta, np.nan), where=positive_mass)
    defined = positive_mass & np.isfinite(raw) & (raw >= 0.)
    guarded = np.minimum(np.where(defined, raw, 0.), config.max_local_vol**2)
    training_variance = np.maximum(guarded, config.min_local_vol**2)
    decoded = decode_densities(support, training_variance, config.expiries_y)
    equation_defect = theta - denominator * training_variance
    payoff_kernel = np.maximum(support[None, :] - support[:, None], 0.)
    reconstructed = decoded @ payoff_kernel.T
    difference = reconstructed - calls
    mass_error = float(np.max(np.abs(decoded.sum(axis=-1) - 1.)))
    mean_error = float(np.max(np.abs(decoded @ support - 1.)))
    max_error = float(np.max(np.abs(difference)))
    if not (np.isfinite(guarded).all() and np.isfinite(decoded).all() and decoded.min() >= -1e-12
            and mass_error < 1e-8 and mean_error < 1e-8):
        raise ValueError("The guarded implicit DLV reconstruction is numerically invalid")
    return {
        "densities": density, "raw_dlvsqs": raw, "dlvsqs": guarded,
        "training_dlvsqs": training_variance, "local_equation_defect": equation_defect,
        "cumulative_equation_defect_bound": float(np.sum(np.max(np.abs(equation_defect), axis=1))),
        "identified": defined, "decoded_densities": decoded,
        "decoded_call_prices": reconstructed, "backward_theta": theta,
        "undefined_nodes": int((~defined).sum()),
        "zero_zero_nodes": int(np.sum((~positive_mass) & (theta == 0.))),
        "low_mass_nodes": int(np.sum(density[:, 1:-1] <= config.density_floor)),
        "floored_nodes": int(np.sum(guarded < config.min_local_vol**2)),
        "capped_nodes": int(np.sum(defined & (raw > config.max_local_vol**2))),
        "extreme_nodes": int(np.sum(defined & (raw > config.extreme_local_vol**2))),
        "max_raw_local_vol": float(np.sqrt(max(float(np.nanmax(raw)), 0.))),
        "max_repricing_error": max_error, "mass_error": mass_error, "mean_error": mean_error,
        "reconstruction_passed": max_error <= config.reconstruction_tolerance,
    }


def build_day(archive: SourceArchive, day: date, *, config: DlvConfig) -> dict[str, object]:
    started = time.monotonic()
    source, provenance = archive.load(day)
    times = config.expiries_y
    variance = np.asarray(source.interpolate_atm_vols(times)) ** 2 * times
    increments = np.diff(variance, prepend=0.)
    forward_vols = np.sqrt(np.maximum(increments / np.diff(times, prepend=0.), 1e-6))
    inner = shared_inner_strikes(forward_vols, config=config)
    original_calls = source.price(expiries=times[:, None], strikes=inner[None, :], mode="surface")
    moment_errors = normalize_surface_moments(source)
    time_values = surface_time_values(source, expiries_y=times, strikes=inner)
    calls = time_values + np.maximum(1. - inner[None, :], 0.)
    normalization_error = float(np.max(np.abs(calls - original_calls)))
    if normalization_error > config.reconstruction_tolerance:
        raise ValueError(f"Source moment normalization changes prices by {normalization_error:.3g}")
    source_theta = np.diff(time_values, axis=0, prepend=np.zeros((1, len(inner))))
    if source_theta.min() < -config.source_calendar_tolerance:
        raise ValueError(f"Source calendar error exceeds its certificate tolerance: {source_theta.min():.3g}")
    calendar_values = np.maximum.accumulate(np.vstack((np.zeros(len(inner)), time_values)), axis=0)[1:]
    calendar_correction = float(np.max(np.abs(calendar_values - time_values)))
    time_values = calendar_values
    calls = time_values + np.maximum(1. - inner[None, :], 0.)
    support, full_calls = add_boundaries(inner, calls)
    full_time_values = np.c_[np.zeros(len(times)), time_values, np.zeros(len(times))]
    original_full_calls = np.c_[np.full(len(times), 1. - support[0]), original_calls, np.zeros(len(times))]
    reference = positive_reference_density(support)
    payoff_kernel = np.maximum(support[None, :] - support[:, None], 0.)
    reference_values = payoff_kernel @ reference - np.maximum(1. - support, 0.)
    recovered = None
    attempts = []
    for regularization_weight in (0., 1e-12, 1e-10, 1e-8, 1e-7, 1e-6):
        calculation_values = (1. - regularization_weight) * full_time_values + regularization_weight * reference_values
        calculation_calls = calculation_values + np.maximum(1. - support[None, :], 0.)
        try:
            recovered = recover_dlvs(support, calculation_calls, forward_vols=forward_vols,
                                     config=config, time_values=calculation_values)
        except ValueError as error:
            attempts.append(str(error))
            continue
        source_repricing_error = float(np.max(np.abs(recovered["decoded_call_prices"] - original_full_calls)))
        recovered["reconstruction_passed"] = source_repricing_error <= config.reconstruction_tolerance
        if recovered["reconstruction_passed"]:
            break
    if recovered is None:
        raise ValueError(f"Direct DLV recovery remained incompatible after numerical guards: {attempts[-1]}")
    native = np.c_[np.log(recovered["training_dlvsqs"]), np.log(forward_vols)].reshape(-1)
    if not np.isfinite(native).all():
        raise ValueError("The logarithmic training state is non-finite")
    return {
        "date": day.isoformat(), **provenance,
        "representation_smoothness": 0., "expiries_y": times,
        "support_strikes": support, "forward_vols": forward_vols,
        "source_call_prices": original_full_calls, "calculation_call_prices": calculation_calls,
        "calculation_time_values": calculation_values,
        "pre_regularization_time_values": full_time_values,
        "density_regularization_weight": regularization_weight,
        "minimum_reference_density": float(reference.min()),
        "density_regularization_price_change": float(np.max(np.abs(calculation_calls - full_calls))),
        "first_interval_regularization_change": float(np.max(np.abs(calculation_calls[0] - full_calls[0]))),
        "regularization_attempt_errors": json.dumps(attempts),
        "raw_u": native, "max_source_repricing_error": source_repricing_error,
        "source_normalization_price_change": normalization_error,
        "source_calendar_correction": calendar_correction,
        "source_first_business_dte": float(source.expiries[0] * DAYS_PER_YEAR),
        "source_extrapolated_tenors": int(np.sum(times < source.expiries[0] - 1e-12)),
        **moment_errors,
        "geometry_variance_floors": int(np.sum(increments <= 0.)),
        **recovered, "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    archive = SourceArchive(root=args.source_root)
    config = DlvConfig()
    days = [day for day in archive.dates() if args.start <= day <= args.end]
    if not days:
        raise ValueError("No source dates in the requested interval")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    configuration = json.loads(json.dumps({"version": DlvConfig.version.unique_id48, **asdict(config)}))
    config_path = args.output_directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("Output directory belongs to a different DLV configuration")
    config_path.write_text(json.dumps(configuration, indent=2) + "\n")
    started = time.monotonic()
    failed = []
    completed = 0
    validated = 0
    rejected = []
    for index, day in enumerate(days):
        output = args.output_directory / "days" / f"{day.isoformat()}.npz"
        if output.exists():
            with np.load(output, allow_pickle=False) as saved:
                passed = bool(saved["reconstruction_passed"])
            validated += int(passed)
            if not passed:
                rejected.append(day.isoformat())
            completed += 1
            continue
        try:
            result = build_day(archive, day, config=config)
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".partial.npz")
            np.savez_compressed(temporary, **result)
            temporary.replace(output)
            completed += 1
            validated += int(result["reconstruction_passed"])
            if not result["reconstruction_passed"]:
                rejected.append(day.isoformat())
            print(f"{index + 1}/{len(days)} {day} seconds={result['elapsed_seconds']:.2f} "
                  f"source_error={result['max_source_repricing_error']:.3g} "
                  f"undefined={result['undefined_nodes']} floor={result['floored_nodes']} "
                  f"cap={result['capped_nodes']} pass={result['reconstruction_passed']}", flush=True)
        except (ValueError, RuntimeError, FileNotFoundError) as error:
            failed.append({"date": day.isoformat(), "error": str(error)})
            print(f"{index + 1}/{len(days)} {day} ERROR {error}", flush=True)
        if (index + 1) % 25 == 0:
            elapsed = time.monotonic() - started
            print(f"progress written={completed}/{len(days)} validated={validated} errors={len(failed)} "
                  f"elapsed_s={elapsed:.1f} eta_s={elapsed / (index + 1) * (len(days)-index-1):.1f}", flush=True)
    (args.output_directory / "summary.json").write_text(json.dumps({
        "source_root": str(archive.root), "written_dates": completed,
        "validated_dates": validated, "total_dates": len(days),
        "failed": failed, "reconstruction_rejected_dates": rejected,
        "elapsed_seconds": time.monotonic() - started,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
