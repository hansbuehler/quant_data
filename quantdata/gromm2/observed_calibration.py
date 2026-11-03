"""Calibrate a date-t GROMM2 tower at observed DYSANOS quote locations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from cdxcore.bs import bs
from cdxcore.err import verify_inp
from cdxcore.subdir import SubDir
from cdxcore.version import version
from filelock import FileLock
from scipy.special import ndtr

from quantdata.biz_calendar import NMS
from quantdata.gromm2.calibration import _atomic_npz
from quantdata.gromm2.dlv import decode_density_tower
from quantdata.gromm2.geometry import (
    Gromm2GeometryConfig,
    expanding_supports,
    project_total_variance,
    transport_tower,
)
from quantdata.gromm2.market import SavedSanosPricer, load_public_sanos_day
from quantdata.sanos.arbfree import ArbFreePricer
from quantdata.widb.day import enrich_day


@version("0.4", dependencies=[Gromm2GeometryConfig])
@dataclass(frozen=True, kw_only=True)
class ObservedFitConfig:
    """One-date fitting controls; source quotes retain their individual widths."""

    model_label: str = "Gromm v2 4,3"
    geometry: Gromm2GeometryConfig = Gromm2GeometryConfig()
    minimum_dlvsq: float = 0.0001
    maximum_dlvsq: float = 9.0
    fair_value_weight: float = 0.05
    maximum_epochs: int = 6_000
    initial_learning_rate: float = 0.02
    minimum_learning_rate: float = 1.0e-6
    plateau_epochs: int = 50
    checkpoint_every: int = 100

    def __post_init__(self) -> None:
        verify_inp(bool(self.model_label), lambda: "model_label must not be empty")
        verify_inp(
            0.0 < self.minimum_dlvsq < self.maximum_dlvsq,
            lambda: "invalid DLV-squared bounds",
        )
        verify_inp(
            0.0 <= self.fair_value_weight <= 1.0,
            lambda: "fair_value_weight must lie in [0,1]",
        )
        verify_inp(
            self.maximum_epochs > 0 and self.plateau_epochs > 0
            and self.checkpoint_every > 0,
            lambda: "epoch counts must be positive",
        )
        verify_inp(
            0.0 < self.minimum_learning_rate <= self.initial_learning_rate,
            lambda: "invalid learning-rate bounds",
        )


def extract_observed_quotes(
    payload: dict[str, object], *, geometry: Gromm2GeometryConfig,
    pricer: ArbFreePricer | SavedSanosPricer | None = None,
    reference_nms_curve: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Retain real call/put books and recenter their widths on saved fair values."""
    market = payload["market"]
    pricer = ArbFreePricer.from_dict(payload["af_pricer"]) if pricer is None else pricer
    expiries = np.asarray(market["expiries"], dtype=np.float64)
    epsilon_vol = float(pricer.interpolate_atm_vols(1.0 / 252.0)) if reference_nms_curve is None else None
    fields = {name: [] for name in (
        "strikes", "expiries_y", "is_call", "fair_values", "half_spreads",
        "market_bids", "market_asks", "nms", "quote_weights", "is_am",
    )}
    for expiry_index, expiry in enumerate(expiries):
        if expiry * 252.0 <= 0.2001 or expiry > geometry.model_business_dtes[-1] / 252.0 + 1.0e-7:
            continue
        strikes = np.asarray(market["strikes"][expiry_index], dtype=np.float64)
        if reference_nms_curve is None:
            annual_vol = float(pricer.interpolate_atm_vols(float(expiry)))
            nms = NMS.to_nms(strikes, t=float(expiry), vol=annual_vol, eps_vol=epsilon_vol)
        else:
            nms = reference_grid_nms(strikes, expiry=float(expiry), reference_nms_curve=reference_nms_curve)
        european = ~np.asarray(market["is_american"][expiry_index], dtype=bool)
        in_range = (nms >= geometry.fit_nms_range[0]) & (nms <= geometry.fit_nms_range[1])
        for is_call, side in ((True, "call"), (False, "put")):
            bids = np.asarray(market[f"{side}_bids"][expiry_index], dtype=np.float64)
            asks = np.asarray(market[f"{side}_asks"][expiry_index], dtype=np.float64)
            fair_values = np.asarray(market[f"af_{side}_mids"][expiry_index], dtype=np.float64)
            active = (
                np.asarray(market[f"{side}_active"][expiry_index], dtype=bool)
                & european & in_range & np.isfinite(fair_values)
                & np.isfinite(bids) & np.isfinite(asks) & (asks > bids) & (bids >= 0.0)
            )
            count = int(np.count_nonzero(active))
            core = (nms[active] >= geometry.core_fit_nms_range[0]) & (nms[active] <= geometry.core_fit_nms_range[1])
            values = {
                "strikes": strikes[active], "expiries_y": np.full(count, expiry),
                "is_call": np.full(count, is_call), "fair_values": fair_values[active],
                "half_spreads": 0.5 * (asks[active] - bids[active]),
                "market_bids": bids[active], "market_asks": asks[active],
                "nms": nms[active],
                "quote_weights": np.where(core, 1.0, geometry.outer_fit_weight),
                "is_am": np.full(count, bool(market["is_am"][expiry_index])),
            }
            for name, value in values.items():
                fields[name].append(value)
    result = {name: np.concatenate(values) for name, values in fields.items()}
    if not len(result["strikes"]):
        raise ValueError("no observed quotes remain in the requested range")
    return result


@version("0.1")
def reference_grid_nms(strikes: np.ndarray, *, expiry: float, reference_nms_curve: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Measure relative strike on the saved origin grid's total-variance scale."""
    expiries, variances = (np.asarray(value, dtype=np.float64) for value in reference_nms_curve)
    if (expiries.ndim != 1 or variances.shape != expiries.shape or not len(expiries)
            or np.any(~np.isfinite(expiries)) or np.any(~np.isfinite(variances))
            or np.any(expiries <= 0.0) or np.any(variances <= 0.0)
            or np.any(np.diff(expiries) <= 0.0) or np.any(np.diff(variances) < 0.0)
            or not 0.0 < expiry <= expiries[-1] + 1.0e-7):
        raise ValueError("invalid reference-date NMS total-variance curve or expiry")
    variance = np.interp(expiry, np.r_[0.0, expiries], np.r_[0.0, variances])
    return np.log(strikes) / np.sqrt(variance)


def _observed_market(payload: dict[str, object], pricer: ArbFreePricer) -> dict[str, object]:
    """Read the complete cached book without inheriting the old fit's NMS mask."""
    frame = enrich_day(payload["ticker"], payload["d"], override_cache_mode="cacheonly")
    market = {name: [] for name in (
        "expiries", "strikes", "is_american", "is_am", "call_bids", "call_asks",
        "put_bids", "put_asks", "call_active", "put_active", "af_call_mids", "af_put_mids",
    )}
    for expiry, rows in frame.groupby("tte_y", sort=True):
        strikes = rows["strike"].to_numpy(dtype=np.float64) / rows["fwd"].to_numpy(dtype=np.float64)
        normalization = rows["fwd"].to_numpy(dtype=np.float64) * rows["df"].to_numpy(dtype=np.float64)
        calls = np.asarray(pricer.price(expiries=float(expiry), strikes=strikes, mode="surface"))
        values = {
            "expiries": float(expiry), "strikes": strikes,
            "is_american": rows["is_american"].to_numpy(dtype=bool),
            "is_am": bool(rows["is_am"].iloc[0]),
            "af_call_mids": calls, "af_put_mids": calls - 1.0 + strikes,
        }
        for side in ("call", "put"):
            bids = rows[f"mkt_{side}_bid"].to_numpy(dtype=np.float64) / normalization
            asks = rows[f"mkt_{side}_ask"].to_numpy(dtype=np.float64) / normalization
            values.update({f"{side}_bids": bids, f"{side}_asks": asks,
                           f"{side}_active": np.isfinite(bids) & np.isfinite(asks) & (bids >= 0.0) & (asks > bids)})
        for name, value in values.items():
            market[name].append(value)
    return market


def quote_interpolation(
    model_expiries_y: np.ndarray, quote_expiries_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bracket exact quote maturities without rounding AM settlements."""
    model = np.asarray(model_expiries_y, dtype=np.float64)
    quotes = np.asarray(quote_expiries_y, dtype=np.float64)
    if np.any(quotes <= 0.0) or np.any(quotes > model[-1] + 1.0e-7):
        raise ValueError("quote maturities must be inside the tower horizon")
    nearest = np.argmin(np.abs(quotes[:, None] - model[None, :]), axis=1)
    quotes = np.where(np.abs(quotes - model[nearest]) < 1.0e-7, model[nearest], quotes)
    upper = np.minimum(np.searchsorted(model, quotes, side="left"), len(model) - 1)
    lower = np.maximum(upper - 1, 0)
    before = quotes <= model[0]
    previous = np.where(before, 0.0, model[lower])
    alpha = (quotes - previous) / (model[upper] - previous)
    return lower.astype(np.int32), upper.astype(np.int32), alpha, before


def black_quote_kernels(
    *, supports: np.ndarray, strikes: np.ndarray, expiries_y: np.ndarray,
    is_call: np.ndarray, annual_volatility: float,
) -> np.ndarray:
    """Evaluate each observed contract on its own latent support and clock."""
    direction = np.where(is_call, 1.0, -1.0)[:, None]
    strikes = strikes[:, None]
    standard_deviation = annual_volatility * np.sqrt(expiries_y[:, None])
    if annual_volatility == 0.0:
        return np.maximum(direction * (supports - strikes), 0.0)
    d1 = np.log(supports / strikes) / standard_deviation + 0.5 * standard_deviation
    d2 = d1 - standard_deviation
    return direction * (supports * ndtr(direction * d1) - strikes * ndtr(direction * d2))


def load_saved_source(source_market: Path) -> tuple[dict[str, object], ArbFreePricer | SavedSanosPricer]:
    """Read existing HDF5 parameters or a reference to existing dataset rows."""
    if source_market.suffix == ".json":
        reference = json.loads(source_market.read_text())
        if reference.get("schema") != "gromm2-existing-sanos-dataset-reference-v1":
            raise ValueError("unsupported saved SANOS reference schema")
        dataset = Path(reference["dataset_path"])
        for filename, expected in reference["parameter_file_sha256"].items():
            with (dataset / filename).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"saved SANOS dataset changed: {filename}")
        trading_date = date.fromisoformat(reference["date"])
        pricer = SavedSanosPricer(day=load_public_sanos_day(dataset, trading_date))
        return {"schema_version": 4, "ticker": reference["ticker"], "d": trading_date}, pricer
    payload = SubDir(str(source_market.parent), fmt=SubDir.PYTREE_HDF5).read(source_market.stem)
    if payload is None or int(payload.get("schema_version", 0)) != 4:
        raise ValueError("a saved DYSANOS schema-4 market is required")
    return payload, ArbFreePricer.from_dict(payload["af_pricer"])


def prepare_observed_fit(
    source_market: Path, *, config: ObservedFitConfig,
    fixed_grid: Path | None = None,
    initializer: Path | None = None,
    reference_nms_curve: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Build the tower from date-t information and keep its quote book separate."""
    payload, pricer = load_saved_source(source_market)
    geometry = config.geometry
    expiries = geometry.model_business_dtes / 252.0
    source_terminal_business_dte = float(pricer.expiries[-1] * 252.0)
    calls = pricer.price(expiries=expiries, strikes=np.ones_like(expiries), mode="surface")
    atm_vol = np.asarray(bs.implied(k=np.ones_like(expiries), prices=calls, sqrtT=np.sqrt(expiries), eps=1.0e-10))
    variances = project_total_variance(expiries, atm_vol**2 * expiries, forward_volatility_floor=geometry.forward_volatility_floor)
    supports = expanding_supports(variances, config=geometry)
    reference_date = str(payload["d"])
    if fixed_grid is not None:
        with np.load(fixed_grid, allow_pickle=False) as saved:
            if not np.array_equal(saved["model_expiries_y"], expiries):
                raise ValueError("reference and fitting expiry grids differ")
            supports = np.asarray(saved["supports"], dtype=np.float64)
            reference_date = str(saved["date"].item())
            if str(payload["d"]) >= reference_date:
                raise ValueError("historical fixed-grid fits must precede the reference date")
        if supports.shape != (len(expiries), geometry.num_interior_nodes + 2):
            raise ValueError("reference and fitting strike-grid dimensions differ")
    quotes = extract_observed_quotes({**payload, "market": _observed_market(payload, pricer)}, geometry=geometry, pricer=pricer,
                                    reference_nms_curve=reference_nms_curve)
    lower, upper, alpha, before = quote_interpolation(expiries, quotes["expiries_y"])
    kernel_arguments = dict(
        strikes=quotes["strikes"], expiries_y=quotes["expiries_y"],
        is_call=quotes["is_call"], annual_volatility=geometry.backbone_annual_volatility,
    )
    lower_kernels = black_quote_kernels(supports=supports[lower], **kernel_arguments)
    upper_kernels = black_quote_kernels(supports=supports[upper], **kernel_arguments)
    front_prices = black_quote_kernels(supports=np.ones((len(lower), 1)), **kernel_arguments)[:, 0]
    intervals = np.diff(expiries, prepend=0.0)
    forward_variances = np.diff(variances, prepend=0.0) / intervals
    initial = np.clip(forward_variances - geometry.backbone_annual_volatility**2, config.minimum_dlvsq + 1.0e-6, config.maximum_dlvsq * 0.9)
    initial_raw = np.broadcast_to(np.log(initial[:, None] - config.minimum_dlvsq), (len(expiries), geometry.num_interior_nodes)).copy()
    initialization_date = ""
    if initializer is not None:
        initial_raw, initialization_date = load_initial_state(
            initializer, target_date=str(payload["d"]), target_supports=supports,
        )
    frame = enrich_day(payload["ticker"], payload["d"], override_cache_mode="cacheonly")
    curve = frame.groupby("tte_y", sort=True)[["fwd", "spot"]].first()
    forward_times = curve.index.to_numpy(dtype=np.float64)
    forward_ratios = curve["fwd"].to_numpy(dtype=np.float64) / curve["spot"].to_numpy(dtype=np.float64)
    forwards = np.exp(np.interp(expiries, np.r_[0.0, forward_times], np.r_[0.0, np.log(forward_ratios)]))
    reference_fields = {} if reference_nms_curve is None else {
        "quote_range_policy": np.asarray("reference-date-relative-strike-range"),
        "quote_range_reference_expiries_y": np.asarray(reference_nms_curve[0]),
        "quote_range_reference_total_variances": np.asarray(reference_nms_curve[1]),
    }
    return {
        **reference_fields,
        **quotes, "supports": supports, "transports": transport_tower(supports),
        "model_expiries_y": expiries, "projected_total_variances": variances,
        "initial_raw": initial_raw,
        "lower": lower, "upper": upper, "alpha": alpha, "before_first": before,
        "lower_kernels": lower_kernels, "upper_kernels": upper_kernels,
        "front_prices": front_prices, "date": np.asarray(str(payload["d"])),
        "reference_date": np.asarray(reference_date),
        "initialization_date": np.asarray(initialization_date),
        "source_terminal_business_dte": np.asarray(source_terminal_business_dte),
        "spot_ref": np.asarray(float(frame["spot"].iloc[0])),
        "forward_over_spot": forwards,
    }


def regrid_initial_state(raw: np.ndarray, old_supports: np.ndarray, new_supports: np.ndarray) -> np.ndarray:
    """Interpolate a starting guess only; all target-grid quotes are refitted."""
    if raw.shape != old_supports[:, 1:-1].shape or new_supports.shape != old_supports.shape:
        raise ValueError("initial state and strike supports must have matching geometry")
    return np.stack([
        np.interp(np.log(new_supports[index, 1:-1]), np.log(old_supports[index, 1:-1]), raw[index])
        for index in range(len(raw))
    ])


def load_initial_state(initializer: Path, *, target_date: str, target_supports: np.ndarray) -> tuple[np.ndarray, str]:
    """Warm-start causally from a saved state without needing its quote kernels."""
    with np.load(initializer / "fit.npz", allow_pickle=False) as saved:
        raw = np.asarray(saved["raw"])
        if "supports" in saved and "date" in saved:
            old_supports = np.asarray(saved["supports"])
            initialization_date = str(saved["date"].item())
        else:
            with np.load(initializer / "inputs.npz", allow_pickle=False) as inputs:
                old_supports = np.asarray(inputs["supports"])
                initialization_date = str(inputs["date"].item())
    if initialization_date > target_date:
        raise ValueError("initialization cannot use a future date")
    return regrid_initial_state(raw, old_supports, target_supports), initialization_date


def price_observed_quotes(densities: jax.Array, arrays: dict[str, jax.Array]) -> jax.Array:
    """Interpolate adjacent latent laws, using each contract's Black kernel."""
    lower_price = jnp.sum(arrays["lower_kernels"] * densities[arrays["lower"]], axis=-1)
    lower_price = jnp.where(arrays["before_first"], arrays["front_prices"], lower_price)
    upper_price = jnp.sum(arrays["upper_kernels"] * densities[arrays["upper"]], axis=-1)
    return (1.0 - arrays["alpha"]) * lower_price + arrays["alpha"] * upper_price


def fit_statistics(errors: np.ndarray, weights: np.ndarray) -> dict[str, float | int]:
    """Report quote-count and objective-weighted errors in actual half-spreads."""
    if not len(errors):
        return {"quotes": 0}
    excess = np.maximum(errors - 1.0, 0.0)
    return {
        "quotes": len(errors), "inside_percent": float(100.0 * np.mean(errors <= 1.0)),
        "weighted_inside_percent": float(100.0 * np.average(errors <= 1.0, weights=weights)),
        "mean_excess_half_spreads": float(np.mean(excess)),
        "fair_value_mae_half_spreads": float(np.mean(errors)),
        "maximum_excess_half_spreads": float(np.max(excess)),
    }


def calibration_functions(
    data: dict[str, np.ndarray], *, config: ObservedFitConfig,
) -> tuple[Callable, Callable]:
    """Return the observed-book loss and its positive-density tower decoder."""
    arrays = {name: jnp.asarray(value) for name, value in data.items() if value.dtype.kind not in "US"}
    intervals = jnp.diff(arrays["model_expiries_y"], prepend=0.0)
    maximum_raw = math.log(config.maximum_dlvsq - config.minimum_dlvsq)

    def decode(raw: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        dlvsq = config.minimum_dlvsq + jnp.exp(jnp.minimum(raw, maximum_raw))
        densities = decode_density_tower(arrays["supports"], arrays["transports"], dlvsq, intervals)
        return dlvsq, densities, price_observed_quotes(densities, arrays)

    def loss(raw: jax.Array) -> jax.Array:
        _, _, prices = decode(raw)
        errors = jnp.abs(prices - arrays["fair_values"]) / arrays["half_spreads"]
        quote_loss = jnp.average(jax.nn.relu(errors - 1.0) + config.fair_value_weight * errors, weights=arrays["quote_weights"])
        barrier = 0.10 * jax.nn.softplus((raw - maximum_raw) / 0.10) / 0.35
        return quote_loss + 0.5 * jnp.sum(barrier**2)

    return loss, decode


def run_observed_fit(
    *, source_market: Path, run_directory: Path, config: ObservedFitConfig,
    fixed_grid: Path | None = None, initializer: Path | None = None,
    retain_inputs: bool = True,
) -> dict[str, object]:
    """Fit one tower; save progress, restart state, and quote-level evidence."""
    run_directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(run_directory / "fit.lock"), timeout=0):
        return _run_observed_fit(source_market=source_market, run_directory=run_directory, config=config, fixed_grid=fixed_grid, initializer=initializer, retain_inputs=retain_inputs)


def _run_observed_fit(
    *, source_market: Path, run_directory: Path, config: ObservedFitConfig,
    fixed_grid: Path | None, initializer: Path | None, retain_inputs: bool,
) -> dict[str, object]:
    data = prepare_observed_fit(source_market, config=config, fixed_grid=fixed_grid, initializer=initializer)
    if retain_inputs:
        _atomic_npz(run_directory / "inputs.npz", data)
    source_hash = hashlib.sha256(source_market.read_bytes()).hexdigest()
    input_digest = hashlib.sha256()
    for name, value in sorted(data.items()):
        input_digest.update(name.encode())
        input_digest.update(np.ascontiguousarray(value).tobytes())
    input_hash = input_digest.hexdigest()
    model_config = {name: value for name, value in asdict(config).items() if name in {"geometry", "minimum_dlvsq", "maximum_dlvsq", "fair_value_weight"}}
    identity = json.dumps({"config": model_config, "source_sha256": source_hash, "inputs_sha256": input_hash, "version": ObservedFitConfig.version.unique_id48}, sort_keys=True)
    loss, decode = calibration_functions(data, config=config)
    value_and_grad = jax.jit(jax.value_and_grad(loss))
    decode_only = jax.jit(decode)

    @jax.jit
    def update(raw, first, second, step, gradient, learning_rate):
        next_step = step + 1
        next_first = 0.9 * first + 0.1 * gradient
        next_second = 0.999 * second + 0.001 * gradient**2
        correction = next_first / (1.0 - 0.9**next_step)
        correction /= jnp.sqrt(next_second / (1.0 - 0.999**next_step)) + 1.0e-8
        return raw - learning_rate * correction, next_first, next_second, next_step

    raw = jnp.asarray(data["initial_raw"])
    first, second = jnp.zeros_like(raw), jnp.zeros_like(raw)
    step = jnp.asarray(0, dtype=jnp.int32)
    best_raw = raw
    best_value = math.inf
    learning_rate = config.initial_learning_rate
    last_improvement = start_epoch = 0
    checkpoint = run_directory / "checkpoint.npz"
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(saved["identity"].item()) != identity:
                raise ValueError("checkpoint identity differs from the requested fit")
            raw, first, second, step = (jnp.asarray(saved[name]) for name in ("raw", "first", "second", "step"))
            best_raw = jnp.asarray(saved["best_raw"])
            best_value, learning_rate = float(saved["best_value"]), float(saved["learning_rate"])
            last_improvement, start_epoch = int(saved["last_improvement"]), int(saved["completed_epochs"])
    stop = threading.Event()
    done = threading.Event()
    started = time.monotonic()
    progress = {"stage": "compiling", "completed": start_epoch, "total": config.maximum_epochs}

    def heartbeat() -> None:
        while not done.wait(60):
            elapsed = time.monotonic() - started
            completed = int(progress["completed"]) - start_epoch
            eta = elapsed * (config.maximum_epochs - int(progress["completed"])) / completed if completed else None
            print(json.dumps(dict(event="heartbeat", **progress, elapsed_seconds=elapsed, remaining_seconds=eta)), flush=True)
            try:
                result = subprocess.run(["gpuctl", "resources", "--json"], capture_output=True, text=True, timeout=15, check=True)
                unhealthy = [item for item in json.loads(result.stdout)["checks"] if item["health"] != "OK"]
                if unhealthy:
                    print(json.dumps(dict(event="resource_alert", checks=unhealthy)), flush=True)
                if any(item["health"] in {"STOP", "ERROR", "UNAVAILABLE"} for item in unhealthy):
                    stop.set()
                    if not done.wait(60):
                        print(json.dumps(dict(event="resource_forced_exit", message="Graceful-stop deadline exceeded; use prior periodic checkpoint")), flush=True)
                        os.kill(os.getpid(), signal.SIGKILL)
                    return
            except Exception as error:
                print(json.dumps(dict(event="resource_warning", message=str(error))), flush=True)

    monitor = threading.Thread(target=heartbeat, daemon=True)
    monitor.start()
    previous_handlers = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGTERM, signal.SIGINT)}

    def save(epoch: int) -> None:
        _atomic_npz(checkpoint, {
            "identity": np.asarray(identity), "raw": np.asarray(raw), "best_raw": np.asarray(best_raw),
            "first": np.asarray(first), "second": np.asarray(second), "step": np.asarray(step),
            "best_value": np.asarray(best_value), "learning_rate": np.asarray(learning_rate),
            "last_improvement": np.asarray(last_improvement), "completed_epochs": np.asarray(epoch),
            "training_config": np.asarray(json.dumps(asdict(config), sort_keys=True)),
        })

    print(json.dumps(dict(event="fit_start", model=config.model_label, date=str(data["date"]), quotes=len(data["strikes"]), tower=list(data["initial_raw"].shape), backend=jax.default_backend(), cold_start=start_epoch == 0)), flush=True)
    stop_reason = "maximum_epochs"
    epoch = start_epoch
    try:
        for epoch in range(start_epoch + 1, config.maximum_epochs + 1):
            objective, gradient = value_and_grad(raw)
            objective_value = float(objective)
            if not np.isfinite(objective_value) or not np.all(np.isfinite(np.asarray(gradient))):
                raise FloatingPointError("non-finite calibration objective or gradient")
            if objective_value < best_value:
                meaningful = objective_value < best_value - max(1.0e-8, abs(best_value) * 1.0e-5) if np.isfinite(best_value) else True
                best_value, best_raw = objective_value, raw
                if meaningful:
                    last_improvement = epoch
            if epoch - last_improvement >= config.plateau_epochs:
                if learning_rate > config.minimum_learning_rate:
                    learning_rate = max(learning_rate * 0.5, config.minimum_learning_rate)
                    last_improvement = epoch
                    print(json.dumps(dict(event="learning_rate_review", epoch=epoch, learning_rate=learning_rate, best_objective=best_value)), flush=True)
                elif epoch >= 1_000:
                    stop_reason = "objective_plateau_at_minimum_learning_rate"
                    break
            raw, first, second, step = update(raw, first, second, step, gradient, jnp.asarray(learning_rate))
            progress.update(stage="fitting", completed=epoch)
            if epoch == 1 or epoch % 50 == 0:
                elapsed = time.monotonic() - started
                print(json.dumps(dict(event="fit_progress", epoch=epoch, total=config.maximum_epochs, objective=objective_value, best_objective=best_value, learning_rate=learning_rate, gradient_norm=float(jnp.linalg.norm(gradient)), elapsed_seconds=elapsed, remaining_seconds=elapsed * (config.maximum_epochs - epoch) / max(epoch - start_epoch, 1))), flush=True)
            if epoch % config.checkpoint_every == 0 or stop.is_set():
                save(epoch)
            if stop.is_set():
                stop_reason = "checkpointed_stop"
                break
        final_value = float(value_and_grad(raw)[0])
        if final_value < best_value:
            best_value, best_raw = final_value, raw
        save(epoch)
        dlvsq, densities, prices = (np.asarray(value) for value in decode_only(best_raw))
        errors = np.abs(prices - data["fair_values"]) / data["half_spreads"]
        core = data["quote_weights"] == 1.0
        groups = {"all": np.ones(len(errors), dtype=bool), "core": core, "outer": ~core, "first_25_days": data["expiries_y"] <= 25.0 / 252.0 + 1.0e-7, "beyond_25_days": data["expiries_y"] > 25.0 / 252.0 + 1.0e-7}
        report = {
            "schema": "gromm2-observed-date-t-fit-v1", "date": str(data["date"]),
            "model_label": config.model_label,
            "reference_date": str(data["reference_date"]),
            "initialization_date": str(data["initialization_date"]) or None,
            "source_market": str(source_market.resolve()), "source_sha256": source_hash,
            "source_refitted": False,
            "source_terminal_business_dte": float(data["source_terminal_business_dte"]),
            "source_native_terminal_pricing_for_tower": bool(data["source_terminal_business_dte"] < data["model_expiries_y"][-1] * 252.0 - 1.0e-5),
            "source_native_terminal_pricing_quote_fraction": float(np.mean(data["expiries_y"] * 252.0 > data["source_terminal_business_dte"] + 1.0e-5)),
            "inputs_sha256": input_hash,
            "config": asdict(config), "backend": jax.default_backend(), "completed_epochs": epoch,
            "stop_reason": stop_reason, "best_objective": best_value,
            "target": "saved DYSANOS surface fair value at each actual observed market strike and expiry",
            "tolerance": "that contract's observed half-spread, with no interpolation of spreads",
            "price_unit": "cash price / (discount factor * forward)",
            "fit": {name: fit_statistics(errors[selected], data["quote_weights"][selected]) for name, selected in groups.items()},
            "density": {"minimum_probability": float(densities.min()), "maximum_mass_error": float(np.max(np.abs(densities.sum(axis=-1) - 1.0))), "maximum_forward_error": float(np.max(np.abs(np.sum(densities * data["supports"], axis=-1) - 1.0))), "maximum_boundary_mass": float(np.max(densities[:, 0] + densities[:, -1]))},
            "dlvsq": {"minimum": float(dlvsq.min()), "maximum": float(dlvsq.max())},
            "original_market_inside_percent": float(100.0 * np.mean((prices >= data["market_bids"]) & (prices <= data["market_asks"]))),
        }
        _atomic_npz(run_directory / "fit.npz", {"raw": np.asarray(best_raw), "dlvsq": dlvsq, "densities": densities, "prices": prices, "errors_half_spreads": errors, "spot_ref": data["spot_ref"], "forward_over_spot": data["forward_over_spot"], "supports": data["supports"], "date": data["date"]})
        (run_directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(event="fit_finished", **report), allow_nan=False), flush=True)
        return report
    finally:
        done.set()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-market", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--maximum-epochs", type=int, default=6_000)
    parser.add_argument("--model-label", default="Gromm v2 4,3")
    arguments = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    run_observed_fit(source_market=arguments.source_market, run_directory=arguments.run_directory, config=ObservedFitConfig(model_label=arguments.model_label, maximum_epochs=arguments.maximum_epochs))


if __name__ == "__main__":
    main()
