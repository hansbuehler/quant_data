"""Own-date calibration followed by causal, reference-grid rolling AR(1) fits."""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing
import os
import signal
import subprocess
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path

import h5py
import jax
import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version
from filelock import FileLock

from quantdata.biz_calendar import count_business_days_between
from quantdata.gromm2.calibration import _atomic_npz
from quantdata.gromm2.estimation import (
    _fixed_contract_endpoint_surface,
    _spot_gaussian_score,
)
from quantdata.gromm2.observed_calibration import (
    ObservedFitConfig,
    load_saved_source,
    run_observed_fit,
)


@version("0.1", dependencies=[ObservedFitConfig])
@dataclass(frozen=True, kw_only=True)
class RollingFitConfig:
    """Window, calibration quality and execution scope of the rolling study."""

    fitting: ObservedFitConfig = ObservedFitConfig()
    start_date: str = "2022-01-03"
    end_date: str = "2025-08-29"
    rolling_start_date: str = "2023-01-01"
    history_business_days: int = 252
    workers: int = 4
    minimum_inside_percent: float = 90.0
    maximum_continuation_epochs: int = 18_000
    diffusion_factors: int = 16

    def __post_init__(self) -> None:
        verify_inp(date.fromisoformat(self.start_date) <= date.fromisoformat(self.end_date), lambda: "invalid calibration date range")
        verify_inp(self.history_business_days >= 3, lambda: "at least three historical business days are required")
        verify_inp(1 <= self.workers <= 4, lambda: "workers must lie between one and four")
        verify_inp(0.0 < self.minimum_inside_percent <= 100.0, lambda: "invalid calibration-quality threshold")
        verify_inp(self.maximum_continuation_epochs >= self.fitting.maximum_epochs, lambda: "continuation ceiling must contain the initial training schedule")
        verify_inp(self.diffusion_factors >= 2, lambda: "diffusion_factors must be at least two")


def write_json(path: Path, payload: object) -> None:
    """Replace a small manifest atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_inventory(source_root: Path, *, config: RollingFitConfig) -> dict[str, object]:
    """Inspect saved surface horizons without calibrating or extrapolating them."""
    candidates: dict[str, list[dict[str, object]]] = {}
    for path in sorted(source_root.rglob("*.h5")):
        if path.parent.name != "SPX":
            continue
        try:
            trading_date = date.fromisoformat(path.stem).isoformat()
        except ValueError:
            continue
        if not config.start_date <= trading_date <= config.end_date:
            continue
        with h5py.File(path, "r") as stream:
            if "root/af_pricer/expiries" not in stream:
                continue
            expiries = np.asarray(stream["root/af_pricer/expiries"], dtype=np.float64)
            smoothness_key = next((key for key in stream["root/af_pricer"] if key == "smoothness" or key.endswith("Nsmoothness")), None)
            smoothness = float(stream["root/af_pricer"][smoothness_key][()]) if smoothness_key is not None else None
        if not len(expiries) or smoothness != 0.25:
            continue
        candidates.setdefault(trading_date, []).append({
            "path": str(path.resolve()), "terminal_business_dte": float(expiries[-1] * 252.0),
            "smoothness": smoothness,
        })
    required = float(config.fitting.geometry.model_business_dtes[-1])
    selected = {day: max(values, key=lambda item: (item["terminal_business_dte"], item["path"])) for day, values in sorted(candidates.items())}
    missing = [day for day, item in selected.items() if item["terminal_business_dte"] < required - 1.0e-5]
    ready = bool(selected) and not missing
    return {
        "schema": "gromm2-rolling-source-inventory-v1", "source_root": str(source_root.resolve()),
        "required_terminal_business_dte": required, "dates": selected,
        "date_count": len(selected), "complete_horizon_dates": len(selected) - len(missing),
        "incomplete_horizon_dates": missing, "ready": ready,
        "policy": "saved smooth DYSANOS surfaces only; no new source fit and no unobserved maturity extrapolation",
    }


def saved_parameter_inventory(path: Path, *, config: RollingFitConfig) -> dict[str, object]:
    """Pin existing SANOS parameter files; native terminal pricing needs no refit."""
    saved = json.loads(path.read_text())
    entries = {}
    required = float(config.fitting.geometry.model_business_dtes[-1])
    for day, record in sorted(saved["dates"].items()):
        date.fromisoformat(day)
        if not config.start_date <= day <= config.end_date:
            continue
        source = Path(record["path"])
        if not source.is_file():
            raise FileNotFoundError(f"saved SANOS parameters are missing for {day}: {source}")
        if source.suffix == ".json":
            _, pricer = load_saved_source(source)
            expiries, smoothness = np.asarray(pricer.expiries), float(pricer.smoothness)
        else:
            with h5py.File(source, "r") as stream:
                state = stream["root/af_pricer"]
                expiries = np.asarray(state["expiries"], dtype=np.float64)
                smoothness_key = next(key for key in state if key == "smoothness" or key.endswith("Nsmoothness"))
                smoothness = float(state[smoothness_key][()])
        if not len(expiries) or np.any(~np.isfinite(expiries)) or np.any(expiries <= 0.0) or np.any(np.diff(expiries) <= 0.0):
            raise ValueError(f"invalid saved SANOS expiry geometry for {day}")
        if smoothness != 0.25:
            raise ValueError(f"saved SANOS smoothness must be 0.25, found {smoothness} for {day}")
        entries[day] = {
            **record, "path": str(source.resolve()), "smoothness": smoothness,
            "terminal_business_dte": float(expiries[-1] * 252.0),
        }
    terminal_dates = [day for day, item in entries.items() if item["terminal_business_dte"] < required - 1.0e-5]
    return {
        "schema": "gromm2-saved-sanos-parameter-inventory-v1", "source_mode": "saved_parameters",
        "pinned_inventory": str(path.resolve()), "dates": entries, "date_count": len(entries),
        "required_terminal_business_dte": required, "complete_horizon_dates": len(entries) - len(terminal_dates),
        "incomplete_horizon_dates": [], "dates_using_native_terminal_pricing": terminal_dates,
        "ready": bool(entries), "source_refitting_permitted": False,
        "policy": "read existing SANOS parameters only; use their native terminal density and flat terminal kernel-volatility pricing rule beyond the last saved node",
    }


def training_prefix(dates: list[str], origin_index: int, *, window: int) -> list[int]:
    """Choose the preceding business-day window, retaining missing observations."""
    if dates != sorted(set(dates)) or not 0 <= origin_index < len(dates):
        raise ValueError("dates must be unique, sorted, and contain the origin")
    origin = date.fromisoformat(dates[origin_index])
    if count_business_days_between(date.fromisoformat(dates[0]), origin) < window:
        raise ValueError("the origin does not have a complete historical window")
    return [index for index in range(origin_index) if 0 < count_business_days_between(date.fromisoformat(dates[index]), origin) <= window]


def select_origin_inventory(inventory: dict[str, object], *, origin: str, window: int) -> dict[str, object]:
    """Retain one estimation origin and only its required historical sources."""
    date.fromisoformat(origin)
    dates = sorted(inventory["dates"])
    if origin not in dates:
        raise ValueError(f"requested origin {origin} is absent from the source inventory")
    origin_index = dates.index(origin)
    indices = training_prefix(dates, origin_index, window=window)
    selected = {dates[index]: inventory["dates"][dates[index]] for index in [*indices, origin_index]}
    required = float(inventory["required_terminal_business_dte"])
    missing = [day for day, item in selected.items() if item["terminal_business_dte"] < required - 1.0e-5]
    native_terminal_pricing = inventory.get("source_mode") == "saved_parameters"
    return {
        **inventory, "dates": selected, "requested_origin": origin,
        "date_count": len(selected), "complete_horizon_dates": len(selected) - len(missing),
        "incomplete_horizon_dates": [] if native_terminal_pricing else missing,
        "dates_using_native_terminal_pricing": missing if native_terminal_pricing else [],
        "ready": native_terminal_pricing or not missing,
    }


def ar1_estimate(origins: np.ndarray, endpoints: np.ndarray) -> dict[str, np.ndarray]:
    """Coordinatewise physical AR(1) with sample innovation variance and OLS SE."""
    if origins.shape != endpoints.shape or len(origins) < 3:
        raise ValueError("AR(1) inputs must contain at least three aligned pairs")
    if not np.all(np.isfinite(origins)) or not np.all(np.isfinite(endpoints)):
        raise ValueError("AR(1) levels must be finite")
    origin_mean = np.mean(origins, axis=0)
    endpoint_mean = np.mean(endpoints, axis=0)
    centered_origin = origins - origin_mean
    centered_endpoint = endpoints - endpoint_mean
    denominator = np.sum(centered_origin**2, axis=0)
    variable = denominator > np.finfo(float).eps * np.maximum(np.sum(origins**2, axis=0), 1.0) * len(origins)
    slope = np.divide(np.sum(centered_origin * centered_endpoint, axis=0), denominator, out=np.zeros_like(denominator), where=variable)
    intercept = endpoint_mean - slope * origin_mean
    residuals = endpoints - intercept - slope * origins
    degrees = np.where(variable, len(origins) - 2, len(origins) - 1)
    innovation_variance = np.sum(residuals**2, axis=0) / degrees
    slope_se = np.sqrt(np.divide(innovation_variance, denominator, out=np.full_like(denominator, np.nan), where=variable))
    return {"intercept": intercept, "slope": slope, "slope_standard_error": slope_se,
            "innovation_variance": innovation_variance, "innovations": residuals,
            "identified_slope": variable}


def estimate_window(
    *, dates: list[str], log_dlvsq: np.ndarray, supports: np.ndarray,
    model_expiries_y: np.ndarray, forward_over_spot: np.ndarray,
    spot_ref: np.ndarray, front_densities: np.ndarray, backbone_vol: float,
) -> dict[str, np.ndarray]:
    """Align daily pairs to fixed cash strikes/expiries before fitting AR(1)."""
    if len(dates) != len(log_dlvsq) or len(dates) != len(spot_ref):
        raise ValueError("historical states and market observations do not align")
    origins, endpoints, scores, pair_dates = [], [], [], []
    clipped = queries = 0
    tenors = model_expiries_y * 252.0
    for endpoint in range(1, len(dates)):
        origin_date, endpoint_date = date.fromisoformat(dates[endpoint - 1]), date.fromisoformat(dates[endpoint])
        if count_business_days_between(origin_date, endpoint_date) != 1:
            continue
        gross_spot = float(spot_ref[endpoint] / spot_ref[endpoint - 1])
        aligned, clip_count = _fixed_contract_endpoint_surface(
            endpoint_log_dlvsq=log_dlvsq[endpoint], supports=supports,
            reference_business_dtes=tenors,
            origin_forward_over_spot=forward_over_spot[endpoint - 1, 1:],
            endpoint_forward_over_spot=forward_over_spot[endpoint], gross_spot_return=gross_spot,
        )
        origins.append(log_dlvsq[endpoint - 1, 1:])
        endpoints.append(aligned)
        clipped += clip_count
        queries += aligned.size
        scores.append(_spot_gaussian_score(
            normalized_spot=gross_spot / float(forward_over_spot[endpoint - 1, 0]),
            support=supports[0], density=front_densities[endpoint - 1],
            expiry_y=float(model_expiries_y[0]), backbone_annual_volatility=backbone_vol,
        ))
        pair_dates.append(dates[endpoint])
    if len(origins) < 3:
        raise ValueError("fewer than three single-session pairs remain")
    result = ar1_estimate(np.stack(origins), np.stack(endpoints))
    result.update(pair_end_dates=np.asarray(pair_dates), spot_gaussian_scores=np.asarray(scores),
                  fixed_contract_clipped_query_fraction=np.asarray(clipped / queries))
    return result


def joint_innovation_loadings(
    innovations: np.ndarray, spot_scores: np.ndarray, *, factor_count: int,
) -> dict[str, np.ndarray]:
    """Estimate the shared P/Q diffusion with the front-spot factor first."""
    samples = len(innovations)
    if samples != len(spot_scores) or samples <= factor_count:
        raise ValueError("too few paired observations for the requested diffusion rank")
    scaled = (innovations - np.mean(innovations, axis=0)).reshape(samples, -1) * np.sqrt(252.0)
    scores = np.asarray(spot_scores) - np.mean(spot_scores)
    score_scale = float(np.std(scores, ddof=1))
    if score_scale <= 0.0 or not np.isfinite(score_scale):
        raise ValueError("spot innovation variance must be positive")
    scores /= score_scale
    spot_loading = scores @ scaled / (scores @ scores)
    orthogonal = scaled - scores[:, None] * spot_loading[None]
    _, singular, right = np.linalg.svd(orthogonal, full_matrices=False)
    residual = right[:factor_count - 1].T * singular[None, :factor_count - 1] / np.sqrt(samples - 1)
    loadings = np.concatenate((spot_loading[:, None], residual), axis=1)
    total = float(np.sum(scaled**2) / (samples - 1))
    captured = float(np.sum(loadings**2))
    return {
        "q_log_dlvsq_loadings": loadings.reshape(innovations.shape[1:] + (factor_count,)),
        "diffusion_explained_variance_fraction": np.asarray(captured / total if total else 1.0),
        "historical_spot_score_mean": np.asarray(np.mean(spot_scores)),
        "historical_spot_score_sd": np.asarray(score_scale),
    }


def _fit_task(
    source: str, output: str, config: RollingFitConfig,
    fixed_grid: str | None = None, initializer: str | None = None,
) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["stop_reason"] != "checkpointed_stop" and report["fit"]["all"]["inside_percent"] >= config.minimum_inside_percent:
            return report
    with (directory / "fit.log").open("a", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        fitting = config.fitting
        report = run_observed_fit(source_market=Path(source), run_directory=directory, config=fitting,
                                  fixed_grid=Path(fixed_grid) if fixed_grid else None,
                                  initializer=Path(initializer) if initializer else None,
                                  retain_inputs=fixed_grid is None)
        while (
            report["fit"]["all"]["inside_percent"] < config.minimum_inside_percent
            and report["completed_epochs"] < config.maximum_continuation_epochs
            and report["stop_reason"] == "maximum_epochs"
        ):
            fitting = replace(fitting, maximum_epochs=min(fitting.maximum_epochs + config.fitting.maximum_epochs, config.maximum_continuation_epochs))
            report = run_observed_fit(source_market=Path(source), run_directory=directory, config=fitting,
                                      fixed_grid=Path(fixed_grid) if fixed_grid else None,
                                      initializer=Path(initializer) if initializer else None,
                                      retain_inputs=fixed_grid is None)
    jax.clear_caches()
    return report


def fit_summary(reports: list[dict[str, object]], *, threshold: float) -> dict[str, object]:
    if not reports:
        return {"dates": 0, "ready": False}
    coverage = np.asarray([report["fit"]["all"]["inside_percent"] for report in reports])
    counts = np.asarray([report["fit"]["all"]["quotes"] for report in reports])
    failed = [report["date"] for report in reports if report["fit"]["all"]["inside_percent"] < threshold]
    unstable = [report["date"] for report in reports if max(report["density"]["maximum_mass_error"], report["density"]["maximum_forward_error"]) > 1.0e-10 or report["density"]["minimum_probability"] < -1.0e-12]
    unfinished = [report["date"] for report in reports if report["stop_reason"] == "checkpointed_stop"]
    return {
        "dates": len(reports), "quotes": int(np.sum(counts)),
        "quote_weighted_inside_percent": float(np.average(coverage, weights=counts)),
        "date_mean_inside_percent": float(np.mean(coverage)),
        "date_coverage_quantiles": dict(zip(("min", "p05", "median", "p95", "max"), np.quantile(coverage, [0.0, 0.05, 0.5, 0.95, 1.0]).tolist(), strict=True)),
        "dates_below_threshold": failed, "unstable_density_dates": unstable,
        "unfinished_dates": unfinished, "ready": not failed and not unstable and not unfinished,
    }


def run_pipeline(
    *, source_root: Path | None = None, run_directory: Path, config: RollingFitConfig,
    inventory_only: bool = False, fit_available_only: bool = False,
    origin_date: str | None = None, source_inventory_path: Path | None = None,
) -> None:
    """Run every own-date fit before starting the rolling reference-grid stage."""
    run_directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(run_directory / "pipeline.lock"), timeout=0):
        write_json(run_directory / "configuration.json", asdict(config))
        if (source_root is None) == (source_inventory_path is None):
            raise ValueError("provide exactly one source root or pinned parameter inventory")
        inventory = saved_parameter_inventory(source_inventory_path, config=config) if source_inventory_path is not None else source_inventory(source_root, config=config)
        if origin_date is not None:
            inventory = select_origin_inventory(inventory, origin=origin_date, window=config.history_business_days)
        write_json(run_directory / "source_inventory.json", inventory)
        print(json.dumps(dict(event="source_inventory", model=config.fitting.model_label, date_count=inventory["date_count"], complete_horizon_dates=inventory["complete_horizon_dates"], native_terminal_pricing_dates=len(inventory.get("dates_using_native_terminal_pricing", [])), source_refitting_permitted=False, ready=inventory["ready"])), flush=True)
        if inventory_only:
            return
        original_date_count = inventory["date_count"]
        if fit_available_only:
            if inventory.get("source_mode") != "saved_parameters":
                inventory = {**inventory, "dates": {day: item for day, item in inventory["dates"].items()
                                                    if item["terminal_business_dte"] >= inventory["required_terminal_business_dte"] - 1.0e-5}}
            inventory["date_count"] = len(inventory["dates"])
            if not inventory["date_count"]:
                raise RuntimeError("no dates have complete saved source surfaces")
        elif not inventory["ready"]:
            raise RuntimeError(f"complete saved six-month source surfaces are missing on {len(inventory['incomplete_horizon_dates'])} dates; see source_inventory.json")
        stop = threading.Event()
        done = threading.Event()
        started = time.monotonic()
        own_dates = inventory["dates"] if origin_date is None else {origin_date: inventory["dates"][origin_date]}
        status = {"stage": "own_date_fits", "completed": 0, "total": len(own_dates)}

        def request_stop(*_: object) -> None:
            stop.set()
            for child in multiprocessing.active_children():
                if child.pid is not None:
                    os.kill(child.pid, signal.SIGTERM)

        previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}

        def monitor() -> None:
            while not done.wait(60):
                elapsed = time.monotonic() - started
                print(json.dumps(dict(event="pipeline_heartbeat", **status, elapsed_seconds=elapsed)), flush=True)
                try:
                    result = subprocess.run(["gpuctl", "resources", "--json"], capture_output=True, text=True, check=True, timeout=15)
                    checks = json.loads(result.stdout)["checks"]
                    unhealthy = [item for item in checks if item["health"] != "OK"]
                    if unhealthy:
                        print(json.dumps(dict(event="resource_alert", checks=unhealthy)), flush=True)
                    if any(item["health"] in {"STOP", "ERROR", "UNAVAILABLE"} for item in unhealthy):
                        request_stop()
                        if not done.wait(60):
                            for child in multiprocessing.active_children():
                                if child.pid is not None and child.is_alive():
                                    os.kill(child.pid, signal.SIGKILL)
                            os.kill(os.getpid(), signal.SIGKILL)
                        return
                except Exception as error:
                    print(json.dumps(dict(event="resource_warning", message=str(error))), flush=True)

        threading.Thread(target=monitor, daemon=True).start()
        context = multiprocessing.get_context("spawn")
        reports = []
        try:
            with ProcessPoolExecutor(max_workers=config.workers, mp_context=context) as executor:
                futures = {
                    executor.submit(_fit_task, item["path"], str(run_directory / "own_dates" / day), config): day
                    for day, item in own_dates.items()
                }
                for future in as_completed(futures):
                    if stop.is_set():
                        for pending in futures:
                            pending.cancel()
                        raise InterruptedError("rolling calibration stopped; saved checkpoints retained")
                    report = future.result()
                    reports.append(report)
                    status["completed"] = len(reports)
                    summary = fit_summary(reports, threshold=config.minimum_inside_percent)
                    write_json(run_directory / "own_date_fit_summary.json", summary)
                    print(json.dumps(dict(event="date_fitted", date=report["date"], completed=len(reports), total=status["total"], inside_percent=report["fit"]["all"]["inside_percent"])), flush=True)
            if fit_available_only:
                summary = {**fit_summary(reports, threshold=config.minimum_inside_percent),
                           "scope": "available six-month sources only", "requested_date_count": original_date_count,
                           "rolling_ar1_started": False}
                write_json(run_directory / "available_own_date_fit_summary.json", summary)
                return
            if not fit_summary(reports, threshold=config.minimum_inside_percent)["ready"]:
                raise RuntimeError("some own-date fits require calibration review before rolling AR(1); see own_date_fit_summary.json")
            rolling_stage(inventory=inventory, run_directory=run_directory, config=config, status=status, origin_date=origin_date)
        finally:
            done.set()
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)


def rolling_stage(*, inventory: dict[str, object], run_directory: Path, config: RollingFitConfig, status: dict[str, object], origin_date: str | None = None) -> None:
    """Recalibrate each past window on the origin's grid, then estimate its AR(1)."""
    dates = sorted(inventory["dates"])
    origins = [index for index, day in enumerate(dates) if day >= config.rolling_start_date
               and (origin_date is None or day == origin_date)
               and count_business_days_between(date.fromisoformat(dates[0]), date.fromisoformat(day)) >= config.history_business_days]
    status.update(stage="rolling_reference_grid_ar1", completed=0, total=len(origins))
    context = multiprocessing.get_context("spawn")
    for number, origin_index in enumerate(origins, start=1):
        origin = dates[origin_index]
        directory = run_directory / "rolling" / origin
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "report.json").exists():
            status["completed"] = number
            continue
        history_indices = training_prefix(dates, origin_index, window=config.history_business_days)
        grid = run_directory / "own_dates" / origin / "inputs.npz"
        historical_reports = []
        if origin_date is not None:
            initializer = None
            for index in history_indices:
                historical_date = dates[index]
                output = directory / "history" / historical_date
                report = _fit_task(
                    inventory["dates"][historical_date]["path"], str(output), config, str(grid),
                    str(initializer) if initializer is not None else None,
                )
                if report["stop_reason"] == "checkpointed_stop":
                    raise InterruptedError("historical fit checkpointed; single-origin estimation stopped")
                historical_reports.append(report)
                initializer = output
                print(json.dumps(dict(event="history_refitted", origin=origin, historical_date=historical_date, completed=len(historical_reports), total=len(history_indices), inside_percent=report["fit"]["all"]["inside_percent"])), flush=True)
        else:
            historical_reports = fit_history_parallel(
                history_indices=history_indices, dates=dates, inventory=inventory,
                directory=directory, grid=grid, run_directory=run_directory,
                config=config, context=context, origin=origin,
            )
        summary = fit_summary(historical_reports, threshold=config.minimum_inside_percent)
        write_json(directory / "fit_summary.json", summary)
        if not summary["ready"]:
            raise RuntimeError(f"historical fits on {origin}'s grid need review; AR(1) was not estimated")
        with np.load(grid, allow_pickle=False) as saved:
            supports = np.asarray(saved["supports"])
            expiries = np.asarray(saved["model_expiries_y"])
        history_dates = [dates[index] for index in history_indices]
        states, fronts, spots, forwards = [], [], [], []
        for day in history_dates:
            with np.load(directory / "history" / day / "fit.npz", allow_pickle=False) as saved:
                states.append(np.asarray(saved["dlvsq"]))
                fronts.append(np.asarray(saved["densities"])[0])
                spots.append(float(saved["spot_ref"]))
                forwards.append(np.asarray(saved["forward_over_spot"]))
        estimate = estimate_window(dates=history_dates, log_dlvsq=np.log(np.asarray(states)), supports=supports,
                                   model_expiries_y=expiries, forward_over_spot=np.asarray(forwards),
                                   spot_ref=np.asarray(spots), front_densities=np.asarray(fronts),
                                   backbone_vol=config.fitting.geometry.backbone_annual_volatility)
        estimate.update(joint_innovation_loadings(estimate["innovations"], estimate["spot_gaussian_scores"], factor_count=config.diffusion_factors))
        _atomic_npz(directory / "ar1.npz", {**estimate, "history_dates": np.asarray(history_dates), "reference_date": np.asarray(origin),
                                           "supports": supports, "model_expiries_y": expiries})
        report = {"model_label": config.fitting.model_label, "origin": origin, "history_start": history_dates[0], "history_end": history_dates[-1],
                  "history_dates": len(history_dates), "one_session_pairs": len(estimate["pair_end_dates"]),
                  "fixed_reference_grid": str(grid), "origin_excluded_from_ar1": True,
                  "historical_calibration": summary,
                  "fixed_contract_clipped_query_fraction": float(estimate["fixed_contract_clipped_query_fraction"]),
                  "absolute_ar1_slope_above_one_fraction": float(np.mean(np.abs(estimate["slope"]) > 1.0)),
                  "innovation_sd_median": float(np.median(np.sqrt(estimate["innovation_variance"])))}
        write_json(directory / "report.json", report)
        status["completed"] = number
        write_json(run_directory / "rolling_progress.json", {"model_label": config.fitting.model_label, "completed_origins": number, "total_origins": len(origins), "last_origin": origin})
        print(json.dumps(dict(event="rolling_origin_finished", completed=number, total=len(origins), **report)), flush=True)
    write_json(run_directory / "rolling_complete.json", {"model_label": config.fitting.model_label, "completed_origins": len(origins)})


def fit_history_parallel(
    *, history_indices: list[int], dates: list[str], inventory: dict[str, object],
    directory: Path, grid: Path, run_directory: Path, config: RollingFitConfig,
    context: multiprocessing.context.BaseContext, origin: str,
) -> list[dict[str, object]]:
    """Preserve the existing independent-history mode for legacy all-date jobs."""
    historical_reports = []
    with ProcessPoolExecutor(max_workers=config.workers, mp_context=context) as executor:
        futures = {
            executor.submit(
                _fit_task, inventory["dates"][dates[index]]["path"], str(directory / "history" / dates[index]),
                config, str(grid), str(run_directory / "own_dates" / dates[index]),
            ): dates[index]
            for index in history_indices
        }
        for future in as_completed(futures):
            report = future.result()
            historical_reports.append(report)
            print(json.dumps(dict(event="history_refitted", origin=origin, historical_date=report["date"], completed=len(historical_reports), total=len(history_indices), inside_percent=report["fit"]["all"]["inside_percent"])), flush=True)
    return historical_reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--source-root", type=Path)
    sources.add_argument("--source-inventory", type=Path, help="pin existing trained SANOS parameters, without source refitting")
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model-label", default="Gromm v2 4,3")
    parser.add_argument("--origin-date", help="estimate only this date, retaining its preceding historical window")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--fit-available-only", action="store_true")
    arguments = parser.parse_args()
    run_pipeline(source_root=arguments.source_root, run_directory=arguments.run_directory,
                 config=RollingFitConfig(fitting=ObservedFitConfig(model_label=arguments.model_label), workers=arguments.workers), inventory_only=arguments.inventory_only,
                 fit_available_only=arguments.fit_available_only, origin_date=arguments.origin_date,
                 source_inventory_path=arguments.source_inventory)


if __name__ == "__main__":
    main()
