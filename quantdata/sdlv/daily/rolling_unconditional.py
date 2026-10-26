"""Rolling 252-observation, restricted-universe Q audits of the wider GROMM fit.

Each date uses 100,000 paths / 50,000 antithetic pairs, 20 business days,
eight factors and eight substeps. Save pair moments, not large path datasets.
The coordinator validates 29 August first and releases the GPU between dates.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from datetime import date
from pathlib import Path

import jax
import numpy as np
from filelock import FileLock

from quantdata import env
from quantdata.sdlv.daily.front_spot import run_oce
from quantdata.sdlv.daily.front_spot.estimation import (
    align_fixed_contract_history,
    fit_joint_innovations,
    rolling_history_indices,
)
from quantdata.sdlv.daily.front_spot.simulation import load_initial_model_state
from quantdata.sdlv.daily.wide_audit import AuditMonitor, emit, load_source, resource_health, sha256

# The pre-backend-switch driver has identical simulation and moment formulas.
# Only this reviewed predecessor may cross a driver-hash change on resumption;
# all imported numerical modules must still match their original hashes.
_GPU_ONLY_DRIVER_SHA256 = "5c57ea71e4e33c5431625bd19e801497c59c9ec485b49ae49d48c6e6432bf5aa"


def validate_resume_identity(previous: dict, current: dict) -> None:
    """Allow a device migration, never changed seeds, data or numerical kernels."""
    previous = {key: value for key, value in previous.items() if key != "compute_backend"}
    current = {key: value for key, value in current.items() if key != "compute_backend"}
    old_code = dict(previous.pop("code_sha256"))
    new_code = dict(current.pop("code_sha256"))
    driver = str(Path(__file__).resolve())
    old_driver, new_driver = old_code.pop(driver), new_code.pop(driver)
    if old_driver not in {new_driver, _GPU_ONLY_DRIVER_SHA256}:
        raise ValueError("Unreviewed resume driver change")
    if previous != current or old_code != new_code:
        raise ValueError("Resume identity changed: seeds, geometry, source or numerical code differ")


def reuse_completed_chunks(source: Path, output: Path, *, paths: int, chunk_size: int) -> dict:
    """Copy only validated, contiguous complete-pair moment checkpoints."""
    if source.resolve() == output.resolve():
        raise ValueError("Backend continuation needs a distinct output directory")
    destination = output / "pair-moments"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    completed = 0
    for start in range(0, paths, chunk_size):
        original = source / "pair-moments" / f"{start:06d}.npz"
        if not original.exists():
            break
        count = min(chunk_size, paths - start)
        with np.load(original, allow_pickle=False) as saved:
            if int(saved["pair_count"]) * 2 != count:
                raise ValueError("Resume chunk has an incomplete pair count")
            if saved["mean"].shape != (20, 45) or saved["m2"].shape != (20, 45):
                raise ValueError("Resume chunk has incompatible coordinate geometry")
            if not np.all(np.isfinite(saved["mean"])) or not np.all(np.isfinite(saved["m2"])):
                raise ValueError("Resume chunk has non-finite moments")
            if np.any(saved["m2"] < 0) or saved["active_mask"].shape != (20, 45):
                raise ValueError("Resume chunk has invalid moments or activity mask")
        target = destination / original.name
        digest = sha256(original)
        if target.exists():
            if sha256(target) != digest:
                raise ValueError("Previously imported chunk changed")
        else:
            temporary = target.with_suffix(".copy-tmp")
            shutil.copy2(original, temporary)
            if sha256(temporary) != digest:
                raise ValueError("Resume chunk changed while being copied")
            temporary.replace(target)
        completed += count
        records.append(dict(start=start, path_count=count, sha256=digest))
    return dict(source_directory=str(source.resolve()), reused_paths=completed,
                remaining_paths=paths - completed, chunks=records)


def pair_moments(returns: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Float64 moments of complete adjacent antithetic pairs."""
    if returns.ndim != 3 or len(returns) < 4 or len(returns) % 2 or not np.all(np.isfinite(returns)):
        raise ValueError("Finite returns of complete pairs are required")
    pairs = (returns[::2].astype(np.float64) + returns[1::2].astype(np.float64)) / 2
    mean = pairs.mean(axis=0)
    return len(pairs), mean, np.sum((pairs - mean)**2, axis=0)


def merge_moments(left: tuple, right: tuple) -> tuple[int, np.ndarray, np.ndarray]:
    """Chan's stable combination of pair counts, means and centered sums."""
    n, mean, m2 = left
    k, other, other_m2 = right
    if n == 0:
        return k, other.copy(), other_m2.copy()
    delta = other - mean
    return n + k, mean + delta * (k / (n + k)), m2 + other_m2 + delta**2 * (n * k / (n + k))


def audit_from_moments(moments: tuple, active_mask: np.ndarray, labels: tuple[str, ...]) -> tuple[dict, dict]:
    """Same coordinate tests as the paper, retaining daily means/SEs/z scores."""
    count, mean, m2 = moments
    if count < 2 or mean.shape != active_mask.shape or mean.shape[1] != len(labels):
        raise ValueError("Invalid pair moments or coordinate labels")
    se = np.sqrt(np.maximum(m2, 0) / (count - 1) / count)
    tested = active_mask & (se > 1e-15)
    z = np.divide(mean, se, out=np.zeros_like(mean), where=tested)
    report = dict(null_hypothesis="E_Q[(V_{t+1}^i-V_t^i)/S_t]=0 for every live day and instrument",
                  path_count=2 * count, antithetic_pair_count=count,
                  sampling_unit="complete antithetic path pair; each coordinate tested separately")
    for name in ("all_instruments", "spot", "options"):
        mask = tested.copy()
        if name == "spot":
            mask[:, 1:] = False
        elif name == "options":
            mask[:, 0] = False
        values = np.abs(z[mask])
        active_indices = np.flatnonzero(mask)
        order = np.argsort(values)[-10:][::-1]
        top = []
        for index in order:
            day, instrument = np.unravel_index(active_indices[index], mean.shape)
            top.append(dict(day_index=int(day), instrument_index=int(instrument),
                            instrument_label=labels[instrument], mean=float(mean[day, instrument]),
                            pair_based_standard_error=float(se[day, instrument]), z_score=float(z[day, instrument])))
        report[name] = dict(tested_coordinate_count=int(mask.sum()),
                            max_absolute_z=float(values.max()) if len(values) else 0.0,
                            p99_absolute_z=float(np.quantile(values, 0.99)) if len(values) else 0.0,
                            rms_z=float(np.sqrt(np.mean(values**2))) if len(values) else 0.0,
                            count_above_2_576=int(np.sum(values > 2.576)), count_above_3=int(np.sum(values > 3)),
                            count_above_5=int(np.sum(values > 5)), top_absolute_z_coordinates=top)
    report["degenerate_live_coordinate_count"] = int(np.sum(active_mask & ~tested))
    arrays = dict(pair_count=np.array(count), mean=mean, pair_based_standard_error=se,
                  z_score=z, active_mask=active_mask, tested_mask=tested, instrument_labels=np.asarray(labels))
    return report, arrays


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _check_environment() -> None:
    if not all(os.environ.get(key) for key in ("QUANTDATA_ROOT", "PYTHONPATH")):
        raise RuntimeError("Explicit QUANTDATA_ROOT and PYTHONPATH are required")
    emit(event="environment", interpreter=sys.executable, cwd=os.getcwd(), locations=str(env.locations))


def run_date(args: argparse.Namespace) -> None:
    """Fit only past-date innovations and checkpoint the unconditional audit."""
    _check_environment()
    backend = "gpu" if args.device == "cuda" else "cpu"
    if jax.default_backend() != backend or not jax.config.x64_enabled:
        raise RuntimeError(f"This job requires the requested JAX {backend} backend and float64")
    if args.paths < 4 or args.paths % 4 or args.chunk_size < 4 or args.chunk_size % 4:
        raise ValueError("Path and chunk counts must be positive multiples of four")
    if any(check["health"] != "OK" for check in resource_health()):
        raise RuntimeError("Resource checks must be OK before a date starts")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    monitor = AuditMonitor(output)
    monitor.thread.start()
    try:
        with FileLock(output / "audit.lock", timeout=0):
            source_directory = output.parent / "shared-source"
            source_directory.mkdir(exist_ok=True)
            raw, source = load_source(args.training_run, source_directory, None)
            source = {**source, "checkpoint_last_fitted_date": source["initial_date"], "initial_date": args.date}
            with np.load(args.training_run / "market.npz", allow_pickle=False) as data:
                dates = tuple(data["dates"].astype(str))
            reference = date.fromisoformat(args.date)
            reference_index = dates.index(args.date)
            window = rolling_history_indices(dates, reference_date=reference, date_count=252)
            window_start, window_end = dates[window[0]], dates[window[-1]]
            code_paths = [Path(__file__), Path(__file__).with_name("wide_audit.py")]
            code_paths += sorted((Path(__file__).parent / "front_spot").glob("*.py"))
            identity = dict(schema="gromm-wide-rolling252-unconditional-v1", source=source,
                            reference_date=args.date, window_start=window_start, window_end=window_end,
                            window_date_count=252, window_dates=[dates[index] for index in window],
                            paths=args.paths, chunk_size=args.chunk_size, generation_batch_size=args.batch_size,
                            seed=args.seed, horizon_days=20, factor_rank=8, dlv_substeps=8,
                            compute_backend=backend,
                            code_sha256={str(path.resolve()): sha256(path) for path in code_paths})
            identity_path = output / "identity.json"
            if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
                raise ValueError("Run identity changed; choose a new output directory")
            run_oce._atomic_json(identity, identity_path)
            if (output / "unconditional.json").exists():
                return
            monitor.stage("calibrate-rolling-diffusion")
            history = align_fixed_contract_history(dataset_path=args.training_run / "market.npz", raw_grids=raw,
                                                    training_start=date.fromisoformat(window_start),
                                                    training_end=date.fromisoformat(window_end))
            if np.any(history.origin_indices < window[0]) or np.any(history.endpoint_indices > window[-1]):
                raise RuntimeError("Historical pairs escaped the past-only training window")
            innovations = fit_joint_innovations(history, factor_rank=8, smoothing_sigma=(1.0, 2.0))
            initial = load_initial_model_state(dataset_path=args.training_run / "market.npz", raw_grids=raw,
                                               date_index=reference_index)
            if args.resume_from is not None:
                previous = json.loads((args.resume_from / "identity.json").read_text())
                validate_resume_identity(previous, identity)
                with np.load(args.resume_from / "diffusion.npz", allow_pickle=False) as saved:
                    for name, value in (("annualized_loadings", innovations.annualized_loadings),
                                        ("eigenvalues", innovations.daily_eigenvalues),
                                        ("support", initial.support), ("dlvsq", initial.dlvsq),
                                        ("density_tower", initial.density_tower)):
                        np.testing.assert_allclose(value, saved[name], rtol=1e-10, atol=1e-12,
                                                   err_msg=f"Resume model changed: {name}")
                    np.testing.assert_array_equal(history.origin_indices, saved["origin_indices"])
                    np.testing.assert_array_equal(history.endpoint_indices, saved["endpoint_indices"])
                    # Retain the exact saved starting state and diffusion, even
                    # if CPU/GPU reconstruction differs by floating-point roundoff.
                    initial = replace(initial, support=saved["support"].copy(), dlvsq=saved["dlvsq"].copy(),
                                      density_tower=saved["density_tower"].copy())
                    innovations = replace(innovations, annualized_loadings=saved["annualized_loadings"].copy(),
                                          daily_eigenvalues=saved["eigenvalues"].copy())
                continuation = reuse_completed_chunks(args.resume_from, output, paths=args.paths,
                                                      chunk_size=args.chunk_size)
                continuation.update(source_backend=previous.get("compute_backend", "gpu"),
                                    continuation_backend=backend, source_identity_sha256=sha256(args.resume_from / "identity.json"),
                                    source_diffusion_sha256=sha256(args.resume_from / "diffusion.npz"),
                                    source_code_sha256=previous["code_sha256"],
                                    numerical_precision="float64 dynamics; existing float32 return storage retained")
                run_oce._atomic_json(continuation, output / "continuation.json")
                emit(event="checkpoint_continuation", reused_paths=continuation["reused_paths"],
                     remaining_paths=continuation["remaining_paths"], backend=backend)
            tower = initial.density_tower
            if np.any(~np.isfinite(tower)) or tower.min() < 0:
                raise FloatingPointError("Reference density is not finite and positive")
            np.testing.assert_allclose(tower.sum(axis=-1), 1, rtol=0, atol=1e-10)
            np.testing.assert_allclose(tower @ initial.support, 1, rtol=0, atol=1e-10)
            calibration = dict(reference_date=args.date, window_start=window_start, window_end=window_end,
                               window_date_count=252, training_pair_count=len(history.origin_indices),
                               first_origin=dates[history.origin_indices[0]], last_endpoint=dates[history.endpoint_indices[-1]],
                               factor_rank=8, smoothing_sigma=[1.0, 2.0],
                               smoothed_explained_variance_fraction=innovations.explained_variance_fraction,
                               spot_score_mean=innovations.spot_score_mean, spot_score_std=innovations.spot_score_std)
            run_oce._atomic_json(calibration, output / "calibration.json")
            _atomic_npz(output / "diffusion.npz", annualized_loadings=innovations.annualized_loadings,
                        eigenvalues=innovations.daily_eigenvalues, origin_indices=history.origin_indices,
                        endpoint_indices=history.endpoint_indices, support=initial.support, dlvsq=initial.dlvsq,
                        density_tower=tower)
            chunks = output / "pair-moments"
            chunks.mkdir(exist_ok=True)
            combined = (0, np.zeros((20, 45)), np.zeros((20, 45)))
            generation_seconds = 0.0
            for start in range(0, args.paths, args.chunk_size):
                count = min(args.chunk_size, args.paths - start)
                monitor.stage("generate-unconditional-Q", start, args.paths)
                chunk = chunks / f"{start:06d}.npz"
                if not chunk.exists():
                    population = run_oce.generate_front_spot_oce_population(
                        None, innovations=innovations, initial_state=initial, record_features=False,
                        path_count=count, batch_size=args.batch_size, dlv_substeps=8,
                        seed=args.seed + start, measure="Q")
                    n, mean, m2 = pair_moments(population.returns)
                    _atomic_npz(chunk, pair_count=np.array(n), mean=mean, m2=m2,
                                active_mask=population.active_mask, strikes=population.strikes,
                                instrument_labels=np.asarray(population.instrument_labels),
                                generation_seconds=np.array(population.generation_seconds))
                    del population
                with np.load(chunk, allow_pickle=False) as saved:
                    if int(saved["pair_count"]) * 2 != count:
                        raise ValueError("Incomplete antithetic moment checkpoint")
                    combined = merge_moments(combined, (int(saved["pair_count"]), saved["mean"], saved["m2"]))
                    active, labels = saved["active_mask"], tuple(saved["instrument_labels"].astype(str))
                    generation_seconds += float(saved["generation_seconds"])
                run_oce._atomic_json(dict(completed_paths=2 * combined[0], requested_paths=args.paths), output / "checkpoint.json")
            result, arrays = audit_from_moments(combined, active, labels)
            _atomic_npz(output / "coordinate-statistics.npz", **arrays)
            result.update(status="complete", reference_date=args.date, calibration=calibration,
                          source=source, generation_seconds=generation_seconds,
                          interpretation="Coordinatewise unconditional tests only; no conditional-policy or all-strategy claim")
            if args.resume_from is not None:
                result["continuation"] = continuation
            result["compute_backend"] = backend
            run_oce._atomic_json(result, output / "unconditional.json")
            monitor.stage("complete", args.paths, args.paths)
            emit(event="date_complete", date=args.date, summary=result["all_instruments"])
    finally:
        monitor.done.set()


def coordinate(args: argparse.Namespace) -> None:
    """Run the reference smoke test, then every date, releasing each allocation."""
    _check_environment()
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError("Run outputs must be outside the source worktree")
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.training_run / "market.npz", allow_pickle=False) as data:
        dates = tuple(data["dates"].astype(str))
    eligible = dates[252:]
    if "2025-08-29" not in eligible:
        raise ValueError("Reference date is not eligible")
    scheduled = ["2025-08-29"] + [value for value in eligible if value != "2025-08-29"]
    plan = dict(reference_preflight="2025-08-29", evaluation_dates=scheduled, total_dates=len(scheduled),
                paths_per_date=args.paths, pairs_per_date=args.paths // 2, historical_dates=252,
                horizon_days=20, factor_rank=8, dlv_substeps=8, conditional_tests=False,
                chunk_size=args.chunk_size, batch_size=args.batch_size, seed=args.seed,
                training_report_sha256=sha256(args.training_run / "fit/report.json"),
                dataset_sha256=sha256(args.training_run / "market.npz"),
                code_sha256={str(path.resolve()): sha256(path) for path in
                             [Path(__file__), Path(__file__).with_name("wide_audit.py"),
                              *sorted((Path(__file__).parent / "front_spot").glob("*.py"))]})
    plan_path = args.output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing run plan differs")
    run_oce._atomic_json(plan, plan_path)
    tasks = [("2025-08-29", 512, args.output / "preflight", args.seed + 100_000_000)]
    tasks += [(value, args.paths, args.output / value, args.seed + dates.index(value) * 1_000_000) for value in scheduled]
    completed = []
    for value, paths, output, seed in tasks:
        report_path = output / "unconditional.json"
        if not report_path.exists():
            while True:
                checks = resource_health()
                if any(check["health"] in {"STOP", "ERROR", "UNAVAILABLE"} for check in checks):
                    raise RuntimeError("Unsafe resources; assess the cause before restarting")
                if all(check["health"] == "OK" for check in checks):
                    status = subprocess.run(["gpuctl", "status", "--json"], capture_output=True, text=True, check=True)
                    if json.loads(status.stdout)["available"] > 0:
                        break
                emit(event="waiting_for_managed_gpu_or_headroom", date=value, checks=checks)
                time.sleep(60)
            command = ["gpuctl", "run", "--gpus", "1", "--task-name", "Gromm-v1-4,3-rolling252-Q",
                       "--label", f"unconditional-{value}-{paths}-paths", "--", sys.executable, "-u", "-m",
                       "quantdata.sdlv.daily.rolling_unconditional", "--one-date", "--date", value,
                       "--training-run", str(args.training_run), "--output", str(output), "--paths", str(paths),
                       "--chunk-size", str(args.chunk_size), "--batch-size", str(args.batch_size), "--seed", str(seed)]
            emit(event="launch", date=value, paths=paths, completed_dates=len(completed), total_dates=len(scheduled))
            subprocess.run(command, check=True, env={**os.environ, "JAX_PLATFORMS": "cuda"},
                           cwd=Path(__file__).resolve().parents[3])
        report = json.loads(report_path.read_text())
        if report["status"] != "complete" or report["path_count"] != paths or report["reference_date"] != value:
            raise ValueError("Saved date result does not match the scheduled audit")
        if output.name == "preflight":
            emit(event="preflight_numerically_complete", date=value,
                 note="Proceed on numerical validity, not a significance-based selection")
            continue
        completed.append(dict(date=value, **report["all_instruments"], calibration=report["calibration"],
                              generation_seconds=report["generation_seconds"]))
        run_oce._atomic_json(dict(status="complete" if len(completed) == len(scheduled) else "running",
                                 completed_dates=len(completed), total_dates=len(scheduled), results=completed),
                             args.output / "summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--date", default="2025-08-29")
    parser.add_argument("--one-date", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume-from", type=Path,
                        help="Import validated moments into a distinct single-date continuation directory")
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    if not args.one_date and (args.device != "cuda" or args.resume_from is not None):
        parser.error("CPU and checkpoint migration require --one-date; they never launch the all-date scheduler")
    with FileLock(args.output.with_suffix(".coordinator.lock") if not args.one_date else args.output.with_suffix(".launch.lock"), timeout=0):
        if args.one_date:
            run_date(args)
        else:
            coordinate(args)


if __name__ == "__main__":
    main()
