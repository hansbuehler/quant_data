"""Unconditional one-day Q martingale diagnostics for each rolling estimate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version
from filelock import FileLock
from scipy.stats import norm

from quantdata.gromm2.calibration import _atomic_npz
from quantdata.gromm2.gpu_forward_queue import resource_report
from quantdata.gromm2.pricing import price_active_calls
from quantdata.gromm2.rolling_calibration import write_json
from quantdata.gromm2.simulation import SimulationState, one_day_risk_neutral_step


@version("0.2")
@dataclass(frozen=True, kw_only=True)
class UnconditionalAuditConfig:
    """Fixed sampling plan with no learned action or empirical measure change."""

    path_count: int = 100_000
    batch_size: int = 256
    substeps: int = 8
    seed: int = 20261024
    standard_error_floor: float = 1.0e-8
    density_loading_floor: float = 1.0e-4
    nms: tuple[float, ...] = (-4.0, -3.5, -2.5, -1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.5, 3.0)

    def __post_init__(self) -> None:
        verify_inp(self.path_count >= 4 and self.path_count % 2 == 0, lambda: "path_count must contain at least two complete antithetic pairs")
        verify_inp(self.batch_size >= 2 and self.batch_size % 2 == 0, lambda: "batch_size must contain complete antithetic pairs")
        verify_inp(self.substeps > 0, lambda: "substeps must be positive")
        verify_inp(self.standard_error_floor > 0.0, lambda: "standard_error_floor must be positive")
        verify_inp(0.0 < self.density_loading_floor < 1.0, lambda: "invalid density localization floor")
        verify_inp(len(self.nms) >= 3 and np.all(np.isfinite(self.nms)), lambda: "audit NMS levels must be finite")


class ResourceHeadroomError(RuntimeError):
    """A monitored resource forbids starting or continuing material work."""


def check_resource_headroom() -> None:
    """Share timeout retries with the supervisor; use only manager thresholds."""
    resources = resource_report()
    if resources["warning"]:
        print(json.dumps(dict(event="resource_alert" if resources["blocked"] else "resource_warning", **resources)), flush=True)
    if resources["blocked"]:
        raise ResourceHeadroomError("resource health forbids material work")


def paired_samples(values: np.ndarray) -> np.ndarray:
    """Treat adjacent antithetic partners as one independent observation."""
    if len(values) % 2:
        raise ValueError("an audit batch must contain complete antithetic pairs")
    return 0.5 * (values[0::2] + values[1::2])


def combine_moments(
    count: int, mean: np.ndarray, centered_sum_squares: np.ndarray,
    observations: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Combine independent pair scores without retaining path arrays."""
    new_count = len(observations)
    if not new_count:
        return count, mean, centered_sum_squares
    batch_mean = np.mean(observations, axis=0)
    batch_squares = np.sum((observations - batch_mean)**2, axis=0)
    total = count + new_count
    difference = batch_mean - mean
    return (
        total, mean + difference * new_count / total,
        centered_sum_squares + batch_squares + difference**2 * count * new_count / total,
    )


def moment_statistics(
    count: int, mean: np.ndarray, centered_sum_squares: np.ndarray, *, epsilon: float,
) -> dict[str, np.ndarray]:
    """Ordinary and economic-floor scores; the latter is not a Gaussian z score."""
    if count < 2:
        raise ValueError("at least two independent antithetic pairs are required")
    deviation = np.sqrt(np.maximum(centered_sum_squares / (count - 1), 0.0))
    standard_error = deviation / np.sqrt(count)
    z = np.divide(mean, standard_error, out=np.zeros_like(mean), where=standard_error > 0.0)
    z = np.where((standard_error == 0.0) & (mean != 0.0), np.copysign(np.inf, mean), z)
    return {
        "mean_gain_over_spot": mean, "pair_standard_deviation": deviation,
        "standard_error": standard_error, "z": z,
        "economic_floor_score": mean / (epsilon + standard_error),
        "pointwise_99_ci_lower": mean - norm.ppf(0.995) * standard_error,
        "pointwise_99_ci_upper": mean + norm.ppf(0.995) * standard_error,
    }


def antithetic_shocks(*, seed: int, completed_paths: int, paths: int, substeps: int, factors: int) -> tuple[np.ndarray, np.ndarray]:
    if paths % 2:
        raise ValueError("shock batches must contain complete antithetic pairs")
    generator = np.random.default_rng(np.random.SeedSequence([seed, completed_paths]))
    factor_half = generator.standard_normal((substeps, paths // 2, factors))
    background_half = generator.standard_normal(paths // 2)
    factor = np.stack((factor_half, -factor_half), axis=2).reshape(substeps, paths, factors)
    background = np.stack((background_half, -background_half), axis=1).reshape(paths)
    return factor, background


def audit_batch_function(
    *, supports: np.ndarray, transports: np.ndarray, expiries_y: np.ndarray,
    dlvsq: np.ndarray, initial_densities: np.ndarray, loadings: np.ndarray,
    projected_variances: np.ndarray, background_volatility: float,
    config: UnconditionalAuditConfig,
):
    """Use the production front bridge and Q drift for one business day."""
    dtes = np.rint(expiries_y * 252.0).astype(int)
    selected = np.flatnonzero((dtes == 1) | (dtes % 5 == 0))
    strikes = np.exp(np.sqrt(projected_variances[selected])[:, None] * np.asarray(config.nms)[None])
    supports_jax, transports_jax = jnp.asarray(supports), jnp.asarray(transports)
    dlvsq_jax, tower_jax, loadings_jax = jnp.asarray(dlvsq), jnp.asarray(initial_densities), jnp.asarray(loadings)
    selected_supports, strikes_jax = supports_jax[selected], jnp.asarray(strikes)
    initial_spot = float(initial_densities[0] @ supports[0])
    initial_calls = price_active_calls(selected_supports, tower_jax[selected][None], jnp.ones(1), jnp.asarray(dtes[selected] / 252.0), strikes_jax, background_annual_volatility=background_volatility)[0]
    initial_prices = jnp.where(strikes_jax >= 1.0, initial_calls, initial_calls - initial_spot + strikes_jax)
    intervals = jnp.asarray(np.diff(expiries_y)[...])

    @jax.jit
    def simulate(factor_normal: jax.Array, background_normal: jax.Array):
        paths = len(background_normal)
        state = SimulationState(
            front_density=jnp.broadcast_to(tower_jax[0], (paths, supports.shape[1])),
            dlvsq=jnp.broadcast_to(dlvsq_jax, (paths,) + dlvsq.shape),
            background_level=jnp.ones(paths),
        )
        result = one_day_risk_neutral_step(
            supports_jax, transports_jax, intervals, state, loadings_jax,
            factor_normal, background_normal, remaining_front_days=jnp.asarray(1),
            substeps=config.substeps, background_annual_volatility=background_volatility,
            density_loading_floor=config.density_loading_floor,
        )
        calls = price_active_calls(
            selected_supports, result.density_tower[:, selected], result.state.background_level,
            jnp.asarray((dtes[selected] - 1) / 252.0), strikes_jax,
            background_annual_volatility=background_volatility,
        )
        prices = jnp.where(strikes_jax >= 1.0, calls, calls - result.spot[:, None, None] + strikes_jax)
        gains = jnp.concatenate(((result.spot - initial_spot)[:, None], (prices - initial_prices).reshape(paths, -1)), axis=1) / initial_spot
        mass_error = jnp.max(jnp.abs(jnp.sum(result.density_tower, axis=-1) - 1.0))
        forward_error = jnp.max(jnp.abs(jnp.sum(result.density_tower * supports_jax, axis=-1) - result.latent_spot[:, None]))
        stable = (
            jnp.all(jnp.isfinite(result.state.dlvsq)) & jnp.all(jnp.isfinite(gains))
            & jnp.all(jnp.isfinite(result.density_tower))
            & jnp.all(result.density_tower >= -1.0e-12)
            & (mass_error <= 1.0e-8) & (forward_error <= 1.0e-8)
        )
        return gains, result.drift_taming_fraction, mass_error, forward_error, stable

    return simulate, {"option_business_dtes": dtes[selected], "option_nms": np.asarray(config.nms), "option_strikes": strikes,
                      "initial_option_prices": np.asarray(initial_prices), "initial_spot": np.asarray(initial_spot)}


def _finite_scalar(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def run_audit(*, calibration_root: Path, origin: str, config: UnconditionalAuditConfig) -> dict[str, object]:
    """Audit one completed rolling estimate, with chunk-level restart state."""
    check_resource_headroom()
    jax.config.update("jax_enable_x64", True)
    directory = calibration_root / "rolling" / origin
    if not (directory / "report.json").is_file():
        raise ValueError("the rolling estimate must finish before its audit")
    inputs_path = calibration_root / "own_dates" / origin / "inputs.npz"
    state_path = calibration_root / "own_dates" / origin / "fit.npz"
    law_path = directory / "ar1.npz"
    identity = json.dumps({
        "schema": "gromm2-unconditional-q-audit-v1", "code_version": config.version.unique_id48,
        "config": asdict(config), "origin": origin,
        "pricing_inputs_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
        "initial_state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        "rolling_law_sha256": hashlib.sha256(law_path.read_bytes()).hexdigest(),
    }, sort_keys=True)
    output = directory / "unconditional_q"
    output.mkdir(exist_ok=True)
    with FileLock(str(output / "audit.lock"), timeout=0):
        report_path = output / "report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report["identity"] != identity:
                raise ValueError("completed audit has a different model or sampling identity")
            return report
        with np.load(inputs_path, allow_pickle=False) as data, np.load(state_path, allow_pickle=False) as state, np.load(law_path, allow_pickle=False) as law:
            if "q_log_dlvsq_loadings" not in law:
                raise ValueError("rolling estimate does not contain the shared P/Q innovation loadings")
            own_report = json.loads(state_path.with_name("report.json").read_text())
            simulate, instruments = audit_batch_function(
                supports=np.asarray(data["supports"]), transports=np.asarray(data["transports"]),
                expiries_y=np.asarray(data["model_expiries_y"]), dlvsq=np.asarray(state["dlvsq"]),
                initial_densities=np.asarray(state["densities"]), loadings=np.asarray(law["q_log_dlvsq_loadings"]),
                projected_variances=np.asarray(data["projected_total_variances"]),
                background_volatility=float(own_report["config"]["geometry"]["backbone_annual_volatility"]), config=config,
            )
            factor_count = law["q_log_dlvsq_loadings"].shape[-1]
        coordinates = 1 + instruments["option_strikes"].size
        completed = pairs = 0
        mean, squares = np.zeros(coordinates), np.zeros(coordinates)
        taming_total = mass_error = forward_error = 0.0
        checkpoint = output / "checkpoint.npz"
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as saved:
                if str(saved["identity"].item()) != identity:
                    raise ValueError("audit checkpoint belongs to another model or sampling plan")
                completed, pairs = int(saved["completed_paths"]), int(saved["pair_count"])
                mean, squares = np.asarray(saved["mean"]), np.asarray(saved["squares"])
                taming_total, mass_error, forward_error = (float(saved[name]) for name in ("taming_total", "mass_error", "forward_error"))
        started, initial_completed = time.monotonic(), completed
        stopped, done = threading.Event(), threading.Event()
        old_handlers = {sig: signal.signal(sig, lambda *_: stopped.set()) for sig in (signal.SIGTERM, signal.SIGINT)}

        def monitor() -> None:
            while not done.wait(60):
                print(json.dumps(dict(event="audit_heartbeat", origin=origin, completed_paths=completed, total_paths=config.path_count, elapsed_seconds=time.monotonic() - started)), flush=True)
                try:
                    check_resource_headroom()
                except ResourceHeadroomError:
                    stopped.set()
                    if not done.wait(60):
                        os.kill(os.getpid(), signal.SIGKILL)
                    return

        threading.Thread(target=monitor, daemon=True).start()
        seed = config.seed + int(origin.replace("-", ""))
        print(json.dumps(dict(event="audit_start", origin=origin, paths=config.path_count, substeps=config.substeps, factors=factor_count, instruments=coordinates, backend=jax.default_backend())), flush=True)
        try:
            for start in range(completed, config.path_count, config.batch_size):
                paths = min(config.batch_size, config.path_count - start)
                factors, background = antithetic_shocks(seed=seed, completed_paths=start, paths=paths, substeps=config.substeps, factors=factor_count)
                gains, taming, mass, forward, stable = simulate(jnp.asarray(factors), jnp.asarray(background))
                values = np.asarray(gains)
                if not bool(stable) or not np.all(np.isfinite(values)):
                    raise FloatingPointError(f"invalid Q simulation or density invariants at {origin}, paths {start}:{start + paths}")
                pairs, mean, squares = combine_moments(pairs, mean, squares, paired_samples(values))
                taming_total += float(taming) * paths
                mass_error, forward_error = max(mass_error, float(mass)), max(forward_error, float(forward))
                completed = start + paths
                _atomic_npz(checkpoint, {
                    "identity": np.asarray(identity), "completed_paths": np.asarray(completed), "pair_count": np.asarray(pairs),
                    "mean": mean, "squares": squares, "taming_total": np.asarray(taming_total),
                    "mass_error": np.asarray(mass_error), "forward_error": np.asarray(forward_error),
                })
                elapsed = time.monotonic() - started
                print(json.dumps(dict(event="audit_progress", origin=origin, completed_paths=completed, total_paths=config.path_count, elapsed_seconds=elapsed,
                                      remaining_seconds=elapsed * (config.path_count - completed) / max(completed - initial_completed, 1))), flush=True)
                if stopped.is_set():
                    raise InterruptedError("audit checkpoint saved; stop requested")
            statistics = moment_statistics(pairs, mean, squares, epsilon=config.standard_error_floor)
            absolute_z = np.abs(statistics["z"])
            family_threshold = float(norm.ppf(1.0 - 0.01 / (2.0 * coordinates)))
            report = {
                "model_label": own_report.get("model_label", "Gromm v2 4,3"), "origin": origin, "identity": identity,
                "measure": "Q", "test": "E_Q[(V(t+1D)-V(t))/S(t)] = 0, separately for each fixed contract and spot",
                "conditional_strategy_training": False, "measure_preconditioning": False,
                "paths": completed, "independent_antithetic_pairs": pairs, "config": asdict(config),
                "coordinate_count": coordinates, "ordinary_99_percentile_absolute_z": _finite_scalar(np.quantile(absolute_z, 0.99, method="higher")),
                "ordinary_maximum_absolute_z": _finite_scalar(np.max(absolute_z)),
                "ordinary_pointwise_99_exceedances": int(np.count_nonzero(absolute_z > norm.ppf(0.995))),
                "ordinary_familywise_99_threshold": family_threshold,
                "ordinary_familywise_99_exceedances": int(np.count_nonzero(absolute_z > family_threshold)),
                "zero_standard_error_nonzero_mean_coordinates": int(np.count_nonzero((statistics["standard_error"] == 0.0) & (mean != 0.0))),
                "maximum_absolute_mean_gain_over_spot": float(np.max(np.abs(mean))),
                "economic_floor_99_percentile_absolute_score": float(np.quantile(np.abs(statistics["economic_floor_score"]), 0.99)),
                "economic_floor_maximum_absolute_score": float(np.max(np.abs(statistics["economic_floor_score"]))),
                "economic_floor_units": "spot-normalized one-day gain; added to the MC standard error, not to the payoff standard deviation",
                "drift_taming_fraction": taming_total / completed,
                "maximum_density_mass_error": mass_error, "maximum_density_forward_error": forward_error,
            }
            _atomic_npz(output / "statistics.npz", {**instruments, **statistics})
            write_json(report_path, report)
            print(json.dumps(dict(event="audit_finished", **report)), flush=True)
            return report
        finally:
            done.set()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def audit_worker_command(
    *, calibration_root: Path, origin: str, config: UnconditionalAuditConfig, backend: str,
) -> list[str]:
    """CPU execution never requests a GPU allocation."""
    if backend not in {"cpu", "cuda"}:
        raise ValueError("audit backend must be cpu or cuda")
    command = ["env", f"JAX_PLATFORMS={backend}", sys.executable, "-u", "-m", "quantdata.gromm2.rolling_audit",
               "--backend", backend,
               "--calibration-root", str(calibration_root), "--origin", origin, "--paths", str(config.path_count),
               "--batch-size", str(config.batch_size), "--substeps", str(config.substeps), "--seed", str(config.seed),
               "--standard-error-floor", str(config.standard_error_floor), "--density-loading-floor", str(config.density_loading_floor),
               "--nms", *(str(level) for level in config.nms)]
    if backend == "cuda":
        gpuctl = shutil.which("gpuctl") or str(Path.home() / ".local/bin/gpuctl")
        command = [gpuctl, "run", "--gpus", "1", "--", *command]
    return command


def watch_audits(
    *, calibration_root: Path, config: UnconditionalAuditConfig, poll_seconds: int = 60,
    origin: str | None = None, backend: str = "cpu",
) -> None:
    """Audit ready origins on CPU by default; support a single requested date."""
    calibration_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(calibration_root / "unconditional_audits.lock"), timeout=0):
        while True:
            check_resource_headroom()
            ready = sorted((calibration_root / "rolling").glob("*/report.json"))
            if origin is not None:
                ready = [path for path in ready if path.parent.name == origin]
            pending = [path.parent.name for path in ready if not (path.parent / "unconditional_q/report.json").exists()]
            if pending:
                next_origin = pending[0]
                log_path = calibration_root / "rolling" / next_origin / "unconditional_audit.log"
                failure_path = calibration_root / "rolling" / next_origin / "unconditional_q/failure.json"
                if failure_path.exists():
                    raise RuntimeError(f"audit worker failed; inspect {failure_path} before restarting")
                command = audit_worker_command(calibration_root=calibration_root, origin=next_origin, config=config, backend=backend)
                print(json.dumps(dict(event="launch_unconditional_audit", origin=next_origin, paths=config.path_count, backend=backend)), flush=True)
                with log_path.open("a", buffering=1) as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    print(json.dumps(dict(event="audit_wait_or_failure", origin=next_origin, returncode=result.returncode, log=str(log_path))), flush=True)
                    time.sleep(poll_seconds)
                    continue
            reports = []
            for report_path in sorted((calibration_root / "rolling").glob("*/unconditional_q/report.json")):
                if origin is None or report_path.parent.parent.name == origin:
                    reports.append(json.loads(report_path.read_text()))
            write_json(calibration_root / "unconditional_audit_summary.json", {
                "model_label": "Gromm v2 4,3", "completed_dates": len(reports), "ready_rolling_dates": len(ready),
                "conditional_tests_run": False, "requested_origin": origin, "backend": backend, "dates": reports,
            })
            if origin is not None and reports:
                return
            if (calibration_root / "rolling_complete.json").exists() and not pending:
                return
            print(json.dumps(dict(event="waiting_for_rolling_estimates", completed_audits=len(reports), ready_rolling_dates=len(ready))), flush=True)
            if not pending:
                time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--origin")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--standard-error-floor", type=float, default=1.0e-8)
    parser.add_argument("--density-loading-floor", type=float, default=1.0e-4)
    parser.add_argument("--nms", type=float, nargs="+", default=UnconditionalAuditConfig().nms)
    arguments = parser.parse_args()
    jax.config.update("jax_platforms", arguments.backend)
    config = UnconditionalAuditConfig(
        path_count=arguments.paths, batch_size=arguments.batch_size, substeps=arguments.substeps,
        seed=arguments.seed, standard_error_floor=arguments.standard_error_floor,
        density_loading_floor=arguments.density_loading_floor, nms=tuple(arguments.nms),
    )
    if arguments.watch:
        watch_audits(calibration_root=arguments.calibration_root, config=config, origin=arguments.origin, backend=arguments.backend)
    elif arguments.origin:
        try:
            run_audit(calibration_root=arguments.calibration_root, origin=arguments.origin, config=config)
        except (InterruptedError, ResourceHeadroomError):
            raise
        except Exception as error:
            write_json(arguments.calibration_root / "rolling" / arguments.origin / "unconditional_q/failure.json", {
                "origin": arguments.origin, "error_type": type(error).__name__, "message": str(error),
            })
            raise
    else:
        parser.error("provide --origin or --watch")


if __name__ == "__main__":
    main()
