"""Isolated wider-grid GROMM v1 calibration, with downweighted full-range quotes.

Reuse the paper's SANOS decoder and exact full-gradient objective. Never mutate
the retained paper dataset/checkpoint or reuse parameters on a different grid.
Preparation is cache-only; fitting supports exact Adam/schedule restart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.util import is_filename
from cdxcore.version import version
from filelock import FileLock

from quantdata.sdlv.daily import DATASET_PATH
from quantdata.sdlv.daily import train as daily


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class CalibrationConfig:
    """Grid, full-range quote weighting, and model identity (not GROMM2)."""

    label: str = "Gromm-v1-4,3"
    grid_nms: tuple[float, float] = (-4.0, 3.0)
    core_nms: tuple[float, float] = (-3.5, 2.5)
    tail_weight: float = 0.1
    num_inner_strikes: int = 200

    def __post_init__(self) -> None:
        verify_inp(is_filename(self.label), lambda: f"Invalid model label: {self.label}")
        for bounds in (self.grid_nms, self.core_nms):
            verify_inp(len(bounds) == 2 and np.all(np.isfinite(bounds)) and bounds[0] < 0 < bounds[1],
                       lambda: f"Invalid NMS bounds: {bounds}")
        verify_inp(0 < self.tail_weight <= 1, lambda: "Tail weight must lie in (0,1]")
        verify_inp(self.num_inner_strikes >= 6 and self.num_inner_strikes % 2 == 0,
                   lambda: "An even number of at least six inner nodes is required")


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    """Paper's 300-epoch initial plus 700-epoch continuation schedule."""

    initial_epochs: int = 300
    continuation_epochs: int = 700
    initial_learning_rate: float = 0.02
    continuation_learning_rate: float = 0.005
    minimum_learning_rate_fraction: float = 0.05
    continuation_minimum_learning_rate_fraction: float = 0.01
    continuation_gradient_tolerance: float = 1e-5
    checkpoint_every: int = 10
    cap_penalty_weight: float = 1.0

    def __post_init__(self) -> None:
        verify_inp(self.initial_epochs > 0 and self.continuation_epochs >= 0 and self.checkpoint_every > 0,
                   lambda: "Invalid epoch/checkpoint counts")
        verify_inp(self.initial_learning_rate > 0 and self.continuation_learning_rate > 0
                   and 0 <= self.minimum_learning_rate_fraction <= 1
                   and 0 <= self.continuation_minimum_learning_rate_fraction <= 1
                   and self.continuation_gradient_tolerance > 0 and self.cap_penalty_weight >= 0,
                   lambda: "Invalid training parameters")

    @property
    def epochs(self) -> int:
        return self.initial_epochs + self.continuation_epochs

    def learning_rate(self, epoch: int) -> float:
        continuation = epoch >= self.initial_epochs
        offset = epoch - self.initial_epochs if continuation else epoch
        count = self.continuation_epochs if continuation else self.initial_epochs
        start = self.continuation_learning_rate if continuation else self.initial_learning_rate
        minimum = self.continuation_minimum_learning_rate_fraction if continuation else self.minimum_learning_rate_fraction
        return start * (minimum + (1 - minimum)
                        * 0.5 * (1 + np.cos(np.pi * offset / max(count - 1, 1))))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def emit(**record: object) -> None:
    print(json.dumps(record, allow_nan=False), flush=True)


def prepare(run_directory: Path, config: CalibrationConfig) -> None:
    """Prepare all paper dates, preserve original ATM scales, and a stress pilot."""
    run_directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(run_directory / "prepare.lock"), timeout=0):
        output = run_directory / "market.npz"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        with np.load(DATASET_PATH, allow_pickle=False) as old:
            dates = old["dates"].astype(str)
            original_initial = np.asarray(old["raw_initial"])
        dates_path = run_directory / "market_dates.json"
        daily._atomic_write_json({"market_dates": dates.tolist()}, dates_path)
        daily.prepare_dataset(
            market_dates_report=dates_path, output=output, ticker="SPX",
            num_inner_strikes=config.num_inner_strikes, quote_chunk_size=128, workers=4,
            grid_nms_range=config.grid_nms, core_nms_range=config.core_nms,
            tail_weight=config.tail_weight, cache_only=True,
        )
        with np.load(output, allow_pickle=False) as archive:
            payload = {key: np.asarray(archive[key]) for key in archive.files}
        # This controls data provenance: only the geometry and quote scope change.
        np.testing.assert_allclose(payload["raw_initial"], original_initial, rtol=0, atol=2e-6)
        pilot_dates = ("2022-01-03", "2022-06-13", "2025-04-08", "2025-08-29")
        indices = np.asarray([int(np.flatnonzero(dates == day)[0]) for day in pilot_dates])
        pilot = {key: value[indices] if value.ndim and value.shape[0] == len(dates) else value
                 for key, value in payload.items()}
        daily._atomic_write_npz(pilot, run_directory / "pilot-market.npz")
        report = dict(model="GROMM v1 / SDLV", config=asdict(config),
                      model_version=CalibrationConfig.version.unique_id48,
                      original_dataset=str(DATASET_PATH), original_dataset_sha256=sha256(DATASET_PATH),
                      dataset=str(output), dataset_sha256=sha256(output),
                      pilot_dates=list(pilot_dates), atm_initializer_matches_paper=True,
                      quote_counts=int(payload["mask"].sum()),
                      core_quotes=int(np.sum(payload["mask"] & (payload["quote_weights"] == 1))),
                      tail_quotes=int(np.sum(payload["mask"] & (payload["quote_weights"] < 1))),
                      conventions=["100 one-day plus 100 25-day nodes; 25 daily expiries; 5% Black smoothing",
                                   "All otherwise eligible cached European books within 25 business days; 0.2-day expiry excluded",
                                   "NMS for quote weights uses each quote expiry's canonical ATM curve, not the 25-day scale",
                                   "Equal-date mean of weighted quote losses divided by each date's sum of weights",
                                   "Contemporaneous individual-date surface fits, not out-of-sample forecasts or a Q-dynamics audit"])
        daily._atomic_write_json(report, run_directory / "manifest.json")
        emit(stage="prepared", **report)


class ProgressMonitor:
    """Flushed heartbeat even during compilation; resource checks every minute."""

    def __init__(self, total: int) -> None:
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.done = threading.Event()
        self.record: dict[str, object] = dict(stage="initializing", completed=0, total=total)
        self.reason: str | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.done.wait(60):
            elapsed = time.monotonic() - self.started
            completed = int(self.record["completed"])
            total = int(self.record["total"])
            emit(event="heartbeat", **self.record, elapsed_seconds=elapsed,
                 estimated_remaining_seconds=elapsed * (total / completed - 1) if completed else None)
            try:
                result = subprocess.run(["gpuctl", "resources", "--json"], capture_output=True, text=True, timeout=15, check=True)
                checks = json.loads(result.stdout)["checks"]
                unhealthy = [(check["name"], check["health"]) for check in checks if check["health"] != "OK"]
                if unhealthy:
                    emit(event="resource_alert", checks=unhealthy)
                if any(health in {"STOP", "ERROR", "UNAVAILABLE"} for _, health in unhealthy):
                    self.reason = f"Resource health: {unhealthy}"
                    self.stop.set()
                    if not self.done.wait(60):
                        emit(event="resource_forced_exit", message="Graceful-stop deadline exceeded; use prior periodic checkpoint")
                        os.kill(os.getpid(), signal.SIGKILL)
                    return
            except Exception as error:
                emit(event="resource_warning", message=str(error))


def density_diagnostics(supports: np.ndarray, densities: np.ndarray) -> dict[str, float]:
    """Numerical calibration stability, not a dynamic no-arbitrage test."""
    if not np.all(np.isfinite(densities)):
        raise FloatingPointError("Non-finite density")
    return dict(minimum_probability=float(densities.min()),
                maximum_mass_error=float(np.max(np.abs(densities.sum(axis=-1) - 1))),
                maximum_forward_error=float(np.max(np.abs(np.einsum("dti,di->dt", densities, supports) - 1))),
                maximum_boundary_mass=float(np.max(densities[..., 0] + densities[..., -1])))


def _validate_resume_identity(saved: dict, current: dict, *, resume_code_sha256: str | None = None) -> None:
    """Allow an explicitly identified runner repair, never a changed dataset/kernel."""
    if saved == current:
        return
    runner = str(Path(__file__))
    old_code = saved.get("code_sha256", {})
    if resume_code_sha256 is not None and old_code.get(runner) == resume_code_sha256:
        candidate = {**saved, "code_sha256": {**old_code, runner: current["code_sha256"][runner]}}
        if candidate == current:
            return
    raise ValueError("Resume identity differs: dataset, model, code or schedule changed")


def independent_density_validation(supports: np.ndarray, raw: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    """Reconstruct in float64 without changing the training precision or state."""
    import jax
    import jax.numpy as jnp

    from quantdata.sdlv.daily.front_spot.jax_model import decode_density_tower

    with jax.enable_x64():
        raw = np.asarray(raw, dtype=np.float64)
        dlvsq = daily._MIN_DLV_SQ + np.exp(np.minimum(raw, np.log(daily._MAX_DLV_SQ - daily._MIN_DLV_SQ)))
        tower = jax.jit(jax.vmap(lambda s, v: decode_density_tower(s, v, dt=1 / 252)))(
            jnp.asarray(supports, dtype=jnp.float64), jnp.asarray(dlvsq))
        tower = np.asarray(tower)
        if tower.dtype != np.float64:
            raise RuntimeError("Independent validation must actually execute in float64")
        diagnostics = density_diagnostics(np.asarray(supports, dtype=np.float64), tower)
    if diagnostics["minimum_probability"] < -1e-13 or max(diagnostics["maximum_mass_error"], diagnostics["maximum_forward_error"]) > 1e-10:
        raise FloatingPointError(f"Float64 reconstruction failed: {diagnostics}")
    return diagnostics, dlvsq


def checkpoint_density_validation(
    supports: np.ndarray, raw: np.ndarray, densities: np.ndarray, initial: np.ndarray,
) -> tuple[dict[str, float], dict[str, float] | None]:
    """Distinguish bounded float32 initial-point-mass roundoff from invalid densities.

The paper's finite-difference initializer can leave a few float32 ulps of
negative boundary mass. A positive, mass-preserving SANOS transition cannot
increase its total negative mass. Only this bounded case may be accepted,
and only after independent float64 positivity/mass/forward validation.
    """
    diagnostics = density_diagnostics(supports, densities)
    if max(diagnostics["maximum_mass_error"], diagnostics["maximum_forward_error"]) > 1e-3:
        raise FloatingPointError(f"Density invariant failure: {diagnostics}")
    if diagnostics["minimum_probability"] >= -1e-7:
        return diagnostics, None
    epsilon = np.finfo(np.float32).eps
    initial_negative_mass = np.maximum(-initial, 0).sum(axis=-1)
    negative_mass = np.maximum(-densities, 0).sum(axis=-1)
    if (not np.all(np.isfinite(initial)) or np.any(initial_negative_mass > 8 * epsilon)
            or np.any(negative_mass > initial_negative_mass[:, None] + epsilon)):
        raise FloatingPointError(f"Negative mass exceeds initializer roundoff: {diagnostics}")
    precise, _ = independent_density_validation(supports, raw)
    return diagnostics, precise


def fit(dataset: Path, output: Path, config: TrainConfig, *, resume_code_sha256: str | None = None) -> bool:
    """Train deterministically with restartable Adam, finite checks and diagnostics."""
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / "training.lock"), timeout=0):
        return _fit_locked(dataset, output, config, resume_code_sha256=resume_code_sha256)


def _fit_locked(dataset: Path, output: Path, config: TrainConfig, *, resume_code_sha256: str | None = None) -> bool:
    import jax
    import jax.numpy as jnp

    if jax.default_backend() not in {"cuda", "gpu"}:
        raise RuntimeError("Launch this training command through gpuctl with a CUDA allocation")
    data_cpu, metadata = daily._load_data(dataset)
    if metadata["schema"] != "daily25-jax-market-weighted-v1":
        raise ValueError("This runner requires a separately prepared weighted dataset")
    model_label = CalibrationConfig().label
    checkpoint_path = output / f"{model_label}.npz"
    identity = dict(model_label=model_label, dataset_sha256=sha256(dataset), train_config=asdict(config),
                    model_version=CalibrationConfig.version.unique_id48,
                    code_sha256={str(path): sha256(path) for path in (Path(__file__), Path(daily.__file__))})
    identity_json = json.dumps(identity, sort_keys=True)
    resume_path = output / "training-state.npz"
    history_path = output / "history.json"
    data = jax.device_put({key: jnp.asarray(value) for key, value in data_cpu.items()})
    params = {"raw": jnp.asarray(data_cpu["raw_initial"])}
    first = {"raw": jnp.zeros_like(params["raw"])}
    second = {"raw": jnp.zeros_like(params["raw"])}
    step, start_epoch, history = jnp.asarray(0, dtype=jnp.int32), 0, []
    resume_provenance = None
    if resume_path.exists():
        with np.load(resume_path, allow_pickle=False) as saved:
            saved_identity = json.loads(str(saved["identity"].item()))
            _validate_resume_identity(saved_identity, identity, resume_code_sha256=resume_code_sha256)
            params = {"raw": jnp.asarray(saved["raw"])}
            first, second = {"raw": jnp.asarray(saved["first"])}, {"raw": jnp.asarray(saved["second"])}
            step, start_epoch = jnp.asarray(saved["step"]), int(saved["epoch"])
            history = json.loads(str(saved["history"].item()))
        resume_provenance = dict(checkpoint_sha256=sha256(resume_path), epoch=start_epoch,
                                 original_identity=saved_identity, runner_repair_accepted=saved_identity != identity)
        emit(event="resume", **resume_provenance)
    functions = daily._make_loss_functions(data, mode="direct", cap_penalty_weight=config.cap_penalty_weight,
                                           quote_chunk_size=int(metadata["quote_chunk_size"]))
    block_grad, adam, decode, predict, blocks = functions
    block_count = next(iter(blocks.values())).shape[0]
    monitor = ProgressMonitor(config.epochs)
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in old_handlers:
        signal.signal(sig, lambda *_: monitor.stop.set())
    monitor.thread.start()
    completed = start_epoch
    started = time.monotonic()

    def save() -> None:
        raw, m, v = (np.asarray(jax.device_get(x["raw"])) for x in (params, first, second))
        if not all(np.all(np.isfinite(x)) for x in (raw, m, v)):
            raise FloatingPointError("Refusing to overwrite a checkpoint with non-finite optimizer state")
        daily._atomic_write_npz(dict(identity=np.asarray(identity_json), epoch=np.asarray(completed),
                                     raw=raw, first=m, second=v, step=np.asarray(step),
                                     history=np.asarray(json.dumps(history))), resume_path)
        daily._atomic_write_npz(dict(schema=np.asarray("daily25-jax-weighted-fit-v1"),
                                     mode=np.asarray("direct"), rank=np.asarray(0), raw=raw,
                                     model_label=np.asarray(model_label),
                                     dataset_sha256=np.asarray(identity["dataset_sha256"])), checkpoint_path)
        daily._atomic_write_json({"history": history}, history_path)

    emit(event="training_start", dates=len(metadata["dates"]), quotes=int(np.sum(metadata["quote_counts"])),
         quote_blocks=block_count, resume_epoch=start_epoch, devices=[str(device) for device in jax.devices()], **identity)
    try:
        for epoch in range(start_epoch, config.epochs):
            if monitor.stop.is_set():
                break
            # Match the paper's continuation: reset Adam but retain the fitted state.
            if epoch == config.initial_epochs:
                first, second = jax.tree.map(jnp.zeros_like, params), jax.tree.map(jnp.zeros_like, params)
                step = jnp.asarray(0, dtype=jnp.int32)
            gradients = jax.tree.map(jnp.zeros_like, params)
            objective = jnp.asarray(0, dtype=jnp.float32)
            for index in range(block_count):
                monitor.record.update(stage="gradient", completed=completed, epoch=epoch + 1, block=index + 1, blocks=block_count)
                (value, _), contribution = block_grad(params, {name: values[index] for name, values in blocks.items()})
                gradients = jax.block_until_ready(jax.tree.map(lambda a, b: a + b, gradients, contribution))
                objective += value
                if monitor.stop.is_set():
                    break
            if monitor.stop.is_set():
                break  # Never apply a partially accumulated gradient.
            if not np.isfinite(float(objective)) or not bool(jnp.all(jnp.isfinite(gradients["raw"]))):
                raise FloatingPointError(f"Non-finite loss/gradient before epoch {epoch + 1} update")
            rate = config.learning_rate(epoch)
            params, first, second, step, norm = adam(params, first, second, step, gradients, jnp.asarray(rate, dtype=jnp.float32))
            if not bool(jnp.all(jnp.isfinite(params["raw"]))):
                raise FloatingPointError("Non-finite parameter update")
            completed = epoch + 1
            row = dict(epoch=completed, objective=float(objective), gradient_l2_norm=float(norm), learning_rate=rate)
            history.append(row)
            reached_tolerance = epoch >= config.initial_epochs and float(norm) <= config.continuation_gradient_tolerance
            if completed == 1 or completed % config.checkpoint_every == 0 or completed == config.epochs or reached_tolerance:
                densities, initial = decode(params)
                diagnostics, precise = checkpoint_density_validation(
                    data_cpu["supports"], np.asarray(params["raw"]), np.asarray(densities), np.asarray(initial))
                if precise is not None:
                    emit(event="float32_initializer_roundoff_verified", epoch=completed,
                         float32=diagnostics, independent_float64=precise)
                save()
                elapsed = time.monotonic() - started
                emit(event="checkpoint", **row, **diagnostics, elapsed_seconds=elapsed,
                     estimated_remaining_seconds=elapsed * (config.epochs - completed) / (completed - start_epoch))
            if reached_tolerance:
                break
        if monitor.stop.is_set():
            save()
            emit(event="stopped", completed=completed, reason=monitor.reason, checkpoint=str(resume_path))
            return False
        monitor.record.update(stage="final_diagnostics", completed=completed)
        densities, density0 = decode(params)
        predictions = np.concatenate([np.asarray(predict(densities, density0, {k: v[i] for k, v in blocks.items()}))
                                      for i in range(block_count)], axis=1)
        if not np.all(np.isfinite(predictions)):
            raise FloatingPointError("Non-finite quote predictions")
        nms = data_cpu["quote_nms"]
        groups = daily._group_statistics(predictions, data_cpu, labels=("core", "tail", "paper_nms_region"),
                                         selections=(data_cpu["quote_weights"] == 1, data_cpu["quote_weights"] < 1,
                                                     (nms >= -2.5) & (nms <= 1.5)))
        reference_date = daily._group_statistics(predictions, data_cpu, labels=("2025-08-29",),
            selections=(np.asarray(metadata["dates"]).astype(str)[:, None] == "2025-08-29",))
        # Independently reconstruct in float64; training remains the paper's float32.
        stable64, dlvsq = independent_density_validation(data_cpu["supports"], np.asarray(params["raw"]))
        report = dict(status="complete", **identity, completed_epochs=completed,
                      checkpoint=str(checkpoint_path.resolve()),
                      elapsed_seconds=time.monotonic() - started, quotes=int(np.sum(metadata["quote_counts"])),
                      dates=len(metadata["dates"]), overall=daily._statistics(predictions, data_cpu),
                      fit_by_weight_group=groups, reference_date=reference_date,
                      stability_float32=density_diagnostics(data_cpu["supports"], np.asarray(densities)),
                      stability_float64=stable64, maximum_dlv_squared=float(dlvsq.max()),
                      resumed_from=resume_provenance,
                      fraction_at_dlv_cap=float(np.mean(dlvsq >= daily._MAX_DLV_SQ * (1 - 1e-7))),
                      dynamic_q_audit="not run; these checks establish calibration numerics only")
        daily._atomic_write_json(report, output / "report.json")
        emit(event="complete", **report)
        return True
    finally:
        monitor.done.set()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "pilot", "fit"))
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--resume-code-sha256", help="Explicitly accept this prior runner hash for a reviewed code-only repair")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.run_directory, CalibrationConfig())
    elif args.command == "pilot":
        fit(args.run_directory / "pilot-market.npz", args.run_directory / "pilot",
            TrainConfig(initial_epochs=100, continuation_epochs=0), resume_code_sha256=args.resume_code_sha256)
    else:
        pilot = json.loads((args.run_directory / "pilot/report.json").read_text())
        if pilot["status"] != "complete":
            raise RuntimeError("The pilot must complete before the full historical fit")
        fit(args.run_directory / "market.npz", args.run_directory / "fit", TrainConfig(), resume_code_sha256=args.resume_code_sha256)


if __name__ == "__main__":
    main()
