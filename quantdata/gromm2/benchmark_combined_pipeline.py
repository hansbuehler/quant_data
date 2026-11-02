"""Compare mapped and host-resident combined-book inputs on an isolated GPU model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import os
import resource
import signal
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from quantdata.gromm2.combined_hedging import (
    CombinedBookConfig,
    CombinedBookPolicy,
    load_combined_scenarios,
)
from quantdata.gromm2.gpu_forward_queue import resource_report
from quantdata.gromm2.hedging_policy import (
    ScenarioData,
    _Progress,
    _write_json,
    training_environment,
)
from quantdata.gromm2.joint_hedging import JointTrainConfig
from quantdata.gromm2.market_batching import (
    MarketBatchIterator,
    MarketHostCache,
    cache_market_training_arrays,
)
from quantdata.gromm2.market_training_runtime import IndependentPortfolioStep
from quantdata.gromm2.portfolio_hedging import LIABILITIES, IndependentHedges
from quantdata.gromm2.rolling_audit import check_resource_headroom
from quantdata.torch.ffw import FeedForwardConfig


def summarize_timings(seconds: list[float]) -> dict[str, float]:
    """Summarize wall-clock timings without excluding slower observations."""
    if not seconds or not all(np.isfinite(value) and value >= 0 for value in seconds):
        raise ValueError("timings must be a nonempty finite nonnegative sequence")
    values = np.asarray(seconds) * 1000
    return dict(mean_ms=float(values.mean()), median_ms=float(np.median(values)),
        p90_ms=float(np.quantile(values, 0.9)), minimum_ms=float(values.min()), maximum_ms=float(values.max()))


def process_io() -> dict[str, int]:
    """Read this benchmark process's I/O counters, not system capacity."""
    counters = {}
    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            name, value = line.split(":", 1)
            counters[name] = int(value.strip())
    except OSError:
        pass
    usage = resource.getrusage(resource.RUSAGE_SELF)
    counters.update(major_faults=usage.ru_majflt, minor_faults=usage.ru_minflt)
    return counters


def checkpoint_configuration(saved: dict) -> CombinedBookConfig:
    """Reconstruct the combined model geometry without selecting best weights."""
    configuration = copy.deepcopy(saved["identity"]["config"])
    configuration["deep"] = FeedForwardConfig(**configuration["deep"])
    return CombinedBookConfig(**configuration)


def pipeline_recommendation(trials: list[dict]) -> dict:
    """Compare resident trials with unpressured mapped controls, not ABBA alone."""
    selected = {kind: [trial for trial in trials if trial["phase"] == kind]
        for kind in ("mapped-with-cache", "resident")}
    rates = {kind: (sum(trial["paths"] for trial in records) / sum(trial["elapsed_seconds"] for trial in records)
        if len(records) == 2 and all(trial["cache_active_throughout"] for trial in records) else None)
        for kind, records in selected.items()}
    controls = [trial for trial in trials if trial["phase"] in {"mapped-before-cache", "mapped-after-release"}]
    mapped = (sum(trial["paths"] for trial in controls) / sum(trial["elapsed_seconds"] for trial in controls)
        if len(controls) == 2 and {trial["phase"] for trial in controls}
            == {"mapped-before-cache", "mapped-after-release"} else None)
    resident = rates["resident"]
    ratio = resident / mapped if mapped and resident else None
    return dict(mapped_paths_per_second=mapped, resident_paths_per_second=resident,
        mapped_with_cache_paths_per_second=rates["mapped-with-cache"],
        resident_to_mapped_ratio=ratio, recommendations=dict(host_cache_training=ratio is not None and ratio >= 1.05,
            basis="Require at least 5% pooled throughput improvement over both mapped controls without the anonymous cache; memory-pressured ABBA comparisons alone cannot recommend caching.",
            limitation="Advisory bounded warmed-path microbenchmark, not full-epoch evidence; compare completed production epochs including population evaluation before claiming a speedup."))


def measure_trial(data: ScenarioData, *, phase: str, order: np.ndarray, batch_size: int,
                  step: IndependentPortfolioStep, progress: _Progress,
                  host_cache: MarketHostCache | None) -> dict:
    """Include iterator setup, input waits and final device drain in total throughput."""
    check_resource_headroom()
    if progress.stop.is_set():
        raise InterruptedError("isolated pipeline benchmark stopped before a trial")
    progress.state.update(stage=phase, trial_completed=0, trial_total=len(order))
    before_io = process_io()
    before_captures, before_capture_seconds = step.capture_count, step.capture_seconds
    waits, updates = [], []
    cache_active = host_cache is not None and host_cache.bytes_cached > 0
    torch.cuda.synchronize(step.device)
    started = time.monotonic()
    with MarketBatchIterator(data, env=step.model.env, batch_size=batch_size, order=order, prefetch_workers=2) as iterator:
        for batch_number in range(len(order) // batch_size):
            waiting = time.monotonic()
            inputs = next(iterator)
            updating = time.monotonic()
            waits.append(updating - waiting)
            step(*inputs, tuple(LIABILITIES))
            updates.append(time.monotonic() - updating)
            progress.state["completed"] += len(inputs[0])
            progress.state["trial_completed"] = (batch_number + 1) * batch_size
            if progress.stop.is_set():
                raise InterruptedError("isolated pipeline benchmark stopped after a complete update")
            if phase in {"resident", "mapped-with-cache"} and (host_cache is None or not host_cache.bytes_cached):
                cache_active = False
    draining = time.monotonic()
    torch.cuda.synchronize(step.device)
    finished = time.monotonic()
    after_io = process_io()
    result = dict(phase=phase, paths=len(order), elapsed_seconds=finished - started,
        paths_per_second=len(order) / (finished - started), input_wait=summarize_timings(waits),
        update=summarize_timings(updates), input_wait_seconds=sum(waits), update_seconds=sum(updates),
        final_device_drain_seconds=finished - draining,
        setup_and_close_seconds=draining - started - sum(waits) - sum(updates),
        input_wait_ms=[value * 1000 for value in waits], update_ms=[value * 1000 for value in updates],
        capture_count_before=before_captures, capture_count_after=step.capture_count,
        capture_seconds_during_trial=step.capture_seconds - before_capture_seconds,
        process_io_delta={name: after_io[name] - before_io[name] for name in before_io.keys() & after_io.keys()},
        cache_active_throughout=cache_active and host_cache is not None and host_cache.bytes_cached > 0)
    progress.emit(trial={key: value for key, value in result.items() if key not in {"input_wait_ms", "update_ms"}})
    return result


def benchmark(*, checkpoint: Path, run_directory: Path, output_directory: Path,
              batches: int = 48, resident_steps: int = 8) -> dict:
    """Benchmark copies only; all generated artifacts stay in a unique external run."""
    if not isinstance(batches, int) or not 8 <= batches <= 256:
        raise ValueError("batches must lie between 8 and 256")
    if not isinstance(resident_steps, int) or not 3 <= resident_steps <= 32:
        raise ValueError("resident_steps must lie between 3 and 32")
    root = os.environ.get("QUANTDATA_ROOT")
    output_directory = output_directory.resolve()
    if not root or not output_directory.is_relative_to(Path(root).resolve()):
        raise ValueError("benchmark output must be below the explicit external QUANTDATA_ROOT")
    source_repository = Path(__file__).resolve().parents[2]
    if output_directory.is_relative_to(source_repository):
        raise ValueError("benchmark output must not be inside a Git working tree")
    check_resource_headroom()
    output_directory.mkdir(parents=True, exist_ok=False)
    serialized = checkpoint.read_bytes()
    digest = hashlib.sha256(serialized).hexdigest()
    saved = torch.load(io.BytesIO(serialized), map_location="cpu", weights_only=False)
    configuration = checkpoint_configuration(saved)
    train_config = replace(JointTrainConfig(**saved["report"]["train_config"]), device="cuda", compile=True,
        host_cache_training=False)
    torch.set_num_threads(train_config.cpu_threads)
    progress = _Progress(label="isolated-combined-pipeline-benchmark", interval=60.0)
    progress.state.update(stage="initializing-allocated-gpu", total=(6 * batches + resident_steps + 3) * train_config.batch_size)
    host_cache = None
    report = dict(complete=False, checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest,
        checkpoint_epoch=saved["state"]["epoch"], checkpoint_next_batch=saved["state"]["next_batch"],
        production_checkpoint_written=False, production_paths_written=False,
        scientific_outcomes=False, optimizer_restored_between_trials=False,
        benchmark_model="Isolated copy of current training parameters and Adam, not validation-selected best parameters.",
        batch_size=train_config.batch_size, batches=batches, prefetch_workers=2, cpu_threads=train_config.cpu_threads,
        precision=train_config.precision, trials=[], initial_resources=resource_report(),
        timing_semantics="Wall-clock input wait includes producer readiness; update includes pending H2D waits, computation and mandatory finite-status read; total includes setup, close and final CUDA drain.",
        limitations=["Same bounded random paths are reused; later mapped trials may benefit from file-cache warming.",
            "Mapped ABBA trials retain the full anonymous host cache; before/after baselines expose this memory-pressure bias.",
            "Only microbenchmark throughput is measured; no full-epoch speedup or convergence claim is made."],
        recommendations=dict(host_cache_training=False))
    _write_json(output_directory / "report.json", report)
    source_path = run_directory / "train.h5"
    source_stat = source_path.stat()
    handlers = {kind: signal.signal(kind, lambda *_: progress.stop.set()) for kind in (signal.SIGINT, signal.SIGTERM)}
    progress.thread.start()
    try:
        env = training_environment(train_config)
        normalization = tuple(saved["model"][f"policies.zero.normalization_{name}"].numpy() for name in ("mean", "scale"))
        model = IndependentHedges(env, config=configuration, normalization=normalization, policy_class=CombinedBookPolicy)
        model.load_state_dict(saved["model"])
        optimizers = {name: torch.optim.Adam(policy.parameters(), lr=train_config.learning_rate)
            for name, policy in model.policies.items()}
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(saved["optimizers"][name])
        torch.set_rng_state(saved["torch_rng"])
        if saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device=env.device)
        step = IndependentPortfolioStep(model, optimizers, gradient_clip=train_config.gradient_clip,
            compile=True, use_cuda_graphs=True, captured_batch_size=train_config.batch_size)
        progress.state.update(stage="mapping-unchanged-training-source")
        mapped, _ = load_combined_scenarios(source_path, split="train", case="black0")
        resident, _ = load_combined_scenarios(source_path, split="train", case="black0")
        if mapped.sha256 != saved["identity"]["training_scenarios_sha256"] or resident.sha256 != mapped.sha256:
            raise ValueError("benchmark scenarios do not match the production checkpoint")
        count = batches * train_config.batch_size
        if count > len(mapped.payoff):
            raise ValueError("benchmark needs more distinct paths than the saved sample contains")
        order = np.random.default_rng(20261101).choice(len(mapped.payoff), count, replace=False)
        report.update(order_sha256=hashlib.sha256(order.tobytes()).hexdigest(), path_selection_seed=20261101,
            source=dict(path=str(source_path.resolve()), sha256=mapped.sha256, bytes=source_stat.st_size,
                mtime_ns=source_stat.st_mtime_ns, paths=len(mapped.payoff),
                arrays={name: dict(shape=source.shape, dtype=str(source.dtype), bytes=source.nbytes)
                    for name, source in mapped.features.arrays.items()}))
        progress.state.update(stage="warming-full-batch-graph")
        with MarketBatchIterator(mapped, env=env, batch_size=train_config.batch_size,
                order=order[:train_config.batch_size], prefetch_workers=2) as iterator:
            device_inputs = next(iterator)
        for _ in range(3):
            step(*device_inputs, tuple(LIABILITIES))
            progress.state["completed"] += train_config.batch_size
            if progress.stop.is_set():
                raise InterruptedError("benchmark stopped during full-graph warmup")
        torch.cuda.synchronize(env.device)
        progress.state.update(stage="resident-device-update-baseline")
        durations = []
        for _ in range(resident_steps):
            started = time.monotonic()
            step(*device_inputs, tuple(LIABILITIES))
            torch.cuda.synchronize(env.device)
            durations.append(time.monotonic() - started)
            progress.state["completed"] += train_config.batch_size
            if progress.stop.is_set():
                raise InterruptedError("benchmark stopped during resident-device baseline")
        del device_inputs
        report["resident_device"] = dict(update=summarize_timings(durations), updates=resident_steps,
            paths_per_second=resident_steps * train_config.batch_size / sum(durations),
            capture_count=step.capture_count, capture_seconds=step.capture_seconds)

        def trial(data: ScenarioData, phase: str) -> None:
            report["trials"].append(measure_trial(data, phase=phase, order=order,
                batch_size=train_config.batch_size, step=step, progress=progress, host_cache=host_cache))
            _write_json(output_directory / "report.json", report)

        trial(mapped, "mapped-before-cache")
        progress.state.update(stage="building-exact-raw-training-host-cache")
        caching_started = time.monotonic()
        host_cache = cache_market_training_arrays(resident, stop_requested=progress.stop.is_set,
            allow_partial=True)
        report["host_cache_build_seconds"] = time.monotonic() - caching_started
        report["host_cache_bytes"] = host_cache.bytes_cached if host_cache is not None else 0
        report["host_cache_array_names"] = list(host_cache.array_names) if host_cache is not None else []
        if host_cache is not None:
            progress.resource_cleanup = host_cache.check_headroom
            progress.resource_release = host_cache.release
            for data, phase in ((mapped, "mapped-with-cache"), (resident, "resident"),
                                (resident, "resident"), (mapped, "mapped-with-cache")):
                trial(data, phase)
            progress.resource_cleanup = progress.resource_release = None
            host_cache.release()
        trial(mapped, "mapped-after-release")
        report.update(pipeline_recommendation(report["trials"]))
        report.update(capture_count=step.capture_count, capture_seconds=step.capture_seconds,
            tail_step_count=step.tail_step_count, final_resources=resource_report())
        if report["final_resources"]["blocked"]:
            raise RuntimeError("resource manager blocked work at benchmark completion")
        final_stat = source_path.stat()
        report["source_metadata_unchanged"] = (final_stat.st_size, final_stat.st_mtime_ns, final_stat.st_ino) == (
            source_stat.st_size, source_stat.st_mtime_ns, source_stat.st_ino)
        report["production_checkpoint_sha256_after"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if not report["source_metadata_unchanged"] or report["production_checkpoint_sha256_after"] != digest:
            raise RuntimeError("production source changed during isolated benchmark; results not accepted")
        report["complete"] = True
        _write_json(output_directory / "report.json", report)
        progress.emit(result_summary={key: report[key] for key in ("mapped_paths_per_second", "resident_paths_per_second", "recommendations")})
        return report
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error), complete=False,
            recommendations=dict(host_cache_training=False))
        _write_json(output_directory / "report.json", report)
        raise
    finally:
        progress.resource_cleanup = progress.resource_release = None
        if host_cache is not None:
            host_cache.release()
        progress.closed.set()
        progress.thread.join(timeout=2.0)
        for kind, handler in handlers.items():
            signal.signal(kind, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=48)
    parser.add_argument("--resident-steps", type=int, default=8)
    arguments = parser.parse_args()
    benchmark(**vars(arguments))


if __name__ == "__main__":
    main()
