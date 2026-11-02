"""Profile checkpoint copies without changing production data, weights or training order."""

from __future__ import annotations

import argparse
import copy
import json
import signal
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch

from quantdata.gromm2.hedging_policy import (
    HedgingNetworkConfig,
    _batch,
    _Progress,
    training_environment,
)
from quantdata.gromm2.joint_hedging import JointTrainConfig, weighted_oce
from quantdata.gromm2.market_hedging import (
    UnitTurnoverBookConfig,
    UnitTurnoverBookPolicy,
    load_market_scenarios,
)
from quantdata.gromm2.portfolio_hedging import LIABILITIES, IndependentHedges
from quantdata.gromm2.rolling_audit import check_resource_headroom


def summarize(values: list[float]) -> dict[str, float]:
    """Return milliseconds, retaining tails as well as central timing."""
    return {"mean_ms": float(np.mean(values)), "median_ms": float(np.median(values)),
            "min_ms": float(np.min(values)), "max_ms": float(np.max(values)),
            "p90_ms": float(np.quantile(values, 0.9))}


def run_benchmark(arguments: argparse.Namespace) -> None:
    """Run bounded measurements on an allocated GPU and save only audit artifacts."""
    check_resource_headroom()
    arguments.output.mkdir(parents=True, exist_ok=False)
    progress = _Progress(label="isolated-market-training-benchmark", interval=60.0)
    progress.state.update(stage="loading-checkpoint", total=arguments.iterations * 3 + arguments.warmup + 4)
    previous_signals = {kind: signal.signal(kind, lambda *_: progress.stop.set())
                        for kind in (signal.SIGINT, signal.SIGTERM)}
    progress.thread.start()
    try:
        saved = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
        configuration = copy.deepcopy(saved["identity"]["config"])
        configuration["deep"] = HedgingNetworkConfig(**configuration["deep"])
        config = UnitTurnoverBookConfig(**configuration)
        train_config = JointTrainConfig(**saved["report"]["train_config"])
        torch.set_num_threads(train_config.cpu_threads)
        env = training_environment(train_config)
        normalization = tuple(saved["model"][f"policies.zero.normalization_{name}"].numpy()
                              for name in ("mean", "scale"))
        model = IndependentHedges(env, config=config, normalization=normalization,
                                  policy_class=UnitTurnoverBookPolicy)
        optimizers = {name: torch.optim.Adam(policy.parameters(), lr=train_config.learning_rate)
                      for name, policy in model.policies.items()}
        forwards = {name: torch.compile(policy, mode=arguments.mode) if arguments.mode != "eager" else policy
                    for name, policy in model.policies.items()}

        def restore() -> None:
            model.load_state_dict(saved["model"])
            for name in LIABILITIES:
                optimizers[name].load_state_dict(copy.deepcopy(saved["optimizers"][name]))
            torch.set_rng_state(saved["torch_rng"])
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device=env.device)
            model.train()

        restore()
        data, _ = load_market_scenarios(arguments.run_directory / "train.h5", split="train",
                                         case="black0", kind="cohorts")
        order = saved["state"]["order"]
        if order is None:
            generator = np.random.default_rng()
            generator.bit_generator.state = saved["numpy_rng"]
            order = generator.permutation(len(data.payoff))
        batch_size = train_config.batch_size
        initial = int(saved["state"]["next_batch"])
        indices = [order[(initial + index * batch_size) % (len(order) - batch_size):
                         (initial + index * batch_size) % (len(order) - batch_size) + batch_size]
                   for index in range(arguments.iterations)]
        host = (data.features[indices[0]], data.prices[indices[0]], data.payoff[indices[0]])
        resident = tuple(torch.as_tensor(values, device=env.device,
                                        dtype=env.fdtype if number == 0 else torch.float64)
                         for number, values in enumerate(host))
        timings: dict[str, list[float]] = defaultdict(list)
        stage_timings: dict[str, list[float]] = defaultdict(list)

        @contextmanager
        def stage(name: str, *, synchronized: bool, annotated: bool):
            if synchronized:
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.profiler.record_function(name) if annotated else nullcontext():
                yield
            if synchronized:
                torch.cuda.synchronize()
                stage_timings[name].append((time.perf_counter() - started) * 1000)

        def step(batch: tuple[torch.Tensor, ...], *, synchronized: bool = False,
                 annotated: bool = False, diagnostics: dict | None = None) -> None:
            torch.compiler.cudagraph_mark_step_begin()
            features, prices, payoff = batch
            for name, policy in model.policies.items():
                if saved["state"]["policies"][name]["early_stopped"]:
                    continue
                with stage(f"{name}/zero_grad", synchronized=synchronized, annotated=annotated):
                    optimizers[name].zero_grad(set_to_none=True)
                with stage(f"{name}/forward", synchronized=synchronized, annotated=annotated):
                    output = forwards[name](features, prices)
                with stage(f"{name}/objective", synchronized=synchronized, annotated=annotated):
                    loss = -weighted_oce(output.trading_gain - LIABILITIES[name] * payoff,
                                         policy.translation, config.risk_aversion, torch.ones_like(payoff))
                    loss += config.penalty_weight * output.penalty.mean()
                with stage(f"{name}/loss_check", synchronized=synchronized, annotated=annotated):
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite {name} benchmark loss")
                if diagnostics is not None:
                    diagnostics[f"{name}/loss"] = loss.detach().cpu().clone()
                    for field, value in zip(output._fields, output):
                        diagnostics[f"{name}/output/{field}"] = value.detach().cpu().clone()
                with stage(f"{name}/backward", synchronized=synchronized, annotated=annotated):
                    loss.backward()
                if diagnostics is not None:
                    for parameter_name, parameter in policy.named_parameters():
                        if parameter.grad is not None:
                            diagnostics[f"{name}/gradient/{parameter_name}"] = parameter.grad.detach().cpu().clone()
                with stage(f"{name}/clip_and_check", synchronized=synchronized, annotated=annotated):
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), train_config.gradient_clip,
                                                   error_if_nonfinite=True)
                with stage(f"{name}/optimizer", synchronized=synchronized, annotated=annotated):
                    optimizers[name].step()
                if diagnostics is not None:
                    for parameter_name, parameter in policy.named_parameters():
                        diagnostics[f"{name}/parameter/{parameter_name}"] = parameter.detach().cpu().clone()
                    for parameter_number, parameter_state in optimizers[name].state_dict()["state"].items():
                        for key, value in parameter_state.items():
                            if isinstance(value, torch.Tensor):
                                diagnostics[f"{name}/optimizer/{parameter_number}/{key}"] = value.detach().cpu().clone()

        def completed(stage_name: str) -> None:
            progress.state.update(stage=stage_name, completed=int(progress.state["completed"]) + 1)
            if progress.stop.is_set():
                raise InterruptedError("isolated benchmark stopped; production checkpoint untouched")

        progress.state.update(stage="compile-and-warmup")
        warmup_started = time.perf_counter()
        for _ in range(arguments.warmup):
            step(resident)
            torch.cuda.synchronize()
            completed("compile-and-warmup")
        warmup_seconds = time.perf_counter() - warmup_started
        progress.emit(warmup_seconds=warmup_seconds)
        equivalence = None
        if arguments.verify_default:
            progress.state.update(stage="default-mode-equivalence")
            candidate_forwards = forwards
            forwards = {name: torch.compile(policy) for name, policy in model.policies.items()}
            restore()
            for _ in range(arguments.warmup):
                step(resident)
            restore()
            reference_diagnostics = {}
            step(resident, diagnostics=reference_diagnostics)
            forwards = candidate_forwards
            restore()
            candidate_diagnostics = {}
            step(resident, diagnostics=candidate_diagnostics)
            differences = {}
            for name, reference in reference_diagnostics.items():
                candidate = candidate_diagnostics[name]
                differences[name] = {"exact": bool(torch.equal(reference, candidate)),
                                     "close": bool(torch.allclose(reference, candidate, rtol=1e-6, atol=1e-7)),
                                     "max_abs": float((reference - candidate).abs().max())}
            equivalence = {"rtol": 1e-6, "atol": 1e-7,
                           "all_exact": all(record["exact"] for record in differences.values()),
                           "all_close": all(record["close"] for record in differences.values()),
                           "tensors_compared": len(differences), "differences": differences}
            (arguments.output / "equivalence.json").write_text(json.dumps(equivalence, indent=2) + "\n")
            progress.emit(equivalence={key: value for key, value in equivalence.items() if key != "differences"})
            if not equivalence["all_close"]:
                raise RuntimeError("candidate fails default-mode numerical equivalence; benchmark aborted")
        for kind in ("resident", "host_to_device", "mmap_end_to_end"):
            restore()
            progress.state.update(stage=kind)
            for index in range(arguments.iterations):
                torch.cuda.synchronize()
                started = time.perf_counter()
                if kind == "resident":
                    batch = resident
                elif kind == "host_to_device":
                    batch = tuple(torch.as_tensor(values, device=env.device,
                                                  dtype=env.fdtype if number == 0 else torch.float64)
                                  for number, values in enumerate(host))
                else:
                    batch = _batch(data, indices[index], env=env)
                loaded = time.perf_counter()
                step(batch)
                torch.cuda.synchronize()
                timings[kind].append((time.perf_counter() - started) * 1000)
                timings[f"{kind}/load"].append((loaded - started) * 1000)
                completed(kind)
            progress.emit(measurement=kind, timing=summarize(timings[kind]))

        restore()
        progress.state.update(stage="synchronized-stage-timings")
        for _ in range(3):
            step(resident, synchronized=True)
            completed("synchronized-stage-timings")
        restore()
        progress.state.update(stage="cpu-cuda-profiler")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True) as profiler:
            with torch.profiler.record_function("three_policy_batch"):
                step(resident, annotated=True)
            torch.cuda.synchronize()
        profiler.export_chrome_trace(str(arguments.output / "trace.json"))
        (arguments.output / "profile_cpu.txt").write_text(profiler.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=60))
        (arguments.output / "profile_cuda.txt").write_text(profiler.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=60))
        completed("finished")
        report = {"checkpoint": str(arguments.checkpoint), "checkpoint_epoch": saved["state"]["epoch"],
                  "checkpoint_next_batch": saved["state"]["next_batch"], "mode": arguments.mode,
                  "batch_size": batch_size, "iterations": arguments.iterations,
                  "warmup_seconds": warmup_seconds, "torch_version": torch.__version__,
                  "gpu": torch.cuda.get_device_name(env.device), "cpu_threads": torch.get_num_threads(),
                  "equivalence": None if equivalence is None else {key: value for key, value in equivalence.items() if key != "differences"},
                  "timings": {name: summarize(values) for name, values in timings.items()},
                  "synchronized_stage_timings": {name: summarize(values) for name, values in stage_timings.items()},
                  "note": "Stage synchronization and profiler instrumentation perturb execution; use unprofiled timings for throughput."}
        (arguments.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        progress.closed.set()
        progress.thread.join(timeout=2)
        for kind, handler in previous_signals.items():
            signal.signal(kind, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("default", "reduce-overhead", "eager"), default="default")
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--verify-default", action="store_true")
    arguments = parser.parse_args()
    if arguments.iterations < 1 or arguments.warmup < 1:
        parser.error("iterations and warmup must be positive")
    run_benchmark(arguments)


if __name__ == "__main__":
    main()
