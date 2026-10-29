"""Supervise shared Q paths and independent H/-H/zero hedges for two market books."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from quantdata.gromm2.gpu_forward_queue import resource_report, terminate_child


def stages(directory: Path, *, history: Path, estimate: Path, factor_law: Path,
           paths: int, batch_size: int, maximum_epochs: int) -> list[tuple[str, list[str]]]:
    sampling = [sys.executable, "-u", "-m", "quantdata.gromm2.paired_forward",
        "--run-directory", str(directory), "--history", str(history), "--estimate", str(estimate),
        "--factor-law", str(factor_law), "--paths", str(paths), "--batch-size", str(batch_size)]
    commands = [("shared-latent-sampling", sampling)]
    for case, kind in (("black0", "factors"), ("black0", "cohorts"), ("black5", "factors"), ("black5", "cohorts")):
        commands.append((f"{case}-{kind}", [sys.executable, "-u", "-m", "quantdata.gromm2.market_hedging",
            "--run-directory", str(directory), "--case", case, "--kind", kind,
            "--device", "cuda", "--compile", "--maximum-epochs", str(maximum_epochs)]))
    return commands


def supervise(directory: Path, commands: list[tuple[str, list[str]]], *, validation: Path) -> int:
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise RuntimeError("an inherited gpuctl allocation is required")
    if not json.loads(validation.read_text()).get("passed"):
        raise ValueError("production requires a passing numerical-equivalence report")
    directory.mkdir(parents=True, exist_ok=True)
    plan_path, status_path = directory / "execution_plan.json", directory / "execution_status.json"
    plan = dict(stages=commands, policies=["zero", "short H", "long H"], path_weights="original equal Q",
        utility="power-exponential OCE, lambda=100", neural_parameter_sharing=False,
        books=["static options and locked 10D cohorts, with the day10 ATM call rebalanced", "spot, real-option PCA baskets and daily 1D ATM call"],
        background_controls=[0.05, 0.0], latent_simulation="shared across background controls and hedge books",
        validation_report=str(validation), option_cost_bps=0.1, spot_cost_bps=0.0,
        cap=10, generation_substeps=8)
    encoded = json.dumps(plan, indent=2)
    if plan_path.exists() and plan_path.read_text() != encoded:
        raise ValueError("execution plan changed; use a new run directory")
    plan_path.write_text(encoded)
    completed = json.loads(status_path.read_text()).get("completed_stages", 0) if status_path.exists() else 0
    stop = threading.Event()
    started = time.monotonic()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGTERM, signal.SIGINT)}
    child = None

    def record(event, **extra):
        state = dict(event=event, completed_stages=completed, total_stages=len(commands),
            elapsed_seconds=time.monotonic() - started, **extra)
        temporary = status_path.with_suffix(".partial")
        temporary.write_text(json.dumps(state, indent=2))
        os.replace(temporary, status_path)
        print(json.dumps(state), flush=True)

    try:
        for index, (name, command) in enumerate(commands):
            if index < completed:
                continue
            if stop.is_set():
                return 130
            resources = resource_report()
            if resources["blocked"]:
                record("resource-stop-awaiting-outer-supervisor", resources=resources)
                stop.wait()
                return 130
            env = dict(os.environ)
            env.update(JAX_PLATFORMS="cuda" if index == 0 else "cpu",
                XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_ALLOCATOR="platform")
            child = subprocess.Popen(command, cwd=Path(__file__).parents[2], env=env, start_new_session=True)
            record("stage-started", stage=name, child_pid=child.pid)
            next_progress = time.monotonic() + 60
            while child.poll() is None:
                if stop.is_set():
                    record("checkpoint-stop", stage=name, **terminate_child(child, timeout=45))
                    return 130
                if time.monotonic() >= next_progress:
                    record("stage-running", stage=name, child_pid=child.pid)
                    next_progress = time.monotonic() + 60
                stop.wait(1)
            if child.returncode:
                if resource_report()["blocked"]:
                    record("resource-stop-awaiting-outer-supervisor", stage=name)
                    stop.wait()
                    return 130
                record("failed", stage=name, returncode=child.returncode)
                return child.returncode
            completed = index + 1
            record("stage-complete", stage=name)
        summaries = {}
        for case in ("black0", "black5"):
            summaries[case] = {}
            for kind in ("cohorts", "factors"):
                files = list((directory / "hedges" / case / kind).glob("*/quotes.json"))
                if len(files) != 1:
                    raise ValueError("expected exactly one completed quote report per scenario")
                summaries[case][kind] = json.loads(files[0].read_text())
        (directory / "comparison_summary.json").write_text(json.dumps(summaries, indent=2))
        record("complete")
        return 0
    finally:
        if child is not None and child.poll() is None:
            terminate_child(child, timeout=45)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--factor-law", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--maximum-epochs", type=int, default=2000)
    args = parser.parse_args()
    commands = stages(args.run_directory, history=args.history, estimate=args.estimate, factor_law=args.factor_law,
        paths=args.paths, batch_size=args.batch_size, maximum_epochs=args.maximum_epochs)
    raise SystemExit(supervise(args.run_directory, commands, validation=args.validation_report))


if __name__ == "__main__":
    main()
