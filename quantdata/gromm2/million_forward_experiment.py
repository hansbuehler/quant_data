"""Supervise paired million-path sampling and independent bid/ask hedge experiments."""

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


def experiment_stages(*, run_directory: Path, history: Path, estimate: Path, factor_law: Path,
                       paths: int, batch_size: int, seed: int, maximum_epochs: int,
                       compile_policies: bool) -> list[tuple[str, list[str]]]:
    """Use common random numbers across smoothing controls and common paths across books."""
    stages = []
    for name, override in (("black5", []), ("black0", ["--background-volatility", "0"])):
        command = [sys.executable, "-u", "-m", "quantdata.gromm2.forward_hedging",
            "--legacy-history", str(history), "--legacy-estimate", str(estimate), "--legacy-factor-law", str(factor_law),
            "--run-directory", str(run_directory / name), "--origin", "2025-08-29", "--paths", str(paths),
            "--batch-size", str(batch_size), "--seed", str(seed), "--substeps", "8", "--backend", "gpu",
            "--linear-solver", "pcr", "--fixing-hedge-moneyness", "0.98", "0.99", "1", "1.01", "1.02",
            "--contiguous-storage", "--scenarios-only", *override]
        stages.append((f"{name}-sampling", command))
    for name in ("black5", "black0"):
        for with_fixing in (True, False):
            command = [sys.executable, "-u", "-m", "quantdata.gromm2.portfolio_hedging",
                "--run-directory", str(run_directory / name), "--device", "cuda", "--risk-aversion", "100",
                "--maximum-epochs", str(maximum_epochs)]
            if not with_fixing:
                command.append("--without-fixing-options")
            if compile_policies:
                command.append("--compile")
            variant = "with_fixing_options" if with_fixing else "without_fixing_options"
            stages.append((f"{name}-{variant}", command))
    return stages


def run_stages(run_directory: Path, stages: list[tuple[str, list[str]]]) -> int:
    """Keep progress, propagate stops, and leave release authorization to gpuctl's supervisor."""
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise RuntimeError("the experiment requires an inherited gpuctl GPU allocation")
    run_directory.mkdir(parents=True, exist_ok=True)
    plan_path = run_directory / "pipeline_plan.json"
    plan = {"stages": stages, "independent_policies": ["zero", "short payoff", "long payoff"],
        "portfolio_parameter_sharing": False, "path_weights": "Original Q, equal weights; no preconditioning.",
        "claim": "abs(S_20/S_10-1)", "cost_bps_of_spot_notional": 0.1,
        "spot_trading": False, "risk_aversion": 100,
        "smoothing_control": "Zero Black background only in simulation and option pricing; no historical refit.",
        "quote_band": "Bid minus three paired MC SEs to ask plus three paired MC SEs; checkpoint-selection uncertainty excluded."}
    serialized = json.dumps(plan, indent=2)
    if plan_path.exists() and plan_path.read_text() != serialized:
        raise ValueError("pipeline plan changed; preserve the prior run and use a new run directory")
    plan_path.write_text(serialized)
    status_path = run_directory / "pipeline_status.json"
    completed = json.loads(status_path.read_text()).get("completed_stages", 0) if status_path.exists() else 0
    started = time.monotonic()
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    child = None

    def record(event: str, **extra) -> None:
        state = {"event": event, "completed_stages": completed, "total_stages": len(stages),
            "elapsed_seconds": time.monotonic() - started, "remaining_seconds": None, **extra}
        temporary = status_path.with_suffix(".partial")
        temporary.write_text(json.dumps(state, indent=2))
        os.replace(temporary, status_path)
        print(json.dumps(state), flush=True)

    try:
        for index, (name, command) in enumerate(stages):
            if index < completed:
                continue
            resources = resource_report()
            if resources["blocked"]:
                record("resource_stopped_waiting_for_supervisor", resources=resources)
                stop.wait()
                return 130
            if resources["warning"]:
                record("resource_warning", resources=resources)
            if stop.is_set():
                return 130
            env = dict(os.environ)
            env.update(JAX_PLATFORMS="cuda" if name.endswith("sampling") else "cpu",
                XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_ALLOCATOR="platform")
            child = subprocess.Popen(command, env=env, cwd=Path(__file__).parents[2], start_new_session=True)
            record("stage_started", stage=name, child_pid=child.pid, command=command)
            next_progress = time.monotonic() + 60.0
            while child.poll() is None:
                if stop.is_set():
                    record("checkpoint_stop_requested", stage=name, **terminate_child(child, timeout=45.0))
                    return 130
                if time.monotonic() >= next_progress:
                    record("stage_running", stage=name, child_pid=child.pid)
                    next_progress = time.monotonic() + 60.0
                stop.wait(1.0)
            if child.returncode:
                resources = resource_report()
                if resources["blocked"]:
                    record("resource_stopped_waiting_for_supervisor", resources=resources)
                    stop.wait()
                    return 130
                record("stage_failed", stage=name, returncode=child.returncode)
                return child.returncode
            completed = index + 1
            record("stage_complete", stage=name)
        summary = {}
        for background in ("black5", "black0"):
            summary[background] = {}
            for variant in ("with_fixing_options", "without_fixing_options"):
                files = list((run_directory / background / variant).glob("*/quotes.json"))
                if len(files) != 1:
                    raise ValueError(f"expected one completed price report for {background}/{variant}")
                report = json.loads(files[0].with_name("training_report.json").read_text())
                summary[background][variant] = {"best_epochs": {
                    name: value["best_epoch"] for name, value in report["policies"].items()},
                    "quotes": json.loads(files[0].read_text())}
        (run_directory / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
        record("completed")
        return 0
    finally:
        if child is not None and child.poll() is None:
            terminate_child(child, timeout=45.0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--factor-law", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026102801)
    parser.add_argument("--maximum-epochs", type=int, default=2000)
    parser.add_argument("--compile", action="store_true")
    arguments = parser.parse_args()
    stages = experiment_stages(run_directory=arguments.run_directory, history=arguments.history,
        estimate=arguments.estimate, factor_law=arguments.factor_law, paths=arguments.paths,
        batch_size=arguments.batch_size, seed=arguments.seed, maximum_epochs=arguments.maximum_epochs,
        compile_policies=arguments.compile)
    raise SystemExit(run_stages(arguments.run_directory, stages))


if __name__ == "__main__":
    main()
