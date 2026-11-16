"""Finish the two rolling fit queues on one explicitly owned GPU allocation."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from filelock import FileLock

from quantdata.hedging.model_audit.legacy_queue import STOP, health_check, run_owned
from quantdata.hedging.model_audit.rolling_versions import origins, resource_health
from quantdata.hedging.model_audit.run import _json

TASK_ID = "01a12f09-4975-7431-a499-7c321ab35fa5"


def worker_environment(path: Path, *, cpu: bool = False) -> dict[str, str]:
    """Load the explicit detached environment, letting gpuctl pin GPU children."""
    environment = dict(os.environ)
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            name, value = line.split("=", 1)
            environment[name] = value
    if cpu:
        environment.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    else:
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        environment.pop("JAX_PLATFORMS", None)
    if not environment.get("QUANTDATA_ROOT") or not environment.get("PYTHONPATH"):
        raise ValueError("Detached workers need explicit QUANTDATA_ROOT and PYTHONPATH")
    return environment


def verify_allocation(reference: str) -> None:
    status = json.loads(subprocess.run(["gpuctl", "status", "--json"], check=True,
        capture_output=True, text=True).stdout)
    allocation = next((item for item in status["allocations"] if item["ref"] == reference), None)
    if allocation is None or allocation.get("task_id") != TASK_ID:
        raise RuntimeError("The requested allocation does not belong to this exact hedging thread")


def execute(args: argparse.Namespace) -> None:
    selected = origins(args.reference)
    dysanos_env = worker_environment(args.dysanos_env)
    gromm_env = worker_environment(args.gromm_env)
    started = time.monotonic()
    verify_allocation(args.allocation)
    args.control.mkdir(parents=True, exist_ok=True)

    def progress(phase: str, origin: str | None = None) -> None:
        record = {"phase": phase, "origin": origin, "allocation": args.allocation,
            "task_id": TASK_ID, "elapsed_seconds": time.monotonic() - started, "eta_seconds": None,
            "dysanos_fits": sum((args.dysanos / f"fits/dysanos1/{day}/FIT_COMPLETE.json").exists() for day in selected),
            "gromm_fits": sum((args.gromm / f"gromm2/{day}/ESTIMATION_COMPLETE.json").exists() for day in selected),
            "total_per_model": len(selected)}
        _json(args.control / "progress.json", record)
        print(json.dumps(record), flush=True)

    def gpu(command: list[str], environment: dict[str, str]) -> None:
        verify_allocation(args.allocation)
        run_owned(["gpuctl", "run", "--allocation", args.allocation, "--", *command], environment=environment)

    try:
        with FileLock(args.control / ".queue.lock"):
            while True:
                health_check()
                if STOP.is_set():
                    raise InterruptedError("Owned completion queue stopped")
                pending_dysanos = [day for day in selected
                    if not (args.dysanos / f"fits/dysanos1/{day}/FIT_COMPLETE.json").exists()]
                pending_gromm = [day for day in selected
                    if not (args.gromm / f"gromm2/{day}/ESTIMATION_COMPLETE.json").exists()]
                if not pending_dysanos and not pending_gromm:
                    break
                if pending_dysanos:
                    day = pending_dysanos[0]
                    progress("DYSANOS v1 paper-objective GPU fit", day)
                    gpu([sys.executable, "-u", "-m", "quantdata.hedging.model_audit.legacy_rolling",
                        "--root", str(args.dysanos), "--reference", str(args.reference), "--origin", day,
                        "--device", "cuda:0", "--paper-midpoint", "--fit-only",
                        "--market-inputs-from", str(args.market_inputs)], dysanos_env)
                ready = next((day for day in pending_gromm
                    if (args.gromm / f"gromm2/{day}/prepared/COMPLETE.json").exists()), None)
                if ready is not None:
                    progress("GROMM2 paper-objective GPU fit and dynamics", ready)
                    gpu([sys.executable, "-u", "-m", "quantdata.hedging.model_audit.repository_gromm2",
                        "--run", str(args.gromm), "--reference", str(args.reference), "--origin", ready,
                        "--fit-only"], gromm_env)
                elif not pending_dysanos:
                    progress("waiting for CPU calibration inputs")
                    if subprocess.run(["systemctl", "--user", "is-active", "--quiet", args.producer_unit]).returncode:
                        raise RuntimeError("GROMM CPU preparation stopped before all required inputs were ready")
                    STOP.wait(60.)
            for day in selected:
                progress("GROMM2 GPU-dependent P/Q forecasts", day)
                gpu([sys.executable, "-u", "-m", "quantdata.hedging.model_audit.repository_gromm2",
                    "--run", str(args.gromm), "--reference", str(args.reference), "--origin", day], gromm_env)
            progress("GPU work complete")
            _json(args.control / "GPU_COMPLETE.json", {"origins_per_model": len(selected)})
    except BaseException as error:
        _json(args.control / "REQUIRES_REVIEW.json", {"error_type": type(error).__name__, "exact_error": str(error)})
        raise
    finally:
        if any(value in ("STOP", "UNAVAILABLE", "ERROR") for value in resource_health()):
            print("Resource stop: allocation retained pending the required release approval", flush=True)
        else:
            subprocess.run(["gpuctl", "release", "--allocation", args.allocation], check=True)
    failures = []
    for day in selected:
        progress("DYSANOS v1 CPU observable statistics; GPU released", day)
        try:
            run_owned([sys.executable, "-u", "-m", "quantdata.hedging.model_audit.legacy_rolling",
                "--root", str(args.dysanos), "--reference", str(args.reference), "--origin", day,
                "--score-only"], environment=worker_environment(args.dysanos_env, cpu=True))
        except subprocess.CalledProcessError as error:
            failures.append({"origin": day, "returncode": error.returncode,
                "exact_error": str(error), "details": "Full numerical exception is preserved in completion.log"})
            _json(args.control / "CPU_STATISTICS_REQUIRES_REVIEW.json", {"failures": failures})
    _json(args.control / "QUEUE_COMPLETE.json", {"fit_origins_per_model": len(selected),
        "dysanos_scoring_failures": failures, "gpu_released": True,
        "status": "CPU scoring requires review" if failures else "fits and one-day observable scores complete"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "dysanos", "gromm", "reference", "market-inputs", "dysanos-env", "gromm-env"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--allocation", required=True)
    parser.add_argument("--producer-unit", required=True)
    args = parser.parse_args()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: STOP.set())
    execute(args)


if __name__ == "__main__":
    main()
