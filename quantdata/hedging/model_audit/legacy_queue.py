"""Resource-managed original-code rolling jobs; never borrow another GPU claim."""

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

from quantdata.hedging.model_audit.rolling_versions import origins, resource_health
from quantdata.hedging.model_audit.run import _json

STOP = threading.Event()


def health_check() -> None:
    if any(value in ("STOP", "UNAVAILABLE", "ERROR") for value in resource_health()):
        raise InterruptedError("Managed resources require a stop; retain prior completed checkpoints")


def run_owned(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    """Monitor and stop only this queue's child process group when required."""
    health_check()
    process = subprocess.Popen(command, env=environment, start_new_session=True)
    while process.poll() is None:
        try:
            process.wait(timeout=60.)
        except subprocess.TimeoutExpired:
            try:
                health_check()
                if STOP.is_set():
                    raise InterruptedError("User requested stopping the owned queue")
            except InterruptedError:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=60.)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def wait_gpu(root: Path, *, origin: str, completed: int, total: int, poll_seconds: float = 60.) -> None:
    started = time.monotonic()
    while not STOP.is_set():
        health_check()
        status = subprocess.run(["gpuctl", "status", "--json"], check=True, capture_output=True, text=True)
        if json.loads(status.stdout)["available"]:
            return
        event = {"phase": "waiting for managed GPU", "origin": origin, "completed": completed,
            "total": total, "wait_elapsed_seconds": time.monotonic() - started, "eta_seconds": None}
        _json(root / "gpu-queue-progress.json", event)
        print(json.dumps(event), flush=True)
        STOP.wait(poll_seconds)
    raise InterruptedError("Queue stopped while waiting for a GPU")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=("dysanos1", "gromm1"), default=["dysanos1", "gromm1"])
    parser.add_argument("--after-dysanos1", action="store_true")
    parser.add_argument("--dysanos-device", choices=("cpu", "cuda:0"), default="cpu")
    args = parser.parse_args()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: STOP.set())
    selected = origins(args.reference)
    args.root.mkdir(parents=True, exist_ok=True)
    if args.after_dysanos1:
        while not STOP.is_set():
            health_check()
            completed = sum((args.root / f"fits/dysanos1/{origin}/COMPLETE.json").exists() for origin in selected)
            if completed == len(selected):
                break
            producer = subprocess.run(["systemctl", "--user", "is-active", "hedging-original-dysanos1-20261113.service"],
                capture_output=True, text=True)
            if producer.returncode:
                raise RuntimeError("DYSANOS producer stopped before completing its window queue; no GROMM fit launched")
            print(f"GROMM waits for DYSANOS v1 completed={completed}/{len(selected)}", flush=True)
            STOP.wait(60.)
        if STOP.is_set():
            raise InterruptedError("Dependent queue stopped before scoring or launching another fit")
        for origin in selected:
            if not (args.root / f"daily/DYSANOS_v1-P/{origin}.json").exists():
                run_owned([sys.executable, "-u", "-m", "quantdata.hedging.model_audit.legacy_rolling",
                    "--root", str(args.root), "--reference", str(args.reference), "--origin", origin, "--score-only"],
                    environment=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    for family, module in (("dysanos1", "legacy_rolling"), ("gromm1", "legacy_gromm")):
        if family not in args.families:
            continue
        for index, origin in enumerate(selected):
            if STOP.is_set():
                raise InterruptedError("Queue stopped before starting another origin")
            if (args.root / f"fits/{family}/{origin}/COMPLETE.json").exists():
                continue
            command = [sys.executable, "-u", "-m", f"quantdata.hedging.model_audit.{module}",
                "--root", str(args.root), "--reference", str(args.reference), "--origin", origin]
            if family == "dysanos1":
                command.extend(["--device", args.dysanos_device])
            print(f"{family} preparing origin={origin} completed={index}/{len(selected)}", flush=True)
            run_owned([*command, "--prepare-only"], environment=dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu"))
            if family == "dysanos1" and args.dysanos_device == "cpu":
                print(f"{family} training on CPU origin={origin}; no GPU allocation required", flush=True)
                run_owned(command, environment=dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu"))
                continue
            wait_gpu(args.root, origin=origin, completed=index, total=len(selected))
            run_owned(["gpuctl", "run", "--gpus", "1", "--task-name", "Hedging original-code models",
                "--task-id", "01a12f09-4975-7431-a499-7c321ab35fa5", "--label", f"{family} affine200 {origin}", "--", *command])
        _json(args.root / f"{family}_TRAINING_COMPLETE.json", {"origins": len(selected),
            "status": "all fits saved and one-day scored; remaining diagnostics still required"})


if __name__ == "__main__":
    main()
