"""Run disjoint recursive-audit origins concurrently without changing model laws or seeds."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from quantdata.hedging.model_audit.refresh import refresh_report


def partition_origins(origins: list[str], workers: int) -> tuple[tuple[str, ...], ...]:
    """Assign each origin exactly once, preserving chronological order within a worker."""
    if workers < 1 or workers > len(origins) or len(set(origins)) != len(origins):
        raise ValueError("Workers must be positive and origins must be unique")
    return tuple(tuple(origins[index::workers]) for index in range(workers))


def completed_origins(output: Path, models: list[str]) -> int:
    return sum(json.loads(path.read_text()).get("status") in ("scored", "model_invalid")
               for model in models for path in (output / "persistence" / model).glob("*.json"))


def stop_workers(processes: list[subprocess.Popen]) -> None:
    """Request session-boundary checkpoints, then enforce the host's sixty-second limit."""
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 60.
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("GROMM-Q", "GROMM-P", "AD-Seq-Vol", "AD-Seq-Vol-FT"), required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments = parser.parse_args()
    origins = json.loads((arguments.output / "observable_reference.json").read_text())["dates"]
    partitions = partition_origins(origins, arguments.workers)
    module = "quantdata.hedging.model_audit.recursive"
    base = [sys.executable, "-u", "-m", module, "--output", str(arguments.output),
        "--models", *arguments.models, "--paths", "32", "--steps", "115", "--device", arguments.device]
    subprocess.run([*base, "--preflight"], check=True)
    directory = arguments.output / "logs" / f"parallel-{arguments.device}-{'-'.join(arguments.models)}"
    directory.mkdir(parents=True, exist_ok=True)
    if any(os.environ.get(name) is None for name in ("QUANTDATA_ROOT", "PYTHONPATH")):
        raise ValueError("Detached workers require explicit QUANTDATA_ROOT and PYTHONPATH")
    processes, streams = [], []
    interrupted = False

    def request_stop(signum, frame) -> None:
        nonlocal interrupted
        interrupted = True
        print(f"signal {signum}: stopping workers at saved session boundaries", flush=True)
        for process in processes:
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.monotonic()
    initial = completed_origins(arguments.output, arguments.models)
    total = len(origins) * len(arguments.models)
    try:
        for index, dates in enumerate(partitions):
            stream = (directory / f"worker-{index}.log").open("a", buffering=1)
            streams.append(stream)
            processes.append(subprocess.Popen([*base, "--dates", *dates], stdout=stream, stderr=subprocess.STDOUT))
            print(f"worker {index} pid={processes[-1].pid} origins={len(dates)} models={arguments.models}", flush=True)
        while True:
            count = completed_origins(arguments.output, arguments.models)
            elapsed = time.monotonic() - started
            eta = elapsed * (total - count) / (count - initial) if count > initial else None
            print(json.dumps({"stage": "parallel-recursive", "models": arguments.models,
                "completed_origins": count, "total_origins": total, "elapsed_seconds": round(elapsed, 1),
                "remaining_seconds": round(eta, 1) if eta is not None else None,
                "live_workers": sum(process.poll() is None for process in processes)}), flush=True)
            if interrupted:
                stop_workers(processes)
                return
            failures = [process.returncode for process in processes if process.poll() not in (None, 0)]
            if failures:
                raise RuntimeError(f"Recursive worker failed with exit codes {failures}; see {directory}")
            if all(process.poll() is not None for process in processes):
                if count != total:
                    raise RuntimeError(f"Workers exited with incomplete coverage: {count}/{total}")
                break
            time.sleep(60.)
        refresh_report(arguments.output)
    finally:
        stop_workers(processes)
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    main()
