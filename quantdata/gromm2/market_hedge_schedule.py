"""Run the spot/option and PCA books sequentially under one GPU allocation."""

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

from quantdata.gromm2.hedging_policy import _require_cuda_visibility
from quantdata.gromm2.rolling_audit import check_resource_headroom


def hedge_commands(run_directory: Path, output_directory: Path, *, maximum_epochs: int,
                    memory_fraction: float) -> dict[str, list[str]]:
    """Keep independent checkpoints and identical economic/optimization controls."""
    if maximum_epochs < 1 or not 0 < memory_fraction <= 1:
        raise ValueError("positive epochs and per-job memory fraction in (0,1] are required")
    common = [sys.executable, "-u", "-m", "quantdata.gromm2.market_hedging", "--run-directory", str(run_directory),
        "--case", "black0", "--device", "cuda", "--compile", "--maximum-epochs", str(maximum_epochs),
        "--spot-cost-bps", "0.1", "--cuda-memory-fraction", str(memory_fraction)]
    return {kind: common + ["--kind", kind, "--output-directory", str(output_directory / "policies" / kind)]
        + (["--with-spot"] if kind == "cohorts" else []) for kind in ("cohorts", "factors")}


def stop_children(children: dict[str, subprocess.Popen], *, timeout: float = 55) -> None:
    """Stop live owned sessions within the supervisor's checkpoint grace period."""
    pending = [child for child in children.values() if child.poll() is None]
    for child in pending:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    for child in pending:
        try:
            child.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for child in pending:
        child.wait(timeout=5)


def run_schedule(run_directory: Path, output_directory: Path, *, maximum_epochs: int = 2000,
                  memory_fraction: float = 1.0) -> int:
    """Never overlap the books; release remains the allocation owner's job."""
    _require_cuda_visibility()
    check_resource_headroom()
    commands = hedge_commands(run_directory, output_directory, maximum_epochs=maximum_epochs,
        memory_fraction=memory_fraction)
    output_directory.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    previous = {kind: signal.signal(kind, lambda *_: stop.set()) for kind in (signal.SIGINT, signal.SIGTERM)}
    children, streams = {}, []
    started = time.monotonic()
    try:
        for kind, command in commands.items():
            if stop.is_set():
                return 130
            check_resource_headroom()
            stream = (output_directory / f"{kind}.log").open("a", buffering=1)
            streams.append(stream)
            children[kind] = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            print(json.dumps(dict(event="scheduled-hedge-launched", book=kind, pid=children[kind].pid,
                command=command, elapsed_seconds=time.monotonic() - started, completed=len(children) - 1, total=2)), flush=True)
            next_report = time.monotonic()
            while children[kind].poll() is None:
                if stop.is_set():
                    return 130
                now = time.monotonic()
                if now >= next_report:
                    check_resource_headroom()
                    print(json.dumps(dict(event="scheduled-hedge-progress", elapsed_seconds=now - started,
                        completed=len(children) - 1, total=2, running=kind,
                        queued=[name for name in commands if name not in children], remaining_seconds=None)), flush=True)
                    next_report = now + 60
                stop.wait(1)
            if children[kind].returncode != 0:
                raise RuntimeError(f"hedge book {kind} failed with code {children[kind].returncode}; later books not launched")
        print(json.dumps(dict(event="scheduled-hedges-completed", elapsed_seconds=time.monotonic() - started,
            completed=2, total=2)), flush=True)
        return 0
    finally:
        stop_children(children)
        for stream in streams:
            stream.close()
        for kind, handler in previous.items():
            signal.signal(kind, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--maximum-epochs", type=int, default=2000)
    parser.add_argument("--memory-fraction", type=float, default=1.0)
    args = parser.parse_args()
    raise SystemExit(run_schedule(args.run_directory, args.output_directory,
        maximum_epochs=args.maximum_epochs, memory_fraction=args.memory_fraction))


if __name__ == "__main__":
    main()
