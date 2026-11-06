"""Finish the active P baselines, then run four learned-policy penalty controls."""

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

from quantdata.sdlv.dhgromm1.persistence import atomic_json
from quantdata.sdlv.dhgromm1.sequence import require_headroom, run_child


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--predecessor-unit", required=True)
    parser.add_argument("--source-run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--l1", type=float, default=1e-4)
    parser.add_argument("--l2", type=float, default=1e-8)
    args = parser.parse_args()
    output = args.output.resolve()
    root = Path(os.environ["QUANTDATA_ROOT"]).resolve()
    if not output.is_relative_to(root) or output.is_relative_to(Path.cwd()):
        raise ValueError("use a unique external sequence directory")
    output.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_:sys.exit(143))
    plan = dict(sources=[str(p.resolve()) for p in args.source_run], epochs_per_regularized_phase=args.epochs,
        action_l1=args.l1, action_l2=args.l2, transaction_costs=0,
        regularizer_treatment="outside_utility", initialization="source-best weights/cash, fresh Adam",
        predecessor=str(args.predecessor), predecessor_unit=args.predecessor_unit,
        Q8="eight simulation substeps with DAILY hedging confirmed; separate new experiment, not a learned-point continuation",
        correction="replaces planned unregularized +500 extensions; original baselines are retained")
    with FileLock(output/"sequence.lock", timeout=0):
        atomic_json(output/"plan.json", plan)
        started = time.monotonic()
        try:
            while True:
                require_headroom()
                status = args.predecessor/"status.json"
                previous = json.loads(status.read_text()) if status.exists() else {}
                if previous.get("status") == "P-complete-Q8-deferred":
                    break
                if previous.get("status") == "stopped":
                    raise RuntimeError("baseline predecessor stopped; do not bypass it")
                active = subprocess.run(["systemctl", "--user", "is-active", args.predecessor_unit],
                                        capture_output=True, text=True)
                if active.stdout.strip() not in {"active", "activating"}:
                    raise RuntimeError("baseline sequence exited without the required completion status")
                atomic_json(output/"status.json", dict(status="waiting", stage="finish-original-P-baselines"))
                print(json.dumps(dict(stage="waiting-for-baselines", completed=0, total=len(args.source_run),
                    elapsed_seconds=time.monotonic()-started, estimated_remaining_seconds=None)), flush=True)
                time.sleep(30)
            for i, source in enumerate(args.source_run):
                folder = output/f"{i+1:02d}-{source.name}"
                status = folder/"status.json"
                if status.exists() and json.loads(status.read_text()).get("status") == "complete":
                    continue
                if json.loads((source/"status.json").read_text()).get("status") != "complete":
                    raise RuntimeError(f"source baseline is not complete: {source}")
                atomic_json(output/"status.json", dict(status="running", stage=source.name,
                    completed=i, total=len(args.source_run), run=str(folder)))
                run_child([sys.executable, "-u", "-m", "quantdata.sdlv.dhgromm1.regularized_run",
                    "--source-run", str(source), "--output", str(folder), "--epochs", str(args.epochs),
                    "--l1", str(args.l1), "--l2", str(args.l2)], output=folder, gpu=True)
                if json.loads(status.read_text()).get("status") != "complete":
                    raise RuntimeError("regularized child stopped; following experiments are not launched")
            atomic_json(output/"status.json", dict(status="complete", completed=len(args.source_run), total=len(args.source_run)))
        except BaseException as error:
            atomic_json(output/"status.json", dict(status="stopped", error=str(error)))
            raise


if __name__ == "__main__":
    main()
