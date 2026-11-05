"""Persist the requested Q1 -> P1(with/without a10) -> Q8 experiment order."""

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

from quantdata.sdlv.daily.wide_audit import resource_health
from quantdata.sdlv.dhgromm1.allocation import release_after_exit
from quantdata.sdlv.dhgromm1.persistence import atomic_json


def require_headroom() -> None:
    checks = resource_health()
    if not checks or any(c["health"] in {"STOP","UNAVAILABLE","ERROR"} for c in checks):
        raise RuntimeError("resource manager blocks the experiment sequence")
    if any(c["health"]!="OK" for c in checks):
        print(json.dumps(dict(stage="sequence-resource-warning",checks=checks)),flush=True)


def run_child(command: list[str], *, output: Path, gpu: bool = False) -> None:
    require_headroom()
    output.mkdir(parents=True,exist_ok=True)
    child = None
    started = time.monotonic()
    try:
        with (output/"job.log").open("a") as log:
            child = subprocess.Popen(command,cwd=Path(__file__).resolve().parents[3],stdout=log,stderr=subprocess.STDOUT)
            while True:
                try:
                    result = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(json.dumps(dict(stage="sequence-child-active",pid=child.pid,output=str(output),
                        completed=0,total=1,elapsed_seconds=time.monotonic()-started,estimated_remaining_seconds=None)),flush=True)
                    require_headroom()
            if result:
                raise RuntimeError(f"experiment child exited {result}; see {output/'job.log'}")
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if gpu and child is not None:
            release_after_exit(output)


def wait_for_q(run: Path, unit: str) -> None:
    started = time.monotonic()
    while True:
        require_headroom()
        status = run/"status.json"
        if status.exists():
            value = json.loads(status.read_text())
            if value.get("status")=="complete":
                return
            raise RuntimeError(f"Q predecessor is not complete: {value}; P training is not launched")
        active = subprocess.run(["systemctl","--user","is-active",unit],capture_output=True,text=True)
        if active.stdout.strip() not in {"active","activating"}:
            raise RuntimeError("Q predecessor has exited without a complete status; do not bypass it")
        print(json.dumps(dict(stage="waiting-for-Q1-no-a10",completed=0,total=1,
            elapsed_seconds=time.monotonic()-started,estimated_remaining_seconds=None)),flush=True)
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-run",type=Path,required=True)
    parser.add_argument("--q-unit",required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--centering-from",type=Path,required=True)
    parser.add_argument("--physical-fit",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    root = Path(os.environ["QUANTDATA_ROOT"]).resolve()
    if not output.is_relative_to(root) or output.is_relative_to(Path.cwd()):
        raise ValueError("the sequence must have a unique external run directory")
    output.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    stages = dict(q1_without_a10=str(args.q_run),p1_with_a10=str(output/"p-with-a10"),
        p1_without_a10=str(output/"p-without-a10"),q8_without_a10="deferred until both P runs finish; trading frequency to be confirmed")
    with FileLock(output/"sequence.lock",timeout=0):
        atomic_json(output/"plan.json",dict(order=stages,costs=0,epochs_per_run=500,
            physical_fit=str(args.physical_fit),Q_centering=str(args.centering_from),
            preparation="P paths may be prepared on CPU while Q trains; GPU training remains sequential"))
        try:
            prepared = output/"p-paths"
            if not (prepared/"status.json").exists():
                run_child([sys.executable,"-u","-m","quantdata.sdlv.dhgromm1.physical_paths",
                    "--reference",str(args.reference),"--fit",str(args.physical_fit),"--output",str(prepared)],output=prepared)
            if json.loads((prepared/"status.json").read_text()).get("status")!="complete":
                raise RuntimeError("P preparation did not complete")
            wait_for_q(args.q_run,args.q_unit)
            for name,flag in (("p-with-a10","--clock-with-a10"),("p-without-a10","--without-a10")):
                folder = output/name
                status = folder/"status.json"
                if status.exists() and json.loads(status.read_text()).get("status")=="complete":
                    continue
                atomic_json(output/"status.json",dict(status="running",stage=name,order=stages))
                run_child([sys.executable,"-u","-m","quantdata.sdlv.dhgromm1.floating_run",
                    "--scenarios",str(prepared/"scenarios"),"--cache",str(prepared/"market-cache"),
                    "--centering-from",str(args.centering_from),"--output",str(folder),
                    "--epochs","500","--batch-size","512","--learning-rate","1e-5",
                    "--trade-l1","0","--trade-l2","0",flag],output=folder,gpu=True)
                if json.loads(status.read_text()).get("status")!="complete":
                    raise RuntimeError(f"{name} stopped; do not start the following experiment")
            atomic_json(output/"status.json",dict(status="P-complete-Q8-deferred",order=stages,
                note="Q8 is not launched by this worker; requires intraday trading-frequency specification and tested data"))
        except BaseException as error:
            atomic_json(output/"status.json",dict(status="stopped",error=str(error),order=stages))
            raise


if __name__ == "__main__":
    main()
