"""Wait for this project's detached pilot, then allocate a GPU for the full fit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--pilot-unit", required=True)
    args = parser.parse_args()
    if not args.pilot_unit.startswith("quant-data-gromm-v1-43-pilot-"):
        raise ValueError("Only this project's exact pilot unit may be monitored")
    if not all(os.environ.get(key) for key in ("QUANTDATA_ROOT", "PYTHONPATH")):
        raise RuntimeError("Detached project environment is missing")
    started = time.monotonic()
    while True:
        state = subprocess.run(["systemctl", "--user", "show", "--property=ActiveState", "--value", args.pilot_unit],
                               capture_output=True, text=True, timeout=15)
        if state.stdout.strip() not in {"active", "activating", "deactivating"}:
            break
        print(json.dumps(dict(stage="waiting_for_pilot", unit=args.pilot_unit,
                              elapsed_seconds=time.monotonic() - started)), flush=True)
        time.sleep(60)
    report_path = args.run_directory / "pilot/report.json"
    if not report_path.exists():
        raise RuntimeError("Pilot exited without a completed stability report; full fit not launched")
    report = json.loads(report_path.read_text())
    if report["status"] != "complete":
        raise RuntimeError("Pilot is not complete; full fit not launched")
    resources = subprocess.run(["gpuctl", "resources", "--json"], capture_output=True,
                               text=True, check=True, timeout=15)
    checks = json.loads(resources.stdout)["checks"]
    if not checks or any(check["health"] != "OK" for check in checks):
        raise RuntimeError("Resource health is not all OK; full fit not launched")
    command = ["gpuctl", "run", "--gpus", "1", "--task-name", "Gromm-v1-4,3",
               "--label", "paper-schedule-full-historical-fit", "--", sys.executable, "-u", "-m",
               "quantdata.sdlv.daily.wide_fit", "fit", "--run-directory", str(args.run_directory)]
    print(json.dumps(dict(stage="launching_full_fit", command=command)), flush=True)
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[3])


if __name__ == "__main__":
    main()
