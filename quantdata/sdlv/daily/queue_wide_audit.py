"""Run a fresh wider-grid Q audit only after the full static fit has completed."""

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-unit", required=True)
    args = parser.parse_args()
    if not args.fit_unit.startswith("quant-data-gromm-v1-43-full-"):
        raise ValueError("Only this project's explicitly named wider-fit unit may be monitored")
    if not all(os.environ.get(key) for key in ("QUANTDATA_ROOT", "PYTHONPATH")):
        raise RuntimeError("Detached project environment is missing")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("A fresh audit requires an empty, uniquely named output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    while True:
        state = subprocess.run(["systemctl", "--user", "show", "--property=ActiveState", "--value", args.fit_unit],
                               capture_output=True, text=True, timeout=15)
        active = state.stdout.strip() in {"active", "activating", "deactivating"}
        try:
            resources = subprocess.run(["gpuctl", "resources", "--json"], capture_output=True,
                                       text=True, check=True, timeout=15)
            checks = json.loads(resources.stdout)["checks"]
            healthy = bool(checks) and all(check["health"] == "OK" for check in checks)
            if not healthy:
                print(json.dumps(dict(event="resource_alert", checks=checks)), flush=True)
        except Exception as error:
            healthy = False
            print(json.dumps(dict(event="resource_warning", message=str(error))), flush=True)
        stage = "waiting_for_completed_fresh_fit" if active else "checking_completed_fit"
        if not active:
            report_path = args.run_directory / "fit/report.json"
            if not report_path.exists() or json.loads(report_path.read_text())["status"] != "complete":
                raise RuntimeError("Training exited without a completed report; no audit launched")
            if healthy:
                status = subprocess.run(["gpuctl", "status", "--json"], capture_output=True,
                                        text=True, check=True, timeout=15)
                if json.loads(status.stdout)["available"] > 0:
                    break
                stage = "waiting_for_managed_gpu"
            else:
                stage = "waiting_for_resource_health"
        print(json.dumps(dict(stage=stage, fit_unit=args.fit_unit,
                              elapsed_seconds=time.monotonic() - started)), flush=True)
        time.sleep(60)
    command = ["gpuctl", "run", "--gpus", "1", "--task-name", "Gromm-v1-4,3-Q-audit",
               "--label", "fresh-100k-paired-unconditional-and-conditional-audit", "--", sys.executable,
               "-u", "-m", "quantdata.sdlv.daily.wide_audit", "--run-directory", str(args.run_directory),
               "--output", str(args.output), "--paths", "100000", "--device", "cuda"]
    environment = {**os.environ, "JAX_PLATFORMS": "cuda"}
    print(json.dumps(dict(stage="launching_fresh_audit", command=command, output=str(args.output))), flush=True)
    subprocess.run(command, check=True, env=environment, cwd=Path(__file__).resolve().parents[3])


if __name__ == "__main__":
    main()
