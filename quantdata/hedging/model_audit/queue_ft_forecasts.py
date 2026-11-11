"""Finish missing forecast and recursive coverage through managed GPU stages."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def wait_for_gpu(gpuctl: Path, *, stage: str) -> None:
    started = time.monotonic()
    while True:
        resource_probe = subprocess.run([str(gpuctl), "resources", "--json"], capture_output=True, text=True)
        status_probe = subprocess.run([str(gpuctl), "status", "--json"], capture_output=True, text=True)
        if resource_probe.returncode or status_probe.returncode:
            print(f"GPU wait elapsed={time.monotonic()-started:.0f}s; manager unavailable; no work launched", flush=True)
        else:
            resources, status = json.loads(resource_probe.stdout), json.loads(status_probe.stdout)
            healthy = resources.get("source") == "daemon" and all(check["health"] == "OK" for check in resources["checks"])
            if healthy and status["available"] > 0:
                return
            print(f"GPU wait stage={stage} elapsed={time.monotonic()-started:.0f}s; available={status['available']} resources_OK={healthy}; ETA depends on current owner", flush=True)
        time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpuctl", type=Path, default=Path.home() / ".local/bin/gpuctl")
    arguments = parser.parse_args()
    started = time.monotonic()
    cpu = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}
    stages = [
        ("May FT forecasts", "complete_diffusion_ft", ["--stage", "sample", "--device", "cuda"]),
        ("Diffusion recursive paths", "recursive", ["--models", "AD-Seq-Vol", "AD-Seq-Vol-FT", "--device", "cuda", "--refresh-report"]),
    ]
    for stage, module, extra in stages:
        wait_for_gpu(arguments.gpuctl, stage=stage)
        command = [str(arguments.gpuctl), "run", "--gpus", "1", "--task-name", "Hedging model-law coverage",
            "--label", stage, "--", sys.executable, "-u", "-m", f"quantdata.hedging.model_audit.{module}",
            "--output", str(arguments.output), *extra]
        print(f"GPU available; launching stage={stage} elapsed={time.monotonic()-started:.0f}s", flush=True)
        subprocess.run(command, check=True)
        if module == "complete_diffusion_ft":
            for refresh, options in (("run", ["--models", "AD-Seq-Vol-FT"]), ("quotes", [])):
                subprocess.run([sys.executable, "-u", "-m", f"quantdata.hedging.model_audit.{refresh}",
                    "--output", str(arguments.output), *options], check=True, env=cpu)
        subprocess.run([sys.executable, "-u", "-m", "quantdata.hedging.model_audit.refresh",
            "--output", str(arguments.output)], check=True, env=cpu)
    print(f"Forecast and diffusion path coverage completed; elapsed={time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
