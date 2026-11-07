"""Stop a managed joint fit when its saved market-fit trend has flattened."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    args = parser.parse_args()
    while True:
        active = subprocess.run(["systemctl", "--user", "is-active", "--quiet", args.unit], check=False)
        if active.returncode:
            print("Training already finished or stopped", flush=True)
            return
        check = subprocess.run(["/home/hans/.local/bin/gpuctl", "resources", "--json"], capture_output=True, text=True, check=False)
        if check.returncode:
            print("WARNING: gpuctl resources could not be read", flush=True)
        else:
            resources = json.loads(check.stdout)
            bad = [(item["name"], item["health"]) for item in resources["checks"] if item["health"] != "OK"]
            if bad:
                print(f"Resource warning: {bad}", flush=True)
            if any(health in ("STOP", "UNAVAILABLE", "ERROR") for _, health in bad):
                subprocess.run(["systemctl", "--user", "stop", args.unit], check=False, timeout=60)
                print("Stopped training due to resource health", flush=True)
                return
        rows = []
        for line in args.log.read_text().splitlines():
            if line.startswith('{"iteration"'):
                rows.append(json.loads(line))
        if rows and rows[-1]["iteration"] >= 100:
            current = rows[-1]
            previous = next((row for row in reversed(rows) if row["iteration"] <= current["iteration"] - 100), None)
            if previous is not None:
                gain = current["inside_percent"] - previous["inside_percent"]
                relative_loss_drop = (previous["quote_and_cap"] - current["quote_and_cap"]) / abs(previous["quote_and_cap"])
                print(f"iteration={current['iteration']} gain_pp={gain:.6f} quote_loss_relative_drop={relative_loss_drop:.6f}", flush=True)
                if current["inside_percent"] >= 90. and gain < .1 and relative_loss_drop < .01:
                    subprocess.run(["systemctl", "--user", "stop", args.unit], check=True, timeout=60)
                    print(f"Stopped at latest saved checkpoint after saturation at iteration {current['iteration']}", flush=True)
                    return
        time.sleep(30)


if __name__ == "__main__":
    main()
