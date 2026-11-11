"""Serialize report refreshes while independent coverage jobs are running."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from filelock import FileLock


def refresh_report(output: Path) -> None:
    """Rebuild checksums, tables, figures and the compiled report under one lock."""
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}
    if (output / "PROTOCOL.json").exists() and not (output / "BASE_READY.json").exists():
        print("Report refresh waits for the corrected native one-day baselines; no mixed-protocol PDF", flush=True)
        return
    with FileLock(output / ".report-refresh.lock"):
        for module in ("provenance", "report"):
            subprocess.run([sys.executable, "-u", "-m", f"quantdata.hedging.model_audit.{module}",
                "--output", str(output)], check=True, env=environment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    refresh_report(arguments.output)


if __name__ == "__main__":
    main()
