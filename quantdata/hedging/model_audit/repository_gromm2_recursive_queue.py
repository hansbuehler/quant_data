"""Run matched GROMM2 recursive diagnostics within one managed allocation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from quantdata.hedging.model_audit.legacy_queue import health_check, run_owned
from quantdata.hedging.model_audit.rolling_versions import origins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    selected = origins(args.reference)
    if len(selected) != 44:
        raise ValueError("Expected the approved 44 origin dates")
    completed, total = 0, 2 * len(selected)
    started = time.monotonic()
    for measure in ("Q", "P"):
        for origin in selected:
            health_check()
            destination = args.output / f"recursive/{measure}/{origin}.json"
            if destination.exists():
                record = json.loads(destination.read_text())
                if (record["origin"] != origin or record["model"] != f"repository GROMM2-{measure}"
                        or record["paths"] != 32 or record["total_static_paths"] != 1000 or record["steps"] != 115):
                    raise ValueError(f"Incompatible saved recursive diagnostic: {destination}")
            else:
                command = [sys.executable, "-u", "-m", "quantdata.hedging.model_audit.repository_gromm2_recursive",
                    "--run", str(args.run), "--reference", str(args.reference), "--output", str(args.output),
                    "--origin", origin, "--measure", measure, "--paths", "32", "--total-paths", "1000", "--steps", "115"]
                print(f"GROMM2 recursive starting {measure} {origin} completed={completed}/{total}", flush=True)
                run_owned(command, environment=dict(os.environ))
            completed += 1
            elapsed = time.monotonic() - started
            print(f"GROMM2 recursive origins completed={completed}/{total} elapsed={elapsed:.1f}s "
                f"eta={elapsed*(total-completed)/completed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
