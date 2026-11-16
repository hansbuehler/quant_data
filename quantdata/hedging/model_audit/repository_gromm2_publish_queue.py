"""Publish completed GROMM diagnostics without touching unrelated Git changes."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from quantdata.hedging.model_audit.current_report import build
from quantdata.hedging.model_audit.legacy_queue import health_check
from quantdata.hedging.model_audit.rolling_versions import origins


def command(*arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(arguments, check=check, text=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "gromm", "source", "output", "retain", "reference", "static", "recursive"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    services = ("hedging-gromm2-static-gpu-20261116.service", "hedging-gromm2-recursive-gpu-wait-20261116.service")
    started = time.monotonic()
    while any(command("systemctl", "--user", "is-active", "--quiet", service, check=False).returncode == 0 for service in services):
        health_check()
        print(f"GROMM2 publication waits for diagnostics; elapsed={time.monotonic()-started:.0f}s", flush=True)
        time.sleep(120)
    dates = origins(args.reference)
    for root in (args.static / "static", args.recursive / "recursive"):
        for measure in ("P", "Q"):
            found = sorted(path.stem for path in (root / measure).glob("*.json"))
            if found != sorted(dates):
                raise RuntimeError(f"GROMM2 diagnostic coverage incomplete: {root / measure} {len(found)}/{len(dates)}")
    if command("git", "branch", "--show-current").stdout.strip() != "hedging-benchmark":
        raise RuntimeError("Refusing publication on a different branch")
    if command("git", "diff", "--cached", "--quiet", check=False).returncode:
        raise RuntimeError("Refusing publication while other staged changes exist")
    target = str(args.retain.relative_to(Path.cwd()))
    if command("git", "status", "--porcelain", "--", target).stdout.strip():
        raise RuntimeError("Refusing to overwrite modified report files")
    if command("git", "diff", "--quiet", "--", "quantdata/hedging/model_audit/current_report.py", check=False).returncode:
        raise RuntimeError("Refusing publication with an uncommitted report generator")
    build(args.root, args.gromm, args.source, args.output, args.retain)
    names = ("CURRENT_RESULTS.json", "MODEL_DYNAMICS_REPORT.pdf", "MODEL_DYNAMICS_REPORT.tex", "RETAINED_ARTIFACTS.json",
        "STATS.md", "current_gromm2_p_static.csv", "current_gromm2_q_static.csv", "legacy/acf.pdf", "legacy/covariance.pdf",
        "legacy/density_acf.pdf", "legacy/density_tails.pdf", "legacy/tails.pdf")
    command("git", "add", *[str(args.retain.relative_to(Path.cwd()) / name) for name in names])
    command("git", "diff", "--cached", "--check")
    if command("git", "diff", "--cached", "--quiet", check=False).returncode:
        commit = command("git", "commit", "-m", "Report complete GROMM P/Q diagnostics")
        print(commit.stdout, flush=True)
        pushed = command("git", "push", "origin", "hedging-benchmark")
        print(pushed.stdout + pushed.stderr, flush=True)
    else:
        print("GROMM2 report already current; no commit needed", flush=True)


if __name__ == "__main__":
    main()
