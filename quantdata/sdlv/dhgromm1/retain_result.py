"""Intentionally retain small completed reports, not paths/checkpoints/generated PDFs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from quantdata.sdlv.daily.wide_audit import sha256
from quantdata.sdlv.dhgromm1.persistence import atomic_json


def retain(*, source: Path, output: Path) -> dict:
    if json.loads((source/"status.json").read_text()).get("status")!="complete":
        raise ValueError("only retain completed results; do not label a running snapshot as final")
    output.mkdir(parents=True,exist_ok=False)
    names = ["identity.json","report.json","status.json","scalar-benchmarks.json","gpu-verification.json",
             "objective-change-rescore.json","triple-losses.csv","figures/losses.png"]
    files = []
    for name in names:
        path = source/name
        if not path.exists():
            continue
        if path.stat().st_size>4_000_000:
            raise ValueError(f"unexpectedly large result artifact: {path}")
        target = output/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if path.suffix==".csv":
            # Canonical LF survives Git checkout; record source and retained
            # hashes separately rather than claiming byte-identical CSVs.
            target.write_text(path.read_text())
        else:
            shutil.copyfile(path,target)
        files.append(dict(path=name,source_sha256=sha256(path),retained_sha256=sha256(target)))
    record = dict(source=str(source.resolve()),status="complete",files=files,
        retained="deliberate compact result snapshot; CSV line endings normalized to LF; raw scenarios, market cache, weights, logs and scratch stay outside Git")
    atomic_json(output/"retention.json",record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(retain(source=args.source,output=args.output),indent=2))
