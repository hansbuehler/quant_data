"""Atomic lightweight records, independent of retired training entry points."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def write_history(output: Path, history: list[dict]) -> None:
    temporary = output/"triple-losses.tmp.csv"
    with temporary.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(output/"triple-losses.csv")
