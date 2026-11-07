"""Resume or partition objective runs using their verified, frozen implementation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import sys
from datetime import date
from pathlib import Path
from zipfile import ZipFile


def restore_implementation(*, input_directory: Path, output_directory: Path,
                           supplemental_directory: Path | None = None) -> Path:
    """Extract only simple Python filenames, verified against the run's source hashes."""
    config = json.loads((input_directory / "config.json").read_text())
    expected = config["implementation_sha256"]
    destination = output_directory / "frozen_implementation"
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(input_directory / "implementation.zip") as archive:
        if set(archive.namelist()) != set(expected):
            raise ValueError("Frozen source archive does not match the recorded implementation files")
        for name, digest in expected.items():
            if Path(name).name != name or not name.endswith(".py"):
                raise ValueError(f"Unsafe frozen source filename: {name}")
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"Frozen source hash mismatch: {name}")
            path = destination / name
            if path.exists() and path.read_bytes() != content:
                raise ValueError(f"Existing frozen source differs from the saved implementation: {path}")
            if not path.exists():
                path.write_bytes(content)
    record_path = output_directory / "supplemental_sources.json"
    supplements = json.loads(record_path.read_text()) if record_path.exists() else {}
    visited, queue = set(), list(expected)
    while queue:
        name = queue.pop()
        if name in visited:
            continue
        visited.add(name)
        content = (destination / name).read_bytes()
        for node in ast.walk(ast.parse(content)):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                       [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                prefix = "quantdata.hedge2."
                if not module.startswith(prefix):
                    continue
                suffix = module[len(prefix):]
                if "." in suffix or not suffix.isidentifier():
                    raise ValueError(f"Unsupported nested project dependency in frozen source: {module}")
                dependency = suffix + ".py"
                if dependency not in expected:
                    path = destination / dependency
                    if dependency in supplements:
                        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != supplements[dependency]["sha256"]:
                            raise ValueError(f"Frozen supplemental dependency changed: {dependency}")
                    elif supplemental_directory is None:
                        raise ValueError(f"Parent snapshot omitted dependency {dependency}; supply and record its source explicitly")
                    else:
                        source = supplemental_directory / dependency
                        data = source.read_bytes()
                        if path.exists() and path.read_bytes() != data:
                            raise ValueError(f"Supplemental dependency has conflicting existing contents: {path}")
                        path.write_bytes(data)
                        supplements[dependency] = {"source": str(source), "sha256": hashlib.sha256(data).hexdigest(),
                            "provenance": "Transitive dependency omitted from original archive; captured explicitly on resume"}
                queue.append(dependency)
    if supplements:
        record_path.write_text(json.dumps(supplements, indent=2) + "\n")
    return destination


def run_snapshot(*, input_directory: Path, output_directory: Path,
                 cohort_config: Path | None = None,
                 supplemental_directory: Path | None = None) -> dict[str, object]:
    parent = json.loads((input_directory / "config.json").read_text())
    dates = parent["dates"] if cohort_config is None else json.loads(cohort_config.read_text())["dates"]
    if not dates or dates != sorted(set(dates)) or not set(dates).issubset(parent["dates"]):
        raise ValueError("A partition must be a nonempty, ordered subset of the original cohort")
    loaded = [name for name in sys.modules if name.startswith("quantdata.hedge2.")
              and name not in (__name__, "quantdata.hedge2.snapshot_run")]
    if loaded:
        raise RuntimeError(f"Frozen runs need a fresh interpreter; these project modules are already loaded: {loaded}")
    frozen = restore_implementation(input_directory=input_directory, output_directory=output_directory,
                                     supplemental_directory=supplemental_directory)
    package = importlib.import_module("quantdata.hedge2")
    package.__path__ = [str(frozen)]
    multiday = parent.get("experiment_type") == "multiday"
    if multiday and cohort_config is not None:
        raise ValueError("Multi-day partitioning uses the saved episode shard, not a replacement daily cohort")
    entrypoint = "episode_run" if multiday else "objective_run"
    module = importlib.import_module(f"quantdata.hedge2.{entrypoint}")
    if Path(module.__file__).resolve() != (frozen / f"{entrypoint}.py").resolve():
        raise RuntimeError("The objective runner did not load from its verified implementation snapshot")
    record = {"implementation_parent": str(input_directory),
              "parent_configuration_sha256": hashlib.sha256((input_directory / "config.json").read_bytes()).hexdigest(),
              "frozen_implementation": str(frozen), "dates": dates,
              "supplemental_sources": str(output_directory / "supplemental_sources.json"),
              "same_data_objectives_costs_and_seed_rule": True}
    (output_directory / "snapshot_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"verified frozen implementation: {frozen}; dates={len(dates)}", flush=True)
    arguments = {name: None if parent[name] is None else Path(parent[name]) for name in ("source_root", "licensed_fit_directory", "raw_quote_directory",
        "market_directory", "daily_directory", "dynamics_directory")}
    if multiday:
        return module.run_episodes(**arguments, output_directory=output_directory,
            coverage_path=Path(parent["coverage_path"]), families=tuple(parent["families"]),
            methods=tuple(parent["methods"]), risks=tuple(parent["risks"]), greek_budgets=tuple(parent["greek_budgets"]),
            shard_index=parent["shard_index"], shard_count=parent["shard_count"],
            objective_gap_tolerance=parent["objective_gap_tolerance"])
    front_arguments = ({"front_scores_path": Path(parent["front_scores_path"]),
        "front_surface_law": parent["front_surface_law"]} if parent.get("front_surface_law") is not None else {})
    return module.run_objectives(method=parent["method"], **arguments, output_directory=output_directory,
        dates=tuple(date.fromisoformat(value) for value in dates),
        objective_gap_tolerance=parent["objective_gap_tolerance"], diagnostic_only=parent["diagnostic_only"],
        itm_premium_book=parent.get("itm_premium_book", False), **front_arguments)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--cohort-config", type=Path)
    parser.add_argument("--supplemental-directory", type=Path)
    print(json.dumps(run_snapshot(**vars(parser.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
