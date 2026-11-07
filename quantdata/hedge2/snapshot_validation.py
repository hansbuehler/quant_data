"""Validate inherited scenario laws before changing only run scheduling."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np

from quantdata.hedge2.snapshot_run import restore_implementation


def validate_scenario_replay(*, input_directory: Path, output_directory: Path,
                             supplemental_directory: Path) -> dict[str, object]:
    frozen = restore_implementation(input_directory=input_directory, output_directory=output_directory,
                                     supplemental_directory=supplemental_directory)
    package = importlib.import_module("quantdata.hedge2")
    package.__path__ = [str(frozen)]
    module = importlib.import_module("quantdata.hedge2.objective_run")
    if Path(module.__file__).parent != frozen:
        raise RuntimeError("Scenario preflight did not load the verified frozen runner")
    config = json.loads((input_directory / "config.json").read_text())
    store = module.MarketStore(market_directory=Path(config["market_directory"]),
        licensed_fit_directory=Path(config["licensed_fit_directory"]),
        raw_quote_directory=Path(config["raw_quote_directory"]))
    archive = module.SourceArchive(root=Path(config["source_root"]))
    records = []
    for checkpoint in sorted(output_directory.glob("*/checkpoint.npz")):
        day = date.fromisoformat(checkpoint.parent.name)
        current = store.load(day)
        source, _ = archive.load(day)
        state = module.DailyState.load(Path(config["daily_directory"]), day)
        dynamics = module.load_dynamics(Path(config["dynamics_directory"]) / "days" / f"{day.isoformat()}.npz")
        book = module.pd.read_parquet(checkpoint.with_name("book.parquet"))
        claims = tuple(claim for claim in module.make_claims(initial_spot=current.curve.spot,
            atm_vols_by_day={tenor: float(source.interpolate_atm_vols(tenor / module.DAYS_PER_YEAR)) for tenor in (1, 2, 3, 5)})
            if claim.kind == "digital" and claim.terminal_day == 1)
        regenerated = asdict(module.generate_scenarios(state=state, dynamics=dynamics, curve=current.curve,
            book=book, claims=claims, initial_spot=current.curve.spot, seed=day.toordinal()))
        with np.load(checkpoint.with_name("scenarios.npz"), allow_pickle=False) as saved:
            if set(saved.files) != set(regenerated):
                raise ValueError(f"Inherited scenario schema changed on {day}")
            for name, values in regenerated.items():
                if not np.array_equal(saved[name], np.asarray(values)):
                    raise ValueError(f"Inherited scenario law changed on {day}: {name}")
        records.append({"date": day.isoformat(), "all_scenario_arrays_bitwise_equal": True,
                        "original_scenarios_sha256": hashlib.sha256(checkpoint.with_name("scenarios.npz").read_bytes()).hexdigest()})
    summary = {"verified_inherited_dates": len(records), "records": records,
               "primary_implementation": "Parent run's hash-verified archive",
               "supplemental_dependencies": "Explicitly captured and pinned; every inherited scenario array replayed exactly"}
    (output_directory / "scenario_replay_preflight.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
