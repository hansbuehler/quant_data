"""Persist small multi-day baseline/hedge checks before rolling expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from quantdata.hedge2.cash import MarketStore
from quantdata.hedge2.episodes import HedgeRule, run_episode
from quantdata.hedge2.source import SourceArchive


def run_episode_pilot(*, source_root: Path, licensed_fit_directory: Path, raw_quote_directory: Path,
                      market_directory: Path, daily_directory: Path, dynamics_directory: Path,
                      output_directory: Path, dates: tuple[date, ...], families: tuple[str, ...],
                      methods: tuple[str, ...], risks: tuple[float, ...], greek_budgets: tuple[float, ...],
                      atm_only: bool = False) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    rules = []
    names = {"unhedged": "unhedged", "black": "BS delta+vega", "greek": "DLV Greek",
             "parametric": "DLV parametric", "statistical": "DLV statistical"}
    for side in (-1, 1):
        for method in methods:
            if method == "greek":
                rules.extend(HedgeRule(method=names[method], side=side, greek_budget=value) for value in greek_budgets)
            elif method in ("parametric", "statistical"):
                rules.extend(HedgeRule(method=names[method], side=side, objective=objective, risk_aversion=risk)
                    for objective in ("mean_variance", "power_exponential") for risk in risks
                    if objective == "mean_variance" or np.isfinite(risk))
            else:
                rules.append(HedgeRule(method=names[method], side=side))
    configuration = {"source_root": str(source_root), "licensed_fit_directory": str(licensed_fit_directory),
        "raw_quote_directory": str(raw_quote_directory), "market_directory": str(market_directory),
        "daily_directory": str(daily_directory), "dynamics_directory": str(dynamics_directory),
        "dates": [day.isoformat() for day in dates], "families": list(families),
        "atm_only_diagnostic": atm_only,
        "rules": [asdict(rule) for rule in rules], "diagnostic_only": True,
        "objective_semantics": "daily receding one-step hedge; not global terminal-wealth optimization"}
    package = Path(__file__).parent
    sources = sorted(package.glob("*.py"))
    configuration["implementation_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    config_path = output_directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("Episode pilot output belongs to a different experiment or implementation")
    config_path.write_text(json.dumps(configuration, indent=2) + "\n")
    if not (output_directory / "implementation.zip").exists():
        with ZipFile(output_directory / "implementation.zip", "w", compression=ZIP_DEFLATED) as archive:
            for path in sources:
                archive.write(path, arcname=path.name)
    store = MarketStore(market_directory=market_directory, licensed_fit_directory=licensed_fit_directory,
                         raw_quote_directory=raw_quote_directory)
    archive = SourceArchive(root=source_root)
    started = time.monotonic()
    summaries, exclusions = [], []
    for day in dates:
        for family in families:
            directory = output_directory / day.isoformat() / family
            directory.mkdir(parents=True, exist_ok=True)
            try:
                labels = None
                if atm_only:
                    labels = ((f"{family}-put-+0.00", f"{family}-call-+0.00") if family.startswith("digital")
                              else ("ko-3D-5D-put-+0.00",) if family == "ko"
                              else ("ratio-straddle-3D-5D-+0.00",))
                result = run_episode(origin=day, family=family, store=store, archive=archive,
                    daily_directory=daily_directory, dynamics_directory=dynamics_directory, rules=tuple(rules),
                    claim_labels=labels, checkpoint_directory=directory / "steps")
            except (FileNotFoundError, ValueError) as error:
                message = str(error)
                missing = isinstance(error, FileNotFoundError) or message.startswith(("Missing observed mids", "No PM contracts"))
                if not missing:
                    raise
                excluded = {"origin": day.isoformat(), "family": family, "reason": message}
                exclusions.append(excluded)
                (directory / "excluded.json").write_text(json.dumps(excluded, indent=2) + "\n")
                print(f"episode excluded origin={day} family={family}: {message}", flush=True)
                continue
            result.summary.to_parquet(directory / "summary.parquet", index=False)
            result.trades.to_parquet(directory / "trades.parquet", index=False)
            result.initial_book.to_parquet(directory / "initial_book.parquet", index=False)
            np.savez_compressed(directory / "positions.npz",
                **{f"step_{index}": values for index, values in enumerate(result.positions)},
                **{f"active_{index}": values for index, values in enumerate(result.active_indices)})
            (directory / "diagnostics.json").write_text(json.dumps(result.diagnostics, indent=2) + "\n")
            summaries.append(result.summary)
            print(f"episode pilot finished origin={day} family={family} rows={len(result.summary)} "
                  f"elapsed_s={time.monotonic()-started:.1f}", flush=True)
    if summaries:
        pd.concat(summaries, ignore_index=True).to_parquet(output_directory / "results.parquet", index=False)
    summary = {"completed_episodes": len(summaries), "excluded_episodes": exclusions,
        "cells": sum(len(frame) for frame in summaries), "elapsed_seconds": time.monotonic()-started,
        "diagnostic_only": True, "global_multiday_policy_optimization": False}
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "licensed-fit-directory", "raw-quote-directory", "market-directory",
                 "daily-directory", "dynamics-directory", "output-directory"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dates", type=date.fromisoformat, nargs="+", required=True)
    parser.add_argument("--families", nargs="+", choices=("digital-3D", "digital-5D", "ko", "ratio_straddle"),
                        default=("digital-3D", "digital-5D", "ko", "ratio_straddle"))
    parser.add_argument("--methods", nargs="+", choices=("unhedged", "black", "greek", "parametric", "statistical"),
                        default=("unhedged", "black", "greek"))
    parser.add_argument("--risks", type=float, nargs="+", default=(10.,))
    parser.add_argument("--greek-budgets", type=float, nargs="+", default=(1.,))
    parser.add_argument("--atm-only", action="store_true")
    args = vars(parser.parse_args())
    for key in ("dates", "families", "methods", "risks", "greek_budgets"):
        args[key] = tuple(args[key])
    print(json.dumps(run_episode_pilot(**args), indent=2), flush=True)


if __name__ == "__main__":
    main()
