"""Checkpointed rolling hedge episodes on explicitly complete observed books."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from filelock import FileLock

from quantdata.hedge2.cash import MarketStore
from quantdata.hedge2.episode_artifacts import publish_episode, verify_episode_artifacts
from quantdata.hedge2.episodes import EpisodeCheckpointedStop, HedgeRule, run_episode
from quantdata.hedge2.source import SourceArchive


def black_coverage_unavailable(error: ValueError) -> bool:
    """Only specified frozen-book bracketing/vega limitations are baseline exclusions."""
    return str(error).startswith(("Black exotic hedge needs observed strikes surrounding ",
        "The Black vega hedge needs two surrounding observed option strikes",
        "The Black vega hedge has no expiry at or after the target",
        "The bracketing option combination has zero Black vega"))


def episode_rules(*, methods: tuple[str, ...], risks: tuple[float, ...],
                  greek_budgets: tuple[float, ...]) -> tuple[HedgeRule, ...]:
    names = {"unhedged": "unhedged", "black": "BS delta+vega", "greek": "DLV Greek",
             "parametric": "DLV parametric", "statistical": "DLV statistical"}
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(names):
        raise ValueError("Choose distinct supported benchmark methods")
    if not risks or len(set(risks)) != len(risks) or any(value <= 0 or np.isnan(value) for value in risks):
        raise ValueError("Risk aversions must be distinct and positive")
    if not greek_budgets or len(set(greek_budgets)) != len(greek_budgets):
        raise ValueError("Specify distinct Greek budgets")
    rules = []
    for side in (-1, 1):
        for method in methods:
            if method == "greek":
                rules.extend(HedgeRule(method=names[method], side=side, greek_budget=budget) for budget in greek_budgets)
            elif method in ("parametric", "statistical"):
                rules.extend(HedgeRule(method=names[method], side=side, objective=objective, risk_aversion=risk)
                    for objective in ("mean_variance", "power_exponential") for risk in risks
                    if objective == "mean_variance" or np.isfinite(risk))
            else:
                rules.append(HedgeRule(method=names[method], side=side))
    return tuple(rules)


def eligible_episodes(coverage: pd.DataFrame, *, families: tuple[str, ...],
                      shard_index: int, shard_count: int) -> tuple[tuple[date, str], ...]:
    groups = {"digital-1D": "digital-1D", "digital-3D": "digital-3D", "digital-5D": "digital-5D",
              "ko": "exotic-3D-5D", "ratio_straddle": "exotic-3D-5D"}
    if (not families or len(set(families)) != len(families) or set(families) - set(groups)
            or shard_count < 1 or not 0 <= shard_index < shard_count
            or coverage.duplicated(["origin", "group"]).any()):
        raise ValueError("Invalid family coverage or episode shard")
    pairs = []
    for row in coverage.loc[coverage["eligible"]].itertuples():
        origin = date.fromisoformat(row.origin)
        if not date(2023, 1, 1) <= origin <= date(2025, 12, 31):
            raise ValueError("Episode coverage leaves the agreed 2023--2025 OOS scope")
        pairs.extend((origin, family) for family in families if groups[family] == row.group)
    ordered = sorted(pairs)
    selected = tuple(ordered[shard_index::shard_count])
    if not selected:
        raise ValueError("The requested shard contains no eligible episodes")
    return selected


def run_episodes(*, source_root: Path, licensed_fit_directory: Path, raw_quote_directory: Path,
                 market_directory: Path, daily_directory: Path, dynamics_directory: Path,
                 coverage_path: Path, output_directory: Path, families: tuple[str, ...],
                 methods: tuple[str, ...], risks: tuple[float, ...], greek_budgets: tuple[float, ...],
                 shard_index: int = 0, shard_count: int = 1,
                 objective_gap_tolerance: float = 1e-4,
                 option_side: str = "otm", include_co_terminal_forwards: bool = False,
                 reference_premium_cap: float | None = None,
                 optimized_premium_cap: float | None = None) -> dict[str, object]:
    """Hold the output lock before any configuration, archive or checkpoint write."""
    if "statistical" in methods:
        raise ValueError("Multi-day statistical benchmarks require terminal-horizon optimization; this one-step controller is diagnostic only for that method")
    arguments = dict(locals())
    resolved = output_directory.resolve()
    if any((parent / ".git").is_file() or (parent / ".git" / "HEAD").is_file()
           for parent in (resolved, *resolved.parents)):
        raise ValueError("Episode outputs must be outside Git working trees")
    output_directory.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(output_directory / "run.lock"), timeout=0)
    with lock:
        return _run_episodes_locked(**arguments, run_lock=lock)


def _run_episodes_locked(*, source_root: Path, licensed_fit_directory: Path, raw_quote_directory: Path,
                        market_directory: Path, daily_directory: Path, dynamics_directory: Path,
                        coverage_path: Path, output_directory: Path, families: tuple[str, ...],
                        methods: tuple[str, ...], risks: tuple[float, ...], greek_budgets: tuple[float, ...],
                        shard_index: int, shard_count: int, objective_gap_tolerance: float,
                        option_side: str, include_co_terminal_forwards: bool,
                        reference_premium_cap: float | None,
                        optimized_premium_cap: float | None,
                        run_lock: FileLock) -> dict[str, object]:
    rules = episode_rules(methods=methods, risks=risks, greek_budgets=greek_budgets)
    if include_co_terminal_forwards and "greek" in methods:
        raise ValueError("Greek-only hedges exclude synthetic co-terminal forwards; use a separate book")
    if (option_side not in ("otm", "itm") or not isinstance(include_co_terminal_forwards, bool)
            or reference_premium_cap is not None and (not np.isfinite(reference_premium_cap) or reference_premium_cap <= 0)
            or optimized_premium_cap is not None and (not np.isfinite(optimized_premium_cap) or optimized_premium_cap <= 0)):
        raise ValueError("Episode book side, forwards and reference premium threshold must be valid")
    if ((option_side != "otm" or include_co_terminal_forwards or reference_premium_cap is not None)
            and set(methods) - {"unhedged", "black"} and optimized_premium_cap is None):
        raise ValueError("Current ITM/forward optimizers require an explicit current-premium limit")
    coverage_config_path = coverage_path.parent / "config.json"
    if option_side != "otm" or include_co_terminal_forwards:
        coverage_config = json.loads(coverage_config_path.read_text())
        if (coverage_config.get("option_side") != option_side
                or coverage_config.get("include_co_terminal_forwards") != include_co_terminal_forwards
                or any(Path(coverage_config[name]).resolve() != path.resolve() for name, path in (
                    ("source_root", source_root), ("market_directory", market_directory),
                    ("licensed_fit_directory", licensed_fit_directory), ("raw_quote_directory", raw_quote_directory),
                    ("dynamics_directory", dynamics_directory)))):
            raise ValueError("Episode coverage must use the same current book and source population")
    cohort = eligible_episodes(pd.read_parquet(coverage_path), families=families,
                              shard_index=shard_index, shard_count=shard_count)
    configuration = {"experiment_type": "one_day" if families == ("digital-1D",) else "multiday", "source_root": str(source_root),
        "licensed_fit_directory": str(licensed_fit_directory), "raw_quote_directory": str(raw_quote_directory),
        "market_directory": str(market_directory), "daily_directory": str(daily_directory),
        "dynamics_directory": str(dynamics_directory), "coverage_path": str(coverage_path),
        "coverage_sha256": hashlib.sha256(coverage_path.read_bytes()).hexdigest(),
        "coverage_configuration_sha256": (hashlib.sha256(coverage_config_path.read_bytes()).hexdigest()
                                          if coverage_config_path.exists() else None),
        "families": list(families), "methods": list(methods), "risks": list(risks), "greek_budgets": list(greek_budgets),
        "rules": [asdict(rule) for rule in rules], "shard_index": shard_index, "shard_count": shard_count,
        "dates": sorted({day.isoformat() for day, _ in cohort}),
        "episodes": [{"origin": day.isoformat(), "family": family} for day, family in cohort],
        "objective_gap_tolerance": objective_gap_tolerance, "diagnostic_only": False,
        "objective_semantics": "daily receding one-step; not globally optimized terminal wealth",
        "option_side": option_side, "include_co_terminal_forwards": include_co_terminal_forwards,
        "reference_premium_cap": reference_premium_cap,
        "optimized_premium_cap": optimized_premium_cap,
        "black_iv_convention": "same saved OTM prices; relative log-price inversion with cdxcore Black pricing",
        "reference_capacity": "analytic references are uncapped; the premium threshold reports breaches without clipping",
        "spot_cost": 0., "option_half_spread_multiplier": 1., "terminal_exit_charge": 0.}
    output_directory.mkdir(parents=True, exist_ok=True)
    sources = sorted(Path(__file__).parent.glob("*.py"))
    configuration["implementation_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    config_path = output_directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("This output belongs to different inputs, economics, cohort or implementation")
    temporary_config = config_path.with_suffix(".partial.json")
    temporary_config.write_text(json.dumps(configuration, indent=2) + "\n")
    temporary_config.replace(config_path)
    configuration_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if not (output_directory / "implementation.zip").exists():
        temporary_archive = output_directory / "implementation.partial.zip"
        with ZipFile(temporary_archive, "w", compression=ZIP_DEFLATED) as archive:
            for path in sources:
                archive.write(path, arcname=path.name)
        temporary_archive.replace(output_directory / "implementation.zip")
    store = MarketStore(market_directory=market_directory, licensed_fit_directory=licensed_fit_directory,
                         raw_quote_directory=raw_quote_directory)
    archive = SourceArchive(root=source_root)
    started = time.monotonic()
    stopping = False

    def request_stop(signum, frame) -> None:
        nonlocal stopping
        stopping = True
        print(f"stop requested signal={signum}; finish and checkpoint the active daily step", flush=True)

    handlers = {number: signal.signal(number, request_stop) for number in (signal.SIGINT, signal.SIGTERM)}
    completed, exclusions = [], []
    try:
        with run_lock:
            for index, (origin, family) in enumerate(cohort):
                directory = output_directory / origin.isoformat() / family
                directory.mkdir(parents=True, exist_ok=True)
                manifest_path = directory / "completed.json"
                excluded_path = directory / "excluded.json"
                if excluded_path.exists():
                    excluded = json.loads(excluded_path.read_text())
                    if methods != ("black",) or not black_coverage_unavailable(ValueError(excluded["exact_error"])):
                        raise ValueError("An unexpected exclusion needs review before resuming")
                    exclusions.append(excluded)
                    continue
                if manifest_path.exists():
                    verify_episode_artifacts(directory=directory, configuration_sha256=configuration_sha256)
                    completed.append(pd.read_parquet(directory / "summary.parquet"))
                else:
                    try:
                        result = run_episode(origin=origin, family=family, store=store, archive=archive,
                            daily_directory=daily_directory, dynamics_directory=dynamics_directory, rules=rules,
                            checkpoint_directory=directory / "steps", stop_requested=lambda: stopping,
                            objective_gap_tolerance=objective_gap_tolerance, option_side=option_side,
                            include_co_terminal_forwards=include_co_terminal_forwards,
                            reference_premium_cap=reference_premium_cap,
                            optimized_premium_cap=optimized_premium_cap)
                    except ValueError as error:
                        if methods != ("black",) or not black_coverage_unavailable(error):
                            raise
                        excluded = {"origin": origin.isoformat(), "family": family, "method": "BS delta+vega",
                            "reason": "The prescribed fixed-book Black hedge is unavailable; no replacement or extrapolation.",
                            "exact_error": str(error), "not_a_solver_failure": True}
                        temporary = excluded_path.with_suffix(".partial.json")
                        temporary.write_text(json.dumps(excluded, indent=2) + "\n")
                        temporary.replace(excluded_path)
                        exclusions.append(excluded)
                        print(f"Black coverage unavailable episodes={index+1}/{len(cohort)} origin={origin} family={family}: {error}", flush=True)
                        continue
                    publish_episode(directory=directory, result=result, origin=origin.isoformat(),
                        family=family, configuration_sha256=configuration_sha256)
                    completed.append(result.summary)
                elapsed = time.monotonic() - started
                remaining = elapsed / (index + 1) * (len(cohort) - index - 1)
                progress = {"status": "running", "completed_episodes": index + 1, "total_episodes": len(cohort),
                            "last_origin": origin.isoformat(), "last_family": family,
                            "elapsed_seconds": elapsed, "estimated_remaining_seconds": remaining}
                (output_directory / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
                print(json.dumps(progress), flush=True)
                if stopping:
                    raise EpisodeCheckpointedStop("Finished episode before requested stop")
            frame = pd.concat(completed, ignore_index=True) if completed else pd.DataFrame()
            frame.to_parquet(output_directory / "results.parquet", index=False)
            summary = {"status": "complete", "episodes": len(completed), "requested_episodes": len(cohort),
                       "unavailable_episodes": exclusions, "cells": len(frame),
                       "elapsed_seconds": time.monotonic() - started, "diagnostic_only": False,
                       "objective_semantics": configuration["objective_semantics"]}
            (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            return summary
    except EpisodeCheckpointedStop as error:
        result = {"status": "checkpointed_stop", "completed_episodes": len(completed), "reason": str(error)}
        (output_directory / "status.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    except Exception as error:
        (output_directory / "status.json").write_text(json.dumps({"status": "review_required",
            "error_type": type(error).__name__, "exact_error": str(error)}, indent=2) + "\n")
        raise
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "licensed-fit-directory", "raw-quote-directory", "market-directory",
                 "daily-directory", "dynamics-directory", "coverage-path", "output-directory"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--families", nargs="+", default=("digital-3D", "digital-5D", "ko", "ratio_straddle"))
    parser.add_argument("--methods", nargs="+", default=("unhedged", "black", "greek"))
    parser.add_argument("--risks", type=float, nargs="+", default=(1., 10., 100., 1000., np.inf))
    parser.add_argument("--greek-budgets", type=float, nargs="+", default=(0., .2, .4, .6, .8, 1.))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--objective-gap-tolerance", type=float, default=1e-4)
    parser.add_argument("--option-side", choices=("otm", "itm"), default="otm")
    parser.add_argument("--include-co-terminal-forwards", action="store_true")
    parser.add_argument("--reference-premium-cap", type=float)
    parser.add_argument("--optimized-premium-cap", type=float)
    arguments = vars(parser.parse_args())
    for name in ("families", "methods", "risks", "greek_budgets"):
        arguments[name] = tuple(arguments[name])
    print(json.dumps(run_episodes(**arguments), indent=2), flush=True)


if __name__ == "__main__":
    main()
