"""Overlapping, causally warm-started terminal-risk episodes on observed markets."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import threading
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from cdxcore.uniquehash import label_unique_filename48_8
from filelock import FileLock

from quantdata.biz_calendar import get_business_days
from quantdata.hedge2.audit_terminal_replay import audit_terminal_replay
from quantdata.hedge2.cash import MarketStore
from quantdata.hedge2.claims import make_claims
from quantdata.hedge2.source import SourceArchive
from quantdata.hedge2.terminal_market import TerminalSimulationConfig
from quantdata.hedge2.terminal_policy import TerminalPolicyConfig
from quantdata.hedge2.terminal_replay import load_origin_decision, replay_terminal_episode
from quantdata.hedge2.terminal_train import TerminalTrainConfig
from quantdata.torch import BaseEnvironment


@dataclass(frozen=True, kw_only=True)
class TerminalRule:
    claim: str
    objective: str
    risk_aversion: float
    side: int

    @property
    def filename(self) -> str:
        return label_unique_filename48_8(
            f"{self.claim} {self.objective} lambda={self.risk_aversion:g} side={self.side:+d}")


def agreed_terminal_claims() -> tuple[str, ...]:
    """Return the agreed labels; actual inception strikes are set by each day's surface."""
    return tuple(claim.label for claim in make_claims(initial_spot=1.,
        atm_vols_by_day={tenor: .2 for tenor in (1, 2, 3, 5)}))


def terminal_rules(*, claims: tuple[str, ...],
                   objectives: tuple[str, ...] = ("mean_variance", "power_exponential"),
                   risk_aversions: tuple[float, ...] = (1., 10., 100., 1000., np.inf),
                   sides: tuple[int, ...] = (-1, 1)) -> tuple[TerminalRule, ...]:
    """Enumerate the full requested tensor; infinite aversion applies only to variance."""
    if (not claims or len(set(claims)) != len(claims) or set(claims) - set(agreed_terminal_claims())
            or not objectives or len(set(objectives)) != len(objectives)
            or set(objectives) - {"mean_variance", "power_exponential"}
            or not risk_aversions or len(set(risk_aversions)) != len(risk_aversions)
            or any(value <= 0 or np.isnan(value) for value in risk_aversions)
            or not sides or len(set(sides)) != len(sides) or set(sides) - {-1, 1}):
        raise ValueError("Choose distinct agreed targets, objectives, positive aversions and claim sides")
    rules = tuple(TerminalRule(claim=claim, objective=objective, risk_aversion=float(risk), side=side)
        for claim in claims for objective in objectives for risk in risk_aversions for side in sides
        if objective == "mean_variance" or np.isfinite(risk))
    if not rules:
        raise ValueError("The selected objective/aversion tensor contains no supported cells")
    return rules


def market_data_unavailable(error: Exception, *, source_directories: tuple[Path, ...] = ()) -> bool:
    """Do not turn arbitrary numerical or identity errors into quote exclusions."""
    if isinstance(error, FileNotFoundError):
        if error.filename is not None:
            missing = Path(error.filename).resolve()
            try:
                date.fromisoformat(missing.stem)
            except ValueError:
                return False
            return missing.suffix in (".npz", ".parquet", ".pdq") and any(
                missing.is_relative_to(path.resolve()) for path in source_directories)
        return str(error).startswith("Missing saved all-0.1 model for ")
    return isinstance(error, ValueError) and str(error).startswith((
        "Missing observed mids", "No PM contracts at required co-terminal expiry ",
        "No PM expiry at or after five business days"))


def _atomic_json(path: Path, values: dict[str, object]) -> None:
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(values, indent=2) + "\n")
    temporary.replace(path)


def run_terminal_roll(*, source_configuration: Path, output_directory: Path,
                      start_date: date, end_date: date, claims: tuple[str, ...], env: BaseEnvironment,
                      objectives: tuple[str, ...] = ("mean_variance", "power_exponential"),
                      risk_aversions: tuple[float, ...] = (1., 10., 100., 1000., np.inf),
                      sides: tuple[int, ...] = (-1, 1),
                      train_config: TerminalTrainConfig = TerminalTrainConfig(),
                      simulation_config: TerminalSimulationConfig = TerminalSimulationConfig(),
                      model_config: TerminalPolicyConfig | None = None,
                      diagnostic_only: bool = False,
                      continue_independent_chains: bool = False) -> dict[str, object]:
    """Retain one independent policy chain per target/objective/aversion/side.

    Only the preceding origin's INCEPTION policy initializes a new episode.
    Its later daily policies must never initialize an overlapping earlier
    date. Execution inside each episode reloads every actual day's model
    and carries actual inventory. Independent episodes are never netted.

    Missing markets are explicit exclusions, not model-filled quotes.
    An unfinished production fit or numerical error stops the rollout and
    preserves accepted records. A changed training budget can resume it;
    changes to model, data, economics or implementation cannot mix silently.
    Optionally defer only an unfinished training chain, while allowing
    independent tensor cells to proceed. Later origins of the deferred
    chain never bypass its unfinished fit. Numerical errors still stop.
    """
    if (not date(2023, 1, 1) <= start_date <= end_date <= date(2025, 12, 31)
            or not isinstance(continue_independent_chains, bool)):
        raise ValueError("Rolling origins must be inside 2023--2025 and independent-chain mode must be boolean")
    rules = terminal_rules(claims=claims, objectives=objectives, risk_aversions=risk_aversions, sides=sides)
    resolved = output_directory.resolve()
    if any((parent / ".git").is_file() or (parent / ".git" / "HEAD").is_file()
           for parent in (resolved, *resolved.parents)):
        raise ValueError("Rolling outputs must be outside Git working trees")
    sources_bytes = source_configuration.read_bytes()
    sources = json.loads(sources_bytes)
    daily_directory, dynamics_directory = Path(sources["daily_directory"]), Path(sources["dynamics_directory"])
    source_directories = tuple(Path(sources[name]) for name in (
        "source_root", "market_directory", "licensed_fit_directory", "raw_quote_directory",
        "daily_directory", "dynamics_directory"))
    store = MarketStore(**{name: Path(sources[name]) for name in (
        "market_directory", "licensed_fit_directory", "raw_quote_directory")})
    archive = SourceArchive(root=Path(sources["source_root"]))
    modules = sorted(Path(__file__).parent.glob("*.py"))
    identity = {"source_configuration": str(source_configuration.resolve()),
        "source_configuration_sha256": hashlib.sha256(sources_bytes).hexdigest(),
        "start_date": start_date.isoformat(), "end_date": end_date.isoformat(),
        "rules": [asdict(rule) for rule in rules], "simulation_config": asdict(simulation_config),
        "policy_config": None if model_config is None else asdict(model_config),
        "policy_config_version": TerminalPolicyConfig.version.unique_id48,
        "simulation_version": TerminalSimulationConfig.version.unique_id48,
        "device_type": env.device.type, "precision": str(env.fdtype), "diagnostic_only": diagnostic_only,
        "continue_independent_chains": continue_independent_chains,
        "warm_start": "preceding available inception policy of the same tensor cell; never a later episode fit",
        "selection": "causal frozen inception books; missing later observed quotes are ex-post exclusions",
        "forecasts": "shared immutable arrays across objectives, aversions and sides; actual inventory remains separate",
        "costs": "option entry half-spreads x1; spot zero; no terminal/KO exit fee; independent episodes",
        "implementation_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in modules}}
    identity = json.loads(json.dumps(identity))
    output_directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_directory / "run.lock"), timeout=0):
        configuration_path = output_directory / "config.json"
        if configuration_path.exists() and json.loads(configuration_path.read_text()) != identity:
            raise ValueError("Rolling scope, source or implementation changed; preserve the original run")
        _atomic_json(configuration_path, identity)
        if not (output_directory / "implementation.zip").exists():
            with ZipFile(output_directory / "implementation.partial.zip", "w", compression=ZIP_DEFLATED) as saved:
                for path in modules:
                    saved.write(path, arcname=path.name)
            (output_directory / "implementation.partial.zip").replace(output_directory / "implementation.zip")
        _atomic_json(output_directory / "training_request.json", asdict(train_config))
        stopping = threading.Event()
        handlers = {}
        if threading.current_thread() is threading.main_thread():
            for number in (signal.SIGINT, signal.SIGTERM):
                handlers[number] = signal.signal(number, lambda signum, frame: stopping.set())
        origins = get_business_days(start_date, end_date)
        previous = {rule: None for rule in rules}
        rows, date_exclusions = [], []
        deferred_chains, deferred_rows = {}, []
        started = time.monotonic()
        status = "complete"
        total, processed = len(origins) * len(rules), 0

        def progress() -> dict[str, object]:
            elapsed = time.monotonic() - started
            counts = pd.Series([row["status"] for row in rows], dtype=str).value_counts().to_dict()
            reported_status = "training_incomplete" if status == "complete" and deferred_chains else status
            information = {"status": reported_status, "requested_origins": len(origins), "rules_per_origin": len(rules),
                "processed_cells": processed, "total_cells": total, "cell_status_counts": counts,
                "completed_episodes": int(counts.get("complete", 0)),
                "unavailable_origin_dates": len(date_exclusions),
                "unavailable_date_cells": sum(row["unavailable_rule_cells"] for row in date_exclusions),
                "deferred_training_chains": len(deferred_chains),
                "deferred_later_cells": len(deferred_rows),
                "elapsed_seconds": elapsed, "remaining_seconds_estimate": (
                    elapsed * (total - processed) / processed if processed and not deferred_chains else None),
                "diagnostic_only": diagnostic_only, "independent_episode_costs": True,
                "inference_units": "business-date blocks; overlapping episodes are not independent samples"}
            _atomic_json(output_directory / "progress.json", information)
            return information

        try:
            for origin in origins:
                if stopping.is_set():
                    status = "checkpointed_stop"
                    break
                runnable_rules = []
                for rule in rules:
                    if rule in deferred_chains:
                        deferred_rows.append({**asdict(rule), "origin": origin.isoformat(),
                            "awaiting_origin": deferred_chains[rule].isoformat()})
                    else:
                        runnable_rules.append(rule)
                if not runnable_rules:
                    progress()
                    continue
                unavailable = [str(directory / "days" / f"{origin}.npz")
                    for directory in (daily_directory, dynamics_directory)
                    if not (directory / "days" / f"{origin}.npz").is_file()]
                if unavailable:
                    date_exclusions.append({"origin": origin.isoformat(), "status": "model_data_unavailable",
                        "missing_files": json.dumps(unavailable), "unavailable_rule_cells": len(runnable_rules)})
                    processed += len(runnable_rules)
                    progress()
                    continue
                for rule in runnable_rules:
                    directory = output_directory / "episodes" / origin.isoformat() / rule.filename
                    try:
                        result = replay_terminal_episode(origin=origin, claim_label=rule.claim,
                            store=store, archive=archive, daily_directory=daily_directory,
                            dynamics_directory=dynamics_directory, output_directory=directory,
                            objective=rule.objective, risk_aversion=rule.risk_aversion, target_side=rule.side,
                            env=env, train_config=train_config, simulation_config=simulation_config,
                            model_config=model_config,
                            previous_origin_decision=previous[rule], diagnostic_only=diagnostic_only,
                            stop_requested=stopping.is_set,
                            shared_forecast_directory=output_directory / "forecasts")
                        row = {**asdict(rule), "origin": origin.isoformat(), "directory": str(directory.resolve()),
                            **result.summary, "status": result.status}
                        if result.status == "complete":
                            audit = audit_terminal_replay(directory=directory, store=store)
                            _atomic_json(directory / "cash_audit.json", audit)
                            previous[rule] = load_origin_decision(directory, env=env, daily_directory=daily_directory)
                        elif result.status == "training_incomplete" and continue_independent_chains:
                            deferred_chains[rule] = origin
                        else:
                            status = result.status
                    except Exception as error:
                        row = {**asdict(rule), "origin": origin.isoformat(), "directory": str(directory.resolve()),
                            "status": "market_data_unavailable" if market_data_unavailable(
                                error, source_directories=source_directories) else "review_required",
                            "error_type": type(error).__name__, "exact_error": str(error)}
                        directory.mkdir(parents=True, exist_ok=True)
                        _atomic_json(directory / "status.json", row)
                        if not market_data_unavailable(error, source_directories=source_directories):
                            status = "review_required"
                            rows.append(row)
                            progress()
                            raise
                        episode_path = directory / "episode.json"
                        if episode_path.exists() and json.loads(episode_path.read_text())["origin_decision"] is not None:
                            previous[rule] = load_origin_decision(directory, env=env, daily_directory=daily_directory)
                    _atomic_json(directory / "status.json", row)
                    rows.append(row)
                    processed += int(row["status"] in ("complete", "market_data_unavailable"))
                    information = progress()
                    remaining = information["remaining_seconds_estimate"]
                    print(f"terminal rolling cells={processed}/{total} origin={origin} claim={rule.claim} "
                        f"objective={rule.objective} lambda={rule.risk_aversion:g} side={rule.side:+d} "
                        f"status={row['status']} elapsed_s={information['elapsed_seconds']:.1f} "
                        f"remaining_s~{remaining:.0f}" if remaining is not None else
                        f"terminal rolling origin={origin} status={row['status']} completed_cells={processed}/{total}", flush=True)
                    if status != "complete" or stopping.is_set():
                        break
                if status != "complete" or stopping.is_set():
                    if stopping.is_set() and status == "complete":
                        status = "checkpointed_stop"
                    break
        finally:
            for number, handler in handlers.items():
                signal.signal(number, handler)
            pd.DataFrame(rows).to_parquet(output_directory / "population.parquet", index=False)
            pd.DataFrame(date_exclusions, columns=("origin", "status", "missing_files", "unavailable_rule_cells")).to_parquet(
                output_directory / "date_exclusions.parquet", index=False)
            pd.DataFrame(deferred_rows, columns=("claim", "objective", "risk_aversion", "side",
                "origin", "awaiting_origin")).to_parquet(output_directory / "deferred_cells.parquet", index=False)
            information = progress()
            _atomic_json(output_directory / "summary.json", information)
        return information


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-configuration", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2023, 1, 1))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--claims", nargs="+", choices=agreed_terminal_claims(), default=agreed_terminal_claims())
    parser.add_argument("--objectives", nargs="+", choices=("mean_variance", "power_exponential"),
        default=("mean_variance", "power_exponential"))
    parser.add_argument("--risk-aversions", nargs="+", type=float, default=(1., 10., 100., 1000., np.inf))
    parser.add_argument("--sides", nargs="+", type=int, choices=(-1, 1), default=(-1, 1))
    parser.add_argument("--paths", type=int)
    parser.add_argument("--maximum-updates", type=int, default=2000)
    parser.add_argument("--minimum-updates", type=int, default=100)
    parser.add_argument("--optimizer", choices=("adam", "lbfgs"), default="adam")
    parser.add_argument("--compact-histories", action="store_true")
    parser.add_argument("--actor-coordinates", choices=("none", "diagonal", "whitened"), default="none")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", type=int, choices=(32, 64), default=64)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--fee-preconditioning", action="store_true")
    parser.add_argument("--continue-independent-chains", action="store_true")
    arguments = parser.parse_args()
    result = run_terminal_roll(source_configuration=arguments.source_configuration,
        output_directory=arguments.output_directory, start_date=arguments.start_date, end_date=arguments.end_date,
        claims=tuple(arguments.claims), objectives=tuple(arguments.objectives), risk_aversions=tuple(arguments.risk_aversions),
        sides=tuple(arguments.sides), env=BaseEnvironment(device=arguments.device, precision=arguments.precision, debug=True),
        train_config=TerminalTrainConfig(maximum_updates=arguments.maximum_updates,
            minimum_updates=arguments.minimum_updates, optimizer=arguments.optimizer,
            compact_histories=arguments.compact_histories, actor_coordinates=arguments.actor_coordinates),
        simulation_config=TerminalSimulationConfig(paths=arguments.paths),
        model_config=(TerminalPolicyConfig(label="terminal-policy", feature_count=12, fee_preconditioning=True)
            if arguments.fee_preconditioning else None), diagnostic_only=arguments.diagnostic_only,
        continue_independent_chains=arguments.continue_independent_chains)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
