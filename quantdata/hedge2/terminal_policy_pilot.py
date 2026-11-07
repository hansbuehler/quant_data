"""Dated terminal-policy diagnostics on immutable simulated market paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, fields
from datetime import date
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from filelock import FileLock

from quantdata.hedge2.claims import Claim
from quantdata.hedge2.terminal_market import TerminalMarketPaths, TerminalSimulationConfig
from quantdata.hedge2.terminal_policy import TerminalPolicy, TerminalPolicyConfig, policy_book
from quantdata.hedge2.terminal_train import (
    TerminalTrainConfig,
    audit_fitted_policy,
    subset_terminal_data,
    train_terminal_policy,
)
from quantdata.torch import BaseEnvironment


def load_simulation_group(directory: Path) -> tuple[TerminalMarketPaths, pd.DataFrame, tuple[Claim, ...]]:
    """Verify completion hashes and source inputs before interpreting saved arrays.

    Loading does not regenerate paths or fit a model. The frozen pricing,
    contracts and simulation implementation must match the current reader.
    Other unrelated project files may have changed since path generation.
    """
    marker = json.loads((directory / "completed.json").read_text())
    required = {"paths.npz", "book.parquet", "claims.json", "diagnostics.json"}
    if set(marker["artifacts_sha256"]) != required:
        raise ValueError("A saved terminal simulation needs all four completed artifacts")
    for name, digest in marker["artifacts_sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Completed terminal simulation artifact changed: {name}")
    configuration = json.loads((directory.parent / "config.json").read_text())
    if configuration["simulation_version"] != TerminalSimulationConfig.version.unique_id48:
        raise ValueError("Saved terminal simulation version is not supported")
    for name in ("terminal_market.py", "terminal.py", "claims.py", "cash.py", "pricing.py", "batch_pricing.py"):
        digest = hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        if digest != configuration["implementation_sha256"][name]:
            raise ValueError(f"Saved terminal simulation requires its original implementation: {name}")
    source_path = Path(configuration["source_configuration"])
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != configuration["source_configuration_sha256"]:
        raise ValueError("The saved terminal simulation's source configuration changed")
    sources, day = json.loads(source_path.read_text()), configuration["day"]
    inputs = {"daily_config": Path(sources["daily_directory"]) / "config.json",
        "daily_state": Path(sources["daily_directory"]) / "days" / f"{day}.npz",
        "dynamics": Path(sources["dynamics_directory"]) / "days" / f"{day}.npz",
        "licensed_quotes": Path(sources["market_directory"]) / "days" / f"{day}.parquet",
        "licensed_fit": Path(sources["licensed_fit_directory"]) / "SPX" / f"{day}.pdq",
        "licensed_raw_quotes": Path(sources["raw_quote_directory"]) / "SPX" / f"{day}.pdq"}
    for name, path in inputs.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != configuration["inputs_sha256"][name]:
            raise ValueError(f"Saved terminal simulation source input changed: {name}")
    metadata = configuration["source_model"]
    if hashlib.sha256(Path(metadata["source_model"]).read_bytes()).hexdigest() != metadata["source_sha256"]:
        raise ValueError("The original saved SANOS interpolation changed")
    with np.load(directory / "paths.npz", allow_pickle=False) as saved:
        expected = {field.name for field in fields(TerminalMarketPaths)}
        if set(saved.files) != expected:
            raise ValueError("Saved terminal simulation array schema changed")
        values = {name: saved[name].copy() for name in saved.files}
    for name in ("model_date", "seed", "elapsed_day", "future_spread_convention", "alpha_convention"):
        values[name] = values[name].item()
    values["claim_labels"] = tuple(values["claim_labels"].tolist())
    result = TerminalMarketPaths(**values)
    claims = tuple(Claim(**entry) for entry in json.loads((directory / "claims.json").read_text())["claims"])
    if result.model_date != day or tuple(claim.label for claim in claims) != result.claim_labels:
        raise ValueError("Saved terminal simulation dates or target labels do not align")
    for index, claim in enumerate(claims):
        result.for_claim(claim, index)
    return result, pd.read_parquet(directory / "book.parquet"), claims


def run_policy_pilot(*, simulation_directory: Path, claim_label: str, output_directory: Path,
                     training_paths: int, holdout_paths: int, objective: str, risk_aversion: float,
                     target_side: int, train_config: TerminalTrainConfig = TerminalTrainConfig(),
                     split_seed: int = 1, precision: int = 64, device: str = "cpu") -> dict[str, object]:
    """Fit only on the declared training subset; evaluate the frozen fit afterward.

    Holdout paths never influence normalization, fitting, stopping or best
    checkpoint selection. They are independent complete iid simulations,
    not individual daily observations. These are model-law diagnostics,
    not realized out-of-sample market hedging results. A sub-10,000 training
    run is explicitly a small pilot, not the requested production sample.
    """
    resolved = output_directory.resolve()
    if any((parent / ".git").is_file() or (parent / ".git" / "HEAD").is_file()
           for parent in (resolved, *resolved.parents)):
        raise ValueError("Terminal policy pilot outputs must be outside Git working trees")
    result, contracts, claims = load_simulation_group(simulation_directory)
    if result.elapsed_day != 0:
        raise ValueError("This zero-inventory pilot is inception-only; a daily restart must carry actual inventory")
    if (not isinstance(training_paths, int) or not isinstance(holdout_paths, int)
            or min(training_paths, holdout_paths) < 2 or training_paths + holdout_paths > len(result.weights)):
        raise ValueError("Training and untouched holdout need disjoint nontrivial complete-path subsets")
    try:
        claim_index = result.claim_labels.index(claim_label)
    except ValueError as error:
        raise ValueError(f"Target is absent from this frozen book: {claim_label}") from error
    paths = result.for_claim(claims[claim_index], claim_index)
    book = policy_book(book=contracts, root_spot=float(result.spots[0, 0]), prices=result.instrument_prices[:, :-1])
    order = np.random.default_rng(split_seed).permutation(len(paths.weights))
    training_indices = order[:training_paths]
    holdout_indices = order[training_paths:training_paths + holdout_paths]
    training, training_book = subset_terminal_data(paths, book, training_indices)
    holdout, holdout_book = subset_terminal_data(paths, book, holdout_indices)
    if device not in ("cpu", "cuda"):
        raise ValueError("Choose CPU or an externally gpuctl-allocated CUDA device")
    env = BaseEnvironment(device=device, precision=precision, debug=True)
    model_config = TerminalPolicyConfig(label="terminal-policy", feature_count=paths.features.shape[2])
    initial_inventory = np.zeros(paths.gains.shape[2])
    implementation = sorted(Path(__file__).parent.glob("*.py"))
    identity = {"simulation_directory": str(simulation_directory.resolve()),
        "simulation_completion_sha256": hashlib.sha256((simulation_directory / "completed.json").read_bytes()).hexdigest(),
        "simulation_configuration_sha256": hashlib.sha256((simulation_directory.parent / "config.json").read_bytes()).hexdigest(),
        "claim": claim_label, "objective": objective, "risk_aversion": risk_aversion, "target_side": target_side,
        "training_paths": training_paths, "holdout_paths": holdout_paths, "split_seed": split_seed,
        "model_config": asdict(model_config), "precision": precision, "device": device,
        "small_sample_diagnostic": training_paths < max(10000, 10 * paths.gains.shape[2]),
        "realized_OOS_results": False,
        "implementation_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in implementation}}
    output_directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_directory / "run.lock"), timeout=0):
        configuration_path = output_directory / "config.json"
        if configuration_path.exists() and json.loads(configuration_path.read_text()) != identity:
            raise ValueError("Terminal policy pilot output belongs to a different scope or implementation")
        configuration_path.write_text(json.dumps(identity, indent=2) + "\n")
        if not (output_directory / "implementation.zip").exists():
            with ZipFile(output_directory / "implementation.zip", "w", compression=ZIP_DEFLATED) as saved:
                for path in implementation:
                    saved.write(path, arcname=path.name)
        np.savez_compressed(output_directory / "split.npz", training=training_indices, holdout=holdout_indices)
        fit = train_terminal_policy(paths=training, book=training_book, initial_inventory=initial_inventory,
            model_config=model_config, train_config=train_config, env=env, objective=objective,
            risk_aversion=risk_aversion, target_side=target_side, as_of_date=date.fromisoformat(result.model_date),
            target_key=claim_label, output_directory=output_directory, progress=lambda message: print(message, flush=True))
        if fit.status == "checkpointed_stop":
            return {"status": fit.status, "updates": fit.updates, "checkpoint": str(fit.checkpoint_path)}
        zero = TerminalPolicy(env=env, config=model_config)
        metrics = {"status": fit.status, "updates": fit.updates, "initial_loss": fit.initial_loss,
            "best_training_loss": fit.best_training_loss, "checkpoint": str(fit.checkpoint_path),
            "training_config": asdict(train_config), "small_sample_diagnostic": identity["small_sample_diagnostic"],
            "realized_OOS_results": False}
        for label, selected_paths, selected_book in (("training", training, training_book), ("holdout", holdout, holdout_book)):
            for method, actor in (("fitted", fit.policy), ("unhedged", zero)):
                metrics[f"{label}_{method}"] = audit_fitted_policy(policy=actor, paths=selected_paths,
                    book=selected_book, initial_inventory=initial_inventory, objective=objective,
                    risk_aversion=risk_aversion, target_side=target_side)
        (output_directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2), flush=True)
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-directory", type=Path, required=True)
    parser.add_argument("--claim", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-paths", type=int, default=256)
    parser.add_argument("--holdout-paths", type=int, default=256)
    parser.add_argument("--objective", choices=("mean_variance", "power_exponential"), default="mean_variance")
    parser.add_argument("--risk-aversion", type=float, default=10.)
    parser.add_argument("--side", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--maximum-updates", type=int, default=2000)
    parser.add_argument("--minimum-updates", type=int, default=100)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--precision", type=int, choices=(32, 64), default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments = parser.parse_args()
    run_policy_pilot(simulation_directory=arguments.simulation_directory, claim_label=arguments.claim,
        output_directory=arguments.output, training_paths=arguments.training_paths, holdout_paths=arguments.holdout_paths,
        objective=arguments.objective, risk_aversion=arguments.risk_aversion, target_side=arguments.side,
        train_config=TerminalTrainConfig(maximum_updates=arguments.maximum_updates, minimum_updates=arguments.minimum_updates),
        split_seed=arguments.split_seed, precision=arguments.precision, device=arguments.device)


if __name__ == "__main__":
    main()
