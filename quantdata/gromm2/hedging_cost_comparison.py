"""Remove transaction costs from frozen market hedges without fitting a new policy."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import signal
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from quantdata.gromm2.hedging_policy import _Progress, _write_json, pair_statistics
from quantdata.gromm2.market_batching import MarketBatchIterator
from quantdata.gromm2.market_hedging import (
    MarketBookConfig,
    MarketBookPolicy,
    SpotCohortBookConfig,
    SpotCohortBookPolicy,
    UnitTurnoverBookConfig,
    UnitTurnoverBookPolicy,
    load_market_scenarios,
)
from quantdata.gromm2.portfolio_hedging import (
    LIABILITIES,
    IndependentHedges,
    portfolio_quote_statistics,
)
from quantdata.gromm2.rolling_audit import check_resource_headroom
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def transaction_cost_quotes(payoff: np.ndarray, gains: dict[str, np.ndarray],
                            costs: dict[str, np.ndarray], *, risk_aversion: float) -> dict[str, object]:
    """Add each path's actual costs back before recomputing its OCE and paired errors."""
    payoff = np.asarray(payoff, dtype=np.float64)
    if set(gains) != set(LIABILITIES) or set(costs) != set(LIABILITIES):
        raise ValueError("all three frozen portfolios require their own gains and costs")
    gross = {}
    for name in LIABILITIES:
        gain, cost = np.asarray(gains[name], dtype=np.float64), np.asarray(costs[name], dtype=np.float64)
        if gain.shape != payoff.shape or cost.shape != payoff.shape:
            raise ValueError("costs and gains must match the payoff path vector")
        if not np.all(np.isfinite(cost)) or np.any(cost < 0.0):
            raise ValueError("removed transaction costs must be finite and nonnegative")
        gross[name] = gain + cost
    return {
        "with_costs": portfolio_quote_statistics(payoff, gains, risk_aversion=risk_aversion),
        "costs_removed": portfolio_quote_statistics(payoff, gross, risk_aversion=risk_aversion),
        "transaction_costs": {name: pair_statistics(value) for name, value in costs.items()},
    }


def frozen_policy_fingerprint(checkpoint: dict) -> str:
    """Identify selected actions and source populations, excluding optimizer progress.

    Precision and the simulated background case affect valuation even when the
    selected weights are unchanged. Epoch and optimizer metadata do not.
    """
    payload = {"identity": checkpoint["identity"],
        "precision": checkpoint["report"]["train_config"]["precision"],
        "simulation_case": checkpoint["report"]["scenario_model"]["simulation_case"]}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode())
    for name in sorted(LIABILITIES):
        state = checkpoint["state"]["policies"][name]["best_state"]
        if not state:
            raise ValueError("selected portfolio state must not be empty")
        for key, value in sorted(state.items()):
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                raise ValueError(f"selected portfolio {name}.{key} must be a finite tensor")
            value = value.detach().cpu().contiguous()
            digest.update(json.dumps([name, key, str(value.dtype), list(value.shape)]).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def frozen_market_model(checkpoint: dict) -> IndependentHedges:
    """Load the previously selected weights and normalization, on CPU only."""
    values = dict(checkpoint["identity"]["config"])
    values["deep"] = FeedForwardConfig(**values["deep"])
    unit_costs = "turnover_cost_basis" in values
    config_class = UnitTurnoverBookConfig if unit_costs else MarketBookConfig
    policy_class = UnitTurnoverBookPolicy if unit_costs else MarketBookPolicy
    if values.get("spot_in_cohort_book", False):
        config_class, policy_class = SpotCohortBookConfig, SpotCohortBookPolicy
    if values.get("kind") == "combined":
        from quantdata.gromm2.combined_hedging import CombinedBookConfig, CombinedBookPolicy

        config_class, policy_class = CombinedBookConfig, CombinedBookPolicy
    config = config_class(**values)
    identity = checkpoint["identity"]
    if (identity["config"] != asdict(config) or identity["code_version"] != config.version.unique_id48
            or identity["user_version"] != config.user_version):
        raise ValueError("frozen market policy version or geometry changed; checkpoint preserved")
    env = BaseEnvironment(device="cpu", precision=checkpoint["report"]["train_config"]["precision"], debug=False)
    model = IndependentHedges(env, config=config,
        normalization=(np.zeros(config.feature_count), np.ones(config.feature_count)), policy_class=policy_class)
    for name, policy in model.policies.items():
        policy.load_state_dict(checkpoint["state"]["policies"][name]["best_state"])
    model.eval()
    return model


def evaluate_frozen_costs(policy_directory: Path, *, run_directory: Path, output_directory: Path,
                          batch_size: int = 4096, cpu_threads: int = 4,
                          splits: tuple[str, ...] = ("validation", "train"),
                          resume: bool = False) -> dict[str, object]:
    """Evaluate frozen net/gross gains, optionally reusing verified completed splits.

    Resuming requires identical checkpoint bytes and source populations. An
    interrupted split is recomputed; a committed split is checked by its archive
    hash and never evaluated again. No training state or normalization is changed.
    """
    if (batch_size <= 0 or cpu_threads <= 0 or not splits or len(set(splits)) != len(splits)
            or any(split not in {"train", "validation"} for split in splits)):
        raise ValueError("evaluation requires positive batch/thread counts and valid splits")
    check_resource_headroom()
    output_directory.mkdir(parents=True, exist_ok=resume)
    with FileLock(str(output_directory) + ".lock", timeout=0):
        return _evaluate_frozen_costs(policy_directory, run_directory=run_directory,
            output_directory=output_directory, batch_size=batch_size, cpu_threads=cpu_threads, splits=splits)


def _archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evaluate_frozen_costs(policy_directory: Path, *, run_directory: Path, output_directory: Path,
                           batch_size: int, cpu_threads: int, splits: tuple[str, ...]) -> dict[str, object]:
    torch.set_num_threads(cpu_threads)
    serialized = (policy_directory / "checkpoint.pt").read_bytes()
    checkpoint = torch.load(io.BytesIO(serialized), map_location="cpu", weights_only=False)
    model = frozen_market_model(checkpoint)
    case = checkpoint["report"]["scenario_model"]["simulation_case"]
    result = {
        "schema": "gromm2-frozen-cost-comparison-v1", "complete": False,
        "policy_directory": str(policy_directory), "checkpoint_sha256": hashlib.sha256(serialized).hexdigest(),
        "case": case, "kind": model.config.kind, "risk_aversion": model.config.risk_aversion,
        "turnover_cost_basis": getattr(model.config, "turnover_cost_basis", "legacy_spot_notional"),
        "checkpoint_epoch": checkpoint["state"]["epoch"],
        "best_epochs": {name: record["best_epoch"] for name, record in checkpoint["state"]["policies"].items()},
        "training_stop_reason": checkpoint["report"]["stop_reason"],
        "method": "Identical frozen holdings and path weights; add actual per-path costs back, then recompute OCE translations. No retraining or checkpoint reselection.",
        "caveat": "Cost-removed evaluation of cost-trained policies, not a newly optimized zero-cost hedge. MC errors exclude fitting and validation-selection uncertainty.",
        "units": "inception spot", "device": "cpu", "splits": {}, "artifacts": {},
        "requested_splits": list(splits),
    }
    report_path = output_directory / "cost_comparison.json"
    frozen_path = output_directory / "frozen_checkpoint.pt"
    if frozen_path.exists():
        if _archive_sha256(frozen_path) != result["checkpoint_sha256"]:
            raise ValueError("frozen cost checkpoint changed; preserve the earlier evaluation")
    elif report_path.exists():
        raise ValueError("frozen cost report has no checkpoint snapshot; preserve the earlier evaluation")
    else:
        temporary = frozen_path.with_suffix(".partial")
        temporary.write_bytes(serialized)
        temporary.replace(frozen_path)
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        for key in ("schema", "checkpoint_sha256", "policy_directory", "case", "kind", "best_epochs"):
            if previous.get(key) != result[key]:
                raise ValueError(f"frozen cost {key} changed; preserve the earlier evaluation")
        result = previous
        result["requested_splits"] = list(dict.fromkeys((*previous.get("requested_splits", previous["splits"]), *splits)))
    result["complete"] = False
    _write_json(report_path, result)
    progress = _Progress(label=f"cost-comparison-{case}-{model.config.kind}", interval=60.0)
    handlers = {sig: signal.signal(sig, lambda *_: progress.stop.set()) for sig in (signal.SIGTERM, signal.SIGINT)}
    progress.thread.start()
    try:
        for split in result["requested_splits"]:
            progress.state.update(stage=f"loading-{split}")
            if model.config.kind == "combined":
                from quantdata.gromm2.combined_hedging import load_combined_scenarios

                data, _ = load_combined_scenarios(run_directory / f"{split}.h5", split=split, case=case)
            else:
                data, _ = load_market_scenarios(run_directory / f"{split}.h5", split=split, case=case, kind=model.config.kind,
                    with_spot=getattr(model.config, "spot_in_cohort_book", False))
            source_key = "training_scenarios_sha256" if split == "train" else "validation_scenarios_sha256"
            if data.sha256 != checkpoint["identity"][source_key]:
                raise ValueError("evaluation scenarios differ from the fitted checkpoint")
            archive = output_directory / f"{split}_frozen_gains.npz"
            if split in result["splits"]:
                artifact = result.get("artifacts", {}).get(split, {})
                if (artifact.get("source_sha256") != data.sha256 or artifact.get("paths") != len(data.payoff)
                        or not archive.exists() or artifact.get("sha256") != _archive_sha256(archive)):
                    raise ValueError(f"frozen cost {split} archive is missing or changed; preserve the earlier evaluation")
                progress.emit(reused_completed_split=split)
                continue
            gains = {name: np.empty(len(data.payoff), dtype=np.float64) for name in LIABILITIES}
            costs = {name: np.empty(len(data.payoff), dtype=np.float64) for name in LIABILITIES}
            progress.state.update(stage=f"evaluating-{split}", completed=0, total=len(data.payoff))
            last_resource_check = time.monotonic()
            with torch.inference_mode(), MarketBatchIterator(data, env=model.env, batch_size=batch_size) as batches:
                for batch_number, (features, prices, _) in enumerate(batches):
                    if progress.stop.is_set():
                        raise InterruptedError("cost evaluation stopped; training checkpoint remains untouched")
                    if time.monotonic() - last_resource_check >= 300.0:
                        check_resource_headroom()
                        last_resource_check = time.monotonic()
                    start = batch_number * batch_size
                    for name, policy in model.policies.items():
                        output = policy(features, prices)
                        stop = start + len(features)
                        gains[name][start:stop] = output.trading_gain.numpy()
                        costs[name][start:stop] = output.transaction_cost.numpy()
                    progress.state["completed"] = stop
            progress.state["stage"] = f"oce-and-paired-errors-{split}"
            result["splits"][split] = transaction_cost_quotes(data.payoff, gains, costs, risk_aversion=model.config.risk_aversion)
            temporary = archive.with_suffix(".partial.npz")
            np.savez_compressed(temporary, payoff=data.payoff,
                **{f"{name}_gain": value for name, value in gains.items()},
                **{f"{name}_cost": value for name, value in costs.items()})
            temporary.replace(archive)
            result["artifacts"][split] = dict(sha256=_archive_sha256(archive), source_sha256=data.sha256, paths=len(data.payoff))
            _write_json(report_path, result)
            progress.emit(completed_split=split, best_epochs=result["best_epochs"])
        result["complete"] = True
        _write_json(report_path, result)
        return result
    finally:
        progress.closed.set()
        progress.thread.join(timeout=2.0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def follow_final_hedges(run_directory: Path, *, output_directory: Path, batch_size: int, cpu_threads: int) -> None:
    """Wait quietly for each finalized training result, then evaluate it once on CPU."""
    completed = []
    started = time.monotonic()
    while len(completed) < 4:
        check_resource_headroom()
        for case in ("black0", "black5"):
            for kind in ("factors", "cohorts"):
                name = f"{case}-{kind}"
                if name in completed:
                    continue
                policies = list((run_directory / "hedges" / case / kind).glob("*/quotes.json"))
                if len(policies) > 1:
                    raise ValueError("multiple finalized policies found for one hedge book")
                if policies:
                    target = output_directory / name
                    evaluate_frozen_costs(policies[0].parent, run_directory=run_directory, output_directory=target,
                        batch_size=batch_size, cpu_threads=cpu_threads, resume=True)
                    completed.append(name)
        print(json.dumps(dict(event="waiting-for-final-hedges", elapsed_seconds=time.monotonic() - started,
            completed=completed, total=4)), flush=True)
        if len(completed) < 4:
            time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--policy-directory", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--follow-finals", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.policy_directory is None and not arguments.follow_finals:
        parser.error("a policy directory or --follow-finals is required")
    if arguments.policy_directory is not None:
        evaluate_frozen_costs(arguments.policy_directory, run_directory=arguments.run_directory,
            output_directory=arguments.output_directory / "current", batch_size=arguments.batch_size,
            cpu_threads=arguments.cpu_threads, resume=arguments.resume)
    if arguments.follow_finals:
        follow_final_hedges(arguments.run_directory, output_directory=arguments.output_directory / "final",
            batch_size=arguments.batch_size, cpu_threads=arguments.cpu_threads)


if __name__ == "__main__":
    main()
