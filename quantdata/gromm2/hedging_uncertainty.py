"""Frozen-checkpoint OCE Monte Carlo errors with antithetic-pair clustering."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from quantdata.gromm2.hedging_policy import (
    _batch,
    _write_json,
    load_scenarios,
    optimized_oce,
    pair_statistics,
    training_environment,
)
from quantdata.gromm2.joint_hedging import JointOptionsHedge, JointTrainConfig, OptionsPolicyConfig
from quantdata.gromm2.rolling_audit import check_resource_headroom
from quantdata.torch.ffw import FeedForwardConfig


def risk_path_scores(pnl: np.ndarray, *, risk_aversion: float) -> tuple[np.ndarray, float]:
    """Use the OCE envelope derivative; the fitted translation has no first-order term."""
    _, translation, utility = optimized_oce(pnl, risk_aversion)
    return -translation - utility, translation


def payoff_hedge_statistics(payoff: np.ndarray, empty_gain: np.ndarray, claim_gain: np.ndarray, *,
                            risk_aversion: float) -> dict[str, object]:
    """Report mean plus first-order MC SE; incremental risks retain paired covariance."""
    arrays = [np.asarray(values, dtype=np.float64) for values in (payoff, empty_gain, claim_gain)]
    if any(values.ndim != 1 or values.shape != arrays[0].shape for values in arrays):
        raise ValueError("payoff and both frozen policy gains must have matching vector shapes")
    for values in arrays:
        pair_statistics(values)
    payoff, empty_gain, claim_gain = arrays
    empty, empty_translation = risk_path_scores(empty_gain, risk_aversion=risk_aversion)
    claim, claim_translation = risk_path_scores(claim_gain - payoff, risk_aversion=risk_aversion)
    unhedged, unhedged_translation = risk_path_scores(-payoff, risk_aversion=risk_aversion)
    return {
        "expectation": pair_statistics(payoff),
        "DH_unhedged_payoff": pair_statistics(unhedged),
        "DH_empty": pair_statistics(empty),
        "DH_claim": pair_statistics(claim),
        "DH_price": pair_statistics(claim - empty),
        "DH_price_minus_expectation": pair_statistics(claim - empty - payoff),
        "evaluation_translations": {
            "empty": empty_translation, "claim": claim_translation, "unhedged": unhedged_translation,
        },
    }


def evaluate_checkpoint_uncertainty(policy_directory: Path, *, run_directory: Path,
                                    output_directory: Path, device: str = "cpu",
                                    batch_size: int = 4096) -> dict[str, object]:
    """Read the best saved policy without fitting, selecting or changing its parameters."""
    if batch_size <= 0:
        raise ValueError("evaluation batch size must be positive")
    check_resource_headroom()
    checkpoint = torch.load(policy_directory / "checkpoint.pt", map_location="cpu", weights_only=False)
    report = checkpoint["report"]
    if report["preconditioned"]:
        raise ValueError("this uncertainty calculation requires original equally weighted Q paths")
    configuration = dict(report["policy_config"])
    configuration["deep"] = FeedForwardConfig(**configuration["deep"])
    configuration["option_expiry_days"] = tuple(configuration["option_expiry_days"])
    config = OptionsPolicyConfig(**configuration)
    identity = checkpoint["identity"]
    if (identity["code_version"] != config.version.unique_id48
            or identity["user_version"] != config.user_version or identity["config"] != asdict(config)):
        raise ValueError("saved model version or geometry changed; the checkpoint is preserved")
    runtime = JointTrainConfig(device=device, precision=report["train_config"]["precision"])
    torch.set_num_threads(runtime.cpu_threads)
    env = training_environment(runtime)
    model = JointOptionsHedge(env, config=config,
        normalization=(np.zeros(config.feature_count), np.ones(config.feature_count)))
    model.load_state_dict(checkpoint["best_state"])
    model.eval()
    output_directory.mkdir(parents=True, exist_ok=False)
    results = {}
    started = time.monotonic()
    for split in ("train", "validation"):
        check_resource_headroom()
        print(json.dumps({"event": "uncertainty_loading", "split": split,
                          "elapsed_seconds": time.monotonic() - started}), flush=True)
        data = load_scenarios(run_directory / f"{split}.h5", expected_split=split)
        key = "training_scenarios_sha256" if split == "train" else "validation_scenarios_sha256"
        if data.sha256 != identity[key]:
            raise ValueError(f"{split} scenario identity differs from the fitted checkpoint")
        gains = {name: np.empty(len(data.payoff), dtype=np.float64) for name in ("empty", "claim")}
        last_report = time.monotonic()
        with torch.no_grad():
            for start in range(0, len(data.payoff), batch_size):
                features, prices, _ = _batch(data, slice(start, start + batch_size), env=env)
                output = model(features, prices)
                for name in gains:
                    gains[name][start:start + len(features)] = getattr(output, name).trading_gain.cpu().numpy()
                completed = min(start + batch_size, len(data.payoff))
                now = time.monotonic()
                if now - last_report >= 60.0 or completed == len(data.payoff):
                    check_resource_headroom()
                    print(json.dumps({"event": "uncertainty_progress", "split": split,
                        "completed_paths": completed, "total_paths": len(data.payoff),
                        "elapsed_seconds": now - started}), flush=True)
                    last_report = now
        results[split] = payoff_hedge_statistics(data.payoff, gains["empty"], gains["claim"],
                                                risk_aversion=config.risk_aversion)
        np.savez_compressed(output_directory / f"{split}_pnl.npz", payoff=data.payoff,
                            empty_gain=gains["empty"], claim_gain=gains["claim"], scenario_sha256=data.sha256)
        del data
    result = {"schema": "gromm2-frozen-hedging-uncertainty-v1", "units": "inception spot",
        "policy_directory": str(policy_directory), "risk_aversion": config.risk_aversion,
        "best_epoch": report["best_epoch"], "device": str(env.device),
        "method": "OCE envelope/influence-function SE; sample SD of adjacent antithetic-pair scores divided by sqrt(pair count).",
        "caveat": "Plug-in Monte Carlo errors with fitted policies held fixed; exclude training and validation-selection uncertainty.",
        "splits": results}
    _write_json(output_directory / "uncertainty_report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-directory", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    arguments = parser.parse_args()
    result = evaluate_checkpoint_uncertainty(arguments.policy_directory, run_directory=arguments.run_directory,
        output_directory=arguments.output_directory, device=arguments.device, batch_size=arguments.batch_size)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
