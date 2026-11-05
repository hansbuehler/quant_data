"""Train independent empty, long and short LSTM hedges on fixed Q scenarios."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from scipy.optimize import brentq

from quantdata.sdlv.daily.wide_audit import AuditMonitor, resource_health, sha256
from quantdata.sdlv.dhgromm1.contracts import SCENARIO_SCHEMA, canonical_gains, scenario_identity
from quantdata.sdlv.dhgromm1.hedge import (
    HedgeConfig,
    RecurrentHedge,
    indifference_quotes,
    oce_value,
)
from quantdata.torch import BaseEnvironment


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    """Operational optimization settings, not part of the model identity."""

    epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 1e-3
    patience: int = 20
    seed: int = 20261102
    gradient_clip: float = 10.0
    resident_data: bool = True
    compile_model: bool = False

    def __post_init__(self) -> None:
        if (self.epochs < 1 or self.batch_size < 2 or self.batch_size % 2 or self.patience < 0
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0):
            raise ValueError("invalid training configuration; batches must preserve antithetic pairs")


def load_checkpoint(path: Path, *, config: HedgeConfig) -> dict:
    """Validate a trusted project checkpoint before reconstructing its model."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved.get("model_code_version") != HedgeConfig.version.unique_id48
            or saved.get("user_version") != config.user_version):
        raise ValueError(f"checkpoint model version differs; cannot continue {path}")
    if saved.get("model_config") != asdict(config):
        raise ValueError(f"checkpoint geometry or objective differs: {path}")
    return saved


def load_frozen_policy(path: Path, *, config: HedgeConfig, device: str) -> RecurrentHedge:
    """Load a selected policy without fitting it or touching its optimizer."""
    saved = load_checkpoint(path, config=config)
    state = saved["best_state"]
    policy = RecurrentHedge(BaseEnvironment(device=device, precision=32), config=config,
                            price_mean=state["price_mean"], price_scale=state["price_scale"])
    policy.load_state_dict(state)
    policy.requires_grad_(False)
    policy.fit_summary = dict(best_training_loss=saved["best_loss"], selected_epoch=saved["selected_epoch"],
                              epochs_run=saved["epoch"] + 1, additional_epochs_run=0, frozen=True)
    return policy


def read_sample(directory: Path, *, paths: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load complete paths, explicitly converting identified legacy return units."""
    identity = scenario_identity(directory)
    prices, returns, payoffs = [], [], []
    for path in sorted(directory.glob("[0-9]*.npz")):
        with np.load(path, allow_pickle=False) as saved:
            price, ret, payoff = saved["prices"], saved["returns"], saved["payoff"]
        if (price.ndim != 3 or price.shape[1] != 21 or price.shape[2] not in (17, 21)
                or ret.shape != (len(price), 20, price.shape[2])
                or payoff.shape != (len(price),) or len(price) % 2):
            raise ValueError(f"invalid pair-preserving scenario chunk: {path}")
        if not np.isfinite(price).all() or not np.isfinite(ret).all() or not np.isfinite(payoff).all():
            raise FloatingPointError(f"non-finite scenario chunk: {path}")
        prices.append(price)
        returns.append(canonical_gains(price, ret, identity=identity))
        payoffs.append(payoff)
    if not prices or sum(map(len, prices)) != paths:
        raise ValueError(f"expected {paths} complete paths in {directory}")
    return (np.concatenate(prices).astype(np.float32), np.concatenate(returns).astype(np.float32),
            np.concatenate(payoffs).astype(np.float64))


def optimize_translation(pnl: np.ndarray, risk_aversion: float) -> float:
    """Find the OCE translation for a fixed P&L without fitting a hedge."""
    values = np.asarray(pnl, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("P&L must be a nonempty finite vector")

    def first_order(m: float) -> float:
        centered = values - m
        derivative = np.where(centered >= 0, np.exp(-risk_aversion * np.maximum(centered, 0)),
                              1.0 - risk_aversion * np.minimum(centered, 0))
        return float(np.mean(derivative, dtype=np.float64) - 1.0)

    if np.ptp(values) == 0:
        return float(values[0])
    return float(brentq(first_order, float(values.min()), float(values.max()), xtol=1e-14))


def pair_summary(values: np.ndarray) -> dict[str, float | int]:
    """Estimate a mean and MC error using antithetic pair averages."""
    observations = np.asarray(values, dtype=np.float64)
    if observations.ndim != 1 or len(observations) < 4 or len(observations) % 2 or not np.isfinite(observations).all():
        raise ValueError("expected finite observations in complete antithetic pairs")
    pairs = observations.reshape(-1, 2).mean(axis=1)
    mean = float(pairs.mean())
    se = float(pairs.std(ddof=1) / np.sqrt(len(pairs)))
    return dict(mean=mean, standard_error=se, ci95_lower=mean - 1.96 * se,
                ci95_upper=mean + 1.96 * se, paths=len(observations), independent_pairs=len(pairs))


def _economic_utility(pnl: np.ndarray, translation: float, risk_aversion: float) -> np.ndarray:
    centered = np.asarray(pnl, dtype=np.float64) - translation
    return translation + np.where(centered >= 0,
        -np.expm1(-risk_aversion * np.maximum(centered, 0)) / risk_aversion,
        np.minimum(centered, 0) - 0.5 * risk_aversion * np.minimum(centered, 0) ** 2)


def day10_straddle_benchmark(prices: np.ndarray, returns: np.ndarray,
                            payoff: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Buy the normalized terminal straddle at its day-10 spot fixing.

    Use two ATM calls and a short co-terminal zero-strike call. In older
    17-instrument scenarios the terminal spot trade is the available substitute.
    """
    spot = prices[:, 10, 0].astype(np.float64)
    call_price = 2 * prices[:, 10, 1].astype(np.float64) / spot
    forward_index = 17 if prices.shape[2] == 21 else 0
    forward_price = prices[:, 10, forward_index].astype(np.float64) / spot
    gain = 2 * returns[:, 10, 1] - returns[:, 10, forward_index]
    premium = call_price - forward_price + 1
    penalty = np.full(len(spot), 3e-4 + 5e-8)
    error = float(np.max(np.abs(gain - payoff + premium)))
    if error > 1e-5:
        raise ValueError(f"day-10 straddle replication identity failed: {error}")
    return gain, penalty, error


def train_policy(*, config: HedgeConfig, train: TrainConfig, prices: np.ndarray, returns: np.ndarray,
                 payoff: np.ndarray, claim_sign: int, output: Path, price_mean: np.ndarray,
                 price_scale: np.ndarray, device: str = "cpu", resume_checkpoint: Path | None = None,
                 warm_start_checkpoint: Path | None = None) -> RecurrentHedge:
    """Optimize OCE less penalty; epochs means ADDITIONAL epochs on continuation."""
    if resume_checkpoint and warm_start_checkpoint:
        raise ValueError("choose exact continuation or a weights-only warm start, not both")
    saved = load_checkpoint(resume_checkpoint, config=config) if resume_checkpoint else None
    warm = load_checkpoint(warm_start_checkpoint, config=config) if warm_start_checkpoint else None
    torch.manual_seed(train.seed + claim_sign)
    env = BaseEnvironment(device=device, precision=32, debug=False)
    policy = RecurrentHedge(env, config=config, price_mean=torch.as_tensor(price_mean),
                            price_scale=torch.as_tensor(price_scale))
    if saved:
        policy.load_state_dict(saved["state_dict"])
    elif warm:
        # A changed simulation law has a new training objective/sample. Restore
        # the selected hedge, including normalization and frozen reference maps,
        # but restart Adam and re-evaluate the initial candidate on the new data.
        policy.load_state_dict(warm["best_state"])
    else:
        with torch.no_grad():
            policy.translation.fill_(optimize_translation(claim_sign * payoff, config.risk_aversion))
    optimizer = torch.optim.Adam(policy.parameters(), lr=train.learning_rate)
    # The full current sample is about 329 MiB. Load it once, then perform
    # shuffled minibatch gathers on device rather than transferring each batch.
    resident = tuple(torch.as_tensor(array, dtype=env.fdtype, device=env.device)
                     for array in (prices, returns, payoff)) if train.resident_data else None
    execute = torch.compile(policy, mode="reduce-overhead", fullgraph=False) if train.compile_model else policy
    print(json.dumps(dict(stage="execution", label=config.label, recurrence="native fused LSTM; batched time heads",
                          resident_sample_bytes=sum(x.numel() * x.element_size() for x in resident) if resident else 0,
                          compile_model=train.compile_model, scalar_metrics="epoch boundaries only")), flush=True)

    def batch_inputs(index: slice | torch.Tensor | np.ndarray) -> tuple[torch.Tensor, ...]:
        if resident is not None:
            return tuple(array[index] for array in resident)
        return tuple(torch.as_tensor(array[index], dtype=env.fdtype, device=env.device)
                     for array in (prices, returns, payoff))
    rng = np.random.default_rng(train.seed + claim_sign)
    start_epoch = saved["epoch"] + 1 if saved else 0
    pair_indices = np.arange(len(prices) // 2)
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = train.learning_rate
        if "pair_order" in saved:
            pair_indices = np.asarray(saved["pair_order"], dtype=np.int64).copy()
            if not np.array_equal(np.sort(pair_indices), np.arange(len(prices) // 2)):
                raise ValueError("saved antithetic-pair order does not match the training population")
        else:
            # Older checkpoints saved RNG but not the shuffled pair order.
            # Replay their shuffles to recover the exact next minibatch order.
            replay = np.random.default_rng(saved["train_config"]["seed"] + claim_sign)
            for _ in range(start_epoch):
                replay.shuffle(pair_indices)
            if json.dumps(replay.bit_generator.state, sort_keys=True) != json.dumps(saved["numpy_rng"], sort_keys=True):
                raise ValueError("could not reconstruct the checkpoint's antithetic-pair ordering")
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        if device == "cuda" and saved.get("cuda_rng"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    def fixed_policy_loss() -> float:
        total = torch.zeros((), dtype=torch.float64, device=env.device)
        with torch.no_grad():
            for begin in range(0, len(prices), train.batch_size):
                end = min(begin + train.batch_size, len(prices))
                inputs, increments, target = batch_inputs(slice(begin, end))
                value = execute(inputs, returns=increments)
                loss = -oce_value(value.gain + claim_sign * target, policy.translation, config.risk_aversion)
                total.add_((loss + value.penalty.mean()).to(torch.float64) * (end - begin))
        return float(total / len(prices))  # One scalar readback per complete scoring pass.

    # Preserve the earlier best candidate on continuation (the unhedged policy
    # on a fresh run). Compare frozen epoch-end policies on the training set,
    # not minibatch losses from changing optimizer states.
    initial_loss = fixed_policy_loss()
    if not math.isfinite(initial_loss):
        raise FloatingPointError(f"non-finite initial policy loss: {config.label}")
    best_loss = saved["best_loss"] if saved else initial_loss
    selected_epoch = saved["selected_epoch"] if saved else 0
    best_state = saved["best_state"] if saved else {k: v.detach().clone() for k, v in policy.state_dict().items()}
    stale = saved["stale_epochs"] if saved else 0
    started = time.monotonic()
    end_epoch = start_epoch + train.epochs
    if saved:
        print(json.dumps(dict(stage="resume", label=config.label, from_epoch=start_epoch, to_epoch=end_epoch,
                              initial_training_loss=initial_loss, inherited_best_loss=best_loss,
                              checkpoint=str(resume_checkpoint))), flush=True)
    if warm:
        print(json.dumps(dict(stage="warm-start", label=config.label, source_selected_epoch=warm["selected_epoch"],
                              initial_training_loss=initial_loss, optimizer="fresh Adam", epochs=end_epoch,
                              checkpoint=str(warm_start_checkpoint))), flush=True)
    for epoch in range(start_epoch, end_epoch):
        rng.shuffle(pair_indices)
        if resident is not None:
            # Preserve the existing pair shuffle/RNG exactly; upload only one
            # small pair-index vector per epoch, not any path observations.
            pair_order = torch.as_tensor(pair_indices, dtype=torch.int64, device=env.device)
            epoch_indices = (2 * pair_order[:, None] + torch.arange(2, device=env.device)).reshape(-1)
        else:
            epoch_indices = (2 * pair_indices[:, None] + np.arange(2)).reshape(-1)
        sum_loss = torch.zeros((), dtype=torch.float64, device=env.device)
        finite_steps = torch.ones((), dtype=torch.bool, device=env.device)
        for begin in range(0, len(pair_indices), train.batch_size // 2):
            indices = epoch_indices[2 * begin:2 * begin + train.batch_size]
            inputs, increments, target = batch_inputs(indices)
            output_value = execute(inputs, returns=increments)
            loss = -oce_value(output_value.gain + claim_sign * target, policy.translation, config.risk_aversion)
            loss = loss + output_value.penalty.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), train.gradient_clip,
                                                 error_if_nonfinite=False, foreach=True)
            finite_steps.logical_and_(torch.isfinite(loss.detach()) & torch.isfinite(norm))
            optimizer.step()
            sum_loss.add_(loss.detach().to(torch.float64) * len(indices))
        mean_loss = fixed_policy_loss()
        online_loss = float(sum_loss / len(prices))
        # Never publish a checkpoint from a nonfinite epoch. Retain the previous
        # atomic checkpoint as the recovery point without per-minibatch fences.
        if not bool(finite_steps) or not math.isfinite(mean_loss) or not math.isfinite(online_loss):
            raise FloatingPointError(f"non-finite loss or gradient: {config.label}, epoch {epoch + 1}; prior checkpoint retained")
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
            selected_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        elapsed = time.monotonic() - started
        done = epoch + 1 - start_epoch
        record = dict(stage="train", label=config.label, epoch=epoch + 1, epochs=end_epoch,
                      training_loss=mean_loss, online_minibatch_loss=online_loss,
                      best_training_loss=best_loss, stale_epochs=stale,
                      elapsed_seconds=elapsed, completed_additional_epochs=done, total_additional_epochs=train.epochs,
                      estimated_remaining_seconds=elapsed * (train.epochs - done) / done)
        print(json.dumps(record), flush=True)
        torch.save(dict(model_config=asdict(config), model_code_version=HedgeConfig.version.unique_id48,
                        user_version=config.user_version, train_config=asdict(train), state_dict=policy.state_dict(),
                        best_state=best_state, best_loss=best_loss, optimizer=optimizer.state_dict(),
                        epoch=epoch, selected_epoch=selected_epoch, stale_epochs=stale, numpy_rng=rng.bit_generator.state,
                        pair_order=pair_indices.copy(), resumed_from=str(resume_checkpoint) if resume_checkpoint else None,
                        warm_started_from=str(warm_start_checkpoint) if warm_start_checkpoint else None,
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device == "cuda" else []),
                   output / f"{config.label}.checkpoint.tmp")
        (output / f"{config.label}.checkpoint.tmp").replace(output / f"{config.label}.checkpoint.pt")
        if train.patience > 0 and stale >= train.patience:
            break
    if best_state is None:
        raise RuntimeError("no trained policy checkpoint")
    policy.load_state_dict(best_state)
    policy.fit_summary = dict(initial_training_loss=initial_loss, best_training_loss=best_loss,
                              selected_epoch=selected_epoch, epochs_run=epoch + 1, start_epoch=start_epoch,
                              additional_epochs_run=epoch + 1 - start_epoch,
                              warm_start=(dict(checkpoint=str(warm_start_checkpoint),
                                  source_selected_epoch=warm["selected_epoch"], optimizer="fresh Adam") if warm else None),
                              execution=dict(recurrence="native fused LSTM; batched time heads", resident_data=train.resident_data,
                                             compile_model=train.compile_model, scalar_metrics="epoch boundaries only"))
    return policy


def evaluate_policy(policy: RecurrentHedge, prices: np.ndarray, returns: np.ndarray, payoff: np.ndarray,
                    claim_sign: int, batch_size: int) -> tuple[np.ndarray, dict]:
    """Freeze the trained policy and summarize independent-sample economics."""
    device = policy.translation.device
    device_prices = torch.as_tensor(prices, dtype=policy.fdtype, device=device)
    device_returns = torch.as_tensor(returns, dtype=policy.fdtype, device=device)
    gain_device = torch.empty(len(prices), dtype=policy.fdtype, device=device)
    penalty_device = torch.empty_like(gain_device)
    with torch.no_grad():
        for start in range(0, len(prices), batch_size):
            stop = min(start + batch_size, len(prices))
            result = policy(device_prices[start:stop], returns=device_returns[start:stop])
            gain_device[start:stop] = result.gain
            penalty_device[start:stop] = result.penalty
    gain = gain_device.cpu().numpy().astype(np.float64)
    penalty = penalty_device.cpu().numpy().astype(np.float64)
    pnl = gain + claim_sign * payoff
    translation = float(policy.translation.detach())
    utility = _economic_utility(pnl, translation, policy.config.risk_aversion)
    optimized_translation = optimize_translation(pnl, policy.config.risk_aversion)
    reoptimized = _economic_utility(pnl, optimized_translation, policy.config.risk_aversion)
    return utility, dict(economic_oce_frozen_translation=pair_summary(utility),
        economic_oce_sample_optimized_translation=pair_summary(reoptimized),
        training_translation=translation, sample_optimized_translation=optimized_translation,
        mean_gain=pair_summary(gain), mean_regularizer=pair_summary(penalty),
        interpretation="The training regularizer is not subtracted from economic P&L; sample translation optimization is in-sample.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20, help="epochs without improvement; 0 disables early stopping")
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--resume-from", type=Path, help="directory of matching previous policy checkpoints")
    parser.add_argument("--resume-incomplete", action="store_true", help="recover atomic checkpoints from an interrupted matching run")
    parser.add_argument("--target-epochs", type=int, help="continue each selected policy only until this total epoch count")
    parser.add_argument("--resident-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warm-start-from", type=Path, help="restore selected weights only; permits new simulation data")
    parser.add_argument("--warm-start-long", type=Path, help="override the long checkpoint, e.g. after an interrupted run")
    parser.add_argument("--policies", choices=["empty", "long", "short"], nargs="+", default=["empty", "long", "short"])
    args = parser.parse_args()
    output, scenarios = args.output.resolve(), args.scenarios.resolve()
    root = Path(os.environ["QUANTDATA_ROOT"]).resolve()
    if not output.is_relative_to(root) or output.is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError("training output must be inside the external data root")
    checks = resource_health()
    if any(check["health"] in {"STOP", "UNAVAILABLE", "ERROR"} for check in checks):
        raise RuntimeError("resource manager blocks launching training")
    if any(check["health"] != "OK" for check in checks):
        print(json.dumps(dict(event="resource_warning", checks=checks)), flush=True)
    source = json.loads((scenarios / "complete.json").read_text())
    if source.get("status") != "complete" or source.get("schema") not in {"dhgromm1-expiry-options-v2", "dhgromm1-expiry-options-v3", SCENARIO_SCHEMA}:
        raise ValueError("complete expiry-return scenarios are required; one-day returns are incompatible")
    resume = args.resume_from.resolve() if args.resume_from else None
    warm = args.warm_start_from.resolve() if args.warm_start_from else None
    if resume and warm:
        raise ValueError("choose exact continuation or a weights-only warm start")
    if args.warm_start_long and warm is None:
        raise ValueError("--warm-start-long requires --warm-start-from")
    if set(args.policies) != {"empty", "long", "short"} and resume is None and warm is None:
        raise ValueError("omitted policies require a prior frozen checkpoint")
    initial_checkpoints = {label: (args.warm_start_long.resolve() if warm and label == "long" and args.warm_start_long
                                  else (resume or warm) / f"{label}.checkpoint.pt")
                           for label in ("empty", "long", "short")} if resume or warm else {}
    if any(path.parent == output for path in initial_checkpoints.values()):
        raise ValueError("use a new output directory; preserve the initialization checkpoints")
    if resume:
        if output == resume:
            raise ValueError("continuation must use a new output directory; preserve the source checkpoints")
        previous_path = resume / "report.json"
        if previous_path.exists():
            previous_report = json.loads(previous_path.read_text())
            if (previous_report.get("status") != "complete"
                    or previous_report.get("scenario_identity_sha256") != sha256(scenarios / "identity.json")):
                raise ValueError("continuation scenarios do not match the previous fitted policies")
        elif args.resume_incomplete:
            previous_path = resume.parent / "pipeline.json"
            previous = json.loads(previous_path.read_text())
            prior_scenarios = Path(previous.get("scenarios", resume.parent / "scenarios")).resolve()
            if (prior_scenarios != scenarios or previous.get("paths_per_sample", source["paths_per_sample"]) != source["paths_per_sample"]
                    or previous.get("simulation_substeps_per_day", source["substeps"]) != source["substeps"]):
                raise ValueError("interrupted run has different simulation data or time geometry")
        else:
            raise ValueError("incomplete source requires explicit --resume-incomplete")
    if args.target_epochs is not None and (not resume or args.target_epochs < 1):
        raise ValueError("a positive --target-epochs requires an exact resume")
    train = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                        patience=args.patience, seed=args.seed, resident_data=args.resident_data, compile_model=args.compile_model)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    monitor = AuditMonitor(output)
    monitor.thread.start()
    try:
        with FileLock(output / "training.lock", timeout=0):
            monitor.stage("load-training-sample")
            prices, returns, payoff = read_sample(scenarios / "train", paths=source["paths_per_sample"])
            instruments = prices.shape[2]
            if instruments != source.get("instruments", 17):
                raise ValueError("scenario instrument geometry differs from its identity")
            mean = prices[:, :-1].mean(axis=(0, 1), dtype=np.float64)
            scale = prices[:, :-1].std(axis=(0, 1), dtype=np.float64)
            scale = np.where(scale > 1e-6, scale, 1.0)
            benchmark_gain, benchmark_penalty, train_replication_error = day10_straddle_benchmark(prices, returns, payoff)
            benchmarks = {}
            for label, sign in (("long", 1), ("short", -1)):
                pnl = sign * (payoff - benchmark_gain)
                translation = optimize_translation(pnl, 100.)
                utility = _economic_utility(pnl, translation, 100.)
                benchmarks[label] = dict(training_translation=translation,
                    training_penalized_objective=float(utility.mean() - benchmark_penalty.mean()))
            policies = {}
            for label, sign in (("empty", 0), ("long", 1), ("short", -1)):
                config = HedgeConfig(label=label, instruments=instruments)
                remaining = train.epochs
                if resume and args.target_epochs is not None:
                    remaining = args.target_epochs - (load_checkpoint(initial_checkpoints[label], config=config)["epoch"] + 1)
                if label in args.policies and remaining > 0:
                    monitor.stage(f"train-{label}")
                    policies[label] = train_policy(config=config, train=replace(train, epochs=remaining), prices=prices,
                        returns=returns, payoff=payoff, claim_sign=sign, output=output, price_mean=mean,
                        price_scale=scale, device=args.device,
                        resume_checkpoint=initial_checkpoints[label] if resume else None,
                        warm_start_checkpoint=initial_checkpoints[label] if warm else None)
                else:
                    monitor.stage(f"load-frozen-{label}")
                    policies[label] = load_frozen_policy(initial_checkpoints[label], config=config, device=args.device)
                    retained = output / f"{label}.checkpoint.pt"
                    if retained.exists():
                        raise FileExistsError(retained)
                    shutil.copy2(initial_checkpoints[label], retained)
            del prices, returns, payoff
            monitor.stage("load-independent-evaluation-sample")
            prices, returns, payoff = read_sample(scenarios / "evaluation", paths=source["paths_per_sample"])
            benchmark_gain, benchmark_penalty, test_replication_error = day10_straddle_benchmark(prices, returns, payoff)
            for label, sign in (("long", 1), ("short", -1)):
                pnl = sign * (payoff - benchmark_gain)
                utility = _economic_utility(pnl, benchmarks[label]["training_translation"], 100.)
                benchmarks[label]["test_economic_oce"] = pair_summary(utility)
                benchmarks[label]["network_training_objective"] = -policies[label].fit_summary["best_training_loss"]
                benchmarks[label]["network_passes_training_benchmark"] = (
                    benchmarks[label]["network_training_objective"] >= benchmarks[label]["training_penalized_objective"] - 1e-6)
            utilities, reports = {}, {}
            for label, sign in (("empty", 0), ("long", 1), ("short", -1)):
                monitor.stage(f"evaluate-{label}")
                utilities[label], reports[label] = evaluate_policy(policies[label], prices, returns, payoff, sign, train.batch_size)
            bid, ask = indifference_quotes(empty_oce=float(utilities["empty"].mean()),
                long_oce=float(utilities["long"].mean()), short_oce=float(utilities["short"].mean()))
            result = dict(status="complete", scenarios=str(scenarios), scenario_identity_sha256=sha256(scenarios / "identity.json"),
                payoff_expectation=pair_summary(payoff), policies=reports,
                quotes=dict(bid=pair_summary(utilities["long"] - utilities["empty"]),
                            ask=pair_summary(utilities["empty"] - utilities["short"])),
                bid=bid, ask=ask, exact_empty_no_trading_oce=0.0,
                training_config=asdict(train), selection="training sample only; evaluation sample not used in selection",
                target_epochs=args.target_epochs,
                model_config=asdict(policies["long"].config), model_code_version=HedgeConfig.version.unique_id48,
                fits={name: policy.fit_summary for name, policy in policies.items()},
                day10_straddle_benchmark=dict(policies=benchmarks, train_replication_error=train_replication_error,
                                              test_replication_error=test_replication_error),
                optimization_status=("fails_day10_benchmark" if any(not b["network_passes_training_benchmark"]
                    for b in benchmarks.values()) else "benchmark_met_convergence_unverified"),
                sources={name: sha256(Path(__file__).parent / name) for name in ("training.py", "hedge.py", "rolling_simulation.py")},
                trained_policies=[label for label in args.policies if not policies[label].fit_summary.get("frozen")],
                initialization=(dict(mode="selected weights and frozen reference maps; fresh Adam; new epoch numbering",
                    checkpoints={label: dict(path=str(path), sha256=sha256(path))
                                 for label, path in initial_checkpoints.items()}) if warm else None),
                continuation=(dict(source_training_directory=str(resume), source_manifest_sha256=sha256(previous_path),
                    recovered_incomplete=not (resume / "report.json").exists(),
                    checkpoints={label: sha256(resume / f"{label}.checkpoint.pt") for label in ("empty", "long", "short")})
                    if resume else None),
                gain_definition="sum of floating quantities times (close-open)/S_open; no premium division",
                interpretation="Original Q weights; fixed-policy pair errors exclude policy-fitting uncertainty.")
            (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
            monitor.stage("complete")
    finally:
        monitor.done.set()


if __name__ == "__main__":
    main()
