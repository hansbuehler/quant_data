"""Independent zero, short-claim and long-claim policies and paired bid/ask errors."""

from __future__ import annotations

import argparse
import copy
import signal
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version
from filelock import FileLock

from quantdata.gromm2.hedging_data import load_mapped_scenarios
from quantdata.gromm2.hedging_policy import (
    _atomic_torch_save,
    _Progress,
    _write_json,
    feature_normalization,
    optimized_oce,
    pair_statistics,
    training_environment,
)
from quantdata.gromm2.hedging_uncertainty import risk_path_scores
from quantdata.gromm2.joint_hedging import (
    JointTrainConfig,
    OptionsOnlyPolicy,
    OptionsPolicyConfig,
)
from quantdata.gromm2.market_batching import (
    GPU_CAPTURE_LOCK,
    MarketBatchIterator,
    cache_market_training_arrays,
)
from quantdata.gromm2.market_training_runtime import IndependentPortfolioStep
from quantdata.gromm2.rolling_audit import check_resource_headroom
from quantdata.torch import BaseEnvironment, BaseModel

LIABILITIES = {"zero": 0.0, "short": 1.0, "long": -1.0}


@version("0.1", dependencies=[OptionsPolicyConfig])
@dataclass(frozen=True, kw_only=True)
class BornOptionConfig(OptionsPolicyConfig):
    """Every option has a known first trading day and a fixed cash-settlement day."""

    group: str = "gromm2/independent-portfolio-hedging"
    option_start_days: tuple[int, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        verify_inp(len(self.option_start_days) == len(self.option_expiry_days)
            and all(0 <= start < expiry for start, expiry in zip(self.option_start_days, self.option_expiry_days)),
            lambda: "option start days must match the book and precede expiry")


class BornOptionPolicy(OptionsOnlyPolicy):
    """Zero holdings before listing prevent gains from the contract's birth mark."""

    def __init__(self, env: BaseEnvironment, *, config: BornOptionConfig,
                 normalization: tuple[np.ndarray, np.ndarray]) -> None:
        super().__init__(env, config=config, normalization_mean=normalization[0], normalization_scale=normalization[1])
        self.register_buffer("option_starts", torch.as_tensor(config.option_start_days, dtype=env.idtype, device=env.device))

    def live_instruments(self, day: int) -> torch.Tensor:
        return (day >= self.option_starts) & (day < self.option_expiries)


class IndependentHedges(BaseModel):
    """All three policies have independent recurrent, affine, deep and translation weights."""

    def __init__(self, env: BaseEnvironment, *, config: BornOptionConfig,
                 normalization: tuple[np.ndarray, np.ndarray], policy_class: type = BornOptionPolicy) -> None:
        super().__init__(env=env)
        self.config = config
        self.policies = torch.nn.ModuleDict({name: policy_class(env,
            config=replace(config, label=f"{config.label}-{name}"), normalization=normalization) for name in LIABILITIES})


def portfolio_quote_statistics(payoff: np.ndarray, gains: dict[str, np.ndarray], *,
                               risk_aversion: float) -> dict[str, object]:
    """Use paired OCE scores for the seller ask, buyer bid and expanded three-SE bracket."""
    scores = {}
    for name, sign in LIABILITIES.items():
        scores[name], _ = risk_path_scores(gains[name] - sign * payoff, risk_aversion=risk_aversion)
    records = {name: pair_statistics(values) for name, values in scores.items()}
    records["DH_sum"] = pair_statistics(scores["zero"] + scores["short"] + scores["long"])
    records["expectation"] = pair_statistics(payoff)
    records["ask"] = pair_statistics(scores["short"] - scores["zero"])
    records["bid"] = pair_statistics(scores["zero"] - scores["long"])
    records["ask_minus_expectation"] = pair_statistics(scores["short"] - scores["zero"] - payoff)
    records["expectation_minus_bid"] = pair_statistics(payoff - scores["zero"] + scores["long"])
    for entry in records.values():
        entry["three_se_lower"] = entry["mean"] - 3.0 * entry["standard_error"]
        entry["three_se_upper"] = entry["mean"] + 3.0 * entry["standard_error"]
    return {"statistics": records,
        "bid_ask_three_se_bracket": [records["bid"]["three_se_lower"], records["ask"]["three_se_upper"]],
        "crossed_point_quotes": records["bid"]["mean"] > records["ask"]["mean"],
        "interpretation": "Frozen-policy plug-in MC errors, clustered by complete antithetic pairs; include covariance in prices, exclude training and validation-selection uncertainty. Three-SE bounds are marginal, not a simultaneous coverage guarantee."}


def portfolio_evaluation_forward(model: IndependentHedges, *, compile: bool = False):
    """Pack only the per-path diagnostics required by population valuation."""
    def forward(features: torch.Tensor, prices: torch.Tensor) -> torch.Tensor:
        results = []
        for policy in model.policies.values():
            output = policy(features, prices)
            results.append(torch.stack((output.trading_gain, output.transaction_cost,
                output.penalty, output.cap_fraction), dim=-1))
        return torch.stack(results, dim=0)

    return torch.compile(forward, mode="reduce-overhead") if compile else forward


def evaluate_portfolios(model: IndependentHedges, data, *, batch_size: int, progress: _Progress | None = None,
                        forward=None, prefetch_workers: int = 2):
    """Evaluate every path, retaining only per-path gains and small diagnostic vectors."""
    arrays = {name: {key: np.empty(len(data.payoff), dtype=np.float64)
        for key in ("gain", "cost", "penalty", "cap")} for name in LIABILITIES}
    forward = portfolio_evaluation_forward(model) if forward is None else forward
    model.eval()
    evaluation_started = time.monotonic()
    with torch.no_grad(), MarketBatchIterator(data, env=model.env, batch_size=batch_size,
            prefetch_workers=prefetch_workers) as batches:
        for batch_number, (features, prices, _) in enumerate(batches):
            start = batch_number * batch_size
            if model.env.device.type == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            with GPU_CAPTURE_LOCK:
                packed_device = forward(features, prices)
            packed = packed_device.cpu().numpy()
            for policy_number, name in enumerate(LIABILITIES):
                for column, key in enumerate(("gain", "cost", "penalty", "cap")):
                    arrays[name][key][start:start + len(features)] = packed[policy_number, :, column]
            if progress is not None:
                completed = min(start + len(features), len(data.payoff))
                elapsed = time.monotonic() - evaluation_started
                progress.state.update(population_completed=completed, population_total=len(data.payoff),
                    population_split=data.identity.get("split"), population_elapsed_seconds=elapsed,
                    population_paths_per_second=completed / elapsed,
                    population_remaining_seconds=(len(data.payoff) - completed) * elapsed / completed)
            if progress is not None and progress.stop.is_set():
                raise InterruptedError("portfolio evaluation stopped; saved training checkpoint remains intact")
    metrics = {}
    for name, policy in model.policies.items():
        if not all(np.all(np.isfinite(value)) for value in arrays[name].values()):
            raise FloatingPointError("non-finite portfolio evaluation; no observations discarded")
        pnl = arrays[name]["gain"] - LIABILITIES[name] * data.payoff
        translation = float(policy.translation.detach().cpu())
        centered = pnl - translation
        utility = -np.expm1(-model.config.risk_aversion * np.maximum(centered, 0.0)) / model.config.risk_aversion
        utility += np.minimum(centered, 0.0) - 0.5 * model.config.risk_aversion * np.minimum(centered, 0.0)**2
        loss = -float(translation + utility.mean())
        metrics[name] = {"risk_at_training_translation": loss,
            "objective": loss + model.config.penalty_weight * float(arrays[name]["penalty"].mean()),
            "DH": -optimized_oce(pnl, model.config.risk_aversion)[0],
            "mean_net_trading_gain": float(arrays[name]["gain"].mean()),
            "mean_transaction_cost": float(arrays[name]["cost"].mean()),
            "near_cap_fraction": float(arrays[name]["cap"].mean())}
    return metrics, {name: value["gain"] for name, value in arrays.items()}


def train_independent_portfolios(train, validation, *, directory: Path, config: BornOptionConfig,
                                 train_config: JointTrainConfig = JointTrainConfig(), just_load: bool = False,
                                 policy_class: type = BornOptionPolicy,
                                 initial_model_state: dict[str, torch.Tensor] | None = None,
                                 initialization: dict | None = None):
    """Train on common minibatches, with separate optimizers, checkpoints and stopping decisions."""
    check_resource_headroom()
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(train_config.cpu_threads)
    env = training_environment(train_config)
    torch.random.default_generator.manual_seed(train_config.seed)
    progress = _Progress(label=config.label, interval=train_config.heartbeat_seconds)
    progress.state.update(stage="training-only-normalization", total=train_config.maximum_epochs * len(train.payoff))
    previous = {sig: signal.signal(sig, lambda *_: progress.stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    progress.thread.start()
    host_cache = None
    try:
        identity = {"config": asdict(config), "code_version": config.version.unique_id48, "user_version": config.user_version,
            "training_scenarios_sha256": train.sha256, "validation_scenarios_sha256": validation.sha256}
        path = directory / "checkpoint.pt"
        if path.exists():
            normalization_checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if normalization_checkpoint["identity"] != identity:
                raise ValueError("portfolio model or source identity changed; use a new run, preserve this checkpoint")
            normalization = tuple(normalization_checkpoint["model"][f"policies.zero.normalization_{name}"].numpy()
                for name in ("mean", "scale"))
            del normalization_checkpoint
        elif just_load:
            raise FileNotFoundError(path)
        elif initial_model_state is not None:
            if initialization is None or initialization.get("training_scenarios_sha256") != train.sha256:
                raise ValueError("warm-start normalization requires the identical training population")
            normalization = tuple(initial_model_state[f"policies.zero.normalization_{name}"].detach().cpu().numpy()
                for name in ("mean", "scale"))
        else:
            normalization = feature_normalization(train.features, floor=config.normalization_floor, progress=progress)
        model = IndependentHedges(env, config=config, normalization=normalization, policy_class=policy_class)
        if initial_model_state is not None and not path.exists():
            model.load_state_dict(initial_model_state, strict=True)
        optimizers = {name: torch.optim.Adam(policy.parameters(), lr=train_config.learning_rate)
            for name, policy in model.policies.items()}
        evaluation_forward = portfolio_evaluation_forward(model,
            compile=train_config.compile and env.device.type == "cuda")
        schedulers = {name: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
            patience=train_config.learning_rate_patience - 1, threshold=train_config.absolute_improvement,
            threshold_mode="abs", min_lr=train_config.minimum_learning_rate) for name, optimizer in optimizers.items()}
        generator = np.random.default_rng(train_config.seed)
        state = {"epoch": 0, "next_batch": 0, "order": None, "history": [], "policies": {
            name: {"best_epoch": 0, "best_validation": float("inf"), "early_stopped": False,
                   "best_state": copy.deepcopy(policy.state_dict())} for name, policy in model.policies.items()}}
        with FileLock(str(path) + ".lock", timeout=0) if not just_load else nullcontext():
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=False)
                if saved["identity"] != identity:
                    raise ValueError("portfolio model or source identity changed; use a new run, preserve this checkpoint")
                model.load_state_dict(saved["model"])
                for name in LIABILITIES:
                    optimizers[name].load_state_dict(saved["optimizers"][name])
                    schedulers[name].load_state_dict(saved["schedulers"][name])
                state = saved["state"]
                initialization = saved["report"].get("initialization", initialization)
                generator.bit_generator.state = saved["numpy_rng"]
                torch.set_rng_state(saved["torch_rng"])
                if env.device.type == "cuda" and saved["cuda_rng"] is not None:
                    torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device=env.device)
            elif just_load:
                raise FileNotFoundError(path)
            progress.initial_completed = state["epoch"] * len(train.payoff) + state["next_batch"]
            training_step = IndependentPortfolioStep(model, optimizers,
                gradient_clip=train_config.gradient_clip, compile=train_config.compile,
                use_cuda_graphs=train_config.compile and env.device.type == "cuda",
                captured_batch_size=train_config.batch_size) if not just_load else None

            save_in_progress = False

            def save(reason: str) -> None:
                nonlocal save_in_progress
                save_in_progress = True
                report = {"config": asdict(config), "train_config": asdict(train_config), "epoch": state["epoch"],
                    "stop_reason": reason, "training_paths": len(train.payoff), "validation_paths": len(validation.payoff),
                    "policies": {name: {key: value for key, value in record.items() if key != "best_state"}
                        for name, record in state["policies"].items()},
                    "parameter_sharing": "None; each portfolio has its own LSTM, affine/deep heads, translation, optimizer and scheduler.",
                    "execution": {"cuda_graphs": train_config.compile and env.device.type == "cuda",
                        "cuda_graph_workspaces": "cuBLAS workspaces allocated inside each manual graph's private pool; caches cleared outside recording under the shared capture lock.",
                        "inputs": "Exact raw training host cache when requested and admitted; bounded shared assembly, pinned double buffering and asynchronous CUDA transfer.",
                        "host_cache_bytes": host_cache.bytes_cached if host_cache is not None else 0,
                        "host_cache_array_names": list(host_cache.array_names) if host_cache is not None else [],
                        "tail_batches": "Every partial batch retained and evaluated eagerly; full-batch CUDA graph preserved.",
                        "safety": "Non-finite inputs, losses or gradients prevent all optimizer commits; exceptions inside partial batches do not overwrite the preceding checkpoint.",
                        "population_evaluation": "Packed compiled diagnostics; every training and validation path retained."},
                    "scenario_model": train.identity["model"],
                    "initialization": initialization,
                    "measure": "Original Q paths with equal weights; no recentering or empirical measure change."}
                _atomic_torch_save(path, {"identity": identity, "model": model.state_dict(), "state": state,
                    "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
                    "schedulers": {name: scheduler.state_dict() for name, scheduler in schedulers.items()},
                    "numpy_rng": generator.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device=env.device) if env.device.type == "cuda" else None,
                    "report": report})
                _write_json(directory / "training_report.json", report)
                _write_json(directory / "training_history.json", {"epochs": state["history"]})
                save_in_progress = False

            if not just_load:
                batch_in_progress = False
                try:
                    if train_config.host_cache_training:
                        progress.state.update(stage="caching-raw-training-host-arrays")
                        host_cache = cache_market_training_arrays(train, stop_requested=progress.stop.is_set,
                            allow_partial=True)
                        if host_cache is not None:
                            progress.resource_cleanup = host_cache.check_headroom
                            progress.resource_release = host_cache.release
                    if not state["history"]:
                        if initial_model_state is None:
                            with torch.no_grad():
                                for name, policy in model.policies.items():
                                    policy.translation.fill_(optimized_oce(-LIABILITIES[name] * train.payoff, config.risk_aversion)[1])
                        progress.state.update(stage="initialized-policy-baseline" if initial_model_state is not None else "zero-action-baseline")
                        initial_train, _ = evaluate_portfolios(model, train, batch_size=train_config.evaluation_batch_size,
                            progress=progress, forward=evaluation_forward, prefetch_workers=train_config.prefetch_workers)
                        initial_validation, _ = evaluate_portfolios(model, validation, batch_size=train_config.evaluation_batch_size,
                            progress=progress, forward=evaluation_forward, prefetch_workers=train_config.prefetch_workers)
                        for name, policy in model.policies.items():
                            state["policies"][name].update(best_validation=initial_validation[name]["objective"],
                                best_state=copy.deepcopy(policy.state_dict()))
                            schedulers[name].best = initial_validation[name]["objective"]
                        state["history"].append({"epoch": 0, "train": initial_train, "validation": initial_validation})
                        save("running")
                    while state["epoch"] < train_config.maximum_epochs and not all(record["early_stopped"] for record in state["policies"].values()):
                        if state["order"] is None:
                            state.update(order=generator.permutation(len(train.payoff)), next_batch=0)
                        model.train()
                        first_batch = state["next_batch"]
                        active_names = tuple(name for name in LIABILITIES if not state["policies"][name]["early_stopped"])
                        window_started, window_completed = time.monotonic(), first_batch
                        progress.state.update(stage="preparing-market-batches", epoch=state["epoch"] + 1,
                            completed=state["epoch"] * len(train.payoff) + first_batch)
                        with MarketBatchIterator(train, env=env, batch_size=train_config.batch_size,
                                order=state["order"], start=first_batch, prefetch_workers=train_config.prefetch_workers) as batches:
                            batch_ready_at = time.monotonic()
                            input_wait_seconds = update_seconds = 0.0
                            for batch_number, (features, prices, payoff) in enumerate(batches):
                                update_started = time.monotonic()
                                input_wait_seconds += update_started - batch_ready_at
                                start = first_batch + batch_number * train_config.batch_size
                                progress.state.update(stage="compiling-or-training-independent-policies",
                                    epoch=state["epoch"] + 1, completed=state["epoch"] * len(train.payoff) + start)
                                batch_in_progress = True
                                training_step(features, prices, payoff, active_names)
                                update_seconds += time.monotonic() - update_started
                                state["next_batch"] = min(start + len(features), len(train.payoff))
                                batch_in_progress = False
                                progress.state.update(stage="independent-portfolio-training", epoch=state["epoch"] + 1,
                                    completed=state["epoch"] * len(train.payoff) + state["next_batch"],
                                    active_policies=active_names)
                                if batch_number % 100 == 99:
                                    now = time.monotonic()
                                    progress.state.update(training_paths_per_second=(state["next_batch"] - window_completed)
                                        / (now - window_started), training_window_seconds=now - window_started,
                                        training_capture_count=training_step.capture_count,
                                        training_capture_seconds=training_step.capture_seconds,
                                        training_tail_steps=training_step.tail_step_count,
                                        training_input_wait_seconds=input_wait_seconds,
                                        training_update_seconds=update_seconds,
                                        training_host_cache_bytes=host_cache.bytes_cached if host_cache is not None else 0)
                                    window_started, window_completed = now, state["next_batch"]
                                    input_wait_seconds = update_seconds = 0.0
                                if progress.stop.is_set():
                                    raise InterruptedError("saved after a complete three-policy minibatch")
                                if progress.checkpoint_requested.is_set():
                                    save("running")
                                    progress.checkpoint_requested.clear()
                                batch_ready_at = time.monotonic()
                                progress.state["stage"] = "waiting-for-next-market-batch"
                        progress.state.update(stage="training-and-validation-population-loss")
                        train_metrics, _ = evaluate_portfolios(model, train, batch_size=train_config.evaluation_batch_size,
                            progress=progress, forward=evaluation_forward, prefetch_workers=train_config.prefetch_workers)
                        validation_metrics, _ = evaluate_portfolios(model, validation, batch_size=train_config.evaluation_batch_size,
                            progress=progress, forward=evaluation_forward, prefetch_workers=train_config.prefetch_workers)
                        state.update(epoch=state["epoch"] + 1, order=None, next_batch=0)
                        for name, policy in model.policies.items():
                            record = state["policies"][name]
                            if record["early_stopped"]:
                                continue
                            objective = validation_metrics[name]["objective"]
                            tolerance = max(train_config.absolute_improvement, train_config.relative_improvement * abs(record["best_validation"]))
                            if objective < record["best_validation"] - tolerance:
                                record.update(best_validation=objective, best_epoch=state["epoch"], best_state=copy.deepcopy(policy.state_dict()))
                            schedulers[name].threshold = tolerance
                            schedulers[name].step(objective)
                            if (state["epoch"] >= train_config.minimum_epochs
                                    and state["epoch"] - record["best_epoch"] >= train_config.convergence_patience):
                                record["early_stopped"] = True
                                policy.load_state_dict(record["best_state"])
                        state["history"].append({"epoch": state["epoch"], "train": train_metrics,
                            "validation": validation_metrics, "learning_rates": {name: optimizer.param_groups[0]["lr"] for name, optimizer in optimizers.items()}})
                        save_in_progress = True
                        save("validation_plateau" if all(record["early_stopped"] for record in state["policies"].values()) else "running")
                        save_in_progress = False
                        progress.emit(best_epochs={name: record["best_epoch"] for name, record in state["policies"].items()})
                        if (state["epoch"] == 1 or state["epoch"] % train_config.plot_every_epochs == 0
                                or all(record["early_stopped"] for record in state["policies"].values())):
                            from quantdata.gromm2.plot_hedging_training import (
                                plot_independent_losses,
                            )

                            plot_independent_losses(directory, output=directory / "training_loss.png")
                    save("validation_plateau" if all(record["early_stopped"] for record in state["policies"].values()) else "maximum_epoch_safety_limit_not_certified_converged")
                except BaseException:
                    if not save_in_progress and not batch_in_progress:
                        save("checkpointed_stop" if progress.stop.is_set() else "checkpointed_exception")
                    raise
            for name, policy in model.policies.items():
                policy.load_state_dict(state["policies"][name]["best_state"])
            results = {"selection": {"epoch": state["epoch"],
                "best_epochs": {name: record["best_epoch"] for name, record in state["policies"].items()}}}
            for split, data in (("train", train), ("validation", validation)):
                metrics, gains = evaluate_portfolios(model, data, batch_size=train_config.evaluation_batch_size,
                    progress=progress, forward=evaluation_forward, prefetch_workers=train_config.prefetch_workers)
                results[split] = {"policies": metrics, **portfolio_quote_statistics(data.payoff, gains, risk_aversion=config.risk_aversion)}
                np.savez_compressed(directory / f"{split}_frozen_pnl.npz", payoff=data.payoff, **gains)
            _write_json(directory / "quotes.json", results)
            return results
    finally:
        progress.resource_cleanup = None
        progress.resource_release = None
        if host_cache is not None:
            host_cache.release()
        progress.closed.set()
        progress.thread.join(timeout=2.0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run_portfolio_experiment(run_directory: Path, *, with_fixing_options: bool,
                             train_config: JointTrainConfig = JointTrainConfig(), risk_aversion: float = 100.0):
    train = load_mapped_scenarios(run_directory / "train.h5", expected_split="train", with_fixing_options=with_fixing_options)
    validation = load_mapped_scenarios(run_directory / "validation.h5", expected_split="validation", with_fixing_options=with_fixing_options)
    if train.identity["model"] != validation.identity["model"] or train.sha256 == validation.sha256:
        raise ValueError("training and validation must be independent samples of exactly the same model")
    geometry = train.identity["config"]
    added = len(geometry["fixing_hedge_moneyness"]) if with_fixing_options else 0
    variant = "with_fixing_options" if with_fixing_options else "without_fixing_options"
    config = BornOptionConfig(label=f"{variant}-lambda{risk_aversion:g}", feature_count=train.features.shape[-1],
        option_expiry_days=train.option_expiry_days, option_start_days=(0,) * (len(train.option_expiry_days) - added)
        + (int(geometry["fixing_day"]),) * added, risk_aversion=risk_aversion)
    return train_independent_portfolios(train, validation, directory=run_directory / variant / config.filename,
        config=config, train_config=train_config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--risk-aversion", type=float, default=100.0)
    parser.add_argument("--maximum-epochs", type=int, default=2000)
    parser.add_argument("--without-fixing-options", action="store_true")
    parser.add_argument("--compile", action="store_true")
    arguments = parser.parse_args()
    run_portfolio_experiment(arguments.run_directory, with_fixing_options=not arguments.without_fixing_options,
        risk_aversion=arguments.risk_aversion,
        train_config=JointTrainConfig(device=arguments.device, maximum_epochs=arguments.maximum_epochs,
            compile=arguments.compile))


if __name__ == "__main__":
    main()
