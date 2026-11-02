"""Joint options-only exotic/zero hedges with validation-selected stopping."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import signal
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version
from filelock import FileLock
from scipy.optimize import brentq

from quantdata.gromm2.hedging_policy import (
    HedgingPolicy,
    HedgingPolicyConfig,
    ScenarioData,
    TrainConfig,
    _atomic_torch_save,
    _batch,
    _Progress,
    _write_json,
    feature_normalization,
    load_scenarios,
    power_exponential_utility,
    training_environment,
)
from quantdata.gromm2.rolling_audit import check_resource_headroom
from quantdata.torch import BaseEnvironment, BaseModel


@version("0.1", dependencies=[HedgingPolicyConfig])
@dataclass(frozen=True, kw_only=True)
class OptionsPolicyConfig(HedgingPolicyConfig):
    """Options-only action geometry and spot-notional transaction costs."""

    group: str = "gromm2/joint-options-hedging"
    transaction_cost_bps: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        verify_inp(np.isfinite(self.transaction_cost_bps) and self.transaction_cost_bps >= 0.0,
                   lambda: "transaction costs must be finite and nonnegative")

    @property
    def instrument_count(self) -> int:
        return len(self.option_expiry_days)


@dataclass(frozen=True, kw_only=True)
class JointTrainConfig(TrainConfig):
    """Validation controls optimization; there is deliberately no third sample."""

    seed: int = 20261028
    batch_size: int = 256
    learning_rate: float = 2.5e-4
    minimum_epochs: int = 1
    learning_rate_patience: int = 25
    convergence_patience: int = 50
    relative_improvement: float = 1.0e-4
    absolute_improvement: float = 1.0e-7
    plot_every_epochs: int = 5
    prefetch_workers: int = 2
    cuda_memory_fraction: float | None = None
    host_cache_training: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        verify_inp(1 <= self.minimum_epochs <= self.maximum_epochs,
                   lambda: "minimum_epochs must lie between one and maximum_epochs")
        verify_inp(self.plot_every_epochs > 0, lambda: "plot interval must be positive")
        verify_inp(self.prefetch_workers in {1, 2}, lambda: "prefetch_workers must be one or two")
        verify_inp(isinstance(self.host_cache_training, bool), lambda: "host_cache_training must be boolean")
        verify_inp(self.cuda_memory_fraction is None or (np.isfinite(self.cuda_memory_fraction)
            and 0 < self.cuda_memory_fraction <= 1), lambda: "cuda_memory_fraction must be in (0,1]")


class OptionsOutput(NamedTuple):
    trading_gain: torch.Tensor
    penalty: torch.Tensor
    action_squared: torch.Tensor
    cap_fraction: torch.Tensor
    positions: torch.Tensor
    transaction_cost: torch.Tensor
    gross_gain: torch.Tensor


class JointOutput(NamedTuple):
    empty: OptionsOutput
    claim: OptionsOutput


class OptionsOnlyPolicy(HedgingPolicy):
    """Spot is observed and used for costs, but is never a tradable action."""

    def live_instruments(self, day: int) -> torch.Tensor:
        return day < self.option_expiries

    def forward(self, features: torch.Tensor, prices: torch.Tensor) -> OptionsOutput:
        assert prices.shape[-1] == self.config.instrument_count + 1
        output = super().forward(features, prices[..., 1:])
        previous = torch.cat((torch.zeros_like(output.positions[:, :1]), output.positions[:, :-1]), dim=1)
        alive = torch.arange(output.positions.shape[1], device=self.device)[:, None] < self.option_expiries
        turnover = (output.positions - previous).abs() * alive
        costs = self.config.transaction_cost_bps * 1.0e-4 * (
            prices[:, :-1, :1] * turnover.to(dtype=prices.dtype)).sum(dim=(1, 2))
        return OptionsOutput(output.trading_gain - costs, output.penalty, output.action_squared,
            output.cap_fraction, output.positions, costs, output.trading_gain)


class JointOptionsHedge(BaseModel):
    """Share recurrent parameters, not portfolio positions or recurrent states."""

    def __init__(self, env: BaseEnvironment, *, config: OptionsPolicyConfig,
                 normalization: tuple[np.ndarray, np.ndarray]) -> None:
        super().__init__(env=env)
        self.config = config
        self.empty = OptionsOnlyPolicy(env, config=replace(config, liability_units=0.0),
            normalization_mean=normalization[0], normalization_scale=normalization[1])
        self.claim = OptionsOnlyPolicy(env, config=replace(config, liability_units=1.0),
            normalization_mean=normalization[0], normalization_scale=normalization[1])
        self.claim.memory = self.empty.memory

    def forward(self, features: torch.Tensor, prices: torch.Tensor) -> JointOutput:
        return JointOutput(self.empty(features, prices), self.claim(features, prices))


def weighted_oce(pnl: torch.Tensor, translation: torch.Tensor, risk_aversion: float,
                 observation_multipliers: torch.Tensor) -> torch.Tensor:
    """Uniform minibatches use N*w multipliers, never batch-renormalized weights."""
    return translation + (observation_multipliers * power_exponential_utility(pnl - translation, risk_aversion)).mean()


def joint_objective(model: JointOptionsHedge, output: JointOutput, payoff: torch.Tensor,
                    observation_multipliers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    empty = -weighted_oce(output.empty.trading_gain, model.empty.translation,
                          model.config.risk_aversion, observation_multipliers)
    claim = -weighted_oce(output.claim.trading_gain - payoff, model.claim.translation,
                          model.config.risk_aversion, observation_multipliers)
    penalty = (observation_multipliers * (output.empty.penalty + output.claim.penalty)).mean()
    return empty + claim + model.config.penalty_weight * penalty, penalty


def optimized_weighted_risk(pnl: np.ndarray, weights: np.ndarray, risk_aversion: float) -> tuple[float, float]:
    """Return negative OCE for a frozen policy; numerical penalties are excluded."""
    pnl = np.asarray(pnl, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if pnl.ndim != 1 or pnl.shape != weights.shape or not np.all(np.isfinite(pnl)):
        raise ValueError("weighted risk requires matching finite P&L and weight vectors")
    if np.any(weights <= 0.0) or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("path weights must be strictly positive and sum to one")

    def residual(translation: float) -> float:
        centered = pnl - translation
        marginal = np.exp(-risk_aversion * np.maximum(centered, 0.0)) - risk_aversion * np.minimum(centered, 0.0)
        return float(weights @ marginal - 1.0)

    lower, upper = float(pnl.min()), float(pnl.max())
    translation = lower if lower == upper else float(brentq(residual, lower, upper, xtol=1.0e-12))
    centered = pnl - translation
    utility = -np.expm1(-risk_aversion * np.maximum(centered, 0.0)) / risk_aversion
    utility += np.minimum(centered, 0.0) - 0.5 * risk_aversion * np.minimum(centered, 0.0)**2
    return -float(translation + weights @ utility), translation


def evaluate_joint(model: JointOptionsHedge, data: ScenarioData, weights: np.ndarray, *,
                   batch_size: int, progress: _Progress | None = None) -> dict[str, object]:
    """Use fixed training translations for loss curves and optimized ones for prices."""
    arrays = {name: np.empty(len(data.payoff), dtype=np.float64) for name in (
        "empty_gain", "claim_gain", "empty_cost", "claim_cost", "empty_penalty", "claim_penalty", "empty_cap", "claim_cap")}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data.payoff), batch_size):
            features, prices, _ = _batch(data, slice(start, start + batch_size), env=model.env)
            output = model(features, prices)
            for name, result in (("empty", output.empty), ("claim", output.claim)):
                for suffix, value in (("gain", result.trading_gain), ("cost", result.transaction_cost),
                                      ("penalty", result.penalty), ("cap", result.cap_fraction)):
                    arrays[f"{name}_{suffix}"][start:start + len(features)] = value.cpu().numpy()
            if progress is not None and progress.stop.is_set():
                raise InterruptedError("joint hedge checkpointed during population evaluation")
    if not all(np.all(np.isfinite(values)) for values in arrays.values()):
        raise FloatingPointError("non-finite joint hedge evaluation")
    records = {}
    economic_loss = 0.0
    for name, policy in (("empty", model.empty), ("claim", model.claim)):
        pnl = arrays[f"{name}_gain"] - (data.payoff if name == "claim" else 0.0)
        translation = float(policy.translation.detach().cpu())
        centered = pnl - translation
        utility = -np.expm1(-model.config.risk_aversion * np.maximum(centered, 0.0)) / model.config.risk_aversion
        utility += np.minimum(centered, 0.0) - 0.5 * model.config.risk_aversion * np.minimum(centered, 0.0)**2
        loss = -float(translation + weights @ utility)
        economic_loss += loss
        risk, optimal_translation = optimized_weighted_risk(pnl, weights, model.config.risk_aversion)
        records[name] = {"loss_at_training_translation": loss, "DH": risk,
            "evaluation_translation": optimal_translation, "mean_pnl": float(weights @ pnl),
            "mean_net_trading_gain": float(weights @ arrays[f"{name}_gain"]),
            "mean_transaction_cost": float(weights @ arrays[f"{name}_cost"]),
            "mean_gross_trading_gain": float(weights @ (arrays[f"{name}_gain"] + arrays[f"{name}_cost"])),
            "near_cap_fraction": float(weights @ arrays[f"{name}_cap"])}
    penalty = float(weights @ (arrays["empty_penalty"] + arrays["claim_penalty"]))
    objective = economic_loss + model.config.penalty_weight * penalty
    return {"objective": objective, "penalty": penalty, "economic_objective": economic_loss,
        "empty": records["empty"], "claim": records["claim"],
        "DH_price": records["claim"]["DH"] - records["empty"]["DH"],
        "weighted_Q_expectation": float(weights @ data.payoff),
        "price_convention": "DH(Z)-DH(0), with DH negative optimized OCE and no initial claim premium in training."}


def load_path_weights(directory: Path, split: str, data: ScenarioData) -> np.ndarray:
    """Refuse altered scenarios, unequal antithetic weights or incomplete fits."""
    report = json.loads((directory / f"{split}.json").read_text())
    with np.load(directory / f"{split}.npz") as source:
        weights = np.asarray(source["path_weights"], dtype=np.float64)
        digest = str(source["scenario_sha256"])
    if not report["passed"] or digest != data.sha256 or report["identity"]["scenario_sha256"] != data.sha256:
        raise ValueError("preconditioning weights do not match the complete scenario set")
    if (weights.shape != data.payoff.shape or np.any(~np.isfinite(weights)) or np.any(weights <= 0.0)
            or not np.isclose(weights.sum(), 1.0, atol=1.0e-12, rtol=0.0)
            or not np.array_equal(weights[0::2], weights[1::2])):
        raise ValueError("invalid complete-path probability weights")
    return weights


def train_joint(train: ScenarioData, validation: ScenarioData, train_weights: np.ndarray, validation_weights: np.ndarray,
                *, directory: Path, config: OptionsPolicyConfig, train_config: JointTrainConfig,
                just_load: bool = False, preconditioned: bool = False) -> dict[str, object]:
    """Select checkpoints and stop using joint validation loss, not a third sample."""
    check_resource_headroom()
    if not preconditioned and (not np.array_equal(train_weights, np.full(len(train.payoff), 1.0 / len(train.payoff)))
                              or not np.array_equal(validation_weights, np.full(len(validation.payoff), 1.0 / len(validation.payoff)))):
        raise ValueError("nonuniform path weights require explicitly enabling preconditioning")
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(train_config.cpu_threads)
    env = training_environment(train_config)
    torch.random.default_generator.manual_seed(train_config.seed)
    normalization = feature_normalization(train.features, floor=config.normalization_floor)
    model = JointOptionsHedge(env, config=config, normalization=normalization)
    identity = {"config": asdict(config), "code_version": config.version.unique_id48, "user_version": config.user_version,
        "training_scenarios_sha256": train.sha256, "validation_scenarios_sha256": validation.sha256,
        "training_weights_sha256": hashlib.sha256(train_weights.tobytes()).hexdigest(),
        "validation_weights_sha256": hashlib.sha256(validation_weights.tobytes()).hexdigest()}
    checkpoint_path = directory / "checkpoint.pt"
    progress = _Progress(label=config.label, interval=train_config.heartbeat_seconds)
    old_handlers = {sig: signal.signal(sig, lambda *_: progress.stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with FileLock(str(checkpoint_path) + ".lock", timeout=0) if not just_load else nullcontext():
            optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                patience=train_config.learning_rate_patience - 1, threshold=train_config.absolute_improvement,
                threshold_mode="abs", min_lr=train_config.minimum_learning_rate)
            generator = np.random.default_rng(train_config.seed)
            state = {"epoch": 0, "next_batch": 0, "order": None, "history": [], "best_epoch": 0,
                "best_validation": float("inf"), "best_state": copy.deepcopy(model.state_dict()),
                "stop_reason": "running", "early_stopped": False, "last_gradient_norm": 0.0}
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if checkpoint["identity"] != identity:
                    raise ValueError("joint checkpoint versions, data, weights or geometry changed; preserve it and use a new run")
                model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])
                generator.bit_generator.state = checkpoint["numpy_rng"]
                torch.set_rng_state(checkpoint["torch_rng"])
                if env.device.type == "cuda":
                    torch.cuda.set_rng_state(checkpoint["torch_cuda_rng"].cpu(), device=env.device)
                state = checkpoint["training_state"]
            elif just_load:
                raise FileNotFoundError(checkpoint_path)
            if just_load:
                model.load_state_dict(state["best_state"])
                return {"training": json.loads((directory / "training_report.json").read_text()),
                    "train": evaluate_joint(model, train, train_weights, batch_size=train_config.evaluation_batch_size),
                    "validation": evaluate_joint(model, validation, validation_weights, batch_size=train_config.evaluation_batch_size)}
            if not state["early_stopped"] and int(state["epoch"]) < train_config.maximum_epochs:
                state["stop_reason"] = "running"
            forward = torch.compile(model) if train_config.compile else model
            progress.initial_completed = int(state["epoch"]) * len(train.payoff) + int(state["next_batch"])
            progress.state.update(completed=progress.initial_completed, total=train_config.maximum_epochs * len(train.payoff))
            progress.thread.start()
            save_in_progress = False

            def save() -> dict[str, object]:
                nonlocal save_in_progress
                save_in_progress = True
                report = {"policy_config": asdict(config), "train_config": asdict(train_config),
                    "training_paths": len(train.payoff), "validation_paths": len(validation.payoff),
                    "epoch": int(state["epoch"]), "best_epoch": int(state["best_epoch"]),
                    "best_validation_objective": float(state["best_validation"]) if np.isfinite(state["best_validation"]) else None,
                    "stop_reason": state["stop_reason"], "early_stopped": state["early_stopped"],
                    "device": str(env.device), "backend": env.device.type,
                    "initialization": "zero actions for the new options-only joint geometry",
                    "selection": "Best joint validation loss; halve LR after 25 stalled epochs, stop after 50 by default. No third sample.",
                    "zero_action_baseline_evaluated": bool(state.get("zero_action_baseline_evaluated", False)),
                    "preconditioning": "Separate fixed complete-path entropy measures for training and validation" if preconditioned else
                        "Disabled: original Q-generated samples with equal path weights",
                    "preconditioned": preconditioned,
                    "shared_parameters": "LSTM weights only; separate action heads, translations, positions and recurrent states",
                    "transaction_cost": "Spot-notional cost on initial entry and live option turnover; cash settlement is free",
                    "normalization": "Training features only; population mean and SD",
                    "statistical_interpretation": "Validation-selected performance, not an untouched conditional-arbitrage significance test."}
                _atomic_torch_save(checkpoint_path, {"identity": identity, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "numpy_rng": generator.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "torch_cuda_rng": torch.cuda.get_rng_state(device=env.device) if env.device.type == "cuda" else None,
                    "training_state": state, "best_state": state["best_state"], "report": report})
                _write_json(directory / "training_report.json", report)
                _write_json(directory / "training_history.json", {"epochs": state["history"]})
                save_in_progress = False
                return report

            try:
                if not state.get("zero_action_baseline_evaluated", False):
                    progress.state.update(stage="zero-action-validation-baseline")
                    current_state = copy.deepcopy(model.state_dict())
                    try:
                        with torch.no_grad():
                            for policy in (model.empty, model.claim):
                                for network in (policy.affine, policy.deep):
                                    final = network.In[-1] if len(network.In) else network.I0
                                    final.weight.zero_()
                                    if final.bias is not None:
                                        final.bias.zero_()
                                policy.translation.zero_()
                            model.claim.translation.fill_(optimized_weighted_risk(-train.payoff, train_weights, config.risk_aversion)[1])
                        baseline_train = evaluate_joint(model, train, train_weights,
                            batch_size=train_config.evaluation_batch_size, progress=progress)
                        baseline_validation = evaluate_joint(model, validation, validation_weights,
                            batch_size=train_config.evaluation_batch_size, progress=progress)
                        if float(baseline_validation["objective"]) < float(state["best_validation"]):
                            state.update(best_validation=float(baseline_validation["objective"]), best_epoch=0,
                                         best_state=copy.deepcopy(model.state_dict()))
                            scheduler.best = float(state["best_validation"])
                            scheduler.num_bad_epochs = int(state["epoch"])
                        state["history"].insert(0, {"epoch": 0, "objective": baseline_train["objective"],
                            "penalty": 0.0, "validation_objective": baseline_validation["objective"],
                            "validation_penalty": 0.0, "learning_rate": train_config.learning_rate,
                            "train": baseline_train, "validation": baseline_validation, "zero_action_baseline": True})
                        state["zero_action_baseline_evaluated"] = True
                    finally:
                        model.load_state_dict(current_state)
                    save()
                while int(state["epoch"]) < train_config.maximum_epochs and not bool(state["early_stopped"]):
                    epoch = int(state["epoch"])
                    if state["order"] is None:
                        state.update(order=generator.permutation(len(train.payoff)), next_batch=0)
                    model.train()
                    for start in range(int(state["next_batch"]), len(train.payoff), train_config.batch_size):
                        indices = state["order"][start:start + train_config.batch_size]
                        features, prices, payoff = _batch(train, indices, env=env)
                        multipliers = torch.as_tensor(len(train.payoff) * train_weights[indices], dtype=prices.dtype, device=env.device)
                        optimizer.zero_grad(set_to_none=True)
                        loss, _ = joint_objective(model, forward(features, prices), payoff, multipliers)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("non-finite joint training loss; no observations discarded")
                        loss.backward()
                        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip, error_if_nonfinite=True)
                        optimizer.step()
                        state.update(next_batch=min(start + train_config.batch_size, len(train.payoff)),
                                     last_gradient_norm=float(gradient_norm))
                        progress.state.update(stage="joint-training", epoch=epoch + 1,
                            completed=epoch * len(train.payoff) + int(state["next_batch"]), learning_rate=optimizer.param_groups[0]["lr"])
                        if progress.stop.is_set():
                            raise InterruptedError("joint hedge checkpointed after a complete optimizer update")
                        if progress.checkpoint_requested.is_set():
                            save()
                            progress.checkpoint_requested.clear()
                    progress.state.update(stage="joint-training-loss")
                    train_metrics = evaluate_joint(model, train, train_weights,
                        batch_size=train_config.evaluation_batch_size, progress=progress)
                    progress.state.update(stage="joint-validation-loss")
                    validation_metrics = evaluate_joint(model, validation, validation_weights,
                        batch_size=train_config.evaluation_batch_size, progress=progress)
                    objective = float(validation_metrics["objective"])
                    tolerance = max(train_config.absolute_improvement, train_config.relative_improvement * abs(float(state["best_validation"])))
                    if not np.isfinite(state["best_validation"]) or objective < float(state["best_validation"]) - tolerance:
                        state.update(best_validation=objective, best_epoch=epoch + 1, best_state=copy.deepcopy(model.state_dict()))
                    scheduler.threshold = max(train_config.absolute_improvement,
                        train_config.relative_improvement * abs(float(state["best_validation"])))
                    previous_lr = optimizer.param_groups[0]["lr"]
                    scheduler.step(objective)
                    state.update(epoch=epoch + 1, next_batch=0, order=None)
                    state["history"].append({"epoch": epoch + 1, "objective": train_metrics["objective"],
                        "penalty": train_metrics["penalty"], "validation_objective": objective,
                        "validation_penalty": validation_metrics["penalty"],
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "train": train_metrics, "validation": validation_metrics})
                    if epoch + 1 - int(state["best_epoch"]) >= train_config.convergence_patience:
                        state.update(early_stopped=True, stop_reason="validation_plateau_user_requested_stopping")
                    save()
                    progress.emit(training_objective=train_metrics["objective"], validation_objective=objective,
                        validation_DH_price=validation_metrics["DH_price"],
                        learning_rate=optimizer.param_groups[0]["lr"], learning_rate_reduced=optimizer.param_groups[0]["lr"] < previous_lr)
                    if (epoch == 0 or (epoch + 1) % train_config.plot_every_epochs == 0 or state["early_stopped"]):
                        from quantdata.gromm2.plot_hedging_training import plot_training_loss

                        plot_training_loss(directory, output=directory / "training_loss.png")
                if not state["early_stopped"]:
                    state["stop_reason"] = "maximum_epoch_safety_limit_not_certified_converged"
                report = save()
            except BaseException:
                if save_in_progress:
                    progress.emit(event_detail="checkpoint-save-failed-no-retry", final_checkpoint_trustworthy=False)
                    raise
                state["stop_reason"] = "resource_pressure_checkpointed_stop" if progress.resource_error is not None else (
                    "checkpointed_stop" if progress.stop.is_set() else "checkpointed_exception")
                save()
                raise
            model.load_state_dict(state["best_state"])
            result = {"training": report,
                "train": evaluate_joint(model, train, train_weights, batch_size=train_config.evaluation_batch_size, progress=progress),
                "validation": evaluate_joint(model, validation, validation_weights, batch_size=train_config.evaluation_batch_size, progress=progress)}
            _write_json(directory / "result.json", result)
            return result
    finally:
        progress.closed.set()
        if progress.thread.is_alive():
            progress.thread.join(timeout=2.0)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def run_joint_experiment(run_directory: Path, *, train_config: JointTrainConfig = JointTrainConfig(),
                         risk_aversions: tuple[float, ...] = (1.0, 10.0), just_load: bool = False,
                         preconditioned: bool = False) -> dict[str, object]:
    """Reuse two complete scenario sets; both validation fitting and stopping are explicit."""
    check_resource_headroom()
    train = load_scenarios(run_directory / "train.h5", expected_split="train")
    validation = load_scenarios(run_directory / "validation.h5", expected_split="validation")
    if (train.sha256 == validation.sha256 or train.identity["model"] != validation.identity["model"]
            or train.features.shape[1:] != validation.features.shape[1:]
            or train.option_expiry_days != validation.option_expiry_days
            or train.option_strikes != validation.option_strikes or train.option_is_call != validation.option_is_call):
        raise ValueError("independent training and validation must use the same financial model and fixed option book")
    if preconditioned:
        train_weights = load_path_weights(run_directory / "measures", "train", train)
        validation_weights = load_path_weights(run_directory / "measures", "validation", validation)
    else:
        train_weights = np.full(len(train.payoff), 1.0 / len(train.payoff))
        validation_weights = np.full(len(validation.payoff), 1.0 / len(validation.payoff))
    results = []
    for risk_aversion in risk_aversions:
        config = OptionsPolicyConfig(label=f"joint-options-lambda{risk_aversion:g}",
            feature_count=train.features.shape[-1], option_expiry_days=train.option_expiry_days, risk_aversion=risk_aversion)
        results.append(train_joint(train, validation, train_weights, validation_weights,
            directory=run_directory / "policies" / config.filename, config=config, train_config=train_config,
            just_load=just_load, preconditioned=preconditioned))
        report = {"schema": "gromm2-joint-options-only-cost-v1", "training_paths": len(train.payoff),
            "validation_paths": len(validation.payoff), "third_sample": False,
            "preconditioned": preconditioned,
            "option_book": {"expiry_business_days": train.option_expiry_days, "strikes": train.option_strikes,
                "is_call": train.option_is_call, "tradable_spot": False}, "policies": results,
            "interpretation": "Validation-selected risk and indifference-price estimates; not a formal untouched no-arbitrage test. " + (
                "Separate entropy-preconditioned empirical measures." if preconditioned else "Original Q samples with equal path weights; no reweighting.")}
        _write_json(run_directory / "joint_hedging_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--maximum-epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--risk-aversions", type=float, nargs="+", default=(1.0, 10.0))
    parser.add_argument("--preconditioned", action="store_true", help="Explicitly opt into saved entropy weights; disabled by default")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--just-load", action="store_true")
    arguments = parser.parse_args()
    run_joint_experiment(arguments.run_directory, train_config=JointTrainConfig(device=arguments.device,
        maximum_epochs=arguments.maximum_epochs, batch_size=arguments.batch_size, learning_rate=arguments.learning_rate,
        compile=arguments.compile), risk_aversions=tuple(arguments.risk_aversions),
        just_load=arguments.just_load, preconditioned=arguments.preconditioned)


if __name__ == "__main__":
    main()
