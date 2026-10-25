"""Restartable GROMM policy training on public-panel generated scenarios."""

from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cdxcore.uniquehash import named_unique_filename48_8
from filelock import FileLock
from numpy.typing import NDArray

from quantdata.hedging.contracts import OptionRight
from quantdata.hedging.mean_reduction import (
    MeanReduction,
    MeanReductionConfig,
    MeanReductionDiagnostics,
    reduce_one_day_means,
)
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    train_config_snapshot,
    training_data_fingerprint,
    validate_train_config_change,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.config import (
    GrommObjective,
    GrommTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.objectives import (
    GrommObjectiveValue,
    gromm_hedging_objective,
    power_exponential_utility,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.policy import (
    GrommPolicy,
    GrommPolicyConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.simulation import (
    GrommDailyScenarios,
)
from quantdata.torch import BaseEnvironment, torch

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, kw_only=True)
class GrommTrainingData:
    """Immutable path tensors supplied to one policy fit."""

    features: FloatArray
    instrument_returns: FloatArray
    claim_returns: FloatArray
    active_mask: BoolArray
    linear_transaction_costs: FloatArray
    scenario_weights: FloatArray | None = None
    mean_reduction: MeanReduction | None = None
    mean_reduction_diagnostics: MeanReductionDiagnostics | None = None
    initial_positions: FloatArray | None = None
    settles_next: BoolArray | None = None
    terminal_transaction_costs: FloatArray | None = None

    def __post_init__(self) -> None:
        if self.features.ndim != 3:
            raise ValueError("features must be path-step-feature")
        path_count, step_count = self.features.shape[:2]
        if (
            path_count < 4
            or path_count % 2
            or self.instrument_returns.ndim != 3
            or self.instrument_returns.shape[:2] != (path_count, step_count)
            or self.claim_returns.shape != (path_count, step_count)
            or self.active_mask.shape
            != (step_count, self.instrument_returns.shape[-1])
            or self.linear_transaction_costs.shape
            not in {
                (step_count, self.instrument_returns.shape[-1]),
                (path_count, step_count, self.instrument_returns.shape[-1]),
            }
        ):
            raise ValueError("GROMM training arrays do not align")
        arrays = (
            self.features,
            self.instrument_returns,
            self.claim_returns,
            self.linear_transaction_costs,
        )
        if any(not np.all(np.isfinite(values)) for values in arrays):
            raise ValueError("GROMM training arrays must be finite")
        if np.any(self.linear_transaction_costs < 0.0):
            raise ValueError("GROMM transaction costs must be non-negative")
        if self.scenario_weights is not None and (
            self.scenario_weights.shape != (path_count,)
            or not np.all(np.isfinite(self.scenario_weights))
            or np.any(self.scenario_weights < 0.0)
            or not float(np.sum(self.scenario_weights)) > 0.0
        ):
            raise ValueError(
                "GROMM scenario weights must align and have finite-positive mass"
            )
        if self.mean_reduction_diagnostics is not None and (
            self.mean_reduction is None
            or self.mean_reduction_diagnostics.method is not self.mean_reduction
        ):
            raise ValueError("GROMM mean-reduction diagnostics do not match method")
        instrument_count = self.instrument_returns.shape[-1]
        if self.initial_positions is not None and self.initial_positions.shape not in {
            (instrument_count,),
            (path_count, instrument_count),
        }:
            raise ValueError("GROMM initial positions do not align")
        if self.settles_next is not None and self.settles_next.shape != (
            step_count,
            instrument_count,
        ):
            raise ValueError("GROMM settlement mask does not align")
        if self.terminal_transaction_costs is not None and (
            self.terminal_transaction_costs.shape
            not in {(instrument_count,), (path_count, instrument_count)}
            or not np.all(np.isfinite(self.terminal_transaction_costs))
            or np.any(self.terminal_transaction_costs < 0.0)
        ):
            raise ValueError("GROMM terminal costs do not align")
        if self.initial_positions is not None and not np.all(
            np.isfinite(self.initial_positions)
        ):
            raise ValueError("GROMM initial positions must be finite")

    @property
    def path_count(self) -> int:
        return len(self.features)

    @property
    def feature_count(self) -> int:
        return self.features.shape[-1]

    @property
    def instrument_count(self) -> int:
        return self.instrument_returns.shape[-1]

    @property
    def normalized_scenario_weights(self) -> FloatArray:
        """Return explicit probability weights for the scenario path axis."""
        if self.scenario_weights is None:
            return np.full(self.path_count, 1.0 / self.path_count, dtype=np.float64)
        return self.scenario_weights / np.sum(self.scenario_weights)


@dataclass(frozen=True, kw_only=True)
class GrommTrainingResult:
    """Trained policy and convergence diagnostics."""

    policy: GrommPolicy
    cash_translation: torch.Tensor | None
    best_epoch: int
    completed_epochs: int
    training_loss: float
    validation_loss: float
    validation_standard_error: float
    cap_hit_fraction: float
    checkpoint_path: Path
    converged: bool


def daily_gromm_training_data(
    scenarios: GrommDailyScenarios,
    *,
    mean_reduction: MeanReduction = MeanReduction.DEMEAN,
    mean_reduction_config: MeanReductionConfig = MeanReductionConfig(),
) -> GrommTrainingData:
    """Convert one public-panel scenario set to the common training schema."""
    calibration = scenarios.calibration
    day = scenarios.sample.transition.origin
    support = day.dlv_support_strikes
    density = calibration.front_density
    front_mean = float(density @ support)
    centered = support - front_mean
    front_variance = float(density @ centered**2)
    front_skew = float(
        density @ centered**3 / max(front_variance, np.finfo(np.float64).tiny) ** 1.5
    )
    log_dlvsq = np.log(day.dlvsq[1:])
    expiry_indices = np.linspace(
        0,
        len(log_dlvsq) - 1,
        num=min(8, len(log_dlvsq)),
        dtype=np.int64,
    )
    expiry_means = np.mean(log_dlvsq[expiry_indices], axis=1)
    rolling = calibration.rolling_physical
    feature = np.asarray(
        [
            float(scenarios.measure.value == "P"),
            scenarios.target.maturity_business_days / 5.0,
            scenarios.target.nms,
            1.0 if scenarios.target.right is OptionRight.CALL else -1.0,
            day.atm_vol_at_business_dte(1),
            day.atm_vol_at_business_dte(5),
            calibration.factors.explained_variance_fraction,
            calibration.front_spot_market_price_of_risk,
            0.0 if rolling is None else rolling.daily_risk_price_norm,
            front_mean,
            front_variance,
            front_skew,
            *expiry_means,
        ],
        dtype=np.float64,
    )
    features = np.broadcast_to(
        feature,
        (scenarios.path_count, 1, len(feature)),
    ).copy()
    raw = GrommTrainingData(
        features=features,
        instrument_returns=scenarios.instrument_returns[:, None, :],
        claim_returns=scenarios.claim_returns[:, None],
        active_mask=np.ones((1, scenarios.instrument_returns.shape[1]), dtype=bool),
        linear_transaction_costs=scenarios.linear_transaction_costs[None, :],
        settles_next=scenarios.settles_next[None],
        terminal_transaction_costs=None,
    )
    return reduce_gromm_one_day_training_data(
        raw,
        mean_reduction=mean_reduction,
        mean_reduction_config=mean_reduction_config,
    )


def reduce_gromm_one_day_training_data(
    data: GrommTrainingData,
    *,
    mean_reduction: MeanReduction,
    mean_reduction_config: MeanReductionConfig = MeanReductionConfig(),
    instrument_returns: FloatArray | None = None,
) -> GrommTrainingData:
    """Apply one common drift reduction to one-day instruments and targets."""
    if data.instrument_returns.shape[1] != 1:
        raise ValueError("mean reduction is defined here only for one-day GROMM data")
    raw_instruments = (
        data.instrument_returns[:, 0]
        if instrument_returns is None
        else np.asarray(instrument_returns, dtype=np.float64)
    )
    if raw_instruments.shape != data.instrument_returns[:, 0].shape:
        raise ValueError("replacement instrument returns do not match GROMM data")
    reduction = reduce_one_day_means(
        raw_instruments,
        data.claim_returns,
        method=mean_reduction,
        config=mean_reduction_config,
    )
    return GrommTrainingData(
        features=data.features,
        instrument_returns=reduction.instrument_returns[:, None],
        claim_returns=reduction.target_returns,
        active_mask=data.active_mask,
        linear_transaction_costs=data.linear_transaction_costs,
        scenario_weights=reduction.weights,
        mean_reduction=mean_reduction,
        mean_reduction_diagnostics=reduction.diagnostics,
        initial_positions=data.initial_positions,
        settles_next=data.settles_next,
        terminal_transaction_costs=data.terminal_transaction_costs,
    )


def fit_gromm_policy(
    data: GrommTrainingData,
    *,
    policy_config: GrommPolicyConfig,
    objective: GrommObjective,
    risk_parameter: float,
    checkpoint_directory: Path,
    train_config: GrommTrainConfig = GrommTrainConfig(),
    environment: BaseEnvironment | None = None,
) -> GrommTrainingResult:
    """Fit one claim or zero-claim policy with restartable checkpoints."""
    if policy_config.feature_count != data.feature_count:
        raise ValueError("policy feature count does not match training data")
    if policy_config.instrument_count != data.instrument_count:
        raise ValueError("policy instrument count does not match training data")
    if not math.isfinite(risk_parameter) or risk_parameter <= 0.0:
        raise ValueError("risk_parameter must be finite-positive")
    if environment is None:
        environment = BaseEnvironment(precision=64, device="cpu", debug=False)
    data_fingerprint = _gromm_data_fingerprint(data)
    precision = str(environment.fdtype)
    checkpoint_directory = Path(checkpoint_directory)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_name = named_unique_filename48_8(
        f"{policy_config.label}-{objective.value}-{risk_parameter:g}",
        policy_config,
        objective.value,
        risk_parameter,
        data_fingerprint,
        precision,
    )
    checkpoint_path = checkpoint_directory / f"{checkpoint_name}.pt"
    lock_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".lock")
    with FileLock(lock_path):
        return _fit_locked(
            data,
            policy_config=policy_config,
            objective=objective,
            risk_parameter=risk_parameter,
            checkpoint_path=checkpoint_path,
            train_config=train_config,
            environment=environment,
            data_fingerprint=data_fingerprint,
            precision=precision,
        )


def _fit_locked(
    data: GrommTrainingData,
    *,
    policy_config: GrommPolicyConfig,
    objective: GrommObjective,
    risk_parameter: float,
    checkpoint_path: Path,
    train_config: GrommTrainConfig,
    environment: BaseEnvironment,
    data_fingerprint: str,
    precision: str,
) -> GrommTrainingResult:
    train_indices, validation_indices = _paired_split(
        data.path_count,
        validation_fraction=train_config.validation_fraction,
        seed=train_config.seed,
    )
    training_features = data.features[train_indices]
    training_weights = data.normalized_scenario_weights[train_indices]
    training_weights = training_weights / np.sum(training_weights)
    feature_mean = np.sum(
        training_weights[:, None]
        * np.mean(training_features, axis=1),
        axis=0,
    )
    feature_variance = np.sum(
        training_weights[:, None, None]
        * np.square(training_features - feature_mean),
        axis=(0, 1),
    ) / training_features.shape[1]
    feature_scale = np.sqrt(feature_variance)
    feature_scale = np.maximum(feature_scale, 1.0e-8)
    torch.manual_seed(train_config.seed)
    policy = GrommPolicy(
        environment,
        config=policy_config,
        feature_mean=torch.as_tensor(
            feature_mean,
            dtype=environment.fdtype,
            device=environment.device,
        ),
        feature_scale=torch.as_tensor(
            feature_scale,
            dtype=environment.fdtype,
            device=environment.device,
        ),
    )
    cash_translation = (
        torch.nn.Parameter(
            torch.zeros((), dtype=environment.fdtype, device=environment.device)
        )
        if objective is GrommObjective.POWER_EXPONENTIAL
        else None
    )
    parameters = list(policy.parameters())
    if cash_translation is not None:
        parameters.append(cash_translation)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(train_config.seed)
    start_epoch = 0
    best_epoch = -1
    best_validation = math.inf
    best_state: dict[str, object] | None = None
    stale_epochs = 0
    checkpoint_converged = False
    current_train_config = train_config_snapshot(train_config)
    if checkpoint_path.exists():
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        compatible = (
            payload.get("checkpoint_format_version") == CHECKPOINT_FORMAT_VERSION
            and payload.get("policy_code_version")
            == policy_config.version.unique_id48
            and payload.get("policy_user_version") == policy_config.user_version
            and payload.get("objective") == objective.value
            and payload.get("risk_parameter") == risk_parameter
            and payload.get("data_fingerprint") == data_fingerprint
            and payload.get("precision") == precision
        )
        if compatible:
            changed_train_fields = validate_train_config_change(
                payload.get("train_config"),
                current_train_config,
                ignored_fields=frozenset(
                    {
                        "maximum_epochs",
                        "progress_seconds",
                        "checkpoint_epochs",
                        "compile_policy",
                        "allow_changed_config_warm_start",
                        "just_load",
                    }
                ),
                immutable_fields=frozenset({"seed", "validation_fraction"}),
                allow_warm_start=train_config.allow_changed_config_warm_start,
                model_name="GROMM",
            )
            policy.load_state_dict(payload["policy"])
            optimizer.load_state_dict(payload["optimizer"])
            if {"learning_rate", "weight_decay"} & set(changed_train_fields):
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = train_config.learning_rate
                    parameter_group["weight_decay"] = train_config.weight_decay
            if cash_translation is not None:
                cash_translation.data.copy_(payload["cash_translation"])
            batch_generator.set_state(payload["batch_generator_state"])
            torch.set_rng_state(payload["torch_rng_state"])
            if environment.device.type == "cuda" and payload.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
            start_epoch = int(payload["epoch"]) + 1
            best_epoch = int(payload["best_epoch"])
            best_validation = float(payload["best_validation"])
            best_state = payload["best_state"]
            stale_epochs = int(payload["stale_epochs"])
            checkpoint_converged = bool(payload.get("converged", False))
            if changed_train_fields:
                if train_config.just_load or start_epoch >= train_config.maximum_epochs:
                    raise ValueError(
                        "changed GROMM training fields require additional epochs"
                    )
                checkpoint_converged = False
                stale_epochs = 0
        else:
            warnings.warn(
                f"Ignoring incompatible GROMM checkpoint {checkpoint_path}",
                stacklevel=2,
            )
    if train_config.just_load or checkpoint_converged:
        if best_state is None:
            raise FileNotFoundError(
                f"no compatible trained GROMM checkpoint at {checkpoint_path}"
            )
        policy.load_state_dict(best_state["policy"])
        if cash_translation is not None:
            cash_translation.data.copy_(best_state["cash_translation"])
        return _training_result(
            policy,
            cash_translation=cash_translation,
            data=data,
            train_indices=train_indices,
            validation_indices=validation_indices,
            objective=objective,
            risk_parameter=risk_parameter,
            best_epoch=best_epoch,
            completed_epochs=start_epoch,
            checkpoint_path=checkpoint_path,
            converged=checkpoint_converged,
        )

    tensors = _training_tensors(data, environment=environment)
    training_policy = torch.compile(policy) if train_config.compile_policy else policy
    started = time.monotonic()
    last_progress = started
    converged = False
    completed_epochs = start_epoch
    for epoch in range(start_epoch, train_config.maximum_epochs):
        policy.train()
        permutation = torch.randperm(
            len(train_indices) // 2,
            generator=batch_generator,
        )
        for pair_start in range(0, len(permutation), train_config.batch_pairs):
            pair_selection = permutation[pair_start : pair_start + train_config.batch_pairs]
            batch_indices = torch.as_tensor(
                train_indices.reshape(-1, 2)[pair_selection.numpy()].reshape(-1),
                dtype=torch.long,
                device=environment.device,
            )
            optimizer.zero_grad(set_to_none=True)
            value = _objective_for_indices(
                training_policy,
                tensors=tensors,
                indices=batch_indices,
                objective=objective,
                risk_parameter=risk_parameter,
                cash_translation=cash_translation,
            )
            value.loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=100.0)
            optimizer.step()
        validation = _objective_for_indices(
            training_policy,
            tensors=tensors,
            indices=torch.as_tensor(
                validation_indices,
                dtype=torch.long,
                device=environment.device,
            ),
            objective=objective,
            risk_parameter=risk_parameter,
            cash_translation=cash_translation,
        )
        validation_loss = float(validation.loss.detach().cpu())
        required = (
            0.0
            if best_state is None
            else train_config.minimum_relative_improvement
            * max(abs(best_validation), 1.0)
        )
        if math.isfinite(validation_loss) and (
            best_state is None or validation_loss < best_validation - required
        ):
            best_validation = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                "policy": {
                    name: value.detach().cpu().clone()
                    for name, value in policy.state_dict().items()
                },
                "cash_translation": (
                    None
                    if cash_translation is None
                    else cash_translation.detach().cpu().clone()
                ),
            }
        else:
            stale_epochs += 1
        completed_epochs = epoch + 1
        now = time.monotonic()
        should_checkpoint = (
            completed_epochs % train_config.checkpoint_epochs == 0
            or now - last_progress >= train_config.progress_seconds
        )
        if should_checkpoint:
            _save_checkpoint(
                checkpoint_path,
                policy=policy,
                cash_translation=cash_translation,
                optimizer=optimizer,
                batch_generator=batch_generator,
                epoch=epoch,
                best_epoch=best_epoch,
                best_validation=best_validation,
                best_state=best_state,
                stale_epochs=stale_epochs,
                objective=objective,
                risk_parameter=risk_parameter,
                converged=False,
                data_fingerprint=data_fingerprint,
                precision=precision,
                train_config=current_train_config,
            )
            elapsed = now - started
            print(
                json.dumps(
                    {
                        "stage": "gromm-policy-training",
                        "epoch": completed_epochs,
                        "maximum_epochs": train_config.maximum_epochs,
                        "best_epoch": best_epoch + 1,
                        "best_validation_loss": best_validation,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": (
                            elapsed / completed_epochs
                            * (train_config.maximum_epochs - completed_epochs)
                        ),
                    }
                ),
                flush=True,
            )
            last_progress = now
        if (
            completed_epochs >= train_config.minimum_epochs
            and stale_epochs >= train_config.patience_epochs
        ):
            converged = True
            break
    if best_state is None:
        raise RuntimeError("GROMM training did not produce a finite validation state")
    _save_checkpoint(
        checkpoint_path,
        policy=policy,
        cash_translation=cash_translation,
        optimizer=optimizer,
        batch_generator=batch_generator,
        epoch=completed_epochs - 1,
        best_epoch=best_epoch,
        best_validation=best_validation,
        best_state=best_state,
        stale_epochs=stale_epochs,
        objective=objective,
        risk_parameter=risk_parameter,
        converged=converged,
        data_fingerprint=data_fingerprint,
        precision=precision,
        train_config=current_train_config,
    )
    policy.load_state_dict(best_state["policy"])
    if cash_translation is not None:
        cash_translation.data.copy_(best_state["cash_translation"])
    return _training_result(
        policy,
        cash_translation=cash_translation,
        data=data,
        train_indices=train_indices,
        validation_indices=validation_indices,
        objective=objective,
        risk_parameter=risk_parameter,
        best_epoch=best_epoch,
        completed_epochs=completed_epochs,
        checkpoint_path=checkpoint_path,
        converged=converged,
    )


def _paired_split(
    path_count: int,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    pair_count = path_count // 2
    generator = np.random.default_rng(seed)
    pairs = generator.permutation(pair_count)
    validation_pairs = max(1, int(round(pair_count * validation_fraction)))
    validation_pair_indices = pairs[-validation_pairs:]
    training_pair_indices = pairs[:-validation_pairs]
    if len(training_pair_indices) == 0:
        raise ValueError("paired split leaves no training paths")

    def expand(values: NDArray[np.int64]) -> NDArray[np.int64]:
        return np.column_stack((2 * values, 2 * values + 1)).reshape(-1)

    return expand(training_pair_indices), expand(validation_pair_indices)


def _training_tensors(
    data: GrommTrainingData,
    *,
    environment: BaseEnvironment,
) -> dict[str, torch.Tensor]:
    tensors = {
        "features": torch.as_tensor(
            data.features, dtype=environment.fdtype, device=environment.device
        ),
        "instrument_returns": torch.as_tensor(
            data.instrument_returns,
            dtype=environment.fdtype,
            device=environment.device,
        ),
        "claim_returns": torch.as_tensor(
            data.claim_returns, dtype=environment.fdtype, device=environment.device
        ),
        "active_mask": torch.as_tensor(
            data.active_mask, dtype=torch.bool, device=environment.device
        ),
        "linear_transaction_costs": torch.as_tensor(
            data.linear_transaction_costs,
            dtype=environment.fdtype,
            device=environment.device,
        ),
        "scenario_weights": torch.as_tensor(
            data.normalized_scenario_weights,
            dtype=environment.fdtype,
            device=environment.device,
        ),
    }
    if data.initial_positions is not None:
        tensors["initial_positions"] = torch.as_tensor(
            data.initial_positions,
            dtype=environment.fdtype,
            device=environment.device,
        )
    if data.settles_next is not None:
        tensors["settles_next"] = torch.as_tensor(
            data.settles_next,
            dtype=torch.bool,
            device=environment.device,
        )
    if data.terminal_transaction_costs is not None:
        tensors["terminal_transaction_costs"] = torch.as_tensor(
            data.terminal_transaction_costs,
            dtype=environment.fdtype,
            device=environment.device,
        )
    return tensors


def _objective_for_indices(
    policy: GrommPolicy,
    *,
    tensors: dict[str, torch.Tensor],
    indices: torch.Tensor,
    objective: GrommObjective,
    risk_parameter: float,
    cash_translation: torch.Tensor | None,
) -> GrommObjectiveValue:
    actions = policy(tensors["features"][indices], tensors["active_mask"])
    return gromm_hedging_objective(
        actions=actions,
        instrument_returns=tensors["instrument_returns"][indices],
        claim_returns=tensors["claim_returns"][indices],
        linear_transaction_costs=(
            tensors["linear_transaction_costs"]
            if tensors["linear_transaction_costs"].ndim == 2
            else tensors["linear_transaction_costs"][indices]
        ),
        objective=objective,
        risk_parameter=risk_parameter,
        scenario_weights=tensors["scenario_weights"][indices],
        cash_translation=cash_translation,
        initial_positions=(
            None
            if "initial_positions" not in tensors
            else (
                tensors["initial_positions"]
                if tensors["initial_positions"].ndim == 1
                else tensors["initial_positions"][indices]
            )
        ),
        settles_next=tensors.get("settles_next"),
        terminal_transaction_costs=(
            None
            if "terminal_transaction_costs" not in tensors
            else (
                tensors["terminal_transaction_costs"]
                if tensors["terminal_transaction_costs"].ndim == 1
                else tensors["terminal_transaction_costs"][indices]
            )
        ),
    )


def _save_checkpoint(
    path: Path,
    *,
    policy: GrommPolicy,
    cash_translation: torch.Tensor | None,
    optimizer: torch.optim.Optimizer,
    batch_generator: torch.Generator,
    epoch: int,
    best_epoch: int,
    best_validation: float,
    best_state: dict[str, object] | None,
    stale_epochs: int,
    objective: GrommObjective,
    risk_parameter: float,
    converged: bool,
    data_fingerprint: str,
    precision: str,
    train_config: dict[str, object],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "policy_code_version": policy.config.version.unique_id48,
            "policy_user_version": policy.config.user_version,
            "objective": objective.value,
            "risk_parameter": risk_parameter,
            "data_fingerprint": data_fingerprint,
            "precision": precision,
            "train_config": train_config,
            "policy": policy.state_dict(),
            "cash_translation": (
                None if cash_translation is None else cash_translation.detach()
            ),
            "optimizer": optimizer.state_dict(),
            "batch_generator_state": batch_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all()
                if policy.device.type == "cuda"
                else None
            ),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation": best_validation,
            "best_state": best_state,
            "stale_epochs": stale_epochs,
            "converged": converged,
        },
        temporary,
    )
    temporary.replace(path)


def _gromm_data_fingerprint(data: GrommTrainingData) -> str:
    return training_data_fingerprint(
        arrays={
            "active_mask": data.active_mask,
            "claim_returns": data.claim_returns,
            "features": data.features,
            "instrument_returns": data.instrument_returns,
            "linear_transaction_costs": data.linear_transaction_costs,
            "scenario_weights": data.normalized_scenario_weights,
            **(
                {}
                if data.initial_positions is None
                else {"initial_positions": data.initial_positions}
            ),
            **(
                {}
                if data.settles_next is None
                else {"settles_next": data.settles_next}
            ),
            **(
                {}
                if data.terminal_transaction_costs is None
                else {
                    "terminal_transaction_costs": data.terminal_transaction_costs
                }
            ),
        },
        metadata={
            "schema": "gromm-training-data-v3",
            "mean_reduction": (
                None if data.mean_reduction is None else data.mean_reduction.value
            ),
        },
    )


def _training_result(
    policy: GrommPolicy,
    *,
    cash_translation: torch.Tensor | None,
    data: GrommTrainingData,
    train_indices: NDArray[np.int64],
    validation_indices: NDArray[np.int64],
    objective: GrommObjective,
    risk_parameter: float,
    best_epoch: int,
    completed_epochs: int,
    checkpoint_path: Path,
    converged: bool,
) -> GrommTrainingResult:
    policy.eval()
    tensors = _training_tensors(data, environment=policy.env)
    validation_tensor_indices = torch.as_tensor(
        validation_indices,
        dtype=torch.long,
        device=policy.device,
    )
    with torch.no_grad():
        training = _objective_for_indices(
            policy,
            tensors=tensors,
            indices=torch.as_tensor(train_indices, device=policy.device),
            objective=objective,
            risk_parameter=risk_parameter,
            cash_translation=cash_translation,
        )
        validation = _objective_for_indices(
            policy,
            tensors=tensors,
            indices=validation_tensor_indices,
            objective=objective,
            risk_parameter=risk_parameter,
            cash_translation=cash_translation,
        )
        validation_influence = _objective_influence(
            validation.wealth,
            objective=objective,
            risk_parameter=risk_parameter,
            cash_translation=cash_translation,
            scenario_weights=tensors["scenario_weights"][validation_tensor_indices],
        ).detach().cpu().numpy()
        if len(validation_influence) % 2:
            raise RuntimeError("validation paths do not preserve antithetic pairs")
        pair_influence = validation_influence.reshape(-1, 2).mean(axis=1)
        standard_error = (
            0.0
            if len(pair_influence) == 1
            else float(
                np.std(pair_influence, ddof=1) / np.sqrt(len(pair_influence))
            )
        )
        actions = policy(tensors["features"], tensors["active_mask"])
        cap_hit_fraction = float(
            torch.mean(
                (
                    torch.abs(actions)
                    >= 0.999 * policy.config.position_cap
                ).to(dtype=policy.fdtype)
            ).cpu()
        )
    return GrommTrainingResult(
        policy=policy,
        cash_translation=cash_translation,
        best_epoch=best_epoch,
        completed_epochs=completed_epochs,
        training_loss=float(training.loss.cpu()),
        validation_loss=float(validation.loss.cpu()),
        validation_standard_error=standard_error,
        cap_hit_fraction=cap_hit_fraction,
        checkpoint_path=checkpoint_path,
        converged=converged,
    )


def _objective_influence(
    wealth: torch.Tensor,
    *,
    objective: GrommObjective,
    risk_parameter: float,
    cash_translation: torch.Tensor | None,
    scenario_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return centered first-order contributions to objective uncertainty."""
    if scenario_weights is None:
        weights = torch.full_like(wealth, 1.0 / len(wealth))
    else:
        if scenario_weights.shape != wealth.shape:
            raise ValueError("scenario weights must align with objective influence")
        total_weight = torch.sum(scenario_weights)
        if not bool(total_weight > 0.0):
            raise ValueError("scenario weights must have positive mass")
        weights = scenario_weights / total_weight
    effective_weights = len(wealth) * weights
    mean_wealth = torch.sum(weights * wealth)
    centered = wealth - mean_wealth
    if objective is GrommObjective.MEAN_VARIANCE:
        variance = torch.sum(weights * centered.square())
        influence = effective_weights * (-centered + 0.5 * risk_parameter * (
            centered.square() - variance
        ))
    elif objective is GrommObjective.MEAN_VOLATILITY:
        variance = torch.sum(weights * centered.square())
        standard_deviation = torch.sqrt(variance + torch.finfo(wealth.dtype).eps)
        influence = effective_weights * (-centered + risk_parameter * (
            centered.square() - variance
        ) / (2.0 * standard_deviation)
        )
    elif objective is GrommObjective.ENTROPY:
        exponent = -risk_parameter * wealth
        positive = weights > 0.0
        log_normalizer = torch.logsumexp(
            exponent[positive] + torch.log(weights[positive]), dim=0
        )
        normalized = torch.zeros_like(weights)
        normalized[positive] = torch.exp(
            exponent[positive] + torch.log(weights[positive]) - log_normalizer
        )
        influence = (len(wealth) * normalized - 1.0) / risk_parameter
    elif objective is GrommObjective.POWER_EXPONENTIAL:
        if cash_translation is None:
            raise ValueError("power-exponential influence requires cash translation")
        utility = power_exponential_utility(
            wealth - cash_translation,
            risk_parameter=risk_parameter,
        )
        mean_utility = torch.sum(weights * utility)
        influence = -effective_weights * (utility - mean_utility)
    else:
        raise ValueError(f"unsupported GROMM objective {objective!r}")
    return influence
