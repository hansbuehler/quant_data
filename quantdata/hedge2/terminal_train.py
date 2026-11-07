"""Checkpointed full-terminal policy fitting using training simulations only."""

from __future__ import annotations

import copy
import hashlib
import json
import signal
import threading
import time
import warnings
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from cdxcore.err import verify_inp
from filelock import FileLock

from quantdata.hedge2.terminal import TerminalPaths, evaluate_terminal_policy, terminal_risk
from quantdata.hedge2.terminal_coordinates import (
    ActorInputCoordinates,
    TerminalCoordinateMap,
    coordinate_checkpoint_binding,
    fit_actor_coordinates,
)
from quantdata.hedge2.terminal_descent import (
    gradient_vector,
    parameter_vector,
    probe_terminal_descent,
)
from quantdata.hedge2.terminal_features import TerminalFeatureMapping
from quantdata.hedge2.terminal_policy import (
    PolicyBook,
    TensorPolicyBook,
    TerminalNetworkConfig,
    TerminalPolicy,
    TerminalPolicyConfig,
)
from quantdata.hedge2.terminal_schedule import step_training_scheduler
from quantdata.hedge2.terminal_torch import (
    TensorTerminalEvaluation,
    TensorTerminalPaths,
    TensorTerminalRisk,
    evaluate_tensor_policy,
    tensor_terminal_risk,
)
from quantdata.torch import BaseEnvironment


@dataclass(frozen=True, kw_only=True)
class TerminalTrainConfig:
    seed: int = 1
    learning_rate: float = 1e-3
    minimum_learning_rate: float = 1e-6
    maximum_updates: int = 2000
    minimum_updates: int = 100
    check_every: int = 10
    stable_checks: int = 10
    learning_rate_patience: int = 5
    relative_loss_tolerance: float = 1e-6
    dollar_action_tolerance: float = 1e-3
    gradient_tolerance: float = 1e-5
    gradient_clip: float = 5.
    compile: bool = False
    compile_evaluation: bool = False
    compact_histories: bool = False
    actor_coordinates: str = "none"
    checkpoint_seconds: float | None = None
    progress_seconds: float = 30.
    optimizer: str = "adam"
    lbfgs_learning_rate: float = 1.
    lbfgs_history_size: int = 20
    lbfgs_max_evaluations: int = 30
    descent_probe_distances: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
    retry_review: bool = False

    def __post_init__(self) -> None:
        verify_inp(0 < self.minimum_updates <= self.maximum_updates
                   and self.check_every > 0 and self.stable_checks > 0 and self.learning_rate_patience > 0,
                   lambda: "Terminal training iteration counts must be positive and ordered")
        verify_inp(0 < self.minimum_learning_rate <= self.learning_rate
                   and self.relative_loss_tolerance >= 0 and self.dollar_action_tolerance > 0
                   and self.gradient_tolerance > 0 and self.gradient_clip > 0
                   and 0 < self.progress_seconds <= 240,
                   lambda: "Terminal learning rates, accuracy and progress cadence must be valid")
        verify_inp(isinstance(self.compile, bool) and isinstance(self.compile_evaluation, bool)
                   and (not self.compile_evaluation or self.compile),
                   lambda: "Full evaluation compilation requires the explicit compilation flag")
        verify_inp(isinstance(self.compact_histories, bool),
                   lambda: "History compaction must be an explicit boolean execution choice")
        verify_inp(self.actor_coordinates in ("none", "diagonal", "whitened")
                   and (self.actor_coordinates == "none" or self.optimizer == "lbfgs"),
                   lambda: "Actor input coordinates are optional float64 L-BFGS execution choices")
        verify_inp(self.checkpoint_seconds is None or (
            isinstance(self.checkpoint_seconds, (int, float)) and not isinstance(self.checkpoint_seconds, bool)
            and np.isfinite(self.checkpoint_seconds) and self.checkpoint_seconds > 0),
            lambda: "Checkpoint cadence must be None (every check) or finite positive seconds")
        verify_inp(self.optimizer in ("adam", "lbfgs")
                   and np.isfinite(self.lbfgs_learning_rate) and self.lbfgs_learning_rate > 0
                   and self.lbfgs_history_size > 0 and self.lbfgs_max_evaluations > 1,
                   lambda: "Choose Adam or full-batch L-BFGS with valid line-search settings")
        verify_inp(bool(self.descent_probe_distances)
                   and all(np.isfinite(distance) and distance > 0 for distance in self.descent_probe_distances)
                   and len(set(self.descent_probe_distances)) == len(self.descent_probe_distances),
                   lambda: "Descent probes need distinct finite positive relative parameter distances")
        verify_inp(isinstance(self.retry_review, bool), lambda: "Review retry must be explicit and Boolean")


@dataclass(frozen=True, kw_only=True)
class TerminalPolicyFit:
    policy: TerminalPolicy
    status: str
    updates: int
    initial_loss: float
    best_training_loss: float
    history: tuple[dict[str, float | int], ...]
    checkpoint_path: Path
    initialisation_date: date | None


@torch.no_grad()
def clip_terminal_gradients(parameters: Iterable[torch.nn.Parameter], *, maximum_norm: float) -> float:
    """Preserve L2 clipping while avoiding overflow from squared float32 gradients.

    A non-finite input is rejected before any gradient is changed. Norms and
    clipping products use scaled float64 arithmetic; the gradients retain
    their original dtype/device and the pre-clipping norm is reported.
    """
    if not np.isfinite(maximum_norm) or maximum_norm <= 0:
        raise ValueError("Gradient clipping needs a finite positive maximum norm")
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.
    if any(gradient.is_sparse or not gradient.is_floating_point()
           or gradient.device != gradients[0].device for gradient in gradients):
        raise ValueError("Terminal gradients must be dense real tensors on one device")
    largest = torch.stack([gradient.detach().abs().amax().to(torch.float64)
        if gradient.numel() else torch.zeros((), dtype=torch.float64, device=gradient.device)
        for gradient in gradients]).amax()
    if not bool(torch.isfinite(largest)):
        raise RuntimeError("Non-finite terminal-policy gradient; preserve the last periodic checkpoint")
    if float(largest) == 0.:
        return 0.
    scaled_norm = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(gradient.detach().to(torch.float64) / largest) for gradient in gradients]))
    norm = largest * scaled_norm
    if not bool(torch.isfinite(norm)):
        raise RuntimeError("Terminal-policy gradient norm exceeds float64; preserve the last periodic checkpoint")
    if float(norm + 1e-6) > maximum_norm:
        target_scale = maximum_norm / (scaled_norm + 1e-6 / largest)
        for gradient in gradients:
            gradient.copy_((gradient.detach().to(torch.float64) / largest * target_scale).to(gradient.dtype))
    return float(norm)


def subset_terminal_data(paths: TerminalPaths, book: PolicyBook,
                         indices: np.ndarray) -> tuple[TerminalPaths, PolicyBook]:
    """Select complete simulated paths; neither time rows nor history information is mixed."""
    rows = np.asarray(indices)
    if (rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or len(rows) == 0
            or np.any(rows < 0) or np.any(rows >= len(paths.gains)) or len(np.unique(rows)) != len(rows)):
        raise ValueError("Select distinct valid complete simulation paths")
    selected = replace(paths, **{name: getattr(paths, name)[rows] for name in
        ("gains", "costs", "features", "history_ids", "terminate_after_step", "claim_gain", "weights")})
    return selected, replace(book, prices=book.prices[rows])


def _data_hash(paths: TerminalPaths, book: PolicyBook, initial_inventory: np.ndarray) -> str:
    digest = hashlib.sha256()
    arrays = {**asdict(paths), **{f"book_{key}": value for key, value in asdict(book).items()},
              "initial_inventory": initial_inventory}
    for name, value in sorted(arrays.items()):
        values = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(values.dtype.str.encode())
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _atomic_checkpoint(path: Path, values: dict[str, object]) -> None:
    temporary = path.with_suffix(".partial.pt")
    torch.save(values, temporary)
    temporary.replace(path)


def _read_checkpoint(path: Path, *, env: BaseEnvironment,
                     expected_config: TerminalPolicyConfig | None = None) -> dict[str, object]:
    values = torch.load(path, map_location="cpu", weights_only=True)
    if values.get("actor_coordinates") is not None and (
            not isinstance(values["actor_coordinates"], dict)
            or not isinstance(values.get("optimization_model"), dict)
            or not isinstance(values.get("optimization_layout"), tuple)):
        raise ValueError("Incomplete optimizer-coordinate checkpoint; physical snapshots are preserved")
    if values.get("actor_coordinates") is not None:
        binding = coordinate_checkpoint_binding(coordinates=values["actor_coordinates"],
            layout=values["optimization_layout"], optimization_state=values["optimization_model"],
            physical_state=values["model"], best_state=values["best_model"])
        if values.get("actor_coordinate_binding") != binding:
            raise ValueError("Actor-coordinate checkpoint binding differs from its frozen inference/training state")
    elif any(name in values for name in ("optimization_model", "optimization_layout", "actor_coordinate_binding")):
        raise ValueError("Incomplete actor-coordinate checkpoint lacks its frozen normalization")
    if (values["code_version"] != TerminalPolicyConfig.version.unique_id48
            or values["user_version"] != values["model_config"]["user_version"]
            or expected_config is not None and values["user_version"] != expected_config.user_version):
        warnings.warn("Terminal policy checkpoint is out of date; cold-start in a new run directory.", UserWarning)
        raise ValueError("Incompatible terminal policy version; the original checkpoint is preserved")
    return values


def load_terminal_policy(path: Path, *, env: BaseEnvironment, best: bool = True,
                         expected_config: TerminalPolicyConfig | None = None) -> TerminalPolicy:
    """Read-only load: no training lock, data discovery, fitting or checkpoint deletion."""
    values = _read_checkpoint(path, env=env, expected_config=expected_config)
    config = dict(values["model_config"])
    config["deep"] = TerminalNetworkConfig(**config["deep"])
    policy = TerminalPolicy(env=env, config=TerminalPolicyConfig(**config))
    policy.load_state_dict(values["best_model"] if best else values["model"], strict=True)
    policy.fit_context = values["context"]
    policy.as_of_date = date.fromisoformat(values["as_of_date"])
    return policy


def _history_actor_inputs(*, policy: TerminalPolicy, paths: TensorTerminalPaths,
                          book: TensorPolicyBook, history_ids: np.ndarray
                          ) -> tuple[tuple[torch.Tensor, torch.Tensor, TensorPolicyBook], ...]:
    """Validate exact predictable history sharing; never merge approximately equal states."""
    count, steps, _ = paths.gains.shape
    histories = np.asarray(history_ids)
    if (histories.shape != (count, steps) or not np.issubdtype(histories.dtype, np.integer)
            or np.any(histories[:, 0] != histories[0, 0])):
        raise ValueError("Actor compaction requires aligned integer histories and one common root")
    forbidden = (torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.dropout._DropoutNd, torch.nn.RReLU)
    if any(isinstance(module, forbidden) for module in policy.modules()):
        raise ValueError("History compaction requires a deterministic pointwise policy with fixed normalization")
    active = torch.full((count,), paths.initially_active, dtype=torch.bool, device=paths.gains.device)
    selections = []
    for step in range(steps):
        _, representatives, inverse = np.unique(histories[:, step], return_index=True, return_inverse=True)
        if step and not np.array_equal(histories[:, step - 1], histories[representatives, step - 1][inverse]):
            raise ValueError("Compacted information histories cannot merge different observed prefixes")
        rows = torch.tensor(representatives, dtype=torch.long, device=paths.gains.device)
        mapping = torch.tensor(inverse, dtype=torch.long, device=paths.gains.device)
        for values in (paths.features[:, step], paths.costs[:, step], book.prices[:, step],
                       book.prices[:, 0, :1], active):
            if not torch.equal(values, values.index_select(0, rows).index_select(0, mapping)):
                raise ValueError("History compaction requires exactly equal predictable inputs within every history")
        selected = replace(book, prices=book.prices.index_select(0, rows))
        selections.append((rows, mapping, selected))
        active &= ~paths.terminate_after_step[:, step]
    return tuple(selections)


def make_terminal_evaluator(*, policy: TerminalPolicy, paths: TensorTerminalPaths,
                            book: TensorPolicyBook, initial_inventory: torch.Tensor,
                            objective: str, risk_aversion: float, target_side: int,
                            compile_policy: bool = False, compile_evaluation: bool = False,
                            history_ids: np.ndarray | None = None,
                            backend: str | None = None
                            ) -> Callable[[], tuple[TensorTerminalEvaluation, TensorTerminalRisk]]:
    """Compile only execution, preserving all public cash and risk validators.

    Compiling the evaluation callable may retain eager graph breaks at
    tensor-dependent validation. No finite checks, cash-root iterations,
    envelope gradients, turnover terms or policy constraints are skipped.
    The optional backend supports explicit numerical/throughput comparisons;
    production uses Torch's default compiler backend. Explicit history IDs
    optionally share only deterministic actor evaluations. All original
    paths, probabilities, gains, fees, wealth and objective reductions remain.
    """
    if (not isinstance(compile_policy, bool) or not isinstance(compile_evaluation, bool)
            or backend is not None and (not isinstance(backend, str) or not backend)):
        raise ValueError("Compilation modes must be booleans with an optional nonempty backend")
    arguments = {} if backend is None else {"backend": backend}
    actor = torch.compile(policy, **arguments) if compile_policy and not compile_evaluation else policy
    selections = (None if history_ids is None else
        _history_actor_inputs(policy=policy, paths=paths, book=book, history_ids=history_ids))

    def positions(features: torch.Tensor, inventory: torch.Tensor, step: int) -> torch.Tensor:
        if selections is None:
            return actor(features, inventory, step, book=book, costs=paths.costs[:, step])
        rows, mapping, selected = selections[step]
        if len(rows) == len(features):
            return actor(features, inventory, step, book=book, costs=paths.costs[:, step])
        proposed = actor(features.index_select(0, rows), inventory.index_select(0, rows), step,
            book=selected, costs=paths.costs[:, step].index_select(0, rows))
        return proposed.index_select(0, mapping)

    def evaluate() -> tuple[TensorTerminalEvaluation, TensorTerminalRisk]:
        result = evaluate_tensor_policy(paths=paths, initial_inventory=initial_inventory,
            policy=positions, target_side=target_side)
        risk = tensor_terminal_risk(result.wealth, paths.weights,
            objective=objective, risk_aversion=risk_aversion)
        return result, risk

    return torch.compile(evaluate, **arguments) if compile_evaluation else evaluate


def train_terminal_policy(*, paths: TerminalPaths, book: PolicyBook, initial_inventory: np.ndarray,
                          model_config: TerminalPolicyConfig, train_config: TerminalTrainConfig,
                          env: BaseEnvironment, objective: str, risk_aversion: float, target_side: int,
                          as_of_date: date, target_key: str, output_directory: Path,
                          previous_policy: TerminalPolicy | None = None,
                          feature_mapping: TerminalFeatureMapping | None = None,
                          progress: Callable[[str], None] | None = None) -> TerminalPolicyFit:
    """Minimize risk of final NET wealth, not next-period marked P&L.

    Training uses the full fixed simulation sample. No validation data
    enters optimization, normalization, checkpoint selection or stopping.
    Completion means training-loss/action stability within this strategy
    class, NOT a globally certified optimum or good realized performance.
    A maximum-update exit is explicitly non-converged and remains resumable.
    """
    book.validate_histories(paths)
    initial = np.asarray(initial_inventory, dtype=float)
    if (initial.shape != (paths.gains.shape[2],) or not np.isfinite(initial).all()
            or model_config.feature_count != paths.features.shape[2] or not target_key
            or objective not in ("mean_variance", "power_exponential") or target_side not in (-1, 1)
            or risk_aversion <= 0 or np.isnan(risk_aversion)
            or objective == "power_exponential" and not np.isfinite(risk_aversion)):
        raise ValueError("Terminal training model, inventory, target and risk specification must align")
    if train_config.optimizer == "lbfgs" and env.fdtype != torch.float64:
        raise ValueError("The L-BFGS acceptance protocol currently requires validated float64 arithmetic")
    resolved = output_directory.resolve()
    if any((parent / ".git").is_file() or (parent / ".git" / "HEAD").is_file()
           for parent in (resolved, *resolved.parents)):
        raise ValueError("Terminal training outputs must be outside Git working trees")
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = output_directory / f"{model_config.filename}.pt"
    context = {"objective": objective, "risk_aversion": risk_aversion, "target_side": target_side,
               "target_key": target_key}
    identity = {"data_sha256": _data_hash(paths, book, initial), "context": context,
        "model_config": asdict(model_config), "as_of_date": as_of_date.isoformat(),
        "implementation_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("terminal_train.py", "terminal_policy.py", "terminal_features.py", "terminal_torch.py", "terminal.py", "terminal_descent.py", "terminal_schedule.py", "terminal_coordinates.py")}}
    if train_config.optimizer == "lbfgs":
        identity["optimizer_protocol"] = {"name": "lbfgs-training-descent-v1", "precision": str(env.fdtype),
            "device_type": env.device.type, **{key: value for key, value in asdict(train_config).items()
                if key not in ("maximum_updates", "checkpoint_seconds", "progress_seconds", "retry_review")}}
    with FileLock(str(checkpoint.with_suffix(".lock")), timeout=0):
        saved = _read_checkpoint(checkpoint, env=env, expected_config=model_config) if checkpoint.exists() else None
        if saved is not None and saved.get("optimizer_type", "adam") != train_config.optimizer:
            raise ValueError("Changing terminal optimizer requires an explicit migration into a new run directory")
        if saved is not None and saved["identity"].get("optimizer_protocol") != identity.get("optimizer_protocol"):
            raise ValueError("Changed terminal numerical protocol requires an explicit reviewed checkpoint migration")
        if saved is not None and saved["identity"] != identity:
            raise ValueError("Terminal checkpoint belongs to different simulations, model, or fitting scope")
        torch.manual_seed(train_config.seed)
        policy = TerminalPolicy(env=env, config=model_config)
        tensor_paths, tensor_book = TensorTerminalPaths.from_paths(paths, env=env), book.tensor(env)
        tensor_initial = torch.tensor(initial, dtype=env.fdtype, device=env.device)
        initialisation_date = None
        if saved is None:
            if previous_policy is not None:
                if (previous_policy.config != model_config or not hasattr(previous_policy, "as_of_date")
                        or previous_policy.as_of_date >= as_of_date or previous_policy.fit_context != context):
                    raise ValueError("Rolling initialization must use an earlier compatible policy for the same objective and target")
                policy.load_state_dict(previous_policy.state_dict(), strict=True)
                initialisation_date = previous_policy.as_of_date
                if feature_mapping is None:
                    raise ValueError("Rolling warm starts require an explicit recorded PCA/state feature mapping")
            policy.rebase_normalization(tensor_paths.features, tensor_paths.weights, mapping=feature_mapping)
        else:
            policy.load_state_dict(saved["model"], strict=True)
            initialisation_date = (None if saved["initialisation_date"] is None
                                   else date.fromisoformat(saved["initialisation_date"]))
            recorded_mapping = saved.get("feature_mapping")
            if recorded_mapping is not None:
                feature_mapping = TerminalFeatureMapping(matrix=np.asarray(recorded_mapping["matrix"]),
                    offset=np.asarray(recorded_mapping["offset"]),
                    elapsed_step_offset=recorded_mapping["elapsed_step_offset"],
                    old_subspace_projection_residual=recorded_mapping["old_subspace_projection_residual"])
        coordinate_map = None
        if train_config.actor_coordinates != "none":
            if saved is None:
                raw_evaluate = make_terminal_evaluator(policy=policy, paths=tensor_paths, book=tensor_book,
                    initial_inventory=tensor_initial, objective=objective, risk_aversion=risk_aversion,
                    target_side=target_side, history_ids=paths.history_ids if train_config.compact_histories else None)
                coordinates = fit_actor_coordinates(policy=policy, evaluate=raw_evaluate,
                    weights=tensor_paths.weights, steps=paths.gains.shape[1], instruments=paths.gains.shape[2],
                    fitting_data_sha256=identity["data_sha256"], mode=train_config.actor_coordinates,
                    history_ids=paths.history_ids if train_config.compact_histories else None)
                coordinate_map = TerminalCoordinateMap(policy=policy, coordinates=coordinates)
                coordinate_map.activate()
            else:
                if saved.get("actor_coordinates") is None:
                    raise ValueError("Actor-coordinate resume lacks its original frozen normalization")
                coordinates = ActorInputCoordinates.from_record(saved["actor_coordinates"], policy=policy,
                    expected_data_sha256=identity["data_sha256"], expected_mode=train_config.actor_coordinates)
                coordinate_map = TerminalCoordinateMap(policy=policy, coordinates=coordinates)
                if saved["optimization_layout"] != coordinate_map.layout:
                    raise ValueError("Actor optimizer-coordinate parameter layout changed")
                optimization_state = {name: value.to(device=policy.device)
                    for name, value in saved["optimization_model"].items()}
                physical = coordinate_map.state_dict(optimization_state, to_physical=True)
                if set(physical) != set(saved["model"]) or any(
                        not torch.equal(value.detach().cpu(), saved["model"][name]) for name, value in physical.items()):
                    raise ValueError("Physical inference snapshot differs from its optimizer-coordinate binding")
                policy.load_state_dict(optimization_state, strict=True)
                coordinate_map.activate(map_weights=False)
        elif saved is not None and saved.get("actor_coordinates") is not None:
            raise ValueError("Actor-coordinate checkpoints cannot resume as untransformed optimizers")

        def physical_state() -> dict[str, torch.Tensor]:
            return (copy.deepcopy(policy.state_dict()) if coordinate_map is None else
                coordinate_map.state_dict(policy.state_dict(), to_physical=True))

        parameters = tuple(policy.parameters())
        use_lbfgs = train_config.optimizer == "lbfgs"
        optimizer = (torch.optim.LBFGS(parameters, lr=train_config.lbfgs_learning_rate,
            max_iter=1, max_eval=train_config.lbfgs_max_evaluations,
            history_size=train_config.lbfgs_history_size, tolerance_grad=0., tolerance_change=0.,
            line_search_fn="strong_wolfe") if use_lbfgs else
            torch.optim.Adam(parameters, lr=train_config.learning_rate))
        scheduler = None if use_lbfgs else torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5,
            patience=train_config.learning_rate_patience, min_lr=train_config.minimum_learning_rate,
            threshold=train_config.relative_loss_tolerance, threshold_mode="abs")
        evaluate = make_terminal_evaluator(policy=policy, paths=tensor_paths, book=tensor_book,
            initial_inventory=tensor_initial, objective=objective, risk_aversion=risk_aversion,
            target_side=target_side, compile_policy=train_config.compile,
            compile_evaluation=train_config.compile_evaluation,
            history_ids=paths.history_ids if train_config.compact_histories else None)

        with torch.no_grad():
            result, initial_risk = evaluate()
        initial_loss = float(initial_risk.value)
        loss_scale = max(1., abs(initial_loss))
        best_loss, best_model = initial_loss, physical_state()
        history = [{"update": 0, "loss": initial_loss, "best_loss": initial_loss,
                    "gradient_norm": 0., "maximum_probe_action_change": 0.}]
        completed_updates, stable, status = 0, 0, "training_limit"
        previous_loss = initial_loss
        probe = (result.positions[:32] * tensor_book.prices[:32]).detach().clone()
        if saved is not None:
            optimizer.load_state_dict(saved["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(saved["scheduler"])
            initial_loss, loss_scale = saved["initial_loss"], saved["loss_scale"]
            best_loss, best_model, history = saved["best_loss"], saved["best_model"], saved["history"]
            completed_updates, stable = saved["updates"], saved["stable_checks"]
            previous_loss, probe = saved["previous_loss"], saved["probe"].to(env.device)
            torch.set_rng_state(saved["torch_rng"].cpu())
            if env.device.type == "cuda" and saved["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
            if saved["status"] == "training_stable":
                status = "training_stable"
            elif saved["status"].endswith("review_required"):
                if train_config.retry_review:
                    stable = 0
                else:
                    status = saved["status"]
        stop = threading.Event()
        handlers = {}
        if threading.current_thread() is threading.main_thread():
            for number in (signal.SIGINT, signal.SIGTERM):
                handlers[number] = signal.signal(number, lambda signum, frame: stop.set())
        started, last_progress, gradient_norm = time.monotonic(), 0., 0.
        last_checkpoint = started
        accepted_since_check = bool(saved.get("accepted_since_check", False)) if saved is not None else False

        def save() -> None:
            nonlocal last_checkpoint
            physical_model = physical_state()
            coordinate_state = {} if coordinate_map is None else {
                "actor_coordinates": coordinate_map.coordinates.record(), "optimization_layout": coordinate_map.layout,
                "optimization_model": policy.state_dict()}
            if coordinate_map is not None:
                coordinate_state["actor_coordinate_binding"] = coordinate_checkpoint_binding(
                    coordinates=coordinate_state["actor_coordinates"], layout=coordinate_state["optimization_layout"],
                    optimization_state=coordinate_state["optimization_model"], physical_state=physical_model,
                    best_state=best_model)
            _atomic_checkpoint(checkpoint, {"identity": identity, "context": context,
                "code_version": TerminalPolicyConfig.version.unique_id48, "user_version": model_config.user_version,
                "model_config": asdict(model_config), "train_config": asdict(train_config),
                "as_of_date": as_of_date.isoformat(), "initialisation_date": None if initialisation_date is None else initialisation_date.isoformat(),
                "model": physical_model, "best_model": best_model, "optimizer": optimizer.state_dict(),
                "optimizer_type": train_config.optimizer,
                "acceptance_protocol": "lbfgs-training-descent-v1" if use_lbfgs else "adam-loss-action-v1",
                "accepted_since_check": accepted_since_check,
                "scheduler": None if scheduler is None else scheduler.state_dict(), "updates": completed_updates, "stable_checks": stable,
                "feature_mapping": None if feature_mapping is None else {
                    "matrix": feature_mapping.matrix.tolist(), "offset": feature_mapping.offset.tolist(),
                    "elapsed_step_offset": feature_mapping.elapsed_step_offset,
                    "old_subspace_projection_residual": feature_mapping.old_subspace_projection_residual},
                "initial_loss": initial_loss, "loss_scale": loss_scale, "best_loss": best_loss,
                "previous_loss": previous_loss, "probe": probe, "history": history, "status": status,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if env.device.type == "cuda" else None,
                **coordinate_state})
            last_checkpoint = time.monotonic()

        try:
            for update in range(completed_updates + 1, train_config.maximum_updates + 1):
                if status != "training_limit":
                    break
                step_details = {}
                if use_lbfgs:
                    before_parameters = parameter_vector(parameters)
                    closure_values = []
                    closure_count = 0

                    def closure() -> torch.Tensor:
                        nonlocal last_progress, closure_count
                        optimizer.zero_grad(set_to_none=True)
                        _, closure_risk = evaluate()
                        normalized_loss = closure_risk.value / loss_scale
                        if not bool(torch.isfinite(normalized_loss)):
                            raise RuntimeError("Non-finite L-BFGS terminal objective")
                        normalized_loss.backward()
                        gradients = gradient_vector(parameters)
                        if not closure_values:
                            closure_values.extend((float(normalized_loss.detach()), gradients.clone()))
                        closure_count += 1
                        now = time.monotonic()
                        if progress is not None and now - last_progress >= train_config.progress_seconds:
                            progress(f"terminal L-BFGS accepted_updates={completed_updates}/{train_config.maximum_updates} "
                                f"line_search_evaluations={closure_count} elapsed_s={now - started:.1f} remaining_s=unknown")
                            last_progress = now
                        return normalized_loss

                    optimizer.step(closure)
                    optimizer.zero_grad(set_to_none=True)
                    result, risk = evaluate()
                    (risk.value / loss_scale).backward()
                    accepted_gradient = gradient_vector(parameters)
                    physical_gradient = (accepted_gradient if coordinate_map is None else
                        coordinate_map.vector(accepted_gradient, gradient=True))
                    gradient_norm = float(torch.linalg.vector_norm(physical_gradient.to(torch.float64)))
                    displacement = parameter_vector(parameters) - before_parameters
                    physical_displacement = displacement if coordinate_map is None else coordinate_map.vector(displacement)
                    step_norm = float(torch.linalg.vector_norm(physical_displacement.to(torch.float64)))
                    initial_value, initial_gradient = closure_values
                    initial_slope = float(initial_gradient.to(torch.float64) @ displacement.to(torch.float64))
                    accepted_slope = float(accepted_gradient.to(torch.float64) @ displacement.to(torch.float64))
                    numerical_allowance = 64. * torch.finfo(env.fdtype).eps * max(1., abs(initial_value))
                    accepted_value = float(risk.value.detach()) / loss_scale
                    armijo = accepted_value <= initial_value + 1e-4 * initial_slope + numerical_allowance
                    curvature = abs(accepted_slope) <= .9 * abs(initial_slope) + numerical_allowance
                    changed = step_norm > 0.
                    resolved = abs(initial_slope) > numerical_allowance and initial_value - accepted_value > numerical_allowance
                    valid_step = changed and initial_slope < 0. and armijo and curvature and resolved
                    accepted_since_check = accepted_since_check or valid_step
                    if not valid_step:
                        status = ("stationary_review_required" if not changed
                            and gradient_norm <= train_config.gradient_tolerance else "line_search_review_required")
                    step_details = {"parameter_step_l2": step_norm, "line_search_armijo": armijo,
                        "line_search_curvature": curvature, "accepted_changed_step": valid_step,
                        "line_search_numerically_resolved": resolved, "line_search_allowance": numerical_allowance,
                        "initial_directional_change": initial_slope, "accepted_directional_change": accepted_slope}
                else:
                    optimizer.zero_grad(set_to_none=True)
                    _, risk = evaluate()
                    (risk.value / loss_scale).backward()
                    gradient_norm = clip_terminal_gradients(parameters, maximum_norm=train_config.gradient_clip)
                    optimizer.step()
                completed_updates = update
                checked = (update % train_config.check_every == 0 or update == train_config.maximum_updates
                           or stop.is_set() or status != "training_limit")
                if checked:
                    with torch.no_grad():
                        result, risk = evaluate()
                    current_loss = float(risk.value)
                    cash_residual = (0. if risk.cash_first_order_residual is None
                                     else float(torch.abs(risk.cash_first_order_residual)))
                    cash_tolerance = 64. * torch.finfo(env.fdtype).eps * (
                        1. + (risk_aversion if np.isfinite(risk_aversion) else 0.) * float(result.wealth.max() - result.wealth.min()))
                    if cash_residual > cash_tolerance:
                        raise RuntimeError(f"Power cash-root accuracy needs review: residual={cash_residual}, tolerance={cash_tolerance}")
                    current_probe = (result.positions[:32] * tensor_book.prices[:32]).detach().clone()
                    action_change = float(torch.max(torch.abs(current_probe - probe)))
                    loss_change = abs(current_loss - previous_loss) / max(1., abs(previous_loss))
                    stable = stable + 1 if (loss_change <= train_config.relative_loss_tolerance
                        and action_change <= train_config.dollar_action_tolerance
                        and (not use_lbfgs or accepted_since_check and status == "training_limit")) else 0
                    if current_loss < best_loss:
                        best_loss, best_model = current_loss, physical_state()
                    if scheduler is not None:
                        step_training_scheduler(scheduler, current_loss, tolerance=train_config.relative_loss_tolerance)
                    previous_loss, probe = current_loss, current_probe
                    history.append({"update": update, "loss": current_loss, "best_loss": best_loss,
                        "gradient_norm": gradient_norm, "maximum_probe_action_change": action_change})
                    history[-1].update(cash_first_order_residual=cash_residual,
                                      cash_first_order_tolerance=cash_tolerance)
                    history[-1].update(step_details)
                    ready = update >= train_config.minimum_updates and stable >= train_config.stable_checks
                    if use_lbfgs:
                        if ready and status == "training_limit":
                            state = optimizer.state[parameters[0]]
                            coordinate_arguments = {} if coordinate_map is None else {
                                "to_probe_coordinates": coordinate_map.vector,
                                "from_probe_coordinates": lambda vector: coordinate_map.vector(vector, to_physical=False)}
                            directions = (physical_gradient, physical_displacement,
                                state["d"] if coordinate_map is None else coordinate_map.vector(state["d"]))
                            certificate = probe_terminal_descent(parameters=parameters,
                                evaluate_loss=lambda: evaluate()[1].value,
                                directions=directions,
                                relative_distances=train_config.descent_probe_distances,
                                tolerance=train_config.relative_loss_tolerance * max(1., abs(current_loss)),
                                **coordinate_arguments)
                            history[-1].update(descent_probe=asdict(certificate),
                                maximum_probe_improvement=certificate.maximum_improvement)
                            if certificate.no_detected_descent and certificate.evaluations > 1:
                                status = "training_stable"
                                best_loss, best_model = current_loss, physical_state()
                        accepted_since_check = False
                    else:
                        low_rate = optimizer.param_groups[0]["lr"] <= train_config.minimum_learning_rate * (1. + 1e-12)
                        if ready and (low_rate or gradient_norm <= train_config.gradient_tolerance):
                            status = "training_stable"
                    if stop.is_set():
                        status = "checkpointed_stop"
                    if (train_config.checkpoint_seconds is None or status != "training_limit"
                            or time.monotonic() - last_checkpoint >= train_config.checkpoint_seconds):
                        save()
                now = time.monotonic()
                if progress is not None and (now - last_progress >= train_config.progress_seconds or checked and status != "training_limit"):
                    elapsed = now - started
                    remaining = elapsed * (train_config.maximum_updates - update) / max(update - (saved["updates"] if saved else 0), 1)
                    progress(f"terminal policy updates={update}/{train_config.maximum_updates} loss={previous_loss:.9g} "
                        f"best={best_loss:.9g} gradient={gradient_norm:.3g} status={status} "
                        f"elapsed_s={elapsed:.1f} remaining_budget_s~{remaining:.0f}")
                    last_progress = now
                if status != "training_limit":
                    break
            save()
        finally:
            for number, handler in handlers.items():
                signal.signal(number, handler)
            if coordinate_map is not None:
                coordinate_map.remove()
        policy.load_state_dict(best_model, strict=True)
        policy.fit_context, policy.as_of_date = context, as_of_date
        checkpoint.with_suffix(".history.json").write_text(json.dumps(history, indent=2) + "\n")
        return TerminalPolicyFit(policy=policy, status=status, updates=completed_updates, initial_loss=initial_loss,
            best_training_loss=best_loss, history=tuple(history), checkpoint_path=checkpoint,
            initialisation_date=initialisation_date)


def audit_fitted_policy(*, policy: TerminalPolicy, paths: TerminalPaths, book: PolicyBook,
                        initial_inventory: np.ndarray, objective: str, risk_aversion: float,
                        target_side: int) -> dict[str, float]:
    """Frozen-policy NumPy replay; usable on independent generated holdout paths."""
    book.validate_histories(paths)
    env = policy.env
    tensor_book, tensor_paths = book.tensor(env), TensorTerminalPaths.from_paths(paths, env=env)

    def replay(features, inventory, step):
        with torch.no_grad():
            return policy(torch.tensor(features, dtype=env.fdtype, device=env.device),
                torch.tensor(inventory, dtype=env.fdtype, device=env.device), step,
                book=tensor_book, costs=tensor_paths.costs[:, step]).cpu().numpy()

    evaluated = evaluate_terminal_policy(paths=paths, initial_inventory=initial_inventory,
        policy=replay, target_side=target_side, position_cap=None)
    with torch.no_grad():
        tensor_result = evaluate_tensor_policy(paths=tensor_paths,
            initial_inventory=torch.tensor(initial_inventory, dtype=env.fdtype, device=env.device),
            policy=policy.bind(book=tensor_book, costs=tensor_paths.costs), target_side=target_side)
    error = float(np.max(np.abs(evaluated.wealth - tensor_result.wealth.cpu().numpy())))
    tolerance = 256. * torch.finfo(env.fdtype).eps * max(1., float(np.max(np.abs(evaluated.wealth))))
    if error > tolerance:
        raise RuntimeError(f"Independent frozen terminal-policy replay differs: {error}")
    allocations = evaluated.positions * book.prices
    if np.max(np.abs(allocations)) > policy.config.instrument_cap * (1. + 64. * torch.finfo(env.fdtype).eps):
        raise RuntimeError("Frozen terminal-policy allocations violate current premium-dollar limits")
    risk = terminal_risk(evaluated.wealth, paths.weights, objective=objective, risk_aversion=risk_aversion)
    weights = paths.weights / paths.weights.sum()
    active = np.full(len(paths.gains), paths.initially_active, dtype=bool)
    inventory = np.broadcast_to(initial_inventory, evaluated.positions[:, 0].shape).copy()
    largest_turnover, forced_liquidations = 0., 0
    live_observations = np.zeros_like(book.prices, dtype=bool)
    for step in range(paths.gains.shape[1]):
        live = active[:, None] & paths.live_contracts[step]
        live_observations[:, step] = live
        forced_liquidations += int(np.sum(live & (book.prices[:, step] == 0.) & (inventory != 0.)))
        change = np.where(live, evaluated.positions[:, step] - inventory, 0.)
        largest_turnover = max(largest_turnover, float(np.abs(change).max()))
        active &= ~paths.terminate_after_step[:, step]
        inventory = np.where(active[:, None] & paths.live_contracts[step + 1], evaluated.positions[:, step], 0.)
    positive_live = book.prices[(book.prices > 0.) & live_observations]
    order = np.argsort(evaluated.wealth)
    cumulative = np.cumsum(weights[order])
    quantiles = evaluated.wealth[order[np.minimum(np.searchsorted(cumulative, [.01, .05]), len(order) - 1)]]
    tail_weights = np.minimum(np.maximum(.05 - (cumulative - weights[order]), 0.), weights[order])
    return {"risk": risk.value, "mean_wealth": risk.mean_wealth, "std_wealth": risk.std_wealth,
        "mean_entry_cost": float(weights @ evaluated.entry_cost), "maximum_replay_error": error,
        "maximum_premium_position": float(np.abs(allocations).max()),
        "maximum_gross_premium": float(np.abs(allocations).sum(axis=2).max()),
        "cap_fraction": float(np.mean((np.abs(allocations) >= policy.config.instrument_cap * (1. - 1e-6))[live_observations])) if live_observations.any() else 0.,
        "minimum_positive_live_price": float(positive_live.min()) if len(positive_live) else 0.,
        "maximum_spot_normalized_contract_coefficient": float(np.abs(evaluated.positions).max()),
        "maximum_spot_normalized_contract_turnover": largest_turnover,
        "maximum_total_entry_cost": float(evaluated.entry_cost.max()),
        "zero_price_forced_liquidations": forced_liquidations,
        "wealth_percentile_01": float(quantiles[0]), "wealth_percentile_05": float(quantiles[1]),
        "loss_expected_shortfall_95": -float(tail_weights @ evaluated.wealth[order]) / .05}
