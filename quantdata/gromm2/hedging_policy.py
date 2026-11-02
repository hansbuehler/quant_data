"""Affine-plus-deep recurrent hedges on an explicitly selected CPU or allocated GPU."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import signal
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.uniquehash import label_unique_filename48_8, named_unique_filename48_8
from cdxcore.util import is_filename
from cdxcore.version import version
from filelock import FileLock
from scipy.optimize import brentq

from quantdata.gromm2.rolling_audit import check_resource_headroom
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class HedgingNetworkConfig(FeedForwardConfig):
    """Versioned project specialization of the shared, unversioned FFW config."""

    activation: str = "SiLU"
    num_depth: int = 2
    num_width: int = 64


@version("0.1", dependencies=[HedgingNetworkConfig])
@dataclass(frozen=True, kw_only=True)
class HedgingPolicyConfig:
    """Logical policy, utility and numerical action regularization specification."""

    label: str
    feature_count: int
    option_expiry_days: tuple[int, ...]
    group: str = "gromm2/forward-straddle"
    user_version: str = "0.1"
    hidden_size: int = 32
    deep: FeedForwardConfig = HedgingNetworkConfig()
    risk_aversion: float = 1.0
    liability_units: float = 0.0
    position_cap: float = 10.0
    soft_cap_fraction: float = 0.8
    penalty_weight: float = 1.0e-4
    normalization_floor: float = 1.0e-4

    def __post_init__(self) -> None:
        verify_inp(bool(self.label) and (self.label.startswith("*") or is_filename(self.label)), lambda: f"invalid policy label {self.label!r}")
        verify_inp(bool(self.group) and all(part not in {"", ".", ".."} and is_filename(part) for part in self.group.split("/")), lambda: f"invalid group {self.group!r}")
        verify_inp(self.feature_count > 0 and self.hidden_size > 0, lambda: "feature and recurrent dimensions must be positive")
        verify_inp(bool(self.option_expiry_days) and all(expiry > 0 for expiry in self.option_expiry_days), lambda: "option expiries must be positive")
        verify_inp(np.isfinite(self.risk_aversion) and self.risk_aversion > 0.0, lambda: "risk aversion must be finite and positive")
        verify_inp(self.liability_units in {0.0, 1.0}, lambda: "this experiment supports empty or short-one-claim portfolios")
        verify_inp(np.isfinite(self.position_cap) and self.position_cap > 0.0, lambda: "position cap must be finite and positive")
        verify_inp(0.0 < self.soft_cap_fraction < 1.0 and self.penalty_weight >= 0.0, lambda: "invalid numerical action penalty")
        verify_inp(self.normalization_floor > 0.0, lambda: "normalization floor must be positive")

    @property
    def instrument_count(self) -> int:
        return len(self.option_expiry_days) + 1

    @property
    def filename(self) -> str:
        if self.label.startswith("*"):
            return label_unique_filename48_8(self.label[1:])
        return named_unique_filename48_8(self.label, self)


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    """Operational schedule; changing it does not invalidate a logical checkpoint."""

    seed: int = 20261027
    device: str = "cpu"
    precision: int = 32
    batch_size: int = 2048
    evaluation_batch_size: int = 4096
    learning_rate: float = 2.0e-3
    minimum_learning_rate: float = 1.0e-6
    maximum_epochs: int = 2000
    minimum_epochs: int = 100
    learning_rate_patience: int = 50
    convergence_patience: int = 100
    relative_improvement: float = 1.0e-5
    absolute_improvement: float = 1.0e-8
    gradient_clip: float = 5.0
    cpu_threads: int = 4
    compile: bool = False
    heartbeat_seconds: float = 60.0

    def __post_init__(self) -> None:
        verify_inp(self.device in {"cpu", "cuda"}, lambda: "device must be explicitly 'cpu' or 'cuda'")
        verify_inp(self.precision in {32, 64}, lambda: "policy precision must be 32 or 64")
        verify_inp(self.batch_size > 0 and self.evaluation_batch_size > 0 and self.cpu_threads > 0, lambda: "batch and thread counts must be positive")
        verify_inp(self.maximum_epochs > 0 and 0 < self.minimum_epochs <= self.maximum_epochs, lambda: "invalid epoch limits")
        verify_inp(0.0 < self.minimum_learning_rate <= self.learning_rate, lambda: "invalid learning rates")
        verify_inp(self.learning_rate_patience > 0 and self.convergence_patience > 0, lambda: "training patience must be positive")
        verify_inp(self.relative_improvement >= 0.0 and self.absolute_improvement >= 0.0 and self.gradient_clip > 0.0, lambda: "invalid convergence or clipping settings")
        verify_inp(0.0 < self.heartbeat_seconds <= 60.0, lambda: "heartbeat must occur at least every sixty seconds")


class HedgeOutput(NamedTuple):
    trading_gain: torch.Tensor
    penalty: torch.Tensor
    action_squared: torch.Tensor
    cap_fraction: torch.Tensor
    positions: torch.Tensor


def power_exponential_utility(value: torch.Tensor, risk_aversion: float) -> torch.Tensor:
    """Exponential positive branch and quadratic negative branch, with u'(0)=1."""
    verify_inp(np.isfinite(risk_aversion) and risk_aversion > 0.0, lambda: "risk aversion must be finite and positive")
    positive = torch.where(value >= 0.0, value, 0.0)
    negative = torch.where(value < 0.0, value, 0.0)
    positive_utility = -torch.expm1(-risk_aversion * positive) / risk_aversion
    negative_utility = negative - 0.5 * risk_aversion * negative.square()
    return torch.where(value >= 0.0, positive_utility, negative_utility)


def oce_value(pnl: torch.Tensor, translation: torch.Tensor, risk_aversion: float) -> torch.Tensor:
    """OCE at a supplied translation; the training maximization learns this scalar."""
    return translation + power_exponential_utility(pnl - translation, risk_aversion).mean()


def optimized_oce(pnl: np.ndarray, risk_aversion: float) -> tuple[float, float, np.ndarray]:
    """Optimize only the evaluation translation of an already frozen P&L sample."""
    values = np.asarray(pnl, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("OCE evaluation requires a nonempty finite P&L vector")
    verify_inp(np.isfinite(risk_aversion) and risk_aversion > 0.0, lambda: "invalid risk aversion")

    def derivative_residual(translation: float) -> float:
        centered = values - translation
        marginal = np.exp(-risk_aversion * np.maximum(centered, 0.0)) - risk_aversion * np.minimum(centered, 0.0)
        return float(marginal.mean() - 1.0)

    lower, upper = float(values.min()), float(values.max())
    translation = lower if lower == upper else float(brentq(derivative_residual, lower, upper, xtol=1.0e-12))
    centered = values - translation
    utility = np.where(centered >= 0.0, -np.expm1(-risk_aversion * np.maximum(centered, 0.0)) / risk_aversion,
                       np.minimum(centered, 0.0) - 0.5 * risk_aversion * np.minimum(centered, 0.0)**2)
    return float(translation + utility.mean()), translation, utility


def feature_normalization(features: np.ndarray, *, floor: float, progress: _Progress | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fit population mean/SD across training path and time axes, never holdout."""
    if features.ndim != 3 or not len(features) or floor <= 0.0:
        raise ValueError("normalization needs nonempty (path,time,feature) training data")
    count = 0
    mean = np.zeros(features.shape[-1], dtype=np.float64)
    squares = np.zeros_like(mean)
    for start in range(0, len(features), 1024):
        block = np.asarray(features[start:start + 1024], dtype=np.float64).reshape(-1, features.shape[-1])
        if not np.all(np.isfinite(block)):
            raise FloatingPointError("non-finite training features")
        block_mean = block.mean(axis=0)
        block_squares = np.sum((block - block_mean)**2, axis=0)
        total = count + len(block)
        difference = block_mean - mean
        squares += block_squares + difference**2 * count * len(block) / total
        mean += difference * len(block) / total
        count = total
        if progress is not None:
            progress.state.update(normalization_completed=min(start + 1024, len(features)),
                normalization_total=len(features))
            if progress.stop.is_set():
                raise InterruptedError("stopped during training-only normalization; existing checkpoints preserved")
    return mean, np.maximum(np.sqrt(squares / count), floor)


class HedgingPolicy(BaseModel):
    """Predict holdings before the price move, then update the recurrent memory."""

    def __init__(self, env: BaseEnvironment, *, config: HedgingPolicyConfig,
                 normalization_mean: np.ndarray, normalization_scale: np.ndarray) -> None:
        super().__init__(env=env)
        verify_inp(env.device.type in {"cpu", "cuda"}, lambda: "policy device must be CPU or allocated CUDA")
        if env.device.type == "cuda":
            _require_cuda_visibility()
        if normalization_mean.shape != (config.feature_count,) or normalization_scale.shape != normalization_mean.shape:
            raise ValueError("feature normalization has the wrong geometry")
        if not np.all(np.isfinite(normalization_mean)) or not np.all(np.isfinite(normalization_scale)) or np.any(normalization_scale <= 0.0):
            raise ValueError("feature normalization must be finite with strictly positive scales")
        self.config = config
        self.register_buffer("normalization_mean", torch.as_tensor(normalization_mean, dtype=env.fdtype, device=env.device).clone())
        self.register_buffer("normalization_scale", torch.as_tensor(normalization_scale, dtype=env.fdtype, device=env.device).clone())
        self.register_buffer("option_expiries", torch.as_tensor(config.option_expiry_days, dtype=env.idtype, device=env.device))
        combined_size = config.feature_count + config.hidden_size + config.instrument_count
        self.affine = FeedForward(env, num_input=combined_size, num_output=config.instrument_count, config=FeedForwardConfig.linear_config())
        self.deep = FeedForward(env, num_input=combined_size, num_output=config.instrument_count, config=config.deep)
        self.memory = torch.nn.LSTMCell(config.feature_count + 2 * config.instrument_count, config.hidden_size, dtype=env.fdtype, device=env.device)
        self.translation = torch.nn.Parameter(torch.zeros((), dtype=env.fdtype, device=env.device))
        for network in (self.affine, self.deep):
            final = network.In[-1] if len(network.In) else network.I0
            with torch.no_grad():
                final.weight.zero_()
                if final.bias is not None:
                    final.bias.zero_()

    def live_instruments(self, day: int) -> torch.Tensor:
        """Return the tradable mask, including spot in the original control."""
        return torch.cat((torch.ones(1, dtype=torch.bool, device=self.device), day < self.option_expiries))

    def forward(self, features: torch.Tensor, prices: torch.Tensor) -> HedgeOutput:
        """Return self-financing gains, including exactly one settlement cashflow."""
        assert features.ndim == 3 and features.shape[-1] == self.config.feature_count
        assert prices.shape == (features.shape[0], features.shape[1] + 1, self.config.instrument_count)
        self.verify(lambda: torch.isfinite(features).all() & torch.isfinite(prices).all(), lambda: "non-finite hedge input")
        normalized = (features.to(dtype=self.fdtype) - self.normalization_mean) / self.normalization_scale
        paths, periods = normalized.shape[:2]
        previous = normalized.new_zeros((paths, self.config.instrument_count))
        hidden = normalized.new_zeros((paths, self.config.hidden_size))
        cell = torch.zeros_like(hidden)
        positions, penalties, cap_fractions = [], [], []
        for day in range(periods):
            alive = self.live_instruments(day)
            previous = previous * alive
            combined = torch.cat((normalized[:, day], hidden, previous / self.config.position_cap), dim=-1)
            raw = self.affine(combined) + self.deep(combined)
            new_position = self.config.position_cap * torch.tanh(raw / self.config.position_cap) * alive
            excess = torch.relu(raw.abs() - self.config.soft_cap_fraction * self.config.position_cap) * alive
            penalties.append(excess.square().sum(dim=-1) / alive.sum().clamp_min(1))
            cap_fractions.append(((new_position.abs() >= 0.95 * self.config.position_cap) & alive).sum(dim=-1).to(dtype=self.fdtype) / alive.sum().clamp_min(1))
            positions.append(new_position)
            recurrent_input = torch.cat((normalized[:, day], previous / self.config.position_cap, new_position / self.config.position_cap), dim=-1)
            hidden, cell = self.memory(recurrent_input, (hidden, cell))
            previous = new_position
        holdings = torch.stack(positions, dim=1)
        gains = (holdings.to(dtype=prices.dtype) * (prices[:, 1:] - prices[:, :-1])).sum(dim=(1, 2))
        return HedgeOutput(gains, torch.stack(penalties, dim=1).mean(dim=1), holdings.square().sum(dim=-1).mean(dim=-1),
                           torch.stack(cap_fractions, dim=1).mean(dim=1), holdings)


@dataclass
class ScenarioData:
    """One fully completed scenario sample; native prices remain float64."""

    features: np.ndarray
    prices: np.ndarray
    payoff: np.ndarray
    option_expiry_days: tuple[int, ...]
    identity: dict[str, object]
    sha256: str
    option_strikes: tuple[float, ...] = ()
    option_is_call: tuple[bool, ...] = ()


def load_scenarios(path: Path, *, expected_split: str) -> ScenarioData:
    """Refuse partial files and preserve independent train/validation identities."""
    with h5py.File(path, "r") as stream:
        identity = json.loads(str(stream.attrs["identity"]))
        expected = int(identity["config"]["paths"])
        if identity.get("split") != expected_split or int(stream.attrs.get("completed_paths", 0)) != expected:
            raise ValueError(f"{expected_split} scenarios are incomplete or incorrectly labelled")
        features = stream["features"][:]
        prices = np.asarray(stream["prices"][:], dtype=np.float64)
        payoff = np.asarray(stream["payoff"][:], dtype=np.float64)
        expiries = tuple(map(int, stream["option_expiry_days"][:]))
        strikes = tuple(map(float, stream["option_strikes"][:]))
        is_call = tuple(map(bool, stream["option_is_call"][:]))
        if len(strikes) != len(expiries) or len(is_call) != len(expiries) or not np.all(np.isfinite(strikes)) or np.any(np.asarray(strikes) <= 0.0):
            raise ValueError("scenario option-book metadata is invalid")
        if not np.array_equal(np.asarray(is_call), np.asarray(strikes) >= 1.0):
            raise ValueError("the fixed book must use one inception-OTM option side per strike")
        if features.ndim != 3 or prices.shape != (expected, features.shape[1] + 1, len(expiries) + 1) or payoff.shape != (expected,) or features.shape[0] != expected:
            raise ValueError("scenario arrays have incompatible path/time/instrument dimensions")
        if expected < 4 or expected % 2 or not all(np.all(np.isfinite(array)) for array in (features, prices, payoff)) or np.any(prices[..., 0] <= 0.0):
            raise ValueError("scenarios require finite values, positive spots and complete antithetic pairs")
        for option, expiry in enumerate(expiries, start=1):
            if expiry < features.shape[1] and np.any(prices[:, expiry + 1:, option] != 0.0):
                raise ValueError("expired option marks must be zero after cash settlement")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return ScenarioData(features, prices, payoff, expiries, identity, digest.hexdigest(), strikes, is_call)


def pair_statistics(values: np.ndarray) -> dict[str, float | int]:
    """Adjacent antithetic partners form the independent units for mean errors."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 4 or len(values) % 2 or not np.all(np.isfinite(values)):
        raise ValueError("statistics require at least two complete finite antithetic pairs")
    paired = 0.5 * (values[0::2] + values[1::2])
    mean = float(paired.mean())
    standard_error = float(paired.std(ddof=1) / np.sqrt(len(paired)))
    return {"mean": mean, "standard_error": standard_error, "pair_count": len(paired),
            "path_standard_deviation": float(values.std(ddof=1)),
            "confidence_95_lower": mean - 1.959963984540054 * standard_error,
            "confidence_95_upper": mean + 1.959963984540054 * standard_error}


class _Progress:
    def __init__(self, *, label: str, interval: float) -> None:
        self.label, self.interval = label, interval
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.closed = threading.Event()
        self.checkpoint_requested = threading.Event()
        self.state: dict[str, object] = {"stage": "initializing", "epoch": 0, "completed": 0, "total": 0}
        self.initial_completed = 0
        self.resource_error: BaseException | None = None
        self.resource_cleanup: Callable[[], bool] | None = None
        self.resource_release: Callable[[], None] | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.closed.wait(self.interval):
            try:
                check_resource_headroom()
                cleanup = self.resource_cleanup
                if cleanup is not None:
                    cleanup()
            except BaseException as error:
                release = self.resource_release
                if release is not None:
                    try:
                        release()
                    except BaseException:
                        pass
                self._resource_stop(error)
                return
            self.checkpoint_requested.set()
            self.emit()

    def _resource_stop(self, error: BaseException) -> None:
        self.resource_error = error
        self.stop.set()
        self.checkpoint_requested.set()
        self.emit(event_detail="resource-stop-requested", graceful_deadline_seconds=60, message=str(error))
        if not self.closed.wait(60.0):
            self.emit(event_detail="resource-stop-deadline-exceeded", final_checkpoint_trustworthy=False,
                      recovery="Only the last fully replaced periodic checkpoint is trustworthy; no save retry under pressure.")
            os.kill(os.getpid(), signal.SIGKILL)

    def emit(self, **extra: object) -> None:
        elapsed = time.monotonic() - self.started
        completed, total = int(self.state["completed"]), int(self.state["total"])
        session_completed = completed - self.initial_completed
        state = {**self.state, "event": "hedging_progress", "policy": self.label, "elapsed_seconds": elapsed,
                 "remaining_seconds_to_safety_ceiling": (total - completed) * elapsed / session_completed if session_completed > 0 else None, **extra}
        print(json.dumps(state, allow_nan=False), flush=True)


def _atomic_torch_save(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _batch(data: ScenarioData, indices: slice | np.ndarray, *, env: BaseEnvironment) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = [data.features[indices], data.prices[indices], data.payoff[indices]]
    arrays = [value if value.flags.writeable else value.copy() for value in values]
    return (torch.as_tensor(arrays[0], dtype=env.fdtype, device=env.device),
            torch.as_tensor(arrays[1], dtype=torch.float64, device=env.device),
            torch.as_tensor(arrays[2], dtype=torch.float64, device=env.device))


def _require_cuda_visibility() -> str:
    """Require the caller's gpuctl allocation before any CUDA runtime query."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or visible.lower() in {"-1", "none", "void"}:
        raise RuntimeError("CUDA training requires a gpuctl allocation with nonempty CUDA_VISIBLE_DEVICES; no GPU is probed or chosen automatically")
    return visible


def training_environment(config: TrainConfig) -> BaseEnvironment:
    """Create only the requested backend; CUDA never silently falls back to CPU."""
    if config.device == "cuda":
        _require_cuda_visibility()
        env = BaseEnvironment(device=torch.device("cuda", 0), precision=config.precision, debug=False)
        if env.device.type != "cuda":
            raise RuntimeError("the allocated CUDA backend is unavailable; refusing to fall back")
        torch.cuda.set_device(env.device)
        memory_fraction = getattr(config, "cuda_memory_fraction", None)
        if memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(memory_fraction, device=env.device)
        torch.cuda.manual_seed(config.seed)
        return env
    return BaseEnvironment(device="cpu", precision=config.precision, debug=False)


def _full_training_objective(model: HedgingPolicy, forward, data: ScenarioData, *, claim_price: float,
                             batch_size: int, progress: _Progress) -> tuple[float, float]:
    total, penalty_total = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data.payoff), batch_size):
            features, prices, payoff = _batch(data, slice(start, start + batch_size), env=model.env)
            result = forward(features, prices)
            pnl = result.trading_gain + model.config.liability_units * (claim_price - payoff)
            loss = -oce_value(pnl, model.translation, model.config.risk_aversion) + model.config.penalty_weight * result.penalty.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite full-training objective")
            total += float(loss) * len(features)
            penalty_total += float(result.penalty.sum())
            if progress.stop.is_set():
                raise InterruptedError("training stopped during full-population objective evaluation")
    return total / len(data.payoff), penalty_total / len(data.payoff)


def fit_policy(data: ScenarioData, *, directory: Path, config: HedgingPolicyConfig, train_config: TrainConfig,
               claim_price: float, normalization: tuple[np.ndarray, np.ndarray], initial_model: HedgingPolicy | None = None,
               just_load: bool = False) -> tuple[HedgingPolicy, dict[str, object]]:
    """Train using every training path; holdout data is not accepted by this API."""
    check_resource_headroom()
    torch.set_num_threads(train_config.cpu_threads)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.pt"
    env = training_environment(train_config)
    torch.random.default_generator.manual_seed(train_config.seed)
    model = HedgingPolicy(env, config=config, normalization_mean=normalization[0], normalization_scale=normalization[1])
    identity = {"config": asdict(config), "training_scenarios_sha256": data.sha256, "claim_price": claim_price,
                "code_version": config.version.unique_id48, "user_version": config.user_version}
    if initial_model is not None:
        model.load_state_dict(initial_model.state_dict())
        with torch.no_grad():
            model.translation.zero_()
    if just_load:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("identity") != identity:
            raise ValueError("read-only policy load has incompatible versions, data or geometry")
        model.load_state_dict(checkpoint["best_state"])
        return model, {**checkpoint["report"], "device": str(env.device), "backend": env.device.type,
            "read_only_load": True, "loaded_from_checkpoint_device": checkpoint.get("runtime_device", "cpu")}
    progress = _Progress(label=config.label, interval=train_config.heartbeat_seconds)
    old_handlers = {sig: signal.signal(sig, lambda *_: progress.stop.set()) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with FileLock(str(checkpoint_path) + ".lock", timeout=0):
            optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                patience=train_config.learning_rate_patience - 1, threshold=train_config.absolute_improvement, threshold_mode="abs",
                min_lr=train_config.minimum_learning_rate)
            generator = np.random.default_rng(train_config.seed)
            state: dict[str, object] = {"epoch": 0, "next_batch": 0, "order": None, "best_objective": float("inf"),
                "best_epoch": 0, "best_state": copy.deepcopy(model.state_dict()), "history": [], "stop_reason": "running",
                "converged": False, "last_gradient_norm": 0.0,
                "initialization": "zero-action" if initial_model is None else "corresponding-empty-policy",
                "initialization_policy_label": None if initial_model is None else initial_model.config.label}
            if checkpoint_path.is_file():
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                saved_identity = checkpoint.get("identity", {})
                if saved_identity.get("code_version") != identity["code_version"] or saved_identity.get("user_version") != identity["user_version"]:
                    warnings.warn("Policy checkpoint version changed; cold-starting without deleting the old file", stacklevel=2)
                    stale = directory / f"checkpoint-incompatible-{time.time_ns()}.pt"
                    os.replace(checkpoint_path, stale)
                elif saved_identity != identity:
                    raise ValueError("logical checkpoint identity changed; use a new run or policy directory")
                else:
                    model.load_state_dict(checkpoint["model"])
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    scheduler.load_state_dict(checkpoint["scheduler"])
                    generator.bit_generator.state = checkpoint["numpy_rng"]
                    torch.set_rng_state(checkpoint["torch_rng"])
                    if env.device.type == "cuda":
                        if checkpoint.get("torch_cuda_rng") is not None:
                            torch.cuda.set_rng_state(checkpoint["torch_cuda_rng"].cpu(), device=env.device)
                        else:
                            torch.cuda.manual_seed(train_config.seed)
                            print(json.dumps({"event": "hedging_rng_backend_warm_start", "policy": config.label,
                                "device": str(env.device), "message": "No CUDA RNG in the source checkpoint; seeded the allocated device without cold-starting model weights."}), flush=True)
                    state = checkpoint["training_state"]
            forward = torch.compile(model) if train_config.compile else model
            if int(state["epoch"]) < train_config.maximum_epochs and not bool(state["converged"]):
                state["stop_reason"] = "running"
            progress.initial_completed = int(state["epoch"]) * len(data.payoff) + int(state["next_batch"])
            progress.state.update(total=train_config.maximum_epochs * len(data.payoff), completed=progress.initial_completed)
            progress.thread.start()
            checkpoint_save_failed, final_checkpoint_complete = False, False

            def save() -> None:
                nonlocal checkpoint_save_failed, final_checkpoint_complete
                checkpoint_save_failed, final_checkpoint_complete = True, False
                report = {"policy_config": asdict(config), "train_config": asdict(train_config), "training_scenarios_sha256": data.sha256,
                    "epoch": int(state["epoch"]), "best_epoch": int(state["best_epoch"]),
                    "best_training_objective": float(state["best_objective"]) if np.isfinite(state["best_objective"]) else None,
                    "stop_reason": state["stop_reason"], "converged": state["converged"], "initialization": state["initialization"],
                    "initialization_policy_label": state["initialization_policy_label"],
                    "device": str(env.device), "backend": env.device.type,
                    "data_staging": "Host scenario arrays with per-minibatch transfers to the selected device.",
                    "normalization": "training path/time population mean and SD only; no validation fitting",
                    "convergence_criterion": "training-objective plateau at minimum learning rate, not a stationarity certificate",
                    "numerical_regularization": "tanh position cap plus separate soft raw-action penalty; no economic trading costs"}
                _atomic_torch_save(checkpoint_path, {"identity": identity, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "numpy_rng": generator.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "torch_cuda_rng": torch.cuda.get_rng_state(device=env.device) if env.device.type == "cuda" else None,
                    "runtime_device": str(env.device), "training_state": state, "best_state": state["best_state"], "report": report})
                final_checkpoint_complete = True
                _write_json(directory / "training_report.json", report)
                _write_json(directory / "training_history.json", {"epochs": state["history"]})
                checkpoint_save_failed = False

            try:
                while int(state["epoch"]) < train_config.maximum_epochs and not bool(state["converged"]):
                    epoch = int(state["epoch"])
                    if state["order"] is None:
                        state["order"] = generator.permutation(len(data.payoff))
                        state["next_batch"] = 0
                    model.train()
                    for start in range(int(state["next_batch"]), len(data.payoff), train_config.batch_size):
                        indices = state["order"][start:start + train_config.batch_size]
                        features, prices, payoff = _batch(data, indices, env=env)
                        optimizer.zero_grad(set_to_none=True)
                        result = forward(features, prices)
                        pnl = result.trading_gain + config.liability_units * (claim_price - payoff)
                        loss = -oce_value(pnl, model.translation, config.risk_aversion) + config.penalty_weight * result.penalty.mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError("non-finite training loss; no paths are discarded")
                        loss.backward()
                        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip, error_if_nonfinite=True)
                        optimizer.step()
                        state["last_gradient_norm"] = float(gradient_norm)
                        state["next_batch"] = min(start + train_config.batch_size, len(data.payoff))
                        progress.state.update(stage="training", epoch=epoch + 1,
                            completed=epoch * len(data.payoff) + int(state["next_batch"]), learning_rate=optimizer.param_groups[0]["lr"])
                        if progress.stop.is_set():
                            raise InterruptedError("training checkpointed at a complete optimizer update")
                        if progress.checkpoint_requested.is_set():
                            save()
                            progress.checkpoint_requested.clear()
                    progress.state.update(stage="training-objective-evaluation")
                    objective, penalty = _full_training_objective(model, forward, data, claim_price=claim_price,
                        batch_size=train_config.evaluation_batch_size, progress=progress)
                    tolerance = max(train_config.absolute_improvement, train_config.relative_improvement * abs(float(state["best_objective"])))
                    if not np.isfinite(state["best_objective"]) or objective < float(state["best_objective"]) - tolerance:
                        state.update(best_objective=objective, best_epoch=epoch + 1, best_state=copy.deepcopy(model.state_dict()))
                    previous_lr = optimizer.param_groups[0]["lr"]
                    scheduler.threshold = max(train_config.absolute_improvement,
                        train_config.relative_improvement * abs(float(state["best_objective"])))
                    scheduler.step(objective)
                    state.update(epoch=epoch + 1, next_batch=0, order=None)
                    state["history"].append({"epoch": epoch + 1, "objective": objective, "penalty": penalty,
                        "learning_rate": optimizer.param_groups[0]["lr"], "last_batch_gradient_norm": state["last_gradient_norm"]})
                    if optimizer.param_groups[0]["lr"] < previous_lr:
                        progress.emit(event_detail="learning-rate-review", objective=objective, learning_rate=optimizer.param_groups[0]["lr"])
                    if epoch + 1 >= train_config.minimum_epochs and optimizer.param_groups[0]["lr"] <= train_config.minimum_learning_rate * (1.0 + 1.0e-12) and epoch + 1 - int(state["best_epoch"]) >= train_config.convergence_patience:
                        state.update(converged=True, stop_reason="training_objective_plateau_at_minimum_learning_rate")
                    save()
                    progress.emit(training_objective=objective, penalty=penalty)
                if not state["converged"]:
                    state["stop_reason"] = "maximum_epoch_safety_limit_not_certified_converged"
                save()
            except BaseException:
                if checkpoint_save_failed:
                    progress.emit(event_detail="checkpoint-save-failed-no-retry", final_checkpoint_trustworthy=final_checkpoint_complete,
                        recovery="Use the last fully replaced periodic checkpoint if the final checkpoint did not finish.")
                    raise
                state["stop_reason"] = "resource_pressure_checkpointed_stop" if progress.resource_error is not None else (
                    "checkpointed_stop" if progress.stop.is_set() else "checkpointed_exception")
                save()
                raise
            model.load_state_dict(state["best_state"])
            model.eval()
            return model, json.loads((directory / "training_report.json").read_text())
    finally:
        progress.closed.set()
        if progress.thread.is_alive():
            progress.thread.join(timeout=2.0)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def evaluate_policy(model: HedgingPolicy, data: ScenarioData, *, claim_price: float, batch_size: int) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Evaluate a frozen policy; optimizing its evaluation translation cannot alter it."""
    arrays = {name: np.empty(len(data.payoff), dtype=np.float64) for name in ("pnl", "trading_gain", "action_squared", "penalty", "cap_fraction")}
    minimum_position = np.full(model.config.instrument_count, np.inf)
    maximum_position = np.full(model.config.instrument_count, -np.inf)
    model.eval()
    started, last_progress = time.monotonic(), time.monotonic()
    with torch.no_grad():
        for start in range(0, len(data.payoff), batch_size):
            features, prices, payoff = _batch(data, slice(start, start + batch_size), env=model.env)
            result = model(features, prices)
            values = {"pnl": result.trading_gain + model.config.liability_units * (claim_price - payoff),
                "trading_gain": result.trading_gain, "action_squared": result.action_squared,
                "penalty": result.penalty, "cap_fraction": result.cap_fraction}
            for name, value in values.items():
                arrays[name][start:start + len(features)] = value.detach().cpu().numpy()
            minimum_position = np.minimum(minimum_position, result.positions.amin(dim=(0, 1)).cpu().numpy())
            maximum_position = np.maximum(maximum_position, result.positions.amax(dim=(0, 1)).cpu().numpy())
            if time.monotonic() - last_progress >= 60.0:
                check_resource_headroom()
                completed = start + len(features)
                elapsed = time.monotonic() - started
                print(json.dumps({"event": "hedging_frozen_evaluation", "policy": model.config.label, "completed": completed,
                    "total": len(data.payoff), "elapsed_seconds": elapsed, "remaining_seconds": elapsed * (len(data.payoff) - completed) / completed}), flush=True)
                last_progress = time.monotonic()
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise FloatingPointError("non-finite frozen-policy result")
    oce, translation, utility = optimized_oce(arrays["pnl"], model.config.risk_aversion)
    unhedged = model.config.liability_units * (claim_price - data.payoff)
    unhedged_oce, unhedged_translation, unhedged_utility = optimized_oce(unhedged, model.config.risk_aversion)
    improvement_influence = utility - unhedged_utility + translation - unhedged_translation
    return {"device": str(model.device), "backend": model.device.type,
        "pnl": pair_statistics(arrays["pnl"]), "trading_gain": pair_statistics(arrays["trading_gain"]),
        "oce": oce, "oce_translation": translation, "oce_mean_utility_standard_error": pair_statistics(utility)["standard_error"],
        "unhedged_oce": unhedged_oce, "oce_improvement_over_unhedged": oce - unhedged_oce,
        "oce_improvement_over_unhedged_statistics": pair_statistics(improvement_influence),
        "oce_comparison_uncertainty": "Antithetic-pair influence-function standard error for frozen-policy OCE minus the fixed zero-action baseline, with each evaluation translation optimized only for its metric.",
        "mean_action_squared": float(arrays["action_squared"].mean()), "mean_cap_near_binding_fraction": float(arrays["cap_fraction"].mean()),
        "near_cap_path_fraction": float(np.mean(arrays["cap_fraction"] > 0.0)), "position_cap": model.config.position_cap,
        "near_cap_absolute_position_threshold": 0.95 * model.config.position_cap,
        "near_cap_fraction_definition": "Path/time mean of the fraction of live instruments with absolute holdings at least 95% of the cap.",
        "mean_numerical_penalty": float(arrays["penalty"].mean()), "penalty_weight": model.config.penalty_weight,
        "minimum_position_by_instrument": minimum_position.tolist(), "maximum_position_by_instrument": maximum_position.tolist()}, arrays


def run_hedging_experiment(run_directory: Path, *, train_config: TrainConfig = TrainConfig(),
                           risk_aversions: tuple[float, ...] = (1.0, 10.0),
                           policy_config: HedgingPolicyConfig | None = None, just_load: bool = False,
                           freeze_empty_risk_aversions: tuple[float, ...] = ()) -> dict[str, object]:
    """Train empty/claim policies first, then use an independent, untouched sample."""
    if any(value not in risk_aversions for value in freeze_empty_risk_aversions):
        raise ValueError("frozen empty policies must belong to the requested risk aversions")
    check_resource_headroom()
    train = load_scenarios(run_directory / "train.h5", expected_split="train")
    base = policy_config or HedgingPolicyConfig(label="forward-straddle", feature_count=train.features.shape[-1], option_expiry_days=train.option_expiry_days)
    if base.feature_count != train.features.shape[-1] or base.option_expiry_days != train.option_expiry_days:
        raise ValueError("policy geometry differs from the training option book")
    normalization = feature_normalization(train.features, floor=base.normalization_floor)
    claim_price = float(train.payoff.mean())
    models: list[tuple[HedgingPolicy, dict[str, object], Path]] = []
    for risk_aversion in risk_aversions:
        empty_model = None
        for liability_units, suffix in ((0.0, "empty"), (1.0, "claim")):
            config = replace(base, label=f"{base.label}-{suffix}-lambda{risk_aversion:g}", risk_aversion=risk_aversion, liability_units=liability_units)
            directory = run_directory / "policies" / config.filename
            frozen_empty = liability_units == 0.0 and risk_aversion in freeze_empty_risk_aversions
            model, training = fit_policy(train, directory=directory, config=config, train_config=train_config,
                claim_price=claim_price, normalization=normalization, initial_model=empty_model if liability_units else None,
                just_load=just_load or frozen_empty)
            if frozen_empty:
                training = {**training, "frozen_by_user_request": True,
                    "freeze_interpretation": "Best training-only checkpoint frozen without asserting convergence; payoff policy starts from these weights."}
                _write_json(directory / "user_frozen_policy.json", training)
            if not frozen_empty and training["stop_reason"] not in {"training_objective_plateau_at_minimum_learning_rate", "maximum_epoch_safety_limit_not_certified_converged"}:
                raise RuntimeError("the policy schedule must finish before opening the final untouched validation sample")
            if liability_units == 0.0:
                empty_model = model
            models.append((model, training, directory))
    validation = load_scenarios(run_directory / "validation.h5", expected_split="validation")
    if (train.identity.get("model") != validation.identity.get("model") or train.features.shape[1:] != validation.features.shape[1:]
        or train.option_expiry_days != validation.option_expiry_days or train.option_strikes != validation.option_strikes
        or train.option_is_call != validation.option_is_call):
        raise ValueError("validation must come from the same model and option book")
    if train.sha256 == validation.sha256 or np.array_equal(train.prices, validation.prices):
        raise ValueError("validation paths must be independent, not copies of training scenarios")
    records = []
    empty_validation: dict[float, np.ndarray] = {}
    for model, training, directory in models:
        train_metrics, _ = evaluate_policy(model, train, claim_price=claim_price, batch_size=train_config.evaluation_batch_size)
        validation_metrics, scores = evaluate_policy(model, validation, claim_price=claim_price, batch_size=train_config.evaluation_batch_size)
        if model.config.liability_units == 0.0:
            empty_validation[model.config.risk_aversion] = scores["trading_gain"]
        else:
            incremental = claim_price - validation.payoff + scores["trading_gain"] - empty_validation[model.config.risk_aversion]
            incremental_oce, _, _ = optimized_oce(incremental, model.config.risk_aversion)
            validation_metrics["claim_minus_empty_hedge"] = {"pnl": pair_statistics(incremental), "oce": incremental_oce}
            scores["incremental_hedge_pnl"] = incremental
        np.savez_compressed(directory / "validation_scores.npz", **scores)
        record = {"config": asdict(model.config), "training": training, "train_metrics": train_metrics,
                  "validation_metrics": validation_metrics, "directory": str(directory)}
        _write_json(directory / "result.json", record)
        records.append(record)
    report = {"schema": "gromm2-forward-straddle-deep-hedging-v1", "claim": "abs(S_20/S_10-1)",
        "claim_price_training_only": pair_statistics(train.payoff), "independent_validation_price": pair_statistics(validation.payoff),
        "train_paths": len(train.payoff), "validation_paths": len(validation.payoff), "backend": train_config.device,
        "device": str(models[0][0].device),
        "option_book": {"expiry_business_days": train.option_expiry_days, "inception_relative_strikes": train.option_strikes,
                        "is_call": train.option_is_call, "price_column_zero": "spot"},
        "train_scenarios_sha256": train.sha256, "validation_scenarios_sha256": validation.sha256,
        "validation_protocol": "All policy weights, stopping, checkpoints, normalization and initial claim price fixed using training only. Validation translation is optimized only to evaluate the frozen P&L distribution.",
        "policies": records}
    _write_json(run_directory / "hedging_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--threads", default=4, type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="CUDA requires launching this command inside a gpuctl allocation")
    parser.add_argument("--batch-size", default=2048, type=int)
    parser.add_argument("--maximum-epochs", default=2000, type=int)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--just-load", action="store_true")
    parser.add_argument("--freeze-empty-risk-aversions", nargs="+", type=float, default=(),
                        help="Explicitly freeze existing empty-policy checkpoints at these risk aversions, then train their payoff hedges")
    arguments = parser.parse_args()
    run_hedging_experiment(arguments.run_directory, train_config=TrainConfig(cpu_threads=arguments.threads,
        device=arguments.device, batch_size=arguments.batch_size, maximum_epochs=arguments.maximum_epochs, compile=arguments.compile),
        just_load=arguments.just_load, freeze_empty_risk_aversions=tuple(arguments.freeze_empty_risk_aversions))


if __name__ == "__main__":
    main()
