"""Cross-fitted conditional no-arbitrage audit for residual-clock RN paths."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.util import is_filename
from cdxcore.version import version
from filelock import FileLock

from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig

_CACHE_SCHEMA = "dysanos-residual-clock-rn-oce-path-cache-v1"
_CHECKPOINT_SCHEMA = "dysanos-residual-clock-rn-conditional-audit-v3"
_REPORT_SCHEMA = "dysanos-residual-clock-rn-conditional-audit-report-v2"
_NUM_EXPIRIES = 4
_NUM_STRIKES = 12
_NUM_INSTRUMENTS = 1 + _NUM_EXPIRIES * _NUM_STRIKES


@version("0.1", dependencies=[FeedForwardConfig])
@dataclass(frozen=True, kw_only=True)
class ResidualClockRnPolicyConfig:
    """Geometry of the uncapped conditional action sieve."""

    label: str
    group: str = "flex/residual-clock-rn-conditional-audit"
    user_version: str = "0.1"
    state_size: int
    num_expiries: int = _NUM_EXPIRIES
    num_strikes: int = _NUM_STRIKES
    include_calendar_time: bool = True
    expiry_specific_linear_skip: bool = True
    network: FeedForwardConfig = FeedForwardConfig(
        activation="SiLU",
        num_depth=2,
        num_width=128,
    )

    def __post_init__(self) -> None:
        verify_inp(
            is_filename(self.label),
            lambda: f"policy label must be a filename, found {self.label!r}",
        )
        verify_inp(self.state_size > 0, lambda: "state_size must be positive")
        verify_inp(
            self.num_expiries > 0 and self.num_strikes > 0,
            lambda: "expiry and strike counts must be positive",
        )
        verify_inp(
            self.include_calendar_time,
            lambda: "the residual-clock audit requires calendar time",
        )
        verify_inp(
            self.expiry_specific_linear_skip,
            lambda: "the residual-clock audit requires expiry-specific affine skips",
        )


class ResidualClockRnPolicy(BaseModel):
    """Shared two-layer SiLU policy plus one affine skip per expiry."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: ResidualClockRnPolicyConfig,
        state_mean: torch.Tensor,
        state_scale: torch.Tensor,
        log_price_mean: torch.Tensor,
        log_price_scale: torch.Tensor,
    ) -> None:
        super().__init__(env)
        if state_mean.shape != (config.state_size,) or state_scale.shape != (config.state_size,):
            raise ValueError("state normalizers do not match the configured state size")
        if log_price_mean.shape != (config.num_strikes,) or log_price_scale.shape != (
            config.num_strikes,
        ):
            raise ValueError("price normalizers do not match the strike count")
        if not bool(
            torch.all(torch.isfinite(state_mean))
            and torch.all(torch.isfinite(state_scale) & (state_scale > 0.0))
            and torch.all(torch.isfinite(log_price_mean))
            and torch.all(torch.isfinite(log_price_scale) & (log_price_scale > 0.0))
        ):
            raise ValueError("policy normalizers must be finite with positive scales")
        self.config = config
        self.register_buffer("state_mean", state_mean.detach().clone())
        self.register_buffer("state_scale", state_scale.detach().clone())
        self.register_buffer("log_price_mean", log_price_mean.detach().clone())
        self.register_buffer("log_price_scale", log_price_scale.detach().clone())
        feature_size = config.state_size + config.num_strikes + 1
        head_size = config.num_strikes + 1
        self.network = FeedForward(
            env,
            feature_size,
            head_size,
            config.network,
        )
        self.linear_skip = torch.nn.ModuleList(
            torch.nn.Linear(
                feature_size,
                head_size,
                bias=True,
                dtype=env.fdtype,
                device=env.device,
            )
            for _ in range(config.num_expiries)
        )
        final_layer = self.network.I0 if len(self.network.In) == 0 else self.network.In[-1]
        with torch.no_grad():
            final_layer.weight.zero_()
            if final_layer.bias is not None:
                final_layer.bias.zero_()
            for layer in self.linear_skip:
                layer.weight.zero_()
                layer.bias.zero_()

    def forward(
        self,
        states: torch.Tensor,
        option_prices: torch.Tensor,
        *,
        calendar_time: torch.Tensor,
    ) -> torch.Tensor:
        """Return uncapped actions ordered as spot, then expiry-major options."""

        observations = len(states)
        if states.shape != (observations, self.config.state_size):
            raise ValueError("policy states have the wrong shape")
        if option_prices.shape != (
            observations,
            self.config.num_expiries,
            self.config.num_strikes,
        ):
            raise ValueError("policy option prices have the wrong shape")
        if calendar_time.shape != (observations,):
            raise ValueError("calendar_time must contain one value per observation")
        safe_prices = torch.clamp_min(
            option_prices,
            torch.finfo(option_prices.dtype).tiny,
        )
        normalized_state = (states - self.state_mean) / self.state_scale
        normalized_prices = (torch.log(safe_prices) - self.log_price_mean) / (self.log_price_scale)
        features = torch.cat(
            (
                normalized_state[:, None].expand(-1, self.config.num_expiries, -1),
                normalized_prices,
                calendar_time[:, None, None].expand(-1, self.config.num_expiries, 1),
            ),
            dim=-1,
        )
        raw_heads = self.network(features) + torch.stack(
            tuple(layer(features[:, expiry]) for expiry, layer in enumerate(self.linear_skip)),
            dim=1,
        )
        spot = torch.sum(raw_heads[..., :1], dim=1)
        options = raw_heads[..., 1:].reshape(observations, -1)
        return torch.cat((spot, options), dim=1)


@dataclass(frozen=True)
class ResidualClockRnPathCache:
    """Memory-mapped RN paths and their validated manifest."""

    directory: Path
    metadata_path: Path
    metadata: dict[str, Any]
    states: np.ndarray
    prices: np.ndarray
    returns: np.ndarray
    state_size: int
    num_paths: int
    num_steps: int


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _atomic_torch(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _resource_snapshot(output: Path) -> dict[str, Any]:
    """Read the host's canonical advisory resource checks through ``gpuctl``."""

    del output  # Resource paths and thresholds belong exclusively to resourced.
    try:
        process = subprocess.run(
            ("gpuctl", "resources", "--json"),
            check=False,
            capture_output=True,
            text=True,
            timeout=15.0,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "source": "gpuctl",
            "health": "WARN",
            "checks": [],
            "error": f"gpuctl resources could not be read: {error}",
        }
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip() or "no diagnostic output"
        return {
            "source": "gpuctl",
            "health": "WARN",
            "checks": [],
            "error": (
                f"gpuctl resources exited with status {process.returncode}: {detail}"
            ),
        }
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        return {
            "source": "gpuctl",
            "health": "WARN",
            "checks": [],
            "error": f"gpuctl resources returned invalid JSON: {error}",
        }
    checks = payload.get("checks") if isinstance(payload, dict) else None
    if not isinstance(checks, list):
        return {
            "source": "gpuctl",
            "health": "WARN",
            "checks": [],
            "error": "gpuctl resources JSON does not contain a checks list",
        }
    health_values = {
        str(check.get("health", "ERROR"))
        for check in checks
        if isinstance(check, Mapping)
    }
    if health_values & {"STOP", "UNAVAILABLE", "ERROR"}:
        health = "STOP"
    elif "WARN" in health_values:
        health = "WARN"
    else:
        health = "OK"
    return {
        "source": payload.get("source", "gpuctl"),
        "health": health,
        "checks": checks,
        "error": None,
        "timestamp": payload.get("timestamp"),
    }


def _check_resources(output: Path) -> dict[str, Any]:
    result = _resource_snapshot(output)
    if result["health"] == "STOP":
        stopped = [
            str(check.get("name", "unknown"))
            for check in result["checks"]
            if isinstance(check, Mapping)
            and check.get("health") in {"STOP", "UNAVAILABLE", "ERROR"}
        ]
        raise RuntimeError(
            "resourced requires the job to stop; failing checks: " + ", ".join(stopped)
        )
    return result


class _Progress:
    def __init__(self, output: Path, *, interval_seconds: float) -> None:
        self.path = output / "progress.jsonl"
        self.output = output
        self.interval_seconds = interval_seconds
        self.started = time.monotonic()
        self.last = -math.inf
        self.work_baselines: dict[tuple[object, ...], tuple[float, int]] = {}
        self.epoch_durations: dict[tuple[object, ...], list[float]] = {}
        self.last_stage_epoch: dict[tuple[object, ...], int] = {}

    def emit(self, payload: Mapping[str, Any], *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last < self.interval_seconds:
            return
        record_payload = dict(payload)
        completed = record_payload.get("completed_paths")
        total = record_payload.get("total_paths")
        if isinstance(completed, int) and isinstance(total, int) and total > 0:
            key = tuple(
                record_payload.get(name)
                for name in ("stage", "direction", "fit_stage", "epoch", "work")
            )
            baseline = self.work_baselines.setdefault(key, (now, completed))
            completed_since_baseline = completed - baseline[1]
            elapsed_since_baseline = now - baseline[0]
            if completed_since_baseline > 0 and elapsed_since_baseline > 0.0:
                paths_per_second = completed_since_baseline / elapsed_since_baseline
                record_payload["estimated_remaining_seconds"] = (
                    max(
                        total - completed,
                        0,
                    )
                    / paths_per_second
                )
        epoch = record_payload.get("epoch")
        total_epochs = record_payload.get("total_epochs")
        epoch_seconds = record_payload.get("epoch_seconds")
        if (
            record_payload.get("stage") == "train-epoch-complete"
            and isinstance(epoch, int)
            and isinstance(total_epochs, int)
            and isinstance(epoch_seconds, (int, float))
            and math.isfinite(epoch_seconds)
            and epoch_seconds >= 0.0
        ):
            key = (
                record_payload.get("diagnostic"),
                record_payload.get("fit_stage"),
            )
            previous_epoch = self.last_stage_epoch.get(key)
            if previous_epoch is not None and epoch <= previous_epoch:
                self.epoch_durations[key] = []
            durations = self.epoch_durations.setdefault(key, [])
            durations.append(float(epoch_seconds))
            self.last_stage_epoch[key] = epoch
            recent_durations = durations[-10:]
            remaining_epochs = max(total_epochs - epoch, 0)
            if record_payload.get("accepted_step_size") == 0.0:
                remaining_epochs = 0
            record_payload["estimated_remaining_seconds"] = (
                math.fsum(recent_durations) / len(recent_durations) * remaining_epochs
            )
            record_payload["estimated_remaining_scope"] = "current-fit-stage"
            record_payload["estimated_remaining_epochs"] = remaining_epochs
        record = {
            **record_payload,
            "elapsed_seconds": now - self.started,
            "resources": _check_resources(self.output),
        }
        line = json.dumps(record)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
        print(line, flush=True)
        self.last = now


def _metadata_integer(metadata: Mapping[str, Any], name: str) -> int | None:
    containers: list[Mapping[str, Any]] = [metadata]
    for key in (
        "policy_state",
        "policy_state_metadata",
        "state",
        "state_metadata",
        "geometry",
        "shapes",
        "feature_metadata",
    ):
        value = metadata.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    values = [container.get(name) for container in containers]
    integers = [int(value) for value in values if isinstance(value, (int, float))]
    if not integers:
        return None
    if any(value != integers[0] for value in integers[1:]):
        raise ValueError(f"metadata has inconsistent {name} values {integers}")
    return integers[0]


def _cache_array_path(
    directory: Path,
    metadata: Mapping[str, Any],
    *,
    metadata_names: tuple[str, ...],
    filenames: tuple[str, ...],
    required: bool = True,
) -> Path | None:
    files = metadata.get("files")
    configured: list[str] = []
    if isinstance(files, Mapping):
        for name in metadata_names:
            value = files.get(name)
            if isinstance(value, str):
                configured.append(value)
            elif isinstance(value, Mapping) and isinstance(value.get("path"), str):
                configured.append(str(value["path"]))
    candidates: list[Path] = []
    for path_value in configured:
        path = Path(path_value)
        candidates.append(path if path.is_absolute() else directory / path)
        if not path.is_absolute():
            candidates.append(path)
    candidates.extend(directory / filename for filename in filenames)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if required:
        joined = ", ".join(filenames)
        raise FileNotFoundError(f"RN path cache is missing one of: {joined}")
    return None


def _require_all_valid(
    path: Path,
    *,
    expected_shape: tuple[int, ...],
    label: str,
    path_batch_size: int = 2_048,
) -> None:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.shape != expected_shape:
        raise ValueError(f"{label} has shape {value.shape}, expected {expected_shape}")
    for first in range(0, len(value), path_batch_size):
        if not bool(np.all(value[first : first + path_batch_size] == 1)):
            raise ValueError(f"{label} contains incomplete path entries")


def load_residual_clock_rn_path_cache(directory: Path) -> ResidualClockRnPathCache:
    """Load and validate the self-describing RN path cache as memmaps."""

    metadata_candidates = (
        directory / "population.json",
        directory / "metadata.json",
        directory / "feature-cache.json",
    )
    metadata_path = next((path for path in metadata_candidates if path.exists()), None)
    if metadata_path is None:
        raise FileNotFoundError("RN path cache is missing metadata.json")
    metadata_value = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata_value, dict):
        raise ValueError("RN path cache metadata must be a JSON object")
    metadata: dict[str, Any] = metadata_value
    pairing = metadata.get(
        "antithetic_pair_ordering",
        metadata.get(
            "pair_ordering",
            metadata.get("antithetic_order", metadata.get("path_order")),
        ),
    )
    if isinstance(pairing, Mapping):
        pairing = pairing.get("layout", pairing.get("ordering"))
    explicit_adjacent = pairing in {
        "adjacent",
        "adjacent-pairs",
        "adjacent_antithetic_pairs",
        "adjacent antithetic pairs",
    }
    described_adjacent = isinstance(pairing, str) and (
        "adjacent" in pairing.lower() and "antithetic pair" in pairing.lower()
    )
    if not explicit_adjacent and not described_adjacent:
        raise ValueError("metadata must declare adjacent antithetic pair ordering")

    states = np.load(
        _cache_array_path(
            directory,
            metadata,
            metadata_names=("policy_states", "states"),
            filenames=("policy-states.npy", "states.npy", "train-states.npy"),
        ),
        mmap_mode="r",
        allow_pickle=False,
    )
    prices = np.load(
        _cache_array_path(
            directory,
            metadata,
            metadata_names=("option_prices", "prices"),
            filenames=(
                "policy-option-prices.npy",
                "prices.npy",
                "train-prices.npy",
            ),
        ),
        mmap_mode="r",
        allow_pickle=False,
    )
    returns = np.load(
        _cache_array_path(
            directory,
            metadata,
            metadata_names=("returns",),
            filenames=("returns.npy", "train-returns.npy"),
        ),
        mmap_mode="r",
        allow_pickle=False,
    )
    if states.ndim != 3:
        raise ValueError(f"states must have shape (N,T,state_size), got {states.shape}")
    num_paths, num_steps, state_size = states.shape
    if prices.shape != (num_paths, num_steps, _NUM_EXPIRIES, _NUM_STRIKES):
        raise ValueError(f"prices have incompatible shape {prices.shape}")
    if returns.shape != (num_paths, num_steps, _NUM_INSTRUMENTS):
        raise ValueError(f"returns have incompatible shape {returns.shape}")
    if num_paths < 4 or num_paths % 4:
        raise ValueError("cache path count must contain an even number of pairs")
    declared_state_size = _metadata_integer(metadata, "state_size")
    if declared_state_size is None:
        declared_state_size = _metadata_integer(metadata, "feature_size")
    if declared_state_size is None:
        declared_state_size = _metadata_integer(metadata, "policy_state_size")
    if declared_state_size is None:
        raise ValueError("metadata must declare state_size or policy-state feature_size")
    if declared_state_size != state_size:
        raise ValueError(f"metadata state size {declared_state_size} disagrees with {state_size}")
    for name, fallback, actual in (
        ("num_paths", None, num_paths),
        ("num_steps", "issue_days", num_steps),
    ):
        declared = _metadata_integer(metadata, name)
        if declared is None and fallback is not None:
            declared = _metadata_integer(metadata, fallback)
        if declared is not None and declared != actual:
            raise ValueError(f"metadata {name} {declared} disagrees with {actual}")
    feature_valid_path = _cache_array_path(
        directory,
        metadata,
        metadata_names=("feature_valid",),
        filenames=("policy-feature-valid.npy",),
        required=metadata_path.name == "population.json",
    )
    return_valid_path = _cache_array_path(
        directory,
        metadata,
        metadata_names=("return_valid",),
        filenames=("return-valid.npy",),
        required=metadata_path.name == "population.json",
    )
    if (feature_valid_path is None) != (return_valid_path is None):
        raise ValueError("RN path cache must provide both validity masks or neither")
    if feature_valid_path is not None and return_valid_path is not None:
        _require_all_valid(
            feature_valid_path,
            expected_shape=(num_paths, num_steps),
            label="policy feature-valid mask",
        )
        _require_all_valid(
            return_valid_path,
            expected_shape=(num_paths, num_steps, _NUM_INSTRUMENTS),
            label="return-valid mask",
        )
    return ResidualClockRnPathCache(
        directory=directory,
        metadata_path=metadata_path,
        metadata=metadata,
        states=states,
        prices=prices,
        returns=returns,
        state_size=state_size,
        num_paths=num_paths,
        num_steps=num_steps,
    )


def streaming_normalizers(
    states: np.ndarray,
    prices: np.ndarray,
    *,
    start: int,
    stop: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Estimate uniform training-fold normalizers without materializing paths."""

    if start < 0 or stop > len(states) or start >= stop:
        raise ValueError("invalid normalizer path interval")
    state_sum = np.zeros(states.shape[-1], dtype=np.float64)
    state_square_sum = np.zeros(states.shape[-1], dtype=np.float64)
    price_sum = np.zeros(prices.shape[-1], dtype=np.float64)
    price_square_sum = np.zeros(prices.shape[-1], dtype=np.float64)
    observations = 0
    price_observations = 0
    tiny = np.finfo(prices.dtype).tiny
    for first in range(start, stop, batch_size):
        last = min(first + batch_size, stop)
        state = np.asarray(states[first:last], dtype=np.float64)
        log_price = np.log(np.maximum(np.asarray(prices[first:last]), tiny))
        state_sum += np.sum(state, axis=(0, 1), dtype=np.float64)
        state_square_sum += np.sum(np.square(state), axis=(0, 1), dtype=np.float64)
        price_sum += np.sum(log_price, axis=(0, 1, 2), dtype=np.float64)
        price_square_sum += np.sum(np.square(log_price), axis=(0, 1, 2), dtype=np.float64)
        observations += state.shape[0] * state.shape[1]
        price_observations += log_price.shape[0] * log_price.shape[1] * log_price.shape[2]
    state_mean = state_sum / observations
    state_variance = np.maximum(
        state_square_sum / observations - np.square(state_mean),
        0.0,
    )
    price_mean = price_sum / price_observations
    price_variance = np.maximum(
        price_square_sum / price_observations - np.square(price_mean),
        0.0,
    )
    state_scale = np.sqrt(state_variance)
    price_scale = np.sqrt(price_variance)
    state_scale = np.where(state_scale > 1.0e-6, state_scale, 1.0)
    price_scale = np.where(price_scale > 1.0e-6, price_scale, 1.0)
    return {
        "state_mean": state_mean.astype(np.float32),
        "state_scale": state_scale.astype(np.float32),
        "log_price_mean": price_mean.astype(np.float32),
        "log_price_scale": price_scale.astype(np.float32),
    }


def _make_policy(
    *,
    state_size: int,
    normalizers: Mapping[str, np.ndarray],
    device: torch.device,
    seed: int,
    label: str,
) -> ResidualClockRnPolicy:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    env = BaseEnvironment(precision=32, device=device, debug=False)
    config = ResidualClockRnPolicyConfig(label=label, state_size=state_size)
    return ResidualClockRnPolicy(
        env,
        config=config,
        state_mean=torch.as_tensor(normalizers["state_mean"], device=device),
        state_scale=torch.as_tensor(normalizers["state_scale"], device=device),
        log_price_mean=torch.as_tensor(normalizers["log_price_mean"], device=device),
        log_price_scale=torch.as_tensor(normalizers["log_price_scale"], device=device),
    )


def _load_batch(
    cache: ResidualClockRnPathCache,
    *,
    first: int,
    last: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(np.array(cache.states[first:last], copy=True), device=device),
        torch.as_tensor(np.array(cache.prices[first:last], copy=True), device=device),
        torch.as_tensor(np.array(cache.returns[first:last], copy=True), device=device),
    )


def _actions_for_path_batch(
    policy: ResidualClockRnPolicy,
    states: torch.Tensor,
    prices: torch.Tensor,
) -> torch.Tensor:
    paths, steps, state_size = states.shape
    if state_size != policy.config.state_size:
        raise ValueError("path state size does not match the policy")
    calendar = torch.arange(steps, dtype=states.dtype, device=states.device) / max(steps - 1, 1)
    calendar = calendar[None].expand(paths, -1).reshape(-1)
    action = policy(
        states.reshape(paths * steps, state_size),
        prices.reshape(paths * steps, _NUM_EXPIRIES, _NUM_STRIKES),
        calendar_time=calendar,
    )
    return action.reshape(paths, steps, _NUM_INSTRUMENTS)


def _daily_gain(
    policy: ResidualClockRnPolicy,
    states: torch.Tensor,
    prices: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    action = _actions_for_path_batch(policy, states, prices)
    return torch.sum(action * returns - 0.5 * action.square(), dim=-1)


def squared_second_moment_reward(
    actions: torch.Tensor,
    returns: torch.Tensor,
    *,
    lambda_second_moment: float = 1.0,
) -> torch.Tensor:
    """Return the main2 squared-diagnostic reward at each path-time point."""

    if actions.shape != returns.shape or actions.ndim != 3:
        raise ValueError("actions and returns must be matching path-step-instrument tensors")
    if not math.isfinite(lambda_second_moment) or lambda_second_moment < 0.0:
        raise ValueError("lambda_second_moment must be finite and nonnegative")
    portfolio_return = torch.sum(actions * returns, dim=-1)
    action_energy = torch.sum(actions.square(), dim=-1)
    return (
        portfolio_return
        - 0.5 * action_energy
        - 0.5 * lambda_second_moment * portfolio_return.square()
    )


def _daily_squared_reward(
    policy: ResidualClockRnPolicy,
    states: torch.Tensor,
    prices: torch.Tensor,
    returns: torch.Tensor,
    *,
    lambda_second_moment: float,
) -> torch.Tensor:
    action = _actions_for_path_batch(policy, states, prices)
    return squared_second_moment_reward(
        action,
        returns,
        lambda_second_moment=lambda_second_moment,
    )


def log_mean_exp_negative_gain(
    gain: np.ndarray,
    *,
    batch_size: int,
) -> tuple[float, float]:
    """Return ``log(mean(exp(-gain)))`` and its exponential-tilt ESS."""

    if gain.ndim != 2 or gain.size == 0:
        raise ValueError("gain must be a nonempty path-time matrix")
    maximum = -math.inf
    for first in range(0, len(gain), batch_size):
        maximum = max(
            maximum,
            float(np.max(-np.asarray(gain[first : first + batch_size]))),
        )
    exponential_sum = 0.0
    squared_exponential_sum = 0.0
    for first in range(0, len(gain), batch_size):
        score = -np.asarray(gain[first : first + batch_size], dtype=np.float64)
        exponential = np.exp(score - maximum)
        exponential_sum += float(np.sum(exponential, dtype=np.float64))
        squared_exponential_sum += float(np.sum(np.square(exponential), dtype=np.float64))
    log_sum = maximum + math.log(exponential_sum)
    objective = log_sum - math.log(gain.size)
    ess = exponential_sum**2 / squared_exponential_sum
    return objective, ess


def _write_objective_cache(
    policy: ResidualClockRnPolicy,
    cache: ResidualClockRnPathCache,
    *,
    diagnostic: str,
    lambda_second_moment: float,
    start: int,
    stop: int,
    batch_size: int,
    path: Path,
    progress: _Progress,
    progress_payload: Mapping[str, Any],
) -> np.ndarray:
    if diagnostic not in {"entropy", "squared"}:
        raise ValueError(f"unknown conditional diagnostic {diagnostic!r}")
    temporary = path.with_name(f".{path.name}.tmp")
    objective_map = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float64,
        shape=(stop - start, cache.num_steps),
    )
    policy.eval()
    with torch.no_grad():
        for first in range(start, stop, batch_size):
            last = min(first + batch_size, stop)
            states, prices, returns = _load_batch(
                cache,
                first=first,
                last=last,
                device=policy.env.device,
            )
            if diagnostic == "entropy":
                value = _daily_gain(policy, states, prices, returns)
            else:
                value = _daily_squared_reward(
                    policy,
                    states,
                    prices,
                    returns,
                    lambda_second_moment=lambda_second_moment,
                )
            objective_map[first - start : last - start] = value.double().cpu().numpy()
            progress.emit(
                {
                    **progress_payload,
                    "work": "objective-forward",
                    "completed_paths": last - start,
                    "total_paths": stop - start,
                }
            )
    objective_map.flush()
    del objective_map
    os.replace(temporary, path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _gradient_statistics(
    parameters: tuple[torch.nn.Parameter, ...],
) -> dict[str, float | int]:
    square = 0.0
    maximum = 0.0
    count = 0
    nonfinite = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        nonfinite += int(torch.count_nonzero(~torch.isfinite(gradient)).item())
        finite = torch.nan_to_num(gradient).double()
        square += float(torch.sum(finite.square()).cpu())
        maximum = max(maximum, float(torch.amax(torch.abs(finite)).cpu()))
        count += gradient.numel()
    return {
        "gradient_l2_norm": math.sqrt(square),
        "gradient_rms": math.sqrt(square / max(count, 1)),
        "gradient_maximum_absolute": maximum,
        "parameter_count": count,
        "nonfinite_gradient_count": nonfinite,
    }


def _trainable_parameters(
    policy: ResidualClockRnPolicy,
    *,
    affine_only: bool,
) -> tuple[torch.nn.Parameter, ...]:
    for parameter in policy.parameters():
        parameter.requires_grad_(not affine_only)
    if affine_only:
        for parameter in policy.linear_skip.parameters():
            parameter.requires_grad_(True)
        return tuple(policy.linear_skip.parameters())
    return tuple(policy.parameters())


def _objective_loss_and_ess(
    objective_values: np.ndarray,
    *,
    diagnostic: str,
    batch_size: int,
) -> tuple[float, float | None]:
    if diagnostic == "entropy":
        return log_mean_exp_negative_gain(objective_values, batch_size=batch_size)
    if diagnostic == "squared":
        return -float(np.mean(objective_values, dtype=np.float64)), None
    raise ValueError(f"unknown conditional diagnostic {diagnostic!r}")


def _fit_stage(
    policy: ResidualClockRnPolicy,
    cache: ResidualClockRnPathCache,
    *,
    diagnostic: str,
    lambda_second_moment: float,
    start: int,
    stop: int,
    stage: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    maximum_gradient_norm: float,
    backtracking_factor: float,
    maximum_line_search_steps: int,
    directory: Path,
    progress: _Progress,
) -> tuple[ResidualClockRnPolicy, list[dict[str, Any]]]:
    if diagnostic not in {"entropy", "squared"}:
        raise ValueError(f"unknown conditional diagnostic {diagnostic!r}")
    affine_only = stage == "affine"
    parameters = _trainable_parameters(policy, affine_only=affine_only)
    checkpoint = directory / f"{stage}.pt"
    objective_path = directory / f"{stage}-objective-values.npy"
    minimum_improvement = 1.0e-12 if diagnostic == "entropy" else 1.0e-15
    optimization_config = {
        "method": "exact-gradient-descent-with-backtracking",
        "learning_rate": learning_rate,
        "maximum_gradient_norm": maximum_gradient_norm,
        "backtracking_factor": backtracking_factor,
        "maximum_line_search_steps": maximum_line_search_steps,
        "minimum_line_search_improvement": minimum_improvement,
    }
    data_identity = {
        "cache_metadata": str(cache.metadata_path.resolve()),
        "fold_start": start,
        "fold_stop": stop,
        "num_steps": cache.num_steps,
        "state_size": cache.state_size,
    }
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    first_epoch = 0
    if checkpoint.exists():
        payload = torch.load(
            checkpoint,
            map_location=policy.env.device,
            weights_only=False,
        )
        if (
            payload.get("schema") != _CHECKPOINT_SCHEMA
            or payload.get("stage") != stage
            or payload.get("diagnostic") != diagnostic
            or payload.get("lambda_second_moment") != lambda_second_moment
            or payload.get("optimization_config") != optimization_config
            or payload.get("data_identity") != data_identity
        ):
            raise RuntimeError(f"checkpoint identity mismatch in {checkpoint}")
        policy.load_state_dict(payload["current_state_dict"])
        history = list(payload["history"])
        best_loss = float(payload["best_loss"])
        best_state = payload["best_state_dict"]
        first_epoch = int(payload["epoch"])
        if payload.get("complete", False):
            policy.load_state_dict(best_state)
            return policy, history
        progress.emit(
            {
                "stage": "resume",
                "diagnostic": diagnostic,
                "fit_stage": stage,
                "epoch": first_epoch,
            },
            force=True,
        )

    for epoch in range(first_epoch, epochs):
        epoch_started = time.monotonic()
        objective_values = _write_objective_cache(
            policy,
            cache,
            diagnostic=diagnostic,
            lambda_second_moment=lambda_second_moment,
            start=start,
            stop=stop,
            batch_size=batch_size,
            path=objective_path,
            progress=progress,
            progress_payload={
                "stage": "train",
                "diagnostic": diagnostic,
                "fit_stage": stage,
                "epoch": epoch + 1,
                "total_epochs": epochs,
            },
        )
        loss, tilt_ess = _objective_loss_and_ess(
            objective_values,
            diagnostic=diagnostic,
            batch_size=batch_size,
        )
        if loss < best_loss:
            best_loss = loss
            best_state = _cpu_state_dict(policy)
        maximum_score: float | None = None
        exponential_sum: float | None = None
        if diagnostic == "entropy":
            maximum_score = float(np.max(-np.asarray(objective_values)))
            exponential_sum = 0.0
            for first in range(0, len(objective_values), batch_size):
                exponential_sum += float(
                    np.sum(
                        np.exp(
                            -np.asarray(objective_values[first : first + batch_size])
                            - maximum_score
                        ),
                        dtype=np.float64,
                    )
                )
        policy.zero_grad(set_to_none=True)
        observation_count = (stop - start) * cache.num_steps
        for first in range(start, stop, batch_size):
            last = min(first + batch_size, stop)
            states, prices, returns = _load_batch(
                cache,
                first=first,
                last=last,
                device=policy.env.device,
            )
            if diagnostic == "entropy":
                gain_tensor = _daily_gain(policy, states, prices, returns)
                stored_gain = np.asarray(
                    objective_values[first - start : last - start],
                    dtype=np.float64,
                )
                if maximum_score is None or exponential_sum is None:
                    raise AssertionError("entropy normalization was not initialized")
                probability = np.exp(-stored_gain - maximum_score) / exponential_sum
                probability_tensor = torch.as_tensor(
                    probability,
                    dtype=gain_tensor.dtype,
                    device=gain_tensor.device,
                )
                torch.sum(-probability_tensor * gain_tensor).backward()
            else:
                reward_tensor = _daily_squared_reward(
                    policy,
                    states,
                    prices,
                    returns,
                    lambda_second_moment=lambda_second_moment,
                )
                (-torch.sum(reward_tensor) / observation_count).backward()
            progress.emit(
                {
                    "stage": "train",
                    "diagnostic": diagnostic,
                    "fit_stage": stage,
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "work": "objective-gradient",
                    "completed_paths": last - start,
                    "total_paths": stop - start,
                }
            )
        preclip_norm = float(torch.nn.utils.clip_grad_norm_(parameters, maximum_gradient_norm))
        statistics = _gradient_statistics(parameters)
        if statistics["nonfinite_gradient_count"]:
            raise FloatingPointError(f"non-finite {stage} gradient")
        original_parameters = tuple(parameter.detach().clone() for parameter in parameters)
        accepted_step = 0.0
        candidate_loss = loss
        candidate_ess = tilt_ess
        line_search_evaluations = 0
        step_size = learning_rate
        for line_search_index in range(maximum_line_search_steps):
            line_search_evaluations = line_search_index + 1
            with torch.no_grad():
                for parameter, original in zip(
                    parameters,
                    original_parameters,
                    strict=True,
                ):
                    if parameter.grad is None:
                        parameter.copy_(original)
                    else:
                        parameter.copy_(original - step_size * parameter.grad)
            candidate_values = _write_objective_cache(
                policy,
                cache,
                diagnostic=diagnostic,
                lambda_second_moment=lambda_second_moment,
                start=start,
                stop=stop,
                batch_size=batch_size,
                path=objective_path,
                progress=progress,
                progress_payload={
                    "stage": "line-search",
                    "diagnostic": diagnostic,
                    "fit_stage": stage,
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "line_search_evaluation": line_search_evaluations,
                    "step_size": step_size,
                },
            )
            candidate_loss, candidate_ess = _objective_loss_and_ess(
                candidate_values,
                diagnostic=diagnostic,
                batch_size=batch_size,
            )
            if candidate_loss < loss - minimum_improvement:
                accepted_step = step_size
                break
            step_size *= backtracking_factor
        if accepted_step == 0.0:
            with torch.no_grad():
                for parameter, original in zip(
                    parameters,
                    original_parameters,
                    strict=True,
                ):
                    parameter.copy_(original)
            candidate_loss = loss
            candidate_ess = tilt_ess
        elif candidate_loss < best_loss:
            best_loss = candidate_loss
            best_state = _cpu_state_dict(policy)
        record = {
            "epoch": epoch + 1,
            "objective_loss": loss,
            "candidate_objective_loss": candidate_loss,
            "objective_improvement_from_zero": -loss,
            "candidate_objective_improvement_from_zero": -candidate_loss,
            "gradient_l2_norm_before_clipping": preclip_norm,
            "accepted_step_size": accepted_step,
            "line_search_evaluations": line_search_evaluations,
            **statistics,
            "epoch_seconds": time.monotonic() - epoch_started,
        }
        if diagnostic == "entropy":
            record.update(
                {
                    "log_mean_exponential_loss": loss,
                    "candidate_log_mean_exponential_loss": candidate_loss,
                    "tilted_path_time_effective_sample_size": tilt_ess,
                    "candidate_tilted_path_time_effective_sample_size": (candidate_ess),
                }
            )
        else:
            record.update(
                {
                    "mean_squared_objective": -loss,
                    "candidate_mean_squared_objective": -candidate_loss,
                    "lambda_second_moment": lambda_second_moment,
                }
            )
        history.append(record)
        if best_state is None:
            raise RuntimeError("training failed to retain a finite policy")
        _atomic_torch(
            {
                "schema": _CHECKPOINT_SCHEMA,
                "diagnostic": diagnostic,
                "lambda_second_moment": lambda_second_moment,
                "stage": stage,
                "epoch": epoch + 1,
                "complete": False,
                "optimization_config": optimization_config,
                "data_identity": data_identity,
                "policy_config": asdict(policy.config),
                "current_state_dict": _cpu_state_dict(policy),
                "best_state_dict": best_state,
                "best_loss": best_loss,
                "history": history,
            },
            checkpoint,
        )
        progress.emit(
            {
                "stage": "train-epoch-complete",
                "diagnostic": diagnostic,
                "fit_stage": stage,
                "epoch": epoch + 1,
                "total_epochs": epochs,
                **record,
            },
            force=True,
        )
        if accepted_step == 0.0:
            break

    final_values = _write_objective_cache(
        policy,
        cache,
        diagnostic=diagnostic,
        lambda_second_moment=lambda_second_moment,
        start=start,
        stop=stop,
        batch_size=batch_size,
        path=objective_path,
        progress=progress,
        progress_payload={
            "stage": "final-evaluation",
            "diagnostic": diagnostic,
            "fit_stage": stage,
        },
    )
    final_loss, final_ess = _objective_loss_and_ess(
        final_values,
        diagnostic=diagnostic,
        batch_size=batch_size,
    )
    if final_loss < best_loss or best_state is None:
        best_loss = final_loss
        best_state = _cpu_state_dict(policy)
    policy.load_state_dict(best_state)
    _atomic_torch(
        {
            "schema": _CHECKPOINT_SCHEMA,
            "diagnostic": diagnostic,
            "lambda_second_moment": lambda_second_moment,
            "stage": stage,
            "epoch": len(history),
            "complete": True,
            "optimization_config": optimization_config,
            "data_identity": data_identity,
            "policy_config": asdict(policy.config),
            "current_state_dict": _cpu_state_dict(policy),
            "best_state_dict": best_state,
            "best_loss": best_loss,
            "final_loss": final_loss,
            "final_tilted_path_time_effective_sample_size": final_ess,
            "history": history,
        },
        checkpoint,
    )
    return policy, history


def _fit_fold(
    cache: ResidualClockRnPathCache,
    *,
    diagnostic: str,
    fold: str,
    start: int,
    stop: int,
    output: Path,
    arguments: argparse.Namespace,
    progress: _Progress,
) -> tuple[ResidualClockRnPolicy, dict[str, Any]]:
    directory = output / diagnostic / fold.lower()
    directory.mkdir(parents=True, exist_ok=True)
    normalizer_path = directory / "normalizers.npz"
    if normalizer_path.exists():
        with np.load(normalizer_path, allow_pickle=False) as payload:
            normalizers = {name: payload[name] for name in payload.files}
    else:
        normalizers = streaming_normalizers(
            cache.states,
            cache.prices,
            start=start,
            stop=stop,
            batch_size=arguments.batch_size,
        )
        temporary = normalizer_path.with_name(f".{normalizer_path.name}.tmp.npz")
        np.savez(temporary, **normalizers)
        os.replace(temporary, normalizer_path)
    policy = _make_policy(
        state_size=cache.state_size,
        normalizers=normalizers,
        device=torch.device(arguments.device),
        seed=arguments.seed,
        label=f"residual-rn-{diagnostic}-{fold.lower()}",
    )
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    policy, affine_history = _fit_stage(
        policy,
        cache,
        diagnostic=diagnostic,
        lambda_second_moment=arguments.lambda_second_moment,
        start=start,
        stop=stop,
        stage="affine",
        epochs=arguments.affine_epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        maximum_gradient_norm=arguments.maximum_gradient_norm,
        backtracking_factor=arguments.backtracking_factor,
        maximum_line_search_steps=arguments.maximum_line_search_steps,
        directory=directory,
        progress=progress,
    )
    policy, full_history = _fit_stage(
        policy,
        cache,
        diagnostic=diagnostic,
        lambda_second_moment=arguments.lambda_second_moment,
        start=start,
        stop=stop,
        stage="full",
        epochs=arguments.full_epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        maximum_gradient_norm=arguments.maximum_gradient_norm,
        backtracking_factor=arguments.backtracking_factor,
        maximum_line_search_steps=arguments.maximum_line_search_steps,
        directory=directory,
        progress=progress,
    )
    return policy, {
        "diagnostic": diagnostic,
        "training_paths": stop - start,
        "training_antithetic_pairs": (stop - start) // 2,
        "state_size": cache.state_size,
        "parameter_count": parameter_count,
        "affine_epochs": len(affine_history),
        "full_epochs": len(full_history),
        "affine_final": affine_history[-1] if affine_history else None,
        "full_final": full_history[-1] if full_history else None,
    }


@torch.no_grad()
def _path_scores(
    policy: ResidualClockRnPolicy,
    cache: ResidualClockRnPathCache,
    *,
    start: int,
    stop: int,
    batch_size: int,
    progress: _Progress,
    label: str,
) -> np.ndarray:
    raw_scores = np.empty(stop - start, dtype=np.float64)
    policy.eval()
    for first in range(start, stop, batch_size):
        last = min(first + batch_size, stop)
        states, prices, returns = _load_batch(
            cache,
            first=first,
            last=last,
            device=policy.env.device,
        )
        action = _actions_for_path_batch(policy, states, prices)
        daily_raw = torch.sum(action * returns, dim=-1)
        raw_scores[first - start : last - start] = (
            torch.mean(daily_raw, dim=1).double().cpu().numpy()
        )
        progress.emit(
            {
                "stage": "held-out-score",
                "direction": label,
                "completed_paths": last - start,
                "total_paths": stop - start,
            }
        )
    return raw_scores


def adjacent_pair_test(values: np.ndarray) -> dict[str, float | int | list[float]]:
    """Test a uniform mean using adjacent antithetic pairs as clusters."""

    value = np.asarray(values, dtype=np.float64)
    if value.ndim != 1 or len(value) < 4 or len(value) % 2:
        raise ValueError("pair test requires an even vector with at least two pairs")
    pair_value = np.mean(value.reshape(-1, 2), axis=1)
    mean = float(np.mean(pair_value, dtype=np.float64))
    standard_error = float(np.std(pair_value, ddof=1) / math.sqrt(len(pair_value)))
    if standard_error > 0.0:
        statistic = mean / standard_error
    else:
        statistic = 0.0 if mean == 0.0 else math.copysign(math.inf, mean)
    p_value = math.erfc(abs(statistic) / math.sqrt(2.0))
    return {
        "paths": len(value),
        "antithetic_pairs": len(pair_value),
        "mean_daily_captured_return": mean,
        "pair_cluster_standard_error": standard_error,
        "statistic": statistic,
        "two_sided_normal_p_value": p_value,
        "confidence_interval_95": [
            mean - 1.96 * standard_error,
            mean + 1.96 * standard_error,
        ],
    }


def unconditional_coordinate_diagnostics(
    returns: np.ndarray,
    *,
    start: int,
    stop: int,
    batch_size: int,
) -> dict[str, Any]:
    """Stream uniform coordinate means and adjacent-pair cluster errors."""

    if start < 0 or stop > len(returns) or start >= stop or (stop - start) % 2:
        raise ValueError("unconditional audit interval must contain complete pairs")
    pair_count = (stop - start) // 2
    if pair_count < 2:
        raise ValueError("unconditional audit needs at least two pairs")
    if batch_size <= 0 or batch_size % 2:
        raise ValueError("unconditional audit batch_size must preserve pairs")
    pair_sum = np.zeros(returns.shape[1:], dtype=np.float64)
    pair_square_sum = np.zeros_like(pair_sum)
    for first in range(start, stop, batch_size):
        last = min(first + batch_size, stop)
        value = np.asarray(returns[first:last], dtype=np.float64)
        pair_value = np.mean(
            value.reshape(-1, 2, *returns.shape[1:]),
            axis=1,
        )
        pair_sum += np.sum(pair_value, axis=0, dtype=np.float64)
        pair_square_sum += np.sum(np.square(pair_value), axis=0, dtype=np.float64)
    mean = pair_sum / pair_count
    variance = np.maximum(
        (pair_square_sum - pair_count * np.square(mean)) / (pair_count - 1),
        0.0,
    )
    standard_error = np.sqrt(variance / pair_count)
    z_score = np.divide(
        mean,
        standard_error,
        out=np.zeros_like(mean),
        where=standard_error > 0.0,
    )
    absolute_z = np.abs(z_score)
    worst_flat = int(np.argmax(absolute_z))
    worst_time, worst_instrument = np.unravel_index(worst_flat, mean.shape)
    return {
        "paths": stop - start,
        "antithetic_pairs": pair_count,
        "coordinates": int(mean.size),
        "maximum_absolute_z": float(np.max(absolute_z)),
        "percentile_99_absolute_z": float(np.percentile(absolute_z, 99.0)),
        "coordinates_absolute_z_above_1_96": int(np.sum(absolute_z > 1.96)),
        "coordinates_absolute_z_above_2_576": int(np.sum(absolute_z > 2.576)),
        "coordinates_absolute_z_above_5": int(np.sum(absolute_z > 5.0)),
        "worst_coordinate": {
            "time_index_zero_based": int(worst_time),
            "instrument_index_zero_based": int(worst_instrument),
            "mean": float(mean[worst_time, worst_instrument]),
            "pair_cluster_standard_error": float(standard_error[worst_time, worst_instrument]),
            "z_score": float(z_score[worst_time, worst_instrument]),
        },
    }


def run(arguments: argparse.Namespace) -> None:
    """Run both cross-fit directions and write the complete audit report."""

    arguments.output.mkdir(parents=True, exist_ok=True)
    with FileLock(arguments.output / "run.lock", timeout=0):
        if arguments.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if arguments.cpu_threads <= 0:
            raise ValueError("cpu_threads must be positive")
        torch.set_num_threads(arguments.cpu_threads)
        if arguments.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        progress = _Progress(
            arguments.output,
            interval_seconds=arguments.progress_interval_seconds,
        )
        progress.emit({"stage": "start"}, force=True)
        cache = load_residual_clock_rn_path_cache(arguments.cache)
        fold_paths = (
            cache.num_paths // 2 if arguments.paths_per_fold is None else arguments.paths_per_fold
        )
        if fold_paths < 2 or fold_paths % 2 or 2 * fold_paths > cache.num_paths:
            raise ValueError("paths_per_fold must select complete adjacent pairs")
        if arguments.batch_size <= 0 or arguments.batch_size % 2:
            raise ValueError("batch_size must be positive and preserve adjacent pairs")
        if arguments.affine_epochs < 0 or arguments.full_epochs < 0:
            raise ValueError("training epoch counts must be nonnegative")
        if not math.isfinite(arguments.learning_rate) or arguments.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if (
            not math.isfinite(arguments.maximum_gradient_norm)
            or arguments.maximum_gradient_norm <= 0.0
        ):
            raise ValueError("maximum_gradient_norm must be finite and positive")
        if not 0.0 < arguments.backtracking_factor < 1.0:
            raise ValueError("backtracking_factor must lie strictly between zero and one")
        if arguments.maximum_line_search_steps <= 0:
            raise ValueError("maximum_line_search_steps must be positive")
        if (
            not math.isfinite(arguments.lambda_second_moment)
            or arguments.lambda_second_moment < 0.0
        ):
            raise ValueError("lambda_second_moment must be finite and nonnegative")
        a_start, a_stop = 0, fold_paths
        b_start, b_stop = fold_paths, 2 * fold_paths

        diagnostic_results: dict[str, dict[str, Any]] = {}
        for diagnostic in ("squared", "entropy"):
            policy_a, fit_a = _fit_fold(
                cache,
                diagnostic=diagnostic,
                fold="A",
                start=a_start,
                stop=a_stop,
                output=arguments.output,
                arguments=arguments,
                progress=progress,
            )
            score_b = _path_scores(
                policy_a,
                cache,
                start=b_start,
                stop=b_stop,
                batch_size=arguments.batch_size,
                progress=progress,
                label=f"{diagnostic}-A-to-B",
            )
            del policy_a
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            policy_b, fit_b = _fit_fold(
                cache,
                diagnostic=diagnostic,
                fold="B",
                start=b_start,
                stop=b_stop,
                output=arguments.output,
                arguments=arguments,
                progress=progress,
            )
            score_a = _path_scores(
                policy_b,
                cache,
                start=a_start,
                stop=a_stop,
                batch_size=arguments.batch_size,
                progress=progress,
                label=f"{diagnostic}-B-to-A",
            )
            del policy_b
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            np.save(
                arguments.output / f"{diagnostic}-A-to-B-raw-scores.npy",
                score_b,
            )
            np.save(
                arguments.output / f"{diagnostic}-B-to-A-raw-scores.npy",
                score_a,
            )
            pooled_test = adjacent_pair_test(np.concatenate((score_a, score_b)))
            diagnostic_results[diagnostic] = {
                "fold_A_fit": fit_a,
                "fold_B_fit": fit_b,
                "A_to_B": adjacent_pair_test(score_b),
                "B_to_A": adjacent_pair_test(score_a),
                "pooled_out_of_fold": pooled_test,
            }

        squared_test = diagnostic_results["squared"]["pooled_out_of_fold"]
        entropy_test = diagnostic_results["entropy"]["pooled_out_of_fold"]
        report: dict[str, Any] = {
            "schema": _REPORT_SCHEMA,
            "complete": True,
            "cache": {
                "directory": str(cache.directory),
                "metadata": str(cache.metadata_path),
                "source_schema": cache.metadata.get("schema"),
                "state_size": cache.state_size,
                "paths": 2 * fold_paths,
                "steps": cache.num_steps,
                "pair_ordering": "adjacent antithetic pairs",
                "measure": "uniform Q path law; no importance weights",
            },
            "definition": {
                "entropy_objective": (
                    "log mean over training-fold path-time observations of "
                    "exp(-a(X,t)'R + 0.5*a(X,t)'a(X,t))"
                ),
                "squared_objective": (
                    "maximize mean over training-fold path-time observations "
                    "of a(X,t)'R - 0.5*a(X,t)'a(X,t) - "
                    "0.5*lambda*(a(X,t)'R)^2"
                ),
                "lambda_second_moment": arguments.lambda_second_moment,
                "action_class": (
                    "uncapped shared two-layer width-128 SiLU plus expiry-specific affine skips"
                ),
                "features": (
                    "actual residual-clock functional state, twelve option "
                    "prices per expiry, and normalized calendar time"
                ),
                "cross_fit": "A policy tested on B and B policy tested on A",
                "conditional_statistics": (
                    "T_sqr and T_ent are their respective pooled out-of-fold "
                    "mean daily a(X,t)'R divided by its adjacent-antithetic-pair "
                    "cluster standard error"
                ),
                "conditional_p_values": ("two-sided standard-normal p-values for T_sqr and T_ent"),
                "optimization": (
                    "exact full-fold gradients with norm clipping and monotone "
                    "backtracking line search"
                ),
            },
            "arguments": {
                name: str(value) if isinstance(value, Path) else value
                for name, value in vars(arguments).items()
            },
            "conditional_diagnostics": diagnostic_results,
            "pooled_out_of_fold": {
                "T_sqr": squared_test["statistic"],
                "p_sqr": squared_test["two_sided_normal_p_value"],
                "T_ent": entropy_test["statistic"],
                "p_ent": entropy_test["two_sided_normal_p_value"],
                "squared": squared_test,
                "entropy": entropy_test,
            },
            "unconditional_coordinate_diagnostics": (
                unconditional_coordinate_diagnostics(
                    cache.returns,
                    start=0,
                    stop=2 * fold_paths,
                    batch_size=arguments.batch_size,
                )
            ),
            "resources_final": _check_resources(arguments.output),
        }
        _atomic_json(report, arguments.output / "report.json")
        progress.emit(
            {
                "stage": "complete",
                "report": str(arguments.output / "report.json"),
                "T_sqr": report["pooled_out_of_fold"]["T_sqr"],
                "p_sqr": report["pooled_out_of_fold"]["p_sqr"],
                "T_ent": report["pooled_out_of_fold"]["T_ent"],
                "p_ent": report["pooled_out_of_fold"]["p_ent"],
            },
            force=True,
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cache", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--paths-per-fold", type=int, default=None)
    result.add_argument("--batch-size", type=int, default=2_000)
    result.add_argument("--affine-epochs", type=int, default=100)
    result.add_argument("--full-epochs", type=int, default=200)
    result.add_argument("--learning-rate", type=float, default=1.0e-3)
    result.add_argument("--maximum-gradient-norm", type=float, default=10.0)
    result.add_argument("--backtracking-factor", type=float, default=0.2)
    result.add_argument("--maximum-line-search-steps", type=int, default=20)
    result.add_argument("--lambda-second-moment", type=float, default=1.0)
    result.add_argument("--progress-interval-seconds", type=float, default=30.0)
    result.add_argument("--seed", type=int, default=20261010)
    result.add_argument("--cpu-threads", type=int, default=8)
    return result


__all__ = [
    "ResidualClockRnPathCache",
    "ResidualClockRnPolicy",
    "ResidualClockRnPolicyConfig",
    "adjacent_pair_test",
    "load_residual_clock_rn_path_cache",
    "log_mean_exp_negative_gain",
    "parser",
    "run",
    "squared_second_moment_reward",
    "streaming_normalizers",
    "unconditional_coordinate_diagnostics",
]


if __name__ == "__main__":
    run(parser().parse_args())
