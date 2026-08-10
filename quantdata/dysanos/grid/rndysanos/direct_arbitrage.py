"""State-conditional direct-arbitrage policies and certification.

The policy sees only the current state ``z``.  A softmax chooses a long/short
direction while a separate sigmoid gate chooses gross exposure.  Thus long
and short positions are represented separately and

``sum(long + short) <= 1``.

For realized return ``dH`` and effective one-way cost ``c``, net return is

``(long - short) @ dH - (long + short) @ c``.

Population almost-sure nonnegativity cannot be proved from Monte Carlo data.
The certification helpers therefore report every validation loss and use
paths, rather than path-time observations, as independent clusters.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.uniquehash import named_unique_filename48_8
from cdxcore.util import is_filename
from cdxcore.version import version
from filelock import FileLock
from torch import Tensor

from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


@version("1.5")
@dataclass(frozen=True, kw_only=True)
class DirectArbitrageConfig:
    """Logical policy, economic threshold, and constrained-loss configuration."""

    label: str
    state_features: int
    instruments: int
    group: str = ""
    user_version: str = "0.1"
    epsilon: float = 0.01
    delta: float = 0.01
    numerical_tolerance: float = 1.0e-4
    smooth_temperature: float = 0.0025
    tail_fraction: float = 0.01
    downside_weight: float = 10.0
    tail_weight: float = 100.0
    overlap_weight: float = 1.0
    mean_reward_weight: float = 0.1
    discovery_fraction: float = 0.05
    constraint_ramp_fraction: float = 0.45
    initial_trade_fraction: float = 0.01
    mask_arbitrage_candidates: bool = False
    objective: str = "simplex-soft-hit"
    cap_weight: float = 1.0
    seed: int = 70_771
    network_config: FeedForwardConfig = FeedForwardConfig(
        num_depth=2,
        num_width=100,
        activation="LeakyReLU",
        use_bias=True,
    )

    def __post_init__(self) -> None:
        verify_inp(
            isinstance(self.label, str) and is_filename(self.label),
            lambda: f"'label' must be a filename, got {self.label!r}",
        )
        verify_inp(
            isinstance(self.group, str) and not self.group.startswith("/"),
            lambda: f"'group' must be relative, got {self.group!r}",
        )
        verify_inp(
            self.state_features > 0,
            lambda: f"'state_features' must be positive, got {self.state_features}",
        )
        verify_inp(
            self.instruments > 0,
            lambda: f"'instruments' must be positive, got {self.instruments}",
        )
        for name in (
            "epsilon",
            "numerical_tolerance",
            "smooth_temperature",
            "downside_weight",
            "tail_weight",
            "overlap_weight",
            "mean_reward_weight",
            "cap_weight",
        ):
            value = getattr(self, name)
            verify_inp(
                math.isfinite(value) and value > 0.0,
                lambda name=name, value=value: (
                    f"'{name}' must be finite and positive, got {value}"
                ),
            )
        for name in (
            "delta",
            "tail_fraction",
            "discovery_fraction",
            "constraint_ramp_fraction",
            "initial_trade_fraction",
        ):
            value = getattr(self, name)
            verify_inp(
                math.isfinite(value) and 0.0 < value < 1.0,
                lambda name=name, value=value: (
                    f"'{name}' must be strictly between zero and one, got {value}"
                ),
            )
        verify_inp(
            self.discovery_fraction + self.constraint_ramp_fraction < 1.0,
            "discovery and constraint-ramp fractions must sum to less than one",
        )
        verify_inp(isinstance(self.seed, int), lambda: f"'seed' is not an int: {self.seed!r}")
        verify_inp(
            isinstance(self.mask_arbitrage_candidates, bool),
            lambda: (
                "'mask_arbitrage_candidates' must be bool, got "
                f"{self.mask_arbitrage_candidates!r}"
            ),
        )
        verify_inp(
            self.objective in {"simplex-soft-hit", "radial-l1-v2"},
            lambda: f"unknown direct-arbitrage objective {self.objective!r}",
        )

    @property
    def filename(self) -> str:
        return named_unique_filename48_8(self.label, self)


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class DirectArbitrageTrainConfig:
    """Operational configuration for direct-policy optimization."""

    batch_size: int = 1_024
    valuation_batch_size: int = 4_096
    lr: float = 1.0e-3
    final_lr: float = 1.0e-6
    weight_decay: float = 1.0e-4
    compile: bool = True
    value_update: int = 10_000
    checkpoint_update: int = 10_000
    checkpoint_root: str = "~/.cache/rndysanos/direct-arbitrage"

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "valuation_batch_size",
            "value_update",
            "checkpoint_update",
        ):
            value = getattr(self, name)
            verify_inp(value > 0, lambda name=name, value=value: f"'{name}' must be positive, got {value}")
        for name in ("lr", "final_lr"):
            value = getattr(self, name)
            verify_inp(
                math.isfinite(value) and value > 0.0,
                lambda name=name, value=value: f"'{name}' must be finite and positive, got {value}",
            )
        verify_inp(
            math.isfinite(self.weight_decay) and self.weight_decay >= 0.0,
            lambda: f"'weight_decay' must be finite and nonnegative, got {self.weight_decay}",
        )


class LongShortSimplexPolicy(BaseModel):
    """State-only long/short direction and independent scalar exposure gate."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: DirectArbitrageConfig,
        state_mean: Tensor,
        state_scale: Tensor,
    ) -> None:
        super().__init__(env)
        expected = (config.state_features,)
        verify_inp(
            state_mean.shape == state_scale.shape == expected,
            lambda: (
                f"state normalization must have shape {expected}, found "
                f"{state_mean.shape} and {state_scale.shape}"
            ),
        )
        if not torch.isfinite(state_mean).all().item():
            raise ValueError("state mean contains non-finite values")
        if not torch.isfinite(state_scale).all().item() or not torch.all(state_scale > 0.0).item():
            raise ValueError("state scale must be finite and positive")
        self.config = config
        self.network = FeedForward(
            env,
            num_input=config.state_features,
            num_output=2 * config.instruments + 1,
            config=config.network_config,
        )
        self.register_buffer("state_mean", state_mean.detach().clone())
        self.register_buffer("state_scale", state_scale.detach().clone())
        self._initialize_no_trade()

    def _initialize_no_trade(self) -> None:
        final = self.network.In[-1] if self.network.In else self.network.I0
        assert final.bias is not None
        gate_logit = math.log(
            self.config.initial_trade_fraction
            / (1.0 - self.config.initial_trade_fraction)
        )
        with torch.no_grad():
            final.weight.zero_()
            final.bias.zero_()
            final.bias[-1] = gate_logit

    def forward(
        self,
        z: Tensor,
        excluded_directions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        verify_inp(
            z.shape[-1] == self.config.state_features,
            lambda: f"state shape {z.shape} has the wrong final dimension",
        )
        normalized = (z - self.state_mean) / self.state_scale
        logits = self.network(normalized)
        J = self.config.instruments
        direction_logits = logits[..., : 2 * J]
        if excluded_directions is not None:
            verify_inp(
                excluded_directions.shape == direction_logits.shape,
                lambda: (
                    f"excluded-direction shape {excluded_directions.shape} != "
                    f"{direction_logits.shape}"
                ),
            )
            verify_inp(
                excluded_directions.dtype == torch.bool,
                lambda: (
                    "excluded directions must have bool dtype, got "
                    f"{excluded_directions.dtype}"
                ),
            )
            direction_logits = direction_logits.masked_fill(
                excluded_directions, -torch.inf
            )
        direction = torch.softmax(direction_logits, dim=-1)
        exposure = torch.sigmoid(logits[..., -1])
        long = exposure[..., None] * direction[..., :J]
        short = exposure[..., None] * direction[..., J:]
        no_trade = 1.0 - exposure
        self.verify(lambda: torch.isfinite(direction), "policy direction is non-finite")
        self.verify(lambda: torch.isfinite(exposure), "policy exposure is non-finite")
        return long, short, no_trade


class RadialL1Policy(BaseModel):
    """V2.0 policy with one unbounded signed output and radial L1 clipping."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        config: DirectArbitrageConfig,
        state_mean: Tensor,
        state_scale: Tensor,
    ) -> None:
        super().__init__(env)
        expected = (config.state_features,)
        verify_inp(
            state_mean.shape == state_scale.shape == expected,
            lambda: (
                f"state normalization must have shape {expected}, found "
                f"{state_mean.shape} and {state_scale.shape}"
            ),
        )
        if not torch.isfinite(state_mean).all().item():
            raise ValueError("state mean contains non-finite values")
        if not torch.isfinite(state_scale).all().item() or not torch.all(
            state_scale > 0.0
        ).item():
            raise ValueError("state scale must be finite and positive")
        self.config = config
        self.network = FeedForward(
            env,
            num_input=config.state_features,
            num_output=config.instruments,
            config=config.network_config,
        )
        self.register_buffer("state_mean", state_mean.detach().clone())
        self.register_buffer("state_scale", state_scale.detach().clone())
        self._initialize_small_trade()

    def _initialize_small_trade(self) -> None:
        final = self.network.In[-1] if self.network.In else self.network.I0
        with torch.no_grad():
            final.weight.mul_(self.config.initial_trade_fraction)
            if final.bias is not None:
                final.bias.zero_()

    def forward(
        self,
        z: Tensor,
        excluded_directions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return the feasible signed action and feasible raw L1 norm."""

        verify_inp(
            z.shape[-1] == self.config.state_features,
            lambda: f"state shape {z.shape} has the wrong final dimension",
        )
        normalized = (z - self.state_mean) / self.state_scale
        raw = self.network(normalized)
        if excluded_directions is not None:
            expected = raw.shape[:-1] + (2 * self.config.instruments,)
            verify_inp(
                excluded_directions.shape == expected,
                lambda: (
                    f"excluded-direction shape {excluded_directions.shape} != "
                    f"{expected}"
                ),
            )
            verify_inp(
                excluded_directions.dtype == torch.bool,
                lambda: (
                    "excluded directions must have bool dtype, got "
                    f"{excluded_directions.dtype}"
                ),
            )
            J = self.config.instruments
            positive = raw.clamp_min(0.0).masked_fill(
                excluded_directions[..., :J], 0.0
            )
            negative = (-raw).clamp_min(0.0).masked_fill(
                excluded_directions[..., J:], 0.0
            )
            raw = positive - negative
        raw_l1 = raw.abs().sum(dim=-1)
        action = raw / raw_l1.clamp_min(1.0)[..., None]
        self.verify(lambda: torch.isfinite(action), "radial policy action is non-finite")
        self.verify(lambda: torch.isfinite(raw_l1), "radial policy norm is non-finite")
        return action, raw_l1


@dataclass
class DirectArbitrageTrainingResult:
    """Trained policy, complete valuations, and restart metadata."""

    policy: LongShortSimplexPolicy | RadialL1Policy
    training: dict[str, Any]
    validation: dict[str, Any] | None
    tracker: dict[str, list[Any]]
    checkpoint_path: Path
    restarted: bool


def net_return(
    *,
    long: Tensor,
    short: Tensor,
    dH: Tensor,
    cost: Tensor,
) -> Tensor:
    """Return net P&L for separate long and short allocations."""

    verify_inp(
        long.shape == short.shape == dH.shape == cost.shape,
        lambda: (
            "long, short, return, and cost shapes differ: "
            f"{long.shape}, {short.shape}, {dH.shape}, {cost.shape}"
        ),
    )
    for name, value in (("long", long), ("short", short), ("dH", dH), ("cost", cost)):
        if not torch.isfinite(value).all().item():
            raise ValueError(f"'{name}' contains non-finite values")
    if not torch.all(long >= 0.0).item() or not torch.all(short >= 0.0).item():
        raise ValueError("long and short allocations must be nonnegative")
    if not torch.all(cost >= 0.0).item():
        raise ValueError("cost must be nonnegative")
    return _net_return_unchecked(long=long, short=short, dH=dH, cost=cost)


def _net_return_unchecked(
    *, long: Tensor, short: Tensor, dH: Tensor, cost: Tensor
) -> Tensor:
    """Performance path after shapes and signs were established by preparation."""

    return ((long - short) * dH - (long + short) * cost).sum(dim=-1)


def direct_arbitrage_loss(
    *,
    long: Tensor,
    short: Tensor,
    dH: Tensor,
    cost: Tensor,
    config: DirectArbitrageConfig,
    penalty_scale: float | Tensor = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Reward threshold hits while heavily penalizing mean and tail downside."""

    pnl = _net_return_unchecked(long=long, short=short, dH=dH, cost=cost)
    soft_hit = torch.sigmoid(
        (pnl - config.epsilon) / config.smooth_temperature
    ).mean()
    mean_reward = pnl.mean() / config.epsilon
    downside = (-pnl - config.numerical_tolerance).clamp_min(0.0) / config.epsilon
    flat_downside = downside.reshape(-1)
    tail_count = max(1, int(math.ceil(flat_downside.numel() * config.tail_fraction)))
    tail_downside = torch.topk(flat_downside, k=tail_count, sorted=False).values.mean()
    overlap = (long * short).sum(dim=-1).mean()
    scale = torch.as_tensor(penalty_scale, dtype=pnl.dtype, device=pnl.device)
    loss = (
        -soft_hit
        - config.mean_reward_weight * mean_reward
        + scale * config.downside_weight * downside.mean()
        + scale * config.tail_weight * tail_downside
        + config.overlap_weight * overlap
    )
    return loss, {
        "pnl": pnl,
        "soft_hit_probability": soft_hit,
        "mean_reward_scaled": mean_reward,
        "downside_mean_scaled": downside.mean(),
        "tail_downside_scaled": tail_downside,
        "long_short_overlap": overlap,
    }


def v2_arbitrage_loss(
    *,
    action: Tensor,
    raw_l1: Tensor,
    dH: Tensor,
    cost: Tensor,
    config: DirectArbitrageConfig,
    penalty_scale: float | Tensor = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """V2.0 mean/downside/CVaR loss with radial-output conditioning penalty."""

    long = action.clamp_min(0.0)
    short = (-action).clamp_min(0.0)
    pnl = _net_return_unchecked(long=long, short=short, dH=dH, cost=cost)
    downside = (-pnl).clamp_min(0.0)
    flat_downside = downside.reshape(-1)
    tail_count = max(1, int(math.ceil(flat_downside.numel() * config.tail_fraction)))
    tail_downside = torch.topk(
        flat_downside, k=tail_count, sorted=False
    ).values.mean()
    cap_excess_sqr = (raw_l1 - 1.0).clamp_min(0.0).square().mean()
    scale = torch.as_tensor(penalty_scale, dtype=pnl.dtype, device=pnl.device)
    loss = (
        -pnl.mean()
        + scale * config.downside_weight * downside.mean()
        + scale * config.tail_weight * tail_downside
        + config.cap_weight * cap_excess_sqr
    )
    return loss, {
        "pnl": pnl,
        "pnl_mean": pnl.mean(),
        "downside_mean": downside.mean(),
        "tail_downside": tail_downside,
        "cap_excess_sqr": cap_excess_sqr,
        "raw_l1_mean": raw_l1.mean(),
        "raw_l1_max": raw_l1.max(),
    }


def _policy_allocations(
    policy: LongShortSimplexPolicy | RadialL1Policy,
    z: Tensor,
    excluded_directions: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    if isinstance(policy, RadialL1Policy):
        action, raw_l1 = policy(z, excluded_directions)
        long = action.clamp_min(0.0)
        short = (-action).clamp_min(0.0)
        no_trade = 1.0 - action.abs().sum(dim=-1)
        return long, short, no_trade, raw_l1
    long, short, no_trade = policy(z, excluded_directions)
    return long, short, no_trade, None


def _iter_path_batches(
    data: Mapping[str, Tensor], batch_size: int
) -> Iterator[dict[str, Tensor]]:
    paths = data["z"].shape[0]
    for start in range(0, paths, batch_size):
        stop = min(start + batch_size, paths)
        yield {name: value[start:stop] for name, value in data.items()}


def _sample_paths(
    data: Mapping[str, Tensor],
    *,
    batch_size: int,
    generator: torch.Generator,
) -> dict[str, Tensor]:
    paths = data["z"].shape[0]
    indices = torch.randint(
        paths,
        (min(batch_size, paths),),
        device=data["z"].device,
        generator=generator,
    )
    return {name: value[indices] for name, value in data.items()}


@torch.no_grad()
def state_normalization(
    data: Mapping[str, Tensor], *, batch_size: int
) -> tuple[Tensor, Tensor]:
    """Return training-only state mean and population standard deviation."""

    features = data["z"].shape[-1]
    total = torch.zeros(features, dtype=torch.float64, device=data["z"].device)
    total_sqr = torch.zeros_like(total)
    observations = 0
    for batch in _iter_path_batches(data, batch_size):
        z = batch["z"]
        total += z.sum(dim=(0, 1), dtype=torch.float64)
        total_sqr += z.square().sum(dim=(0, 1), dtype=torch.float64)
        observations += z.shape[0] * z.shape[1]
    mean = total / float(observations)
    variance = (total_sqr / float(observations) - mean.square()).clamp_min(0.0)
    scale = variance.sqrt().clamp_min(1.0e-6)
    return mean.to(data["z"].dtype), scale.to(data["z"].dtype)


@torch.no_grad()
def unconditional_cost(
    data: Mapping[str, Tensor],
    *,
    cost_uncond: float,
    batch_size: int,
) -> Tensor:
    """Return the full-training population standard error per instrument."""

    verify_inp(
        math.isfinite(cost_uncond) and cost_uncond >= 0.0,
        lambda: f"'cost_uncond' must be finite and nonnegative, got {cost_uncond}",
    )
    instruments = data["dH"].shape[-1]
    total = torch.zeros(instruments, dtype=torch.float64, device=data["dH"].device)
    total_sqr = torch.zeros_like(total)
    observations = 0
    for batch in _iter_path_batches(data, batch_size):
        dH = batch["dH"]
        total += dH.sum(dim=(0, 1), dtype=torch.float64)
        total_sqr += dH.square().sum(dim=(0, 1), dtype=torch.float64)
        observations += dH.shape[0] * dH.shape[1]
    mean = total / float(observations)
    variance = (total_sqr / float(observations) - mean.square()).clamp_min(0.0)
    return (cost_uncond * torch.sqrt(variance / float(observations))).to(
        data["dH"].dtype
    )


@torch.no_grad()
def effective_cost(
    data: Mapping[str, Tensor],
    *,
    bar_c: Tensor,
    cost_prop: float,
    conditional_variance_model: Any | None,
    n_cost: int,
    batch_size: int,
) -> Tensor:
    """Materialize ``max(c_cond, bar_c, c_prop)`` without retaining gradients."""

    verify_inp(n_cost > 0, lambda: f"'n_cost' must be positive, got {n_cost}")
    verify_inp(
        math.isfinite(cost_prop) and cost_prop >= 0.0,
        lambda: f"'cost_prop' must be finite and nonnegative, got {cost_prop}",
    )
    H = data["H"]
    verify_inp(
        bar_c.shape == (H.shape[-1],),
        lambda: f"global cost shape {bar_c.shape} != {(H.shape[-1],)}",
    )
    output = torch.empty_like(H)
    start = 0
    for batch in _iter_path_batches(data, batch_size):
        stop = start + batch["z"].shape[0]
        c_prop = cost_prop * batch["H"].abs()
        cost = torch.maximum(c_prop, bar_c.reshape(1, 1, -1))
        if conditional_variance_model is not None:
            _, _, variance = conditional_variance_model.moment_outputs(
                batch["z"], batch["dH"]
            )
            c_cond = torch.sqrt(variance.clamp_min(0.0) / float(n_cost))
            cost = torch.maximum(cost, c_cond)
        output[start:stop] = cost
        start = stop
    if not torch.isfinite(output).all().item() or not torch.all(output >= 0.0).item():
        raise RuntimeError("effective cost contains invalid values")
    return output


def _clustered_lower_bound(path_rates: np.ndarray, confidence: float = 0.95) -> float:
    if path_rates.size == 0:
        raise ValueError("at least one path is required")
    if path_rates.size == 1:
        return float(path_rates[0])
    z_score = NormalDist().inv_cdf(confidence)
    error = path_rates.std(ddof=1) / math.sqrt(path_rates.size)
    return float(max(0.0, path_rates.mean() - z_score * error))


def _wilson_upper(successes: int, observations: int, confidence: float = 0.95) -> float:
    verify_inp(0 <= successes <= observations, "invalid binomial counts")
    verify_inp(observations > 0, "at least one binomial observation is required")
    z_score = NormalDist().inv_cdf(confidence)
    probability = successes / observations
    denominator = 1.0 + z_score**2 / observations
    center = probability + z_score**2 / (2.0 * observations)
    radius = z_score * math.sqrt(
        probability * (1.0 - probability) / observations
        + z_score**2 / (4.0 * observations**2)
    )
    return float(min(1.0, (center + radius) / denominator))


@torch.no_grad()
def valuate_policy(
    policy: LongShortSimplexPolicy | RadialL1Policy,
    data: Mapping[str, Tensor],
    *,
    batch_size: int,
    split: str,
) -> dict[str, Any]:
    """Value a policy and conservatively certify it using independent paths."""

    verify_inp(split in {"training", "validation"}, lambda: f"unknown split {split!r}")
    pnl_chunks: list[np.ndarray] = []
    gross_chunks: list[np.ndarray] = []
    no_trade_chunks: list[np.ndarray] = []
    raw_l1_chunks: list[np.ndarray] = []
    long_total = torch.zeros(
        policy.config.instruments, dtype=torch.float64, device=policy.device
    )
    short_total = torch.zeros_like(long_total)
    overlap_total = torch.zeros_like(long_total)
    observations = 0
    policy.eval()
    for batch in _iter_path_batches(data, batch_size):
        excluded = batch.get("excluded_directions")
        long, short, no_trade, raw_l1 = _policy_allocations(
            policy, batch["z"], excluded
        )
        pnl = net_return(
            long=long,
            short=short,
            dH=batch["dH"],
            cost=batch["cost"],
        )
        pnl_chunks.append(pnl.cpu().numpy())
        gross_chunks.append((long + short).sum(dim=-1).cpu().numpy())
        no_trade_chunks.append(no_trade.cpu().numpy())
        if raw_l1 is not None:
            raw_l1_chunks.append(raw_l1.cpu().numpy())
        long_total += long.sum(dim=(0, 1), dtype=torch.float64)
        short_total += short.sum(dim=(0, 1), dtype=torch.float64)
        overlap_total += (long * short).sum(dim=(0, 1), dtype=torch.float64)
        observations += pnl.numel()

    pnl = np.concatenate(pnl_chunks, axis=0).astype(np.float64, copy=False)
    gross = np.concatenate(gross_chunks, axis=0).astype(np.float64, copy=False)
    no_trade = np.concatenate(no_trade_chunks, axis=0).astype(np.float64, copy=False)
    raw_l1 = (
        None
        if not raw_l1_chunks
        else np.concatenate(raw_l1_chunks, axis=0).astype(np.float64, copy=False)
    )
    epsilon_hits = pnl > policy.config.epsilon
    negative = pnl < -policy.config.numerical_tolerance
    strict_negative = pnl < 0.0
    path_hit_rates = epsilon_hits.mean(axis=1)
    losing_paths = np.any(negative, axis=1)
    hit_probability = float(epsilon_hits.mean())
    hit_lower = _clustered_lower_bound(path_hit_rates)
    losing_path_count = int(losing_paths.sum())
    losing_path_probability_upper = _wilson_upper(losing_path_count, len(losing_paths))
    mean_long = (long_total / float(observations)).cpu().numpy()
    mean_short = (short_total / float(observations)).cpu().numpy()
    mean_abs = mean_long + mean_short
    order = np.argsort(-mean_abs)[: min(20, len(mean_abs))]
    quantile_levels = np.asarray([0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    empirical_nonnegative = bool(not np.any(negative))
    excluded = data.get("excluded_directions")
    return {
        "split": split,
        "paths": int(pnl.shape[0]),
        "steps": int(pnl.shape[1]),
        "observations": int(pnl.size),
        "epsilon": policy.config.epsilon,
        "delta": policy.config.delta,
        "numerical_tolerance": policy.config.numerical_tolerance,
        "pnl_mean": float(pnl.mean()),
        "pnl_std_population": float(pnl.std(ddof=0)),
        "pnl_quantile_levels": quantile_levels.tolist(),
        "pnl_quantiles": np.quantile(pnl, quantile_levels).tolist(),
        "pnl_min": float(pnl.min()),
        "pnl_max": float(pnl.max()),
        "strict_negative_observations": int(strict_negative.sum()),
        "negative_observations_beyond_tolerance": int(negative.sum()),
        "empirical_nonnegative_within_tolerance": empirical_nonnegative,
        "epsilon_hit_probability": hit_probability,
        "epsilon_hit_probability_path_clustered_lower_95": hit_lower,
        "losing_paths": losing_path_count,
        "losing_path_probability_upper_95": losing_path_probability_upper,
        "qualifies_empirically": bool(
            empirical_nonnegative and hit_lower > policy.config.delta
        ),
        "gross_exposure_mean": float(gross.mean()),
        "gross_exposure_quantiles": np.quantile(gross, quantile_levels).tolist(),
        "no_trade_mean": float(no_trade.mean()),
        "raw_l1_mean": None if raw_l1 is None else float(raw_l1.mean()),
        "raw_l1_max": None if raw_l1 is None else float(raw_l1.max()),
        "raw_l1_above_one_probability": (
            None if raw_l1 is None else float((raw_l1 > 1.0).mean())
        ),
        "raw_l1_above_two_probability": (
            None if raw_l1 is None else float((raw_l1 > 2.0).mean())
        ),
        "boundary_exposure_probability": float((gross > 0.99).mean()),
        "excluded_direction_fraction": (
            0.0
            if excluded is None
            else float(excluded.to(torch.float64).mean().item())
        ),
        "mean_long_by_instrument": mean_long.tolist(),
        "mean_short_by_instrument": mean_short.tolist(),
        "mean_long_short_overlap_by_instrument": (
            overlap_total / float(observations)
        ).cpu().numpy().tolist(),
        "top_instruments_by_mean_gross_exposure": [
            {
                "instrument": int(index),
                "mean_long": float(mean_long[index]),
                "mean_short": float(mean_short[index]),
                "mean_gross": float(mean_abs[index]),
            }
            for index in order
        ],
        "mc_convention": (
            "hit-rate uncertainty clusters all time steps by simulated path; "
            "loss certification rejects on any path-time loss beyond tolerance"
        ),
    }


def _checkpoint_path(
    config: DirectArbitrageConfig,
    train_config: DirectArbitrageTrainConfig,
) -> Path:
    root = Path(train_config.checkpoint_root).expanduser()
    if config.group:
        root = root.joinpath(*Path(config.group).parts)
    return root / f"{config.filename}.pt"


def _compact_valuation(value: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "pnl_mean",
        "pnl_std_population",
        "pnl_min",
        "pnl_max",
        "strict_negative_observations",
        "negative_observations_beyond_tolerance",
        "epsilon_hit_probability",
        "epsilon_hit_probability_path_clustered_lower_95",
        "losing_paths",
        "losing_path_probability_upper_95",
        "qualifies_empirically",
        "gross_exposure_mean",
        "no_trade_mean",
        "raw_l1_mean",
        "raw_l1_max",
        "raw_l1_above_one_probability",
        "raw_l1_above_two_probability",
        "boundary_exposure_probability",
        "excluded_direction_fraction",
    )
    return {name: value[name] for name in names}


def _atomic_checkpoint(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(state), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _training_checkpoint_state(
    *,
    config: DirectArbitrageConfig,
    train_config: DirectArbitrageTrainConfig,
    policy: LongShortSimplexPolicy | RadialL1Policy,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    epoch: int,
    tracker: Mapping[str, list[Any]],
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "schema": 1,
        "config_version": DirectArbitrageConfig.version.unique_id48,
        "policy_version": (
            "radial-l1-v2-v1"
            if isinstance(policy, RadialL1Policy)
            else "long-short-simplex-v3"
        ),
        "user_version": config.user_version,
        "config": asdict(config),
        "train_config": asdict(train_config),
        "epoch": epoch,
        "policy_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "generator_state": generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "tracker": dict(tracker),
    }
    if policy.device.type == "cuda":
        state["cuda_rng_state"] = torch.cuda.get_rng_state(policy.device)
    return state


def train_direct_policy(
    training: Mapping[str, Tensor],
    validation: Mapping[str, Tensor] | None,
    *,
    config: DirectArbitrageConfig,
    train_config: DirectArbitrageTrainConfig,
    epochs: int,
    env: BaseEnvironment,
    restart: bool = False,
) -> DirectArbitrageTrainingResult:
    """Train a direct policy and atomically checkpoint resumable optimizer state."""

    verify_inp(epochs > 0, lambda: f"'epochs' must be positive, got {epochs}")
    required = {"z", "dH", "cost"}
    if config.mask_arbitrage_candidates:
        required.add("excluded_directions")
    verify_inp(
        required <= training.keys(),
        lambda: f"training data is missing {sorted(required - training.keys())}",
    )
    if validation is not None:
        verify_inp(
            required <= validation.keys(),
            lambda: f"validation data is missing {sorted(required - validation.keys())}",
        )
    verify_inp(
        training["dH"].shape == training["cost"].shape,
        lambda: (
            f"training return/cost shapes differ: {training['dH'].shape} and "
            f"{training['cost'].shape}"
        ),
    )
    for split, data in (("training", training), ("validation", validation)):
        if data is None or "excluded_directions" not in data:
            continue
        excluded = data["excluded_directions"]
        expected = data["dH"].shape[:-1] + (2 * config.instruments,)
        verify_inp(
            excluded.shape == expected and excluded.dtype == torch.bool,
            lambda split=split, excluded=excluded, expected=expected: (
                f"{split} excluded directions must be bool with shape {expected}, "
                f"got {excluded.shape}/{excluded.dtype}"
            ),
        )
        if torch.any(excluded.all(dim=-1)).item():
            raise ValueError(f"{split} mask excludes every direction at a state")
    torch.manual_seed(config.seed)
    if env.device.type == "cuda":
        torch.cuda.manual_seed(config.seed)
    mean, scale = state_normalization(
        training, batch_size=train_config.valuation_batch_size
    )
    policy_class = (
        RadialL1Policy
        if config.objective == "radial-l1-v2"
        else LongShortSimplexPolicy
    )
    policy = policy_class(
        env, config=config, state_mean=mean, state_scale=scale
    )
    policy_version = (
        "radial-l1-v2-v1"
        if isinstance(policy, RadialL1Policy)
        else "long-short-simplex-v3"
    )
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    gamma = (
        train_config.final_lr / train_config.lr
    ) ** (1.0 / float(max(epochs - 1, 1)))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    generator = torch.Generator(device=env.device)
    generator.manual_seed(config.seed + 1)
    tracker: dict[str, list[Any]] = {
        "epochs": [],
        "batch_loss": [],
        "learning_rate": [],
        "gradient_norm": [],
        "valuation_epochs": [],
        "training_values": [],
        "validation_values": [],
    }
    checkpoint_path = _checkpoint_path(config, train_config)
    epoch = 0
    restarted = False

    lock = FileLock(f"{checkpoint_path}.lock")
    with lock:
        if checkpoint_path.exists() and not restart:
            state = torch.load(
                checkpoint_path, map_location=env.device, weights_only=False
            )
            compatible = (
                state.get("schema") == 1
                and state.get("config_version")
                == DirectArbitrageConfig.version.unique_id48
                and state.get("policy_version") == policy_version
                and state.get("user_version") == config.user_version
                and state.get("config") == asdict(config)
            )
            if compatible:
                policy.load_state_dict(state["policy_state"])
                optimizer.load_state_dict(state["optimizer_state"])
                scheduler.load_state_dict(state["scheduler_state"])
                generator.set_state(state["generator_state"])
                torch.set_rng_state(state["torch_rng_state"])
                if env.device.type == "cuda" and "cuda_rng_state" in state:
                    torch.cuda.set_rng_state(state["cuda_rng_state"], env.device)
                epoch = int(state["epoch"])
                tracker = state["tracker"]
                restarted = True
            else:
                print(
                    f"Ignoring incompatible direct-arbitrage checkpoint {checkpoint_path}",
                    flush=True,
                )

        def loss_step(
            batch: Mapping[str, Tensor], penalty_scale: Tensor
        ) -> Tensor:
            excluded = batch.get("excluded_directions")
            if isinstance(policy, RadialL1Policy):
                action, raw_l1 = policy(batch["z"], excluded)
                loss, _ = v2_arbitrage_loss(
                    action=action,
                    raw_l1=raw_l1,
                    dH=batch["dH"],
                    cost=batch["cost"],
                    config=config,
                    penalty_scale=penalty_scale,
                )
            else:
                long, short, _ = policy(batch["z"], excluded)
                loss, _ = direct_arbitrage_loss(
                    long=long,
                    short=short,
                    dH=batch["dH"],
                    cost=batch["cost"],
                    config=config,
                    penalty_scale=penalty_scale,
                )
            return loss

        compiled_loss_step = loss_step
        if train_config.compile and not policy.debug:
            compiled_loss_step = torch.compile(loss_step, dynamic=True)

        penalty_scale = torch.ones((), dtype=env.fdtype, device=env.device)
        def save() -> None:
            _atomic_checkpoint(
                checkpoint_path,
                _training_checkpoint_state(
                    config=config,
                    train_config=train_config,
                    policy=policy,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    generator=generator,
                    epoch=epoch,
                    tracker=tracker,
                ),
            )

        for epoch_index in range(epoch, epochs):
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            batch = _sample_paths(
                training,
                batch_size=train_config.batch_size,
                generator=generator,
            )
            progress = (epoch_index + 1) / float(epochs)
            if progress <= config.discovery_fraction:
                scale_value = 0.0
            elif progress < (
                config.discovery_fraction + config.constraint_ramp_fraction
            ):
                scale_value = (
                    progress - config.discovery_fraction
                ) / config.constraint_ramp_fraction
            else:
                scale_value = 1.0
            penalty_scale.fill_(scale_value)
            loss = compiled_loss_step(batch, penalty_scale)
            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"non-finite direct-arbitrage loss at epoch {epoch_index + 1}"
                )
            loss.backward()
            gradient_sqr = torch.zeros((), dtype=torch.float64, device=env.device)
            nonfinite_gradients = 0
            for parameter in policy.parameters():
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach()
                nonfinite_gradients += int(
                    torch.count_nonzero(~torch.isfinite(gradient)).item()
                )
                gradient_sqr += torch.nan_to_num(gradient.to(torch.float64)).square().sum()
            if nonfinite_gradients:
                raise RuntimeError(
                    f"found {nonfinite_gradients} non-finite gradient elements"
                )
            optimizer.step()
            epoch = epoch_index + 1
            if epoch < epochs:
                scheduler.step()
            tracker["epochs"].append(epoch)
            tracker["batch_loss"].append(float(loss.detach().item()))
            tracker["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
            tracker["gradient_norm"].append(float(torch.sqrt(gradient_sqr).item()))

            should_value = (
                epoch == 1
                or epoch % train_config.value_update == 0
                or epoch == epochs
            )
            if should_value:
                training_value = valuate_policy(
                    policy,
                    training,
                    batch_size=train_config.valuation_batch_size,
                    split="training",
                )
                validation_value = (
                    valuate_policy(
                        policy,
                        validation,
                        batch_size=train_config.valuation_batch_size,
                        split="validation",
                    )
                    if validation is not None
                    else None
                )
                tracker["valuation_epochs"].append(epoch)
                tracker["training_values"].append(
                    _compact_valuation(training_value)
                )
                tracker["validation_values"].append(
                    None
                    if validation_value is None
                    else _compact_valuation(validation_value)
                )
                print(
                    f"Epoch {epoch}/{epochs}: loss={loss.item():.7g}, "
                    f"train hit={training_value['epsilon_hit_probability']:.6g}, "
                    f"train losses={training_value['negative_observations_beyond_tolerance']}, "
                    f"validation hit={0.0 if validation_value is None else validation_value['epsilon_hit_probability']:.6g}, "
                    f"validation losses={0 if validation_value is None else validation_value['negative_observations_beyond_tolerance']}",
                    flush=True,
                )
                save()
            elif epoch % train_config.checkpoint_update == 0:
                save()

        training_value = valuate_policy(
            policy,
            training,
            batch_size=train_config.valuation_batch_size,
            split="training",
        )
        validation_value = (
            valuate_policy(
                policy,
                validation,
                batch_size=train_config.valuation_batch_size,
                split="validation",
            )
            if validation is not None
            else None
        )
        save()

    return DirectArbitrageTrainingResult(
        policy=policy,
        training=training_value,
        validation=validation_value,
        tracker=tracker,
        checkpoint_path=checkpoint_path,
        restarted=restarted,
    )


__all__ = [
    "DirectArbitrageConfig",
    "DirectArbitrageTrainConfig",
    "DirectArbitrageTrainingResult",
    "LongShortSimplexPolicy",
    "RadialL1Policy",
    "direct_arbitrage_loss",
    "effective_cost",
    "net_return",
    "state_normalization",
    "train_direct_policy",
    "unconditional_cost",
    "v2_arbitrage_loss",
    "valuate_policy",
]
