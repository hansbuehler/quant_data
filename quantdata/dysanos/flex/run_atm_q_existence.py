"""Find a P-free conditional front-state law satisfying the W2 ATM roll."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, cast

import torch
from cdxcore.err import verify_inp
from cdxcore.version import version
from filelock import FileLock

from quantdata import env as quantdata_env
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.model import (
    JointHierarchicalDynamicsModel,
    bound_native_raw,
    dlvsq_cap_barrier,
)
from quantdata.dysanos.flex.run_joint_hierarchical_q1 import _load_inputs
from quantdata.torch import BaseEnvironment, BaseModel

OBJECTIVE_ID = "pooled-fixed-stream-atm-and-branch-moments-v1"
_SHARED_STRIKES_TRAINING_ROOT = (
    Path(quantdata_env.locations.training.path) / "flex-shared-strikes"
)
_DEFAULT_JOINT_COMPLETION = (
    _SHARED_STRIKES_TRAINING_ROOT
    / "joint-hierarchical"
    / "v1"
    / "reports"
    / "joint-hierarchical-200-complete.json"
)
_DEFAULT_OUTPUT_ROOT = (
    _SHARED_STRIKES_TRAINING_ROOT
    / "q-existence"
    / "atm-fixed-state-exact-pooled-v1"
)


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class AtmQExistenceLawConfig:
    """Geometry of the arbitrary conditional law used by this experiment."""

    label: str = "atm-q-existence-exact-v1"
    front_state_size: int = 23
    minimum_state_noise_std: float = 1.0e-4
    initial_state_noise_std: float = 1.0e-2
    branch_atm_target_bps: float = 25.0

    def __post_init__(self) -> None:
        verify_inp(
            self.label not in {"", ".", ".."},
            lambda: f"Expected a non-empty law label, found {self.label!r}",
        )
        verify_inp(
            self.front_state_size > 0,
            lambda: (
                "'front_state_size' must be positive, found "
                f"{self.front_state_size}"
            ),
        )
        verify_inp(
            0.0 < self.minimum_state_noise_std < self.initial_state_noise_std,
            lambda: (
                "Expected 0 < minimum_state_noise_std < "
                f"initial_state_noise_std, found {self.minimum_state_noise_std} "
                f"and {self.initial_state_noise_std}"
            ),
        )
        verify_inp(
            self.branch_atm_target_bps > 0.0,
            lambda: (
                "'branch_atm_target_bps' must be positive, found "
                f"{self.branch_atm_target_bps}"
            ),
        )


class AtmQExistenceLaw(BaseModel):
    """Full-support Q law for ``h1_next`` conditional on current h1 and spot."""

    def __init__(
        self,
        env: BaseEnvironment,
        *,
        current_front_state: torch.Tensor,
        config: AtmQExistenceLawConfig,
        initialization_seed: int,
    ) -> None:
        super().__init__(env)
        if current_front_state.shape != (config.front_state_size,):
            raise ValueError(
                "Expected current front state of shape "
                f"{(config.front_state_size,)}, found {current_front_state.shape}"
            )
        self.config = config
        self.register_buffer("current_front_state", current_front_state.detach().clone())
        self.location_shift = torch.nn.Parameter(torch.zeros_like(current_front_state))
        self.spot_loading = torch.nn.Parameter(torch.zeros_like(current_front_state))
        generator = torch.Generator(device=env.device).manual_seed(initialization_seed)
        branch = torch.randn(
            config.front_state_size,
            dtype=env.fdtype,
            device=env.device,
            generator=generator,
        )
        branch = 0.05 * branch / torch.linalg.vector_norm(branch)
        self.branch_direction = torch.nn.Parameter(branch)
        initial_excess_std = (
            config.initial_state_noise_std - config.minimum_state_noise_std
        )
        self.raw_noise_std = torch.nn.Parameter(
            torch.tensor(
                math.log(math.expm1(initial_excess_std)),
                dtype=env.fdtype,
                device=env.device,
            )
        )

    @property
    def noise_std(self) -> torch.Tensor:
        """Strictly positive state-noise scale preserving full real support."""
        return self.config.minimum_state_noise_std + torch.nn.functional.softplus(
            self.raw_noise_std
        )

    def forward(
        self,
        *,
        spot_rank: torch.Tensor,
        branch: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Draw the next front state from the conditional two-Gaussian mixture."""
        expected_shape = (len(spot_rank), self.config.front_state_size)
        if branch.shape != spot_rank.shape or epsilon.shape != expected_shape:
            raise ValueError(
                "Conditional-law inputs have inconsistent shapes: "
                f"spot={spot_rank.shape}, branch={branch.shape}, "
                f"epsilon={epsilon.shape}, expected epsilon={expected_shape}"
            )
        return (
            self.current_front_state
            + self.location_shift
            + spot_rank.unsqueeze(-1) * self.spot_loading
            + branch.unsqueeze(-1) * self.branch_direction
            + self.noise_std * epsilon
        )


@dataclass(frozen=True)
class FrontWeekExactMixtureSampler:
    """Exact discrete-SANOS then Black-Scholes front-week spot sampler."""

    support: torch.Tensor
    density: torch.Tensor
    sqrt_variance: torch.Tensor
    diagnostics: dict[str, float]

    @classmethod
    def from_sanos_components(
        cls,
        *,
        support: torch.Tensor,
        density: torch.Tensor,
        sqrt_variance: torch.Tensor,
    ) -> FrontWeekExactMixtureSampler:
        """Normalize numerical residue without changing the implied law's shape."""
        if support.ndim != 1 or density.shape != support.shape:
            raise ValueError("front support and density must be aligned vectors")
        if (
            sqrt_variance.numel() != 1
            or not bool(torch.all(torch.isfinite(sqrt_variance)))
            or float(sqrt_variance) <= 0.0
        ):
            raise ValueError("front backbone total volatility must be positive")
        if (
            not bool(torch.all(torch.isfinite(support) & (support > 0.0)))
            or not bool(torch.all(torch.diff(support) > 0.0))
            or not bool(torch.all(torch.isfinite(density)))
        ):
            raise ValueError("invalid front SANOS components for exact sampling")
        sampling_density = torch.clamp_min(density.detach(), 0.0)
        mass_before = torch.sum(sampling_density)
        if not bool(torch.isfinite(mass_before)) or float(mass_before) <= 0.0:
            raise ValueError("front SANOS density has no positive finite mass")
        sampling_density = sampling_density / mass_before
        mean_before = torch.sum(sampling_density * support.detach())
        if not bool(torch.isfinite(mean_before)) or float(mean_before) <= 0.0:
            raise ValueError("front SANOS density has no positive finite mean")
        sampling_support = support.detach() / mean_before
        return cls(
            support=sampling_support,
            density=sampling_density,
            sqrt_variance=sqrt_variance.detach().reshape(()),
            diagnostics={
                "density_mass_before_normalization": float(mass_before),
                "density_mean_before_normalization": float(mean_before),
                "density_mass_after_normalization": float(torch.sum(sampling_density)),
                "density_mean_after_normalization": float(
                    torch.sum(sampling_density * sampling_support)
                ),
            },
        )

    def sample(
        self,
        *,
        num_paths: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw spot and its exact finite-mixture probability-integral rank."""
        if num_paths <= 1:
            raise ValueError(f"'num_paths' must exceed one, found {num_paths}")
        device = self.support.device
        dtype = self.support.dtype
        base = (
            torch.arange(num_paths, dtype=dtype, device=device) + 0.5
        ) / num_paths
        component_shift = torch.rand((), dtype=dtype, device=device, generator=generator)
        backbone_shift = torch.rand((), dtype=dtype, device=device, generator=generator)
        component_uniform = torch.remainder(base + component_shift, 1.0)
        backbone_order = torch.randperm(num_paths, device=device, generator=generator)
        backbone_uniform = torch.remainder(
            base[backbone_order] + backbone_shift,
            1.0,
        )
        floor = 4.0 * torch.finfo(dtype).eps
        component_uniform = torch.clamp(component_uniform, floor, 1.0 - floor)
        backbone_uniform = torch.clamp(backbone_uniform, floor, 1.0 - floor)
        cumulative_density = torch.cumsum(self.density, dim=0)
        cumulative_density[-1] = 1.0
        component = torch.searchsorted(
            cumulative_density,
            component_uniform,
        ).clamp_max(len(cumulative_density) - 1)
        backbone_normal = torch.special.ndtri(backbone_uniform)
        variance = self.sqrt_variance
        gross = self.support[component] * torch.exp(
            -0.5 * variance.square() + variance * backbone_normal
        )

        standardized = (
            torch.log(gross).unsqueeze(-1)
            - torch.log(self.support).unsqueeze(0)
            + 0.5 * variance.square()
        ) / variance
        mixture_probability = torch.sum(
            self.density.unsqueeze(0) * torch.special.ndtr(standardized),
            dim=-1,
        )
        mixture_probability = torch.clamp(
            mixture_probability,
            floor,
            1.0 - floor,
        )
        spot_rank = torch.special.ndtri(mixture_probability)
        if not bool(torch.all(torch.isfinite(gross) & (gross > 0.0))):
            raise FloatingPointError("exact front mixture produced invalid spot")
        if not bool(torch.all(torch.isfinite(spot_rank))):
            raise FloatingPointError("exact front mixture produced invalid PIT rank")
        return gross, spot_rank


@dataclass(frozen=True)
class QExistenceBatch:
    """One fixed Monte Carlo replication of the conditional roll experiment."""

    spot_rank: torch.Tensor
    gross_spot_return: torch.Tensor
    branch: torch.Tensor
    epsilon: torch.Tensor

    @property
    def num_paths(self) -> int:
        return len(self.spot_rank)


def make_q_existence_batch(
    *,
    front_sampler: FrontWeekExactMixtureSampler,
    front_state_size: int,
    spot_contexts: int,
    surface_paths: int,
    seed: int,
) -> QExistenceBatch:
    """Make fixed, balanced spot/state shocks for one replication."""
    if spot_contexts < 2 or surface_paths < 4 or surface_paths % 4:
        raise ValueError("batch needs at least two spots and surface paths divisible by four")
    generator = torch.Generator(device=front_sampler.support.device).manual_seed(seed)
    gross, spot_rank = front_sampler.sample(
        num_paths=spot_contexts,
        generator=generator,
    )
    quarter = surface_paths // 4
    epsilon_quarter = torch.randn(
        (spot_contexts, quarter, front_state_size),
        dtype=front_sampler.support.dtype,
        device=front_sampler.support.device,
        generator=generator,
    )
    epsilon_half = torch.cat((epsilon_quarter, -epsilon_quarter), dim=1)
    epsilon = torch.cat((epsilon_half, epsilon_half), dim=1)
    branch = torch.cat(
        (
            -torch.ones(
                surface_paths // 2,
                dtype=front_sampler.support.dtype,
                device=front_sampler.support.device,
            ),
            torch.ones(
                surface_paths // 2,
                dtype=front_sampler.support.dtype,
                device=front_sampler.support.device,
            ),
        )
    ).unsqueeze(0).expand(spot_contexts, -1)
    return QExistenceBatch(
        spot_rank=spot_rank[:, None].expand(-1, surface_paths).reshape(-1),
        gross_spot_return=gross[:, None].expand(-1, surface_paths).reshape(-1),
        branch=branch.reshape(-1),
        epsilon=epsilon.reshape(-1, front_state_size),
    )


def _price_calls(
    model: JointHierarchicalDynamicsModel,
    state: torch.Tensor,
    *,
    expiry_index: int,
    normalized_strikes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = model.raw_states_from_states(state)
    parameters = bound_native_raw(raw, geometry=model.config.geometry)
    surface = model.sanos_decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )
    expiry = model.sanos_decoder.model_expiries_y[expiry_index].expand_as(
        normalized_strikes
    )
    return (
        model.sanos_decoder.price_calls_batch(
            surface,
            expiries_y=expiry,
            strikes=normalized_strikes,
        ),
        raw,
    )


def atm_roll_returns(
    *,
    model: JointHierarchicalDynamicsModel,
    law: AtmQExistenceLaw,
    current_state: torch.Tensor,
    current_w2_atm_call: torch.Tensor,
    batch: QExistenceBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``S_next C_W1(K/S_next)-C_W2(K)`` at fixed cash strike K."""
    next_front_state = law(
        spot_rank=batch.spot_rank,
        branch=batch.branch,
        epsilon=batch.epsilon,
    )
    next_state = current_state.unsqueeze(0).expand(batch.num_paths, -1).clone()
    next_state[:, : law.config.front_state_size] = next_front_state
    next_normalized_strike = (1.0 / batch.gross_spot_return).unsqueeze(-1)
    next_w1_call, raw = _price_calls(
        model,
        next_state,
        expiry_index=0,
        normalized_strikes=next_normalized_strike,
    )
    rolled_call = batch.gross_spot_return * next_w1_call[:, 0]
    return rolled_call - current_w2_atm_call, raw


def spot_control_variate(
    returns: torch.Tensor,
    gross_spot_return: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use the exact population identity ``E[S_next/S_now-1]=0``."""
    if returns.shape != gross_spot_return.shape or returns.ndim != 1:
        raise ValueError("spot control variate requires aligned path vectors")
    spot_return = gross_spot_return - 1.0
    centered_spot = spot_return - torch.mean(spot_return)
    centered_return = returns - torch.mean(returns)
    denominator = torch.sum(centered_spot.square()).clamp_min(
        torch.finfo(returns.dtype).tiny
    )
    beta = (torch.sum(centered_spot * centered_return) / denominator).detach()
    adjusted = returns - beta * spot_return
    return adjusted, torch.mean(adjusted), beta


def _replicated_statistics(
    returns_by_replication: list[torch.Tensor],
    batches: list[QExistenceBatch],
    *,
    branch_target_bps: float,
) -> dict[str, object]:
    if len(returns_by_replication) != len(batches) or not batches:
        raise ValueError("replicated statistics require aligned non-empty inputs")
    adjusted_means: list[torch.Tensor] = []
    raw_means: list[torch.Tensor] = []
    negative_means: list[torch.Tensor] = []
    positive_means: list[torch.Tensor] = []
    betas: list[torch.Tensor] = []
    all_returns: list[torch.Tensor] = []
    for returns, batch in zip(returns_by_replication, batches, strict=True):
        _, adjusted_mean, beta = spot_control_variate(
            returns,
            batch.gross_spot_return,
        )
        negative = batch.branch < 0.0
        positive = batch.branch > 0.0
        _, negative_mean, _ = spot_control_variate(
            returns[negative],
            batch.gross_spot_return[negative],
        )
        _, positive_mean, _ = spot_control_variate(
            returns[positive],
            batch.gross_spot_return[positive],
        )
        adjusted_means.append(adjusted_mean)
        raw_means.append(torch.mean(returns))
        negative_means.append(negative_mean)
        positive_means.append(positive_mean)
        betas.append(beta)
        all_returns.append(returns)
    adjusted = torch.stack(adjusted_means)
    raw = torch.stack(raw_means)
    negative = torch.stack(negative_means)
    positive = torch.stack(positive_means)
    combined = torch.cat(all_returns)
    adjusted_mean = torch.mean(adjusted)
    adjusted_se = torch.std(adjusted, correction=1) / math.sqrt(len(adjusted))
    raw_mean = torch.mean(raw)
    raw_se = torch.std(raw, correction=1) / math.sqrt(len(raw))
    std = torch.std(combined, correction=1)
    quantiles = torch.quantile(
        combined,
        torch.tensor(
            (0.01, 0.05, 0.50, 0.95, 0.99),
            dtype=combined.dtype,
            device=combined.device,
        ),
    )
    return {
        "replications": len(batches),
        "paths_per_replication": batches[0].num_paths,
        "total_paths": len(combined),
        "adjusted_mean_bps_spot": float(adjusted_mean) * 1.0e4,
        "adjusted_mean_mc_standard_error_bps_spot": float(adjusted_se) * 1.0e4,
        "adjusted_mean_z": float(
            adjusted_mean / adjusted_se.clamp_min(torch.finfo(adjusted.dtype).tiny)
        ),
        "raw_mean_bps_spot": float(raw_mean) * 1.0e4,
        "raw_mean_mc_standard_error_bps_spot": float(raw_se) * 1.0e4,
        "return_sample_std_bps_spot": float(std) * 1.0e4,
        "negative_return_percent": float(torch.mean((combined < 0.0).float())) * 100.0,
        "positive_return_percent": float(torch.mean((combined > 0.0).float())) * 100.0,
        "negative_branch_adjusted_mean_bps_spot": float(torch.mean(negative)) * 1.0e4,
        "positive_branch_adjusted_mean_bps_spot": float(torch.mean(positive)) * 1.0e4,
        "branch_target_bps_spot": branch_target_bps,
        "spot_control_beta_mean": float(torch.mean(torch.stack(betas))),
        "return_quantiles_bps_spot": {
            label: float(value) * 1.0e4
            for label, value in zip(
                ("q01", "q05", "q50", "q95", "q99"),
                quantiles,
                strict=True,
            )
        },
        "gross_spot_return_mean": float(
            torch.mean(torch.cat([batch.gross_spot_return for batch in batches]))
        ),
    }


def _atomic_torch_save(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--joint-completion",
        type=Path,
        default=_DEFAULT_JOINT_COMPLETION,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT / "reports" / "atm-q-existence.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT / "checkpoints" / "atm-q-existence.pt",
    )
    parser.add_argument("--worker-heartbeat", type=Path, default=None)
    parser.add_argument("--generation-token", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--selected-validation-index", type=int, default=62)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--fit-replications", type=int, default=24)
    parser.add_argument("--validation-replications", type=int, default=24)
    parser.add_argument("--spot-contexts", type=int, default=125)
    parser.add_argument("--surface-paths", type=int, default=8)
    parser.add_argument("--fit-seed", type=int, default=9_284_101)
    parser.add_argument("--validation-seed", type=int, default=9_285_101)
    parser.add_argument("--initialization-seed", type=int, default=9_283_101)
    parser.add_argument("--stable-epochs", type=int, default=20)
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.device != "cuda:0":
        raise ValueError("ATM Q-existence training is restricted to GPU0")
    if arguments.selected_validation_index < 0:
        raise ValueError("selected validation index must be non-negative")
    positive = (
        arguments.epochs,
        arguments.fit_replications,
        arguments.validation_replications,
        arguments.spot_contexts,
        arguments.surface_paths,
        arguments.stable_epochs,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("all ATM Q-existence training counts must be positive")
    if arguments.fit_replications < 12 or arguments.validation_replications < 12:
        raise ValueError("Monte Carlo inference requires at least twelve replications")
    if arguments.surface_paths < 4 or arguments.surface_paths % 4:
        raise ValueError("surface paths must be at least four and divisible by four")
    if not math.isfinite(arguments.learning_rate) or arguments.learning_rate <= 0.0:
        raise ValueError("learning rate must be finite and positive")
    if arguments.fit_seed == arguments.validation_seed:
        raise ValueError("fit and validation random streams must be disjoint")
    if (arguments.worker_heartbeat is None) != (arguments.generation_token is None):
        raise ValueError("worker heartbeat and generation token must be specified together")


def _write_heartbeat(
    arguments: argparse.Namespace,
    *,
    phase: str,
    completed: int,
    total: int,
    status: str = "RUNNING",
) -> None:
    if arguments.worker_heartbeat is None:
        return
    write_json_atomic(
        {
            "schema": "flex-atm-q-existence-worker-heartbeat-v1",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "generation": arguments.generation_token,
            "worker_pid": os.getpid(),
            "status": status,
            "last_progress_value": {
                "phase": phase,
                "completed": completed,
                "total": total,
            },
        },
        arguments.worker_heartbeat,
    )


def _make_batches(
    *,
    front_sampler: FrontWeekExactMixtureSampler,
    config: AtmQExistenceLawConfig,
    replications: int,
    spot_contexts: int,
    surface_paths: int,
    seed: int,
) -> list[QExistenceBatch]:
    return [
        make_q_existence_batch(
            front_sampler=front_sampler,
            front_state_size=config.front_state_size,
            spot_contexts=spot_contexts,
            surface_paths=surface_paths,
            seed=seed + replication * 1_000_003,
        )
        for replication in range(replications)
    ]


@torch.no_grad()
def _evaluate(
    *,
    model: JointHierarchicalDynamicsModel,
    law: AtmQExistenceLaw,
    current_state: torch.Tensor,
    current_w2_atm_call: torch.Tensor,
    batches: list[QExistenceBatch],
) -> dict[str, object]:
    returns = [
        atm_roll_returns(
            model=model,
            law=law,
            current_state=current_state,
            current_w2_atm_call=current_w2_atm_call,
            batch=batch,
        )[0]
        for batch in batches
    ]
    return _replicated_statistics(
        returns,
        batches,
        branch_target_bps=law.config.branch_atm_target_bps,
    )


def _training_checkpoint(
    *,
    arguments: argparse.Namespace,
    config: AtmQExistenceLawConfig,
    model_sources: Mapping[str, str],
    model_date: str,
    law: AtmQExistenceLaw,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    stable: int,
    history: list[dict[str, float]],
    status: str,
) -> dict[str, object]:
    model_state_dict = law.state_dict()
    return {
        "schema": "flex-atm-q-existence-training-v1",
        "status": status,
        "phase": "complete" if status == "complete" else "training",
        "objective": OBJECTIVE_ID,
        "generation": arguments.generation_token,
        "law_config": asdict(config),
        "source_completion_sha256": model_sources["completion_sha256"],
        "market_date": model_date,
        "epoch": epoch,
        "stable_epochs": stable,
        # Publish both the experiment-specific and generic names.  The latter
        # lets the durable pipeline validator inspect tensor finiteness without
        # needing a one-off schema exception for this research worker.
        "law_state_dict": model_state_dict,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "fit_seed": arguments.fit_seed,
        "validation_seed": arguments.validation_seed,
    }


def run(arguments: argparse.Namespace) -> None:
    """Train and independently validate the fixed-state Q-existence law."""
    _validate_arguments(arguments)
    torch.set_float32_matmul_precision("high")
    env = BaseEnvironment(precision=32, device=arguments.device, debug=False)
    (
        model,
        _,
        _,
        _,
        validation_dates,
        validation_states,
        _,
        sources,
    ) = _load_inputs(arguments.joint_completion, env=env)
    if arguments.selected_validation_index >= len(validation_states):
        raise ValueError("selected validation index is outside the 2025 state history")
    if not arguments.no_compile:
        model.sanos_decoder.enable_solver_compile()
    current_state = validation_states[arguments.selected_validation_index].detach()
    model_date = validation_dates[arguments.selected_validation_index].isoformat()
    front_state_size = model.config.state_sizes_by_expiry[0]
    config = AtmQExistenceLawConfig(front_state_size=front_state_size)
    law = AtmQExistenceLaw(
        env,
        current_front_state=current_state[:front_state_size],
        config=config,
        initialization_seed=arguments.initialization_seed,
    )
    with torch.no_grad():
        raw = model.raw_states_from_states(current_state.unsqueeze(0))
        parameters = bound_native_raw(raw, geometry=model.config.geometry)
        surface = model.sanos_decoder.forward_batch(
            dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols,
            eta=parameters.eta,
        )
        current_w2_atm_call = _price_calls(
            model,
            current_state.unsqueeze(0),
            expiry_index=1,
            normalized_strikes=torch.ones((1, 1), dtype=env.fdtype, device=env.device),
        )[0][0, 0].detach()
    front_sampler = FrontWeekExactMixtureSampler.from_sanos_components(
        support=surface.support_strikes[0, 0],
        density=surface.densities[0, 0],
        sqrt_variance=surface.backbone_total_vols[0, 0],
    )
    fit_batches = _make_batches(
        front_sampler=front_sampler,
        config=config,
        replications=arguments.fit_replications,
        spot_contexts=arguments.spot_contexts,
        surface_paths=arguments.surface_paths,
        seed=arguments.fit_seed,
    )
    validation_batches = _make_batches(
        front_sampler=front_sampler,
        config=config,
        replications=arguments.validation_replications,
        spot_contexts=arguments.spot_contexts,
        surface_paths=arguments.surface_paths,
        seed=arguments.validation_seed,
    )
    optimizer = torch.optim.Adam(law.parameters(), lr=arguments.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=15,
        min_lr=5.0e-5,
    )
    history: list[dict[str, float]] = []
    start_epoch = 0
    stable = 0
    if arguments.checkpoint.exists():
        checkpoint = torch.load(
            arguments.checkpoint,
            map_location=env.device,
            weights_only=False,
        )
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("schema") != "flex-atm-q-existence-training-v1"
            or checkpoint.get("law_config") != asdict(config)
            or checkpoint.get("source_completion_sha256")
            != sources["completion_sha256"]
            or checkpoint.get("market_date") != model_date
            or checkpoint.get("fit_seed") != arguments.fit_seed
            or checkpoint.get("validation_seed") != arguments.validation_seed
        ):
            raise ValueError("ATM Q-existence checkpoint identity is incompatible")
        law.load_state_dict(cast(Mapping[str, torch.Tensor], checkpoint["law_state_dict"]))
        if checkpoint.get("objective") == OBJECTIVE_ID:
            optimizer.load_state_dict(
                cast(dict[str, Any], checkpoint["optimizer_state_dict"])
            )
            scheduler.load_state_dict(
                cast(dict[str, Any], checkpoint["scheduler_state_dict"])
            )
            start_epoch = int(checkpoint["epoch"])
            stable = int(checkpoint["stable_epochs"])
            history = list(cast(list[dict[str, float]], checkpoint["history"]))
        else:
            # Retain the useful law parameters from the exploratory worker, but
            # reset optimizer/history because it minimized squared moments by
            # replication rather than the one pooled empirical Q restriction.
            print(
                "warm-starting pooled objective from the pre-pooled ATM law; "
                "optimizer, scheduler, and convergence history reset",
                flush=True,
            )
        if checkpoint.get("status") == "complete" and arguments.output.exists():
            print(f"ATM Q-existence already complete: {arguments.output}", flush=True)
            return

    _write_heartbeat(
        arguments,
        phase="training",
        completed=start_epoch,
        total=arguments.epochs,
    )
    target = config.branch_atm_target_bps
    for epoch in range(start_epoch, arguments.epochs):
        # First evaluate all fixed replications without retaining 24 decoder
        # graphs. The pooled moments are the empirical Q restriction;
        # individual scrambles estimate its Monte Carlo uncertainty rather
        # than defining separate moment conditions to fit.
        epoch_returns: list[torch.Tensor] = []
        cap_values: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in fit_batches:
                returns, raw_next = atm_roll_returns(
                    model=model,
                    law=law,
                    current_state=current_state,
                    current_w2_atm_call=current_w2_atm_call,
                    batch=batch,
                )
                epoch_returns.append(returns)
                cap_values.append(
                    1.0e-4
                    * torch.mean(
                        dlvsq_cap_barrier(
                            raw_next,
                            geometry=model.config.geometry,
                        ).square()
                    )
                )
        fit_statistics = _replicated_statistics(
            epoch_returns,
            fit_batches,
            branch_target_bps=target,
        )
        pooled_mean_bps = torch.tensor(
            float(fit_statistics["adjusted_mean_bps_spot"]),
            dtype=env.fdtype,
            device=env.device,
        )
        pooled_negative_bps = torch.tensor(
            float(fit_statistics["negative_branch_adjusted_mean_bps_spot"]),
            dtype=env.fdtype,
            device=env.device,
        )
        pooled_positive_bps = torch.tensor(
            float(fit_statistics["positive_branch_adjusted_mean_bps_spot"]),
            dtype=env.fdtype,
            device=env.device,
        )
        loss = (
            pooled_mean_bps.square()
            + 0.01
            * (
                (pooled_negative_bps + target).square()
                + (pooled_positive_bps - target).square()
            )
            + torch.mean(torch.stack(cap_values))
        )
        loss_value = float(loss)

        # A second microbatched pass applies the analytic derivative of the
        # pooled objective. Each surrogate has the same parameter gradient as
        # the full 24,000-path graph while only one 1,000-path graph is alive.
        optimizer.zero_grad(set_to_none=True)
        for batch in fit_batches:
            returns, raw_next = atm_roll_returns(
                model=model,
                law=law,
                current_state=current_state,
                current_w2_atm_call=current_w2_atm_call,
                batch=batch,
            )
            _, adjusted_mean, _ = spot_control_variate(
                returns,
                batch.gross_spot_return,
            )
            negative = batch.branch < 0.0
            positive = batch.branch > 0.0
            _, negative_mean, _ = spot_control_variate(
                returns[negative],
                batch.gross_spot_return[negative],
            )
            _, positive_mean, _ = spot_control_variate(
                returns[positive],
                batch.gross_spot_return[positive],
            )
            replication_count = len(fit_batches)
            gradient_surrogate = (
                2.0
                * pooled_mean_bps
                * (1.0e4 * adjusted_mean)
                / replication_count
                + 0.02
                * (pooled_negative_bps + target)
                * (1.0e4 * negative_mean)
                / replication_count
                + 0.02
                * (pooled_positive_bps - target)
                * (1.0e4 * positive_mean)
                / replication_count
                + 1.0e-4
                * torch.mean(
                    dlvsq_cap_barrier(
                        raw_next,
                        geometry=model.config.geometry,
                    ).square()
                )
                / replication_count
            )
            gradient_surrogate.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(law.parameters(), 10.0)
        optimizer.step()
        scheduler.step(loss_value)
        record = {
            "epoch": float(epoch + 1),
            "loss": loss_value,
            "adjusted_mean_bps_spot": float(fit_statistics["adjusted_mean_bps_spot"]),
            "raw_mean_bps_spot": float(fit_statistics["raw_mean_bps_spot"]),
            "return_sample_std_bps_spot": float(
                fit_statistics["return_sample_std_bps_spot"]
            ),
            "negative_branch_adjusted_mean_bps_spot": float(
                fit_statistics["negative_branch_adjusted_mean_bps_spot"]
            ),
            "positive_branch_adjusted_mean_bps_spot": float(
                fit_statistics["positive_branch_adjusted_mean_bps_spot"]
            ),
            "state_noise_std": float(law.noise_std.detach()),
            "gradient_norm": float(gradient_norm.detach()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if not all(math.isfinite(value) for value in record.values()):
            raise FloatingPointError("non-finite ATM Q-existence training record")
        history.append(record)
        passed_fit_gate = (
            abs(record["adjusted_mean_bps_spot"]) <= 0.50
            and abs(record["negative_branch_adjusted_mean_bps_spot"] + target) <= 1.0
            and abs(record["positive_branch_adjusted_mean_bps_spot"] - target) <= 1.0
            and record["return_sample_std_bps_spot"] >= 10.0
        )
        stable = stable + 1 if passed_fit_gate else 0
        if epoch == start_epoch or (epoch + 1) % 5 == 0:
            print(
                f"epoch={epoch + 1} loss={loss_value:.6g} "
                f"mean={record['adjusted_mean_bps_spot']:.4f}bp "
                f"branches=({record['negative_branch_adjusted_mean_bps_spot']:.3f},"
                f"{record['positive_branch_adjusted_mean_bps_spot']:.3f})bp "
                f"std={record['return_sample_std_bps_spot']:.3f}bp "
                f"lr={record['learning_rate']:.6g}",
                flush=True,
            )
        _write_heartbeat(
            arguments,
            phase="training",
            completed=epoch + 1,
            total=arguments.epochs,
        )
        if (epoch + 1) % 5 == 0 or stable >= arguments.stable_epochs:
            _atomic_torch_save(
                _training_checkpoint(
                    arguments=arguments,
                    config=config,
                    model_sources=sources,
                    model_date=model_date,
                    law=law,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    stable=stable,
                    history=history,
                    status="training",
                ),
                arguments.checkpoint,
            )
        if stable >= arguments.stable_epochs:
            break

    _write_heartbeat(arguments, phase="validation", completed=0, total=1)
    fit_audit = _evaluate(
        model=model,
        law=law,
        current_state=current_state,
        current_w2_atm_call=current_w2_atm_call,
        batches=fit_batches,
    )
    validation_audit = _evaluate(
        model=model,
        law=law,
        current_state=current_state,
        current_w2_atm_call=current_w2_atm_call,
        batches=validation_batches,
    )
    validation_mean = abs(float(validation_audit["adjusted_mean_bps_spot"]))
    validation_se = float(
        validation_audit["adjusted_mean_mc_standard_error_bps_spot"]
    )
    validation_std = float(validation_audit["return_sample_std_bps_spot"])
    validation_positive = float(validation_audit["positive_return_percent"])
    validation_negative = float(validation_audit["negative_return_percent"])
    semantic_success = (
        validation_mean <= max(1.0, 3.0 * validation_se)
        and validation_std >= 10.0
        and validation_positive >= 20.0
        and validation_negative >= 20.0
    )
    parameters = {
        "location_shift": law.location_shift.detach().cpu().tolist(),
        "spot_loading": law.spot_loading.detach().cpu().tolist(),
        "branch_direction": law.branch_direction.detach().cpu().tolist(),
        "state_noise_std": float(law.noise_std.detach()),
    }
    report = {
        "schema": "flex-atm-q-existence-exact-v1",
        "status": "complete" if semantic_success else "failed_semantic_validation",
        "semantic_success": semantic_success,
        "generation": arguments.generation_token,
        "experiment": "Q-existence",
        "measure": {
            "P_used": False,
            "P_transition_used": False,
            "P_density_or_distance_penalty_used": False,
            "spot": "exact current front-week SANOS/lognormal mixture",
            "conditional_front_state_law": (
                "h1_next = h1_now + a + B*z_spot + b*d + sigma*epsilon; "
                "b is equiprobable +/-1 and epsilon is standard Gaussian"
            ),
            "full_real_support": True,
        },
        "timing": {
            "current_option": "normalized ATM W2 call at fixed cash strike K=S_t",
            "rolled_option": "W1 call at t+1 with the same cash strike and expiry",
            "restriction": (
                "C_t_W2(1) = E_Q[(S_t+1/S_t) "
                "C_t+1_W1(1/(S_t+1/S_t)) | fixed current state]"
            ),
        },
        "selection": {
            "validation_index": arguments.selected_validation_index,
            "market_date": model_date,
            "current_state": current_state.detach().cpu().tolist(),
            "current_front_state_h1": current_state[:front_state_size].cpu().tolist(),
            "current_w2_atm_call_per_spot": float(current_w2_atm_call),
        },
        "configuration": {
            "law": asdict(config),
            "fit_replications": arguments.fit_replications,
            "validation_replications": arguments.validation_replications,
            "spot_contexts_per_replication": arguments.spot_contexts,
            "surface_paths_per_spot": arguments.surface_paths,
            "fit_total_paths": (
                arguments.fit_replications
                * arguments.spot_contexts
                * arguments.surface_paths
            ),
            "validation_total_paths": (
                arguments.validation_replications
                * arguments.spot_contexts
                * arguments.surface_paths
            ),
            "fit_random_numbers_fixed_during_training": True,
            "fit_seed": arguments.fit_seed,
            "validation_seed": arguments.validation_seed,
            "validation_stream_disjoint_from_fit": True,
            "compiled_sanos_solver": not arguments.no_compile,
        },
        "front_spot_sampler": {
            "method": (
                "categorical SANOS atom followed by independent mean-one "
                "Black-Scholes lognormal factor"
            ),
            "conditioning_coordinate": (
                "exact finite-lognormal-mixture CDF rank transformed by "
                "the standard-normal inverse CDF"
            ),
            "population_mean_gross_return": 1.0,
            **front_sampler.diagnostics,
        },
        "training": {
            "objective": OBJECTIVE_ID,
            "epochs_completed": len(history),
            "converged_fit_gate": stable >= arguments.stable_epochs,
            "stable_epochs": stable,
            "history": history,
            "audit": fit_audit,
        },
        "independent_validation": validation_audit,
        "parameters": parameters,
        "sources": sources,
    }
    write_json_atomic(report, arguments.output)
    checkpoint_payload = _training_checkpoint(
        arguments=arguments,
        config=config,
        model_sources=sources,
        model_date=model_date,
        law=law,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=len(history),
        stable=stable,
        history=history,
        status="complete" if semantic_success else "failed_terminal",
    )
    checkpoint_payload["report"] = str(arguments.output)
    checkpoint_payload["report_sha256"] = checkpoint_digest(arguments.output)
    _atomic_torch_save(checkpoint_payload, arguments.checkpoint)
    completion_path = arguments.output.with_name(f"{arguments.output.stem}-complete.json")
    write_json_atomic(
        {
            "schema": "flex-atm-q-existence-completion-v1",
            "status": "complete" if semantic_success else "FAILED_TERMINAL",
            "semantic_success": semantic_success,
            "generation": arguments.generation_token,
            "report": str(arguments.output),
            "report_sha256": checkpoint_digest(arguments.output),
            "checkpoint": str(arguments.checkpoint),
            "checkpoint_sha256": checkpoint_digest(arguments.checkpoint),
            "source_completion": str(arguments.joint_completion),
            "source_completion_sha256": sources["completion_sha256"],
        },
        completion_path,
    )
    _write_heartbeat(
        arguments,
        phase="complete" if semantic_success else "failed-terminal",
        completed=1,
        total=1,
        status="SUCCEEDED" if semantic_success else "FAILED_TERMINAL",
    )
    print(json.dumps(validation_audit, indent=2), flush=True)
    if not semantic_success:
        raise RuntimeError("ATM Q-existence law failed independent validation")


def main() -> None:
    """CLI entry point with one lock per durable experiment checkpoint."""
    arguments = _parse_arguments()
    arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(arguments.checkpoint.with_suffix(".lock")):
        run(arguments)


if __name__ == "__main__":
    run_guarded_cli(
        main,
        program="run_atm_q_existence",
        clean_sigterm=True,
    )
