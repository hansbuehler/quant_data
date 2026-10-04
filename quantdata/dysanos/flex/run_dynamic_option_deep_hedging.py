"""Run the 25-day dynamic-option FLEX OCE deep-hedging experiment."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version
from filelock import FileLock

from quantdata.dysanos.flex.dynamic_option_hedging import (
    DynamicOptionPolicy,
    DynamicOptionPolicyConfig,
    five_day_straddle_claim,
    interpolated_option_returns,
    marginal_utility_density,
    oce_value,
)
from quantdata.dysanos.flex.front_spot_copula import fit_front_spot_copula_from_history
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.oce_curvature_optimizers import Shampoo
from quantdata.dysanos.flex.presentation_hedging import (
    exact_daily_transition,
    load_presentation_model,
)
from quantdata.dysanos.flex.sanos import SanosSurface
from quantdata.dysanos.flex.weekly_simulation import front_week_sanos_inverse_cdf


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class DynamicHedgeTrainConfig:
    """Operational controls for one dynamic-policy fit."""

    batch_size: int = 384
    maximum_epochs: int = 100
    validation_every: int = 2
    validation_batch_size: int = 250
    antithetic_block_size: int = 1_000
    learning_rate: float = 2.0e-3
    minimum_learning_rate: float = 1.0e-5
    gradient_clip_norm: float = 100.0
    optimizer: str = "shampoo"
    shampoo_damping: float = 1.0e-4
    shampoo_precondition_frequency: int = 5
    maximum_parameter_step: float = 0.05
    validation_gradient_z: float = 2.0
    validation_gradient_rms_tolerance: float = 1.0e-2
    validation_patience: int = 3
    minimum_validation_ess_fraction: float = 1.0e-3
    maximum_validation_value_drop: float = 5.0
    maximum_cap_fraction: float = 0.95
    maximum_raw_action: float = 1.0e6
    maximum_option_return: float = 1.0e8
    maximum_gradient_norm: float = 1.0e8
    minimum_ess_fraction: float = 1.0e-5

    def __post_init__(self) -> None:
        verify_inp(
            min(
                self.batch_size,
                self.maximum_epochs,
                self.validation_every,
                self.validation_batch_size,
                self.antithetic_block_size,
                self.validation_patience,
            )
            > 0,
            lambda: "training integer controls must be positive",
        )
        verify_inp(
            self.optimizer in {"adamw", "shampoo"},
            lambda: f"unknown optimizer {self.optimizer!r}",
        )
        positive = (
            self.learning_rate,
            self.minimum_learning_rate,
            self.gradient_clip_norm,
            self.shampoo_damping,
            self.maximum_parameter_step,
            self.validation_gradient_z,
            self.validation_gradient_rms_tolerance,
            self.minimum_validation_ess_fraction,
            self.maximum_validation_value_drop,
            self.maximum_raw_action,
            self.maximum_option_return,
            self.maximum_gradient_norm,
            self.minimum_ess_fraction,
        )
        verify_inp(
            all(math.isfinite(value) and value > 0.0 for value in positive),
            lambda: "training scalar controls must be finite-positive",
        )
        verify_inp(
            self.minimum_learning_rate <= self.learning_rate,
            lambda: "minimum learning rate exceeds the initial rate",
        )
        verify_inp(
            self.shampoo_precondition_frequency > 0,
            lambda: "Shampoo precondition frequency must be positive",
        )
        verify_inp(
            0.0 < self.maximum_cap_fraction <= 1.0,
            lambda: "maximum cap fraction must lie in (0,1]",
        )
        verify_inp(
            self.minimum_ess_fraction <= 1.0,
            lambda: "minimum ESS fraction cannot exceed one",
        )
        verify_inp(
            self.validation_batch_size % 2 == 0
            and self.antithetic_block_size % self.validation_batch_size == 0,
            lambda: (
                "validation batch size must be even and divide the antithetic block"
            ),
        )


@dataclass(frozen=True)
class PathPopulation:
    """Memory-mapped fixed state and spot paths."""

    states: np.ndarray
    spots: np.ndarray

    def __len__(self) -> int:
        return len(self.states)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-history", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--stage", choices=("generate", "train", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--paths-per-population", type=int, default=100_000)
    parser.add_argument("--generation-batch-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2_026_100_1)
    parser.add_argument("--maximum-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--validation-batch-size", type=int, default=250)
    parser.add_argument("--antithetic-block-size", type=int, default=1_000)
    parser.add_argument("--validation-every", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--optimizer", choices=("adamw", "shampoo"), default="shampoo")
    parser.add_argument("--action-l1-cap", type=float, default=16.0)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--network-width", type=int, default=128)
    parser.add_argument("--network-depth", type=int, default=2)
    parser.add_argument("--spot-cost-bps", type=float, default=1.0)
    parser.add_argument("--option-cost-bps", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--copula-fit-before", type=date.fromisoformat, default=date(2025, 1, 1))
    return parser.parse_args()


def _decode(model: Any, states: torch.Tensor) -> Any:
    with torch.no_grad():
        raw = model.raw_states_from_states(states)
        parameters = bound_native_raw(raw, geometry=model.config.geometry)
        return model.sanos_decoder.forward_batch(
            dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols,
            eta=parameters.eta,
        )


def _decode_path_surfaces(model: Any, states: torch.Tensor) -> SanosSurface:
    """Decode all issue-day surfaces in one frozen batched operation."""

    if states.ndim != 3 or states.shape[1] != 25:
        raise ValueError("surface path states must have shape (paths,25,state_size)")
    paths, days, state_size = states.shape
    flat = _decode(model, states.reshape(paths * days, state_size))

    def reshape(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(paths, days, *value.shape[1:])

    return SanosSurface(
        expiries_y=reshape(flat.expiries_y),
        support_strikes=reshape(flat.support_strikes),
        density0=reshape(flat.density0),
        densities=reshape(flat.densities),
        dlvsqs=reshape(flat.dlvsqs),
        forward_vols=reshape(flat.forward_vols),
        total_vols=reshape(flat.total_vols),
        eta=reshape(flat.eta),
        backbone_total_vols=reshape(flat.backbone_total_vols),
    )


def _surface_day(surfaces: SanosSurface, day: int) -> SanosSurface:
    """Return one issue-day view from decoded path surfaces."""

    return SanosSurface(
        expiries_y=surfaces.expiries_y[:, day],
        support_strikes=surfaces.support_strikes[:, day],
        density0=surfaces.density0[:, day],
        densities=surfaces.densities[:, day],
        dlvsqs=surfaces.dlvsqs[:, day],
        forward_vols=surfaces.forward_vols[:, day],
        total_vols=surfaces.total_vols[:, day],
        eta=surfaces.eta[:, day],
        backbone_total_vols=surfaces.backbone_total_vols[:, day],
    )


def _load_history(path: Path) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    dates = payload.get("dates")
    states = payload.get("states")
    spots = payload.get("spot_refs")
    if not isinstance(dates, list) or not isinstance(states, torch.Tensor) or not isinstance(spots, torch.Tensor):
        raise ValueError("target history is missing dates, states, or spot references")
    if dates[-1] != "2025-08-29":
        raise ValueError(f"expected initial date 2025-08-29, found {dates[-1]!r}")
    return dates, states, spots


def _atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _population_paths(directory: Path, name: str) -> tuple[Path, Path]:
    return directory / f"{name}-states.npy", directory / f"{name}-spots.npy"


def _generate_population(
    *,
    name: str,
    output_directory: Path,
    model: Any,
    initial_state: torch.Tensor,
    daily_transition: torch.Tensor,
    daily_factor: torch.Tensor,
    copula_loading: torch.Tensor,
    copula_orthogonal_std: float,
    copula_total_std: float,
    paths: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """Generate one independent fixed 25-day path population."""

    if paths <= 0 or paths % 2 or batch_size <= 0 or batch_size % 2:
        raise ValueError("path and generation batch counts must be positive and even")
    state_path, spot_path = _population_paths(output_directory, name)
    if state_path.exists() and spot_path.exists():
        states = np.load(state_path, mmap_mode="r", allow_pickle=False)
        spots = np.load(spot_path, mmap_mode="r", allow_pickle=False)
        if states.shape == (paths, 26, len(initial_state)) and spots.shape == (paths, 26):
            return {"name": name, "paths": paths, "reused": True}
        raise ValueError(f"existing {name} population has incompatible geometry")
    state_tmp = state_path.with_name(f".{state_path.name}.tmp")
    spot_tmp = spot_path.with_name(f".{spot_path.name}.tmp")
    output_directory.mkdir(parents=True, exist_ok=True)
    state_map = np.lib.format.open_memmap(
        state_tmp, mode="w+", dtype=np.float32, shape=(paths, 26, len(initial_state))
    )
    spot_map = np.lib.format.open_memmap(
        spot_tmp, mode="w+", dtype=np.float32, shape=(paths, 26)
    )
    generator = torch.Generator(device=initial_state.device).manual_seed(seed)
    equilibrium = model.equilibrium_mean.detach()
    started = time.monotonic()
    for first in range(0, paths, batch_size):
        last = min(first + batch_size, paths)
        count = last - first
        if count % 2:
            raise RuntimeError("final generation batch is not antithetic")
        states = initial_state[None].expand(count, -1).clone()
        path_spot = torch.ones(count, dtype=initial_state.dtype, device=initial_state.device)
        state_map[first:last, 0] = states.cpu().numpy()
        spot_map[first:last, 0] = 1.0
        for day in range(25):
            surface = _decode(model, states)
            half_state = torch.randn(
                (count // 2, len(initial_state)),
                dtype=initial_state.dtype,
                device=initial_state.device,
                generator=generator,
            )
            half_spot = torch.randn(
                count // 2,
                dtype=initial_state.dtype,
                device=initial_state.device,
                generator=generator,
            )
            state_normals = torch.cat((half_state, -half_state), dim=0)
            independent_spot = torch.cat((half_spot, -half_spot), dim=0)
            states = equilibrium + (states - equilibrium) @ daily_transition
            states = states + state_normals @ daily_factor.T
            gaussian_score = (
                state_normals @ copula_loading
                + copula_orthogonal_std * independent_spot
            ) / copula_total_std
            path_spot = path_spot * front_week_sanos_inverse_cdf(surface, gaussian_score)
            state_map[first:last, day + 1] = states.cpu().numpy()
            spot_map[first:last, day + 1] = path_spot.cpu().numpy()
        if first == 0 or last == paths or last % (10 * batch_size) == 0:
            rate = last / max(time.monotonic() - started, 1.0e-12)
            print(f"generated {name}: {last}/{paths} paths at {rate:.1f}/s", flush=True)
    state_map.flush()
    spot_map.flush()
    del state_map, spot_map
    os.replace(state_tmp, state_path)
    os.replace(spot_tmp, spot_path)
    return {"name": name, "paths": paths, "seed": seed, "reused": False}


def _load_population(directory: Path, name: str) -> PathPopulation:
    state_path, spot_path = _population_paths(directory, name)
    states = np.load(state_path, mmap_mode="r", allow_pickle=False)
    spots = np.load(spot_path, mmap_mode="r", allow_pickle=False)
    if states.ndim != 3 or states.shape[1:] != (26, 200) or spots.shape != states.shape[:2]:
        raise ValueError(f"invalid {name} population geometry")
    return PathPopulation(states=states, spots=spots)


def _population_batch(
    population: PathPopulation,
    indices: np.ndarray,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    states = torch.as_tensor(np.asarray(population.states[indices]), device=device)
    spots = torch.as_tensor(np.asarray(population.spots[indices]), device=device)
    return states, spots


def _antithetic_validation_chunks(
    paths: int,
    *,
    batch_size: int,
    block_size: int,
) -> list[np.ndarray]:
    """Return chunks that keep every generated antithetic pair together."""

    if paths % block_size or block_size % batch_size or batch_size % 2:
        raise ValueError("validation paths are incompatible with antithetic blocks")
    chunks: list[np.ndarray] = []
    pairs_per_chunk = batch_size // 2
    half_block = block_size // 2
    for block in range(0, paths, block_size):
        for pair_first in range(0, half_block, pairs_per_chunk):
            left = np.arange(block + pair_first, block + pair_first + pairs_per_chunk)
            right = left + half_block
            chunks.append(np.concatenate((left, right)))
    return chunks


def _policy_wealth(
    *,
    model: Any,
    policy: DynamicOptionPolicy,
    states: torch.Tensor,
    spots: torch.Tensor,
    claim_sign: int,
    spot_cost: float,
    option_cost: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate terminal wealth and policy stability diagnostics."""

    paths = len(states)
    hidden = policy.initial_hidden(paths)
    pnl = torch.zeros(paths, dtype=states.dtype, device=states.device)
    surfaces = _decode_path_surfaces(model, states[:, :25])
    raw_max = torch.zeros((), dtype=states.dtype, device=states.device)
    option_return_max = torch.zeros((), dtype=states.dtype, device=states.device)
    option_price_min = states.new_tensor(torch.inf)
    action_l1 = torch.zeros((), dtype=torch.float64, device=states.device)
    cap_count = torch.zeros((), dtype=torch.float64, device=states.device)
    filtered_option_count = torch.zeros((), dtype=torch.float64, device=states.device)
    for day in range(25):
        decision = policy(
            states[:, day],
            torch.log(spots[:, day]),
            day / 25.0,
            hidden,
        )
        hidden = decision.hidden
        raw_l1 = torch.sum(torch.abs(decision.raw_quantities), dim=1)
        cap_count += torch.sum((raw_l1 >= policy.config.maximum_action_l1).double())
        raw_max = torch.maximum(raw_max, torch.amax(torch.abs(decision.raw_quantities)))
        surface = _surface_day(surfaces, day)
        option_return, option_details = interpolated_option_returns(
            model,
            surface,
            issue_day=day,
            remaining_days=25 - day,
            path_spots=spots,
            nms=decision.nms,
            maturity_fraction=decision.maturity_fraction,
            put_weight=decision.put_weight,
            price_floor=policy.config.minimum_option_price,
        )
        eligible = option_details["eligible"]
        effective_option_quantity = decision.quantities[:, 1:] * eligible
        filtered_option_count += torch.sum((~eligible).double())
        effective_action_l1 = torch.abs(decision.quantities[:, 0]) + torch.sum(
            torch.abs(effective_option_quantity), dim=1
        )
        action_l1 += torch.sum(effective_action_l1.double())
        option_return_max = torch.maximum(
            option_return_max,
            torch.amax(torch.abs(option_return.detach())),
        )
        option_price_min = torch.minimum(
            option_price_min,
            torch.amin(option_details["price"].detach()),
        )
        spot_return = spots[:, -1] / spots[:, day] - 1.0
        pnl = pnl + decision.quantities[:, 0] * spot_return
        pnl = pnl + torch.sum(effective_option_quantity * option_return, dim=1)
        pnl = pnl - spot_cost * torch.abs(decision.quantities[:, 0])
        pnl = pnl - option_cost * torch.sum(torch.abs(effective_option_quantity), dim=1)
    wealth = pnl + claim_sign * five_day_straddle_claim(spots)
    if not bool(torch.all(torch.isfinite(wealth))):
        raise FloatingPointError("policy wealth is non-finite")
    observations = paths * 25
    return wealth, {
        "mean_action_l1": action_l1 / observations,
        "cap_active_fraction": cap_count / observations,
        "maximum_absolute_raw_action": raw_max,
        "maximum_absolute_option_return": option_return_max,
        "minimum_option_price": option_price_min,
        "filtered_option_fraction": filtered_option_count
        / (observations * policy.config.option_slots),
        "wealth_mean": torch.mean(wealth.detach().double()),
        "wealth_std": torch.std(wealth.detach().double(), correction=0),
        "wealth_minimum": torch.amin(wealth.detach().double()),
        "wealth_maximum": torch.amax(wealth.detach().double()),
    }


def _parameter_gradient(policy: DynamicOptionPolicy) -> tuple[float, float, int]:
    square = 0.0
    maximum = 0.0
    count = 0
    for parameter in policy.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().double()
        if not bool(torch.all(torch.isfinite(gradient))):
            raise FloatingPointError("policy gradient is non-finite")
        square += float(torch.sum(gradient.square()))
        maximum = max(maximum, float(torch.amax(torch.abs(gradient))))
        count += gradient.numel()
    return math.sqrt(square), maximum, count


def _make_optimizer(
    policy: DynamicOptionPolicy,
    cash_translation: torch.nn.Parameter | None,
    config: DynamicHedgeTrainConfig,
) -> torch.optim.Optimizer:
    parameters = list(policy.parameters())
    if cash_translation is not None:
        parameters.append(cash_translation)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=0.0)
    return Shampoo(
        parameters,
        lr=config.learning_rate,
        damping=config.shampoo_damping,
        precondition_frequency=config.shampoo_precondition_frequency,
        maximum_parameter_step=config.maximum_parameter_step,
    )


def _epoch_learning_rate(config: DynamicHedgeTrainConfig, epoch: int) -> float:
    """Cosine decay from the configured initial to minimum learning rate."""

    progress = epoch / max(config.maximum_epochs - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return config.minimum_learning_rate + (
        config.learning_rate - config.minimum_learning_rate
    ) * cosine


def _write_action_audit(
    *,
    model: Any,
    policy: DynamicOptionPolicy,
    population: PathPopulation,
    output: Path,
    batch_size: int,
) -> None:
    """Persist frozen test actions as spot plus five fully specified option slots."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    values = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(len(population), 25, 1 + 4 * policy.config.option_slots),
    )
    policy.eval()
    with torch.no_grad():
        for first in range(0, len(population), batch_size):
            last = min(first + batch_size, len(population))
            indices = np.arange(first, last)
            states, spots = _population_batch(population, indices, device=policy.device)
            hidden = policy.initial_hidden(last - first)
            surfaces = _decode_path_surfaces(model, states[:, :25])
            for day in range(25):
                decision = policy(
                    states[:, day],
                    torch.log(spots[:, day]),
                    day / 25.0,
                    hidden,
                )
                hidden = decision.hidden
                remaining = 25 - day
                maturity_days = (
                    torch.ones_like(decision.maturity_fraction)
                    if remaining == 1
                    else 1.0 + (remaining - 1.0) * decision.maturity_fraction
                )
                _, option_details = interpolated_option_returns(
                    model,
                    _surface_day(surfaces, day),
                    issue_day=day,
                    remaining_days=remaining,
                    path_spots=spots,
                    nms=decision.nms,
                    maturity_fraction=decision.maturity_fraction,
                    put_weight=decision.put_weight,
                    price_floor=policy.config.minimum_option_price,
                )
                effective_option_quantity = (
                    decision.quantities[:, 1:] * option_details["eligible"]
                )
                slot_values = torch.stack(
                    (
                        effective_option_quantity,
                        decision.nms,
                        maturity_days,
                        decision.put_weight,
                    ),
                    dim=2,
                ).reshape(last - first, -1)
                row = torch.cat((decision.quantities[:, :1], slot_values), dim=1)
                values[first:last, day] = row.cpu().numpy()
    values.flush()
    del values
    os.replace(temporary, output)


def _incremental_hedge_summary(claim_path: Path, empty_path: Path) -> dict[str, Any]:
    """Summarize the signed union of claim and empty frozen test books."""

    claim = np.load(claim_path, mmap_mode="r", allow_pickle=False)
    empty = np.load(empty_path, mmap_mode="r", allow_pickle=False)
    if claim.shape != empty.shape or claim.ndim != 3 or claim.shape[2] != 21:
        raise ValueError("action audit files have incompatible geometry")
    spot_sum = np.zeros(25, dtype=np.float64)
    spot_square = np.zeros(25, dtype=np.float64)
    spot_max = np.zeros(25, dtype=np.float64)
    option_gross_sum = np.zeros(25, dtype=np.float64)
    paths = len(claim)
    for first in range(0, paths, 2_000):
        last = min(first + 2_000, paths)
        spot = np.asarray(claim[first:last, :, 0], dtype=np.float64)
        spot -= np.asarray(empty[first:last, :, 0], dtype=np.float64)
        spot_sum += np.sum(spot, axis=0)
        spot_square += np.sum(np.square(spot), axis=0)
        spot_max = np.maximum(spot_max, np.max(np.abs(spot), axis=0))
        claim_quantities = np.asarray(claim[first:last, :, 1::4], dtype=np.float64)
        empty_quantities = np.asarray(empty[first:last, :, 1::4], dtype=np.float64)
        # Different selectors generally choose different contracts.  The exact
        # excess option hedge is therefore their signed union, not slotwise subtraction.
        option_gross_sum += np.sum(
            np.abs(claim_quantities) + np.abs(empty_quantities), axis=(0, 2)
        )
    return {
        "representation": "claim policy book plus the negative empty-policy book",
        "claim_action_file": str(claim_path),
        "empty_action_file": str(empty_path),
        "spot_excess_mean_by_day": (spot_sum / paths).tolist(),
        "spot_excess_rms_by_day": np.sqrt(spot_square / paths).tolist(),
        "spot_excess_maximum_absolute_by_day": spot_max.tolist(),
        "option_union_mean_gross_quantity_by_day": (option_gross_sum / paths).tolist(),
    }


def _validation_gradient(
    *,
    model: Any,
    policy: DynamicOptionPolicy,
    population: PathPopulation,
    utility: str,
    risk_aversion: float,
    claim_sign: int,
    cash_translation: torch.nn.Parameter | None,
    spot_cost: float,
    option_cost: float,
    batch_size: int,
    antithetic_block_size: int,
    reoptimize_test_translation: bool = False,
) -> dict[str, float]:
    """Compute a full validation gradient and a chunk-based noise scale."""

    policy.zero_grad(set_to_none=True)
    if cash_translation is not None:
        cash_translation.grad = None
    chunk_gradient_estimates: list[torch.Tensor] = []
    gradient_contributions: list[torch.Tensor] = []
    cap_sum = 0.0
    raw_max = 0.0
    option_return_max = 0.0
    option_price_min = math.inf
    filtered_option_sum = 0.0
    wealth_values: list[torch.Tensor] = []
    validation_chunks = _antithetic_validation_chunks(
        len(population),
        batch_size=batch_size,
        block_size=antithetic_block_size,
    )
    # Entropy needs the global empirical normalizer before its exact gradient pass.
    if utility == "entropy":
        with torch.no_grad():
            for indices in validation_chunks:
                states, spots = _population_batch(population, indices, device=policy.device)
                wealth, _ = _policy_wealth(
                    model=model,
                    policy=policy,
                    states=states,
                    spots=spots,
                    claim_sign=claim_sign,
                    spot_cost=spot_cost,
                    option_cost=option_cost,
                )
                wealth_values.append(wealth.cpu().double())
        all_wealth = torch.cat(wealth_values)
        log_normalizer = torch.logsumexp(-risk_aversion * all_wealth, dim=0)
        log_normalizer -= math.log(len(population))
        value = -float(log_normalizer) / risk_aversion
    else:
        value = 0.0

    for indices in validation_chunks:
        count = len(indices)
        states, spots = _population_batch(population, indices, device=policy.device)
        wealth, diagnostics = _policy_wealth(
            model=model,
            policy=policy,
            states=states,
            spots=spots,
            claim_sign=claim_sign,
            spot_cost=spot_cost,
            option_cost=option_cost,
        )
        policy.zero_grad(set_to_none=True)
        if cash_translation is not None:
            cash_translation.grad = None
        if utility == "entropy":
            global_weights = torch.exp(
                -risk_aversion * wealth.double() - log_normalizer.to(policy.device)
            )
            loss = -torch.sum(global_weights.detach() * wealth.double()) / len(population)
        else:
            assert cash_translation is not None
            batch_value = oce_value(
                wealth,
                utility=utility,
                risk_aversion=risk_aversion,
                cash_translation=cash_translation,
            )
            loss = -batch_value * (count / len(population))
            value += float(batch_value.detach()) * (count / len(population))
            wealth_values.append(wealth.detach().cpu().double())
        loss.backward()
        pieces = [
            parameter.grad.detach().reshape(-1).cpu().float()
            if parameter.grad is not None
            else torch.zeros(parameter.numel())
            for parameter in policy.parameters()
        ]
        if cash_translation is not None:
            pieces.append(
                cash_translation.grad.detach().reshape(-1).cpu().float()
                if cash_translation.grad is not None
                else torch.zeros(1)
            )
        # Convert each weighted chunk contribution to an estimate of the full mean
        # gradient so its dispersion estimates sampling noise across path chunks.
        contribution = torch.cat(pieces)
        gradient_contributions.append(contribution)
        chunk_gradient_estimates.append(
            contribution * (len(population) / count)
        )
        cap_sum += float(diagnostics["cap_active_fraction"].detach()) * count
        raw_max = max(
            raw_max,
            float(diagnostics["maximum_absolute_raw_action"].detach()),
        )
        option_return_max = max(
            option_return_max,
            float(diagnostics["maximum_absolute_option_return"].detach()),
        )
        option_price_min = min(
            option_price_min,
            float(diagnostics["minimum_option_price"].detach()),
        )
        filtered_option_sum += float(
            diagnostics["filtered_option_fraction"].detach()
        ) * count
    stacked = torch.stack(chunk_gradient_estimates).double()
    mean_gradient = torch.sum(torch.stack(gradient_contributions).double(), dim=0)
    gradient_norm = float(torch.linalg.vector_norm(mean_gradient))
    gradient_rms = gradient_norm / math.sqrt(len(mean_gradient))
    if len(stacked) > 1:
        se_norm = float(
            torch.sqrt(torch.sum(torch.var(stacked, dim=0, correction=1)) / len(stacked))
        )
    else:
        se_norm = math.inf
    # Restore the exact full-population mean gradient for optional diagnostics.
    offset = 0
    all_parameters = list(policy.parameters())
    if cash_translation is not None:
        all_parameters.append(cash_translation)
    for parameter in all_parameters:
        size = parameter.numel()
        parameter.grad = mean_gradient[offset : offset + size].reshape(parameter.shape).to(
            device=parameter.device, dtype=parameter.dtype
        )
        offset += size
    result = {
        "value": value,
        "gradient_l2_norm": gradient_norm,
        "gradient_rms": gradient_rms,
        "gradient_standard_error_l2": se_norm,
        "gradient_signal_to_noise": gradient_norm / max(se_norm, 1.0e-30),
        "cap_active_fraction": cap_sum / len(population),
        "maximum_absolute_raw_action": raw_max,
        "maximum_absolute_option_return": option_return_max,
        "minimum_option_price": option_price_min,
        "filtered_option_fraction": filtered_option_sum / len(population),
    }
    density_wealth = all_wealth if utility == "entropy" else torch.cat(wealth_values)
    density_cash = (
        None
        if utility == "entropy"
        else cash_translation.detach().cpu().double()
    )
    validation_density = marginal_utility_density(
        density_wealth,
        utility=utility,
        risk_aversion=risk_aversion,
        cash_translation=density_cash,
    )
    result["density_ess_fraction"] = float(
        len(density_wealth) / torch.sum(validation_density.double().square())
    )
    if utility == "power-exp" and reoptimize_test_translation:
        all_wealth = density_wealth
        lower = float(torch.amin(all_wealth)) - 20.0 / risk_aversion
        upper = float(torch.amax(all_wealth)) + 20.0 / risk_aversion
        for _ in range(100):
            middle = 0.5 * (lower + upper)
            residual = all_wealth - middle
            derivative = torch.where(
                residual >= 0.0,
                torch.exp(-risk_aversion * torch.clamp_min(residual, 0.0)),
                1.0 - risk_aversion * residual,
            )
            if float(torch.mean(derivative)) > 1.0:
                upper = middle
            else:
                lower = middle
        optimal_cash = all_wealth.new_tensor(0.5 * (lower + upper))
        test_value = oce_value(
            all_wealth,
            utility="power-exp",
            risk_aversion=risk_aversion,
            cash_translation=optimal_cash,
        )
        test_density = marginal_utility_density(
            all_wealth,
            utility="power-exp",
            risk_aversion=risk_aversion,
            cash_translation=optimal_cash,
        )
        result["training_cash_translation"] = float(cash_translation.detach())
        result["test_optimal_cash_translation"] = float(optimal_cash)
        result["value_at_training_cash_translation"] = value
        result["value"] = float(test_value)
        result["test_density_ess_fraction"] = float(
            len(all_wealth) / torch.sum(test_density.double().square())
        )
    return result


def _train_one(
    *,
    model: Any,
    env: Any,
    train: PathPopulation,
    validation: PathPopulation,
    test: PathPopulation,
    historical_state_mean: torch.Tensor,
    historical_state_scale: torch.Tensor,
    policy_config: DynamicOptionPolicyConfig,
    train_config: DynamicHedgeTrainConfig,
    output_directory: Path,
    utility: str,
    risk_aversion: float,
    claim_sign: int,
    spot_cost: float,
    option_cost: float,
    seed: int,
    resume: bool,
    warm_start_checkpoint: Path | None = None,
) -> dict[str, Any]:
    label = f"{utility}-lambda-{risk_aversion:g}-claim-{claim_sign:+d}"
    checkpoint = output_directory / "checkpoints" / f"{label}.pt"
    lock = FileLock(str(checkpoint) + ".lock")
    policy = DynamicOptionPolicy(
        env,
        config=policy_config,
        state_mean=historical_state_mean,
        state_scale=historical_state_scale,
    )
    cash_translation = (
        torch.nn.Parameter(torch.zeros((), dtype=env.fdtype, device=env.device))
        if utility == "power-exp"
        else None
    )
    optimizer = _make_optimizer(policy, cash_translation, train_config)
    generator = np.random.default_rng(seed)
    start_epoch = 0
    history: list[dict[str, Any]] = []
    close_count = 0
    with lock:
        if resume and checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved.get("schema") != "flex-dynamic-option-deep-hedging-v1":
                raise ValueError(f"invalid checkpoint {checkpoint}")
            if saved.get("policy_code_version") != policy_config.version.unique_id48:
                print(f"stale policy checkpoint ignored: {checkpoint}", flush=True)
            elif saved.get("policy_user_version") != policy_config.user_version:
                print(f"user-version mismatch; checkpoint ignored: {checkpoint}", flush=True)
            else:
                policy.load_state_dict(saved["policy_state_dict"])
                optimizer.load_state_dict(saved["optimizer_state_dict"])
                if cash_translation is not None:
                    cash_translation.data.copy_(saved["cash_translation"].to(env.device))
                generator.bit_generator.state = saved["numpy_generator_state"]
                start_epoch = (
                    train_config.maximum_epochs
                    if saved.get("phase") == "complete"
                    else int(saved["epoch"])
                )
                history = list(saved.get("history", []))
        elif warm_start_checkpoint is not None:
            saved = torch.load(
                warm_start_checkpoint, map_location="cpu", weights_only=False
            )
            if (
                saved.get("schema") != "flex-dynamic-option-deep-hedging-v1"
                or saved.get("phase") != "complete"
                or saved.get("policy_code_version")
                != policy_config.version.unique_id48
                or saved.get("policy_user_version") != policy_config.user_version
            ):
                raise ValueError(f"invalid warm-start checkpoint {warm_start_checkpoint}")
            policy.load_state_dict(saved["policy_state_dict"])
            if cash_translation is not None:
                cash_translation.data.copy_(saved["cash_translation"].to(env.device))
            print(f"warm started {label} from {warm_start_checkpoint}", flush=True)
        validation_values = [
            float(record["validation"]["value"])
            for record in history
            if isinstance(record.get("validation"), Mapping)
        ]
        best_validation_value = max(validation_values, default=-math.inf)
        for epoch in range(start_epoch, train_config.maximum_epochs):
            learning_rate = _epoch_learning_rate(train_config, epoch)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            permutation = generator.permutation(len(train))
            epoch_started = time.monotonic()
            for first in range(0, len(train), train_config.batch_size):
                indices = permutation[first : first + train_config.batch_size]
                states, spots = _population_batch(train, indices, device=env.device)
                optimizer.zero_grad(set_to_none=True)
                wealth, diagnostics = _policy_wealth(
                    model=model,
                    policy=policy,
                    states=states,
                    spots=spots,
                    claim_sign=claim_sign,
                    spot_cost=spot_cost,
                    option_cost=option_cost,
                )
                value = oce_value(
                    wealth,
                    utility=utility,
                    risk_aversion=risk_aversion,
                    cash_translation=cash_translation,
                )
                loss = -value
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"{label}: non-finite training loss")
                loss.backward()
                gradient_norm, _, _ = _parameter_gradient(policy)
                if cash_translation is not None and cash_translation.grad is not None:
                    cash_gradient = float(torch.abs(cash_translation.grad.detach()))
                    if not math.isfinite(cash_gradient):
                        raise FloatingPointError(f"{label}: non-finite cash gradient")
                    gradient_norm = math.hypot(gradient_norm, cash_gradient)
                if gradient_norm > train_config.maximum_gradient_norm:
                    raise FloatingPointError(f"{label}: gradient blow-up {gradient_norm:.6g}")
                with torch.no_grad():
                    density = marginal_utility_density(
                        wealth,
                        utility=utility,
                        risk_aversion=risk_aversion,
                        cash_translation=cash_translation,
                    )
                    ess_fraction = float(
                        len(wealth) / torch.sum(density.double().square())
                    )
                if ess_fraction < train_config.minimum_ess_fraction:
                    raise FloatingPointError(
                        f"{label}: marginal-utility density collapsed; ESS={ess_fraction:.3e}"
                    )
                clipped_parameters = list(policy.parameters())
                if cash_translation is not None:
                    clipped_parameters.append(cash_translation)
                torch.nn.utils.clip_grad_norm_(
                    clipped_parameters, train_config.gradient_clip_norm
                )
                optimizer.step()
                if cash_translation is not None and not bool(torch.isfinite(cash_translation)):
                    raise FloatingPointError(f"{label}: non-finite cash translation")
                cap_fraction = float(diagnostics["cap_active_fraction"].detach())
                raw_max = float(
                    diagnostics["maximum_absolute_raw_action"].detach()
                )
                option_return_max = float(
                    diagnostics["maximum_absolute_option_return"].detach()
                )
                option_price_min = float(
                    diagnostics["minimum_option_price"].detach()
                )
                filtered_option_fraction = float(
                    diagnostics["filtered_option_fraction"].detach()
                )
                if raw_max > train_config.maximum_raw_action:
                    raise FloatingPointError(f"{label}: raw action blow-up {raw_max:.6g}")
                if option_return_max > train_config.maximum_option_return:
                    raise FloatingPointError(
                        f"{label}: normalized option-return blow-up "
                        f"{option_return_max:.6g} at minimum premium {option_price_min:.6g}"
                    )
                if cap_fraction > train_config.maximum_cap_fraction:
                    raise FloatingPointError(f"{label}: cap saturation {cap_fraction:.2%}")
                if not all(bool(torch.all(torch.isfinite(p))) for p in policy.parameters()):
                    raise FloatingPointError(f"{label}: non-finite policy parameters")
                batch_number = first // train_config.batch_size + 1
                if batch_number % 50 == 0:
                    print(
                        json.dumps(
                            {
                                "fit": label,
                                "epoch": epoch + 1,
                                "batch": batch_number,
                                "training_value": float(value.detach()),
                                "gradient_l2": gradient_norm,
                                "ess_fraction": ess_fraction,
                                "cap_active_fraction": cap_fraction,
                                "maximum_absolute_raw_action": raw_max,
                                "maximum_absolute_option_return": option_return_max,
                                "minimum_option_price": option_price_min,
                                "filtered_option_fraction": filtered_option_fraction,
                            }
                        ),
                        flush=True,
                    )
            record: dict[str, Any] = {
                "epoch": epoch + 1,
                "training_value_last_batch": float(value.detach()),
                "training_gradient_l2_last_batch": gradient_norm,
                "training_ess_fraction_last_batch": ess_fraction,
                "training_cap_active_fraction_last_batch": cap_fraction,
                "training_maximum_absolute_raw_action_last_batch": raw_max,
                "training_maximum_absolute_option_return_last_batch": option_return_max,
                "training_minimum_option_price_last_batch": option_price_min,
                "training_filtered_option_fraction_last_batch": filtered_option_fraction,
                "learning_rate": learning_rate,
                "seconds": time.monotonic() - epoch_started,
            }
            if (epoch + 1) % train_config.validation_every == 0:
                validation_result = _validation_gradient(
                    model=model,
                    policy=policy,
                    population=validation,
                    utility=utility,
                    risk_aversion=risk_aversion,
                    claim_sign=claim_sign,
                    cash_translation=cash_translation,
                    spot_cost=spot_cost,
                    option_cost=option_cost,
                    batch_size=train_config.validation_batch_size,
                    antithetic_block_size=train_config.antithetic_block_size,
                )
                record["validation"] = validation_result
                if (
                    validation_result["density_ess_fraction"]
                    < train_config.minimum_validation_ess_fraction
                ):
                    raise FloatingPointError(
                        f"{label}: validation density collapsed; ESS fraction="
                        f"{validation_result['density_ess_fraction']:.6g}"
                    )
                if (
                    validation_result["maximum_absolute_option_return"]
                    > train_config.maximum_option_return
                    or validation_result["cap_active_fraction"]
                    > train_config.maximum_cap_fraction
                ):
                    raise FloatingPointError(
                        f"{label}: validation action/return blow-up; "
                        f"max option return="
                        f"{validation_result['maximum_absolute_option_return']:.6g}, "
                        f"cap fraction={validation_result['cap_active_fraction']:.3%}"
                    )
                if (
                    math.isfinite(best_validation_value)
                    and validation_result["value"]
                    < best_validation_value - train_config.maximum_validation_value_drop
                ):
                    raise FloatingPointError(
                        f"{label}: validation OCE blow-up; value="
                        f"{validation_result['value']:.6g}, best={best_validation_value:.6g}"
                    )
                best_validation_value = max(
                    best_validation_value, validation_result["value"]
                )
                statistically_zero = (
                    validation_result["gradient_signal_to_noise"]
                    <= train_config.validation_gradient_z
                    and validation_result["gradient_rms"]
                    <= train_config.validation_gradient_rms_tolerance
                )
                close_count = close_count + 1 if statistically_zero else 0
            history.append(record)
            print(json.dumps({"fit": label, **record}), flush=True)
            _atomic_torch_save(
                {
                    "schema": "flex-dynamic-option-deep-hedging-v1",
                    "phase": "training",
                    "policy_code_version": policy_config.version.unique_id48,
                    "policy_user_version": policy_config.user_version,
                    "policy_config": asdict(policy_config),
                    "train_config": asdict(train_config),
                    "epoch": epoch + 1,
                    "policy_state_dict": copy.deepcopy(policy.state_dict()),
                    "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                    "cash_translation": None if cash_translation is None else cash_translation.detach().cpu(),
                    "numpy_generator_state": generator.bit_generator.state,
                    "history": history,
                },
                checkpoint,
            )
            if close_count >= train_config.validation_patience:
                break
        test_result = _validation_gradient(
            model=model,
            policy=policy,
            population=test,
            utility=utility,
            risk_aversion=risk_aversion,
            claim_sign=claim_sign,
            cash_translation=cash_translation,
            spot_cost=spot_cost,
            option_cost=option_cost,
            batch_size=train_config.validation_batch_size,
            antithetic_block_size=train_config.antithetic_block_size,
            reoptimize_test_translation=True,
        )
        _atomic_torch_save(
            {
                "schema": "flex-dynamic-option-deep-hedging-v1",
                "phase": "complete",
                "policy_code_version": policy_config.version.unique_id48,
                "policy_user_version": policy_config.user_version,
                "policy_config": asdict(policy_config),
                "train_config": asdict(train_config),
                "epoch": len(history),
                "policy_state_dict": copy.deepcopy(policy.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "cash_translation": None if cash_translation is None else cash_translation.detach().cpu(),
                "numpy_generator_state": generator.bit_generator.state,
                "history": history,
                "test": test_result,
            },
            checkpoint,
        )
        action_path = output_directory / "test-actions" / f"{label}.npy"
        _write_action_audit(
            model=model,
            policy=policy,
            population=test,
            output=action_path,
            batch_size=train_config.validation_batch_size,
        )
    return {
        "label": label,
        "test": test_result,
        "epochs": len(history),
        "checkpoint": str(checkpoint),
        "test_actions": str(action_path),
    }


def run(arguments: argparse.Namespace) -> None:
    """Generate paths, fit all 18 policies, and write excess price results."""

    if arguments.paths_per_population <= 0 or arguments.paths_per_population % 2:
        raise ValueError("paths per population must be positive and even")
    env, model = load_presentation_model(arguments.checkpoint, device=arguments.device)
    dates, historical_states, historical_spots = _load_history(arguments.target_history)
    initial_state = historical_states[-1].to(dtype=env.fdtype, device=env.device)
    daily_transition, daily_factor, transition_diagnostics = exact_daily_transition(model)
    copula_fit, copula_diagnostics = fit_front_spot_copula_from_history(
        model=model,
        states=historical_states.double().numpy(),
        spots=historical_spots.double().numpy(),
        dates=dates,
        daily_transition=daily_transition.cpu().double().numpy(),
        daily_factor=daily_factor.cpu().double().numpy(),
        fit_before=arguments.copula_fit_before,
    )
    copula_loading = torch.as_tensor(copula_fit.state_loading, dtype=env.fdtype, device=env.device)
    output = arguments.output_directory
    populations_directory = output / "populations"
    metadata = {
        "schema": "flex-dynamic-option-paths-v1",
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_sha256": checkpoint_digest(arguments.checkpoint),
        "initial_date": dates[-1],
        "num_days": 25,
        "paths_per_population": arguments.paths_per_population,
        "action_audit_columns": [
            "spot_quantity",
            *(
                f"slot_{slot}_{field}"
                for slot in range(5)
                for field in ("quantity", "nms", "maturity_days", "is_put")
            ),
        ],
        "transition_diagnostics": transition_diagnostics,
        "copula_diagnostics": copula_diagnostics,
    }
    if arguments.stage in {"generate", "all"}:
        generated = []
        for offset, name in enumerate(("training", "validation", "test")):
            generated.append(
                _generate_population(
                    name=name,
                    output_directory=populations_directory,
                    model=model,
                    initial_state=initial_state,
                    daily_transition=daily_transition,
                    daily_factor=daily_factor,
                    copula_loading=copula_loading,
                    copula_orthogonal_std=copula_fit.orthogonal_std,
                    copula_total_std=copula_fit.total_std,
                    paths=arguments.paths_per_population,
                    batch_size=arguments.generation_batch_size,
                    seed=arguments.seed + offset,
                )
            )
        metadata["populations"] = generated
        write_json_atomic(metadata, output / "path-metadata.json")
    if arguments.stage == "generate":
        return

    train = _load_population(populations_directory, "training")
    validation = _load_population(populations_directory, "validation")
    test = _load_population(populations_directory, "test")
    from quantdata.torch.ffw import FeedForwardConfig

    policy_config = DynamicOptionPolicyConfig(
        label="dynamic-five-option-affine-deep",
        state_size=200,
        hidden_size=arguments.hidden_size,
        option_slots=5,
        maximum_action_l1=arguments.action_l1_cap,
        decision_network=FeedForwardConfig(
            activation="SiLU",
            num_depth=arguments.network_depth,
            num_width=arguments.network_width,
        ),
        memory_network=FeedForwardConfig(
            activation="SiLU", num_depth=1, num_width=arguments.network_width
        ),
    )
    train_config = DynamicHedgeTrainConfig(
        batch_size=arguments.batch_size,
        maximum_epochs=arguments.maximum_epochs,
        validation_every=arguments.validation_every,
        validation_batch_size=arguments.validation_batch_size,
        antithetic_block_size=arguments.antithetic_block_size,
        learning_rate=arguments.learning_rate,
        optimizer=arguments.optimizer,
    )
    results: list[dict[str, Any]] = []
    historical = historical_states.to(dtype=env.fdtype, device=env.device)
    historical_state_mean = torch.mean(historical, dim=0)
    historical_state_scale = torch.std(historical, dim=0, correction=0)
    historical_state_scale = torch.where(
        historical_state_scale > 1.0e-5,
        historical_state_scale,
        torch.ones_like(historical_state_scale),
    )
    for utility in ("entropy", "power-exp"):
        for risk_aversion in (0.1, 1.0, 10.0):
            for claim_sign in (0, 1, -1):
                empty_checkpoint = (
                    output
                    / "checkpoints"
                    / f"{utility}-lambda-{risk_aversion:g}-claim-+0.pt"
                )
                results.append(
                    _train_one(
                        model=model,
                        env=env,
                        train=train,
                        validation=validation,
                        test=test,
                        historical_state_mean=historical_state_mean,
                        historical_state_scale=historical_state_scale,
                        policy_config=policy_config,
                        train_config=train_config,
                        output_directory=output,
                        utility=utility,
                        risk_aversion=risk_aversion,
                        claim_sign=claim_sign,
                        spot_cost=arguments.spot_cost_bps / 1.0e4,
                        option_cost=arguments.option_cost_bps / 1.0e4,
                        seed=arguments.seed + len(results) + 100,
                        resume=arguments.resume,
                        warm_start_checkpoint=(
                            empty_checkpoint if claim_sign != 0 else None
                        ),
                    )
                )
    indexed = {(result["label"]): result for result in results}
    prices: list[dict[str, Any]] = []
    for utility in ("entropy", "power-exp"):
        for risk_aversion in (0.1, 1.0, 10.0):
            empty = indexed[f"{utility}-lambda-{risk_aversion:g}-claim-+0"]["test"]["value"]
            long_value = indexed[f"{utility}-lambda-{risk_aversion:g}-claim-+1"]["test"]["value"]
            short_value = indexed[f"{utility}-lambda-{risk_aversion:g}-claim--1"]["test"]["value"]
            prices.append(
                {
                    "utility": utility,
                    "risk_aversion": risk_aversion,
                    "empty_value": empty,
                    "long_excess_value": long_value - empty,
                    "short_ask_value": empty - short_value,
                    "long_excess_hedge": _incremental_hedge_summary(
                        Path(indexed[f"{utility}-lambda-{risk_aversion:g}-claim-+1"]["test_actions"]),
                        Path(indexed[f"{utility}-lambda-{risk_aversion:g}-claim-+0"]["test_actions"]),
                    ),
                    "short_excess_hedge": _incremental_hedge_summary(
                        Path(indexed[f"{utility}-lambda-{risk_aversion:g}-claim--1"]["test_actions"]),
                        Path(indexed[f"{utility}-lambda-{risk_aversion:g}-claim-+0"]["test_actions"]),
                    ),
                }
            )
    write_json_atomic(
        {"metadata": metadata, "fits": results, "prices": prices},
        output / "results.json",
    )


if __name__ == "__main__":
    run(_arguments())
