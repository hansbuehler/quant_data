"""Exact local-support masks for cost-adjusted option directions.

For direction ``d`` equal to ``+1`` (long) or ``-1`` (short), the empirical
candidate function is

``arb_d(z, omega) = 1``

when every one of the ``k`` nearest training observations has
``d * dH_omega - c_omega >= -e_omega`` and at least one has a gain strictly
above ``e_omega``.  The tolerance ``e`` only absorbs floating-point error;
the economic cost ``c`` is subtracted explicitly.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version
from torch import Tensor

from .unconditional_option_test import DEFAULT_ROUNDOFF_MULTIPLIER

LONG_DIRECTION = 0
SHORT_DIRECTION = 1
DIRECTIONS = ("long_dH_minus_cost", "short_minus_dH_minus_cost")


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class LocalArbitrageConfig:
    """Logical and batching configuration for the exact local mask."""

    neighbors: int = 10_240
    query_batch_size: int = 2_048
    instrument_batch_size: int = 10
    roundoff_multiplier: float = DEFAULT_ROUNDOFF_MULTIPLIER
    state_scale_floor: float = 1.0e-6

    def __post_init__(self) -> None:
        for name in ("neighbors", "query_batch_size", "instrument_batch_size"):
            value = getattr(self, name)
            verify_inp(
                isinstance(value, int) and value > 0,
                lambda name=name, value=value: (
                    f"'{name}' must be a positive integer, got {value!r}"
                ),
            )
        verify_inp(
            math.isfinite(self.state_scale_floor) and self.state_scale_floor > 0.0
            and math.isfinite(self.roundoff_multiplier)
            and self.roundoff_multiplier > 0.0,
            lambda: (
                "state scale floor and roundoff multiplier must be finite and "
                f"positive, got {self.state_scale_floor}/{self.roundoff_multiplier}"
            ),
        )


@dataclass(frozen=True, kw_only=True)
class LocalArbitrageProgress:
    """Cumulative disjoint candidate-state counts during mask valuation."""

    split: str
    processed_states: int
    total_states: int
    long_only_candidate_states: int
    short_only_candidate_states: int
    both_direction_candidate_states: int


def cost_adjusted_direction_candidates(
    neighbor_returns: Tensor,
    query_cost: Tensor,
    neighbor_roundoff: Tensor,
) -> Tensor:
    """Return long/short candidate flags with shape ``(B, 2, J)``."""

    verify_inp(
        neighbor_returns.shape == neighbor_roundoff.shape
        and query_cost.shape
        == (neighbor_returns.shape[0], neighbor_returns.shape[2]),
        lambda: (
            "expected neighbour return/roundoff shape (B, k, J) and query-cost "
            f"shape (B, J), got {neighbor_returns.shape}/"
            f"{neighbor_roundoff.shape}/{query_cost.shape}"
        ),
    )
    verify_inp(
        neighbor_returns.ndim == 3,
        lambda: (
            "neighbour inputs must have shape (batch, neighbours, instruments), "
            f"got {neighbor_returns.shape}"
        ),
    )
    cost = query_cost[:, None, :]
    long_gain = neighbor_returns - cost
    short_gain = -neighbor_returns - cost
    long = (~(long_gain < -neighbor_roundoff).any(dim=1)) & (
        long_gain > neighbor_roundoff
    ).any(dim=1)
    short = (~(short_gain < -neighbor_roundoff).any(dim=1)) & (
        short_gain > neighbor_roundoff
    ).any(dim=1)
    return torch.stack((long, short), dim=1)


def _tensor_list(value: Tensor) -> list[Any]:
    return value.detach().cpu().tolist()


@torch.inference_mode()
def valuate_local_arbitrage_mask(
    *,
    reference_state: Tensor,
    reference_returns: Tensor,
    reference_roundoff: Tensor,
    query_state: Tensor,
    query_returns: Tensor,
    query_cost: Tensor,
    query_roundoff: Tensor,
    config: LocalArbitrageConfig,
    split: str,
    leave_one_out: bool,
    candidate_sink: Callable[[int, int, np.ndarray], None] | None = None,
    progress: Callable[[LocalArbitrageProgress], None] | None = None,
    progress_interval_seconds: float = 30.0,
) -> dict[str, Any]:
    """Evaluate the training-fitted exact kNN ``arb(z; omega)`` function."""

    verify_inp(isinstance(split, str) and split != "", "split must be nonempty")
    verify_inp(
        math.isfinite(progress_interval_seconds) and progress_interval_seconds > 0.0,
        lambda: (
            "progress_interval_seconds must be finite and positive, got "
            f"{progress_interval_seconds}"
        ),
    )
    verify_inp(
        reference_state.ndim == query_state.ndim == 2
        and reference_state.shape[1] == query_state.shape[1],
        lambda: (
            f"state tensors must be compatible matrices, got "
            f"{reference_state.shape}/{query_state.shape}"
        ),
    )
    reference_values = (reference_returns, reference_roundoff)
    query_values = (query_returns, query_cost, query_roundoff)
    verify_inp(
        all(value.shape == reference_returns.shape for value in reference_values)
        and all(value.shape == query_returns.shape for value in query_values),
        "return, cost, and roundoff shapes must agree within each split",
    )
    verify_inp(
        reference_returns.ndim == query_returns.ndim == 2
        and reference_returns.shape[1] == query_returns.shape[1]
        and reference_state.shape[0] == reference_returns.shape[0]
        and query_state.shape[0] == query_returns.shape[0],
        "state/return observation or instrument axes do not agree",
    )
    verify_inp(
        config.neighbors < reference_state.shape[0]
        if leave_one_out
        else config.neighbors <= reference_state.shape[0],
        lambda: (
            f"not enough reference observations for {config.neighbors} neighbours"
        ),
    )
    tensors = (
        reference_state,
        reference_returns,
        reference_roundoff,
        query_state,
        query_returns,
        query_cost,
        query_roundoff,
    )
    device = reference_state.device
    if any(value.device != device for value in tensors):
        raise ValueError("all local-arbitrage tensors must be on the same device")
    if any(not torch.isfinite(value).all().item() for value in tensors):
        raise ValueError("local-arbitrage tensors contain non-finite values")
    if not torch.all(query_cost >= 0.0).item():
        raise ValueError("local-arbitrage costs must be nonnegative")
    if not torch.all(reference_roundoff >= 0.0).item() or not torch.all(
        query_roundoff >= 0.0
    ).item():
        raise ValueError("local-arbitrage roundoff bands must be nonnegative")

    queries = query_state.shape[0]
    instruments = reference_returns.shape[1]
    candidate_count = torch.zeros(2, instruments, dtype=torch.int64, device=device)
    counter_count = torch.zeros_like(candidate_count)
    exclusive_states = torch.zeros(3, dtype=torch.int64, device=device)
    reference_norm = reference_state.square().sum(dim=1)
    radius_sample = np.empty(min(1_000_000, queries), dtype=np.float32)
    radius_sample_count = 0
    radius_sum = 0.0
    radius_minimum = math.inf
    radius_maximum = -math.inf
    started = time.monotonic()
    last_progress = -math.inf

    for start in range(0, queries, config.query_batch_size):
        stop = min(start + config.query_batch_size, queries)
        batch_state = query_state[start:stop]
        distance = batch_state @ reference_state.T
        distance.mul_(-2.0).add_(reference_norm)
        distance.add_(batch_state.square().sum(dim=1, keepdim=True))
        if leave_one_out:
            row = torch.arange(stop - start, device=device)
            distance[row, torch.arange(start, stop, device=device)] = torch.inf
        nearest_distance, nearest = torch.topk(
            distance,
            config.neighbors,
            dim=1,
            largest=False,
            sorted=False,
        )
        del distance
        nearest_distance.clamp_min_(0.0)
        batch_radius = nearest_distance.max(dim=1).values.sqrt()
        radius_values = batch_radius.cpu().numpy()
        take = min(radius_values.size, radius_sample.size - radius_sample_count)
        if take > 0:
            radius_sample[
                radius_sample_count : radius_sample_count + take
            ] = radius_values[:take]
            radius_sample_count += take
        radius_sum += float(batch_radius.sum(dtype=torch.float64).item())
        radius_minimum = min(radius_minimum, float(batch_radius.min().item()))
        radius_maximum = max(radius_maximum, float(batch_radius.max().item()))
        del nearest_distance, batch_radius

        batch_candidates = torch.empty(
            stop - start, 2, instruments, dtype=torch.bool, device=device
        )
        batch_query_returns = query_returns[start:stop]
        batch_query_cost = query_cost[start:stop]
        batch_query_roundoff = query_roundoff[start:stop]
        for left in range(0, instruments, config.instrument_batch_size):
            right = min(left + config.instrument_batch_size, instruments)
            neighbor_returns = reference_returns[:, left:right][nearest]
            neighbor_roundoff = reference_roundoff[:, left:right][nearest]
            candidates = cost_adjusted_direction_candidates(
                neighbor_returns,
                batch_query_cost[:, left:right],
                neighbor_roundoff,
            )
            del neighbor_returns, neighbor_roundoff
            batch_candidates[..., left:right] = candidates
            candidate_count[:, left:right] += candidates.sum(dim=0)
            query_long_gain = (
                batch_query_returns[:, left:right]
                - batch_query_cost[:, left:right]
            )
            query_short_gain = (
                -batch_query_returns[:, left:right]
                - batch_query_cost[:, left:right]
            )
            query_gain = torch.stack((query_long_gain, query_short_gain), dim=1)
            counter_count[:, left:right] += (
                candidates
                & (query_gain < -batch_query_roundoff[:, None, left:right])
            ).sum(dim=0)

        long_state = batch_candidates[:, LONG_DIRECTION].any(dim=1)
        short_state = batch_candidates[:, SHORT_DIRECTION].any(dim=1)
        exclusive_states += torch.stack(
            (
                (long_state & ~short_state).sum(),
                (short_state & ~long_state).sum(),
                (long_state & short_state).sum(),
            )
        )
        if candidate_sink is not None:
            candidate_sink(start, stop, batch_candidates.cpu().numpy())
        now = time.monotonic()
        if progress is not None and (
            now - last_progress >= progress_interval_seconds or stop == queries
        ):
            counts = exclusive_states.cpu().tolist()
            progress(
                LocalArbitrageProgress(
                    split=split,
                    processed_states=stop,
                    total_states=queries,
                    long_only_candidate_states=counts[0],
                    short_only_candidate_states=counts[1],
                    both_direction_candidate_states=counts[2],
                )
            )
            last_progress = now

    total_direction_comparisons = queries * instruments
    counter_total = counter_count.sum(dim=1)
    candidate_total = candidate_count.sum(dim=1)
    sampled_radius = radius_sample[:radius_sample_count]
    return {
        "split": split,
        "queries": queries,
        "reference_observations": reference_state.shape[0],
        "instruments": instruments,
        "directions": list(DIRECTIONS),
        "leave_one_out": leave_one_out,
        "neighbors": config.neighbors,
        "candidate_count_by_direction_instrument": _tensor_list(candidate_count),
        "candidate_count_by_direction": _tensor_list(candidate_total),
        "candidate_fraction_by_direction": _tensor_list(
            candidate_total.to(torch.float64) / total_direction_comparisons
        ),
        "realized_counter_count_by_direction_instrument": _tensor_list(
            counter_count
        ),
        "realized_counter_count_by_direction": _tensor_list(counter_total),
        "realized_counter_rate_by_direction": _tensor_list(
            torch.where(
                candidate_total > 0,
                counter_total.to(torch.float64) / candidate_total,
                0.0,
            )
        ),
        "states_by_direction_exclusivity": {
            "long_only": int(exclusive_states[0].item()),
            "short_only": int(exclusive_states[1].item()),
            "both_directions": int(exclusive_states[2].item()),
            "neither_direction": int(queries - exclusive_states.sum().item()),
        },
        "neighbor_radius": {
            "minimum": radius_minimum,
            "mean": radius_sum / queries,
            "quantiles": {
                f"{probability:g}": float(np.quantile(sampled_radius, probability))
                for probability in (0.05, 0.5, 0.95)
            },
            "maximum": radius_maximum,
            "quantile_sample_states": radius_sample_count,
        },
        "candidate_definition": (
            "all k training-neighbour net gains d*dH-c are >= -roundoff "
            "and at least one is > roundoff"
        ),
        "elapsed_seconds": time.monotonic() - started,
    }


def pack_direction_mask(mask: np.ndarray) -> np.ndarray:
    """Pack the option axis of a bool ``(..., 2, J)`` candidate mask."""

    mask = np.asarray(mask)
    verify_inp(
        mask.dtype == np.bool_ and mask.ndim >= 2 and mask.shape[-2] == 2,
        lambda: f"mask must be bool with shape (..., 2, J), got {mask.shape}/{mask.dtype}",
    )
    return np.packbits(mask, axis=-1, bitorder="little")


def unpack_direction_mask(packed: Tensor, *, instruments: int) -> Tensor:
    """Unpack a uint8 ``(..., 2, ceil(J/8))`` mask on its Torch device."""

    verify_inp(
        packed.dtype == torch.uint8 and packed.ndim >= 2 and packed.shape[-2] == 2,
        lambda: (
            "packed mask must be uint8 with shape (..., 2, bytes), got "
            f"{packed.shape}/{packed.dtype}"
        ),
    )
    verify_inp(instruments > 0, lambda: f"'instruments' must be positive, got {instruments}")
    verify_inp(
        packed.shape[-1] == math.ceil(instruments / 8),
        lambda: (
            f"packed byte axis {packed.shape[-1]} does not encode {instruments} instruments"
        ),
    )
    bit = torch.arange(8, dtype=torch.uint8, device=packed.device)
    unpacked = ((packed[..., None] >> bit) & 1).to(torch.bool)
    return unpacked.flatten(start_dim=-2)[..., :instruments]


__all__ = [
    "DIRECTIONS",
    "LONG_DIRECTION",
    "LocalArbitrageConfig",
    "LocalArbitrageProgress",
    "SHORT_DIRECTION",
    "cost_adjusted_direction_candidates",
    "pack_direction_mask",
    "unpack_direction_mask",
    "valuate_local_arbitrage_mask",
]
