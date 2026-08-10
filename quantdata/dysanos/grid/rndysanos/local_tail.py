"""Exact adaptive-neighbour diagnostics for conditional option-return tails.

The estimator uses a uniform ``k``-nearest-neighbour kernel.  Its weights are
positive, so estimates of positive and negative return parts remain
nonnegative.  Training valuation is leave-one-out; validation queries always
use the training split as their reference sample.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.version import version
from torch import Tensor

from .torch_drift_change import UniverseGeometry, validate_universe
from .unconditional_option_test import (
    DEFAULT_ROUNDOFF_MULTIPLIER,
    delta_hedged_option_returns,
    delta_hedged_roundoff_band,
)

StateMode = Literal["raw", "instrument"]
LONG_DIRECTION = 0
SHORT_DIRECTION = 1
DIRECTIONS = ("long_dH", "short_minus_dH")


@dataclass(frozen=True, kw_only=True)
class LocalTailProgress:
    """Cumulative one-sided-support counts for a valuation in progress."""

    split: str
    processed_states: int
    total_states: int
    states_with_any_long_candidate: int
    states_with_any_short_candidate: int
    long_only_candidate_states: int
    short_only_candidate_states: int
    both_direction_candidate_states: int


@version("1.2")
@dataclass(frozen=True, kw_only=True)
class LocalTailConfig:
    """Configuration for an exact uniform adaptive-neighbour estimator."""

    state_mode: StateMode
    neighbors: int = 1_024
    instrument_candidate_neighbors: int = 8_192
    tail_ranks: tuple[int, ...] = (1, 2, 10, 20, 50)
    signal_thresholds: tuple[float, ...] = (2.0, 3.0)
    query_batch_size: int = 3_584
    instrument_batch_size: int = 20
    roundoff_multiplier: float = DEFAULT_ROUNDOFF_MULTIPLIER
    state_scale_floor: float = 1.0e-6

    def __post_init__(self) -> None:
        verify_inp(
            self.state_mode in {"raw", "instrument"},
            lambda: f"unknown state mode {self.state_mode!r}",
        )
        for name in (
            "neighbors",
            "instrument_candidate_neighbors",
            "query_batch_size",
            "instrument_batch_size",
        ):
            value = getattr(self, name)
            verify_inp(
                isinstance(value, int) and value > 0,
                lambda name=name, value=value: f"'{name}' must be positive, got {value!r}",
            )
        verify_inp(
            self.instrument_candidate_neighbors >= self.neighbors,
            lambda: (
                "instrument candidate neighbours must be at least the final "
                f"neighbour count, got {self.instrument_candidate_neighbors} < {self.neighbors}"
            ),
        )
        verify_inp(bool(self.tail_ranks), "at least one local tail rank is required")
        verify_inp(
            all(isinstance(rank, int) and rank > 0 for rank in self.tail_ranks),
            lambda: f"tail ranks must be positive integers, got {self.tail_ranks}",
        )
        verify_inp(
            tuple(sorted(set(self.tail_ranks))) == self.tail_ranks,
            lambda: f"tail ranks must be unique and increasing, got {self.tail_ranks}",
        )
        verify_inp(
            bool(self.signal_thresholds)
            and all(
                math.isfinite(value) and value > 0.0
                for value in self.signal_thresholds
            )
            and tuple(sorted(set(self.signal_thresholds)))
            == self.signal_thresholds,
            lambda: (
                "signal thresholds must be finite, positive, unique, and "
                f"increasing, got {self.signal_thresholds}"
            ),
        )
        verify_inp(
            2 * self.tail_ranks[-1] <= self.neighbors,
            lambda: (
                f"largest tail rank {self.tail_ranks[-1]} needs at most half of "
                f"{self.neighbors} neighbours"
            ),
        )
        for name in ("roundoff_multiplier", "state_scale_floor"):
            value = getattr(self, name)
            verify_inp(
                math.isfinite(value) and value > 0.0,
                lambda name=name, value=value: f"'{name}' must be finite and positive, got {value}",
            )

    @property
    def local_tail_probabilities(self) -> tuple[float, ...]:
        return tuple(rank / self.neighbors for rank in self.tail_ranks)

    @property
    def zero_event_upper_probability_95(self) -> float:
        """Exact iid binomial upper bound after zero events in ``neighbors`` draws."""

        return 1.0 - 0.05 ** (1.0 / self.neighbors)


def prepare_delta_hedged_local_data(
    universe: Mapping[str, Any],
    *,
    roundoff_multiplier: float = DEFAULT_ROUNDOFF_MULTIPLIER,
) -> tuple[dict[str, np.ndarray], UniverseGeometry]:
    """Return delta-hedged option returns, roundoff bands, and state fields."""

    geometry = validate_universe(universe)
    dH = delta_hedged_option_returns(
        dC=np.asarray(universe["dC"]),
        delta=np.asarray(universe["delta"]),
        dS=np.asarray(universe["dS"]),
    )
    roundoff = delta_hedged_roundoff_band(
        C=np.asarray(universe["C"]),
        dC=np.asarray(universe["dC"]),
        delta=np.asarray(universe["delta"]),
        dS=np.asarray(universe["dS"]),
        multiplier=roundoff_multiplier,
    )
    xstate = np.asarray(universe["xstate"], dtype=np.float32)[
        :, : geometry.steps
    ]
    log_strikes = np.asarray(universe["log_strikes"], dtype=np.float32).reshape(
        geometry.samples, geometry.steps, -1
    )
    expiries = np.asarray(universe["expiries"], dtype=np.float32).reshape(-1)
    expected = (geometry.samples, geometry.steps, geometry.expiries * geometry.strikes)
    assert dH.shape == roundoff.shape == log_strikes.shape == expected
    assert xstate.shape == (
        geometry.samples,
        geometry.steps,
        geometry.state_variables,
    )
    if not all(np.isfinite(value).all() for value in (dH, roundoff, xstate, log_strikes)):
        raise ValueError("prepared local-tail data contain non-finite values")
    if np.any(roundoff <= 0.0):
        raise ValueError("local-tail roundoff bands must be positive")
    return {
        "returns": dH.astype(np.float32, copy=False),
        "roundoff": roundoff.astype(np.float32, copy=False),
        "xstate": xstate,
        "log_strikes": log_strikes,
        "expiries": expiries,
    }, geometry


def reduced_option_state(
    data: Mapping[str, np.ndarray],
    *,
    geometry: UniverseGeometry,
    mode: StateMode,
    instrument: int = 0,
) -> np.ndarray:
    """Return ``xstate`` or one option's ``(xstate, log-strike, expiry)``."""

    verify_inp(mode in {"raw", "instrument"}, lambda: f"unknown state mode {mode!r}")
    instruments = geometry.expiries * geometry.strikes
    verify_inp(
        0 <= instrument < instruments,
        lambda: f"instrument {instrument} is outside [0, {instruments})",
    )
    xstate = np.asarray(data["xstate"], dtype=np.float32)
    if mode == "raw":
        return xstate
    log_strike = np.asarray(data["log_strikes"], dtype=np.float32)[
        ..., instrument : instrument + 1
    ]
    expiry_index = instrument // geometry.strikes
    expiry = np.full(
        (*xstate.shape[:-1], 1),
        np.asarray(data["expiries"], dtype=np.float32)[expiry_index],
        dtype=np.float32,
    )
    output = np.concatenate((xstate, log_strike, expiry), axis=-1)
    assert output.shape[:-1] == xstate.shape[:-1]
    assert output.shape[-1] == geometry.state_variables + 2
    return output


def state_normalization(
    training_state: np.ndarray,
    *,
    scale_floor: float,
    path_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return training-only weighted population location and scale."""

    verify_inp(
        training_state.ndim == 3 and training_state.shape[-1] > 0,
        lambda: f"training state must have shape (S, TAU, F), got {training_state.shape}",
    )
    verify_inp(
        math.isfinite(scale_floor) and scale_floor > 0.0,
        lambda: f"scale floor must be finite and positive, got {scale_floor}",
    )
    flat = training_state.reshape(-1, training_state.shape[-1]).astype(
        np.float64, copy=False
    )
    if path_weights is None:
        mean = flat.mean(axis=0, dtype=np.float64)
        scale = flat.std(axis=0, dtype=np.float64)
    else:
        weights = np.asarray(path_weights, dtype=np.float64)
        verify_inp(
            weights.shape == (training_state.shape[0],),
            lambda: (
                "path weights must have one value per independent path, got "
                f"{weights.shape} for state {training_state.shape}"
            ),
        )
        if not np.all(np.isfinite(weights)) or not np.all(weights > 0.0):
            raise ValueError("state-normalization path weights must be finite and positive")
        observation_weights = np.repeat(weights, training_state.shape[1])
        mass = float(observation_weights.sum())
        mean = np.sum(observation_weights[:, None] * flat, axis=0) / mass
        variance = (
            np.sum(observation_weights[:, None] * np.square(flat - mean), axis=0)
            / mass
        )
        scale = np.sqrt(np.maximum(variance, 0.0))
    constant = scale < scale_floor
    scale = np.maximum(scale, scale_floor)
    return mean.astype(np.float32), scale.astype(np.float32), constant


def normalize_state(
    state: np.ndarray, *, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Normalize a state array without changing its shape or float32 dtype."""

    verify_inp(
        state.shape[-1] == mean.shape[0] == scale.shape[0],
        lambda: f"state/normalization shapes differ: {state.shape}/{mean.shape}/{scale.shape}",
    )
    if not np.isfinite(state).all() or not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("state normalization contains non-finite values")
    if np.any(scale <= 0.0):
        raise ValueError("state normalization scale must be positive")
    return ((state - mean) / scale).astype(np.float32, copy=False)


def shared_instrument_metric_diagnostics(
    data: Mapping[str, np.ndarray],
    *,
    training_paths: int,
    geometry: UniverseGeometry,
    scale_floor: float,
    sample_states: int = 4_096,
) -> dict[str, Any]:
    """Test whether every option produces the same standardized strike metric.

    When log strikes differ only by instrument-specific constants, their
    standardized values are identical.  Expiries are constant and therefore
    have zero distance.  In that case one exact neighbour graph is valid for
    every separately estimated option.
    """

    log_strikes = np.asarray(data["log_strikes"], dtype=np.float32)[
        :training_paths
    ].reshape(-1, geometry.expiries * geometry.strikes)
    mean = log_strikes.mean(axis=0, dtype=np.float64)
    scale = log_strikes.std(axis=0, dtype=np.float64)
    constant_strikes = scale < scale_floor
    safe_scale = np.maximum(scale, scale_floor)
    count = min(sample_states, log_strikes.shape[0])
    sample_index = np.linspace(0, log_strikes.shape[0] - 1, count, dtype=np.int64)
    standardized = (
        log_strikes[sample_index].astype(np.float64) - mean
    ) / safe_scale
    deviation = np.max(np.abs(standardized - standardized[:, :1]))
    expiries = np.asarray(data["expiries"], dtype=np.float32)
    output = {
        "sample_states": count,
        "max_abs_standardized_log_strike_difference": float(deviation),
        "constant_log_strike_instruments": int(constant_strikes.sum()),
        "expiry_values": expiries.astype(float).tolist(),
        "expiry_is_constant_across_states": True,
        "shared_metric": bool(deviation <= 5.0e-5),
    }
    return output


def _weighted_lower_tail(
    values: Tensor,
    normalized_weights: Tensor,
    *,
    probabilities: tuple[float, ...],
) -> tuple[Tensor, Tensor]:
    """Return weighted lower quantiles and CVaRs along the neighbour axis."""

    sorted_values, order = torch.sort(values, dim=1)
    sorted_weights = torch.gather(normalized_weights, 1, order)
    cumulative = torch.cumsum(sorted_weights, dim=1)
    cumulative_before = cumulative - sorted_weights
    quantiles: list[Tensor] = []
    cvars: list[Tensor] = []
    for probability in probabilities:
        threshold = torch.as_tensor(
            probability, dtype=values.dtype, device=values.device
        )
        index = torch.argmax((cumulative >= threshold).to(torch.int8), dim=1)
        quantiles.append(
            torch.gather(sorted_values, 1, index[:, None, :]).squeeze(1)
        )
        included = torch.minimum(
            sorted_weights,
            torch.relu(threshold - cumulative_before),
        )
        cvars.append((sorted_values * included).sum(dim=1) / threshold)
    return torch.stack(quantiles, dim=1), torch.stack(cvars, dim=1)


def local_neighbor_statistics(
    neighbor_returns: Tensor,
    neighbor_roundoff: Tensor,
    *,
    tail_ranks: tuple[int, ...],
    neighbor_log_weights: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return positive/negative parts and signed lower-tail order statistics.

    Inputs have shape ``(queries, neighbours, instruments)``.  Direction zero
    is the long return and direction one is the negated return.
    """

    verify_inp(
        neighbor_returns.shape == neighbor_roundoff.shape
        and neighbor_returns.ndim == 3,
        lambda: (
            "neighbour returns/roundoff must share shape (Q, K, J), got "
            f"{neighbor_returns.shape}/{neighbor_roundoff.shape}"
        ),
    )
    neighbors = neighbor_returns.shape[1]
    verify_inp(
        bool(tail_ranks)
        and tuple(sorted(set(tail_ranks))) == tail_ranks
        and 2 * tail_ranks[-1] <= neighbors,
        lambda: f"invalid tail ranks {tail_ranks} for {neighbors} neighbours",
    )
    if not torch.isfinite(neighbor_returns).all().item():
        raise ValueError("neighbour returns contain non-finite values")
    if not torch.isfinite(neighbor_roundoff).all().item() or not torch.all(
        neighbor_roundoff >= 0.0
    ).item():
        raise ValueError("neighbour roundoff is non-finite or negative")

    weighted = neighbor_log_weights is not None
    if neighbor_log_weights is None:
        normalized_weights = torch.full_like(
            neighbor_returns,
            1.0 / neighbors,
        )
    else:
        verify_inp(
            neighbor_log_weights.shape
            in {neighbor_returns.shape[:2], neighbor_returns.shape},
            lambda: (
                "neighbour log weights must have shape (Q,K) or (Q,K,J), got "
                f"{neighbor_log_weights.shape} for {neighbor_returns.shape}"
            ),
        )
        if not torch.isfinite(neighbor_log_weights).all().item():
            raise ValueError("neighbour log weights contain non-finite values")
        log_weights = neighbor_log_weights.to(neighbor_returns.dtype)
        if log_weights.ndim == 2:
            log_weights = log_weights[..., None].expand_as(neighbor_returns)
        normalized_weights = torch.softmax(log_weights, dim=1)

    positive = torch.relu(neighbor_returns - neighbor_roundoff)
    negative = torch.relu(-neighbor_returns - neighbor_roundoff)
    positive_event = neighbor_returns > neighbor_roundoff
    negative_event = neighbor_returns < -neighbor_roundoff
    mean = (normalized_weights * neighbor_returns).sum(dim=1)
    second_moment = (
        normalized_weights * neighbor_returns.square()
    ).sum(dim=1)
    variance = (second_moment - mean.square()).clamp_min(0.0)
    if weighted:
        probabilities = tuple(rank / neighbors for rank in tail_ranks)
        long_quantile, long_cvar = _weighted_lower_tail(
            neighbor_returns,
            normalized_weights,
            probabilities=probabilities,
        )
        short_quantile, short_cvar = _weighted_lower_tail(
            -neighbor_returns,
            normalized_weights,
            probabilities=probabilities,
        )
    else:
        maximum_rank = tail_ranks[-1]
        lower = torch.topk(
            neighbor_returns,
            maximum_rank,
            dim=1,
            largest=False,
            sorted=True,
        ).values
        upper = torch.topk(
            neighbor_returns,
            maximum_rank,
            dim=1,
            largest=True,
            sorted=True,
        ).values
        long_quantile = torch.stack(
            [lower[:, rank - 1] for rank in tail_ranks], dim=1
        )
        short_quantile = torch.stack(
            [-upper[:, rank - 1] for rank in tail_ranks], dim=1
        )
        long_cvar = torch.stack(
            [lower[:, :rank].mean(dim=1) for rank in tail_ranks], dim=1
        )
        short_cvar = torch.stack(
            [-upper[:, :rank].mean(dim=1) for rank in tail_ranks], dim=1
        )
    return {
        "mean": mean,
        "second_moment": second_moment,
        "variance": variance,
        "positive_mean": (normalized_weights * positive).sum(dim=1),
        "negative_mean": (normalized_weights * negative).sum(dim=1),
        "positive_probability": (
            normalized_weights * positive_event.to(neighbor_returns.dtype)
        ).sum(dim=1),
        "negative_probability": (
            normalized_weights * negative_event.to(neighbor_returns.dtype)
        ).sum(dim=1),
        "positive_count": positive_event.sum(dim=1),
        "negative_count": negative_event.sum(dim=1),
        "effective_sample_size": normalized_weights.square().sum(dim=1).reciprocal(),
        "quantile": torch.stack((long_quantile, short_quantile), dim=1),
        "lower_cvar": torch.stack((long_cvar, short_cvar), dim=1),
    }


def _tensor_list(value: Tensor) -> list[Any]:
    return value.detach().cpu().tolist()


def _append_prefix_sample(
    sample: np.ndarray, count: int, values: np.ndarray
) -> int:
    """Append only the prefix which fits in a fixed diagnostic reservoir."""

    if count >= sample.size:
        return count
    take = min(values.size, sample.size - count)
    sample[count : count + take] = values[:take]
    return count + take


@torch.inference_mode()
def valuate_local_tail(
    *,
    reference_state: Tensor,
    reference_returns: Tensor,
    reference_roundoff: Tensor,
    reference_log_weights: Tensor | None = None,
    query_state: Tensor,
    query_returns: Tensor,
    query_roundoff: Tensor,
    query_log_weights: Tensor | None = None,
    reference_instrument_feature: Tensor | None = None,
    query_instrument_feature: Tensor | None = None,
    config: LocalTailConfig,
    split: str,
    leave_one_out: bool,
    leave_one_out_reference_indices: Tensor | None = None,
    progress: Callable[[LocalTailProgress], None] | None = None,
    progress_interval_seconds: float = 30.0,
) -> dict[str, Any]:
    """Value all query states with an exact adaptive-neighbour box kernel."""

    verify_inp(isinstance(split, str) and split != "", "split must be nonempty")
    verify_inp(
        math.isfinite(progress_interval_seconds) and progress_interval_seconds > 0.0,
        lambda: (
            "progress_interval_seconds must be finite and positive, got "
            f"{progress_interval_seconds}"
        ),
    )
    verify_inp(
        reference_state.ndim == query_state.ndim == 2,
        lambda: f"state tensors must be matrices, got {reference_state.shape}/{query_state.shape}",
    )
    verify_inp(
        reference_returns.shape == reference_roundoff.shape
        and query_returns.shape == query_roundoff.shape,
        lambda: "return and roundoff shapes must agree",
    )
    verify_inp(
        reference_returns.ndim == query_returns.ndim == 2
        and reference_returns.shape[1] == query_returns.shape[1],
        lambda: f"return tensors must have matching instrument axes",
    )
    verify_inp(
        reference_state.shape[0] == reference_returns.shape[0]
        and query_state.shape[0] == query_returns.shape[0],
        lambda: "state and return observation counts must agree",
    )
    for name, log_weights, observations in (
        ("reference_log_weights", reference_log_weights, reference_state.shape[0]),
        ("query_log_weights", query_log_weights, query_state.shape[0]),
    ):
        if log_weights is None:
            continue
        verify_inp(
            log_weights.shape == (observations,)
            and log_weights.dtype.is_floating_point,
            lambda name=name, log_weights=log_weights, observations=observations: (
                f"'{name}' must be floating with shape {(observations,)}, got "
                f"{log_weights.shape}/{log_weights.dtype}"
            ),
        )
    verify_inp(
        config.neighbors < reference_state.shape[0]
        if leave_one_out
        else config.neighbors <= reference_state.shape[0],
        lambda: f"not enough reference observations for {config.neighbors} neighbours",
    )
    if leave_one_out_reference_indices is not None:
        verify_inp(
            leave_one_out,
            "leave-one-out reference indices require leave_one_out=True",
        )
        verify_inp(
            leave_one_out_reference_indices.shape == (query_state.shape[0],)
            and leave_one_out_reference_indices.dtype == torch.int64,
            lambda: (
                "leave-one-out reference indices must be int64 with shape "
                f"{(query_state.shape[0],)}, got "
                f"{leave_one_out_reference_indices.shape}/"
                f"{leave_one_out_reference_indices.dtype}"
            ),
        )
        if (
            torch.any(leave_one_out_reference_indices < 0).item()
            or torch.any(
                leave_one_out_reference_indices >= reference_state.shape[0]
            ).item()
        ):
            raise ValueError("leave-one-out reference indices are out of range")
    elif leave_one_out:
        verify_inp(
            query_state.shape[0] == reference_state.shape[0],
            lambda: (
                "implicit leave-one-out requires identical query/reference "
                f"counts, got {query_state.shape[0]} and {reference_state.shape[0]}"
            ),
        )
    candidate_neighbors = (
        config.instrument_candidate_neighbors
        if config.state_mode == "instrument"
        else config.neighbors
    )
    verify_inp(
        candidate_neighbors < reference_state.shape[0]
        if leave_one_out
        else candidate_neighbors <= reference_state.shape[0],
        lambda: (
            f"not enough reference observations for {candidate_neighbors} "
            "candidate neighbours"
        ),
    )
    if config.state_mode == "instrument":
        verify_inp(
            reference_instrument_feature is not None
            and query_instrument_feature is not None,
            "instrument mode requires reference and query instrument features",
        )
        assert reference_instrument_feature is not None
        assert query_instrument_feature is not None
        verify_inp(
            reference_instrument_feature.shape == reference_returns.shape
            and query_instrument_feature.shape == query_returns.shape,
            lambda: (
                "instrument-feature shapes must match returns, got "
                f"{reference_instrument_feature.shape}/{reference_returns.shape} and "
                f"{query_instrument_feature.shape}/{query_returns.shape}"
            ),
        )
    else:
        verify_inp(
            reference_instrument_feature is None and query_instrument_feature is None,
            "raw mode does not consume instrument features",
        )
    tensors = (
        reference_state,
        reference_returns,
        reference_roundoff,
        query_state,
        query_returns,
        query_roundoff,
    )
    if reference_log_weights is not None:
        tensors += (reference_log_weights,)
    if query_log_weights is not None:
        tensors += (query_log_weights,)
    if reference_instrument_feature is not None:
        tensors += (reference_instrument_feature,)
    if query_instrument_feature is not None:
        tensors += (query_instrument_feature,)
    if leave_one_out_reference_indices is not None:
        tensors += (leave_one_out_reference_indices,)
    device = reference_state.device
    if any(value.device != device for value in tensors):
        raise ValueError("all local-tail tensors must be on the same device")
    if any(not torch.isfinite(value).all().item() for value in tensors):
        raise ValueError("local-tail tensors contain non-finite values")

    queries = query_state.shape[0]
    instruments = reference_returns.shape[1]
    levels = len(config.tail_ranks)
    signal_levels = len(config.signal_thresholds)
    moment_sum = torch.zeros(4, instruments, dtype=torch.float64, device=device)
    query_weight = (
        torch.ones(queries, dtype=torch.float64, device=device)
        if query_log_weights is None
        else torch.exp(
            query_log_weights.to(torch.float64)
            - query_log_weights.to(torch.float64).max()
        )
    )
    query_weight_sum = query_weight.sum()
    query_effective_sample_size = (
        query_weight_sum.square() / query_weight.square().sum()
    )
    support_count = torch.zeros(2, instruments, dtype=torch.int64, device=device)
    support_counter = torch.zeros_like(support_count)
    support_favorable = torch.zeros_like(support_count)
    quantile_count = torch.zeros(2, levels, instruments, dtype=torch.int64, device=device)
    quantile_counter = torch.zeros_like(quantile_count)
    cvar_count = torch.zeros_like(quantile_count)
    cvar_counter = torch.zeros_like(quantile_count)
    support_states = torch.zeros(2, dtype=torch.int64, device=device)
    exclusive_support_states = torch.zeros(3, dtype=torch.int64, device=device)
    support_state_weight = torch.zeros(2, dtype=torch.float64, device=device)
    exclusive_support_state_weight = torch.zeros(
        3, dtype=torch.float64, device=device
    )
    quantile_states = torch.zeros(2, levels, dtype=torch.int64, device=device)
    cvar_states = torch.zeros_like(quantile_states)
    quantile_both = torch.zeros(levels, dtype=torch.int64, device=device)
    cvar_both = torch.zeros_like(quantile_both)
    signal_count = torch.zeros(
        2, signal_levels, instruments, dtype=torch.int64, device=device
    )
    signal_counter = torch.zeros_like(signal_count)
    signal_states = torch.zeros(
        2, signal_levels, dtype=torch.int64, device=device
    )
    signal_both = torch.zeros(signal_levels, dtype=torch.int64, device=device)
    signal_max = torch.zeros(instruments, dtype=torch.float64, device=device)
    zero_sigma_count = torch.zeros(instruments, dtype=torch.int64, device=device)
    support_policy_pnl: list[np.ndarray] = []
    support_policy_band: list[np.ndarray] = []
    support_policy_instrument: list[np.ndarray] = []
    support_policy_direction: list[np.ndarray] = []
    support_policy_weight: list[np.ndarray] = []
    radius_comparisons = queries * (
        instruments if config.state_mode == "instrument" else 1
    )
    radius_sample = np.empty(min(1_000_000, radius_comparisons), dtype=np.float32)
    radius_sample_count = 0
    radius_sum = 0.0
    radius_minimum = math.inf
    radius_maximum = -math.inf
    certified_count = torch.zeros(instruments, dtype=torch.int64, device=device)
    local_ess_sum = torch.zeros((), dtype=torch.float64, device=device)
    local_ess_minimum = math.inf
    local_ess_maximum = -math.inf
    reference_norm = reference_state.square().sum(dim=1)
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
            excluded = (
                torch.arange(start, stop, device=device)
                if leave_one_out_reference_indices is None
                else leave_one_out_reference_indices[start:stop]
            )
            distance[row, excluded] = torch.inf
        nearest_distance, nearest = torch.topk(
            distance,
            candidate_neighbors,
            dim=1,
            largest=False,
            sorted=True,
        )
        del distance
        nearest_distance.clamp_min_(0.0)
        if config.state_mode == "raw":
            batch_radius = nearest_distance[:, -1].sqrt()
            values = batch_radius.cpu().numpy()
            radius_sample_count = _append_prefix_sample(
                radius_sample, radius_sample_count, values
            )
            radius_sum += float(batch_radius.sum(dtype=torch.float64).item())
            radius_minimum = min(radius_minimum, float(batch_radius.min().item()))
            radius_maximum = max(radius_maximum, float(batch_radius.max().item()))
            certified_count += stop - start
        batch_query_returns = query_returns[start:stop]
        batch_query_roundoff = query_roundoff[start:stop]
        batch_query_weight = query_weight[start:stop]
        batch_support_states = torch.zeros(
            2, stop - start, dtype=torch.bool, device=device
        )
        batch_quantile_states = torch.zeros(
            2, levels, stop - start, dtype=torch.bool, device=device
        )
        batch_cvar_states = torch.zeros_like(batch_quantile_states)
        batch_signal_states = torch.zeros(
            2,
            signal_levels,
            stop - start,
            dtype=torch.bool,
            device=device,
        )
        batch_policy_score = torch.full(
            (stop - start,), -torch.inf, dtype=query_returns.dtype, device=device
        )
        batch_policy_pnl = torch.zeros_like(batch_policy_score)
        batch_policy_band = torch.zeros_like(batch_policy_score)
        batch_policy_instrument = torch.full(
            (stop - start,), -1, dtype=torch.int64, device=device
        )
        batch_policy_direction = torch.full_like(batch_policy_instrument, -1)

        for left in range(0, instruments, config.instrument_batch_size):
            right = min(left + config.instrument_batch_size, instruments)
            if config.state_mode == "instrument":
                assert reference_instrument_feature is not None
                assert query_instrument_feature is not None
                reference_feature = reference_instrument_feature[:, left:right][nearest]
                query_feature = query_instrument_feature[
                    start:stop, left:right
                ][:, None, :]
                augmented_distance = nearest_distance[..., None] + (
                    reference_feature - query_feature
                ).square()
                selected_distance, selected_local = torch.topk(
                    augmented_distance,
                    config.neighbors,
                    dim=1,
                    largest=False,
                    sorted=True,
                )
                del augmented_distance, reference_feature, query_feature
                expanded_nearest = nearest[..., None].expand(
                    -1, -1, right - left
                )
                selected = torch.gather(expanded_nearest, 1, selected_local)
                del expanded_nearest, selected_local
                instrument_index = torch.arange(left, right, device=device).view(
                    1, 1, -1
                )
                neighbor_returns = reference_returns[selected, instrument_index]
                neighbor_roundoff = reference_roundoff[selected, instrument_index]
                neighbor_log_weights = (
                    None
                    if reference_log_weights is None
                    else reference_log_weights[selected]
                )
                certified = selected_distance[:, -1] < nearest_distance[:, -1, None]
                certified_count[left:right] += certified.sum(dim=0)
                batch_radius = selected_distance[:, -1].sqrt()
                radius_sum += float(batch_radius.sum(dtype=torch.float64).item())
                radius_minimum = min(radius_minimum, float(batch_radius.min().item()))
                radius_maximum = max(radius_maximum, float(batch_radius.max().item()))
                radius_sample_count = _append_prefix_sample(
                    radius_sample,
                    radius_sample_count,
                    batch_radius.flatten().cpu().numpy(),
                )
                del selected, selected_distance, certified, batch_radius
            else:
                neighbor_returns = reference_returns[:, left:right][nearest]
                neighbor_roundoff = reference_roundoff[:, left:right][nearest]
                neighbor_log_weights = (
                    None
                    if reference_log_weights is None
                    else reference_log_weights[nearest]
                )
            statistics = local_neighbor_statistics(
                neighbor_returns,
                neighbor_roundoff,
                tail_ranks=config.tail_ranks,
                neighbor_log_weights=neighbor_log_weights,
            )
            del neighbor_returns, neighbor_roundoff, neighbor_log_weights
            query_value = batch_query_returns[:, left:right]
            query_band = batch_query_roundoff[:, left:right]

            conditional_mean = statistics["mean"]
            conditional_sigma = statistics["variance"].sqrt()
            positive_sigma = conditional_sigma > 0.0
            signal_max[left:right] = torch.maximum(
                signal_max[left:right],
                torch.where(
                    positive_sigma,
                    conditional_mean.abs() / conditional_sigma,
                    torch.where(
                        conditional_mean == 0.0,
                        torch.zeros_like(conditional_mean),
                        torch.full_like(conditional_mean, torch.inf),
                    ),
                )
                .amax(dim=0)
                .to(torch.float64),
            )
            zero_sigma_count[left:right] += (~positive_sigma).sum(dim=0)

            local_ess = statistics["effective_sample_size"]
            local_ess_sum += local_ess.sum(dtype=torch.float64)
            local_ess_minimum = min(
                local_ess_minimum, float(local_ess.min().item())
            )
            local_ess_maximum = max(
                local_ess_maximum, float(local_ess.max().item())
            )
            for moment_index, name in enumerate(
                (
                    "positive_mean",
                    "negative_mean",
                    "positive_probability",
                    "negative_probability",
                )
            ):
                moment_sum[moment_index, left:right] += (
                    batch_query_weight[:, None]
                    * statistics[name].to(torch.float64)
                ).sum(dim=0)

            support_long = (statistics["negative_count"] == 0) & (
                statistics["positive_count"] > 0
            )
            support_short = (statistics["positive_count"] == 0) & (
                statistics["negative_count"] > 0
            )
            support = torch.stack((support_long, support_short), dim=0)
            signed_query = torch.stack((query_value, -query_value), dim=0)

            thresholds = torch.as_tensor(
                config.signal_thresholds,
                dtype=conditional_mean.dtype,
                device=device,
            ).reshape(1, -1, 1)
            signal_long = conditional_mean[:, None, :] > (
                thresholds * conditional_sigma[:, None, :]
            )
            signal_short = conditional_mean[:, None, :] < -(
                thresholds * conditional_sigma[:, None, :]
            )
            signal = torch.stack((signal_long, signal_short), dim=0).permute(
                0, 2, 1, 3
            )
            signal_count[:, :, left:right] += signal.sum(dim=2)
            signal_counter[:, :, left:right] += (
                signal & (signed_query[:, None] < -query_band[None, None])
            ).sum(dim=2)
            batch_signal_states |= signal.any(dim=3)
            signal_both += (signal_long & signal_short).sum(dim=(0, 2))
            support_count[:, left:right] += support.sum(dim=1)
            support_counter[:, left:right] += (
                support & (signed_query < -query_band)
            ).sum(dim=1)
            support_favorable[:, left:right] += (
                support & (signed_query > query_band)
            ).sum(dim=1)
            batch_support_states |= support.any(dim=2)

            predicted_gain = torch.stack(
                (statistics["positive_mean"], statistics["negative_mean"]),
                dim=0,
            )
            local_score = torch.where(
                support,
                predicted_gain,
                torch.full_like(predicted_gain, -torch.inf),
            ).permute(1, 0, 2).reshape(stop - start, -1)
            local_best_score, local_flat_index = local_score.max(dim=1)
            local_trade = torch.isfinite(local_best_score)
            local_direction = torch.div(
                local_flat_index, right - left, rounding_mode="floor"
            )
            local_instrument = local_flat_index % (right - left)
            local_pnl = signed_query.permute(1, 0, 2).reshape(
                stop - start, -1
            ).gather(1, local_flat_index[:, None]).squeeze(1)
            local_band = query_band[:, None, :].expand(
                -1, 2, -1
            ).reshape(stop - start, -1).gather(
                1, local_flat_index[:, None]
            ).squeeze(1)
            replace = local_trade & (local_best_score > batch_policy_score)
            batch_policy_score = torch.where(
                replace, local_best_score, batch_policy_score
            )
            batch_policy_pnl = torch.where(replace, local_pnl, batch_policy_pnl)
            batch_policy_band = torch.where(replace, local_band, batch_policy_band)
            batch_policy_instrument = torch.where(
                replace, local_instrument + left, batch_policy_instrument
            )
            batch_policy_direction = torch.where(
                replace, local_direction, batch_policy_direction
            )

            quantile = statistics["quantile"]
            cvar = statistics["lower_cvar"]
            tail_candidate = quantile > query_band[:, None, None]
            cvar_candidate = cvar > query_band[:, None, None]
            tail_candidate = tail_candidate.permute(1, 2, 0, 3)
            cvar_candidate = cvar_candidate.permute(1, 2, 0, 3)
            signed_counter = (
                signed_query[:, None] < -query_band[None, None]
            )
            quantile_count[:, :, left:right] += tail_candidate.sum(dim=2)
            quantile_counter[:, :, left:right] += (
                tail_candidate & signed_counter
            ).sum(dim=2)
            cvar_count[:, :, left:right] += cvar_candidate.sum(dim=2)
            cvar_counter[:, :, left:right] += (
                cvar_candidate & signed_counter
            ).sum(dim=2)
            batch_quantile_states |= tail_candidate.any(dim=3)
            batch_cvar_states |= cvar_candidate.any(dim=3)
            quantile_both += (
                tail_candidate[LONG_DIRECTION] & tail_candidate[SHORT_DIRECTION]
            ).sum(dim=(1, 2))
            cvar_both += (
                cvar_candidate[LONG_DIRECTION] & cvar_candidate[SHORT_DIRECTION]
            ).sum(dim=(1, 2))

        support_states += batch_support_states.sum(dim=1)
        support_state_weight += (
            batch_support_states.to(torch.float64)
            * batch_query_weight[None, :]
        ).sum(dim=1)
        long_state = batch_support_states[LONG_DIRECTION]
        short_state = batch_support_states[SHORT_DIRECTION]
        exclusive_support_states += torch.stack(
            (
                (long_state & ~short_state).sum(),
                (short_state & ~long_state).sum(),
                (long_state & short_state).sum(),
            )
        )
        exclusive_support_state_weight += torch.stack(
            (
                (
                    (long_state & ~short_state).to(torch.float64)
                    * batch_query_weight
                ).sum(),
                (
                    (short_state & ~long_state).to(torch.float64)
                    * batch_query_weight
                ).sum(),
                (
                    (long_state & short_state).to(torch.float64)
                    * batch_query_weight
                ).sum(),
            )
        )
        quantile_states += batch_quantile_states.sum(dim=2)
        cvar_states += batch_cvar_states.sum(dim=2)
        signal_states += batch_signal_states.sum(dim=2)
        support_policy_pnl.append(batch_policy_pnl.cpu().numpy())
        support_policy_band.append(batch_policy_band.cpu().numpy())
        support_policy_instrument.append(batch_policy_instrument.cpu().numpy())
        support_policy_direction.append(batch_policy_direction.cpu().numpy())
        support_policy_weight.append(batch_query_weight.cpu().numpy())
        now = time.monotonic()
        if progress is not None and (
            now - last_progress >= progress_interval_seconds or stop == queries
        ):
            state_counts = support_states.detach().cpu().tolist()
            exclusive_counts = exclusive_support_states.detach().cpu().tolist()
            progress(
                LocalTailProgress(
                    split=split,
                    processed_states=stop,
                    total_states=queries,
                    states_with_any_long_candidate=state_counts[LONG_DIRECTION],
                    states_with_any_short_candidate=state_counts[SHORT_DIRECTION],
                    long_only_candidate_states=exclusive_counts[0],
                    short_only_candidate_states=exclusive_counts[1],
                    both_direction_candidate_states=exclusive_counts[2],
                )
            )
            last_progress = now

    elapsed = time.monotonic() - started
    denominator = float(query_weight_sum.item())
    radius_values = radius_sample[:radius_sample_count]

    def candidate_block(
        count: Tensor,
        counter: Tensor,
        states: Tensor,
        both: Tensor | None = None,
    ) -> dict[str, Any]:
        rate = torch.where(count > 0, counter.to(torch.float64) / count, 0.0)
        output: dict[str, Any] = {
            "count_by_direction_level_instrument": _tensor_list(count),
            "count_by_direction_level": _tensor_list(count.sum(dim=-1)),
            "states_with_any_by_direction_level": _tensor_list(states),
            "realized_counter_sign_count_by_direction_level": _tensor_list(
                counter.sum(dim=-1)
            ),
            "realized_counter_sign_rate_by_direction_level": _tensor_list(
                torch.where(
                    count.sum(dim=-1) > 0,
                    counter.sum(dim=-1).to(torch.float64) / count.sum(dim=-1),
                    0.0,
                )
            ),
            "realized_counter_sign_rate_by_direction_level_instrument": _tensor_list(
                rate
            ),
        }
        if both is not None:
            output["both_directions_positive_count_by_level"] = _tensor_list(both)
        return output

    support_rate = torch.where(
        support_count > 0,
        support_counter.to(torch.float64) / support_count,
        0.0,
    )
    policy_pnl = np.concatenate(support_policy_pnl).astype(np.float64, copy=False)
    policy_band = np.concatenate(support_policy_band).astype(np.float64, copy=False)
    policy_instrument = np.concatenate(support_policy_instrument)
    policy_direction = np.concatenate(support_policy_direction)
    policy_weight = np.concatenate(support_policy_weight).astype(
        np.float64, copy=False
    )
    policy_trade = policy_instrument >= 0
    policy_loss = policy_trade & (policy_pnl < -policy_band)
    policy_gain = policy_trade & (policy_pnl > policy_band)
    policy_neutral = policy_trade & ~policy_loss & ~policy_gain
    policy_trade_count = int(policy_trade.sum())
    policy_quantile_levels = np.asarray(
        [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0], dtype=np.float64
    )
    policy_selected_count = np.bincount(
        policy_instrument[policy_trade], minlength=instruments
    )
    policy_direction_count = np.bincount(
        policy_direction[policy_trade], minlength=2
    )
    policy_weight_sum = float(policy_weight.sum())
    policy_trade_weight = float(policy_weight[policy_trade].sum())
    policy_gain_weight = float(policy_weight[policy_gain].sum())
    policy_loss_weight = float(policy_weight[policy_loss].sum())

    def weighted_policy_quantiles() -> list[float] | None:
        if policy_trade_count == 0:
            return None
        values = policy_pnl[policy_trade]
        weights = policy_weight[policy_trade]
        order = np.argsort(values)
        values = values[order]
        weights = weights[order]
        cumulative = np.cumsum(weights) / np.sum(weights)
        return [
            float(values[min(np.searchsorted(cumulative, level), values.size - 1)])
            for level in policy_quantile_levels
        ]
    return {
        "split": split,
        "queries": queries,
        "reference_observations": reference_state.shape[0],
        "instruments": instruments,
        "leave_one_out": leave_one_out,
        "neighbors": config.neighbors,
        "effective_sample_size": float(
            local_ess_sum.item() / (queries * instruments)
        ),
        "local_effective_sample_size": {
            "minimum": local_ess_minimum,
            "mean": float(local_ess_sum.item() / (queries * instruments)),
            "maximum": local_ess_maximum,
        },
        "query_measure": (
            "likelihood-weighted original P"
            if query_log_weights is not None
            else "uniform empirical"
        ),
        "query_weight_sum": float(query_weight_sum.item()),
        "query_effective_sample_size": float(
            query_effective_sample_size.item()
        ),
        "tail_ranks": list(config.tail_ranks),
        "tail_probabilities": list(config.local_tail_probabilities),
        "zero_event_upper_probability_95": (
            None
            if reference_log_weights is not None
            else config.zero_event_upper_probability_95
        ),
        "mean_positive_part_by_instrument": _tensor_list(moment_sum[0] / denominator),
        "mean_negative_part_by_instrument": _tensor_list(moment_sum[1] / denominator),
        "mean_positive_probability_by_instrument": _tensor_list(moment_sum[2] / denominator),
        "mean_negative_probability_by_instrument": _tensor_list(moment_sum[3] / denominator),
        "mean_positive_part": float(moment_sum[0].sum().item() / (denominator * instruments)),
        "mean_negative_part": float(moment_sum[1].sum().item() / (denominator * instruments)),
        "mean_positive_probability": float(moment_sum[2].sum().item() / (denominator * instruments)),
        "mean_negative_probability": float(moment_sum[3].sum().item() / (denominator * instruments)),
        "zero_adverse": {
            "count_by_direction_instrument": _tensor_list(support_count),
            "count_by_direction": _tensor_list(support_count.sum(dim=-1)),
            "states_with_any_by_direction": _tensor_list(support_states),
            "weighted_state_probability_by_direction": _tensor_list(
                support_state_weight / query_weight_sum
            ),
            "states_by_direction_exclusivity": {
                "long_only": int(exclusive_support_states[0].item()),
                "short_only": int(exclusive_support_states[1].item()),
                "both_directions": int(exclusive_support_states[2].item()),
                "neither_direction": int(
                    queries - exclusive_support_states.sum().item()
                ),
            },
            "weighted_state_probability_by_direction_exclusivity": {
                "long_only": float(
                    exclusive_support_state_weight[0].item() / denominator
                ),
                "short_only": float(
                    exclusive_support_state_weight[1].item() / denominator
                ),
                "both_directions": float(
                    exclusive_support_state_weight[2].item() / denominator
                ),
                "neither_direction": float(
                    1.0
                    - exclusive_support_state_weight.sum().item() / denominator
                ),
            },
            "realized_counter_sign_count_by_direction": _tensor_list(
                support_counter.sum(dim=-1)
            ),
            "realized_counter_sign_rate_by_direction": _tensor_list(
                torch.where(
                    support_count.sum(dim=-1) > 0,
                    support_counter.sum(dim=-1).to(torch.float64)
                    / support_count.sum(dim=-1),
                    0.0,
                )
            ),
            "realized_favorable_sign_count_by_direction": _tensor_list(
                support_favorable.sum(dim=-1)
            ),
            "realized_favorable_sign_rate_by_direction": _tensor_list(
                torch.where(
                    support_count.sum(dim=-1) > 0,
                    support_favorable.sum(dim=-1).to(torch.float64)
                    / support_count.sum(dim=-1),
                    0.0,
                )
            ),
            "realized_counter_sign_rate_by_direction_instrument": _tensor_list(
                support_rate
            ),
        },
        "zero_adverse_single_instrument_policy": {
            "definition": (
                "At each query state, among directions with zero adverse "
                "outcomes in the training-neighbour sample, select the one "
                "instrument/direction with the largest estimated favorable "
                "part and use unit L1 exposure; otherwise do not trade."
            ),
            "query_states": queries,
            "trading_states": policy_trade_count,
            "trading_state_probability": policy_trade_count / queries,
            "weighted_trading_state_probability": (
                policy_trade_weight / policy_weight_sum
            ),
            "long_trades": int(policy_direction_count[LONG_DIRECTION]),
            "short_trades": int(policy_direction_count[SHORT_DIRECTION]),
            "realized_gains_beyond_roundoff": int(policy_gain.sum()),
            "realized_losses_beyond_roundoff": int(policy_loss.sum()),
            "realized_neutral_within_roundoff": int(policy_neutral.sum()),
            "empirical_nonnegative_within_roundoff": bool(not np.any(policy_loss)),
            "positive_return_probability_all_queries": (
                policy_gain_weight / policy_weight_sum
            ),
            "positive_return_probability_when_trading": (
                0.0
                if policy_trade_count == 0
                else policy_gain_weight / policy_trade_weight
            ),
            "loss_probability_when_trading": (
                0.0
                if policy_trade_count == 0
                else policy_loss_weight / policy_trade_weight
            ),
            "zero_loss_probability_upper_95_when_trading": (
                None
                if (
                    policy_trade_count == 0
                    or np.any(policy_loss)
                    or query_log_weights is not None
                )
                else 1.0 - 0.05 ** (1.0 / policy_trade_count)
            ),
            "pnl_mean_all_queries": float(
                np.dot(policy_weight, policy_pnl) / policy_weight_sum
            ),
            "pnl_mean_when_trading": (
                0.0
                if policy_trade_count == 0
                else float(
                    np.dot(
                        policy_weight[policy_trade], policy_pnl[policy_trade]
                    )
                    / policy_trade_weight
                )
            ),
            "pnl_quantile_levels": policy_quantile_levels.tolist(),
            "pnl_quantiles_when_trading": (
                weighted_policy_quantiles()
            ),
            "selected_count_by_instrument": policy_selected_count.tolist(),
        },
        "conditional_moment_signal": {
            "definition": (
                "directional E[dH|z] exceeds g times the population "
                "conditional standard deviation estimated from the same "
                "uniform k-nearest-neighbour sample"
            ),
            "thresholds": list(config.signal_thresholds),
            **candidate_block(
                signal_count, signal_counter, signal_states, signal_both
            ),
            "maximum_abs_mean_over_sigma_by_instrument": _tensor_list(
                signal_max
            ),
            "maximum_abs_mean_over_sigma": float(signal_max.max().item()),
            "zero_sigma_count_by_instrument": _tensor_list(zero_sigma_count),
            "zero_sigma_count": int(zero_sigma_count.sum().item()),
        },
        "quantile_positive": candidate_block(
            quantile_count, quantile_counter, quantile_states, quantile_both
        ),
        "lower_cvar_positive": candidate_block(
            cvar_count, cvar_counter, cvar_states, cvar_both
        ),
        "neighbor_radius": {
            "minimum": radius_minimum,
            "mean": radius_sum / float(radius_comparisons),
            "quantiles": {
                f"{probability:g}": float(np.quantile(radius_values, probability))
                for probability in (0.05, 0.5, 0.95)
            },
            "maximum": radius_maximum,
            "quantile_sample_comparisons": radius_sample_count,
            "total_comparisons": radius_comparisons,
            "metric": "Euclidean distance after training-only componentwise standardization",
        },
        "neighbor_search": {
            "candidate_neighbors": candidate_neighbors,
            "final_neighbors": config.neighbors,
            "certified_exact_count_by_instrument": _tensor_list(certified_count),
            "certified_exact_fraction_by_instrument": _tensor_list(
                certified_count.to(torch.float64) / queries
            ),
            "certified_exact_fraction": float(
                certified_count.sum().item() / (queries * instruments)
            ),
            "certification": (
                "selected augmented kth distance is strictly below the raw "
                "candidate-pool boundary"
                if config.state_mode == "instrument"
                else "exact all-reference search"
            ),
        },
        "elapsed_seconds": elapsed,
    }


__all__ = [
    "DIRECTIONS",
    "LONG_DIRECTION",
    "LocalTailConfig",
    "LocalTailProgress",
    "SHORT_DIRECTION",
    "local_neighbor_statistics",
    "normalize_state",
    "prepare_delta_hedged_local_data",
    "reduced_option_state",
    "shared_instrument_metric_diagnostics",
    "state_normalization",
    "valuate_local_tail",
]
