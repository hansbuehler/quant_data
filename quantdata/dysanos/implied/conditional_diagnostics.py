"""Shared SANOS local-tail diagnostics from a compact PCA universe store."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cdxcore.err import verify_inp

from quantdata.dysanos.grid.gendysanos.run_terminal_arbitrage_tests import (
    _returns_and_roundoff,
    _sample_indices,
    _stable_path_weights,
)
from quantdata.dysanos.grid.rndysanos.local_tail import (
    LocalTailConfig,
    LocalTailProgress,
    _weighted_lower_tail,
    normalize_state,
    state_normalization,
    valuate_local_tail,
)
from quantdata.dysanos.grid.rndysanos.unconditional_option_test import (
    delta_hedged_option_cost_notionals,
    net_return_roundoff_band,
)
from quantdata.torch import DeviceLock

from .experiment import ImpliedPCAExperiment
from .universe import PCAUniverseBlock

_LOCAL_TAIL_STORE_SCHEMA_VERSION = "0.2"


class CompactLocalTailStore:
    """Disk-backed state/return arrays required by exact neighbour searches."""

    def __init__(
        self,
        root: str | Path,
        *,
        experiment: ImpliedPCAExperiment,
    ) -> None:
        self.directory = Path(root) / "local-tail-compact"
        self.experiment = experiment
        self.manifest_path = self.directory / "manifest.json"

    @property
    def is_complete(self) -> bool:
        if not self.manifest_path.exists():
            return False
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            value.get("status") == "complete"
            and value.get("cache_uid") == self.experiment.filename
            and value.get("schema_version")
            == _LOCAL_TAIL_STORE_SCHEMA_VERSION
            and all((self.directory / name).exists() for name in value["files"].values())
        )

    def build(self, blocks: Iterable[PCAUniverseBlock]) -> None:
        """Persist only raw state, return, roundoff, likelihood, and component."""

        self.directory.mkdir(parents=True, exist_ok=True)
        state = returns = roundoff = cost = net_roundoff = weights = components = None
        files = {
            "state": "state.npy",
            "returns": "returns.npy",
            "roundoff": "roundoff.npy",
            "cost": "cost.npy",
            "net_roundoff": "net_roundoff.npy",
            "log_path_weight": "log_path_weight.npy",
            "importance_component": "importance_component.npy",
        }
        last_stop = 0
        for block in blocks:
            universe = block.universe
            block_returns, block_roundoff = _returns_and_roundoff(universe)
            block_state = np.asarray(universe["xstate"], dtype=np.float32)
            option_cost = self.experiment.cost_proportion * (
                delta_hedged_option_cost_notionals(
                    C=np.asarray(universe["C"], dtype=np.float32),
                    delta=np.asarray(universe["delta"], dtype=np.float32),
                    spot=np.asarray(universe["spot"], dtype=np.float32),
                )
            )
            block_cost = np.zeros_like(block_returns)
            block_cost[..., 0] = self.experiment.cost_proportion * np.abs(
                np.asarray(universe["spot"], dtype=np.float32)
            )
            block_cost[..., 1:] = option_cost
            block_net_roundoff = net_return_roundoff_band(
                block_returns, block_cost
            )
            if state is None:
                S = self.experiment.num_samples
                state = np.lib.format.open_memmap(
                    self.directory / files["state"],
                    mode="w+",
                    dtype=np.float32,
                    shape=(S, *block_state.shape[1:]),
                )
                returns = np.lib.format.open_memmap(
                    self.directory / files["returns"],
                    mode="w+",
                    dtype=np.float32,
                    shape=(S, *block_returns.shape[1:]),
                )
                roundoff = np.lib.format.open_memmap(
                    self.directory / files["roundoff"],
                    mode="w+",
                    dtype=np.float32,
                    shape=(S, *block_roundoff.shape[1:]),
                )
                cost = np.lib.format.open_memmap(
                    self.directory / files["cost"],
                    mode="w+",
                    dtype=np.float32,
                    shape=(S, *block_cost.shape[1:]),
                )
                net_roundoff = np.lib.format.open_memmap(
                    self.directory / files["net_roundoff"],
                    mode="w+",
                    dtype=np.float32,
                    shape=(S, *block_net_roundoff.shape[1:]),
                )
                weights = np.lib.format.open_memmap(
                    self.directory / files["log_path_weight"],
                    mode="w+",
                    dtype=np.float64,
                    shape=(S,),
                )
                components = np.lib.format.open_memmap(
                    self.directory / files["importance_component"],
                    mode="w+",
                    dtype=np.int16,
                    shape=(S,),
                )
            assert all(
                value is not None
                for value in (
                    state,
                    returns,
                    roundoff,
                    cost,
                    net_roundoff,
                    weights,
                    components,
                )
            )
            verify_inp(
                block.start == last_stop,
                lambda: f"compact local-tail blocks are not contiguous at {block.start}",
            )
            state[block.start : block.stop] = block_state
            returns[block.start : block.stop] = block_returns
            roundoff[block.start : block.stop] = block_roundoff
            cost[block.start : block.stop] = block_cost
            net_roundoff[block.start : block.stop] = block_net_roundoff
            weights[block.start : block.stop] = np.asarray(
                universe.get(
                    "log_path_weight", np.zeros(block.stop - block.start)
                ),
                dtype=np.float64,
            )
            components[block.start : block.stop] = np.asarray(
                universe.get(
                    "importance_component", np.zeros(block.stop - block.start)
                ),
                dtype=np.int16,
            )
            last_stop = block.stop
        verify_inp(
            state is not None and last_stop == self.experiment.num_samples,
            "compact local-tail store did not receive the complete universe",
        )
        for value in (
            state,
            returns,
            roundoff,
            cost,
            net_roundoff,
            weights,
            components,
        ):
            assert value is not None
            value.flush()
        manifest = {
            "status": "complete",
            "schema_version": _LOCAL_TAIL_STORE_SCHEMA_VERSION,
            "cache_uid": self.experiment.filename,
            "configuration": asdict(self.experiment),
            "files": files,
            "semantics": {
                "state": "log-spot plus PCA surface scores",
                "returns": "spot and matched fixed-contract delta-hedged returns",
                "roundoff": "32 times float32 unit roundoff on unreduced arithmetic",
                "cost": (
                    "spot: cost_proportion*abs(S); delta-hedged option: "
                    "cost_proportion*(abs(C)+abs(delta*S)), without netting"
                ),
                "net_roundoff": "32 times float32 unit roundoff for directional net return",
                "weights": "complete path log(p/q)",
            },
        }
        temporary = self.manifest_path.with_name(
            f".{self.manifest_path.name}.{os.getpid()}.tmp"
        )
        try:
            temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            os.replace(temporary, self.manifest_path)
        finally:
            temporary.unlink(missing_ok=True)

    def arrays(self) -> dict[str, np.ndarray]:
        verify_inp(self.is_complete, "compact local-tail store is incomplete")
        files = json.loads(self.manifest_path.read_text(encoding="utf-8"))["files"]
        return {
            name: np.load(self.directory / filename, mmap_mode="r")
            for name, filename in files.items()
        }


def _cost_adjusted_local_directions(
    *,
    reference_state: torch.Tensor,
    reference_returns: torch.Tensor,
    reference_cost: torch.Tensor,
    reference_roundoff: torch.Tensor,
    reference_log_weights: torch.Tensor,
    query_state: torch.Tensor,
    query_returns: torch.Tensor,
    query_cost: torch.Tensor,
    query_roundoff: torch.Tensor,
    config: LocalTailConfig,
    split: str,
    leave_one_out_indices: torch.Tensor | None,
    progress_interval_seconds: float,
    return_neighbor_indices: bool = False,
) -> dict[str, Any]:
    """Evaluate long ``dH-c`` and short ``-dH-c`` as separate directions."""

    import time

    verify_inp(
        reference_returns.shape == reference_cost.shape == reference_roundoff.shape
        and query_returns.shape == query_cost.shape == query_roundoff.shape,
        "cost-adjusted local arrays must have matching return geometry",
    )
    queries, instruments = query_returns.shape
    directions = 2
    levels = len(config.tail_ranks)
    signals = len(config.signal_thresholds)
    device = reference_state.device
    dtype = reference_returns.dtype
    reference_norm = reference_state.square().sum(dim=1)
    favorable_probability_sum = torch.zeros(
        directions, instruments, dtype=torch.float64, device=device
    )
    adverse_probability_sum = torch.zeros_like(favorable_probability_sum)
    favorable_part_sum = torch.zeros_like(favorable_probability_sum)
    adverse_part_sum = torch.zeros_like(favorable_probability_sum)
    quantile_sum = torch.zeros(
        directions, levels, instruments, dtype=torch.float64, device=device
    )
    cvar_sum = torch.zeros_like(quantile_sum)
    one_sided_count = torch.zeros(
        directions, instruments, dtype=torch.int64, device=device
    )
    realized_adverse = torch.zeros_like(one_sided_count)
    signal_count = torch.zeros(
        directions, signals, instruments, dtype=torch.int64, device=device
    )
    local_ess_sum = 0.0
    local_ess_minimum = math.inf
    local_ess_maximum = -math.inf
    radius_sum = 0.0
    radius_minimum = math.inf
    radius_maximum = -math.inf
    policy_trades = policy_losses = policy_gains = 0
    screened_candidates: list[dict[str, Any]] = []
    neighbor_index_batches: list[np.ndarray] = []
    neighbor_radius_batches: list[np.ndarray] = []
    calibration_bins = 10
    calibration_count = {
        "loss": np.zeros(calibration_bins, dtype=np.int64),
        "gain": np.zeros(calibration_bins, dtype=np.int64),
    }
    calibration_prediction = {
        "loss": np.zeros(calibration_bins, dtype=np.float64),
        "gain": np.zeros(calibration_bins, dtype=np.float64),
    }
    calibration_realized = {
        "loss": np.zeros(calibration_bins, dtype=np.float64),
        "gain": np.zeros(calibration_bins, dtype=np.float64),
    }
    mean_residual_count = 0
    mean_residual_sum = 0.0
    mean_residual_square_sum = 0.0
    outcome_interval_coverage = 0
    long_only_states = short_only_states = both_states = 0
    started = time.monotonic()
    last_progress = -math.inf

    for start in range(0, queries, config.query_batch_size):
        stop = min(start + config.query_batch_size, queries)
        batch_state = query_state[start:stop]
        distance = batch_state @ reference_state.T
        distance.mul_(-2.0).add_(reference_norm)
        distance.add_(batch_state.square().sum(dim=1, keepdim=True))
        if leave_one_out_indices is not None:
            row = torch.arange(stop - start, device=device)
            distance[row, leave_one_out_indices[start:stop]] = torch.inf
        nearest_distance, nearest = torch.topk(
            distance,
            config.neighbors,
            dim=1,
            largest=False,
            sorted=True,
        )
        radius = nearest_distance[:, -1].clamp_min(0.0).sqrt()
        if return_neighbor_indices:
            neighbor_index_batches.append(nearest.detach().cpu().numpy())
            neighbor_radius_batches.append(radius.detach().cpu().numpy())
        radius_sum += float(radius.sum(dtype=torch.float64).item())
        radius_minimum = min(radius_minimum, float(radius.min().item()))
        radius_maximum = max(radius_maximum, float(radius.max().item()))
        neighbor_log_weight = reference_log_weights[nearest]
        normalized_weight = torch.softmax(neighbor_log_weight, dim=1)
        local_ess = normalized_weight.square().sum(dim=1).reciprocal()
        local_ess_sum += float(local_ess.sum(dtype=torch.float64).item())
        local_ess_minimum = min(
            local_ess_minimum, float(local_ess.min().item())
        )
        local_ess_maximum = max(
            local_ess_maximum, float(local_ess.max().item())
        )
        batch_any = torch.zeros(
            directions, stop - start, dtype=torch.bool, device=device
        )
        policy_score = torch.full(
            (stop - start,), -torch.inf, dtype=dtype, device=device
        )
        policy_value = torch.zeros_like(policy_score)
        policy_band = torch.zeros_like(policy_score)
        policy_direction = torch.full(
            (stop - start,), -1, dtype=torch.int64, device=device
        )
        policy_instrument = torch.full_like(policy_direction, -1)

        for left in range(0, instruments, config.instrument_batch_size):
            right = min(left + config.instrument_batch_size, instruments)
            neighbor_return = reference_returns[:, left:right][nearest]
            neighbor_cost = reference_cost[:, left:right][nearest]
            neighbor_band = reference_roundoff[:, left:right][nearest]
            directional = torch.stack(
                (
                    neighbor_return - neighbor_cost,
                    -neighbor_return - neighbor_cost,
                ),
                dim=0,
            )
            weight = normalized_weight[None, :, :, None]
            favorable = directional > neighbor_band[None]
            adverse = directional < -neighbor_band[None]
            favorable_probability = (
                weight * favorable.to(dtype)
            ).sum(dim=2)
            adverse_probability = (weight * adverse.to(dtype)).sum(dim=2)
            favorable_part = (
                weight * torch.relu(directional - neighbor_band[None])
            ).sum(dim=2)
            adverse_part = (
                weight * torch.relu(-directional - neighbor_band[None])
            ).sum(dim=2)
            favorable_probability_sum[:, left:right] += (
                favorable_probability.sum(dim=1, dtype=torch.float64)
            )
            adverse_probability_sum[:, left:right] += adverse_probability.sum(
                dim=1, dtype=torch.float64
            )
            favorable_part_sum[:, left:right] += favorable_part.sum(
                dim=1, dtype=torch.float64
            )
            adverse_part_sum[:, left:right] += adverse_part.sum(
                dim=1, dtype=torch.float64
            )
            one_sided = (~adverse.any(dim=2)) & favorable.any(dim=2)
            one_sided_count[:, left:right] += one_sided.sum(dim=1)
            batch_any |= one_sided.any(dim=2)

            means = (weight * directional).sum(dim=2)
            variances = (
                (weight * directional.square()).sum(dim=2) - means.square()
            ).clamp_min(0.0)
            thresholds = torch.as_tensor(
                config.signal_thresholds, dtype=dtype, device=device
            )[None, :, None, None]
            signal = means[:, None] > thresholds * variances.sqrt()[:, None]
            signal_count[:, :, left:right] += signal.sum(dim=2)

            for direction in range(directions):
                quantile, cvar = _weighted_lower_tail(
                    directional[direction],
                    normalized_weight[:, :, None].expand(
                        -1, -1, right - left
                    ),
                    probabilities=config.local_tail_probabilities,
                )
                quantile_sum[direction, :, left:right] += quantile.sum(
                    dim=0, dtype=torch.float64
                )
                cvar_sum[direction, :, left:right] += cvar.sum(
                    dim=0, dtype=torch.float64
                )

            query_directional = torch.stack(
                (
                    query_returns[start:stop, left:right]
                    - query_cost[start:stop, left:right],
                    -query_returns[start:stop, left:right]
                    - query_cost[start:stop, left:right],
                ),
                dim=0,
            )
            query_band = query_roundoff[start:stop, left:right]
            query_loss = query_directional < -query_band[None]
            query_gain = query_directional > query_band[None]
            realized_adverse[:, left:right] += (
                one_sided & query_loss
            ).sum(dim=1)

            for calibration_name, prediction, realized in (
                ("loss", adverse_probability, query_loss),
                ("gain", favorable_probability, query_gain),
            ):
                prediction_array = prediction.detach().cpu().numpy().reshape(-1)
                realized_array = (
                    realized.detach().cpu().numpy().reshape(-1).astype(np.float64)
                )
                bin_index = np.minimum(
                    (prediction_array * calibration_bins).astype(np.int64),
                    calibration_bins - 1,
                )
                calibration_count[calibration_name] += np.bincount(
                    bin_index, minlength=calibration_bins
                )
                calibration_prediction[calibration_name] += np.bincount(
                    bin_index,
                    weights=prediction_array,
                    minlength=calibration_bins,
                )
                calibration_realized[calibration_name] += np.bincount(
                    bin_index,
                    weights=realized_array,
                    minlength=calibration_bins,
                )
            residual = query_directional - means
            mean_residual_count += residual.numel()
            mean_residual_sum += float(residual.sum(dtype=torch.float64).item())
            mean_residual_square_sum += float(
                residual.square().sum(dtype=torch.float64).item()
            )
            outcome_interval_coverage += int(
                (
                    residual.abs()
                    <= 1.96 * variances.sqrt().clamp_min(torch.finfo(dtype).tiny)
                ).sum().item()
            )

            score = torch.where(
                one_sided,
                favorable_part,
                torch.full_like(favorable_part, -torch.inf),
            ).permute(1, 0, 2).reshape(stop - start, -1)
            best_score, flat = score.max(dim=1)
            best_direction = torch.div(flat, right - left, rounding_mode="floor")
            best_instrument = flat % (right - left)
            value = query_directional.permute(1, 0, 2).reshape(
                stop - start, -1
            ).gather(1, flat[:, None]).squeeze(1)
            band = query_band.gather(1, best_instrument[:, None]).squeeze(1)
            replace = best_score > policy_score
            policy_score = torch.where(replace, best_score, policy_score)
            policy_value = torch.where(replace, value, policy_value)
            policy_band = torch.where(replace, band, policy_band)
            policy_direction = torch.where(
                replace, best_direction, policy_direction
            )
            policy_instrument = torch.where(
                replace, left + best_instrument, policy_instrument
            )

        traded = torch.isfinite(policy_score)
        policy_trades += int(traded.sum().item())
        policy_losses += int((traded & (policy_value < -policy_band)).sum().item())
        policy_gains += int((traded & (policy_value > policy_band)).sum().item())
        traded_rows = torch.nonzero(traded, as_tuple=False).flatten()
        for row in traded_rows.tolist():
            screened_candidates.append(
                {
                    "query_index": start + row,
                    "direction": (
                        "long" if int(policy_direction[row].item()) == 0 else "short"
                    ),
                    "instrument": int(policy_instrument[row].item()),
                    "neighbor_radius": float(radius[row].item()),
                    "reference_score": float(policy_score[row].item()),
                }
            )
        long_state, short_state = batch_any
        long_only_states += int((long_state & ~short_state).sum().item())
        short_only_states += int((short_state & ~long_state).sum().item())
        both_states += int((long_state & short_state).sum().item())
        now = time.monotonic()
        if now - last_progress >= progress_interval_seconds or stop == queries:
            print(
                f"Cost-adjusted local tail {split}: {stop:,}/{queries:,}; "
                f"one-sided long={long_only_states:,}, short={short_only_states:,}, "
                f"both={both_states:,}",
                flush=True,
            )
            last_progress = now

    denominator = float(queries)

    def calibration_table(name: str) -> list[dict[str, Any]]:
        return [
            {
                "bin_lower": index / calibration_bins,
                "bin_upper": (index + 1) / calibration_bins,
                "observations": int(calibration_count[name][index]),
                "mean_predicted_probability": (
                    float(
                        calibration_prediction[name][index]
                        / calibration_count[name][index]
                    )
                    if calibration_count[name][index]
                    else None
                ),
                "realized_frequency": (
                    float(
                        calibration_realized[name][index]
                        / calibration_count[name][index]
                    )
                    if calibration_count[name][index]
                    else None
                ),
            }
            for index in range(calibration_bins)
        ]

    residual_mean = mean_residual_sum / mean_residual_count
    residual_variance = max(
        mean_residual_square_sum / mean_residual_count - residual_mean**2,
        0.0,
    )
    report = {
        "split": split,
        "queries": queries,
        "reference_observations": int(reference_state.shape[0]),
        "directions": ["long: dH-cost", "short: -dH-cost"],
        "cost_tested_separately_by_direction": True,
        "neighbors": config.neighbors,
        "tail_ranks": list(config.tail_ranks),
        "tail_probabilities": list(config.local_tail_probabilities),
        "mean_favorable_probability_by_direction_instrument": (
            favorable_probability_sum / denominator
        ).cpu().tolist(),
        "mean_adverse_probability_by_direction_instrument": (
            adverse_probability_sum / denominator
        ).cpu().tolist(),
        "mean_favorable_part_by_direction_instrument": (
            favorable_part_sum / denominator
        ).cpu().tolist(),
        "mean_adverse_part_by_direction_instrument": (
            adverse_part_sum / denominator
        ).cpu().tolist(),
        "mean_lower_quantile_by_direction_rank_instrument": (
            quantile_sum / denominator
        ).cpu().tolist(),
        "mean_lower_cvar_by_direction_rank_instrument": (
            cvar_sum / denominator
        ).cpu().tolist(),
        "one_sided_candidate_count_by_direction_instrument": (
            one_sided_count.cpu().tolist()
        ),
        "realized_adverse_count_by_direction_instrument": (
            realized_adverse.cpu().tolist()
        ),
        "signal_count_by_direction_threshold_instrument": (
            signal_count.cpu().tolist()
        ),
        "states_by_direction_exclusivity": {
            "long_only": long_only_states,
            "short_only": short_only_states,
            "both": both_states,
            "neither": queries - long_only_states - short_only_states - both_states,
        },
        "single_instrument_policy": {
            "trading_states": policy_trades,
            "realized_adverse_outcomes": policy_losses,
            "realized_favorable_outcomes": policy_gains,
        },
        "screened_candidates": screened_candidates,
        "held_out_calibration": {
            "applicability": (
                "independent audit calibration"
                if split.startswith("audit")
                else "reference leave-one-path-out diagnostic"
            ),
            "loss_probability_reliability": calibration_table("loss"),
            "gain_probability_reliability": calibration_table("gain"),
            "conditional_mean_residual": {
                "observations": mean_residual_count,
                "mean": residual_mean,
                "root_mean_square": math.sqrt(
                    residual_variance + residual_mean**2
                ),
            },
            "empirical_95_percent_outcome_interval_coverage": (
                outcome_interval_coverage / mean_residual_count
            ),
        },
        "neighbor_radius": {
            "minimum": radius_minimum,
            "mean": radius_sum / queries,
            "maximum": radius_maximum,
        },
        "local_effective_sample_size": {
            "minimum": local_ess_minimum,
            "mean": local_ess_sum / queries,
            "maximum": local_ess_maximum,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    if return_neighbor_indices:
        report["_neighbor_indices"] = np.concatenate(
            neighbor_index_batches, axis=0
        )
        report["_neighbor_radius"] = np.concatenate(
            neighbor_radius_batches, axis=0
        )
    return report


def run_conditional_local_tail(
    store: CompactLocalTailStore,
    *,
    device: torch.device | None = None,
    progress_interval_seconds: float = 30.0,
) -> dict[str, Any]:
    """Fit on the 80% reference paths and evaluate the 20% audit paths."""

    experiment = store.experiment
    arrays = store.arrays()
    state = arrays["state"]
    returns = arrays["returns"]
    roundoff = arrays["roundoff"]
    cost = arrays["cost"]
    net_roundoff = arrays["net_roundoff"]
    log_path_weight = arrays["log_path_weight"]
    component = arrays["importance_component"]
    S, steps, features = state.shape
    verify_inp(
        S == experiment.num_samples and S >= 2,
        lambda: f"local-tail split requires at least two paths, got {S}",
    )
    training_paths = experiment.reference_paths
    validation_paths = S - training_paths
    training_weight = _stable_path_weights(log_path_weight[:training_paths])
    mean, scale, constant = state_normalization(
        state[:training_paths],
        scale_floor=1.0e-6,
        path_weights=training_weight,
    )
    normalized_training = normalize_state(
        state[:training_paths], mean=mean, scale=scale
    ).reshape(training_paths * steps, features)
    normalized_validation = normalize_state(
        state[training_paths:], mean=mean, scale=scale
    ).reshape(validation_paths * steps, features)
    training_returns = np.asarray(returns[:training_paths]).reshape(
        training_paths * steps, -1
    )
    validation_returns = np.asarray(returns[training_paths:]).reshape(
        validation_paths * steps, -1
    )
    training_roundoff = np.asarray(roundoff[:training_paths]).reshape(
        training_paths * steps, -1
    )
    validation_roundoff = np.asarray(roundoff[training_paths:]).reshape(
        validation_paths * steps, -1
    )
    training_cost = np.asarray(cost[:training_paths]).reshape(
        training_paths * steps, -1
    )
    validation_cost = np.asarray(cost[training_paths:]).reshape(
        validation_paths * steps, -1
    )
    training_net_roundoff = np.asarray(net_roundoff[:training_paths]).reshape(
        training_paths * steps, -1
    )
    validation_net_roundoff = np.asarray(net_roundoff[training_paths:]).reshape(
        validation_paths * steps, -1
    )
    training_log_weight = np.repeat(log_path_weight[:training_paths], steps)

    importance_enabled = experiment.importance_sampling is not None
    training_eligible = (
        np.flatnonzero(component[:training_paths] == 0)
        if importance_enabled
        else None
    )
    validation_eligible = (
        np.flatnonzero(component[training_paths:] == 0)
        if importance_enabled
        else None
    )
    training_indices = _sample_indices(
        paths=training_paths,
        steps=steps,
        maximum=experiment.query_paths_per_split,
        seed=experiment.seed + 11,
        eligible_paths=training_eligible,
    )
    validation_indices = _sample_indices(
        paths=validation_paths,
        steps=steps,
        maximum=experiment.query_paths_per_split,
        seed=experiment.seed + 29,
        eligible_paths=validation_eligible,
    )
    reference_observations = training_paths * steps
    verify_inp(
        reference_observations >= 2,
        "local-tail diagnostics require at least two training observations",
    )
    effective_neighbors = min(
        experiment.neighbors, reference_observations - 1
    )
    effective_candidate_neighbors = min(
        experiment.candidate_neighbors, reference_observations - 1
    )
    config = LocalTailConfig(
        state_mode="raw",
        neighbors=effective_neighbors,
        instrument_candidate_neighbors=max(
            effective_neighbors, effective_candidate_neighbors
        ),
        tail_ranks=experiment.tail_ranks,
        signal_thresholds=experiment.signal_thresholds,
        query_batch_size=experiment.query_paths_per_split,
        instrument_batch_size=experiment.instrument_block_size,
        state_scale_floor=1.0e-6,
    )

    def valuate(target_device: torch.device) -> dict[str, Any]:
        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(
                np.array(value, dtype=np.float32, order="C", copy=True),
                dtype=torch.float32,
                device=target_device,
            )

        reference_state = tensor(normalized_training)
        reference_returns = tensor(training_returns)
        reference_roundoff = tensor(training_roundoff)
        reference_log_weights = tensor(training_log_weight)
        reference_cost = tensor(training_cost)
        reference_net_roundoff = tensor(training_net_roundoff)

        def progress(value: LocalTailProgress) -> None:
            print(
                f"PCA {value.split}: {value.processed_states:,}/"
                f"{value.total_states:,}; one-sided long="
                f"{value.long_only_candidate_states:,}, short="
                f"{value.short_only_candidate_states:,}, both="
                f"{value.both_direction_candidate_states:,}",
                flush=True,
            )

        training = valuate_local_tail(
            reference_state=reference_state,
            reference_returns=reference_returns,
            reference_roundoff=reference_roundoff,
            reference_log_weights=reference_log_weights,
            query_state=tensor(normalized_training[training_indices]),
            query_returns=tensor(training_returns[training_indices]),
            query_roundoff=tensor(training_roundoff[training_indices]),
            query_log_weights=(
                None
                if importance_enabled
                else tensor(training_log_weight[training_indices])
            ),
            config=config,
            split="training",
            leave_one_out=True,
            leave_one_out_reference_indices=torch.as_tensor(
                training_indices, dtype=torch.int64, device=target_device
            ),
            progress=progress,
            progress_interval_seconds=progress_interval_seconds,
        )
        validation = valuate_local_tail(
            reference_state=reference_state,
            reference_returns=reference_returns,
            reference_roundoff=reference_roundoff,
            reference_log_weights=reference_log_weights,
            query_state=tensor(normalized_validation[validation_indices]),
            query_returns=tensor(validation_returns[validation_indices]),
            query_roundoff=tensor(validation_roundoff[validation_indices]),
            query_log_weights=None,
            config=config,
            split="validation",
            leave_one_out=False,
            progress=progress,
            progress_interval_seconds=progress_interval_seconds,
        )
        cost_training = _cost_adjusted_local_directions(
            reference_state=reference_state,
            reference_returns=reference_returns,
            reference_cost=reference_cost,
            reference_roundoff=reference_net_roundoff,
            reference_log_weights=reference_log_weights,
            query_state=tensor(normalized_training[training_indices]),
            query_returns=tensor(training_returns[training_indices]),
            query_cost=tensor(training_cost[training_indices]),
            query_roundoff=tensor(training_net_roundoff[training_indices]),
            config=config,
            split="training",
            leave_one_out_indices=torch.as_tensor(
                training_indices, dtype=torch.int64, device=target_device
            ),
            progress_interval_seconds=progress_interval_seconds,
        )
        cost_validation = _cost_adjusted_local_directions(
            reference_state=reference_state,
            reference_returns=reference_returns,
            reference_cost=reference_cost,
            reference_roundoff=reference_net_roundoff,
            reference_log_weights=reference_log_weights,
            query_state=tensor(normalized_validation[validation_indices]),
            query_returns=tensor(validation_returns[validation_indices]),
            query_cost=tensor(validation_cost[validation_indices]),
            query_roundoff=tensor(validation_net_roundoff[validation_indices]),
            config=config,
            split="validation",
            leave_one_out_indices=None,
            progress_interval_seconds=progress_interval_seconds,
        )
        return {
            "training": training,
            "validation": validation,
            "cost_adjusted": {
                "cost_proportion": experiment.cost_proportion,
                "cost_definition": "cost_proportion * (abs(C) + abs(delta*S))",
                "training": cost_training,
                "validation": cost_validation,
            },
        }

    if device is not None:
        valuation = valuate(device)
        device_name = str(device)
    else:
        with DeviceLock() as lock:
            valuation = valuate(lock.device)
            device_name = str(lock.device)
    return {
        "status": "complete",
        "device": device_name,
        "configuration": asdict(config),
        "state": {
            "definition": "log-spot plus PCA surface scores",
            "training_only_mean": mean.tolist(),
            "training_only_population_scale": scale.tolist(),
            "constant_feature_indices": np.flatnonzero(constant).tolist(),
        },
        "split": {
            "training_paths": training_paths,
            "validation_paths": validation_paths,
            "training_queries": int(training_indices.size),
            "validation_queries": int(validation_indices.size),
            "training_queries_leave_one_path_out": True,
            "validation_queries_independent": True,
            "importance_query_paths": (
                "direct-P base-mixture paths" if importance_enabled else "direct-P paths"
            ),
            "reference_measure": "likelihood-weighted proposal paths",
        },
        **valuation,
    }


__all__ = ["CompactLocalTailStore", "run_conditional_local_tail"]
