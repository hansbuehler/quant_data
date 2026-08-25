"""Diagnose an expiry-causal allocation of a fitted Global FLEX state."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quantdata.dysanos.flex.failure_reporting import run_guarded_cli

_EXPLAINED_LEVELS = (0.90, 0.95, 0.99, 0.995, 0.999, 0.9999)
_RETENTION_LEVELS = (0.50, 0.70, 0.80, 0.90, 0.95, 0.99)
_RIDGE_LEVELS = (1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 0.1, 1.0, 10.0)
_GRID_ZERO_CANDIDATES = (10, 20, 30, 40)


@dataclass(frozen=True)
class _FoldFeatures:
    """One standardized feature fold and its reusable thin SVD."""

    train_indices: torch.Tensor
    validation_indices: torch.Tensor
    train_left: torch.Tensor
    singular_values: torch.Tensor
    right: torch.Tensor
    validation_features: torch.Tensor


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("phase") != "complete":
        raise ValueError(f"Expected a complete checkpoint, found {path}")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no model_state_dict: {path}")
    return state


def _rank_at_fraction(singular_values: torch.Tensor, fraction: float) -> int:
    energy = singular_values.square()
    total = torch.sum(energy)
    if singular_values.numel() == 0 or total <= 0.0:
        return 0
    cumulative = torch.cumsum(energy, dim=0) / total
    target = torch.as_tensor(
        fraction,
        dtype=cumulative.dtype,
        device=cumulative.device,
    )
    return int(torch.searchsorted(cumulative, target).item()) + 1


def _rank_spectrum(singular_values: torch.Tensor) -> dict[str, Any]:
    energy = singular_values.square()
    total = torch.sum(energy)
    if singular_values.numel() == 0 or total <= 0.0:
        return {
            "ranks": {str(level): 0 for level in _EXPLAINED_LEVELS},
            "top_10_fraction": 0.0,
            "smallest_to_largest_ratio": 0.0,
        }
    return {
        "ranks": {
            str(level): _rank_at_fraction(singular_values, level)
            for level in _EXPLAINED_LEVELS
        },
        "top_10_fraction": float(torch.sum(energy[:10]) / total),
        "smallest_to_largest_ratio": float(
            singular_values[-1] / singular_values[0]
        ),
    }


def _global_rank_diagnostics(
    centered_global_raw: torch.Tensor,
    *,
    block_width: int,
    num_expiries: int,
) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for expiry_index in range(num_expiries):
        start = expiry_index * block_width
        stop = start + block_width
        singular_values = torch.linalg.svdvals(
            centered_global_raw[:, start:stop]
        )
        blocks.append(
            {
                "expiry_index": expiry_index,
                **_rank_spectrum(singular_values),
            }
        )

    grid_zero = centered_global_raw[:, :block_width]
    left, singular_values, _ = torch.linalg.svd(
        grid_zero,
        full_matrices=False,
    )
    tolerance = (
        max(grid_zero.shape)
        * torch.finfo(grid_zero.dtype).eps
        * singular_values[0]
    )
    numerical_rank = int(torch.sum(singular_values > tolerance).item())
    observable_basis = left[:, :numerical_rank]
    surface_singular_values = torch.linalg.svdvals(
        observable_basis.transpose(0, 1) @ centered_global_raw
    )
    return {
        "individual_expiry_blocks": blocks,
        "grid_0_to_full_surface": {
            "grid_0_numerical_rank": numerical_rank,
            **_rank_spectrum(surface_singular_values),
        },
    }


def _balanced_allocation(
    grid_zero_size: int,
    *,
    total_size: int,
    num_expiries: int,
) -> tuple[int, ...]:
    remaining = total_size - grid_zero_size
    if grid_zero_size <= 0 or remaining < num_expiries - 1:
        raise ValueError(
            "Allocation must leave at least one state for every later expiry"
        )
    quotient, remainder = divmod(remaining, num_expiries - 1)
    return (grid_zero_size,) + tuple(
        quotient + (1 if index < remainder else 0)
        for index in range(num_expiries - 1)
    )


def _triangular_rrr_residual_coefficients(
    global_coefficients: torch.Tensor,
    *,
    allocation: tuple[int, ...],
    block_width: int,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    residual = global_coefficients.clone()
    selected = torch.empty(
        (global_coefficients.shape[0], 0),
        dtype=global_coefficients.dtype,
    )
    selected_sizes: list[int] = []
    tolerance = (
        max(global_coefficients.shape)
        * torch.finfo(global_coefficients.dtype).eps
        * torch.linalg.norm(global_coefficients)
    )
    for expiry_index, requested_rank in enumerate(allocation):
        start = expiry_index * block_width
        current = residual[:, start : start + block_width]
        left, singular_values, _ = torch.linalg.svd(
            current,
            full_matrices=False,
        )
        numerical_rank = int(torch.sum(singular_values > tolerance).item())
        available_rank = global_coefficients.shape[0] - selected.shape[1]
        rank = min(requested_rank, numerical_rank, available_rank)
        if rank == 0:
            selected_sizes.append(0)
            continue
        current_basis = left[:, :numerical_rank]
        descendants = residual[:, start:]
        predictive_left, _, _ = torch.linalg.svd(
            current_basis.transpose(0, 1) @ descendants,
            full_matrices=False,
        )
        new_states = current_basis @ predictive_left[:, :rank]
        selected = torch.cat((selected, new_states), dim=1)
        residual[:, start:] = descendants - new_states @ (
            new_states.transpose(0, 1) @ descendants
        )
        selected_sizes.append(rank)
    return residual, tuple(selected_sizes)


def _triangular_rrr_residual_fraction(
    global_coefficients: torch.Tensor,
    *,
    allocation: tuple[int, ...],
    block_width: int,
) -> tuple[float, tuple[int, ...]]:
    residual, selected_sizes = _triangular_rrr_residual_coefficients(
        global_coefficients,
        allocation=allocation,
        block_width=block_width,
    )
    base_energy = torch.sum(global_coefficients.square())
    return float(torch.sum(residual.square()) / base_energy), selected_sizes


def _coordinate_allocation_search(
    global_coefficients: torch.Tensor,
    *,
    initial_allocation: tuple[int, ...],
    block_width: int,
    max_iterations: int = 5,
) -> dict[str, Any]:
    allocation = initial_allocation
    objective, selected_sizes = _triangular_rrr_residual_fraction(
        global_coefficients,
        allocation=allocation,
        block_width=block_width,
    )
    history = [objective]
    for _ in range(max_iterations):
        removal_costs: dict[int, float] = {}
        for donor in range(1, len(allocation)):
            if allocation[donor] <= 1:
                continue
            candidate = list(allocation)
            candidate[donor] -= 1
            candidate_objective, _ = _triangular_rrr_residual_fraction(
                global_coefficients,
                allocation=tuple(candidate),
                block_width=block_width,
            )
            removal_costs[donor] = candidate_objective - objective

        addition_gains: dict[int, float] = {}
        for recipient in range(1, len(allocation)):
            candidate = list(allocation)
            candidate[recipient] += 1
            candidate_objective, _ = _triangular_rrr_residual_fraction(
                global_coefficients,
                allocation=tuple(candidate),
                block_width=block_width,
            )
            addition_gains[recipient] = objective - candidate_objective

        candidate_pairs = [
            (
                addition_gains[recipient] - removal_costs[donor],
                donor,
                recipient,
            )
            for donor in removal_costs
            for recipient in addition_gains
            if donor != recipient
        ]
        if not candidate_pairs:
            break
        _, donor, recipient = max(candidate_pairs)
        best_objective = objective
        best_allocation = allocation
        best_selected_sizes = selected_sizes
        # The marginal screen identifies a promising donor/recipient pair.
        # Line-search the integer transfer size so a long run of identical
        # one-state moves does not dominate analysis time.
        for transfer_size in range(1, allocation[donor]):
            candidate = list(allocation)
            candidate[donor] -= transfer_size
            candidate[recipient] += transfer_size
            candidate_allocation = tuple(candidate)
            candidate_objective, candidate_selected_sizes = (
                _triangular_rrr_residual_fraction(
                    global_coefficients,
                    allocation=candidate_allocation,
                    block_width=block_width,
                )
            )
            if candidate_objective < best_objective:
                best_objective = candidate_objective
                best_allocation = candidate_allocation
                best_selected_sizes = candidate_selected_sizes
        relative_improvement = (objective - best_objective) / max(
            objective,
            torch.finfo(global_coefficients.dtype).tiny,
        )
        if relative_improvement <= 1.0e-8:
            break
        allocation = best_allocation
        objective = best_objective
        selected_sizes = best_selected_sizes
        history.append(objective)
    return {
        "grid_0_states": allocation[0],
        "initial_allocation": list(initial_allocation),
        "optimized_allocation": list(allocation),
        "selected_numerical_ranks": list(selected_sizes),
        "residual_global_surface_variance_fraction": objective,
        "explained_global_surface_variance_fraction": 1.0 - objective,
        "num_exchange_iterations": len(history) - 1,
        "objective_history": history,
    }


def _conditional_grid_one_diagnostics(
    global_coefficients: torch.Tensor,
    *,
    grid_zero_sizes: tuple[int, ...],
    block_width: int,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    base_energy = torch.sum(global_coefficients.square())
    for grid_zero_size in grid_zero_sizes:
        residual = global_coefficients.clone()
        grid_zero = residual[:, :block_width]
        grid_zero_basis = torch.linalg.svd(
            grid_zero,
            full_matrices=False,
        ).U
        descendants = residual
        predictive_left, _, _ = torch.linalg.svd(
            grid_zero_basis.transpose(0, 1) @ descendants,
            full_matrices=False,
        )
        grid_zero_states = (
            grid_zero_basis @ predictive_left[:, :grid_zero_size]
        )
        residual = descendants - grid_zero_states @ (
            grid_zero_states.transpose(0, 1) @ descendants
        )

        grid_one = residual[:, block_width : 2 * block_width]
        grid_one_basis = torch.linalg.svd(
            grid_one,
            full_matrices=False,
        ).U
        grid_one_descendants = residual[:, block_width:]
        _, grid_one_singular_values, _ = torch.linalg.svd(
            grid_one_basis.transpose(0, 1) @ grid_one_descendants,
            full_matrices=False,
        )
        rows.append(
            {
                "grid_0_states": grid_zero_size,
                "residual_surface_variance_fraction_after_grid_0": float(
                    torch.sum(residual.square()) / base_energy
                ),
                "conditional_grid_1_predictive_ranks": _rank_spectrum(
                    grid_one_singular_values
                ),
            }
        )
    return tuple(rows)


def _contiguous_folds(num_rows: int, num_folds: int) -> tuple[np.ndarray, ...]:
    if not 2 <= num_folds <= num_rows:
        raise ValueError(
            f"Expected 2 <= num_folds <= num_rows, found {num_folds} and "
            f"{num_rows}"
        )
    return tuple(
        np.asarray(values, dtype=np.int64)
        for values in np.array_split(np.arange(num_rows), num_folds)
    )


def _prepare_fold_features(
    features: torch.Tensor,
    folds: tuple[np.ndarray, ...],
) -> tuple[_FoldFeatures, ...]:
    prepared: list[_FoldFeatures] = []
    all_indices = np.arange(features.shape[0])
    eps = torch.finfo(features.dtype).eps
    for validation_values in folds:
        train_values = np.setdiff1d(
            all_indices,
            validation_values,
            assume_unique=True,
        )
        train_indices = torch.as_tensor(train_values, dtype=torch.long)
        validation_indices = torch.as_tensor(
            validation_values,
            dtype=torch.long,
        )
        train = features[train_indices]
        validation = features[validation_indices]
        mean = torch.mean(train, dim=0)
        scale = torch.std(train, dim=0, correction=0)
        scale = torch.where(scale > eps, scale, torch.ones_like(scale))
        train = (train - mean) / scale
        validation = (validation - mean) / scale
        left, singular_values, right_t = torch.linalg.svd(
            train,
            full_matrices=False,
        )
        prepared.append(
            _FoldFeatures(
                train_indices=train_indices,
                validation_indices=validation_indices,
                train_left=left,
                singular_values=singular_values,
                right=right_t.transpose(0, 1).contiguous(),
                validation_features=validation,
            )
        )
    return tuple(prepared)


def _ridge_coefficients(
    fold: _FoldFeatures,
    target: torch.Tensor,
    *,
    relative_ridge: float,
) -> torch.Tensor:
    train_target = target[fold.train_indices]
    singular_values = fold.singular_values
    # Standardized feature cross-products are O(num_train). Scaling ridge by
    # that count makes the reported value comparable across fold sizes.
    ridge = relative_ridge * len(fold.train_indices)
    multiplier = singular_values / (singular_values.square() + ridge)
    return (
        fold.right
        @ (multiplier.unsqueeze(1) * (fold.train_left.transpose(0, 1) @ train_target))
    )


def _select_relative_ridge(
    prepared: tuple[_FoldFeatures, ...],
    target: torch.Tensor,
) -> tuple[float, tuple[dict[str, float], ...]]:
    rows: list[dict[str, float]] = []
    for relative_ridge in _RIDGE_LEVELS:
        prediction = torch.zeros_like(target)
        for fold in prepared:
            coefficients = _ridge_coefficients(
                fold,
                target,
                relative_ridge=relative_ridge,
            )
            prediction[fold.validation_indices] = (
                fold.validation_features @ coefficients
            )
        mse = torch.mean((prediction - target).square())
        rows.append(
            {
                "relative_ridge": relative_ridge,
                "oof_mse": float(mse),
            }
        )
    best = min(rows, key=lambda row: row["oof_mse"])
    return float(best["relative_ridge"]), tuple(rows)


def _oof_reduced_rank_curve(
    prepared: tuple[_FoldFeatures, ...],
    target: torch.Tensor,
    *,
    relative_ridge: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_rows, target_size = target.shape
    predictions = torch.zeros(
        (target_size + 1, num_rows, target_size),
        dtype=target.dtype,
    )
    for fold in prepared:
        coefficients = _ridge_coefficients(
            fold,
            target,
            relative_ridge=relative_ridge,
        )
        train_prediction = (
            (
                (fold.train_left * fold.singular_values.unsqueeze(0))
                @ fold.right.transpose(0, 1)
            )
            @ coefficients
        )
        _, _, target_right_t = torch.linalg.svd(
            train_prediction,
            full_matrices=False,
        )
        validation_prediction = fold.validation_features @ coefficients
        cumulative = torch.zeros_like(validation_prediction)
        predictions[0, fold.validation_indices] = cumulative
        for rank, direction in enumerate(target_right_t, start=1):
            projection = validation_prediction @ direction
            cumulative = cumulative + projection.unsqueeze(1) * direction
            predictions[rank, fold.validation_indices] = cumulative

    residual = predictions - target.unsqueeze(0)
    sse = torch.sum(residual.square(), dim=(1, 2))
    return predictions, sse


def _selected_rank(sse: torch.Tensor, retention: float) -> tuple[int, int]:
    baseline = sse[0]
    best_rank = int(torch.argmin(sse).item())
    achievable_gain = baseline - sse[best_rank]
    if best_rank == 0 or achievable_gain <= 0.0:
        return 0, best_rank
    retained_target = baseline - retention * achievable_gain
    eligible = torch.nonzero(sse[: best_rank + 1] <= retained_target)
    return int(eligible[0].item()), best_rank


def _sequential_policy(
    prepared_by_expiry: tuple[tuple[_FoldFeatures, ...], ...],
    global_scores: torch.Tensor,
    *,
    relative_ridge: float,
    retention: float,
) -> dict[str, Any]:
    residual = global_scores.clone()
    total_prediction = torch.zeros_like(global_scores)
    initial_energy = torch.sum(global_scores.square())
    allocations: list[int] = []
    steps: list[dict[str, Any]] = []
    for expiry_index, prepared in enumerate(prepared_by_expiry):
        predictions, sse = _oof_reduced_rank_curve(
            prepared,
            residual,
            relative_ridge=relative_ridge,
        )
        selected_rank, best_rank = _selected_rank(sse, retention)
        selected_prediction = predictions[selected_rank]
        before = torch.sum(residual.square())
        residual = residual - selected_prediction
        total_prediction = total_prediction + selected_prediction
        after = torch.sum(residual.square())
        allocations.append(selected_rank)
        steps.append(
            {
                "expiry_index": expiry_index,
                "selected_rank": selected_rank,
                "best_oof_rank": best_rank,
                "residual_energy_before": float(before),
                "residual_energy_after": float(after),
                "selected_relative_reduction": float(
                    (before - after) / before if before > 0.0 else 0.0
                ),
                "best_relative_reduction": float(
                    (sse[0] - sse[best_rank]) / sse[0]
                    if sse[0] > 0.0
                    else 0.0
                ),
            }
        )
    explained = 1.0 - torch.sum(residual.square()) / initial_energy
    return {
        "retention": retention,
        "allocation": allocations,
        "total_states": sum(allocations),
        "oof_global_surface_variance_explained": float(explained),
        "steps": steps,
        "oof_score_prediction": total_prediction,
    }


def _json_policy(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in policy.items()
        if key != "oof_score_prediction"
    }


def analyze_state_allocation(
    *,
    direct_checkpoint: Path,
    global_checkpoint: Path,
    num_folds: int,
    grid_zero_candidates: tuple[int, ...] = _GRID_ZERO_CANDIDATES,
    exchange_iterations: int = 5,
) -> dict[str, Any]:
    """Run Global/grid-0 spectra and sequential out-of-fold rank diagnostics."""
    if len(grid_zero_candidates) == 0:
        raise ValueError("At least one grid-zero candidate is required")
    if exchange_iterations < 0:
        raise ValueError(
            f"exchange_iterations must be non-negative, found {exchange_iterations}"
        )
    direct_state = _checkpoint_state(direct_checkpoint)
    global_state = _checkpoint_state(global_checkpoint)
    direct_raw = direct_state["daily_raw_embedding.weight"].double()
    global_daily = global_state["daily_state_embedding.weight"].double()
    global_weight = global_state["raw_decoder.I0.weight"].double()
    global_bias = global_state["raw_decoder.I0.bias"].double()
    if direct_raw.shape[0] != global_daily.shape[0]:
        raise ValueError(
            "Direct and Global checkpoints contain different numbers of days: "
            f"{direct_raw.shape[0]} versus {global_daily.shape[0]}"
        )
    if direct_raw.shape[1] != global_weight.shape[0]:
        raise ValueError(
            "Direct state and Global decoder geometries differ: "
            f"{direct_raw.shape[1]} versus {global_weight.shape[0]}"
        )
    num_expiries = 14
    if direct_raw.shape[1] % num_expiries != 0:
        raise ValueError(
            f"Native state size {direct_raw.shape[1]} is not divisible by 14"
        )
    block_width = direct_raw.shape[1] // num_expiries
    maximum_grid_zero_size = global_daily.shape[1] - (num_expiries - 1)
    if any(
        not 1 <= grid_zero_size <= maximum_grid_zero_size
        for grid_zero_size in grid_zero_candidates
    ):
        raise ValueError(
            "Every grid-zero candidate must leave one state for each later "
            f"expiry; expected values in [1,{maximum_grid_zero_size}], found "
            f"{grid_zero_candidates}"
        )
    global_raw = global_daily @ global_weight.transpose(0, 1) + global_bias
    global_mean = torch.mean(global_raw, dim=0)
    centered_global_raw = global_raw - global_mean
    left, singular_values, right_t = torch.linalg.svd(
        centered_global_raw,
        full_matrices=False,
    )
    global_rank = global_daily.shape[1]
    global_scores = left[:, :global_rank] * singular_values[:global_rank]
    global_loadings = right_t[:global_rank]
    global_coefficients = (
        singular_values[:global_rank].unsqueeze(1) * global_loadings
    )

    folds = _contiguous_folds(len(direct_raw), num_folds)
    prepared_by_expiry = tuple(
        _prepare_fold_features(
            direct_raw[
                :,
                expiry_index * block_width : (expiry_index + 1) * block_width,
            ],
            folds,
        )
        for expiry_index in range(num_expiries)
    )
    relative_ridge, ridge_diagnostics = _select_relative_ridge(
        prepared_by_expiry[0],
        global_scores,
    )
    policies = tuple(
        _sequential_policy(
            prepared_by_expiry,
            global_scores,
            relative_ridge=relative_ridge,
            retention=retention,
        )
        for retention in _RETENTION_LEVELS
    )
    ideal_triangular_allocations = tuple(
        _coordinate_allocation_search(
            global_coefficients,
            initial_allocation=_balanced_allocation(
                grid_zero_size,
                total_size=global_rank,
                num_expiries=num_expiries,
            ),
            block_width=block_width,
            max_iterations=exchange_iterations,
        )
        for grid_zero_size in grid_zero_candidates
    )
    return {
        "schema": "flex-expiry-state-allocation-diagnostic-v1",
        "direct_checkpoint": str(direct_checkpoint),
        "global_checkpoint": str(global_checkpoint),
        "num_days": int(direct_raw.shape[0]),
        "num_expiries": num_expiries,
        "expiry_block_width": block_width,
        "global_state_size": global_rank,
        "num_contiguous_oof_folds": num_folds,
        "global_raw_mean": global_mean,
        "global_score_loadings": global_loadings,
        "global_rank_diagnostics": _global_rank_diagnostics(
            centered_global_raw,
            block_width=block_width,
            num_expiries=num_expiries,
        ),
        "conditional_grid_1_diagnostics": list(
            _conditional_grid_one_diagnostics(
                global_coefficients,
                grid_zero_sizes=grid_zero_candidates,
                block_width=block_width,
            )
        ),
        "selected_relative_ridge": relative_ridge,
        "ridge_diagnostics": list(ridge_diagnostics),
        "sequential_policies": [_json_policy(policy) for policy in policies],
        "ideal_global_surface_triangular_allocations": list(
            ideal_triangular_allocations
        ),
        "exchange_iterations": exchange_iterations,
    }


def _write_outputs(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    json_report = {
        key: value
        for key, value in report.items()
        if not isinstance(value, torch.Tensor)
    }
    output.write_text(
        json.dumps(json_report, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = [
        "# FLEX expiry-state allocation diagnostic",
        "",
        f"Days: `{report['num_days']}`. Global state: `{report['global_state_size']}`. ",
        f"Contiguous out-of-fold partitions: `{report['num_contiguous_oof_folds']}`.",
        "",
        "## Grid 0 rank spectrum",
        "",
        "| Target | 90% | 95% | 99% | 99.5% | 99.9% | 99.99% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    diagnostics = report["global_rank_diagnostics"]
    grid_0 = diagnostics["individual_expiry_blocks"][0]["ranks"]
    full = diagnostics["grid_0_to_full_surface"]["ranks"]
    level_keys = [str(level) for level in _EXPLAINED_LEVELS]
    markdown.append(
        "| Grid 0 itself | "
        + " | ".join(str(grid_0[key]) for key in level_keys)
        + " |"
    )
    markdown.append(
        "| Full Global surface observable through grid 0 | "
        + " | ".join(str(full[key]) for key in level_keys)
        + " |"
    )
    markdown.extend(
        [
            "",
            "## Sequential Direct-grid to Global-state diagnostic",
            "",
            f"Selected relative ridge: `{report['selected_relative_ridge']}`.",
            "",
            "| Retained predictable gain per step | Total states | OOF surface variance explained | Allocation, grid 0 to 13 |",
            "|---:|---:|---:|---|",
        ]
    )
    for policy in report["sequential_policies"]:
        markdown.append(
            f"| {100.0 * policy['retention']:.0f}% | "
            f"{policy['total_states']} | "
            f"{100.0 * policy['oof_global_surface_variance_explained']:.4f}% | "
            f"`{policy['allocation']}` |"
        )
    markdown.extend(
        [
            "",
            "These out-of-fold allocations are a rejection diagnostic, not a "
            "recommended budget: even all Direct expiry blocks recover only "
            "about 63% of the independently fine-tuned Global surface through "
            "this linear map. Raw Direct and Global coordinates therefore do "
            "not share a sufficiently stable gauge.",
            "",
            "## Conditional grid 1 rank after grid 0",
            "",
            "| Grid 0 states | Residual Global variation | Grid 1 rank for 90% | 95% | 99% |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["conditional_grid_1_diagnostics"]:
        ranks = row["conditional_grid_1_predictive_ranks"]["ranks"]
        markdown.append(
            f"| {row['grid_0_states']} | "
            f"{100.0 * row['residual_surface_variance_fraction_after_grid_0']:.4f}% | "
            f"{ranks[str(0.9)]} | {ranks[str(0.95)]} | {ranks[str(0.99)]} |"
        )
    markdown.extend(
        [
            "",
            "## Ideal triangular approximation of the fitted Global surface",
            "",
            "Grid 0 is fixed at each candidate size. One-state exchanges then "
            "optimize the remaining 140-state budget against decoded Global "
            "raw-surface variation.",
            "",
            "| Grid 0 states | Explained Global variation | Optimized allocation |",
            "|---:|---:|---|",
        ]
    )
    for candidate in report["ideal_global_surface_triangular_allocations"]:
        markdown.append(
            f"| {candidate['grid_0_states']} | "
            f"{100.0 * candidate['explained_global_surface_variance_fraction']:.4f}% | "
            f"`{candidate['optimized_allocation']}` |"
        )
    output.with_suffix(".md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-checkpoint", type=Path, required=True)
    parser.add_argument("--global-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument(
        "--grid-zero-candidates",
        type=int,
        nargs="+",
        default=list(_GRID_ZERO_CANDIDATES),
    )
    parser.add_argument("--exchange-iterations", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = analyze_state_allocation(
        direct_checkpoint=args.direct_checkpoint,
        global_checkpoint=args.global_checkpoint,
        num_folds=args.num_folds,
        grid_zero_candidates=tuple(args.grid_zero_candidates),
        exchange_iterations=args.exchange_iterations,
    )
    _write_outputs(report, args.output)
    print(args.output)


if __name__ == "__main__":
    run_guarded_cli(main, program="analyze_state_allocation")
