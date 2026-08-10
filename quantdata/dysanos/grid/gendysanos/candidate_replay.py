"""Replay a frozen conditional basket on an independent matched universe."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from cdxcore.err import verify_inp
from scipy.stats import norm

from .matched_cache import MatchedUniverseBlockCache
from .required_runs import (
    clopper_pearson_one_sided,
    derive_case_arrays,
    positive_relative_weights,
    propagated_basket_band,
)


def summarize_frozen_replay(
    *,
    activated_pnl: np.ndarray,
    activated_band: np.ndarray,
    in_region: np.ndarray,
    log_path_weights: np.ndarray,
    importance_components: np.ndarray,
    family_size: int,
    prerequisite_checks_passed: bool,
) -> dict[str, Any]:
    """Summarize an independently replayed state-activated basket."""

    pnl = np.asarray(activated_pnl, dtype=np.float64)
    band = np.asarray(activated_band, dtype=np.float64)
    region = np.asarray(in_region, dtype=bool)
    log_weights = np.asarray(log_path_weights, dtype=np.float64)
    components = np.asarray(importance_components)
    verify_inp(
        pnl.shape
        == band.shape
        == region.shape
        == log_weights.shape
        == components.shape,
        "frozen replay arrays have inconsistent shapes",
    )
    verify_inp(pnl.ndim == 1 and pnl.size > 0, "frozen replay is empty")
    verify_inp(family_size >= 1, "frozen replay family size must be positive")
    if not all(np.all(np.isfinite(value)) for value in (pnl, band, log_weights)):
        raise FloatingPointError("frozen replay contains NaN or infinity")

    direct = components == 0
    direct_paths = int(np.count_nonzero(direct))
    verify_inp(direct_paths > 0, "frozen replay has no direct-P paths")
    loss = pnl < -band
    gain = pnl > band
    adjusted_alpha = 0.05 / family_size
    z = float(norm.ppf(1.0 - adjusted_alpha))

    direct_loss = int(np.count_nonzero(loss & direct))
    direct_gain = int(np.count_nonzero(gain & direct))
    gain_lower, _ = clopper_pearson_one_sided(
        direct_gain, direct_paths, alpha=adjusted_alpha
    )
    _, loss_upper = clopper_pearson_one_sided(
        direct_loss, direct_paths, alpha=adjusted_alpha
    )
    direct_values = pnl[direct]
    direct_mean = float(np.mean(direct_values))
    direct_error = float(np.std(direct_values, ddof=0) / math.sqrt(direct_paths))

    weights = positive_relative_weights(log_weights)
    mass = float(weights.sum())
    weighted_mean = float(np.dot(weights, pnl) / mass)
    weighted_error = float(
        np.sqrt(np.dot(np.square(weights), np.square(pnl - weighted_mean))) / mass
    )
    gain_value = gain.astype(np.float64)
    weighted_gain = float(np.dot(weights, gain_value) / mass)
    weighted_gain_error = float(
        np.sqrt(
            np.dot(np.square(weights), np.square(gain_value - weighted_gain))
        )
        / mass
    )
    weighted_gain_lower = weighted_gain - z * weighted_gain_error
    direct_mean_lower = direct_mean - z * direct_error
    weighted_mean_lower = weighted_mean - z * weighted_error
    economic_conditions = {
        "no_target_loss_beyond_numerical_band": int(np.count_nonzero(loss)) == 0,
        "adjusted_global_direct_P_gain_lower_positive": gain_lower > 0.0,
        "adjusted_global_weighted_P_gain_lower_positive": weighted_gain_lower > 0.0,
        "target_prerequisite_checks_passed": prerequisite_checks_passed,
    }
    additional_mean_conditions = {
        "adjusted_global_direct_P_mean_lower_positive": direct_mean_lower > 0.0,
        "adjusted_global_weighted_P_mean_lower_positive": weighted_mean_lower > 0.0,
    }
    region_pnl = pnl[region]
    return {
        "definition": (
            "the source action, source normalization, and source normalized "
            "state ball are frozen before replay on the independent target universe"
        ),
        "paths": int(pnl.size),
        "direct_P_paths": direct_paths,
        "paths_in_region": int(np.count_nonzero(region)),
        "direct_P_paths_in_region": int(np.count_nonzero(region & direct)),
        "loss_count": int(np.count_nonzero(loss)),
        "neutral_count": int(np.count_nonzero(~loss & ~gain)),
        "gain_count": int(np.count_nonzero(gain)),
        "direct_P_loss_count": direct_loss,
        "direct_P_gain_count": direct_gain,
        "minimum_region_economic_pnl": (
            float(np.min(region_pnl)) if region_pnl.size else None
        ),
        "maximum_region_economic_pnl": (
            float(np.max(region_pnl)) if region_pnl.size else None
        ),
        "adjusted_global_direct_P_loss_probability_upper": loss_upper,
        "adjusted_global_direct_P_gain_probability_lower": gain_lower,
        "global_direct_P_mean": direct_mean,
        "global_direct_P_mean_standard_error": direct_error,
        "adjusted_global_direct_P_mean_lower": direct_mean_lower,
        "global_weighted_P_gain_probability": weighted_gain,
        "global_weighted_P_gain_standard_error": weighted_gain_error,
        "adjusted_global_weighted_P_gain_probability_lower": weighted_gain_lower,
        "global_weighted_P_mean": weighted_mean,
        "global_weighted_P_mean_standard_error": weighted_error,
        "adjusted_global_weighted_P_mean_lower": weighted_mean_lower,
        "multiplicity": {
            "method": "Bonferroni familywise correction",
            "family_size": family_size,
            "adjusted_alpha": adjusted_alpha,
        },
        "economic_conditions": economic_conditions,
        "economic_certified": bool(all(economic_conditions.values())),
        "additional_mean_conditions": additional_mean_conditions,
        "additional_mean_certified": bool(all(additional_mean_conditions.values())),
    }


def replay_report_candidate(
    *,
    source_report: dict[str, Any],
    target_report: dict[str, Any],
    candidate_index: int = 0,
) -> dict[str, Any]:
    """Replay one source local-basket survivor on the target report's cache."""

    candidates = source_report["candidate_certification"][
        "local_conditional_baskets"
    ]["surviving"]
    verify_inp(
        0 <= candidate_index < len(candidates),
        lambda: f"invalid source candidate index {candidate_index}",
    )
    candidate = candidates[candidate_index]
    time_index = int(candidate["trading_time_index"])
    normalization = source_report["conditional"]["times"][time_index][
        "state_normalization"
    ]
    mean = np.asarray(normalization["reference_only_mean"], dtype=np.float64)
    scale = np.asarray(
        normalization["reference_only_population_scale"], dtype=np.float64
    )
    center = np.asarray(candidate["state_region"]["center"], dtype=np.float64)
    radius = float(candidate["state_region"]["radius"])
    action = np.asarray(candidate["reference_solution"]["action"], dtype=np.float64)
    source_configuration = source_report["configuration"]
    target_configuration = target_report["configuration"]
    verify_inp(
        source_configuration["generator"] == target_configuration["generator"]
        and source_configuration["trading_frequency"]
        == target_configuration["trading_frequency"]
        and source_configuration["return_horizon"]
        == target_configuration["return_horizon"]
        and source_configuration["hedge"] == target_configuration["hedge"],
        "source and target reports describe different markets",
    )

    cache_directory = Path(target_report["base_cache"])
    manifest = json.loads(
        (cache_directory / "manifest.json").read_text(encoding="utf-8")
    )
    cache = MatchedUniverseBlockCache(
        cache_directory.parent,
        cache_uid=manifest["cache_uid"],
        configuration=manifest["configuration"],
        num_samples=int(manifest["num_samples"]),
    )
    pnl_parts: list[np.ndarray] = []
    band_parts: list[np.ndarray] = []
    region_parts: list[np.ndarray] = []
    log_weight_parts: list[np.ndarray] = []
    component_parts: list[np.ndarray] = []
    for block in cache.iter_blocks():
        universe = getattr(block, source_configuration["return_horizon"])
        case = derive_case_arrays(
            universe,
            hedge=source_configuration["hedge"],
            cost_rate=float(source_configuration["cost_rate"]),
        )
        state = np.asarray(universe["xstate"][:, time_index], dtype=np.float64)
        normalized = (state - mean) / scale
        in_region = np.linalg.norm(normalized - center, axis=1) <= radius
        activated_pnl = np.zeros(state.shape[0], dtype=np.float64)
        activated_band = np.zeros(state.shape[0], dtype=np.float64)
        if np.any(in_region):
            returns = np.asarray(case.returns[in_region, time_index], dtype=np.float64)
            costs = np.asarray(case.costs[in_region, time_index], dtype=np.float64)
            bands = np.asarray(
                case.numerical_band[in_region, time_index], dtype=np.float64
            )
            activated_pnl[in_region] = returns @ action - costs @ np.abs(action)
            activated_band[in_region] = propagated_basket_band(
                action, returns, costs, bands
            )
        pnl_parts.append(activated_pnl)
        band_parts.append(activated_band)
        region_parts.append(in_region)
        log_weight_parts.append(
            np.asarray(universe["log_path_weight"], dtype=np.float64)
        )
        component_parts.append(
            np.asarray(universe["importance_component"], dtype=np.int16)
        )

    summary = summarize_frozen_replay(
        activated_pnl=np.concatenate(pnl_parts),
        activated_band=np.concatenate(band_parts),
        in_region=np.concatenate(region_parts),
        log_path_weights=np.concatenate(log_weight_parts),
        importance_components=np.concatenate(component_parts),
        family_size=48 * 10 * 1_024,
        prerequisite_checks_passed=(
            target_report.get("functional_gate", {}).get("passed") is True
        ),
    )
    return {
        "status": "complete",
        "source_configuration": source_configuration,
        "target_configuration": target_configuration,
        "source_candidate_index": candidate_index,
        "source_query_index": candidate["query_index"],
        "trading_time_index": time_index,
        "source_state_region": candidate["state_region"],
        "source_action_l1_norm": float(np.abs(action).sum()),
        "uses_target_outcomes_to_select_action_or_region": False,
        "summary": summary,
    }


def main() -> None:
    """Replay a report survivor and write a machine-readable result."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--target-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, default=0)
    args = parser.parse_args()
    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    target = json.loads(args.target_report.read_text(encoding="utf-8"))
    result = replay_report_candidate(
        source_report=source,
        target_report=target,
        candidate_index=args.candidate_index,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(result["summary"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
