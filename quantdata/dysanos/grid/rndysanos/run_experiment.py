"""Run reproducible compiled-GPU OCE measure-change experiments.

Examples
--------
Small mechanics run::

    python -m quantdata.dysanos.grid.rndysanos.run_experiment \
        --paths 1000 --epochs 100 --stages 1 --regimes conditional --restart

Full comparison::

    python -m quantdata.dysanos.grid.rndysanos.run_experiment \
        --paths 100000 --epochs 100 --stages 3 --regimes both --restart
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import torch

from quantdata.torch import BaseEnvironment, DeviceLock

from .torch_drift_change import DysanosConfig, RealizedDensity, TrainConfig, train
from .torch_prep import RNDysanosExperiment, prep_experiment

_REGIMES = {
    "costfree": {"cost_cond": 0.0, "cost_uncond": 0.0, "cost_prop": 0.0},
    "unconditional": {"cost_cond": 0.0, "cost_uncond": 1.0, "cost_prop": 0.0001},
    "conditional": {"cost_cond": 1.0, "cost_uncond": 0.0, "cost_prop": 0.0001},
}


def _proc_memory() -> dict[str, float]:
    fields = {"VmRSS": "rss_gib", "VmHWM": "peak_rss_gib"}
    output: dict[str, float] = {}
    with Path("/proc/self/status").open(encoding="utf-8") as stream:
        for line in stream:
            name, _, value = line.partition(":")
            if name in fields:
                output[fields[name]] = float(value.split()[0]) / (1024.0**2)
    return output


def memory_snapshot(device: torch.device) -> dict[str, float]:
    """Return current/peak process and CUDA memory in GiB."""

    output = _proc_memory()
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        output.update(
            cuda_allocated_gib=torch.cuda.memory_allocated(device) / (1024.0**3),
            cuda_reserved_gib=torch.cuda.memory_reserved(device) / (1024.0**3),
            cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / (1024.0**3),
            cuda_peak_reserved_gib=torch.cuda.max_memory_reserved(device) / (1024.0**3),
            cuda_free_gib=free / (1024.0**3),
            cuda_total_gib=total / (1024.0**3),
        )
    return output


def _dynamo_counters() -> dict[str, int]:
    try:
        from torch._dynamo.utils import counters

        return {
            "frames_total": int(counters["frames"]["total"]),
            "frames_ok": int(counters["frames"]["ok"]),
            "unique_graphs": int(counters["stats"]["unique_graphs"]),
            "graph_breaks": int(sum(counters["graph_break"].values())),
        }
    except (ImportError, KeyError, TypeError):
        return {}


def _counter_difference(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {name: value - before.get(name, 0) for name, value in after.items()}


def stage_universes(
    prepared: list[dict[str, np.ndarray]],
    *,
    stages: int,
    same_universe: bool,
) -> list[dict[str, np.ndarray]]:
    """Select independent universes or reuse one exact universe at every stage."""

    if stages <= 0:
        raise ValueError("'stages' must be positive")
    required = 1 if same_universe else stages
    if len(prepared) != required:
        raise ValueError(f"expected {required} prepared universes, got {len(prepared)}")
    return [prepared[0]] * stages if same_universe else list(prepared)


def pair_training_validation_universes(
    prepared: list[dict[str, np.ndarray]],
    *,
    consume: bool = False,
) -> list[dict[str, np.ndarray]]:
    """Pair independent equal-size universes into 50/50 train/validation data.

    ``consume=True`` releases each source pair as soon as its combined universe
    is allocated.  This keeps peak host memory near the final paired-data size
    for full multi-stage experiments.
    """

    if len(prepared) == 0 or len(prepared) % 2:
        raise ValueError("an even positive number of source universes is required")
    sources = prepared if consume else list(prepared)
    paired: list[dict[str, np.ndarray]] = []
    while sources:
        training = sources.pop(0)
        validation = sources.pop(0)
        if training.keys() != validation.keys():
            raise ValueError("paired universes have different fields")
        combined: dict[str, np.ndarray] = {}
        training_paths = training["dS"].shape[0]
        validation_paths = validation["dS"].shape[0]
        if training_paths != validation_paths:
            raise ValueError(
                "paired universes must have the same path count; found "
                f"{training_paths} and {validation_paths}"
            )
        if np.array_equal(training["dS"], validation["dS"]) and np.array_equal(
            training["xstate"], validation["xstate"]
        ):
            raise ValueError(
                "paired training and validation universes are exact duplicates"
            )
        for name in training:
            left = np.asarray(training[name])
            right = np.asarray(validation[name])
            if left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError(
                    f"paired field {name!r} differs: {left.shape}/{left.dtype} and "
                    f"{right.shape}/{right.dtype}"
                )
            if left.ndim > 0 and left.shape[0] == training_paths:
                combined[name] = np.concatenate((left, right), axis=0)
            else:
                if not np.array_equal(left, right):
                    raise ValueError(
                        f"non-path field {name!r} differs between paired universes"
                    )
                combined[name] = left
        paired.append(combined)
    return paired


def attach_arb_direction_mask(
    universe: dict[str, np.ndarray],
    *,
    report_path: str | Path,
    validation_fraction: float,
    universe_seed_offset: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Attach a saved cost-aware ``arb(z; omega)`` direction mask.

    The local-arbitrage files contain option directions only and are packed
    along the option axis.  This function restores them in the OCE convention
    (long instruments followed by short instruments), leaving spot active.
    An all-zero mask is represented without allocating a large boolean array.
    """

    path = Path(report_path)
    mask_report = json.loads(path.read_text(encoding="utf-8"))
    if mask_report.get("status") != "complete":
        raise ValueError(f"arbitrage-mask report {path} is not complete")
    paths, steps = np.asarray(universe["dS"]).shape
    options = int(np.prod(np.asarray(universe["dC"]).shape[-2:]))
    instruments = options + 1
    validation_paths = int(math.floor(paths * validation_fraction))
    if validation_fraction > 0.0 and paths > 1:
        validation_paths = max(1, validation_paths)
    validation_paths = min(validation_paths, paths - 1)
    training_paths = paths - validation_paths
    expected = {
        "paths": paths,
        "training_paths": training_paths,
        "validation_paths": validation_paths,
        "universe_seed_offset": universe_seed_offset,
    }
    for name, value in expected.items():
        if int(mask_report.get(name, -1)) != value:
            raise ValueError(
                f"arbitrage-mask report {name}={mask_report.get(name)!r}; "
                f"current universe requires {value}"
            )
    geometry = mask_report.get("geometry", {})
    if int(geometry.get("steps", -1)) != steps:
        raise ValueError(
            f"arbitrage-mask report steps={geometry.get('steps')!r}; expected {steps}"
        )

    counts = {
        split: [int(value) for value in mask_report[split]["candidate_count_by_direction"]]
        for split in ("training", "validation")
    }
    candidate_count = sum(sum(value) for value in counts.values())
    metadata: dict[str, Any] = {
        "mode": "arb-masked",
        "source_report": str(path),
        "candidate_count_by_split_and_direction": counts,
        "candidate_count": candidate_count,
        "candidate_definition": mask_report["training"]["candidate_definition"],
        "source_n_cost": int(mask_report["n_cost"]),
        "current_n_cost": training_paths,
        "empty_mask_optimized": candidate_count == 0,
    }
    if candidate_count == 0:
        return dict(universe), metadata

    split_sizes = {"training": training_paths, "validation": validation_paths}
    unpacked: list[np.ndarray] = []
    for split in ("training", "validation"):
        packed_path = Path(mask_report["mask_files"][split])
        packed = np.load(packed_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (split_sizes[split] * steps, 2, math.ceil(options / 8))
        if packed.shape != expected_shape or packed.dtype != np.uint8:
            raise ValueError(
                f"packed {split} mask must be {expected_shape}/uint8, got "
                f"{packed.shape}/{packed.dtype}"
            )
        bits = np.unpackbits(packed, axis=-1, count=options, bitorder="little")
        unpacked.append(bits.reshape(split_sizes[split], steps, 2, options).astype(bool))

    option_mask = np.concatenate(unpacked, axis=0)
    excluded = np.zeros((paths, steps, 2 * instruments), dtype=bool)
    excluded[..., 1:instruments] = option_mask[..., 0, :]
    excluded[..., instruments + 1 :] = option_mask[..., 1, :]
    masked = dict(universe)
    masked["excluded_directions"] = excluded
    metadata["materialized_bytes"] = int(excluded.nbytes)
    return masked, metadata


def _series_convergence(
    epochs: list[int],
    values: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    count = min(len(epochs), len(values))
    if count == 0:
        return {"points": 0}
    x = np.asarray(epochs[-count:], dtype=np.float64)
    y = np.asarray([value[key] for value in values[-count:]], dtype=np.float64)
    tail = min(5, count)
    x_tail = x[-tail:]
    y_tail = y[-tail:]
    if tail >= 2:
        slope = float(np.polyfit(x_tail - x_tail[0], y_tail, 1)[0])
        interval = float(np.median(np.diff(x_tail)))
        projected_change = slope * interval
    else:
        slope = 0.0
        projected_change = math.inf
    scale = max(
        abs(float(y_tail[-1])),
        float(np.mean(np.abs(y_tail))),
        1.0e-12,
    )
    return {
        "points": count,
        "first_epoch": int(x[0]),
        "final_epoch": int(x[-1]),
        "first": float(y[0]),
        "final": float(y[-1]),
        "last_change": None if count < 2 else float(y[-1] - y[-2]),
        "tail_points": tail,
        "tail_slope_per_epoch": slope,
        "projected_relative_change_per_valuation": abs(projected_change) / scale,
    }


def convergence_summary(tracker: Any) -> dict[str, Any]:
    """Summarize whether full-training objectives flattened late in a run."""

    deterministic = getattr(tracker, "deterministic_optimizer", None)
    if deterministic is not None:
        return {
            "criterion": (
                "SciPy L-BFGS-B full-training optimizer-coordinate gradient "
                "infinity norm is no larger than its configured tolerance"
            ),
            "clearly_converged": bool(
                deterministic["gradient_tolerance_met"]
            ),
            "main": deterministic,
            "moment_drift": {"points": 0},
            "moment_sqr": {"points": 0},
            "final_lr": None,
            "final_moment_lr": None,
        }

    main = _series_convergence(
        tracker.valuation_epochs,
        tracker.training_values,
        "loss",
    )
    drift = _series_convergence(
        tracker.moment_valuation_epochs,
        tracker.moment_training_values,
        "drift_loss_mean",
    )
    sqr = _series_convergence(
        tracker.moment_valuation_epochs,
        tracker.moment_training_values,
        "sqr_loss_mean",
    )
    final_lr = None if not tracker.lr else float(tracker.lr[-1])
    final_moment_lr = None if not tracker.moment_lr else float(tracker.moment_lr[-1])
    main_flat = (
        main.get("points", 0) >= 5
        and main["projected_relative_change_per_valuation"] <= 5.0e-3
    )
    moments_flat = (
        drift.get("points", 0) == 0
        or (
            drift.get("points", 0) >= 5
            and sqr.get("points", 0) >= 5
            and drift["projected_relative_change_per_valuation"] <= 5.0e-3
            and sqr["projected_relative_change_per_valuation"] <= 5.0e-3
        )
    )
    return {
        "criterion": (
            "at least five full-training valuations and <=0.5% projected "
            "relative change over one valuation interval for each trained objective"
        ),
        "clearly_converged": main_flat and moments_flat,
        "main": main,
        "moment_drift": drift,
        "moment_sqr": sqr,
        "final_lr": final_lr,
        "final_moment_lr": final_moment_lr,
    }


def martingale_summary(output: dict[str, Any]) -> dict[str, Any]:
    """Summarize zero-drift and cost-band evidence with multiplicity visible."""

    pooled_z_score = np.asarray(output["mean_dH_over_err"], dtype=np.float64)
    pooled_return_on_cost = np.asarray(output["return_on_cost"], dtype=np.float64)
    z_score = np.asarray(
        output.get("mean_dH_over_err_by_time", pooled_z_score), dtype=np.float64
    )
    return_on_cost = np.asarray(
        output.get("return_on_cost_by_time", pooled_return_on_cost),
        dtype=np.float64,
    )
    instruments = int(z_score.shape[-1])
    times = int(z_score.size // instruments)
    conditions = int(z_score.size)
    bonferroni_95 = NormalDist().inv_cdf(1.0 - 0.05 / (2.0 * conditions))
    max_abs_z = float(np.max(np.abs(z_score)))
    return {
        "instruments": instruments,
        "times": times,
        "time_instrument_conditions": conditions,
        "max_abs_z": max_abs_z,
        "rms_z": float(np.sqrt(np.mean(z_score**2))),
        "count_abs_z_gt_2": int(np.count_nonzero(np.abs(z_score) > 2.0)),
        "fraction_abs_z_le_2": float(np.mean(np.abs(z_score) <= 2.0)),
        "bonferroni_95_threshold": bonferroni_95,
        "within_population_mc_noise_95_familywise": max_abs_z <= bonferroni_95,
        "max_abs_return_on_cost": float(np.max(np.abs(return_on_cost))),
        "fraction_abs_return_on_cost_le_1": float(
            np.mean(np.abs(return_on_cost) <= 1.0)
        ),
        "pooled_max_abs_z": float(np.max(np.abs(pooled_z_score))),
        "pooled_rms_z": float(np.sqrt(np.mean(pooled_z_score**2))),
        "pooled_max_abs_return_on_cost": float(
            np.max(np.abs(pooled_return_on_cost))
        ),
        "density_mean": float(output["density_mean"]),
        "density_std": float(output["density_std"]),
        "density_err": float(output["density_err"]),
        "density_mean_minus_one_over_err": float(
            output["density_mean_minus_one_over_err"]
        ),
        "density_mean_within_5_mc_errors_of_one": bool(
            abs(float(output["density_mean"]) - 1.0)
            <= max(5.0 * float(output["density_err"]), 1.0e-6)
        ),
        "density_ess_fraction": float(
            output["density_ess"] / output["n_error_observations"]
        ),
        "density_normalization_error": float(output["density_normalization_error"]),
        "density_pre_normalization_mean": float(
            output.get("density_pre_normalization_mean", output["density_mean"])
        ),
        "density_split_rescaling_factor": float(
            output.get("density_split_rescaling_factor", 1.0)
        ),
        "path_density_mean": float(output["path_density_mean"]),
        "path_density_ess": float(output["path_density_ess"]),
        "path_density_normalization_error": float(
            output["path_density_normalization_error"]
        ),
    }


def _instrument_outputs(output: dict[str, Any]) -> dict[str, Any]:
    names = (
        "mean_dH",
        "std_dH",
        "err_dH",
        "mean_dH_over_err",
        "return_on_cost",
        "mean_dH_by_time",
        "std_dH_by_time",
        "err_dH_by_time",
        "mean_dH_over_err_by_time",
        "return_on_cost_by_time",
        "c_cond",
        "bar_c",
        "c_prop",
        "c",
        "binding_cond_fraction",
        "binding_uncond_fraction",
        "binding_prop_fraction",
        "excess_mean",
        "excess_std",
        "excess_quantiles",
        "outside_cost_fraction",
        "mean_absolute_violation",
        "excess_cost_multiples",
        "signed_excess_mean_by_multiple",
        "signed_excess_std_by_multiple",
        "unsigned_excess_mean_by_multiple",
        "unsigned_excess_std_by_multiple",
        "unsigned_excess_mean_across_instruments",
        "action_mean",
        "action_std",
        "action_max",
        "action_cap",
        "action_cap_saturation_rate",
        "action_cap_objective_derivative",
        "deterministic_preprocessed_action",
        "deterministic_correction_action",
        "deterministic_total_action",
        "quantile_levels",
    )
    return {
        name: np.asarray(output[name]).tolist()
        for name in names
        if name in output
    }


def _baseline_split(output: dict[str, Any]) -> dict[str, Any]:
    """Extract the pre-stage reference-density analytics from a valuation."""

    pooled_z_score = np.asarray(
        output["reference_mean_dH_over_err"], dtype=np.float64
    )
    pooled_return_on_cost = np.asarray(
        output["reference_return_on_cost"], dtype=np.float64
    )
    z_score = np.asarray(
        output.get("reference_mean_dH_over_err_by_time", pooled_z_score),
        dtype=np.float64,
    )
    return_on_cost = np.asarray(
        output.get("reference_return_on_cost_by_time", pooled_return_on_cost),
        dtype=np.float64,
    )
    density_mean = float(output["reference_density_mean"])
    density_err = float(output["reference_density_err"])
    return {
        "mc_error_convention": output["mc_error_convention"],
        "std_convention": output["std_convention"],
        "martingale": {
            "instruments": int(z_score.shape[-1]),
            "times": int(z_score.size // z_score.shape[-1]),
            "time_instrument_conditions": int(z_score.size),
            "max_abs_z": float(np.max(np.abs(z_score))),
            "rms_z": float(np.sqrt(np.mean(z_score**2))),
            "max_abs_return_on_cost": float(np.max(np.abs(return_on_cost))),
            "fraction_abs_return_on_cost_le_1": float(
                np.mean(np.abs(return_on_cost) <= 1.0)
            ),
            "pooled_max_abs_z": float(np.max(np.abs(pooled_z_score))),
            "pooled_rms_z": float(np.sqrt(np.mean(pooled_z_score**2))),
            "pooled_max_abs_return_on_cost": float(
                np.max(np.abs(pooled_return_on_cost))
            ),
            "density_mean": density_mean,
            "density_std": float(output["reference_density_std"]),
            "density_err": density_err,
            "density_mean_minus_one_over_err": float(
                output["reference_density_mean_minus_one_over_err"]
            ),
            "density_mean_within_5_mc_errors_of_one": bool(
                abs(density_mean - 1.0) <= max(5.0 * density_err, 1.0e-6)
            ),
            "density_ess_fraction": float(
                output["reference_density_ess"] / output["n_error_observations"]
            ),
        },
        "instruments": {
            "mean_dH": np.asarray(output["reference_mean_dH"]).tolist(),
            "std_dH": np.asarray(output["reference_std_dH"]).tolist(),
            "err_dH": np.asarray(output["reference_err_dH"]).tolist(),
            "mean_dH_over_err": pooled_z_score.tolist(),
            "return_on_cost": pooled_return_on_cost.tolist(),
            "mean_dH_over_err_by_time": z_score.tolist(),
            "return_on_cost_by_time": return_on_cost.tolist(),
            "mean_dH_by_time": np.asarray(
                output.get("reference_mean_dH_by_time", output["reference_mean_dH"])
            ).tolist(),
            "std_dH_by_time": np.asarray(
                output.get("reference_std_dH_by_time", output["reference_std_dH"])
            ).tolist(),
            "err_dH_by_time": np.asarray(
                output.get("reference_err_dH_by_time", output["reference_err_dH"])
            ).tolist(),
            "excess_cost_multiples": np.asarray(
                output["excess_cost_multiples"]
            ).tolist(),
            "signed_excess_mean_by_multiple": np.asarray(
                output["reference_signed_excess_mean_by_multiple"]
            ).tolist(),
            "signed_excess_std_by_multiple": np.asarray(
                output["reference_signed_excess_std_by_multiple"]
            ).tolist(),
            "unsigned_excess_mean_by_multiple": np.asarray(
                output["reference_unsigned_excess_mean_by_multiple"]
            ).tolist(),
            "unsigned_excess_std_by_multiple": np.asarray(
                output["reference_unsigned_excess_std_by_multiple"]
            ).tolist(),
            "unsigned_excess_mean_across_instruments": np.asarray(
                output["reference_unsigned_excess_mean_across_instruments"]
            ).tolist(),
        },
    }


def _stage_report(
    *,
    result: Any,
    elapsed_seconds: float,
    device: torch.device,
    compilation: dict[str, int],
) -> dict[str, Any]:
    splits = {"training": result.training}
    if result.validation is not None:
        splits["validation"] = result.validation
    return {
        "checkpoint": str(result.checkpoint_path),
        "elapsed_seconds": elapsed_seconds,
        "compiled_graphs": compilation,
        "memory": memory_snapshot(device),
        "convergence": convergence_summary(result.tracker),
        "deterministic_preprocessing": (
            result.tracker.deterministic_preprocessing
        ),
        "deterministic_optimizer": result.tracker.deterministic_optimizer,
        "action_initialization": result.tracker.action_initialization,
        "splits": {
            name: {
                "loss": float(output["loss"]),
                "utility": float(output["utility"]),
                "n_cost": int(output["n_cost"]),
                "mc_error_convention": output["mc_error_convention"],
                "std_convention": output["std_convention"],
                "oce_variant": output["oce_variant"],
                "cash_offset": float(output["cash_offset"]),
                "marginal_utility_training_normalizer": float(
                    output["marginal_utility_training_normalizer"]
                ),
                "cash_offset_first_order_residual": float(
                    output["cash_offset_first_order_residual"]
                ),
                "importance_sampling": {
                    "density_measure": output["density_measure"],
                    "sampling_weight_mean": float(
                        output["sampling_weight_mean"]
                    ),
                    "sampling_weight_min": float(
                        output["sampling_weight_min"]
                    ),
                    "sampling_weight_max": float(
                        output["sampling_weight_max"]
                    ),
                    "sampling_weight_ess": float(
                        output["sampling_weight_ess"]
                    ),
                },
                "objective_gradient": output["objective_gradient"],
                "path_pnl": {
                    "mean_under_P": float(output["path_pnl_mean_under_P"]),
                    "std_under_P": float(output["path_pnl_std_under_P"]),
                    "minimum_sampled": float(
                        output["path_pnl_minimum_sampled"]
                    ),
                    "maximum_sampled": float(
                        output["path_pnl_maximum_sampled"]
                    ),
                    "quantile_probabilities": np.asarray(
                        output["path_pnl_quantile_probabilities"]
                    ).tolist(),
                    "weighted_quantiles_under_P": np.asarray(
                        output["path_pnl_weighted_quantiles_under_P"]
                    ).tolist(),
                    "downside_thresholds": np.asarray(
                        output["path_pnl_downside_thresholds"]
                    ).tolist(),
                    "probability_below_threshold_under_P": np.asarray(
                        output[
                            "path_pnl_probability_below_threshold_under_P"
                        ]
                    ).tolist(),
                },
                "density_selection": {
                    "positive_increment_fraction": float(
                        output["density_increment_positive_fraction"]
                    ),
                    "zero_increment_fraction": float(
                        output["density_increment_zero_fraction"]
                    ),
                    "previous_measure_active_mass": float(
                        output["previous_measure_active_mass"]
                    ),
                    **(
                        {
                            "tail_fraction": float(output["tail_fraction"]),
                            "tail_cutoff": float(output["tail_cutoff"]),
                            "tail_boundary_fraction": float(
                                output["tail_boundary_fraction"]
                            ),
                        }
                        if "tail_fraction" in output
                        else {}
                    ),
                },
                "martingale": martingale_summary(output),
                "instruments": _instrument_outputs(output),
            }
            for name, output in splits.items()
        },
    }


def run_regime(
    universes: list[dict[str, np.ndarray]],
    *,
    regime: str,
    paths: int,
    epochs: int,
    moment_epochs: int,
    train_config: TrainConfig,
    env: BaseEnvironment,
    restart: bool,
    same_universe: bool = False,
    universe_seed_offset: int = 0,
    cost_band_tolerance: float = 0.0,
    minimum_unsigned_improvement: float = 0.0,
    continue_after_failed_gate: bool = False,
    mask_label: str = "unmasked",
    mask_metadata: dict[str, Any] | None = None,
    utility: str = "powexp",
    lmbda: float = 1.0,
    action_type: str = "network",
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run gated measure changes until convergence or excess stops improving."""

    costs = _REGIMES[regime]
    previous_density: RealizedDensity | None = None
    stages: list[dict[str, Any]] = []
    baseline: dict[str, Any] | None = None
    previous_unsigned_excess2: float | None = None
    stop_reason = "maximum_model_steps_reached"

    def payload() -> dict[str, Any]:
        final = stages[-1]["splits"] if stages else None
        return {
            "regime": regime,
            "cost_cond": costs["cost_cond"],
            "cost_uncond": costs["cost_uncond"],
            "delta_hedged": True,
            "utility": utility,
            "lmbda": lmbda,
            "action_type": action_type,
            "universe_mode": "same" if same_universe else "independent",
            "strategy_mask": mask_metadata or {"mode": mask_label},
            "baseline": baseline,
            "stages": stages,
            "model_steps_completed": len(stages),
            "stop_reason": stop_reason,
            "final_training_martingale": (
                None if final is None else final["training"]["martingale"]
            ),
            "final_validation_martingale": (
                None
                if final is None or "validation" not in final
                else final["validation"]["martingale"]
            ),
        }

    for stage, universe in enumerate(universes, start=1):
        if env.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(env.device)
            torch.cuda.synchronize(env.device)
        gc.collect()
        before_compile = _dynamo_counters()
        started = time.monotonic()
        model_config = DysanosConfig(
            label=(
                f"oce-{utility}-{lmbda:g}-{regime}-{mask_label}-{paths}-"
                f"stage-{stage}"
            ),
            group=(
                f"validation/{paths}-paths-{epochs}-epochs/"
                f"{'same-universe' if same_universe else 'independent'}/"
                f"{utility}-{lmbda:g}/{regime}/"
                f"{mask_label}/seed-{universe_seed_offset}"
            ),
            user_version="1.0",
            utility=utility,
            lmbda=lmbda,
            action_type=action_type,
            delta_hedged=True,
            cash_returns=False,
            cost_cond=costs["cost_cond"],
            cost_uncond=costs["cost_uncond"],
            cost_prop=costs["cost_prop"],
        )
        result = train(
            universe,
            config=model_config,
            train_config=train_config,
            epochs=epochs,
            moment_epochs=moment_epochs,
            previous_density=previous_density,
            restart=restart,
            env=env,
            universe_index=stage - 1,
        )
        if env.device.type == "cuda":
            torch.cuda.synchronize(env.device)
        elapsed = time.monotonic() - started
        compilation = _counter_difference(_dynamo_counters(), before_compile)
        stage_report = _stage_report(
            result=result,
            elapsed_seconds=elapsed,
            device=env.device,
            compilation=compilation,
        )
        if baseline is None:
            baseline = {
                "source": "reference density and stage-1 calibrated costs before OCE",
                "splits": {
                    "training": _baseline_split(result.training),
                    **(
                        {"validation": _baseline_split(result.validation)}
                        if result.validation is not None
                        else {}
                    ),
                },
            }
            previous_unsigned_excess2 = float(
                np.mean(result.training["reference_unsigned_excess_mean_by_multiple"][1])
            )

        training_unsigned_excess2 = float(
            np.mean(result.training["unsigned_excess_mean_by_multiple"][1])
        )
        assert previous_unsigned_excess2 is not None
        unsigned_improvement = previous_unsigned_excess2 - training_unsigned_excess2
        relative_unsigned_improvement = unsigned_improvement / max(
            abs(previous_unsigned_excess2), 1.0e-12
        )
        convergence = stage_report["convergence"]
        return_on_cost = np.asarray(
            result.training["return_on_cost_by_time"], dtype=np.float64
        )
        cost_band_met = bool(
            np.all(np.abs(return_on_cost) <= 1.0 + cost_band_tolerance)
        )
        unsigned_improved = relative_unsigned_improvement > minimum_unsigned_improvement
        stage_report["continuation_gate"] = {
            "clearly_converged": bool(convergence["clearly_converged"]),
            "cost_band_met": cost_band_met,
            "cost_band_tolerance": cost_band_tolerance,
            "training_unsigned_excess2": training_unsigned_excess2,
            "previous_training_unsigned_excess2": previous_unsigned_excess2,
            "absolute_unsigned_excess2_improvement": unsigned_improvement,
            "relative_unsigned_excess2_improvement": relative_unsigned_improvement,
            "minimum_relative_improvement": minimum_unsigned_improvement,
            "unsigned_excess2_improved": unsigned_improved,
        }
        stages.append(stage_report)
        previous_density = result.density
        del result
        gc.collect()
        if not convergence["clearly_converged"]:
            stop_reason = "maximum_epochs_without_clear_convergence"
        elif not cost_band_met:
            stop_reason = "maximum_epochs_without_training_cost_band"
        elif not unsigned_improved:
            stop_reason = "training_unsigned_excess2_stopped_improving"
        else:
            stop_reason = "ready_for_next_model_step"
        gate_failed = stop_reason != "ready_for_next_model_step"
        stage_report["continuation_gate"]["override_applied"] = bool(
            continue_after_failed_gate and gate_failed
        )
        if not gate_failed or continue_after_failed_gate:
            previous_unsigned_excess2 = training_unsigned_excess2
        if progress is not None:
            progress(payload())
        if gate_failed and not continue_after_failed_gate:
            break
    else:
        stop_reason = "maximum_model_steps_reached"
    result_payload = payload()
    if progress is not None:
        progress(result_payload)
    return result_payload


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_experiment(
    args: argparse.Namespace,
    *,
    prepared_universes: list[dict[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    universe_seed_offset = getattr(args, "universe_seed_offset", 0)
    source_paths = args.paths
    source_universes = 1 if args.same_universe else args.stages
    if args.independent_split_universes:
        if args.paths % 2:
            raise ValueError(
                "--paths must be even when --independent-split-universes is used"
            )
        source_paths = args.paths // 2
        source_universes *= 2
    preparation_started = time.monotonic()
    experiment = RNDysanosExperiment(
        num_samples=source_paths,
        pcnt_validation=args.validation_fraction,
        cnt_universe=source_universes,
        universe_seed_offset=universe_seed_offset,
    )
    source_prepared = (
        prep_experiment(experiment, plot_sampler=False)
        if prepared_universes is None
        else prepared_universes
    )
    if len(source_prepared) != source_universes:
        raise ValueError(
            f"expected {source_universes} prepared source universes, got "
            f"{len(source_prepared)}"
        )
    prepared_source_universes = len(source_prepared)
    prepared = (
        pair_training_validation_universes(source_prepared, consume=True)
        if args.independent_split_universes
        else source_prepared
    )
    if prepared is not source_prepared:
        del source_prepared
        gc.collect()
    mask_metadata: dict[str, Any] | None = None
    mask_label = "unmasked"
    arb_mask_report = getattr(args, "arb_mask_report", None)
    if arb_mask_report is not None:
        if len(prepared) != 1:
            raise ValueError(
                "--arb-mask-report currently requires one prepared universe; "
                "use --same-universe and do not use --independent-split-universes"
            )
        prepared[0], mask_metadata = attach_arb_direction_mask(
            prepared[0],
            report_path=arb_mask_report,
            validation_fraction=args.validation_fraction,
            universe_seed_offset=universe_seed_offset,
        )
        mask_label = "arb-masked"
    preparation_seconds = time.monotonic() - preparation_started
    universes = stage_universes(
        prepared,
        stages=args.stages,
        same_universe=args.same_universe,
    )

    regime_names = (
        ["unconditional", "conditional"]
        if args.regimes == "both"
        else [args.regimes]
    )
    report: dict[str, Any] = {
        "paths": args.paths,
        "epochs": args.epochs,
        "moment_epochs": args.moment_epochs,
        "lr": args.lr,
        "final_lr": args.final_lr,
        "optimizer": getattr(args, "optimizer", "AdamW"),
        "action_type": getattr(args, "action_type", "network"),
        "moment_lr": args.moment_lr,
        "moment_final_lr": args.moment_final_lr,
        "action_initialization_epochs": args.action_initialization_epochs,
        "action_initialization_lr": args.action_initialization_lr,
        "validation_fraction": args.validation_fraction,
        "utility": args.utility,
        "lmbda": args.lmbda,
        "training_paths": int(args.paths * (1.0 - args.validation_fraction)),
        "validation_paths": int(args.paths * args.validation_fraction),
        "universe_seed_offset": universe_seed_offset,
        "minimum_epochs": args.minimum_epochs,
        "minimum_moment_epochs": args.minimum_moment_epochs,
        "early_stopping": args.early_stopping,
        "cost_band_tolerance": args.cost_band_tolerance,
        "minimum_unsigned_improvement": args.minimum_unsigned_improvement,
        "continue_after_failed_gate": args.continue_after_failed_gate,
        "stages": args.stages,
        "universe_mode": "same" if args.same_universe else "independent",
        "prepared_universes": len(prepared),
        "prepared_source_universes": prepared_source_universes,
        "independent_split_universes": args.independent_split_universes,
        "compile": not args.no_compile,
        "strategy_mask": mask_metadata or {"mode": mask_label},
        "preparation_seconds": preparation_seconds,
        "regimes": {},
    }
    with DeviceLock() as device_lock:
        env = BaseEnvironment(device=device_lock.device, debug=False)
        train_config = TrainConfig(
            trn_batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            validation_fraction=experiment.pcnt_validation,
            optimizer=getattr(args, "optimizer", "AdamW"),
            lr=args.lr,
            final_lr=args.final_lr,
            moment_lr=args.moment_lr,
            moment_final_lr=args.moment_final_lr,
            action_initialization_epochs=args.action_initialization_epochs,
            action_initialization_lr=args.action_initialization_lr,
            weight_decay=args.weight_decay,
            clip_grad_norm=None,
            compile=not args.no_compile,
            value_seconds=args.value_seconds,
            value_update=args.value_update,
            checkpoint_seconds=args.checkpoint_seconds,
            checkpoint_update=args.checkpoint_update,
            early_stopping=args.early_stopping,
            minimum_epochs=args.minimum_epochs,
            minimum_moment_epochs=args.minimum_moment_epochs,
            convergence_points=args.convergence_points,
            convergence_relative_tolerance=args.convergence_relative_tolerance,
            require_cost_band=True,
            cost_band_tolerance=args.cost_band_tolerance,
            checkpoint_root=args.checkpoint_root,
        )
        report["device"] = str(env.device)
        report["memory_before"] = memory_snapshot(env.device)
        for regime in regime_names:
            def write_progress(regime_output: dict[str, Any]) -> None:
                report["regimes"][regime] = regime_output
                report["memory_current"] = memory_snapshot(env.device)
                _atomic_json(Path(args.report), report)

            report["regimes"][regime] = run_regime(
                universes,
                regime=regime,
                paths=args.paths,
                epochs=args.epochs,
                moment_epochs=args.moment_epochs,
                train_config=train_config,
                env=env,
                restart=args.restart,
                same_universe=args.same_universe,
                universe_seed_offset=universe_seed_offset,
                cost_band_tolerance=args.cost_band_tolerance,
                minimum_unsigned_improvement=args.minimum_unsigned_improvement,
                continue_after_failed_gate=args.continue_after_failed_gate,
                mask_label=mask_label,
                mask_metadata=mask_metadata,
                utility=args.utility,
                lmbda=args.lmbda,
                action_type=getattr(args, "action_type", "network"),
                progress=write_progress,
            )
            _atomic_json(Path(args.report), report)
        report["memory_after"] = memory_snapshot(env.device)
    _atomic_json(Path(args.report), report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--moment-epochs", type=int)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument("--universe-seed-offset", type=int, default=0)
    parser.add_argument(
        "--independent-split-universes",
        action="store_true",
        help=(
            "construct every 50/50 split by pairing two independently simulated "
            "half-sized universes"
        ),
    )
    parser.add_argument(
        "--same-universe",
        action="store_true",
        help="reuse the exact first universe and train/test split at every stage",
    )
    parser.add_argument(
        "--regimes",
        choices=["costfree", "unconditional", "conditional", "both"],
        default="both",
    )
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--val-batch-size", type=int, default=4_096)
    parser.add_argument(
        "--optimizer",
        choices=["Adam", "AdamW", "LBFGS", "SciPy"],
        default="AdamW",
    )
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument(
        "--final-lr",
        type=float,
        default=1.0e-3,
        help="must equal --lr; the main learning rate is constant",
    )
    parser.add_argument("--moment-lr", type=float, default=1.0e-3)
    parser.add_argument("--moment-final-lr", type=float, default=1.0e-5)
    parser.add_argument("--action-initialization-epochs", type=int, default=0)
    parser.add_argument("--action-initialization-lr", type=float, default=1.0e-3)
    parser.add_argument(
        "--utility",
        choices=["powexp", "entropy", "cvar"],
        default="powexp",
    )
    parser.add_argument("--lmbda", type=float, default=1.0)
    parser.add_argument(
        "--action-type",
        choices=["network", "deterministic"],
        default="network",
    )
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--minimum-epochs", type=int, default=0)
    parser.add_argument("--minimum-moment-epochs", type=int, default=0)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--convergence-points", type=int, default=5)
    parser.add_argument("--convergence-relative-tolerance", type=float, default=5.0e-3)
    parser.add_argument("--cost-band-tolerance", type=float, default=0.0)
    parser.add_argument("--minimum-unsigned-improvement", type=float, default=0.0)
    parser.add_argument(
        "--continue-after-failed-gate",
        action="store_true",
        help=(
            "continue through all requested model steps even when a preceding "
            "step fails convergence, cost-band, or excess-improvement gates"
        ),
    )
    parser.add_argument("--value-seconds", type=float, default=3_600.0)
    parser.add_argument("--value-update", type=int, default=100)
    parser.add_argument("--checkpoint-seconds", type=float, default=3_600.0)
    parser.add_argument("--checkpoint-update", type=int, default=100)
    parser.add_argument("--checkpoint-root", default="~/.cache/rndysanos/oce")
    parser.add_argument(
        "--arb-mask-report",
        help=(
            "cost-local-arbitrage-mask JSON for direction-specific option "
            "inactivation; requires a matching same-universe simulation"
        ),
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.moment_epochs is None:
        args.moment_epochs = args.epochs
    for name in ("paths", "epochs", "moment_epochs", "stages", "batch_size", "val_batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be strictly between zero and one")
    if args.independent_split_universes and args.validation_fraction != 0.5:
        raise ValueError(
            "--independent-split-universes requires --validation-fraction 0.5"
        )
    report = run_experiment(args)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "convergence_summary",
    "main",
    "martingale_summary",
    "memory_snapshot",
    "pair_training_validation_universes",
    "run_experiment",
    "run_regime",
    "stage_universes",
]
