"""Audit the exact P-free Q transition implicit in shared-strike FLEX SANOS."""

from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import torch
from cdxcore.version import version
from scipy.special import ndtr

from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.finite_martingale_transport import (
    SanosStepMartingaleTransition,
    build_revealed_node_rolled_sanos_chain,
    condition_sanos_step_on_spot,
    solve_shared_sanos_density_step_from_dlvsqs,
    solve_shared_sanos_step_transition_from_dlvsqs,
)
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.run_joint_hierarchical_q1 import _load_inputs
from quantdata.torch import BaseEnvironment


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class NativeSanosQAuditConfig:
    """Numerical controls and semantic gates for the native-Q audit."""

    maximum_expiry: int = 2
    mc_paths_per_replication: int = 24_000
    mc_replications: int = 2
    posterior_grid_points: int = 257
    posterior_probe_spots: tuple[float, ...] = (0.85, 1.0, 1.15)
    seed: int = 8_240_824
    exact_tolerance: float = 1.0e-8
    minimum_weighted_conditional_std: float = 1.0e-5
    minimum_weighted_two_sided_mass: float = 1.0e-5
    minimum_atm_split_percent: float = 5.0
    maximum_mc_standard_errors: float = 5.0

    def __post_init__(self) -> None:
        if self.maximum_expiry < 2 or self.maximum_expiry > 14:
            raise ValueError("'maximum_expiry' must lie in [2,14]")
        if self.mc_paths_per_replication < 32:
            raise ValueError("'mc_paths_per_replication' must be at least 32")
        if self.mc_replications < 2:
            raise ValueError("'mc_replications' must be at least two")
        if self.posterior_grid_points < 33:
            raise ValueError("'posterior_grid_points' must be at least 33")
        if any(not np.isfinite(value) or value <= 0.0 for value in self.posterior_probe_spots):
            raise ValueError("Posterior probe spots must be finite and positive")
        if not np.isfinite(self.exact_tolerance) or self.exact_tolerance <= 0.0:
            raise ValueError("'exact_tolerance' must be finite and positive")
        if (
            not np.isfinite(self.minimum_weighted_conditional_std)
            or self.minimum_weighted_conditional_std < 0.0
            or not np.isfinite(self.minimum_weighted_two_sided_mass)
            or self.minimum_weighted_two_sided_mass < 0.0
        ):
            raise ValueError("Noncollapse gates must be finite and non-negative")
        if not 0.0 <= self.minimum_atm_split_percent < 50.0:
            raise ValueError("ATM split gate must lie in [0,50)")
        if (
            not np.isfinite(self.maximum_mc_standard_errors)
            or self.maximum_mc_standard_errors <= 0.0
        ):
            raise ValueError("MC standard-error gate must be finite and positive")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-completion", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--dataset", choices=("validation", "train"), default="validation")
    parser.add_argument(
        "--maximum-dates",
        type=int,
        default=0,
        help="Evenly spaced date count; zero means every date in the selected dataset.",
    )
    parser.add_argument("--maximum-expiry", type=int, default=2)
    parser.add_argument("--mc-paths", type=int, default=24_000)
    parser.add_argument("--mc-replications", type=int, default=2)
    parser.add_argument("--posterior-grid-points", type=int, default=257)
    parser.add_argument("--seed", type=int, default=8_240_824)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Total audit process budget including the parent; must lie in [1,4].",
    )
    return parser.parse_args()


def _black_calls(
    spots: np.ndarray,
    strikes: np.ndarray,
    total_vol: float,
) -> np.ndarray:
    spots = np.asarray(spots, dtype=np.float64).reshape(-1, 1)
    strikes = np.asarray(strikes, dtype=np.float64).reshape(1, -1)
    if total_vol == 0.0:
        return np.maximum(spots - strikes, 0.0)
    d1 = (np.log(spots / strikes) + 0.5 * total_vol**2) / total_vol
    return spots * ndtr(d1) - strikes * ndtr(d1 - total_vol)


def _step_noncollapse(
    step: SanosStepMartingaleTransition,
    *,
    tolerance: float,
) -> dict[str, float]:
    transition = step.transition_probabilities
    support = step.support
    differences = support[None, :] - support[:, None]
    row_variance = np.sum(transition * differences**2, axis=1)
    upward = np.sum(transition * (differences > tolerance), axis=1)
    downward = np.sum(transition * (differences < -tolerance), axis=1)
    weights = step.previous_density
    return {
        "weighted_conditional_variance": float(weights @ row_variance),
        "weighted_conditional_std": float(np.sqrt(max(0.0, weights @ row_variance))),
        "density_weighted_nonzero_variance_percent": float(
            100.0 * np.sum(weights[row_variance > tolerance**2])
        ),
        "density_weighted_two_sided_row_percent": float(
            100.0 * np.sum(weights[(upward > tolerance) & (downward > tolerance)])
        ),
        "density_weighted_upward_mass": float(weights @ upward),
        "density_weighted_downward_mass": float(weights @ downward),
        "density_weighted_min_directional_mass": float(weights @ np.minimum(upward, downward)),
    }


def _replication_statistics(values: np.ndarray) -> dict[str, float]:
    sample_std = float(np.std(values, ddof=1))
    mean = float(np.mean(values))
    return {
        "paths": int(len(values)),
        "mean": mean,
        "mean_bps": 10_000.0 * mean,
        "sample_std": sample_std,
        "mc_standard_error": sample_std / math.sqrt(len(values)),
        "positive_percent": float(100.0 * np.mean(values > 0.0)),
        "negative_percent": float(100.0 * np.mean(values < 0.0)),
        "above_one_bp_percent": float(100.0 * np.mean(values > 1.0e-4)),
        "below_minus_one_bp_percent": float(100.0 * np.mean(values < -1.0e-4)),
    }


def _posterior_atm_grid(
    step: SanosStepMartingaleTransition,
    observed_spots: np.ndarray,
    *,
    previous_backbone_total_vol: float,
    next_backbone_total_vol: float,
) -> np.ndarray:
    prices = np.empty_like(observed_spots)
    strike = np.asarray((1.0,), dtype=np.float64)
    for index, observed_spot in enumerate(observed_spots):
        law = condition_sanos_step_on_spot(
            step,
            float(observed_spot),
            previous_backbone_total_vol=previous_backbone_total_vol,
            next_backbone_total_vol=next_backbone_total_vol,
        )
        prices[index] = law.call_prices(strike)[0]
    return prices


def monte_carlo_atm_audit(
    step: SanosStepMartingaleTransition,
    *,
    previous_backbone_total_vol: float,
    next_backbone_total_vol: float,
    config: NativeSanosQAuditConfig,
    seed_offset: int,
) -> dict[str, object]:
    """Fresh-path ATM tower and split audit with independent replications."""
    increment = float(
        np.sqrt(
            max(
                0.0,
                next_backbone_total_vol**2 - previous_backbone_total_vol**2,
            )
        )
    )
    current_atm = float(
        step.next_density
        @ _black_calls(step.support, np.asarray((1.0,)), next_backbone_total_vol)[:, 0]
    )
    paths_by_replication: list[tuple[np.ndarray, np.ndarray]] = []
    all_front_spots: list[np.ndarray] = []
    transition_cdf = np.cumsum(step.transition_probabilities, axis=1)
    transition_cdf[:, -1] = 1.0
    for replication in range(config.mc_replications):
        rng = np.random.default_rng(config.seed + seed_offset + 1_000_003 * replication)
        latent_previous = rng.choice(
            len(step.support),
            size=config.mc_paths_per_replication,
            p=step.previous_density,
        )
        front_normal = rng.standard_normal(config.mc_paths_per_replication)
        front_spot = step.support[latent_previous] * np.exp(
            -0.5 * previous_backbone_total_vol**2 + previous_backbone_total_vol * front_normal
        )
        uniforms = rng.random(config.mc_paths_per_replication)
        latent_next = np.sum(
            uniforms[:, None] > transition_cdf[latent_previous],
            axis=1,
        )
        increment_normal = rng.standard_normal(config.mc_paths_per_replication)
        next_spot = (
            front_spot
            * step.support[latent_next]
            / step.support[latent_previous]
            * np.exp(-0.5 * increment**2 + increment * increment_normal)
        )
        paths_by_replication.append((front_spot, next_spot))
        all_front_spots.append(front_spot)

    combined_front = np.concatenate(all_front_spots)
    log_lower = float(np.min(np.log(combined_front))) - 1.0e-6
    log_upper = float(np.max(np.log(combined_front))) + 1.0e-6
    log_grid = np.linspace(log_lower, log_upper, config.posterior_grid_points)
    spot_grid = np.exp(log_grid)
    price_grid = _posterior_atm_grid(
        step,
        spot_grid,
        previous_backbone_total_vol=previous_backbone_total_vol,
        next_backbone_total_vol=next_backbone_total_vol,
    )
    midpoint_spots = np.exp(0.5 * (log_grid[:-1] + log_grid[1:]))
    probe_indices = (
        np.linspace(
            0,
            len(midpoint_spots) - 1,
            min(32, len(midpoint_spots)),
        )
        .round()
        .astype(np.int64)
    )
    exact_midpoint = _posterior_atm_grid(
        step,
        midpoint_spots[probe_indices],
        previous_backbone_total_vol=previous_backbone_total_vol,
        next_backbone_total_vol=next_backbone_total_vol,
    )
    interpolated_midpoint = np.interp(
        np.log(midpoint_spots[probe_indices]),
        log_grid,
        price_grid,
    )
    interpolation_error = float(np.max(np.abs(exact_midpoint - interpolated_midpoint)))

    mark_statistics: list[dict[str, float]] = []
    payoff_statistics: list[dict[str, float]] = []
    mark_values: list[np.ndarray] = []
    payoff_values: list[np.ndarray] = []
    for front_spot, next_spot in paths_by_replication:
        conditional_atm = np.interp(np.log(front_spot), log_grid, price_grid)
        mark_pnl = conditional_atm - current_atm
        payoff_pnl = np.maximum(next_spot - 1.0, 0.0) - current_atm
        mark_values.append(mark_pnl)
        payoff_values.append(payoff_pnl)
        mark_statistics.append(_replication_statistics(mark_pnl))
        payoff_statistics.append(_replication_statistics(payoff_pnl))
    pooled_mark = _replication_statistics(np.concatenate(mark_values))
    pooled_payoff = _replication_statistics(np.concatenate(payoff_values))
    mean_tolerance = max(
        config.maximum_mc_standard_errors * pooled_mark["mc_standard_error"] + interpolation_error,
        config.exact_tolerance,
    )
    passed = (
        abs(pooled_mark["mean"]) <= mean_tolerance
        and pooled_mark["positive_percent"] >= config.minimum_atm_split_percent
        and pooled_mark["negative_percent"] >= config.minimum_atm_split_percent
    )
    return {
        "current_atm_call": current_atm,
        "posterior_grid_points": config.posterior_grid_points,
        "maximum_checked_interpolation_error": interpolation_error,
        "mark_to_market": {
            "replications": mark_statistics,
            "pooled": pooled_mark,
        },
        "terminal_payoff": {
            "replications": payoff_statistics,
            "pooled": pooled_payoff,
        },
        "mean_tolerance": mean_tolerance,
        "passed": bool(passed),
    }


def audit_native_sanos_surface(
    *,
    support_strikes: np.ndarray,
    densities: np.ndarray,
    dlvsqs: np.ndarray,
    model_time_increments_y: np.ndarray,
    backbone_total_vols: np.ndarray,
    config: NativeSanosQAuditConfig,
    seed_offset: int = 0,
) -> dict[str, object]:
    """Audit W2 and optional later native SANOS transitions for one date."""
    support_strikes = np.asarray(support_strikes, dtype=np.float64)
    densities = np.asarray(densities, dtype=np.float64)
    dlvsqs = np.asarray(dlvsqs, dtype=np.float64)
    model_time_increments_y = np.asarray(model_time_increments_y, dtype=np.float64)
    backbone_total_vols = np.asarray(backbone_total_vols, dtype=np.float64)
    if support_strikes.ndim != 2:
        raise ValueError("'support_strikes' must have shape (expiry,support)")
    expiries, support_size = support_strikes.shape
    if (
        densities.shape != (expiries, support_size)
        or dlvsqs.shape != (expiries, support_size - 2)
        or model_time_increments_y.shape != (expiries,)
        or backbone_total_vols.shape != (expiries,)
        or config.maximum_expiry > expiries
    ):
        raise ValueError("Native SANOS surface arrays have inconsistent geometry")
    if not np.allclose(
        support_strikes,
        support_strikes[:1],
        atol=0.0,
        rtol=0.0,
    ):
        raise ValueError("Native-Q audit requires shared strikes across expiries")

    expiry_reports: list[dict[str, object]] = []
    steps: list[SanosStepMartingaleTransition] = []
    for target_index in range(1, config.maximum_expiry):
        support = support_strikes[target_index]
        step = solve_shared_sanos_step_transition_from_dlvsqs(
            support,
            densities[target_index - 1],
            densities[target_index],
            dlvsqs=dlvsqs[target_index],
            time_increment_y=float(model_time_increments_y[target_index]),
            tolerance=config.exact_tolerance,
        )
        steps.append(step)
        propagated = step.previous_density @ step.transition_probabilities
        strike_calls = _black_calls(
            support,
            support,
            float(backbone_total_vols[target_index]),
        )
        propagated_calls = propagated @ strike_calls
        target_calls = step.next_density @ strike_calls
        call_residual = float(np.max(np.abs(propagated_calls - target_calls)))
        previous_vol = float(backbone_total_vols[target_index - 1])
        next_vol = float(backbone_total_vols[target_index])
        variance_increment = next_vol**2 - previous_vol**2
        if variance_increment < -config.exact_tolerance:
            raise RuntimeError("Backbone total variance decreases between expiries")
        recombined_vol = math.sqrt(previous_vol**2 + max(0.0, variance_increment))
        noncollapse = _step_noncollapse(step, tolerance=config.exact_tolerance)
        posterior_probes: list[dict[str, object]] = []
        for observed_spot in config.posterior_probe_spots:
            law = condition_sanos_step_on_spot(
                step,
                observed_spot,
                previous_backbone_total_vol=previous_vol,
                next_backbone_total_vol=next_vol,
            )
            posterior_probes.append(
                {
                    "observed_spot": observed_spot,
                    "atm_call": float(law.call_prices(np.asarray((1.0,)))[0]),
                    "diagnostics": asdict(law.diagnostics),
                }
            )
        exact_maximum = max(
            step.diagnostics.maximum_row_mass_residual,
            step.diagnostics.maximum_row_martingale_residual,
            step.diagnostics.maximum_density_propagation_residual,
            call_residual,
            abs(recombined_vol - next_vol),
            max(
                abs(float(probe["diagnostics"]["conditional_spot_mean_residual"]))
                for probe in posterior_probes
            ),
        )
        exact_passed = (
            exact_maximum <= config.exact_tolerance
            and step.diagnostics.minimum_transition_probability >= -config.exact_tolerance
        )
        noncollapse_passed = (
            noncollapse["weighted_conditional_std"] >= config.minimum_weighted_conditional_std
            and noncollapse["density_weighted_min_directional_mass"]
            >= config.minimum_weighted_two_sided_mass
        )
        mc = monte_carlo_atm_audit(
            step,
            previous_backbone_total_vol=previous_vol,
            next_backbone_total_vol=next_vol,
            config=config,
            seed_offset=seed_offset + target_index * 10_000_019,
        )
        expiry_reports.append(
            {
                "source_expiry": target_index,
                "target_expiry": target_index + 1,
                "transition_diagnostics": asdict(step.diagnostics),
                "backbone": {
                    "previous_total_vol": previous_vol,
                    "next_total_vol": next_vol,
                    "increment_total_vol": math.sqrt(max(0.0, variance_increment)),
                    "recombination_residual": recombined_vol - next_vol,
                },
                "all_strike_call_tower_method": (
                    "propagated discrete density with recombined backbone variance"
                ),
                "all_strike_call_tower_num_strikes": len(support),
                "all_strike_call_tower_maximum_absolute_residual": call_residual,
                "noncollapse": noncollapse,
                "posterior_spot_probes": posterior_probes,
                "monte_carlo_atm": mc,
                "exact_passed": bool(exact_passed),
                "noncollapse_passed": bool(noncollapse_passed),
                "passed": bool(exact_passed and noncollapse_passed and mc["passed"]),
            }
        )
    front_cdf = np.cumsum(densities[0])
    representative_nodes = np.unique(
        np.searchsorted(front_cdf, np.asarray((0.1, 0.5, 0.9)), side="left")
    )
    rolled_node_reports: list[dict[str, object]] = []
    for node in representative_nodes:
        chain = build_revealed_node_rolled_sanos_chain(
            support_strikes[0],
            tuple(steps),
            source_node_index=int(node),
            old_dlvsqs=dlvsqs,
            old_time_increments_y=model_time_increments_y,
            old_backbone_total_vols=backbone_total_vols,
            tolerance=config.exact_tolerance,
        )
        rolled_node_reports.append(
            {
                "source_node_index": int(node),
                "source_relative_spot": float(support_strikes[0, node]),
                "density0_atom_mass": float(chain.density0[node]),
                "first_density_transition_row_residual": float(
                    np.max(np.abs(chain.densities[0] - steps[0].transition_probabilities[node]))
                ),
                "maximum_mass_residual": chain.maximum_mass_residual,
                "maximum_mean_residual": chain.maximum_mean_residual,
                "dlvsq_time_scale_factors": chain.dlvsq_time_scale_factors.tolist(),
                "backbone_total_vols": chain.backbone_total_vols.tolist(),
            }
        )
    rolled_maximum_residual = max(
        max(
            float(report["first_density_transition_row_residual"]),
            float(report["maximum_mass_residual"]),
            float(report["maximum_mean_residual"]),
        )
        for report in rolled_node_reports
    )
    revealed_audit = {
        "conditioning": "front discrete SANOS node I is revealed",
        "custom_grid": "support/support[I]",
        "dlv_coordinate": (
            "exact oracle uses integrated DLVsq*dt; annualized rolled DLVsq may "
            "exceed the ordinary market-state cap solely because 5 days are "
            "represented on the 4.8-day front node"
        ),
        "ordinary_market_state_dlvsq_cap": 9.0,
        "maximum_rolled_front_dlvsq": float(
            np.max(dlvsqs[1] * model_time_increments_y[1] / model_time_increments_y[0])
        ),
        "rolled_front_coordinates_above_ordinary_cap": int(
            np.sum(dlvsqs[1] * model_time_increments_y[1] / model_time_increments_y[0] > 9.0)
        ),
        "front_timing_note": ("old W2-W1 is 5 business days; rolled front is 4.8 business days"),
        "representative_nodes": rolled_node_reports,
        "maximum_residual": rolled_maximum_residual,
        "passed": bool(rolled_maximum_residual <= config.exact_tolerance),
    }
    exact_rolled_front_dlvsq = dlvsqs[1] * model_time_increments_y[1] / model_time_increments_y[0]
    clipped_rolled_front_dlvsq = np.minimum(exact_rolled_front_dlvsq, 9.0)
    equivalent_old_dlvsq = (
        clipped_rolled_front_dlvsq * model_time_increments_y[0] / model_time_increments_y[1]
    )
    strict_cap_density = solve_shared_sanos_density_step_from_dlvsqs(
        support_strikes[0],
        densities[0],
        dlvsqs=equivalent_old_dlvsq,
        time_increment_y=float(model_time_increments_y[1]),
        tolerance=config.exact_tolerance,
    )
    strict_cap_strikes = np.exp(np.linspace(-0.5, 0.4, 181))
    strict_cap_node_calls = _black_calls(
        support_strikes[0],
        strict_cap_strikes,
        float(backbone_total_vols[1]),
    )
    strict_cap_call_error = (strict_cap_density - densities[1]) @ strict_cap_node_calls
    revealed_audit["strict_ordinary_cap_projection"] = {
        "status": "exact" if np.all(exact_rolled_front_dlvsq <= 9.0) else "approximate",
        "num_dense_strikes": len(strict_cap_strikes),
        "log_strike_range": [-0.5, 0.4],
        "maximum_absolute_call_tower_error": float(np.max(np.abs(strict_cap_call_error))),
        "maximum_absolute_call_tower_error_bps_spot": float(
            10_000.0 * np.max(np.abs(strict_cap_call_error))
        ),
        "maximum_density_error": float(np.max(np.abs(strict_cap_density - densities[1]))),
    }
    w2 = expiry_reports[0]
    w2["revealed_node_rolled_chain"] = revealed_audit
    w2["passed"] = bool(w2["passed"] and revealed_audit["passed"])
    return {
        "expiry_reports": expiry_reports,
        "w2_passed": bool(w2["passed"]),
        "all_requested_expiries_passed": bool(
            all(bool(report["passed"]) for report in expiry_reports)
        ),
    }


def summarize_date_audits(date_audits: list[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate state audits with exactly one vote per selected date."""
    if not date_audits:
        raise ValueError("At least one date audit is required")
    w2_reports = [
        cast(list[Mapping[str, object]], audit["expiry_reports"])[0] for audit in date_audits
    ]
    w2_passes = np.asarray([bool(audit["w2_passed"]) for audit in date_audits])
    requested_passes = np.asarray(
        [bool(audit["all_requested_expiries_passed"]) for audit in date_audits]
    )
    weighted_stds = np.asarray(
        [float(report["noncollapse"]["weighted_conditional_std"]) for report in w2_reports]  # type: ignore[index]
    )
    mark_means = np.asarray(
        [
            float(report["monte_carlo_atm"]["mark_to_market"]["pooled"]["mean"])  # type: ignore[index]
            for report in w2_reports
        ]
    )
    return {
        "date_weighting": "equal",
        "num_dates": len(date_audits),
        "w2_pass_percent": float(100.0 * np.mean(w2_passes)),
        "all_requested_expiries_pass_percent": float(100.0 * np.mean(requested_passes)),
        "w2_weighted_conditional_std_mean": float(np.mean(weighted_stds)),
        "w2_weighted_conditional_std_minimum": float(np.min(weighted_stds)),
        "w2_atm_mark_mean_equal_date_average": float(np.mean(mark_means)),
        "w2_atm_mark_maximum_absolute_mean": float(np.max(np.abs(mark_means))),
        "w2_semantic_gate_passed": bool(np.all(w2_passes)),
        "all_requested_semantic_gates_passed": bool(np.all(requested_passes)),
    }


def _even_indices(num_dates: int, maximum_dates: int) -> np.ndarray:
    if num_dates <= 0:
        raise ValueError("No dates are available for the native-Q audit")
    if maximum_dates < 0:
        raise ValueError("'maximum_dates' must be non-negative")
    if maximum_dates == 0 or maximum_dates >= num_dates:
        return np.arange(num_dates, dtype=np.int64)
    return np.unique(np.linspace(0, num_dates - 1, maximum_dates).round().astype(np.int64))


def _audit_worker(payload: tuple[Any, ...]) -> dict[str, object]:
    (
        trading_date,
        support,
        density,
        dlv,
        dts,
        backbone,
        config,
        seed_offset,
    ) = payload
    audit = audit_native_sanos_surface(
        support_strikes=support,
        densities=density,
        dlvsqs=dlv,
        model_time_increments_y=dts,
        backbone_total_vols=backbone,
        config=config,
        seed_offset=seed_offset,
    )
    return {"date": trading_date, **audit}


def run(arguments: argparse.Namespace) -> None:
    """Load provenance-bound states and execute the durable P-free Q audit."""
    if arguments.maximum_dates < 0:
        raise ValueError("'maximum_dates' must be non-negative")
    if arguments.workers < 1 or arguments.workers > 4:
        raise ValueError("'workers' must lie in [1,4]")
    if arguments.mc_paths < 24_000:
        raise ValueError("Durable audit requires at least 24,000 paths per replication")
    config = NativeSanosQAuditConfig(
        maximum_expiry=arguments.maximum_expiry,
        mc_paths_per_replication=arguments.mc_paths,
        mc_replications=arguments.mc_replications,
        posterior_grid_points=arguments.posterior_grid_points,
        seed=arguments.seed,
    )
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    (
        model,
        train_dates,
        train_states,
        _,
        validation_dates,
        validation_states,
        _,
        sources,
    ) = _load_inputs(arguments.joint_completion, env=env)
    dates, states = (
        (validation_dates, validation_states)
        if arguments.dataset == "validation"
        else (train_dates, train_states)
    )
    if any(value.year == 2026 for value in dates):
        raise ValueError("2026 data are excluded from the native-Q audit")
    if arguments.dataset == "validation" and any(value.year != 2025 for value in dates):
        raise ValueError("Validation native-Q audit must use only 2025")
    indices = _even_indices(len(dates), arguments.maximum_dates)
    selected_dates = tuple(dates[int(index)] for index in indices)
    selected_states = states[torch.as_tensor(indices, dtype=torch.long, device=env.device)]
    with torch.no_grad():
        raw = model.raw_states_from_states(selected_states)
        parameters = bound_native_raw(raw, geometry=model.config.geometry)
        surface = model.sanos_decoder.forward_batch(
            dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols,
            eta=parameters.eta,
        )
    supports = surface.support_strikes.detach().cpu().double().numpy()
    densities = surface.densities.detach().cpu().double().numpy()
    dlvs = surface.dlvsqs.detach().cpu().double().numpy()
    backbones = surface.backbone_total_vols.detach().cpu().double().numpy()
    dts = model.sanos_decoder.model_dts_y.detach().cpu().double().numpy()
    payloads = [
        (
            selected_dates[index].isoformat(),
            supports[index],
            densities[index],
            dlvs[index],
            dts,
            backbones[index],
            config,
            index * 100_000_007,
        )
        for index in range(len(selected_dates))
    ]
    arguments.report_directory.mkdir(parents=True, exist_ok=True)
    progress_path = arguments.report_directory / "native-sanos-q-audit-progress.json"

    def record_progress(
        completed: int,
        *,
        latest_date: str | None,
        status: str,
    ) -> None:
        write_json_atomic(
            {
                "schema": "flex-native-sanos-q-audit-progress-v1",
                "status": status,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "dataset": arguments.dataset,
                "maximum_expiry": config.maximum_expiry,
                "completed_dates": completed,
                "total_dates": len(payloads),
                "latest_completed_date": latest_date,
            },
            progress_path,
        )

    record_progress(0, latest_date=None, status="running")
    date_audits: list[dict[str, object]] = []
    if arguments.workers == 1:
        results = map(_audit_worker, payloads)
    else:
        # The process budget includes this parent, hence at most workers-1
        # children.  No downstream waiter is created.
        with ProcessPoolExecutor(max_workers=arguments.workers - 1) as executor:
            for audit in executor.map(_audit_worker, payloads):
                date_audits.append(audit)
                record_progress(
                    len(date_audits),
                    latest_date=str(audit["date"]),
                    status="running",
                )
                print(
                    "native SANOS Q audit progress "
                    f"{len(date_audits)}/{len(payloads)} date={audit['date']}",
                    flush=True,
                )
            results = ()
    for audit in results:
        date_audits.append(audit)
        record_progress(
            len(date_audits),
            latest_date=str(audit["date"]),
            status="running",
        )
        print(
            "native SANOS Q audit progress "
            f"{len(date_audits)}/{len(payloads)} date={audit['date']}",
            flush=True,
        )
    summary = summarize_date_audits(date_audits)
    record_progress(
        len(date_audits),
        latest_date=str(date_audits[-1]["date"]),
        status="completed-unverified",
    )
    report_path = arguments.report_directory / "native-sanos-q-audit.json"
    completion_path = arguments.report_directory / "native-sanos-q-audit-complete.json"
    report = {
        "schema": "flex-native-sanos-q-audit-v1",
        "status": ("complete" if summary["all_requested_semantic_gates_passed"] else "failed"),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "measure": "Q constructed directly from SANOS; no P dynamics or entropy",
        "p_dynamics_used": False,
        "entropy_objective_used": False,
        "dataset": arguments.dataset,
        "selected_dates": [value.isoformat() for value in selected_dates],
        "selection": "evenly spaced" if len(indices) < len(dates) else "all dates",
        "config": asdict(config),
        "sources": sources,
        "summary": summary,
        "date_audits": date_audits,
    }
    write_json_atomic(report, report_path)
    if not summary["w2_semantic_gate_passed"]:
        raise RuntimeError(
            "Hard W2 native-Q semantic gate failed; later expiries cannot override it"
        )
    if not summary["all_requested_semantic_gates_passed"]:
        raise RuntimeError("A requested later-expiry native-Q semantic gate failed after W2 passed")
    write_json_atomic(
        {
            "schema": "flex-native-sanos-q-audit-completion-v1",
            "status": "complete",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "report": str(report_path),
            "report_sha256": checkpoint_digest(report_path),
            "joint_completion": str(arguments.joint_completion),
            "joint_completion_sha256": checkpoint_digest(arguments.joint_completion),
            "selected_date_count": len(selected_dates),
            "w2_semantic_gate_passed": True,
            "all_requested_semantic_gates_passed": summary["all_requested_semantic_gates_passed"],
        },
        completion_path,
    )
    record_progress(
        len(date_audits),
        latest_date=str(date_audits[-1]["date"]),
        status="succeeded",
    )


def main() -> None:
    run(_parse_arguments())


if __name__ == "__main__":
    run_guarded_cli(main, program="run_native_sanos_q_audit")
