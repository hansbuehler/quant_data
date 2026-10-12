"""Simulate a few fixed-chart DYSANOS martingale-coordinate steps.

This is the pathwise extension of ``run_local_martingale_chart_scaling_gate``.
The relative support is fixed and floats with spot, while the Black backbone
is the deterministic total volatility ``sigma * sqrt(T)``.  At every state the
finite density law is mapped through selected fixed-cash call coordinates to
the next residual expiry clock.  The selected surviving calls and spot have
zero conditional drift by construction.

This remains a chart experiment: spot follows the gate's finite mean-one law,
not the complete front-expiry DYSANOS marginal.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from quantdata.dysanos.flex.lvsq_drift import (
    density_martingale_lvsq_step,
    smooth_lvsq_log_loadings,
)
from quantdata.dysanos.flex.martingale_coordinate_chart import (
    recover_density_from_cash_calls,
)
from quantdata.dysanos.flex.model import (
    NativeParameters,
    bound_native_raw,
    raw_from_native_parameters,
)
from quantdata.dysanos.flex.run_local_martingale_chart_scaling_gate import (
    Config as LocalConfig,
)
from quantdata.dysanos.flex.run_local_martingale_chart_scaling_gate import (
    _cash_delta,
    _price_from_density,
    _price_grid,
    _problem,
    _spot_loadings,
    _successor_coordinates,
)
from quantdata.dysanos.flex.sanos import FlexSanosDecoder, SanosSurface

SCHEMA = "dysanos-fixed-chart-multistep-paths-v1"


@dataclass(frozen=True, kw_only=True)
class Config:
    """Controls for the short path experiment."""

    output: Path
    num_paths: int = 10_000
    num_steps: int = 4
    path_batch_size: int = 500
    seed: int = 47
    num_chart_strikes: int = 3
    num_audit_strikes: int = 17
    fixed_backbone_annualized_volatility: float = 0.05
    annualized_dlvsq_log_volatility: float = 0.05
    annualized_spot_volatility: float = 0.10


@dataclass(frozen=True)
class _BatchResult:
    """One path batch advanced by one business day."""

    successor: FlexSanosDecoder
    valid_sampled: torch.Tensor
    next_raw_valid: torch.Tensor
    next_spot_valid: torch.Tensor
    next_cash_strikes_valid: torch.Tensor
    next_audit_strikes_valid: torch.Tensor
    option_increments: torch.Tensor
    spot_increments: torch.Tensor
    conditional_state_valid: torch.Tensor
    minimum_density_by_branch: torch.Tensor
    minimum_dlvsq_margin_by_branch: torch.Tensor
    maximum_dlvsq_margin_by_branch: torch.Tensor
    condition_numbers: torch.Tensor
    conditional_selected_errors: torch.Tensor
    conditional_audit_errors: torch.Tensor


def _arguments() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-paths", type=int, default=10_000)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--path-batch-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=47)
    arguments = parser.parse_args()
    return Config(
        output=arguments.output.resolve(),
        num_paths=arguments.num_paths,
        num_steps=arguments.num_steps,
        path_batch_size=arguments.path_batch_size,
        seed=arguments.seed,
    )


def _validate(config: Config) -> None:
    if config.output in (Path("/tmp"), Path("/data")):
        raise ValueError("output must be a project-specific directory")
    if config.num_paths <= 0 or config.num_steps <= 0 or config.path_batch_size <= 0:
        raise ValueError("path, step, and batch counts must be positive")
    if config.num_chart_strikes <= 0 or config.num_audit_strikes <= 0:
        raise ValueError("strike counts must be positive")
    for value in (
        config.fixed_backbone_annualized_volatility,
        config.annualized_dlvsq_log_volatility,
        config.annualized_spot_volatility,
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("volatilities must be finite and positive")


def _atomic_json(payload: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _recover_dlvsqs_unchecked(
    decoder: FlexSanosDecoder,
    *,
    support: torch.Tensor,
    densities: torch.Tensor,
) -> torch.Tensor:
    """Recover DLV squares without rejecting inadmissible diagnostic branches."""

    density0 = decoder._density_from_calls(
        support[:, 0],
        torch.relu(1.0 - support[:, 0]),
    )
    previous_support = support[:, 0]
    previous_density = density0
    rows: list[torch.Tensor] = []
    for expiry_index in range(support.shape[1]):
        current_support = support[:, expiry_index]
        current_density = densities[:, expiry_index]
        inner = current_support[:, 1:-1]
        current_payoff = torch.relu(
            current_support[:, None] - inner[:, :, None]
        )
        previous_payoff = torch.relu(
            previous_support[:, None] - inner[:, :, None]
        )
        current_calls = torch.einsum(
            "bkn,bn->bk", current_payoff, current_density
        )
        previous_calls = torch.einsum(
            "bkn,bn->bk", previous_payoff, previous_density
        )
        box = (
            inner.square()
            * decoder.model_dts_y[expiry_index]
            / (current_support[:, 2:] - current_support[:, :-2])
        )
        rows.append(
            (current_calls - previous_calls)
            / (box * current_density[:, 1:-1])
        )
        previous_support = current_support
        previous_density = current_density
    return torch.stack(rows, dim=1)


def _sample_branches(
    num_states: int,
    num_branches: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return torch.randint(
        num_branches,
        (num_states,),
        generator=generator,
        device=device,
    )


def _advance_batch(
    config: Config,
    decoder: FlexSanosDecoder,
    *,
    raw: torch.Tensor,
    spot: torch.Tensor,
    cash_strikes: torch.Tensor,
    audit_strikes: torch.Tensor,
    generator: torch.Generator,
) -> _BatchResult:
    parameters = bound_native_raw(raw, geometry=decoder.geometry)
    surface = decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )
    base_calls = _price_grid(
        decoder,
        surface,
        spot=spot,
        cash_strikes=cash_strikes,
    )
    base_audit_calls = _price_grid(
        decoder,
        surface,
        spot=spot,
        cash_strikes=audit_strikes,
    )
    cash_delta = _cash_delta(
        support=surface.support_strikes,
        density=surface.densities,
        backbone_total_vols=surface.backbone_total_vols,
        cash_strikes=cash_strikes / spot[:, None, None],
    )
    loadings = smooth_lvsq_log_loadings(
        surface.support_strikes[0, 0],
        num_model_expiries=decoder.geometry.num_model_expiries,
        annualized_log_volatility=config.annualized_dlvsq_log_volatility,
    )
    density_step = density_martingale_lvsq_step(
        decoder,
        raw,
        raw_log_loadings=loadings,
        calendar_time_increment_y=1.0 / 252.0,
    )
    innovations = density_step.standardized_innovations
    probabilities = density_step.branch_probabilities
    num_states = len(raw)
    num_branches = len(innovations)
    successor, successor_forward, successor_eta, live = _successor_coordinates(
        decoder.env,
        decoder,
        surface,
        business_day_fraction=1.0,
        fixed_backbone_annualized_volatility=(
            config.fixed_backbone_annualized_volatility
        ),
    )
    live_cash_strikes = cash_strikes[:, live]
    live_audit_strikes = audit_strikes[:, live]
    live_base_calls = base_calls[:, live]
    live_base_audit_calls = base_audit_calls[:, live]

    _, successor_support = successor._support_strikes_from_forward_vols(
        successor_forward
    )
    successor_backbone = successor._total_vols(
        successor_forward * successor_eta
    )
    baseline_density, _ = recover_density_from_cash_calls(
        spot=spot,
        support=successor_support,
        backbone_total_vols=successor_backbone,
        cash_strikes=live_cash_strikes,
        cash_calls=live_base_calls,
        reference_density=surface.densities[:, live],
    )

    expanded_spot = spot[:, None].expand(-1, num_branches).reshape(-1)
    expanded_support = (
        surface.support_strikes[:, None]
        .expand(-1, num_branches, -1, -1)
        .reshape(-1, surface.support_strikes.shape[1], surface.support_strikes.shape[2])
    )
    expanded_backbone = (
        surface.backbone_total_vols[:, None]
        .expand(-1, num_branches, -1)
        .reshape(-1, surface.backbone_total_vols.shape[1])
    )
    expanded_cash_strikes = (
        cash_strikes[:, None]
        .expand(-1, num_branches, -1, -1)
        .reshape(-1, cash_strikes.shape[1], cash_strikes.shape[2])
    )
    fixed_calls = _price_from_density(
        spot=expanded_spot,
        support=expanded_support,
        density=density_step.densities_by_branch.reshape_as(expanded_support),
        backbone_total_vols=expanded_backbone,
        cash_strikes=expanded_cash_strikes,
    ).reshape(num_states, num_branches, *base_calls.shape[1:])

    spot_loadings = _spot_loadings(
        LocalConfig(
            output=config.output,
            annualized_spot_volatility=config.annualized_spot_volatility,
        ),
        dtype=raw.dtype,
    )
    spot_log_shock = math.sqrt(1.0 / 252.0) * (innovations @ spot_loadings)
    gross_spot = torch.exp(spot_log_shock)
    gross_spot = gross_spot / torch.sum(probabilities * gross_spot)
    next_spot = spot[:, None] * gross_spot[None]
    fixed_call_shock = fixed_calls - base_calls[:, None]
    spot_call_shock = (
        cash_delta[:, None]
        * spot[:, None, None, None]
        * spot_log_shock[None, :, None, None]
    )
    relative_call_shock = (fixed_call_shock + spot_call_shock) / torch.clamp_min(
        base_calls[:, None],
        torch.finfo(raw.dtype).tiny,
    )
    target_calls = base_calls[:, None] * torch.exp(relative_call_shock)
    target_calls = target_calls / torch.sum(
        probabilities[None, :, None, None]
        * target_calls
        / torch.clamp_min(base_calls[:, None], torch.finfo(raw.dtype).tiny),
        dim=1,
        keepdim=True,
    )
    target_calls = target_calls[:, :, live]

    fixed_density_shock = (
        density_step.densities_by_branch[:, :, live]
        - surface.densities[:, None, live]
    )
    reference_density = baseline_density[:, None] + fixed_density_shock
    flat_spot = next_spot.reshape(-1)
    flat_forward = (
        successor_forward[:, None]
        .expand(-1, num_branches, -1)
        .reshape(-1, successor.geometry.num_model_expiries)
    )
    flat_eta = (
        successor_eta[:, None]
        .expand(-1, num_branches, -1)
        .reshape(-1, successor.geometry.num_model_expiries)
    )
    flat_support = (
        successor_support[:, None]
        .expand(-1, num_branches, -1, -1)
        .reshape(
            -1,
            successor.geometry.num_model_expiries,
            successor.geometry.num_inner_strikes + 2,
        )
    )
    flat_backbone = (
        successor_backbone[:, None]
        .expand(-1, num_branches, -1)
        .reshape(-1, successor.geometry.num_model_expiries)
    )
    flat_cash_strikes = (
        live_cash_strikes[:, None]
        .expand(-1, num_branches, -1, -1)
        .reshape(-1, live_cash_strikes.shape[1], live_cash_strikes.shape[2])
    )
    flat_target_calls = target_calls.reshape_as(flat_cash_strikes)
    flat_reference_density = reference_density.reshape_as(flat_support)
    recovered_density, condition = recover_density_from_cash_calls(
        spot=flat_spot,
        support=flat_support,
        backbone_total_vols=flat_backbone,
        cash_strikes=flat_cash_strikes,
        cash_calls=flat_target_calls,
        reference_density=flat_reference_density,
    )
    recovered_dlvsq = _recover_dlvsqs_unchecked(
        successor,
        support=flat_support,
        densities=recovered_density,
    )
    geometry = successor.geometry
    minimum_density = torch.amin(recovered_density, dim=(-2, -1))
    minimum_dlvsq_margin = torch.amin(
        recovered_dlvsq - geometry.min_dlvsq,
        dim=(-2, -1),
    )
    maximum_dlvsq_margin = torch.amin(
        geometry.max_dlvsq - recovered_dlvsq,
        dim=(-2, -1),
    )
    valid = (
        torch.isfinite(minimum_density)
        & torch.isfinite(minimum_dlvsq_margin)
        & torch.isfinite(maximum_dlvsq_margin)
        & (minimum_density > 0.0)
        & (minimum_dlvsq_margin > 0.0)
        & (maximum_dlvsq_margin > 0.0)
    ).reshape(num_states, num_branches)

    repriced = _price_from_density(
        spot=flat_spot,
        support=flat_support,
        density=recovered_density,
        backbone_total_vols=flat_backbone,
        cash_strikes=flat_cash_strikes,
    ).reshape_as(target_calls)
    flat_audit_strikes = (
        live_audit_strikes[:, None]
        .expand(-1, num_branches, -1, -1)
        .reshape(-1, live_audit_strikes.shape[1], live_audit_strikes.shape[2])
    )
    audit_prices = _price_from_density(
        spot=flat_spot,
        support=flat_support,
        density=recovered_density,
        backbone_total_vols=flat_backbone,
        cash_strikes=flat_audit_strikes,
    ).reshape(
        num_states,
        num_branches,
        live_audit_strikes.shape[1],
        live_audit_strikes.shape[2],
    )
    conditional_selected_error = torch.sum(
        probabilities[None, :, None, None] * repriced,
        dim=1,
    ) - live_base_calls
    conditional_audit_error = torch.sum(
        probabilities[None, :, None, None] * audit_prices,
        dim=1,
    ) - live_base_audit_calls

    sampled_branch = _sample_branches(
        num_states,
        num_branches,
        generator=generator,
        device=raw.device,
    )
    flat_index = torch.arange(num_states, device=raw.device) * num_branches + sampled_branch
    sampled_valid = valid.reshape(-1)[flat_index]
    sampled_spot = flat_spot[flat_index]
    sampled_calls = repriced.reshape(
        num_states * num_branches,
        repriced.shape[-2],
        repriced.shape[-1],
    )[flat_index]
    selected_dlvsq = recovered_dlvsq[flat_index][sampled_valid]
    selected_forward = flat_forward[flat_index][sampled_valid]
    selected_eta = flat_eta[flat_index][sampled_valid]
    next_raw = raw_from_native_parameters(
        NativeParameters(
            dlvsqs=selected_dlvsq,
            forward_vols=selected_forward,
            eta=selected_eta,
        ),
        geometry=geometry,
    )
    return _BatchResult(
        successor=successor,
        valid_sampled=sampled_valid,
        next_raw_valid=next_raw,
        next_spot_valid=sampled_spot[sampled_valid],
        next_cash_strikes_valid=live_cash_strikes[sampled_valid],
        next_audit_strikes_valid=live_audit_strikes[sampled_valid],
        option_increments=sampled_calls - live_base_calls,
        spot_increments=sampled_spot - spot,
        conditional_state_valid=torch.all(valid, dim=1),
        minimum_density_by_branch=minimum_density,
        minimum_dlvsq_margin_by_branch=minimum_dlvsq_margin,
        maximum_dlvsq_margin_by_branch=maximum_dlvsq_margin,
        condition_numbers=condition,
        conditional_selected_errors=conditional_selected_error,
        conditional_audit_errors=conditional_audit_error,
    )


def _mc_summary(values: torch.Tensor) -> dict[str, float | int]:
    observations = len(values)
    flat = values.reshape(observations, -1)
    mean = torch.mean(flat, dim=0)
    standard_error = torch.std(flat, dim=0, correction=1) / math.sqrt(observations)
    z = mean / torch.clamp_min(standard_error, torch.finfo(values.dtype).tiny)
    return {
        "observations": observations,
        "maximum_absolute_mean": float(torch.amax(torch.abs(mean))),
        "maximum_standard_error": float(torch.amax(standard_error)),
        "maximum_absolute_z": float(torch.amax(torch.abs(z))),
        "num_absolute_z_above_1_96": int(torch.sum(torch.abs(z) > 1.96)),
        "num_absolute_z_above_2_576": int(torch.sum(torch.abs(z) > 2.576)),
        "root_mean_square_z": float(torch.sqrt(torch.mean(z.square()))),
    }


def run(config: Config) -> dict[str, object]:
    """Run the short path experiment and persist a JSON report."""

    _validate(config)
    local_config = LocalConfig(
        output=config.output,
        num_chart_strikes=config.num_chart_strikes,
        annualized_dlvsq_log_volatility=config.annualized_dlvsq_log_volatility,
        annualized_spot_volatility=config.annualized_spot_volatility,
        fixed_relative_grid=True,
        fixed_backbone_annualized_volatility=(
            config.fixed_backbone_annualized_volatility
        ),
    )
    env, decoder, initial_raw, initial_parameters, _ = _problem(local_config)
    initial_surface = decoder.forward_batch(
        dlvsqs=initial_parameters.dlvsqs,
        forward_vols=initial_parameters.forward_vols,
        eta=initial_parameters.eta,
    )
    strike_indices = torch.linspace(
        0,
        decoder.geometry.num_inner_strikes - 1,
        config.num_chart_strikes,
        dtype=env.fdtype,
    ).round().to(torch.int64)
    initial_cash_strikes = initial_surface.support_strikes[
        :, :, 1:-1
    ][..., strike_indices]
    initial_audit_strikes = torch.exp(
        torch.linspace(-0.28, 0.20, config.num_audit_strikes, dtype=env.fdtype)
    )[None, None].expand(1, decoder.geometry.num_model_expiries, -1)
    raw = initial_raw.expand(config.num_paths, -1).clone()
    spot = torch.ones(config.num_paths, dtype=env.fdtype)
    cash_strikes = initial_cash_strikes.expand(config.num_paths, -1, -1).clone()
    audit_strikes = initial_audit_strikes.expand(config.num_paths, -1, -1).clone()
    generator = torch.Generator(device=env.device).manual_seed(config.seed)
    started = time.monotonic()
    steps: list[dict[str, object]] = []
    initial_paths = config.num_paths
    for step_index in range(config.num_steps):
        current_paths = len(raw)
        if current_paths == 0:
            break
        batch_results: list[_BatchResult] = []
        for first in range(0, current_paths, config.path_batch_size):
            last = min(first + config.path_batch_size, current_paths)
            batch_results.append(
                _advance_batch(
                    config,
                    decoder,
                    raw=raw[first:last],
                    spot=spot[first:last],
                    cash_strikes=cash_strikes[first:last],
                    audit_strikes=audit_strikes[first:last],
                    generator=generator,
                )
            )
            print(
                json.dumps(
                    {
                        "elapsed_seconds": time.monotonic() - started,
                        "step": step_index + 1,
                        "total_steps": config.num_steps,
                        "completed_paths": last,
                        "total_paths": current_paths,
                    }
                ),
                flush=True,
            )
        successor = batch_results[0].successor
        option_increments = torch.cat(
            tuple(result.option_increments for result in batch_results)
        )
        spot_increments = torch.cat(
            tuple(result.spot_increments for result in batch_results)
        )
        sampled_valid = torch.cat(
            tuple(result.valid_sampled for result in batch_results)
        )
        conditional_valid = torch.cat(
            tuple(result.conditional_state_valid for result in batch_results)
        )
        minimum_density = torch.cat(
            tuple(result.minimum_density_by_branch for result in batch_results)
        )
        minimum_dlvsq_margin = torch.cat(
            tuple(result.minimum_dlvsq_margin_by_branch for result in batch_results)
        )
        maximum_dlvsq_margin = torch.cat(
            tuple(result.maximum_dlvsq_margin_by_branch for result in batch_results)
        )
        conditions = torch.cat(
            tuple(result.condition_numbers.reshape(-1) for result in batch_results)
        )
        conditional_selected_errors = torch.cat(
            tuple(result.conditional_selected_errors for result in batch_results)
        )
        conditional_audit_errors = torch.cat(
            tuple(result.conditional_audit_errors for result in batch_results)
        )
        step_report: dict[str, object] = {
            "step": step_index + 1,
            "input_paths": current_paths,
            "input_business_dtes": [
                float(value) for value in decoder.model_expiries_y * 252.0
            ],
            "output_business_dtes": [
                float(value) for value in successor.model_expiries_y * 252.0
            ],
            "strictly_admissible_conditional_state_fraction": float(
                torch.mean(conditional_valid.to(env.fdtype))
            ),
            "sampled_admissible_branch_fraction": float(
                torch.mean(sampled_valid.to(env.fdtype))
            ),
            "minimum_density": float(torch.amin(minimum_density)),
            "minimum_dlvsq_margin": float(torch.amin(minimum_dlvsq_margin)),
            "minimum_upper_dlvsq_margin": float(torch.amin(maximum_dlvsq_margin)),
            "maximum_condition_number": float(torch.amax(conditions)),
            "99_percent_condition_number": float(torch.quantile(conditions, 0.99)),
            "maximum_conditional_selected_call_error": float(
                torch.amax(torch.abs(conditional_selected_errors))
            ),
            "maximum_conditional_audit_call_error": float(
                torch.amax(torch.abs(conditional_audit_errors))
            ),
            "sampled_spot_increment": _mc_summary(spot_increments[:, None]),
            "sampled_selected_call_increments": _mc_summary(option_increments),
        }
        steps.append(step_report)
        raw = torch.cat(tuple(result.next_raw_valid for result in batch_results))
        spot = torch.cat(tuple(result.next_spot_valid for result in batch_results))
        cash_strikes = torch.cat(
            tuple(result.next_cash_strikes_valid for result in batch_results)
        )
        audit_strikes = torch.cat(
            tuple(result.next_audit_strikes_valid for result in batch_results)
        )
        decoder = successor
        print(
            json.dumps(
                {
                    "elapsed_seconds": time.monotonic() - started,
                    "completed_steps": step_index + 1,
                    "total_steps": config.num_steps,
                    "surviving_paths": len(raw),
                    "estimated_remaining_seconds": (
                        (time.monotonic() - started)
                        / (step_index + 1)
                        * (config.num_steps - step_index - 1)
                    ),
                }
            ),
            flush=True,
        )

    report: dict[str, object] = {
        "schema": SCHEMA,
        "config": {**asdict(config), "output": str(config.output)},
        "scope": {
            "spot_law": "finite exactly mean-one gate law",
            "option_constraints": "selected surviving fixed-cash calls",
            "front_expiry_payoffs_tested": False,
            "full_front_marginal_used": False,
            "independent_paths": True,
        },
        "initial_paths": initial_paths,
        "terminal_admissible_paths": len(raw),
        "terminal_admissible_path_fraction": len(raw) / initial_paths,
        "elapsed_seconds": time.monotonic() - started,
        "steps": steps,
    }
    config.output.mkdir(parents=True, exist_ok=True)
    _atomic_json(report, config.output / "report.json")
    return report


def main() -> None:
    """CLI entry point."""

    print(json.dumps(run(_arguments()), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
