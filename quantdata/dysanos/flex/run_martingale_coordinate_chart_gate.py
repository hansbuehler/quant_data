"""Run a small one-day semi-analytic DYSANOS martingale-chart gate.

The initial surface has the inhomogeneous 1D/5D/10D clock. Spot is integrated
from its decoded one-day DYSANOS marginal. The successor surface has the
correct residual 4D/9D clock and stochastic forward volatilities. A linear
program completes its branch densities so selected fixed-cash options are
martingales while every recovered branch DLV stays inside its configured
bounds. No AAD or learned transition enters the construction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from quantdata.dysanos.flex.config import (
    FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.martingale_coordinate_chart import (
    complete_branch_dlvs_to_martingales,
)
from quantdata.dysanos.flex.sanos import FlexSanosDecoder, SanosSurface
from quantdata.torch import BaseEnvironment

SCHEMA = "dysanos-martingale-coordinate-chart-gate-v3"


@dataclass(frozen=True, kw_only=True)
class Config:
    """Controls for the small residual-clock gate."""

    output: Path
    calendar_step_y: float = 1.0 / 252.0
    annualized_forward_log_volatility: float = 0.05
    num_inner_strikes: int = 12
    num_chart_strikes: int = 6
    spot_quadrature_nodes: int = 5
    support_width_multiplier: float = 1.5
    support_spot_elasticity: float = 0.0
    backbone_spot_elasticity: float = 0.0
    density_floor: float = 1.0e-5
    dlvsq_bound_margin: float = 1.0e-4
    smooth_quadratic_projection: bool = False
    dlvsq_projection_weight: float = 0.0
    dlvsq_level_shift: float = 0.0
    dlvsq_skew_multiplier: float = 1.0
    forward_vol_scale: float = 1.0


def _arguments() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calendar-step-y", type=float, default=1.0 / 252.0)
    parser.add_argument(
        "--annualized-forward-log-volatility",
        type=float,
        default=0.05,
    )
    parser.add_argument("--num-inner-strikes", type=int, default=12)
    parser.add_argument("--num-chart-strikes", type=int, default=6)
    parser.add_argument("--spot-quadrature-nodes", type=int, default=5)
    parser.add_argument("--support-width-multiplier", type=float, default=1.5)
    parser.add_argument("--support-spot-elasticity", type=float, default=0.0)
    parser.add_argument("--backbone-spot-elasticity", type=float, default=0.0)
    parser.add_argument("--density-floor", type=float, default=1.0e-5)
    parser.add_argument("--dlvsq-bound-margin", type=float, default=1.0e-4)
    parser.add_argument("--smooth-quadratic-projection", action="store_true")
    parser.add_argument("--dlvsq-projection-weight", type=float, default=0.0)
    parser.add_argument("--dlvsq-level-shift", type=float, default=0.0)
    parser.add_argument("--dlvsq-skew-multiplier", type=float, default=1.0)
    parser.add_argument("--forward-vol-scale", type=float, default=1.0)
    arguments = parser.parse_args()
    return Config(
        output=arguments.output.resolve(),
        calendar_step_y=arguments.calendar_step_y,
        annualized_forward_log_volatility=(
            arguments.annualized_forward_log_volatility
        ),
        num_inner_strikes=arguments.num_inner_strikes,
        num_chart_strikes=arguments.num_chart_strikes,
        spot_quadrature_nodes=arguments.spot_quadrature_nodes,
        support_width_multiplier=arguments.support_width_multiplier,
        support_spot_elasticity=arguments.support_spot_elasticity,
        backbone_spot_elasticity=arguments.backbone_spot_elasticity,
        density_floor=arguments.density_floor,
        dlvsq_bound_margin=arguments.dlvsq_bound_margin,
        smooth_quadratic_projection=arguments.smooth_quadratic_projection,
        dlvsq_projection_weight=arguments.dlvsq_projection_weight,
        dlvsq_level_shift=arguments.dlvsq_level_shift,
        dlvsq_skew_multiplier=arguments.dlvsq_skew_multiplier,
        forward_vol_scale=arguments.forward_vol_scale,
    )


def _validate_config(config: Config) -> None:
    if config.output in (Path("/tmp"), Path("/data")):
        raise ValueError("output must be a project-specific directory")
    values = (
        config.calendar_step_y,
        config.annualized_forward_log_volatility,
        config.support_width_multiplier,
        config.density_floor,
        config.dlvsq_bound_margin,
        config.forward_vol_scale,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("step and forward volatility must be finite and positive")
    elasticities = (
        config.support_spot_elasticity,
        config.backbone_spot_elasticity,
        config.dlvsq_level_shift,
        config.dlvsq_skew_multiplier,
        config.dlvsq_projection_weight,
    )
    if any(not math.isfinite(value) for value in elasticities):
        raise ValueError("spot elasticities must be finite")
    if config.dlvsq_projection_weight < 0.0:
        raise ValueError("DLV projection weight must be nonnegative")
    if (
        config.num_inner_strikes < 6
        or config.num_inner_strikes % 2
        or not 1 <= config.num_chart_strikes <= config.num_inner_strikes
        or config.spot_quadrature_nodes < 2
    ):
        raise ValueError("invalid support, chart-strike or quadrature size")


def _atomic_json(payload: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _decoder(
    env: BaseEnvironment,
    *,
    num_inner_strikes: int,
    business_dtes: tuple[float, ...],
) -> FlexSanosDecoder:
    geometry = FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=num_inner_strikes,
        num_model_expiries=len(business_dtes),
        explicit_model_business_dtes=business_dtes,
        nms_range=(-1.5, 1.0),
    )
    return FlexSanosDecoder(env, geometry=geometry)


def _initial_problem(
    config: Config,
) -> tuple[BaseEnvironment, FlexSanosDecoder, torch.Tensor, torch.Tensor, torch.Tensor]:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    decoder = _decoder(
        env,
        num_inner_strikes=config.num_inner_strikes,
        business_dtes=(1.0, 5.0, 10.0),
    )
    strike = torch.linspace(-1.0, 1.0, config.num_inner_strikes, dtype=env.fdtype)
    expiry_level = torch.tensor((0.11, 0.13, 0.15), dtype=env.fdtype)
    dlvsqs = (
        expiry_level[:, None]
        + config.dlvsq_level_shift
        + 0.035 * strike[None].square()
        + 0.015 * config.dlvsq_skew_multiplier * strike[None]
    )[None]
    forward_vols = config.forward_vol_scale * torch.tensor(
        [[0.22, 0.24, 0.25]],
        dtype=env.fdtype,
    )
    eta = torch.tensor([[0.18, 0.24, 0.24]], dtype=env.fdtype)
    if not bool(
        torch.all(
            (dlvsqs > decoder.geometry.min_dlvsq)
            & (dlvsqs < decoder.geometry.max_dlvsq)
        )
        and torch.all(
            (forward_vols > decoder.geometry.min_forward_vol)
            & (forward_vols < decoder.geometry.max_forward_vol)
        )
    ):
        raise ValueError("synthetic initial surface lies outside decoder bounds")
    return env, decoder, dlvsqs, forward_vols, eta


def _price_cash_grid(
    decoder: FlexSanosDecoder,
    surface: SanosSurface,
    *,
    spot: torch.Tensor,
    cash_strikes: torch.Tensor,
    residual_expiries_y: torch.Tensor,
) -> torch.Tensor:
    """Price a fixed-cash expiry-by-strike grid at each branch."""

    branches, expiries, num_strikes = cash_strikes.shape
    if spot.shape != (branches,) or residual_expiries_y.shape != (expiries,):
        raise ValueError("cash pricing inputs have incompatible shapes")
    query_expiries = residual_expiries_y.reshape(1, expiries, 1).expand(
        branches,
        expiries,
        num_strikes,
    )
    normalized = decoder.price_calls_batch(
        surface,
        expiries_y=query_expiries.reshape(branches, -1),
        strikes=(cash_strikes / spot[:, None, None]).reshape(branches, -1),
    ).reshape(branches, expiries, num_strikes)
    return spot[:, None, None] * normalized


def _remaining_coordinates(
    current: SanosSurface,
    next_decoder: FlexSanosDecoder,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map current cumulative support/backbone variances to the residual clock."""

    support_total = torch.sqrt(
        torch.clamp_min(
            current.total_vols[:, 1:].square() - current.total_vols[:, :1].square(),
            0.0,
        )
    )
    backbone_total = torch.sqrt(
        torch.clamp_min(
            current.backbone_total_vols[:, 1:].square()
            - current.backbone_total_vols[:, :1].square(),
            0.0,
        )
    )
    sqrt_dts = torch.sqrt(next_decoder.model_dts_y)
    forward = torch.diff(
        support_total,
        dim=-1,
        prepend=torch.zeros_like(support_total[:, :1]),
    ) / sqrt_dts
    backbone_forward = torch.diff(
        backbone_total,
        dim=-1,
        prepend=torch.zeros_like(backbone_total[:, :1]),
    ) / sqrt_dts
    return forward, backbone_forward / forward


def _front_quadrature(
    surface: SanosSurface,
    *,
    quadrature_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine every front DLV node, Gaussian node and symmetric vol factor."""

    nodes_numpy, weights_numpy = np.polynomial.hermite.hermgauss(quadrature_nodes)
    dtype = surface.densities.dtype
    nodes = torch.as_tensor(math.sqrt(2.0) * nodes_numpy, dtype=dtype)
    weights = torch.as_tensor(weights_numpy / math.sqrt(math.pi), dtype=dtype)
    support = surface.support_strikes[0, 0]
    density = surface.densities[0, 0]
    support_size = len(support)
    factor_sign = torch.tensor((-1.0, 1.0), dtype=dtype)

    support_index = torch.arange(support_size).repeat_interleave(
        quadrature_nodes * len(factor_sign)
    )
    gaussian_index = torch.arange(quadrature_nodes).repeat_interleave(
        len(factor_sign)
    ).repeat(support_size)
    forward_sign = factor_sign.repeat(quadrature_nodes).repeat(support_size)
    probabilities = (
        density[support_index]
        * weights[gaussian_index]
        / len(factor_sign)
    )
    front_vol = surface.backbone_total_vols[0, 0]
    backbone_multiplier = torch.exp(
        -0.5 * front_vol.square() + front_vol * nodes[gaussian_index]
    )
    backbone_multiplier = backbone_multiplier / torch.sum(
        weights
        * torch.exp(-0.5 * front_vol.square() + front_vol * nodes)
    )
    spot = support[support_index] * backbone_multiplier
    return spot, probabilities, forward_sign


def run(config: Config) -> dict[str, object]:
    """Run the residual-clock martingale-chart gate and return its report."""

    _validate_config(config)
    env, current_decoder, base_dlvsqs, base_forward, base_eta = _initial_problem(config)
    current_surface = current_decoder.forward_batch(
        dlvsqs=base_dlvsqs,
        forward_vols=base_forward,
        eta=base_eta,
    )
    next_decoder = _decoder(
        env,
        num_inner_strikes=config.num_inner_strikes,
        business_dtes=(4.0, 9.0),
    )
    next_forward, next_eta = _remaining_coordinates(current_surface, next_decoder)
    next_forward = config.support_width_multiplier * next_forward
    next_eta = next_eta / config.support_width_multiplier
    next_dlvsqs = base_dlvsqs[:, 1:].clone()
    dtype = base_dlvsqs.dtype

    chart_strikes = torch.exp(
        torch.linspace(-0.13, 0.13, config.num_chart_strikes, dtype=dtype)
    ).reshape(1, 1, -1).expand(1, 2, -1)
    current_calls = _price_cash_grid(
        current_decoder,
        current_surface,
        spot=torch.ones(1, dtype=dtype),
        cash_strikes=chart_strikes,
        residual_expiries_y=current_decoder.model_expiries_y[1:],
    )

    spot, probabilities, forward_sign = _front_quadrature(
        current_surface,
        quadrature_nodes=config.spot_quadrature_nodes,
    )
    branches = len(spot)
    sqrt_h = math.sqrt(config.calendar_step_y)
    expiry_loading = torch.linspace(0.7, 1.0, 2, dtype=dtype)
    centered_log_spot = torch.log(spot) - torch.sum(probabilities * torch.log(spot))
    support_spot_factor = torch.exp(
        config.support_spot_elasticity * centered_log_spot[:, None]
    )
    forward_vols = next_forward.expand(branches, -1) * torch.exp(
        config.annualized_forward_log_volatility
        * sqrt_h
        * forward_sign[:, None]
        * expiry_loading[None]
    ) * support_spot_factor
    eta = next_eta.expand(branches, -1) * torch.exp(
        (
            config.backbone_spot_elasticity
            - config.support_spot_elasticity
        )
        * centered_log_spot[:, None]
    )
    if not bool(
        torch.all(
            (forward_vols > next_decoder.geometry.min_forward_vol)
            & (forward_vols < next_decoder.geometry.max_forward_vol)
        )
        and torch.all(
            (eta > next_decoder.geometry.min_eta)
            & (eta < next_decoder.geometry.max_eta)
        )
    ):
        raise RuntimeError("spot-dependent forward-vol law left the decoder bounds")
    auxiliary = next_decoder.forward_batch(
        dlvsqs=next_dlvsqs.expand(branches, -1, -1),
        forward_vols=forward_vols,
        eta=eta,
    )
    branch_strikes = chart_strikes.expand(branches, -1, -1)
    completion = complete_branch_dlvs_to_martingales(
        next_decoder,
        branch_probabilities=probabilities,
        spot=spot,
        support=auxiliary.support_strikes,
        density0=auxiliary.density0,
        backbone_total_vols=auxiliary.backbone_total_vols,
        reference_densities=auxiliary.densities,
        cash_strikes=branch_strikes,
        target_mean_cash_calls=current_calls[0],
        density_floor=config.density_floor,
        dlvsq_bound_margin=config.dlvsq_bound_margin,
        smooth_quadratic_projection=config.smooth_quadratic_projection,
        dlvsq_projection_weight=config.dlvsq_projection_weight,
    )
    recovered_dlvsqs = completion.dlvsqs
    recovered_surface = next_decoder.forward_batch(
        dlvsqs=recovered_dlvsqs,
        forward_vols=forward_vols,
        eta=eta,
    )
    recovered_calls = _price_cash_grid(
        next_decoder,
        recovered_surface,
        spot=spot,
        cash_strikes=branch_strikes,
        residual_expiries_y=next_decoder.model_expiries_y,
    )

    audit_strikes = torch.exp(torch.linspace(-0.28, 0.20, 17, dtype=dtype))
    audit_grid = audit_strikes.reshape(1, 1, -1).expand(1, 2, -1)
    current_audit_calls = _price_cash_grid(
        current_decoder,
        current_surface,
        spot=torch.ones(1, dtype=dtype),
        cash_strikes=audit_grid,
        residual_expiries_y=current_decoder.model_expiries_y[1:],
    )
    branch_audit_calls = _price_cash_grid(
        next_decoder,
        recovered_surface,
        spot=spot,
        cash_strikes=audit_grid.expand(branches, -1, -1),
        residual_expiries_y=next_decoder.model_expiries_y,
    )
    audit_error = torch.sum(
        probabilities[:, None, None] * branch_audit_calls,
        dim=0,
    ) - current_audit_calls[0]

    front_audit_grid = audit_strikes.reshape(1, 1, -1)
    exact_front_calls = _price_cash_grid(
        current_decoder,
        current_surface,
        spot=torch.ones(1, dtype=dtype),
        cash_strikes=front_audit_grid,
        residual_expiries_y=current_decoder.model_expiries_y[:1],
    )[0, 0]
    quadrature_front_calls = torch.sum(
        probabilities[:, None] * torch.relu(spot[:, None] - audit_strikes[None]),
        dim=0,
    )

    weighted_dlvsq_mean = torch.sum(
        probabilities[:, None, None] * recovered_dlvsqs,
        dim=0,
    )
    weighted_forward_mean = torch.sum(probabilities[:, None] * forward_vols, dim=0)
    dlvsq_drift = (weighted_dlvsq_mean - next_dlvsqs[0]) / config.calendar_step_y
    forward_drift = (weighted_forward_mean - next_forward[0]) / config.calendar_step_y
    dlvsq_factor_loading = torch.sum(
        probabilities[:, None, None]
        * (recovered_dlvsqs - weighted_dlvsq_mean[None])
        * forward_sign[:, None, None],
        dim=0,
    ) / sqrt_h
    lower_dlvsq_distance = recovered_dlvsqs - next_decoder.geometry.min_dlvsq
    upper_dlvsq_distance = next_decoder.geometry.max_dlvsq - recovered_dlvsqs
    active_threshold = 2.0 * config.dlvsq_bound_margin
    weighted_active_lower = torch.sum(
        probabilities[:, None, None]
        * (lower_dlvsq_distance <= active_threshold).to(dtype)
    ) / recovered_dlvsqs.shape[1] / recovered_dlvsqs.shape[2]
    weighted_active_upper = torch.sum(
        probabilities[:, None, None]
        * (upper_dlvsq_distance <= active_threshold).to(dtype)
    ) / recovered_dlvsqs.shape[1] / recovered_dlvsqs.shape[2]
    weighted_relative_density_rms = torch.sqrt(
        torch.sum(
            probabilities[:, None, None]
            * (completion.densities / auxiliary.densities - 1.0).square()
        )
        / completion.densities.shape[1]
        / completion.densities.shape[2]
    )
    mean_chart_calls = torch.sum(
        probabilities[:, None, None] * recovered_calls,
        dim=0,
    )
    density_mass_residual = torch.amax(
        torch.abs(torch.sum(completion.densities, dim=-1) - 1.0)
    )
    density_mean_residual = torch.amax(
        torch.abs(
            torch.sum(completion.densities * auxiliary.support_strikes, dim=-1)
            - 1.0
        )
    )
    density_recursion_residual = torch.amax(
        torch.abs(recovered_surface.densities - completion.densities)
    )

    report: dict[str, object] = {
        "schema": SCHEMA,
        "config": {**asdict(config), "output": str(config.output)},
        "geometry": {
            "initial_business_dtes": [1.0, 5.0, 10.0],
            "successor_business_dtes": [4.0, 9.0],
            "num_inner_strikes": config.num_inner_strikes,
            "num_chart_strikes": config.num_chart_strikes,
            "num_branches": branches,
        },
        "construction": {
            "spot_law": "front DYSANOS density times Gauss-Hermite Black backbone",
            "free_coordinates": "stochastic successor forward vols; eta held at residual values",
            "density_completion": (
                "DLV-bounded linear feasibility with an optional convex projection"
            ),
            "dlv_inverse": "closed-form adjacent-density SANOS identity",
            "automatic_differentiation_used": False,
            "solver": (
                "HiGHS linear feasibility; CLARABEL convex projection when enabled"
            ),
            "selection_method": completion.selection_method,
            "projection_solver_status": completion.projection_solver_status,
        },
        "martingale": {
            "spot_mean_error": float(torch.sum(probabilities * spot) - 1.0),
            "maximum_front_quadrature_call_error": float(
                torch.amax(torch.abs(quadrature_front_calls - exact_front_calls))
            ),
            "maximum_chart_call_mean_error": float(
                torch.amax(torch.abs(mean_chart_calls - current_calls[0]))
            ),
            "maximum_off_grid_call_mean_error": float(torch.amax(torch.abs(audit_error))),
            "root_mean_square_off_grid_call_mean_error": float(
                torch.sqrt(torch.mean(audit_error.square()))
            ),
        },
        "admissibility": {
            "minimum_density": float(torch.amin(completion.densities)),
            "minimum_dlvsq_margin": float(
                torch.amin(recovered_dlvsqs - next_decoder.geometry.min_dlvsq)
            ),
            "maximum_dlvsq_margin": float(
                torch.amin(next_decoder.geometry.max_dlvsq - recovered_dlvsqs)
            ),
            "maximum_mass_residual": float(density_mass_residual),
            "maximum_mean_residual": float(density_mean_residual),
            "maximum_density_recursion_residual": float(density_recursion_residual),
            "maximum_completion_constraint_residual": float(
                max(
                    completion.maximum_equality_residual,
                    completion.maximum_inequality_residual,
                )
            ),
            "maximum_absolute_relative_density_change": float(
                torch.amax(
                    torch.abs(completion.densities / auxiliary.densities - 1.0)
                )
            ),
            "weighted_absolute_density_change": (
                completion.weighted_absolute_density_change
            ),
            "weighted_quadratic_relative_density_change": (
                completion.weighted_quadratic_relative_density_change
            ),
            "weighted_relative_density_rms": float(weighted_relative_density_rms),
            "weighted_root_mean_square_dlvsq_change": (
                completion.weighted_root_mean_square_dlvsq_change
            ),
            "weighted_fraction_near_lower_dlvsq_bound": float(
                weighted_active_lower
            ),
            "weighted_fraction_near_upper_dlvsq_bound": float(
                weighted_active_upper
            ),
        },
        "induced_characteristics": {
            "maximum_absolute_dlvsq_drift": float(torch.amax(torch.abs(dlvsq_drift))),
            "maximum_absolute_forward_vol_drift": float(
                torch.amax(torch.abs(forward_drift))
            ),
            "maximum_absolute_dlvsq_factor_loading": float(
                torch.amax(torch.abs(dlvsq_factor_loading))
            ),
            "forward_vol_drift": forward_drift.tolist(),
            "dlvsq_drift": dlvsq_drift.tolist(),
            "dlvsq_forward_factor_loading": dlvsq_factor_loading.tolist(),
        },
    }
    config.output.mkdir(parents=True, exist_ok=True)
    _atomic_json(report, config.output / "report.json")
    return report


def main() -> None:
    """CLI entry point."""

    report = run(_arguments())
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
