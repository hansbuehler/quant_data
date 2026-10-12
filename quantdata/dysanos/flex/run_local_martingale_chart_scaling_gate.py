"""Test the local DYSANOS chart around an exact fixed-grid SDLV law.

The fixed-grid density-martingale generator supplies four-factor, exactly
zero-mean call-price shocks.  Spot is an exactly mean-one binary exponential
branch and forward/backbone volatilities move on the native shared DYSANOS
support.  For each branch we solve the square local chart from spot, native
volatility coordinates, and selected fixed-cash calls back to densities and
DLVs by the minimum-relative-change completion.

The experiment repeats this construction at 1, 1/2, 1/4, and 1/8 business
day, reducing the residual expiry clock at the same time.  It also
differentiates the relative-density projection's KKT system implicitly along
all four factors.  The latter uses explicit finite differences of the Black
pricing matrix followed by the analytic differentiated linear solve; it does
not use automatic differentiation or differentiate through a solver.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from quantdata.dysanos.flex.config import (
    FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
    FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.lvsq_drift import (
    density_martingale_lvsq_step,
    fixed_contract_lvsq_drift,
    smooth_lvsq_log_loadings,
)
from quantdata.dysanos.flex.martingale_coordinate_chart import (
    black_call_matrix,
    decode_martingale_coordinates,
    implicit_linear_system_directional_derivatives,
)
from quantdata.dysanos.flex.model import NativeParameters, raw_from_native_parameters
from quantdata.dysanos.flex.sanos import FlexSanosDecoder, SanosSurface
from quantdata.torch import BaseEnvironment

SCHEMA = "dysanos-local-martingale-chart-scaling-gate-v2"


@dataclass(frozen=True, kw_only=True)
class Config:
    """Controls for the shrinking-step local chart experiment."""

    output: Path
    business_day_fractions: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
    num_inner_strikes: int = 6
    num_chart_strikes: int = 3
    annualized_dlvsq_log_volatility: float = 0.05
    annualized_forward_log_volatility: float = 0.05
    annualized_spot_volatility: float = 0.10
    derivative_step: float = 1.0e-3
    fixed_relative_grid: bool = False
    fixed_backbone_annualized_volatility: float | None = None


def _arguments() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-inner-strikes", type=int, default=6)
    parser.add_argument("--num-chart-strikes", type=int, default=3)
    parser.add_argument(
        "--annualized-dlvsq-log-volatility", type=float, default=0.05
    )
    parser.add_argument(
        "--annualized-forward-log-volatility", type=float, default=0.05
    )
    parser.add_argument("--annualized-spot-volatility", type=float, default=0.10)
    parser.add_argument("--derivative-step", type=float, default=1.0e-3)
    parser.add_argument("--fixed-relative-grid", action="store_true")
    parser.add_argument(
        "--fixed-backbone-annualized-volatility",
        type=float,
        default=None,
    )
    arguments = parser.parse_args()
    return Config(
        output=arguments.output.resolve(),
        num_inner_strikes=arguments.num_inner_strikes,
        num_chart_strikes=arguments.num_chart_strikes,
        annualized_dlvsq_log_volatility=(
            arguments.annualized_dlvsq_log_volatility
        ),
        annualized_forward_log_volatility=(
            arguments.annualized_forward_log_volatility
        ),
        annualized_spot_volatility=arguments.annualized_spot_volatility,
        derivative_step=arguments.derivative_step,
        fixed_relative_grid=arguments.fixed_relative_grid,
        fixed_backbone_annualized_volatility=(
            arguments.fixed_backbone_annualized_volatility
        ),
    )


def _validate_config(config: Config) -> None:
    if config.output in (Path("/tmp"), Path("/data")):
        raise ValueError("output must be a project-specific directory")
    positive = (
        *config.business_day_fractions,
        config.annualized_dlvsq_log_volatility,
        config.annualized_forward_log_volatility,
        config.annualized_spot_volatility,
        config.derivative_step,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("time steps, volatilities, and derivative step must be positive")
    if (
        config.fixed_backbone_annualized_volatility is not None
        and (
            not math.isfinite(config.fixed_backbone_annualized_volatility)
            or config.fixed_backbone_annualized_volatility <= 0.0
        )
    ):
        raise ValueError("fixed backbone annualized volatility must be positive")
    if config.num_inner_strikes != 6:
        raise ValueError("the v1 gate uses its explicit six-node reference state")
    if not 1 <= config.num_chart_strikes <= config.num_inner_strikes:
        raise ValueError("num_chart_strikes must lie between one and six")


def _atomic_json(payload: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _problem(
    config: Config,
) -> tuple[
    BaseEnvironment,
    FlexSanosDecoder,
    torch.Tensor,
    NativeParameters,
    torch.Tensor,
]:
    # The gate reports excursions beyond the admissible chart instead of
    # stopping inside the decoder's debug assertions on the first excursion.
    env = BaseEnvironment(precision=64, device="cpu", debug=False)
    moving_geometry = FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=config.num_inner_strikes,
        num_model_expiries=3,
        explicit_model_business_dtes=(1.0, 5.0, 10.0),
        nms_range=(-1.5, 1.0),
    )
    parameters = NativeParameters(
        dlvsqs=torch.tensor(
            [[
                [0.11, 0.09, 0.07, 0.08, 0.10, 0.13],
                [0.12, 0.10, 0.08, 0.09, 0.11, 0.14],
                [0.13, 0.11, 0.09, 0.10, 0.12, 0.15],
            ]],
            dtype=env.fdtype,
        ),
        forward_vols=torch.tensor([[0.22, 0.24, 0.25]], dtype=env.fdtype),
        eta=torch.tensor([[0.18, 0.24, 0.24]], dtype=env.fdtype),
    )
    if config.fixed_relative_grid:
        moving_decoder = FlexSanosDecoder(env, geometry=moving_geometry)
        moving_surface = moving_decoder.forward_batch(
            dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols,
            eta=parameters.eta,
        )
        fixed_inner = tuple(
            float(value)
            for value in moving_surface.support_strikes[0, 0, 1:-1]
        )
        geometry = FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig(
            num_inner_strikes=config.num_inner_strikes,
            num_model_expiries=3,
            explicit_model_business_dtes=(1.0, 5.0, 10.0),
            nms_range=(-1.5, 1.0),
            inner_relative_strikes=fixed_inner,
        )
    else:
        geometry = moving_geometry
    decoder = FlexSanosDecoder(env, geometry=geometry)
    if config.fixed_backbone_annualized_volatility is not None:
        parameters = NativeParameters(
            dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols,
            eta=_fixed_backbone_eta(
                decoder,
                forward_vols=parameters.forward_vols,
                annualized_volatility=(
                    config.fixed_backbone_annualized_volatility
                ),
            ),
        )
    raw = raw_from_native_parameters(parameters, geometry=geometry)
    surface = decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )
    loadings = smooth_lvsq_log_loadings(
        surface.support_strikes[0, 0],
        num_model_expiries=geometry.num_model_expiries,
        annualized_log_volatility=config.annualized_dlvsq_log_volatility,
    )
    return env, decoder, raw, parameters, loadings


def _forward_loadings(config: Config, *, dtype: torch.dtype) -> torch.Tensor:
    shape = torch.tensor(
        [
            [-0.45, 0.75, -0.20, 0.00],
            [-0.30, 0.70, 0.45, -0.25],
            [-0.20, 0.55, 0.70, 0.40],
        ],
        dtype=dtype,
    )
    return (
        config.annualized_forward_log_volatility
        * shape
        / torch.linalg.vector_norm(shape, dim=-1, keepdim=True)
    )


def _fixed_backbone_eta(
    decoder: FlexSanosDecoder,
    *,
    forward_vols: torch.Tensor,
    annualized_volatility: float,
) -> torch.Tensor:
    """Return eta giving Black total vol ``sigma * sqrt(T)`` exactly."""

    target_total = annualized_volatility * torch.sqrt(
        decoder.model_expiries_y
    )
    backbone_forward = torch.diff(
        target_total,
        prepend=torch.zeros_like(target_total[:1]),
    ) / torch.sqrt(decoder.model_dts_y)
    eta = backbone_forward.unsqueeze(0) / forward_vols
    geometry = decoder.geometry
    if not bool(
        torch.all(
            (eta >= geometry.min_eta) & (eta <= geometry.max_eta)
        )
    ):
        raise ValueError(
            "fixed backbone requires eta outside the configured bounds: "
            f"[{float(torch.amin(eta))}, {float(torch.amax(eta))}]"
        )
    return eta


def _decoder(
    env: BaseEnvironment,
    *,
    num_inner_strikes: int,
    business_dtes: tuple[float, ...],
    fixed_inner_relative_strikes: tuple[float, ...] | None,
) -> FlexSanosDecoder:
    geometry = (
        FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig(
            num_inner_strikes=num_inner_strikes,
            num_model_expiries=len(business_dtes),
            explicit_model_business_dtes=business_dtes,
            nms_range=(-1.5, 1.0),
            inner_relative_strikes=fixed_inner_relative_strikes,
        )
        if fixed_inner_relative_strikes is not None
        else FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig(
            num_inner_strikes=num_inner_strikes,
            num_model_expiries=len(business_dtes),
            explicit_model_business_dtes=business_dtes,
            nms_range=(-1.5, 1.0),
        )
    )
    return FlexSanosDecoder(env, geometry=geometry)


def _successor_coordinates(
    env: BaseEnvironment,
    decoder: FlexSanosDecoder,
    surface: SanosSurface,
    *,
    business_day_fraction: float,
    fixed_backbone_annualized_volatility: float | None = None,
) -> tuple[FlexSanosDecoder, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Subtract elapsed front variance and return the residual DYSANOS clock."""

    current_dtes = decoder.model_expiries_y * 252.0
    expiry_tolerance = (
        64.0
        * torch.finfo(current_dtes.dtype).eps
        * torch.clamp_min(torch.amax(current_dtes), 1.0)
    )
    live = torch.nonzero(
        current_dtes > business_day_fraction + expiry_tolerance,
        as_tuple=False,
    ).flatten()
    if len(live) == 0:
        raise ValueError("calendar step must leave at least one retained expiry")
    residual_dtes = current_dtes[live] - business_day_fraction
    if not bool(torch.all(residual_dtes > 0.0)):
        raise ValueError("calendar step must leave every retained expiry positive")
    successor = _decoder(
        env,
        num_inner_strikes=decoder.geometry.num_inner_strikes,
        business_dtes=tuple(float(value) for value in residual_dtes),
        fixed_inner_relative_strikes=(
            decoder.geometry.inner_relative_strikes
            if isinstance(
                decoder.geometry,
                FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
            )
            else None
        ),
    )
    elapsed_fraction = torch.as_tensor(
        business_day_fraction,
        dtype=surface.densities.dtype,
        device=surface.densities.device,
    )
    support_elapsed_variance = surface.total_vols[:, :1].square() * elapsed_fraction
    backbone_elapsed_variance = (
        surface.backbone_total_vols[:, :1].square() * elapsed_fraction
    )
    fixed_relative_grid = isinstance(
        decoder.geometry,
        FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
    )
    support_total = torch.sqrt(
        torch.clamp_min(
            surface.total_vols[:, live].square() - support_elapsed_variance,
            0.0,
        )
    )
    if fixed_backbone_annualized_volatility is None:
        backbone_total = torch.sqrt(
            torch.clamp_min(
                surface.backbone_total_vols[:, live].square()
                - backbone_elapsed_variance,
                0.0,
            )
        )
    else:
        backbone_total = fixed_backbone_annualized_volatility * torch.sqrt(
            successor.model_expiries_y
        ).unsqueeze(0)
    sqrt_dts = torch.sqrt(successor.model_dts_y)
    forward = (
        surface.forward_vols[:, live]
        if fixed_relative_grid
        else torch.diff(
            support_total,
            dim=-1,
            prepend=torch.zeros_like(support_total[:, :1]),
        )
        / sqrt_dts
    )
    backbone_forward = torch.diff(
        backbone_total,
        dim=-1,
        prepend=torch.zeros_like(backbone_total[:, :1]),
    ) / sqrt_dts
    eta = backbone_forward / forward
    return successor, forward, eta, live


def _spot_loadings(config: Config, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(
        [config.annualized_spot_volatility, 0.0, 0.0, 0.0], dtype=dtype
    )


def _price_from_density(
    *,
    spot: torch.Tensor,
    support: torch.Tensor,
    density: torch.Tensor,
    backbone_total_vols: torch.Tensor,
    cash_strikes: torch.Tensor,
) -> torch.Tensor:
    pricing = black_call_matrix(
        support,
        cash_strikes / spot[:, None, None],
        backbone_total_vols,
    )
    return spot[:, None, None] * torch.einsum(
        "btkn,btn->btk", pricing, density
    )


def _cash_delta(
    *,
    support: torch.Tensor,
    density: torch.Tensor,
    backbone_total_vols: torch.Tensor,
    cash_strikes: torch.Tensor,
) -> torch.Tensor:
    """Return analytic spot deltas of the fixed-relative mixture."""

    node = support.unsqueeze(-2)
    strike = cash_strikes.unsqueeze(-1)
    volatility = backbone_total_vols[..., None, None]
    safe_volatility = torch.clamp_min(volatility, torch.finfo(support.dtype).tiny)
    d1 = (
        torch.log(node / strike) + 0.5 * safe_volatility.square()
    ) / safe_volatility
    component_delta = node * torch.special.ndtr(d1)
    component_delta = torch.where(
        volatility > 0.0,
        component_delta,
        node * (node > strike).to(node.dtype),
    )
    return torch.einsum("btkn,btn->btk", component_delta, density)


def _chart_system(
    decoder: FlexSanosDecoder,
    *,
    spot: torch.Tensor,
    forward_vols: torch.Tensor,
    eta: torch.Tensor,
    cash_strikes: torch.Tensor,
    cash_calls: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, support = decoder._support_strikes_from_forward_vols(forward_vols)
    backbone_forward = forward_vols * eta
    backbone_total = decoder._total_vols(backbone_forward)
    pricing = black_call_matrix(
        support,
        cash_strikes / spot[:, None, None],
        backbone_total,
    )
    matrix = torch.cat(
        (
            torch.ones_like(support[:, :, None]),
            support[:, :, None],
            pricing,
        ),
        dim=2,
    )
    right_hand_side = torch.cat(
        (
            torch.ones(
                (*spot.shape, support.shape[1], 2),
                dtype=support.dtype,
                device=support.device,
            ),
            cash_calls / spot[:, None, None],
        ),
        dim=-1,
    )
    return matrix, right_hand_side


def _relative_projection_system(
    matrix: torch.Tensor,
    right_hand_side: torch.Tensor,
    reference_density: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the KKT system for minimum squared relative density change."""

    if matrix.ndim != 3 or right_hand_side.shape != matrix.shape[:2]:
        raise ValueError("pricing matrix and right-hand side do not align")
    if reference_density.shape != matrix.shape[:1] + matrix.shape[-1:]:
        raise ValueError("reference density does not align with pricing columns")
    expiries, constraints, nodes = matrix.shape
    weighted = matrix * reference_density[:, None]
    identity = torch.eye(nodes, dtype=matrix.dtype, device=matrix.device)[None].expand(
        expiries, -1, -1
    )
    zero = torch.zeros(
        (expiries, constraints, constraints),
        dtype=matrix.dtype,
        device=matrix.device,
    )
    system = torch.cat(
        (
            torch.cat((identity, weighted.mT), dim=-1),
            torch.cat((weighted, zero), dim=-1),
        ),
        dim=-2,
    )
    residual = right_hand_side - torch.einsum(
        "tij,tj->ti", matrix, reference_density
    )
    rhs = torch.cat(
        (
            torch.zeros(
                (expiries, nodes), dtype=matrix.dtype, device=matrix.device
            ),
            residual,
        ),
        dim=-1,
    )
    return system, rhs


def _price_grid(
    decoder: FlexSanosDecoder,
    surface: SanosSurface,
    *,
    spot: torch.Tensor,
    cash_strikes: torch.Tensor,
) -> torch.Tensor:
    batch, expiries, num_strikes = cash_strikes.shape
    query_expiries = decoder.model_expiries_y.reshape(1, expiries, 1).expand(
        batch, expiries, num_strikes
    )
    normalized = decoder.price_calls_batch(
        surface,
        expiries_y=query_expiries.reshape(batch, -1),
        strikes=(cash_strikes / spot[:, None, None]).reshape(batch, -1),
    ).reshape(batch, expiries, num_strikes)
    return spot[:, None, None] * normalized


def _implicit_diagnostics(
    config: Config,
    decoder: FlexSanosDecoder,
    *,
    raw: torch.Tensor,
    parameters: NativeParameters,
    loadings: torch.Tensor,
    surface: SanosSurface,
    cash_strikes: torch.Tensor,
    base_calls: torch.Tensor,
    cash_delta: torch.Tensor,
    forward_loadings: torch.Tensor,
    spot_loadings: torch.Tensor,
) -> dict[str, object]:
    local = fixed_contract_lvsq_drift(
        decoder,
        raw,
        raw_log_loadings=loadings,
    )
    base_pricing = black_call_matrix(
        surface.support_strikes,
        cash_strikes,
        surface.backbone_total_vols,
    )
    call_diffusion = torch.einsum(
        "btkn,btnf->btkf", base_pricing, local.density_diffusion
    )[0]
    total_call_diffusion = call_diffusion + (
        cash_delta[0, ..., None] * spot_loadings[None, None]
    )
    relative_call_diffusion = total_call_diffusion / base_calls[0, ..., None]

    def system(
        direction: int, epsilon: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        value = torch.as_tensor(epsilon, dtype=surface.densities.dtype)
        spot = torch.exp(spot_loadings[direction] * value).reshape(1)
        forward = parameters.forward_vols * torch.exp(
            forward_loadings[:, direction][None] * value
        )
        calls = base_calls * torch.exp(
            relative_call_diffusion[..., direction][None] * value
        )
        matrix, right_hand_side = _chart_system(
            decoder,
            spot=spot,
            forward_vols=forward,
            eta=parameters.eta,
            cash_strikes=cash_strikes,
            cash_calls=calls,
        )
        reference_density = (
            surface.densities
            + local.density_diffusion[..., direction] * value
        )
        projection, projection_rhs = _relative_projection_system(
            matrix[0], right_hand_side[0], reference_density[0]
        )
        return projection, projection_rhs, reference_density[0], matrix[0]

    base_matrix, base_rhs, base_reference, base_constraint = system(0, 0.0)
    matrix_first: list[torch.Tensor] = []
    matrix_second: list[torch.Tensor] = []
    rhs_first: list[torch.Tensor] = []
    rhs_second: list[torch.Tensor] = []
    direct_first: list[torch.Tensor] = []
    direct_second: list[torch.Tensor] = []
    epsilon = config.derivative_step
    nodes = base_reference.shape[-1]
    base_solution = torch.linalg.solve(
        base_matrix, base_rhs.unsqueeze(-1)
    ).squeeze(-1)
    base_density = base_reference * (1.0 + base_solution[..., :nodes])
    for direction in range(len(spot_loadings)):
        plus_matrix, plus_rhs, plus_reference, _ = system(direction, epsilon)
        minus_matrix, minus_rhs, minus_reference, _ = system(direction, -epsilon)
        matrix_first.append((plus_matrix - minus_matrix) / (2.0 * epsilon))
        matrix_second.append(
            (plus_matrix - 2.0 * base_matrix + minus_matrix) / epsilon**2
        )
        rhs_first.append((plus_rhs - minus_rhs) / (2.0 * epsilon))
        rhs_second.append((plus_rhs - 2.0 * base_rhs + minus_rhs) / epsilon**2)
        plus_solution = torch.linalg.solve(
            plus_matrix, plus_rhs.unsqueeze(-1)
        ).squeeze(-1)
        minus_solution = torch.linalg.solve(
            minus_matrix, minus_rhs.unsqueeze(-1)
        ).squeeze(-1)
        plus_density = plus_reference * (1.0 + plus_solution[..., :nodes])
        minus_density = minus_reference * (1.0 + minus_solution[..., :nodes])
        direct_first.append((plus_density - minus_density) / (2.0 * epsilon))
        direct_second.append(
            (plus_density - 2.0 * base_density + minus_density) / epsilon**2
        )

    implicit = implicit_linear_system_directional_derivatives(
        matrix=base_matrix,
        right_hand_side=base_rhs,
        matrix_first=torch.stack(matrix_first),
        matrix_second=torch.stack(matrix_second),
        right_hand_side_first=torch.stack(rhs_first),
        right_hand_side_second=torch.stack(rhs_second),
    )
    relative_first = implicit.first[..., :nodes]
    relative_second = implicit.second[..., :nodes]
    reference_first = local.density_diffusion[0].permute(2, 0, 1)
    density_first = reference_first + base_reference[None] * relative_first
    density_second = (
        2.0 * reference_first * relative_first
        + base_reference[None] * relative_second
    )
    direct_first_tensor = torch.stack(direct_first)
    direct_second_tensor = torch.stack(direct_second)
    weighted_constraint = base_constraint * base_reference[:, None]
    constraint_singular_values = torch.linalg.svdvals(weighted_constraint)
    constraint_condition = (
        constraint_singular_values[..., 0]
        / constraint_singular_values[..., -1]
    )
    rank = torch.linalg.matrix_rank(weighted_constraint)
    first_scale = torch.clamp_min(
        torch.amax(torch.abs(direct_first_tensor)),
        torch.finfo(base_matrix.dtype).tiny,
    )
    second_scale = torch.clamp_min(
        torch.amax(torch.abs(direct_second_tensor)),
        torch.finfo(base_matrix.dtype).tiny,
    )
    first_error = torch.amax(torch.abs(density_first - direct_first_tensor))
    second_error = torch.amax(torch.abs(density_second - direct_second_tensor))
    return {
        "automatic_differentiation_used": False,
        "derivative_of_pricing_matrix": "centered numerical difference",
        "inverse_derivative": "analytic implicit linear-system formula",
        "derivative_step": config.derivative_step,
        "minimum_constraint_rank": int(torch.amin(rank)),
        "num_constraints": int(weighted_constraint.shape[-2]),
        "num_density_nodes": int(weighted_constraint.shape[-1]),
        "maximum_weighted_constraint_condition_number": float(
            torch.amax(constraint_condition)
        ),
        "maximum_kkt_condition_number": float(
            torch.amax(implicit.condition_numbers)
        ),
        "maximum_first_order_equation_residual": float(
            implicit.maximum_first_order_equation_residual
        ),
        "maximum_second_order_equation_residual": float(
            implicit.maximum_second_order_equation_residual
        ),
        "maximum_first_derivative_crosscheck_error": float(first_error),
        "relative_first_derivative_crosscheck_error": float(
            first_error / first_scale
        ),
        "maximum_second_derivative_crosscheck_error": float(second_error),
        "relative_second_derivative_crosscheck_error": float(
            second_error / second_scale
        ),
    }


def _run_step(
    config: Config,
    env: BaseEnvironment,
    decoder: FlexSanosDecoder,
    *,
    business_day_fraction: float,
    raw: torch.Tensor,
    parameters: NativeParameters,
    loadings: torch.Tensor,
    surface: SanosSurface,
    cash_strikes: torch.Tensor,
    base_calls: torch.Tensor,
    cash_delta: torch.Tensor,
    forward_loadings: torch.Tensor,
    spot_loadings: torch.Tensor,
) -> dict[str, object]:
    calendar_step_y = business_day_fraction / 252.0
    fixed_step = density_martingale_lvsq_step(
        decoder,
        raw,
        raw_log_loadings=loadings,
        calendar_time_increment_y=calendar_step_y,
    )
    innovations = fixed_step.standardized_innovations
    probabilities = fixed_step.branch_probabilities
    branches = len(innovations)
    successor, successor_forward, successor_eta, live = _successor_coordinates(
        env,
        decoder,
        surface,
        business_day_fraction=business_day_fraction,
        fixed_backbone_annualized_volatility=(
            config.fixed_backbone_annualized_volatility
        ),
    )
    live_strikes = cash_strikes[:, live]
    live_base_calls = base_calls[:, live]
    base_reference_density = surface.densities[:, live]
    baseline = decode_martingale_coordinates(
        successor,
        spot=torch.ones(1, dtype=surface.densities.dtype),
        forward_vols=successor_forward,
        eta=successor_eta,
        cash_strikes=live_strikes,
        cash_calls=live_base_calls,
        reference_densities=base_reference_density,
    )
    support = surface.support_strikes.expand(branches, -1, -1)
    backbone = surface.backbone_total_vols.expand(branches, -1)
    strikes = cash_strikes.expand(branches, -1, -1)
    fixed_calls = _price_from_density(
        spot=torch.ones(branches, dtype=surface.densities.dtype),
        support=support,
        density=fixed_step.densities_by_branch[0],
        backbone_total_vols=backbone,
        cash_strikes=strikes,
    )

    sqrt_h = math.sqrt(calendar_step_y)
    spot_log_shock = sqrt_h * (innovations @ spot_loadings)
    spot = torch.exp(spot_log_shock)
    spot = spot / torch.sum(probabilities * spot)
    forward_log_shock = sqrt_h * (innovations @ forward_loadings.mT)
    forward_log_shock = forward_log_shock[:, live]
    forward_vols = successor_forward.expand(branches, -1) * torch.exp(
        forward_log_shock
    )
    forward_vols = forward_vols / torch.sum(
        probabilities[:, None] * forward_vols / successor_forward,
        dim=0,
    )
    eta = (
        (
            successor_forward * successor_eta
        ).expand(branches, -1) / forward_vols
        if config.fixed_backbone_annualized_volatility is not None
        else successor_eta.expand(branches, -1)
    )
    fixed_call_shock = fixed_calls - base_calls
    spot_call_shock = cash_delta.expand(branches, -1, -1) * (
        sqrt_h * (innovations @ spot_loadings)[:, None, None]
    )
    relative_call_shock = (fixed_call_shock + spot_call_shock) / base_calls
    target_calls = base_calls * torch.exp(relative_call_shock)
    target_calls = target_calls / torch.sum(
        probabilities[:, None, None] * target_calls / base_calls,
        dim=0,
    )
    target_calls = target_calls[:, live]
    mean_target_error = torch.sum(
        probabilities[:, None, None] * target_calls, dim=0
    ) - live_base_calls[0]

    fixed_density_shock = (
        fixed_step.densities_by_branch[0][:, live]
        - base_reference_density.expand(branches, -1, -1)
    )
    branch_reference_density = (
        baseline.surface.densities.expand(branches, -1, -1)
        + fixed_density_shock
    )

    recovered = decode_martingale_coordinates(
        successor,
        spot=spot,
        forward_vols=forward_vols,
        eta=eta,
        cash_strikes=live_strikes.expand(branches, -1, -1),
        cash_calls=target_calls,
        reference_densities=branch_reference_density,
    )
    repriced = _price_grid(
        successor,
        recovered.surface,
        spot=spot,
        cash_strikes=live_strikes.expand(branches, -1, -1),
    )
    mean_reprice_error = torch.sum(
        probabilities[:, None, None] * repriced, dim=0
    ) - live_base_calls[0]

    audit_strikes = torch.exp(
        torch.linspace(-0.28, 0.20, 17, dtype=surface.densities.dtype)
    )[None, None].expand(1, len(live), -1)
    base_audit = _price_grid(
        decoder,
        surface,
        spot=torch.ones(1, dtype=surface.densities.dtype),
        cash_strikes=torch.exp(
            torch.linspace(-0.28, 0.20, 17, dtype=surface.densities.dtype)
        )[None, None].expand(1, surface.densities.shape[1], -1),
    )[:, live]
    branch_audit = _price_grid(
        successor,
        recovered.surface,
        spot=spot,
        cash_strikes=audit_strikes.expand(branches, -1, -1),
    )
    audit_error = torch.sum(
        probabilities[:, None, None] * branch_audit, dim=0
    ) - base_audit[0]

    dlvsq_change = recovered.surface.dlvsqs - baseline.surface.dlvsqs
    weighted_mean_dlvsq_change = torch.sum(
        probabilities[:, None, None] * dlvsq_change, dim=0
    )
    weighted_dlvsq_rms = torch.sqrt(
        torch.mean(
            torch.sum(
                probabilities[:, None, None] * dlvsq_change.square(), dim=0
            )
        )
    )
    geometry = decoder.geometry
    bound_distance = torch.minimum(
        recovered.surface.dlvsqs - geometry.min_dlvsq,
        geometry.max_dlvsq - recovered.surface.dlvsqs,
    )
    active_threshold = 1.0e-4
    minimum_decoded_density = torch.amin(recovered.surface.densities)
    admissible = bool(
        recovered.diagnostics.minimum_density > 0.0
        and recovered.diagnostics.minimum_dlvsq_margin > 0.0
        and recovered.diagnostics.maximum_dlvsq_margin > 0.0
        and minimum_decoded_density >= -1.0e-12
        and recovered.diagnostics.maximum_density_recursion_residual <= 1.0e-10
    )
    return {
        "business_day_fraction": business_day_fraction,
        "calendar_step_y": calendar_step_y,
        "num_branches": branches,
        "initial_business_dtes": [float(value) for value in decoder.model_expiries_y * 252.0],
        "successor_business_dtes": [
            float(value) for value in successor.model_expiries_y * 252.0
        ],
        "fixed_grid": {
            "maximum_density_mean_residual": float(
                fixed_step.maximum_density_mean_residual
            ),
            "maximum_decoder_density_residual": float(
                fixed_step.maximum_decoder_density_residual
            ),
            "minimum_density": float(fixed_step.minimum_density),
            "minimum_dlvsq": float(fixed_step.minimum_dlvsq),
            "maximum_dlvsq": float(fixed_step.maximum_dlvsq),
        },
        "martingale": {
            "spot_mean_error": float(torch.sum(probabilities * spot) - 1.0),
            "maximum_input_call_mean_error": float(
                torch.amax(torch.abs(mean_target_error))
            ),
            "maximum_selected_call_mean_error": float(
                torch.amax(torch.abs(mean_reprice_error))
            ),
            "maximum_off_grid_call_mean_error": float(
                torch.amax(torch.abs(audit_error))
            ),
            "root_mean_square_off_grid_call_mean_error": float(
                torch.sqrt(torch.mean(audit_error.square()))
            ),
        },
        "chart": {
            "maximum_condition_number": float(
                torch.amax(recovered.diagnostics.density_system_condition_numbers)
            ),
            "minimum_density": float(recovered.diagnostics.minimum_density),
            "minimum_decoded_density": float(minimum_decoded_density),
            "minimum_dlvsq_margin": float(
                recovered.diagnostics.minimum_dlvsq_margin
            ),
            "maximum_dlvsq_margin": float(
                recovered.diagnostics.maximum_dlvsq_margin
            ),
            "maximum_density_recursion_residual": float(
                recovered.diagnostics.maximum_density_recursion_residual
            ),
            "maximum_cash_repricing_error": float(
                recovered.diagnostics.maximum_absolute_cash_repricing_error
            ),
            "fraction_within_1e-4_of_a_dlvsq_bound": float(
                torch.mean((bound_distance <= active_threshold).to(bound_distance.dtype))
            ),
            "admissible": admissible,
        },
        "scaling": {
            "maximum_absolute_deterministic_clock_dlvsq_change": float(
                torch.amax(
                    torch.abs(
                        baseline.surface.dlvsqs
                        - parameters.dlvsqs[:, live]
                    )
                )
            ),
            "weighted_root_mean_square_dlvsq_increment": float(
                weighted_dlvsq_rms
            ),
            "rms_increment_over_sqrt_h": float(weighted_dlvsq_rms / sqrt_h),
            "maximum_absolute_mean_dlvsq_drift": float(
                torch.amax(torch.abs(weighted_mean_dlvsq_change))
                / calendar_step_y
            ),
        },
    }


def run(config: Config) -> dict[str, object]:
    """Run every shrinking-step gate and write its JSON report."""

    _validate_config(config)
    env, decoder, raw, parameters, loadings = _problem(config)
    surface = decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )
    strike_indices = torch.linspace(
        0,
        config.num_inner_strikes - 1,
        config.num_chart_strikes,
        dtype=surface.densities.dtype,
    ).round().to(torch.int64)
    cash_strikes = surface.support_strikes[:, :, 1:-1][..., strike_indices].clone()
    base_calls = _price_grid(
        decoder,
        surface,
        spot=torch.ones(1, dtype=surface.densities.dtype),
        cash_strikes=cash_strikes,
    )
    cash_delta = _cash_delta(
        support=surface.support_strikes,
        density=surface.densities,
        backbone_total_vols=surface.backbone_total_vols,
        cash_strikes=cash_strikes,
    )
    forward_loadings = _forward_loadings(config, dtype=surface.densities.dtype)
    if (
        config.fixed_relative_grid
        and config.fixed_backbone_annualized_volatility is not None
    ):
        # Forward volatility only moves the support and Black backbone in this
        # construction.  Once both are frozen it is a redundant coordinate.
        forward_loadings = torch.zeros_like(forward_loadings)
    spot_loadings = _spot_loadings(config, dtype=surface.densities.dtype)

    implicit = _implicit_diagnostics(
        config,
        decoder,
        raw=raw,
        parameters=parameters,
        loadings=loadings,
        surface=surface,
        cash_strikes=cash_strikes,
        base_calls=base_calls,
        cash_delta=cash_delta,
        forward_loadings=forward_loadings,
        spot_loadings=spot_loadings,
    )
    steps: list[dict[str, object]] = []
    for business_day_fraction in config.business_day_fractions:
        try:
            step = _run_step(
                config,
                env,
                decoder,
                business_day_fraction=business_day_fraction,
                raw=raw,
                parameters=parameters,
                loadings=loadings,
                surface=surface,
                cash_strikes=cash_strikes,
                base_calls=base_calls,
                cash_delta=cash_delta,
                forward_loadings=forward_loadings,
                spot_loadings=spot_loadings,
            )
            step["status"] = (
                "passed" if bool(step["chart"]["admissible"]) else "inadmissible"
            )
        except (RuntimeError, ValueError) as error:
            step = {
                "business_day_fraction": business_day_fraction,
                "calendar_step_y": business_day_fraction / 252.0,
                "status": "failed",
                "error": str(error),
            }
        steps.append(step)

    report: dict[str, object] = {
        "schema": SCHEMA,
        "config": {**asdict(config), "output": str(config.output)},
        "construction": {
            "call_coordinate_law": (
                "positive exponential martingale with instantaneous diffusion "
                "from the exact four-factor fixed-grid SDLV cubature"
            ),
            "spot_law": "exactly mean-one exponential Rademacher branch",
            "native_free_coordinates": (
                "fixed spot-relative support and deterministic Black backbone"
                if (
                    config.fixed_relative_grid
                    and config.fixed_backbone_annualized_volatility is not None
                )
                else "stochastic shared support and Black backbone through forward vols"
            ),
            "inverse": (
                "minimum-relative-change fixed-cash chart followed by "
                "algebraic DLV recovery"
            ),
            "automatic_differentiation_used": False,
        },
        "implicit_derivatives": implicit,
        "steps": steps,
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
