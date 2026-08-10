"""Search frozen period portfolios over their conditional Gaussian support.

This diagnostic complements finite conditional Monte Carlo.  It writes each
five-day state continuation as an affine function of independent standard
normal shocks and differentiates the complete frozen portfolio P&L through
the DYSANOS decoder.  Projected Adam searches nested Mahalanobis balls for a
loss.  A negative minimum is a counterexample to arbitrage; a non-negative
minimum is only a numerical search result, not a global certificate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.verbose import Context
from scipy.stats import chi2

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.grid.torch_pricer import _bbtorch
from quantdata.dysanos.grid.torch_simulator import (
    _xar1_gaussian_step_factor,
    _xar1_matrix_exp_terms,
)

from .run_incremental_candidate_replay import (
    _conservative_mean_lower,
    _decode_initial_surface,
    _sparse_portfolio,
    recover_candidate_state,
    select_strongest_candidates,
)
from .terminal_universe import expiry_step_offsets
from .torch_prep import GenDysanosExperiment, _load_generator_components

_DEFAULT_SOURCE_REPORT = Path(
    "quantdata/dysanos/grid/gendysanos/reports/required/"
    "dysanos-weekly-1000000-incremental-raw-1e-03.json"
)
_DEFAULT_MONTE_CARLO_REPORT = Path(
    "quantdata/dysanos/grid/gendysanos/reports/incremental-candidate-replay/"
    "dysanos-weekly-1000000-incremental-raw-1e-03-top10-n10000.json"
)
_DEFAULT_OUTPUT = Path(
    "quantdata/dysanos/grid/gendysanos/reports/incremental-candidate-replay/"
    "dysanos-weekly-1000000-incremental-raw-1e-03-top10-adversarial.json"
)
_DEFAULT_BALL_PROBABILITIES = (0.90, 0.99, 0.999, 0.9999, 0.999999)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def project_rows_to_l2_balls_(
    values: torch.Tensor, radii: torch.Tensor
) -> None:
    """Project each row of ``values`` onto its corresponding Euclidean ball."""

    verify_inp(values.ndim >= 2, "'values' must have a leading batch dimension")
    verify_inp(
        radii.shape == (values.shape[0],)
        and bool(torch.all(torch.isfinite(radii) & (radii > 0.0)).item()),
        "'radii' must contain one finite-positive radius per row",
    )
    with torch.no_grad():
        flat = values.reshape(values.shape[0], -1)
        norms = torch.linalg.vector_norm(flat, dim=1)
        factors = torch.minimum(torch.ones_like(norms), radii / norms.clamp_min(1.0e-30))
        values.mul_(factors.reshape((-1,) + (1,) * (values.ndim - 1)))


def first_loss_radius_on_ray(
    pnl_function,
    shocks: torch.Tensor,
    *,
    iterations: int = 50,
) -> tuple[float, float]:
    """Bisect the first sign change from zero toward a loss-producing shock."""

    verify_inp(iterations > 0, lambda: f"'iterations' must be positive, got {iterations}")
    with torch.no_grad():
        zero = torch.zeros_like(shocks)
        pnl_zero = float(pnl_function(zero)[0].item())
        pnl_full = float(pnl_function(shocks)[0].item())
        if pnl_zero < 0.0:
            return 0.0, pnl_zero
        if pnl_full >= 0.0:
            return math.nan, pnl_full
        low = 0.0
        high = 1.0
        pnl_high = pnl_full
        for _ in range(iterations):
            middle = 0.5 * (low + high)
            pnl_middle = float(pnl_function(shocks * middle)[0].item())
            if pnl_middle < 0.0:
                high = middle
                pnl_high = pnl_middle
            else:
                low = middle
        radius = high * float(torch.linalg.vector_norm(shocks).item())
        return radius, pnl_high


def _conditional_daily_transition(sampler) -> dict[str, np.ndarray]:
    """Return the exact one-day affine transition in standardized coordinates."""

    states = sampler.num_xstate
    center = np.asarray(sampler.pcaxar1.state_center, dtype=np.float64)
    scale = np.asarray(sampler.pcaxar1.state_scale, dtype=np.float64)
    raw_k = np.asarray(sampler.pcaxar1.k, dtype=np.float64)
    k = raw_k * scale[:, None] / scale[None, :]
    U0 = (
        np.asarray(sampler.pcaxar1.U0, dtype=np.float64)
        - center[None, :] @ raw_k
    ) / scale[None, :]
    dU = np.asarray(sampler.pcaxar1.dU, dtype=np.float64) / scale[None, :]
    alpha_covariance = sampler.cholesky @ sampler.cholesky.T
    covariance_rate = dU.T @ alpha_covariance @ dU
    h = float((1.0 / DAYS_PER_YEAR) / sampler.pcaxar1.mean_dt)
    A, b = _xar1_matrix_exp_terms(U0=U0, k=k, h=h)
    factor = _xar1_gaussian_step_factor(k=k, C=covariance_rate, h=h, A=A)
    verify_inp(
        center.shape == scale.shape == (states,)
        and A.shape == (states, states)
        and b.shape == (1, states)
        and factor.shape[0] == states,
        "conditional Gaussian transition has inconsistent geometry",
    )
    return {"center": center, "scale": scale, "A": A, "b": b, "factor": factor}


def _raw_state_path_from_shocks(
    shocks: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    transition: torch.Tensor,
    intercept: torch.Tensor,
    factor: torch.Tensor,
) -> torch.Tensor:
    """Map independent standard-normal daily shocks to raw-coordinate states."""

    batches, days, rank = shocks.shape
    states = initial_state.numel()
    assert factor.shape == (states, rank)
    standardized = (initial_state - center) / scale
    current = standardized.expand(batches, states)
    path = [current]
    for day in range(days):
        current = current @ transition + intercept + shocks[:, day] @ factor.T
        path.append(current)
    return torch.stack(path, dim=1) * scale[None, None] + center[None, None]


def _price_alive_calls_at_dt(
    pricer,
    market_strikes: torch.Tensor,
    *,
    dt: float,
    alive: np.ndarray,
) -> torch.Tensor:
    """Price only unexpired maturities, avoiding undefined expired branches."""

    batches, expiries, strikes = market_strikes.shape
    verify_inp(
        alive.shape == (expiries,),
        lambda: f"'alive' must have shape {(expiries,)}, got {alive.shape}",
    )
    values: list[torch.Tensor] = []
    variances = pricer.sqrtVars.square()
    smoothing_scale = (
        0.0 if pricer.smoothness is None else math.sqrt(pricer.smoothness)
    )
    for expiry_index in range(expiries):
        if not alive[expiry_index]:
            values.append(
                torch.zeros(
                    (batches, strikes),
                    dtype=market_strikes.dtype,
                    device=market_strikes.device,
                )
            )
            continue

        strike = market_strikes[:, expiry_index, :, None]
        expiry = pricer.expiries[:, expiry_index]
        if expiry_index == 0:
            alpha = (expiry - dt) / expiry
            interpolated_variance = variances[:, 0] * alpha
            weights = (
                pricer.xdensity0 * (1.0 - alpha[:, None])
                + pricer.xdensities[:, 0] * alpha[:, None]
            )
            component_spot = pricer.xstrikes[:, 0]
            if smoothing_scale == 0.0:
                value = torch.sum(
                    torch.relu(component_spot[:, None] - strike)
                    * weights[:, None],
                    dim=-1,
                )
            else:
                value = torch.sum(
                    _bbtorch.bs_price(
                        spot=component_spot[:, None],
                        strikes=strike,
                        sqrtVars=(
                            smoothing_scale
                            * torch.sqrt(interpolated_variance)[:, None, None]
                        ),
                    )
                    * weights[:, None],
                    dim=-1,
                )
        else:
            previous_expiry = pricer.expiries[:, expiry_index - 1]
            alpha = (expiry - dt - previous_expiry) / (expiry - previous_expiry)
            interpolated_variance = (
                variances[:, expiry_index] * alpha
                + variances[:, expiry_index - 1] * (1.0 - alpha)
            )
            high_spot = pricer.xstrikes[:, expiry_index]
            low_spot = pricer.xstrikes[:, expiry_index - 1]
            high_weight = pricer.xdensities[:, expiry_index]
            low_weight = pricer.xdensities[:, expiry_index - 1]
            if smoothing_scale == 0.0:
                high = torch.sum(
                    torch.relu(high_spot[:, None] - strike)
                    * high_weight[:, None],
                    dim=-1,
                )
                low = torch.sum(
                    torch.relu(low_spot[:, None] - strike)
                    * low_weight[:, None],
                    dim=-1,
                )
            else:
                sqrt_variance = (
                    smoothing_scale
                    * torch.sqrt(interpolated_variance)[:, None, None]
                )
                high = torch.sum(
                    _bbtorch.bs_price(
                        spot=high_spot[:, None],
                        strikes=strike,
                        sqrtVars=sqrt_variance,
                    )
                    * high_weight[:, None],
                    dim=-1,
                )
                low = torch.sum(
                    _bbtorch.bs_price(
                        spot=low_spot[:, None],
                        strikes=strike,
                        sqrtVars=sqrt_variance,
                    )
                    * low_weight[:, None],
                    dim=-1,
                )
            value = high * alpha[:, None] + low * (1.0 - alpha[:, None])
        values.append(value)
    return torch.stack(values, dim=1)


def _candidate_pnl_function(
    *,
    model,
    initial_state: np.ndarray,
    action: np.ndarray,
    normalized_strikes: np.ndarray,
    normalized_calls: np.ndarray,
    expiries: np.ndarray,
    expiry_offsets: np.ndarray,
    transition_geometry: dict[str, np.ndarray],
    days: int,
    cost_rate: float,
):
    """Build a differentiable exact-state P&L function for one frozen action."""

    device = model.env.device
    dtype = model.env.fdtype

    def tensor(value) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=device)

    torch_initial_state = tensor(initial_state)
    center = tensor(transition_geometry["center"])
    scale = tensor(transition_geometry["scale"])
    transition = tensor(transition_geometry["A"])
    intercept = tensor(transition_geometry["b"])
    factor = tensor(transition_geometry["factor"])
    torch_action = tensor(action)
    torch_strikes = tensor(normalized_strikes)
    torch_calls = tensor(normalized_calls)
    torch_expiries = tensor(expiries[None])
    offsets = np.asarray(expiry_offsets, dtype=np.int64)
    alive = offsets > days
    spot_instruments = int(expiries.size)
    verify_inp(
        torch_action.numel() == spot_instruments + torch_calls.numel(),
        lambda: (
            f"candidate action has {torch_action.numel()} instruments, expected "
            f"{spot_instruments + torch_calls.numel()}"
        ),
    )

    initial_spot = torch.exp(torch_initial_state[0])
    cash_strikes = torch_strikes * initial_spot
    entry_calls = torch_calls * initial_spot
    fixed_cost = cost_rate * (
        torch.sum(torch.abs(torch_action[:spot_instruments])) * initial_spot
        + torch.sum(
            torch.abs(torch_action[spot_instruments:]).reshape_as(entry_calls)
            * entry_calls
        )
    )

    def pnl(shocks: torch.Tensor) -> torch.Tensor:
        path = _raw_state_path_from_shocks(
            shocks,
            initial_state=torch_initial_state,
            center=center,
            scale=scale,
            transition=transition,
            intercept=intercept,
            factor=factor,
        )
        spots = torch.exp(path[..., 0])
        endpoint_spot = spots[:, days]
        endpoint_pricer = model.decoder(
            state=path[:, days, 1:], expiries=torch_expiries, training=False
        )
        endpoint_relative_strikes = cash_strikes[None] / endpoint_spot[:, None, None]
        endpoint_normalized_calls = _price_alive_calls_at_dt(
            endpoint_pricer,
            endpoint_relative_strikes,
            dt=days / DAYS_PER_YEAR,
            alive=alive,
        )

        values: list[torch.Tensor] = []
        for expiry_index, offset in enumerate(offsets):
            if offset <= days:
                settlement_spot = spots[:, int(offset), None]
                value = torch.relu(settlement_spot - cash_strikes[expiry_index][None])
            else:
                value = torch.clamp_min(
                    endpoint_normalized_calls[:, expiry_index], 0.0
                ) * endpoint_spot[:, None]
            values.append(value)
        endpoint_calls = torch.stack(values, dim=1)
        spot_endpoints = torch.stack(
            [
                spots[:, int(offset)] if offset <= days else endpoint_spot
                for offset in offsets
            ],
            dim=1,
        )
        returns = torch.cat(
            (
                spot_endpoints - initial_spot,
                (endpoint_calls - entry_calls[None]).flatten(start_dim=1),
            ),
            dim=1,
        )
        result = returns @ torch_action - fixed_cost
        if not bool(torch.all(torch.isfinite(result)).item()):
            raise FloatingPointError("adversarial replay produced NaN or infinity")
        return result

    return pnl, int(factor.shape[1])


def _optimize_candidate(
    *,
    pnl_function,
    shock_days: int,
    shock_rank: int,
    ball_probabilities: tuple[float, ...],
    restarts: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    dimensions = shock_days * shock_rank
    radii_np = np.sqrt(chi2.ppf(np.asarray(ball_probabilities), df=dimensions))
    radii = torch.as_tensor(
        np.repeat(radii_np, restarts), dtype=dtype, device=device
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    shocks = torch.randn(
        (len(radii), shock_days, shock_rank),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    flat_norm = torch.linalg.vector_norm(shocks.flatten(start_dim=1), dim=1)
    radial = torch.rand(
        len(radii), dtype=dtype, device=device, generator=generator
    ).pow(1.0 / dimensions)
    shocks *= (radii * radial / flat_norm.clamp_min(1.0e-30))[:, None, None]
    for probability_index in range(len(ball_probabilities)):
        shocks[probability_index * restarts].zero_()
    shocks = torch.nn.Parameter(shocks)
    optimizer = torch.optim.Adam([shocks], lr=learning_rate)
    best_pnl = torch.full((len(radii),), math.inf, dtype=dtype, device=device)
    best_shocks = torch.zeros_like(shocks)

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        pnl = pnl_function(shocks)
        with torch.no_grad():
            improved = pnl < best_pnl
            best_pnl[improved] = pnl[improved]
            best_shocks[improved] = shocks[improved]
        (pnl.sum() * 1.0e6).backward()
        if shocks.grad is None or not bool(torch.all(torch.isfinite(shocks.grad)).item()):
            raise FloatingPointError("adversarial replay gradient is NaN or infinity")
        optimizer.step()
        project_rows_to_l2_balls_(shocks, radii)

    with torch.no_grad():
        pnl = pnl_function(shocks)
        improved = pnl < best_pnl
        best_pnl[improved] = pnl[improved]
        best_shocks[improved] = shocks[improved]

    results: list[dict[str, Any]] = []
    previous_minimum = math.inf
    for probability_index, (probability, radius) in enumerate(
        zip(ball_probabilities, radii_np, strict=True)
    ):
        selection = slice(
            probability_index * restarts, (probability_index + 1) * restarts
        )
        local_index = int(torch.argmin(best_pnl[selection]).item())
        index = probability_index * restarts + local_index
        minimum = float(best_pnl[index].item())
        selected_shocks = best_shocks[index : index + 1].detach()
        shock_norm = float(torch.linalg.vector_norm(selected_shocks).item())
        ray_radius, ray_pnl = first_loss_radius_on_ray(
            pnl_function, selected_shocks
        )
        radial_tail = (
            float(chi2.sf(ray_radius**2, df=dimensions))
            if math.isfinite(ray_radius)
            else None
        )
        results.append(
            {
                "ball_probability": probability,
                "ball_radius": float(radius),
                "minimum_pnl_found": minimum,
                "minimizer_shock_norm": shock_norm,
                "loss_found": minimum < 0.0,
                "first_loss_radius_on_selected_ray": (
                    ray_radius if math.isfinite(ray_radius) else None
                ),
                "pnl_at_first_loss_bisection_point": ray_pnl,
                "gaussian_radial_tail_beyond_first_loss_radius": radial_tail,
                "minimum_not_worse_than_previous_ball": minimum <= previous_minimum + 1.0e-9,
            }
        )
        previous_minimum = min(previous_minimum, minimum)
    zero_pnl = float(
        pnl_function(
            torch.zeros(
                (1, shock_days, shock_rank), dtype=dtype, device=device
            )
        )[0].item()
    )
    return {
        "standard_normal_dimensions": dimensions,
        "daily_shock_rank": shock_rank,
        "zero_shock_pnl": zero_pnl,
        "balls": results,
        "any_loss_found": any(item["loss_found"] for item in results),
        "minimum_pnl_found": min(item["minimum_pnl_found"] for item in results),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the automatic-differentiation support search."""

    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    configuration = source_report["configuration"]
    expected = {
        "generator": "dysanos",
        "trading_frequency": "weekly",
        "return_horizon": "incremental",
        "hedge": "raw",
        "cost_rate": 1.0e-3,
        "path_count": 1_000_000,
    }
    verify_inp(
        all(configuration[name] == value for name, value in expected.items()),
        lambda: f"source report is not the requested experiment: {configuration}",
    )
    verify_inp(
        all(0.0 < value < 1.0 for value in args.ball_probabilities),
        "all Gaussian-ball probabilities must lie strictly between zero and one",
    )
    candidates = select_strongest_candidates(source_report, count=args.candidates)
    experiment = GenDysanosExperiment(
        num_samples=2,
        num_alpha=None,
        use_matrix_exp=True,
        bandwidth=1.0,
        trading_frequency="weekly",
        return_horizon="incremental",
        num_trading_times=10,
        seed=args.seed,
        importance_sampling=None,
    )
    grid_config, _, model, sampler = _load_generator_components(
        experiment, verbose=Context.quiet
    )
    compute_device = torch.device(args.device)
    verify_inp(
        compute_device.type != "cuda" or torch.cuda.is_available(),
        lambda: f"CUDA device {args.device!r} was requested but is unavailable",
    )
    # Generator fitting is intentionally CPU-only. Only after the sampler has
    # captured the fitted NumPy transition do we move the differentiable
    # decoder and its non-buffer NMS grid to the requested replay device.
    model.env.device = compute_device
    model.to(compute_device)
    model.decoder.xnms = model.decoder.xnms.to(compute_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    expiries = np.asarray(grid_config.grid_expiries, dtype=np.float64)
    expiry_offsets, expiry_error = expiry_step_offsets(
        expiries, dt=1.0 / DAYS_PER_YEAR
    )
    transition_geometry = _conditional_daily_transition(sampler)
    results: list[dict[str, Any]] = []
    monte_carlo = (
        json.loads(args.monte_carlo_report.read_text(encoding="utf-8"))
        if args.monte_carlo_report.exists()
        else None
    )

    for rank, candidate in enumerate(candidates, start=1):
        initial_state = recover_candidate_state(source_report, candidate)
        action = np.asarray(candidate["reference_solution"]["action"], dtype=np.float64)
        normalized_strikes, normalized_calls, _ = _decode_initial_surface(
            model, state=initial_state, expiries=expiries
        )
        pnl_function, shock_rank = _candidate_pnl_function(
            model=model,
            initial_state=initial_state,
            action=action,
            normalized_strikes=normalized_strikes,
            normalized_calls=normalized_calls,
            expiries=expiries,
            expiry_offsets=expiry_offsets,
            transition_geometry=transition_geometry,
            days=5,
            cost_rate=1.0e-3,
        )
        optimization = _optimize_candidate(
            pnl_function=pnl_function,
            shock_days=5,
            shock_rank=shock_rank,
            ball_probabilities=args.ball_probabilities,
            restarts=args.restarts,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed + 1_000_003 * rank,
            device=model.env.device,
            dtype=model.env.fdtype,
        )
        monte_carlo_summary = (
            monte_carlo["candidates"][rank - 1]["conditional_replay"]
            if monte_carlo is not None
            else None
        )
        results.append(
            {
                "rank": rank,
                "source_trading_time_index": int(candidate["trading_time_index"]),
                "source_query_index": int(candidate["query_index"]),
                "ranking_score": _conservative_mean_lower(candidate),
                "starting_state": initial_state.tolist(),
                "portfolio": {
                    "l1_norm": float(np.abs(action).sum()),
                    "nonzero_positions": _sparse_portfolio(
                        action=action,
                        normalized_strikes=normalized_strikes,
                        normalized_calls=normalized_calls,
                        expiries=expiries,
                        spot=math.exp(float(initial_state[0])),
                    ),
                },
                "conditional_monte_carlo": monte_carlo_summary,
                "automatic_differentiation_search": optimization,
            }
        )
        print(
            f"candidate {rank}: min={optimization['minimum_pnl_found']:.8g}, "
            f"loss={optimization['any_loss_found']}",
            flush=True,
        )

    output = {
        "status": "complete",
        "experiment": "automatic-differentiation conditional Gaussian support replay",
        "source_report": str(args.source_report),
        "monte_carlo_report": str(args.monte_carlo_report),
        "interpretation": (
            "A negative value is a constructive counterexample to arbitrage. "
            "Failure to find a negative value is not a global certificate. The "
            "reported chi-square tail is radial Gaussian mass outside the first "
            "loss radius, not the probability of portfolio loss."
        ),
        "configuration": {
            "candidates": args.candidates,
            "conditional_days": 5,
            "ball_probabilities": list(args.ball_probabilities),
            "restarts_per_ball": args.restarts,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "device": str(compute_device),
            "seed": args.seed,
            "cost_rate": 1.0e-3,
            "hedge": "raw",
            "maximum_expiry_time_error": expiry_error,
        },
        "candidates": results,
        "candidates_with_constructive_loss": sum(
            item["automatic_differentiation_search"]["any_loss_found"]
            for item in results
        ),
    }
    _atomic_json(args.output, output)
    print(f"Wrote {args.output}", flush=True)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, default=_DEFAULT_SOURCE_REPORT)
    parser.add_argument(
        "--monte-carlo-report", type=Path, default=_DEFAULT_MONTE_CARLO_REPORT
    )
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--restarts", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--ball-probabilities",
        type=float,
        nargs="+",
        default=list(_DEFAULT_BALL_PROBABILITIES),
    )
    parser.add_argument("--seed", type=int, default=2_026_807)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    arguments.ball_probabilities = tuple(arguments.ball_probabilities)
    run(arguments)
