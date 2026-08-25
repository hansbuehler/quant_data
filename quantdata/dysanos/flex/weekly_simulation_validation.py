"""Real-checkpoint GPU diagnostics for the Torch weekly FLEX simulator."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.spread_cost import load_torch_spread_cost_model
from quantdata.dysanos.flex.weekly_dynamics import (
    TorchWeeklyDynamics,
    WeeklyDynamicsFit,
    WeeklyTrainingHistory,
)
from quantdata.dysanos.flex.weekly_simulation import (
    WeeklySanosSimulator,
    load_strong_calibrated_model,
)
from quantdata.dysanos.flex.weekly_validation import (
    exact_weekly_pairs,
    project_strong_states,
)
from quantdata.torch import BaseEnvironment


def _realized_endpoint_indices(
    history: WeeklyTrainingHistory,
    *,
    initial_date: date,
) -> dict[int, int]:
    try:
        initial_index = history.dates.index(initial_date)
    except ValueError as error:
        raise ValueError(f"Initial simulation date is unavailable: {initial_date}") from error
    endpoints: dict[int, int] = {}
    for weeks in (1, 2, 4):
        origin, endpoint = exact_weekly_pairs(history.dates, weeks=weeks)
        matches = np.flatnonzero(origin == initial_index)
        if len(matches) != 1:
            raise ValueError(
                f"Initial date {initial_date} has no exact {weeks}-week endpoint"
            )
        endpoints[weeks] = int(endpoint[int(matches[0])])
    return endpoints


def _mode_diagnostics(
    simulator: WeeklySanosSimulator,
    initial_state: torch.Tensor,
    initial_spot: torch.Tensor,
    *,
    num_paths: int,
    spot_mode: str,
    realized_log_spot: np.ndarray,
    realized_reduced_states: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], Any]:
    generator = torch.Generator(device=simulator.device)
    generator.manual_seed(seed)
    cuda_device = str(simulator.device).startswith("cuda")
    if cuda_device:
        torch.cuda.synchronize(torch.device(simulator.device))
    started = perf_counter()
    with torch.no_grad():
        paths = simulator.simulate_paths(
            initial_state,
            initial_spot,
            num_paths=num_paths,
            num_steps=4,
            spot_mode=spot_mode,
            generator=generator,
        )
    if cuda_device:
        torch.cuda.synchronize(torch.device(simulator.device))
    elapsed = perf_counter() - started
    initial_log_spot = math.log(float(initial_spot.cpu()))
    horizon_results: dict[str, Any] = {}
    for weeks in (1, 2, 4):
        simulated_return = (
            paths.log_spots[:, weeks] - paths.log_spots[:, 0]
        ).detach().cpu().numpy()
        realized_return = float(realized_log_spot[weeks] - initial_log_spot)
        standard_deviation = float(np.std(simulated_return, ddof=0))
        if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
            raise RuntimeError(
                f"{spot_mode} spot distribution collapsed at {weeks} weeks"
            )
        quantiles = {
            coverage: np.quantile(
                simulated_return,
                ((1.0 - coverage) / 2.0, 1.0 - (1.0 - coverage) / 2.0),
            )
            for coverage in (0.50, 0.80, 0.90, 0.95, 0.99)
        }
        simulated_state_mean = (
            paths.reduced_states[:, weeks].mean(dim=0).detach().cpu().numpy()
        )
        horizon_results[f"{weeks}w"] = {
            "realized_log_spot_return": realized_return,
            "simulated_log_spot_return_mean": float(np.mean(simulated_return)),
            "simulated_log_spot_return_std": standard_deviation,
            "realized_spot_z": (
                (realized_return - float(np.mean(simulated_return)))
                / standard_deviation
            ),
            "realized_spot_percentile": float(
                np.mean(simulated_return <= realized_return)
            ),
            "realized_spot_inside_central_intervals": {
                f"{int(100 * coverage)}pct": bool(
                    bounds[0] <= realized_return <= bounds[1]
                )
                for coverage, bounds in quantiles.items()
            },
            "realized_state_vs_simulated_mean_rmse": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            realized_reduced_states[weeks] - simulated_state_mean
                        )
                    )
                )
            ),
        }
    return (
        {
            "num_paths": num_paths,
            "elapsed_seconds": elapsed,
            "path_steps_per_second": num_paths * 4 / elapsed,
            "finite": bool(
                torch.all(torch.isfinite(paths.reduced_states))
                and torch.all(torch.isfinite(paths.log_spots))
            ),
            "horizons": horizon_results,
        },
        paths,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _long_run_surface_diagnostics(
    dynamics: TorchWeeklyDynamics,
    *,
    num_paths: int = 2_048,
    num_weeks: int = 260,
) -> dict[str, float | int | bool]:
    """Compare a five-year simulation from equilibrium with stationary moments."""
    generator = torch.Generator(device=dynamics.device)
    generator.manual_seed(71_927)
    states = torch.zeros(
        (num_paths, dynamics.reduced_state_size),
        dtype=dynamics.fdtype,
        device=dynamics.device,
    )
    log_spot = torch.zeros(num_paths, dtype=dynamics.fdtype, device=dynamics.device)
    with torch.no_grad():
        for _ in range(num_weeks):
            step = dynamics.step(states, log_spot, generator=generator)
            states = step.reduced_state
            log_spot = step.normal_log_spot
    empirical_mean = torch.mean(states, dim=0)
    centered = states - empirical_mean
    empirical_covariance = centered.T @ centered / num_paths
    target = dynamics.stationary_surface_covariance
    target_trace = torch.trace(target)
    empirical_trace = torch.trace(empirical_covariance)
    variance_ratio = empirical_trace / torch.clamp_min(
        target_trace,
        torch.finfo(target.dtype).tiny,
    )
    covariance_relative_error = torch.linalg.vector_norm(
        empirical_covariance - target
    ) / torch.clamp_min(
        torch.linalg.vector_norm(target),
        torch.finfo(target.dtype).tiny,
    )
    mean_scale = torch.sqrt(
        torch.clamp_min(
            target_trace / dynamics.reduced_state_size,
            torch.finfo(target.dtype).tiny,
        )
    )
    normalized_mean_rms = torch.sqrt(torch.mean(empirical_mean.square())) / mean_scale
    finite = bool(
        torch.all(torch.isfinite(states))
        and torch.isfinite(variance_ratio)
        and torch.isfinite(covariance_relative_error)
        and torch.isfinite(normalized_mean_rms)
    )
    return {
        "num_paths": num_paths,
        "num_weeks": num_weeks,
        "finite": finite,
        "stationary_variance_trace_ratio": float(variance_ratio.cpu()),
        "stationary_covariance_relative_frobenius_error": float(
            covariance_relative_error.cpu()
        ),
        "stationary_mean_normalized_rms": float(normalized_mean_rms.cpu()),
        "noncollapsed": bool(finite and 0.1 < variance_ratio < 10.0),
    }


def _gradient_diagnostics(
    simulator: WeeklySanosSimulator,
    initial_state: torch.Tensor,
    initial_spot: torch.Tensor,
    spread_cost_model: Any,
) -> dict[str, float | bool]:
    """Run one real-decoder option-return gradient check."""
    state = initial_state.detach().clone().requires_grad_(True)
    spot = initial_spot.detach().clone().requires_grad_(True)
    generator = torch.Generator(device=simulator.device)
    generator.manual_seed(91_177)
    result = simulator.simulate_option_returns(
        state,
        spot,
        num_paths=2,
        num_steps=1,
        return_mode="period",
        spot_mode="normal",
        spread_cost_model=spread_cost_model,
        generator=generator,
    )
    state_gradient, spot_gradient = torch.autograd.grad(
        result.returns.sum(),
        (state, spot),
    )
    finite = bool(
        torch.all(torch.isfinite(state_gradient))
        and torch.all(torch.isfinite(spot_gradient))
    )
    return {
        "finite": finite,
        "state_gradient_l2": float(torch.linalg.vector_norm(state_gradient).cpu()),
        "spot_gradient_abs": float(torch.abs(spot_gradient).cpu()),
        # Normalized returns are theoretically invariant to a common scaling
        # of initial spot and cash strikes. State sensitivity, not a non-zero
        # spot derivative, is therefore the relevant differentiability gate.
        "state_nonzero": bool(
            finite and torch.linalg.vector_norm(state_gradient) > 0.0
        ),
    }


def run_simulation_diagnostics(
    fit: WeeklyDynamicsFit,
    history: WeeklyTrainingHistory,
    *,
    strong_checkpoint: Path,
    initial_date: date = date(2025, 1, 2),
    normal_paths: int = 32_768,
    implied_paths: int = 4_096,
    option_paths: int = 64,
    long_run_paths: int = 2_048,
    long_run_weeks: int = 260,
    spread_checkpoint: Path = Path(
        "quantdata/dysanos/flex/reports/phase3-spread/feed_forward.pt"
    ),
) -> dict[str, Any]:
    """Run normal/implied real-decoder checks on physical GPU 0."""
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 3 real simulation validation requires CUDA GPU 0")
    env = BaseEnvironment(precision=32, device="cuda:0", debug=False)
    dynamics = TorchWeeklyDynamics(env, fit=fit)
    calibrated_model = load_strong_calibrated_model(env, strong_checkpoint)
    simulator = WeeklySanosSimulator(
        env,
        dynamics=dynamics,
        calibrated_model=calibrated_model,
    )
    spread_cost_model, spread_metadata = load_torch_spread_cost_model(
        spread_checkpoint,
        env=env,
    )
    spread_report = spread_checkpoint.parent / "phase3-spread-ffw.json"
    if not spread_report.exists():
        raise ValueError(f"Spread provenance report is missing: {spread_report}")
    spread_payload = json.loads(spread_report.read_text(encoding="utf-8"))
    expected_split = {
        "train_start": "2022-01-03",
        "train_end": "2024-12-31",
        "validation_start": "2025-01-02",
        "validation_end": "2025-08-29",
        "no_2026": True,
    }
    if spread_payload.get("status") != "complete" or spread_payload.get(
        "split"
    ) != expected_split:
        raise ValueError("Spread model does not prove the locked Phase 3 split")
    if spread_metadata.get("split") != expected_split:
        raise ValueError("Torch spread checkpoint metadata does not prove the split")
    reduced = project_strong_states(fit, history.states)
    endpoints = _realized_endpoint_indices(history, initial_date=initial_date)
    initial_index = history.dates.index(initial_date)
    initial_state = torch.as_tensor(
        reduced[initial_index],
        dtype=env.fdtype,
        device=env.device,
    )
    initial_spot = torch.tensor(
        math.exp(history.log_spot[initial_index]),
        dtype=env.fdtype,
        device=env.device,
    )
    realized_log_spot = np.asarray(
        [
            history.log_spot[initial_index],
            history.log_spot[endpoints[1]],
            history.log_spot[endpoints[2]],
            math.nan,
            history.log_spot[endpoints[4]],
        ]
    )
    realized_reduced = np.stack(
        [
            reduced[initial_index],
            reduced[endpoints[1]],
            reduced[endpoints[2]],
            np.full(fit.reduced_state_size, math.nan),
            reduced[endpoints[4]],
        ]
    )
    normal, normal_batch = _mode_diagnostics(
        simulator,
        initial_state,
        initial_spot,
        num_paths=normal_paths,
        spot_mode="normal",
        realized_log_spot=realized_log_spot,
        realized_reduced_states=realized_reduced,
        seed=4815,
    )
    implied, _ = _mode_diagnostics(
        simulator,
        initial_state,
        initial_spot,
        num_paths=implied_paths,
        spot_mode="implied",
        realized_log_spot=realized_log_spot,
        realized_reduced_states=realized_reduced,
        seed=4815,
    )
    long_run = _long_run_surface_diagnostics(
        dynamics,
        num_paths=long_run_paths,
        num_weeks=long_run_weeks,
    )

    with torch.no_grad():
        audit_states = normal_batch.reduced_states[:512].reshape(
            -1,
            fit.reduced_state_size,
        )
        full_states = dynamics.decode_full_state(audit_states)
        raw = calibrated_model.raw_states_from_states(full_states)
        parameters = bound_native_raw(
            raw,
            geometry=calibrated_model.config.geometry,
        )
        surface = simulator.decode_surface(audit_states)
    saturation = {
        "num_audit_path_steps": len(audit_states),
        "raw_abs_gt_8_fraction": float(torch.mean((torch.abs(raw) > 8.0).float()).cpu()),
        "raw_abs_gt_12_fraction": float(
            torch.mean((torch.abs(raw) > 12.0).float()).cpu()
        ),
        "dlvsq_min": float(torch.min(parameters.dlvsqs).cpu()),
        "dlvsq_max": float(torch.max(parameters.dlvsqs).cpu()),
        "forward_vol_min": float(torch.min(parameters.forward_vols).cpu()),
        "forward_vol_max": float(torch.max(parameters.forward_vols).cpu()),
        "mu_min": float(torch.min(parameters.eta).cpu()),
        "mu_max": float(torch.max(parameters.eta).cpu()),
        "density_min": float(torch.min(surface.densities).cpu()),
        "density_mass_max_abs_error": float(
            torch.max(torch.abs(surface.densities.sum(dim=-1) - 1.0)).cpu()
        ),
        "density_mean_max_abs_error": float(
            torch.max(
                torch.abs(
                    torch.sum(
                        surface.support_strikes * surface.densities,
                        dim=-1,
                    )
                    - 1.0
                )
            ).cpu()
        ),
        "float_density_tolerance": max(
            1.0e-8,
            256.0 * torch.finfo(env.fdtype).eps,
        ),
    }

    option_checks: dict[str, Any] = {}
    for spot_index, spot_mode in enumerate(("normal", "implied")):
        for return_index, return_mode in enumerate(("period", "expiry")):
            generator = torch.Generator(device=simulator.device)
            generator.manual_seed(62_000 + 10 * spot_index + return_index)
            with torch.no_grad():
                result = simulator.simulate_option_returns(
                    initial_state,
                    initial_spot,
                    num_paths=option_paths,
                    num_steps=simulator.num_model_expiries,
                    return_mode=return_mode,
                    spot_mode=spot_mode,
                    spread_cost_model=spread_cost_model,
                    generator=generator,
                )
            nonzero = ~result.grid.is_zero_strike
            proportional_entry_cost = (
                result.entry_costs[..., nonzero]
                / torch.clamp_min(
                    result.entry_prices[..., nonzero],
                    torch.finfo(result.entry_prices.dtype).tiny,
                )
            )
            option_checks[f"{spot_mode}_{return_mode}"] = {
                "shape": list(result.returns.shape),
                "num_options": result.grid.num_options,
                "internal_simulation_steps": result.internal_simulation_steps,
                "finite": bool(torch.all(torch.isfinite(result.returns))),
                "zero_strike_calls": int(torch.sum(result.grid.is_zero_strike).cpu()),
                "proportional_entry_cost_mean": float(
                    torch.mean(proportional_entry_cost).cpu()
                ),
                "proportional_entry_cost_median": float(
                    torch.median(proportional_entry_cost).cpu()
                ),
                "cost_model": "date-balanced linear half-spread attribution",
                "cost_feature_volatility": (
                    "SANOS strike-total-volatility ATM proxy"
                ),
            }
    gradient = _gradient_diagnostics(
        simulator,
        initial_state,
        initial_spot,
        spread_cost_model,
    )
    return {
        "device": str(env.device),
        "precision": 32,
        "initial_date": initial_date.isoformat(),
        "endpoint_dates": {
            f"{weeks}w": history.dates[index].isoformat()
            for weeks, index in endpoints.items()
        },
        "normal_p": normal,
        "implied_spot_test_mode": implied,
        "implied_mode_status": (
            "structural simulator test only; not the calibrated historical-P spot model"
        ),
        "decoder_saturation": saturation,
        "long_run_surface": long_run,
        "real_decoder_gradient": gradient,
        "option_return_checks": option_checks,
        "spread_checkpoint": str(spread_checkpoint),
        "spread_checkpoint_sha256": _sha256(spread_checkpoint),
        "spread_report": str(spread_report),
        "spread_report_sha256": _sha256(spread_report),
        "spread_split": expected_split,
        "nondegenerate": bool(
            normal["finite"]
            and implied["finite"]
            and all(check["finite"] for check in option_checks.values())
            and long_run["noncollapsed"]
            and gradient["finite"]
            and gradient["state_nonzero"]
            and saturation["density_min"]
            >= -saturation["float_density_tolerance"]
            and saturation["density_mass_max_abs_error"]
            <= saturation["float_density_tolerance"]
            and saturation["density_mean_max_abs_error"]
            <= saturation["float_density_tolerance"]
        ),
    }


__all__ = ["run_simulation_diagnostics"]
