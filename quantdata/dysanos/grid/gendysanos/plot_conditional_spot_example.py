"""Plot conditional physical and widened-proposal spot ranges from a real state."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import date
from pathlib import Path

import numpy as np
from cdxcore.verbose import Context
from scipy.stats import norm

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.grid.torch_simulator import (
    _xar1_gaussian_step_factor,
    _xar1_matrix_exp_terms,
)

from .torch_prep import GenDysanosExperiment, _load_generator_components


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _conditional_mean_surface_path(
    *,
    sampler,
    initial_state: np.ndarray,
    num_steps: int,
    dt: float,
) -> np.ndarray:
    """Continue the selected surface state along its conditional mean path."""

    center = np.asarray(sampler.pcaxar1.state_center, dtype=np.float64)
    scale = np.asarray(sampler.pcaxar1.state_scale, dtype=np.float64)
    raw_k = sampler.pcaxar1.k
    raw_intercept = sampler.pcaxar1.U0
    k = raw_k * scale[:, None] / scale[None, :]
    intercept = (raw_intercept - center[None, :] @ raw_k) / scale[None, :]
    transition, offset = _xar1_matrix_exp_terms(
        U0=intercept,
        k=k,
        h=float(dt / sampler.pcaxar1.mean_dt),
    )
    autonomy_tolerance = 1.0e3 * np.finfo(float).eps * max(
        1.0, float(np.linalg.norm(transition, ord=2))
    )
    if np.max(np.abs(transition[0, 1:])) > autonomy_tolerance:
        raise ValueError(
            "surface conditional mean depends on current spot by "
            f"{transition[0, 1:]}"
        )
    standardized = np.empty((num_steps, initial_state.size), dtype=np.float64)
    standardized[0] = (initial_state - center) / scale
    for step in range(num_steps - 1):
        standardized[step + 1, 0] = standardized[step, 0]
        standardized[step + 1, 1:] = (
            standardized[step, 1:] @ transition[1:, 1:]
            + offset[0, 1:]
        )
    return center[None, :] + standardized * scale[None, :]


def _conditional_log_spot_moments(
    *,
    sampler,
    reference_path: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return conditional log-spot means and standard deviations.

    The complete surface component of ``reference_path`` is held fixed. The
    moments therefore refer only to the scalar spot residual left after
    conditioning on every realized surface innovation.
    """

    center = np.asarray(sampler.pcaxar1.state_center, dtype=np.float64)
    scale = np.asarray(sampler.pcaxar1.state_scale, dtype=np.float64)
    if center.shape != scale.shape or center.shape != (reference_path.shape[1],):
        raise ValueError(
            f"invalid state normalization {center.shape}/{scale.shape} for "
            f"path {reference_path.shape}"
        )
    standardized = (reference_path - center[None, :]) / scale[None, :]
    normalized_dt = float(dt / sampler.pcaxar1.mean_dt)
    raw_k = sampler.pcaxar1.k
    raw_intercept = sampler.pcaxar1.U0
    k = raw_k * scale[:, None] / scale[None, :]
    intercept = (raw_intercept - center[None, :] @ raw_k) / scale[None, :]
    loadings = sampler.pcaxar1.dU / scale[None, :]
    transition, offset = _xar1_matrix_exp_terms(
        U0=intercept,
        k=k,
        h=normalized_dt,
    )
    alpha_covariance = sampler.cholesky @ sampler.cholesky.T
    covariance_rate = loadings.T @ alpha_covariance @ loadings
    noise_factor = _xar1_gaussian_step_factor(
        k=k,
        C=covariance_rate,
        h=normalized_dt,
        A=transition,
    )
    covariance = noise_factor @ noise_factor.T
    surface_covariance = 0.5 * (
        covariance[1:, 1:] + covariance[1:, 1:].T
    )
    surface_inverse = np.linalg.pinv(
        surface_covariance,
        rcond=1.0e3 * np.finfo(float).eps,
        hermitian=True,
    )
    spot_given_surface = surface_inverse @ covariance[1:, 0]
    residual_variance = float(
        covariance[0, 0] - covariance[0, 1:] @ spot_given_surface
    )
    if not math.isfinite(residual_variance) or residual_variance <= 0.0:
        raise ValueError(
            "conditional spot residual variance must be positive, got "
            f"{residual_variance:g}"
        )

    innovations = (
        standardized[1:]
        - standardized[:-1] @ transition
        - offset
    )
    surface_innovations = innovations[:, 1:]
    conditional_mean = np.empty(reference_path.shape[0], dtype=np.float64)
    conditional_variance = np.empty(reference_path.shape[0], dtype=np.float64)
    conditional_mean[0] = standardized[0, 0]
    conditional_variance[0] = 0.0
    spot_autoregression = float(transition[0, 0])
    for step in range(reference_path.shape[0] - 1):
        conditional_mean[step + 1] = (
            conditional_mean[step] * spot_autoregression
            + standardized[step, 1:] @ transition[1:, 0]
            + offset[0, 0]
            + surface_innovations[step] @ spot_given_surface
        )
        conditional_variance[step + 1] = (
            spot_autoregression**2 * conditional_variance[step]
            + residual_variance
        )

    raw_mean = center[0] + scale[0] * conditional_mean
    raw_std = scale[0] * np.sqrt(conditional_variance)
    return raw_mean, raw_std, math.sqrt(residual_variance) * scale[0]


def create_figure(
    *,
    start_date: date,
    tail_multiplier: float,
    seed: int,
    output_path: Path,
) -> dict[str, object]:
    if not math.isfinite(tail_multiplier) or tail_multiplier <= 1.0:
        raise ValueError(
            f"tail_multiplier must be finite and greater than one, got {tail_multiplier}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    experiment = GenDysanosExperiment(
        num_samples=1,
        num_alpha=None,
        use_matrix_exp=True,
        bandwidth=1.0,
        seed=seed,
    )
    grid_config, market, _, sampler = _load_generator_components(
        experiment,
        verbose=Context.quiet,
    )
    matching_dates = [
        index
        for index, market_date in enumerate(market.dates)
        if str(market_date) == start_date.isoformat()
    ]
    if len(matching_dates) != 1:
        raise ValueError(
            f"expected one observation for {start_date}, found {len(matching_dates)}"
        )
    start_index = matching_dates[0]
    dt = 1.0 / DAYS_PER_YEAR
    expiries = np.asarray(grid_config.grid_expiries, dtype=np.float64)
    expiry_steps = np.rint(expiries / dt).astype(np.int64)
    if np.any(expiry_steps <= 0) or np.any(np.diff(expiry_steps) <= 0):
        raise ValueError(f"invalid expiry steps {expiry_steps}")
    maximum_expiry_error = float(np.max(np.abs(expiry_steps * dt - expiries)))
    if maximum_expiry_error > 0.5 * dt:
        raise ValueError(
            f"expiry grid cannot be represented on the daily path: {maximum_expiry_error:g}"
        )

    reference_path = _conditional_mean_surface_path(
        sampler=sampler,
        initial_state=sampler.xstate[start_index].astype(np.float64),
        num_steps=int(expiry_steps[-1]) + 1,
        dt=dt,
    )
    if not np.allclose(reference_path[0], sampler.xstate[start_index]):
        raise RuntimeError("simulated path does not start at the selected historical state")
    conditional_log_mean, conditional_log_std, daily_residual_std = (
        _conditional_log_spot_moments(
            sampler=sampler,
            reference_path=reference_path,
            dt=dt,
        )
    )

    initial_spot = float(market["spot_ref"][start_index])
    log_spot_scale = math.log(initial_spot) - float(reference_path[0, 0])
    absolute_log_mean = log_spot_scale + conditional_log_mean
    normal_quantile = float(norm.ppf(0.95))
    calibrated_lower = np.exp(
        absolute_log_mean - normal_quantile * conditional_log_std
    )
    calibrated_upper = np.exp(
        absolute_log_mean + normal_quantile * conditional_log_std
    )
    proposal_lower = np.exp(
        absolute_log_mean
        - normal_quantile * tail_multiplier * conditional_log_std
    )
    proposal_upper = np.exp(
        absolute_log_mean
        + normal_quantile * tail_multiplier * conditional_log_std
    )
    conditional_median = np.exp(absolute_log_mean)

    model_strikes = (
        sampler.pricer.strikes[start_index].detach().cpu().numpy().astype(np.float64)
    )
    if model_strikes.shape != (expiries.size, 20):
        raise ValueError(
            f"unexpected initial model-strike grid {model_strikes.shape}"
        )
    cash_strikes = initial_spot * model_strikes

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/quantdata-matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    day_grid = np.arange(1, expiry_steps[-1] + 1)
    time_grid = day_grid / DAYS_PER_YEAR
    fig, axis = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    axis.fill_between(
        time_grid,
        proposal_lower[1:],
        proposal_upper[1:],
        color="#d98c28",
        alpha=0.20,
        label=rf"widened component, $r={tail_multiplier:g}$ (5–95%)",
    )
    axis.plot(
        time_grid,
        proposal_lower[1:],
        color="#b66c0d",
        linewidth=0.8,
    )
    axis.plot(
        time_grid,
        proposal_upper[1:],
        color="#b66c0d",
        linewidth=0.8,
    )
    axis.fill_between(
        time_grid,
        calibrated_lower[1:],
        calibrated_upper[1:],
        color="#3976af",
        alpha=0.32,
        label="calibrated conditional law (5–95%)",
    )
    axis.plot(
        time_grid,
        conditional_median[1:],
        color="#174b73",
        linewidth=1.3,
        label="conditional spot median",
    )
    for expiry_index, (expiry, strikes) in enumerate(
        zip(expiries, cash_strikes, strict=True)
    ):
        axis.scatter(
            np.full(strikes.size, expiry),
            strikes,
            marker="_",
            s=80,
            linewidths=0.85,
            color="#222222",
            label=r"cash strikes of $C_0$" if expiry_index == 0 else None,
            zorder=4,
        )
    axis.axhline(
        initial_spot,
        color="#555555",
        linestyle=":",
        linewidth=0.9,
        label=rf"initial spot $S_0={initial_spot:,.0f}$",
    )
    axis.set_xscale("log")
    axis.set_xticks(
        expiries,
        tuple(
            f"{int(step)}d" if step < DAYS_PER_YEAR else "1y"
            for step in expiry_steps
        ),
    )
    axis.set_xlim(time_grid[0], time_grid[-1] * 1.04)
    axis.set_ylim(0.0, float(np.max(proposal_upper[expiry_steps])) * 1.04)
    axis.set_xlabel("time from initial surface (log scale)")
    axis.set_ylabel("SPX level / cash strike")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(output_path, dpi=240)
    plt.close(fig)

    report = {
        "definition": (
            "Spot bands conditional on the mean future surface path starting from "
            "a historical ML-SANOS state. The widened ranges are individual "
            "importance-mixture components and are not physical-law confidence bands."
        ),
        "start_date": start_date.isoformat(),
        "start_index": start_index,
        "initial_spot": initial_spot,
        "surface_factors": sampler.pcaxar1.num_surface_alpha,
        "tail_multiplier": tail_multiplier,
        "seed": seed,
        "daily_conditional_log_spot_residual_std": daily_residual_std,
        "expiries_years": expiries,
        "expiry_steps": expiry_steps,
        "cash_strike_minimum": np.min(cash_strikes, axis=1),
        "cash_strike_maximum": np.max(cash_strikes, axis=1),
        "conditional_spot_median": conditional_median[expiry_steps],
        "calibrated_spot_q05": calibrated_lower[expiry_steps],
        "calibrated_spot_q95": calibrated_upper[expiry_steps],
        "proposal_spot_q05": proposal_lower[expiry_steps],
        "proposal_spot_q95": proposal_upper[expiry_steps],
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2025, 5, 6))
    parser.add_argument("--tail-multiplier", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=614_203)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "tex"
            / "generated"
            / "conditional_spot_importance_example.png"
        ),
    )
    arguments = parser.parse_args()
    report = create_figure(
        start_date=arguments.start_date,
        tail_multiplier=arguments.tail_multiplier,
        seed=arguments.seed,
        output_path=arguments.output_path,
    )
    print(
        f"Wrote conditional spot example for {report['start_date']} to "
        f"{arguments.output_path}"
    )


if __name__ == "__main__":
    main()
