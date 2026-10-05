"""Generate paper diagnostics for the daily front-spot model."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.sdlv.daily.front_spot.calibration import (
    FrontSpotCalibration,
    calibrate_front_spot_model,
)
from quantdata.sdlv.daily.front_spot.estimation import (
    JointInnovationFit,
    fit_joint_innovations,
)
from quantdata.sdlv.daily.front_spot.jax_model import (
    RiskNeutralState,
    black_mixture_call_prices,
    decode_density_tower,
    one_day_physical_substepped,
    one_day_risk_neutral_substepped,
)
from quantdata.sdlv.daily.front_spot.simulation import load_initial_model_state

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class FactorSpectrum:
    """Variance fractions of the spot-oriented DLV factor system."""

    individual_fraction: NDArray[np.float64]
    cumulative_fraction: NDArray[np.float64]
    first_rank_above_90: int | None


@dataclass(frozen=True)
class PaperConfiguration:
    """One factor-rank and time-step configuration used in the paper."""

    label: str
    factor_rank: int
    dlv_substeps: int


@dataclass(frozen=True)
class ImpliedVolatilityFactorSurfaces:
    """Base IV surface and central daily factor responses on the paper grid."""

    log_moneyness: NDArray[np.float64]
    expiry_days: NDArray[np.int64]
    base_iv: NDArray[np.float64]
    factor_responses_bp: NDArray[np.float64]


def factor_spectrum(innovations: JointInnovationFit) -> FactorSpectrum:
    """Recover individual and cumulative variance fractions from a fit."""
    factor_rank = innovations.factor_rank
    daily_loadings = innovations.annualized_loadings / np.sqrt(DAYS_PER_YEAR)
    factor_variance = np.sum(
        daily_loadings.reshape(-1, factor_rank) ** 2,
        axis=0,
    )
    retained_variance = float(np.sum(factor_variance))
    explained = innovations.explained_variance_fraction
    total_variance = retained_variance / explained
    individual = factor_variance / total_variance
    cumulative = np.cumsum(individual)
    above = np.flatnonzero(cumulative >= 0.9)
    return FactorSpectrum(
        individual_fraction=individual,
        cumulative_fraction=cumulative,
        first_rank_above_90=(int(above[0]) + 1 if above.size else None),
    )


def implied_volatility_from_calls(
    forward: NDArray[np.float64] | float,
    strikes: NDArray[np.float64],
    maturity_years: NDArray[np.float64],
    call_prices: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Invert undiscounted Black calls with vectorized total-vol bisection."""
    call = np.asarray(call_prices, dtype=np.float64)
    forward_array, strike_array, maturity_array, call = np.broadcast_arrays(
        np.asarray(forward, dtype=np.float64),
        np.asarray(strikes, dtype=np.float64),
        np.asarray(maturity_years, dtype=np.float64),
        call,
    )
    if np.any(forward_array <= 0.0) or np.any(strike_array <= 0.0):
        raise ValueError("Forwards and strikes must be positive")
    if np.any(maturity_array <= 0.0):
        raise ValueError("Maturities must be positive")

    intrinsic = np.maximum(forward_array - strike_array, 0.0)
    target = np.clip(call, intrinsic, forward_array)
    lower = np.full_like(target, 1.0e-10)
    upper = np.full_like(target, 5.0)
    log_moneyness = np.log(forward_array / strike_array)
    for _ in range(70):
        total_vol = 0.5 * (lower + upper)
        d1 = log_moneyness / total_vol + 0.5 * total_vol
        d2 = d1 - total_vol
        price = forward_array * ndtr(d1) - strike_array * ndtr(d2)
        lower = np.where(price < target, total_vol, lower)
        upper = np.where(price >= target, total_vol, upper)
    return 0.5 * (lower + upper) / np.sqrt(maturity_array)


def _call_implied_vol_surface(
    support: NDArray[np.float64],
    density_tower: NDArray[np.float64],
    strikes: NDArray[np.float64],
) -> NDArray[np.float64]:
    expiry_count = density_tower.shape[-2]
    remaining_days = np.arange(1, expiry_count + 1, dtype=np.float64)
    call_prices = np.asarray(
        black_mixture_call_prices(
            jnp.asarray(support, dtype=jnp.float64),
            jnp.asarray(density_tower, dtype=jnp.float64),
            jnp.asarray(strikes, dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
            jnp.asarray(remaining_days, dtype=jnp.float64),
        )
    )
    return implied_volatility_from_calls(
        np.ones((expiry_count, 1), dtype=np.float64),
        strikes[None, :],
        remaining_days[:, None] / DAYS_PER_YEAR,
        call_prices,
    )


def plot_factor_power(
    spectrum: FactorSpectrum,
    output_path: Path,
    *,
    model_rank_cap: int,
) -> None:
    """Plot individual and cumulative explained variance by factor."""
    ranks = np.arange(1, len(spectrum.individual_fraction) + 1)
    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    axis.bar(
        ranks,
        100.0 * spectrum.individual_fraction,
        color="#4C78A8",
        label="individual",
    )
    axis.set_xlabel("Factor rank")
    axis.set_ylabel("Individual variance explained (%)")
    axis.set_xticks(ranks)
    cumulative_axis = axis.twinx()
    cumulative_axis.plot(
        ranks,
        100.0 * spectrum.cumulative_fraction,
        color="#E45756",
        marker="o",
        linewidth=2.0,
        label="cumulative",
    )
    cumulative_axis.axhline(90.0, color="black", linestyle="--", linewidth=1.0)
    cumulative_axis.axvline(
        model_rank_cap,
        color="#54A24B",
        linestyle=":",
        linewidth=1.5,
        label=f"model cap = {model_rank_cap}",
    )
    cumulative_axis.set_ylabel("Cumulative variance explained (%)")
    cumulative_axis.set_ylim(0.0, 100.0)
    handles = axis.get_legend_handles_labels()[0] + cumulative_axis.get_legend_handles_labels()[0]
    labels = axis.get_legend_handles_labels()[1] + cumulative_axis.get_legend_handles_labels()[1]
    axis.legend(handles, labels, loc="center right")
    axis.set_title("Spot-oriented DLV factor power")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _implied_volatility_factor_surfaces(
    calibration: FrontSpotCalibration,
    *,
    fit_label: str = "full",
    date_index: int = -1,
) -> ImpliedVolatilityFactorSurfaces:
    """Compute base IV and central one-daily-sigma factor responses."""
    initial = load_initial_model_state(fit_label=fit_label, date_index=date_index)
    support = initial.support
    log_strikes = np.log(support[1:-1])
    strike_mask = (log_strikes >= -0.30) & (log_strikes <= 0.25)
    strikes = support[1:-1][strike_mask]
    plotted_log_strikes = log_strikes[strike_mask]
    base_iv = _call_implied_vol_surface(support, initial.density_tower, strikes)
    factor_rank = calibration.innovations.factor_rank
    loadings = calibration.innovations.annualized_loadings
    factor_shocks: list[NDArray[np.float64]] = []
    daily_scale = np.sqrt(1.0 / DAYS_PER_YEAR)
    for factor_index in range(factor_rank):
        log_shock = daily_scale * loadings[..., factor_index]
        plus_dlvsq = initial.dlvsq.copy()
        minus_dlvsq = initial.dlvsq.copy()
        plus_dlvsq[1:] *= np.exp(log_shock)
        minus_dlvsq[1:] *= np.exp(-log_shock)
        plus_tower = np.asarray(
            decode_density_tower(
                jnp.asarray(support, dtype=jnp.float64),
                jnp.asarray(plus_dlvsq, dtype=jnp.float64),
                dt=1.0 / DAYS_PER_YEAR,
            )
        )
        minus_tower = np.asarray(
            decode_density_tower(
                jnp.asarray(support, dtype=jnp.float64),
                jnp.asarray(minus_dlvsq, dtype=jnp.float64),
                dt=1.0 / DAYS_PER_YEAR,
            )
        )
        plus_iv = _call_implied_vol_surface(support, plus_tower, strikes)
        minus_iv = _call_implied_vol_surface(support, minus_tower, strikes)
        factor_shocks.append(5_000.0 * (plus_iv - minus_iv))

    factor_responses = np.stack(factor_shocks)
    if not np.all(np.isfinite(base_iv)) or not np.all(np.isfinite(factor_responses)):
        raise RuntimeError("Implied-volatility factor surfaces contain non-finite values")
    return ImpliedVolatilityFactorSurfaces(
        log_moneyness=plotted_log_strikes,
        expiry_days=np.arange(1, initial.density_tower.shape[0] + 1),
        base_iv=base_iv,
        factor_responses_bp=factor_responses,
    )


def plot_implied_volatility_shocks(
    calibration: FrontSpotCalibration,
    output_path: Path,
    *,
    fit_label: str = "full",
    date_index: int = -1,
    surfaces: ImpliedVolatilityFactorSurfaces | None = None,
) -> None:
    """Plot base implied volatility and signed one-daily-sigma factor shocks."""
    if surfaces is None:
        surfaces = _implied_volatility_factor_surfaces(
            calibration,
            fit_label=fit_label,
            date_index=date_index,
        )
    factor_responses = surfaces.factor_responses_bp
    factor_rank = factor_responses.shape[0]

    panel_count = factor_rank + 1
    column_count = 4
    row_count = int(np.ceil(panel_count / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(4.1 * column_count, 3.0 * row_count),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    flat_axes = np.asarray(axes).reshape(-1)
    base_image = flat_axes[0].pcolormesh(
        surfaces.log_moneyness,
        surfaces.expiry_days,
        100.0 * surfaces.base_iv,
        shading="auto",
        cmap="viridis",
    )
    flat_axes[0].set_title("Base implied volatility (%)")
    shock_limit = max(
        float(np.nanquantile(np.abs(factor_responses), 0.99)),
        1.0,
    )
    shock_images = []
    for factor_index, shock in enumerate(factor_responses):
        axis = flat_axes[factor_index + 1]
        image = axis.pcolormesh(
            surfaces.log_moneyness,
            surfaces.expiry_days,
            shock,
            shading="auto",
            cmap="coolwarm",
            vmin=-shock_limit,
            vmax=shock_limit,
        )
        shock_images.append(image)
        qualifier = "spot-aligned" if factor_index == 0 else "orthogonal"
        axis.set_title(f"Factor {factor_index + 1} ({qualifier})")
    for axis in flat_axes[panel_count:]:
        axis.set_visible(False)
    for axis in flat_axes[:panel_count]:
        axis.set_xlabel("Log-moneyness")
        axis.set_ylabel("Remaining business days")
    figure.colorbar(base_image, ax=flat_axes[0], label="Annualized volatility (%)")
    figure.colorbar(
        shock_images[0],
        ax=list(flat_axes[1:panel_count]),
        label=r"Central $\pm1\sigma$ implied-volatility shock (bp)",
    )
    figure.suptitle(
        "Risk-neutral daily DLV diffusion shocks in implied-volatility coordinates"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_baseline_implied_volatility_3d(
    surfaces: ImpliedVolatilityFactorSurfaces,
    output_path: Path,
) -> None:
    """Plot the baseline implied-volatility surface in three dimensions."""
    log_moneyness, expiry_days = np.meshgrid(
        surfaces.log_moneyness,
        surfaces.expiry_days,
    )
    base_iv_percent = 100.0 * surfaces.base_iv
    figure = plt.figure(figsize=(10.0, 7.2), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    plotted_surface = axis.plot_surface(
        log_moneyness,
        expiry_days,
        base_iv_percent,
        cmap="viridis",
        linewidth=0.0,
        antialiased=True,
        rcount=len(surfaces.expiry_days),
        ccount=len(surfaces.log_moneyness),
    )
    axis.set_xlabel("Log-moneyness")
    axis.set_ylabel("Remaining business days")
    axis.set_zlabel("Annualized implied volatility (%)")
    axis.set_title("Baseline risk-neutral implied-volatility surface")
    axis.view_init(elev=27.0, azim=-128.0)
    axis.set_box_aspect((1.35, 1.0, 0.75))
    figure.colorbar(
        plotted_surface,
        ax=axis,
        shrink=0.68,
        pad=0.08,
        label="Annualized implied volatility (%)",
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_factor_responses_3d(
    surfaces: ImpliedVolatilityFactorSurfaces,
    output_path: Path,
) -> None:
    """Plot central daily PCA-factor IV responses as 3D surfaces."""
    factor_responses = surfaces.factor_responses_bp
    factor_rank = factor_responses.shape[0]
    column_count = 4
    row_count = int(np.ceil(factor_rank / column_count))
    log_moneyness, expiry_days = np.meshgrid(
        surfaces.log_moneyness,
        surfaces.expiry_days,
    )
    response_limit = max(float(np.max(np.abs(factor_responses))), 1.0)
    figure = plt.figure(
        figsize=(4.5 * column_count, 3.8 * row_count),
        constrained_layout=True,
    )
    axes = []
    plotted_surface = None
    for factor_index, response in enumerate(factor_responses):
        axis = figure.add_subplot(
            row_count,
            column_count,
            factor_index + 1,
            projection="3d",
        )
        plotted_surface = axis.plot_surface(
            log_moneyness,
            expiry_days,
            response,
            cmap="coolwarm",
            vmin=-response_limit,
            vmax=response_limit,
            linewidth=0.0,
            antialiased=True,
            rcount=len(surfaces.expiry_days),
            ccount=len(surfaces.log_moneyness),
        )
        qualifier = "spot-aligned" if factor_index == 0 else "orthogonal"
        axis.set_title(f"Factor {factor_index + 1} ({qualifier})", fontsize=10)
        axis.set_xlabel("Log-moneyness", fontsize=8, labelpad=2)
        axis.set_ylabel("Days", fontsize=8, labelpad=2)
        axis.set_zlabel("IV response (bp)", fontsize=8, labelpad=2)
        axis.set_zlim(-response_limit, response_limit)
        axis.tick_params(labelsize=7, pad=0)
        axis.view_init(elev=25.0, azim=-128.0)
        axis.set_box_aspect((1.25, 1.0, 0.75))
        axes.append(axis)
    if plotted_surface is None:
        raise RuntimeError("At least one factor response is required")
    figure.colorbar(
        plotted_surface,
        ax=axes,
        shrink=0.55,
        pad=0.02,
        label=r"Central $\pm1\sigma$ implied-volatility response (bp)",
    )
    figure.suptitle(
        f"Rank-{factor_rank} risk-neutral daily PCA responses in IV coordinates"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _path_surface_metrics(
    support: NDArray[np.float64],
    density_tower: NDArray[np.float64],
    background_level: NDArray[np.float64],
    spot: NDArray[np.float64],
    *,
    log_strike_bump: float = 0.01,
) -> dict[str, NDArray[np.float64]]:
    path_count = density_tower.shape[0]
    tenor_indices = np.asarray([0, 4])
    remaining_days = np.asarray([1.0, 5.0])
    selected_tower = density_tower[:, tenor_indices]
    strike_multipliers = np.exp(
        np.asarray([-log_strike_bump, 0.0, log_strike_bump])
    )
    strikes = spot[:, None] * strike_multipliers[None, :]

    def one_path_calls(
        path_density: jax.Array,
        path_background: jax.Array,
        path_strikes: jax.Array,
    ) -> jax.Array:
        return black_mixture_call_prices(
            jnp.asarray(support, dtype=jnp.float64),
            path_density,
            path_strikes,
            path_background,
            jnp.asarray(remaining_days, dtype=jnp.float64),
        )

    call_prices = np.asarray(
        jax.vmap(one_path_calls)(
            jnp.asarray(selected_tower, dtype=jnp.float64),
            jnp.asarray(background_level, dtype=jnp.float64),
            jnp.asarray(strikes, dtype=jnp.float64),
        )
    )
    implied_vol = implied_volatility_from_calls(
        spot[:, None, None],
        strikes[:, None, :],
        remaining_days[None, :, None] / DAYS_PER_YEAR,
        call_prices,
    )
    return {
        "spot": spot,
        "atm_1d_vol": implied_vol[:, 0, 1],
        "atm_5d_vol": implied_vol[:, 1, 1],
        "atm_1d_skew": (
            (implied_vol[:, 0, 2] - implied_vol[:, 0, 0])
            / (2.0 * log_strike_bump)
        ),
        "atm_5d_skew": (
            (implied_vol[:, 1, 2] - implied_vol[:, 1, 0])
            / (2.0 * log_strike_bump)
        ),
    }


def simulate_coupled_path_metrics(
    calibration: FrontSpotCalibration,
    *,
    dlv_substeps: int,
    path_count: int = 4,
    day_count: int = 10,
    seed: int = 20261005,
    fit_label: str = "full",
    date_index: int = -1,
) -> dict[str, Any]:
    """Simulate matched P/Q paths and return spot, ATM vol, and ATM skew."""
    if not 1 <= day_count <= 20:
        raise ValueError("day_count must lie in [1,20] so the 5D tenor remains live")
    if path_count < 1:
        raise ValueError("path_count must be positive")
    initial = load_initial_model_state(fit_label=fit_label, date_index=date_index)
    factor_rank = calibration.innovations.factor_rank
    generator = np.random.default_rng(seed)
    factor_normals = jnp.asarray(
        generator.standard_normal(
            (day_count, dlv_substeps, path_count, factor_rank)
        ),
        dtype=jnp.float64,
    )
    background_normals = jnp.asarray(
        generator.standard_normal((day_count, path_count)),
        dtype=jnp.float64,
    )
    support = jnp.asarray(initial.support, dtype=jnp.float64)
    initial_front = jnp.asarray(initial.density_tower[0], dtype=jnp.float64)
    initial_future = jnp.asarray(initial.dlvsq[1:], dtype=jnp.float64)
    q_state = RiskNeutralState(initial_front, initial_future, jnp.asarray(1.0))
    p_state = RiskNeutralState(initial_front, initial_future, jnp.asarray(1.0))
    loadings = jnp.asarray(
        calibration.innovations.annualized_loadings,
        dtype=jnp.float64,
    )
    risk_price = jnp.asarray(
        calibration.physical.constant_market_price_of_risk,
        dtype=jnp.float64,
    )
    q_step = jax.jit(
        one_day_risk_neutral_substepped,
        static_argnames=("expiry_dt", "time_power"),
    )
    p_step = jax.jit(
        one_day_physical_substepped,
        static_argnames=("expiry_dt", "time_power"),
    )
    q_metrics: list[dict[str, NDArray[np.float64]]] = []
    p_metrics: list[dict[str, NDArray[np.float64]]] = []
    initial_tower = np.broadcast_to(
        initial.density_tower,
        (path_count,) + initial.density_tower.shape,
    )
    initial_background = np.ones(path_count, dtype=np.float64)
    initial_spot = np.ones(path_count, dtype=np.float64)
    initial_metrics = _path_surface_metrics(
        initial.support,
        initial_tower,
        initial_background,
        initial_spot,
    )
    q_metrics.append(initial_metrics)
    p_metrics.append(initial_metrics)

    for day_index in range(day_count):
        future_count = q_state.future_dlvsq.shape[-2]
        active_loadings = loadings[:future_count]
        q_result = q_step(
            support,
            q_state,
            active_loadings,
            factor_normals[day_index],
            background_normals[day_index],
            expiry_dt=1.0 / DAYS_PER_YEAR,
            time_power=1.0,
        )
        p_result = p_step(
            support,
            p_state,
            active_loadings,
            risk_price,
            factor_normals[day_index],
            background_normals[day_index],
            expiry_dt=1.0 / DAYS_PER_YEAR,
            time_power=1.0,
        )
        q_state = q_result.state
        p_state = p_result.state
        q_metrics.append(
            _path_surface_metrics(
                initial.support,
                np.asarray(q_result.density_tower),
                np.asarray(q_state.background_level),
                np.asarray(q_result.spot),
            )
        )
        p_metrics.append(
            _path_surface_metrics(
                initial.support,
                np.asarray(p_result.density_tower),
                np.asarray(p_state.background_level),
                np.asarray(p_result.spot),
            )
        )

    keys = tuple(initial_metrics)
    return {
        "initial_date": initial.date,
        "factor_rank": factor_rank,
        "dlv_substeps": dlv_substeps,
        "path_count": path_count,
        "day_count": day_count,
        "seed": seed,
        "q": {key: np.stack([value[key] for value in q_metrics]) for key in keys},
        "p": {key: np.stack([value[key] for value in p_metrics]) for key in keys},
    }


def plot_coupled_path_metrics(
    metrics: dict[str, Any],
    output_path: Path,
    *,
    title: str,
) -> None:
    """Plot matched P/Q paths for spot, ATM volatility, and ATM skew."""
    keys = (
        "spot",
        "atm_1d_vol",
        "atm_5d_vol",
        "atm_1d_skew",
        "atm_5d_skew",
    )
    labels = (
        "Spot / initial spot",
        "ATM 1D implied vol (%)",
        "ATM 5D implied vol (%)",
        "ATM 1D skew (vol points / log-strike)",
        "ATM 5D skew (vol points / log-strike)",
    )
    scales = (1.0, 100.0, 100.0, 100.0, 100.0)
    days = np.arange(metrics["day_count"] + 1)
    figure, axes = plt.subplots(5, 1, figsize=(10.0, 13.0), sharex=True, constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    for axis, key, label, scale in zip(axes, keys, labels, scales, strict=True):
        q_values = np.asarray(metrics["q"][key])
        p_values = np.asarray(metrics["p"][key])
        for path_index in range(metrics["path_count"]):
            color = colors[path_index % len(colors)]
            axis.plot(days, scale * q_values[:, path_index], color=color, linewidth=1.6)
            axis.plot(
                days,
                scale * p_values[:, path_index],
                color=color,
                linewidth=1.3,
                linestyle="--",
            )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].plot([], [], color="black", linewidth=1.6, label=r"$\mathbb{Q}$")
    axes[0].plot([], [], color="black", linewidth=1.3, linestyle="--", label=r"$\mathbb{P}$")
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Business-day step")
    figure.suptitle(title)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def run_paper_diagnostics(
    output_dir: Path,
    *,
    max_factor_rank: int = 16,
    spectrum_factor_rank: int = 32,
    path_count: int = 4,
    path_days: int = 10,
    seed: int = 20261005,
) -> dict[str, Any]:
    """Generate spectrum, IV-shock, and coupled-path paper artifacts."""
    if not 8 <= max_factor_rank <= 16:
        raise ValueError("max_factor_rank must lie in [8,16]")
    if not max_factor_rank <= spectrum_factor_rank < 710:
        raise ValueError("spectrum_factor_rank must lie in [max_factor_rank,709]")
    output_dir.mkdir(parents=True, exist_ok=True)
    maximum_calibration = calibrate_front_spot_model(factor_rank=max_factor_rank)
    spectrum_innovations = fit_joint_innovations(
        maximum_calibration.history,
        factor_rank=spectrum_factor_rank,
        smoothing_sigma=(1.0, 2.0),
    )
    spectrum = factor_spectrum(spectrum_innovations)
    selected_rank = (
        spectrum.first_rank_above_90
        if spectrum.first_rank_above_90 is not None
        and spectrum.first_rank_above_90 <= max_factor_rank
        else max_factor_rank
    )
    reaches_90_by_max_rank = bool(
        spectrum.cumulative_fraction[max_factor_rank - 1] >= 0.9
    )
    selected_calibration = (
        maximum_calibration
        if selected_rank == max_factor_rank
        else calibrate_front_spot_model(factor_rank=selected_rank)
    )
    rank_eight_calibration = (
        selected_calibration
        if selected_rank == 8
        else calibrate_front_spot_model(factor_rank=8)
    )
    calibrations = {8: rank_eight_calibration, selected_rank: selected_calibration}
    configurations = (
        PaperConfiguration("rank8_one_step", 8, 1),
        PaperConfiguration(f"rank{selected_rank}_eight_step", selected_rank, 8),
        PaperConfiguration(f"rank{selected_rank}_one_step", selected_rank, 1),
    )

    plot_factor_power(
        spectrum,
        output_dir / "pca_explained_variance.png",
        model_rank_cap=max_factor_rank,
    )
    factor_surfaces = {
        rank: _implied_volatility_factor_surfaces(calibration)
        for rank, calibration in calibrations.items()
    }
    plot_baseline_implied_volatility_3d(
        factor_surfaces[selected_rank],
        output_dir / "baseline_implied_volatility_3d.png",
    )
    for rank, calibration in calibrations.items():
        plot_implied_volatility_shocks(
            calibration,
            output_dir / f"pca_implied_vol_shocks_rank{rank}.png",
            surfaces=factor_surfaces[rank],
        )
        plot_factor_responses_3d(
            factor_surfaces[rank],
            output_dir / f"pca_factor_responses_3d_rank{rank}.png",
        )

    path_outputs: dict[str, Any] = {}
    for configuration in configurations:
        metrics = simulate_coupled_path_metrics(
            calibrations[configuration.factor_rank],
            dlv_substeps=configuration.dlv_substeps,
            path_count=path_count,
            day_count=path_days,
            seed=seed,
        )
        plot_coupled_path_metrics(
            metrics,
            output_dir / f"paths_{configuration.label}.png",
            title=(
                f"Rank {configuration.factor_rank}, "
                f"{configuration.dlv_substeps} DLV substep(s) per day"
            ),
        )
        path_outputs[configuration.label] = metrics

    output = {
        "schema": "front-spot-paper-diagnostics-v1",
        "max_factor_rank": max_factor_rank,
        "spectrum_factor_rank": spectrum_factor_rank,
        "selected_factor_rank": selected_rank,
        "first_rank_above_90_percent": spectrum.first_rank_above_90,
        "reaches_90_percent_by_max_rank": reaches_90_by_max_rank,
        "selected_explained_variance_fraction": float(
            spectrum.cumulative_fraction[selected_rank - 1]
        ),
        "individual_variance_fraction": spectrum.individual_fraction,
        "cumulative_variance_fraction": spectrum.cumulative_fraction,
        "configurations": [configuration.__dict__ for configuration in configurations],
        "paths": path_outputs,
    }
    with (output_dir / "paper_diagnostics.json").open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(output), stream, indent=2)
        stream.write("\n")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-factor-rank", type=int, default=16)
    parser.add_argument("--spectrum-factor-rank", type=int, default=32)
    parser.add_argument("--path-count", type=int, default=4)
    parser.add_argument("--path-days", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261005)
    return parser


def main() -> None:
    """Generate all paper diagnostics from the command line."""
    arguments = _parser().parse_args()
    output = run_paper_diagnostics(
        arguments.output_dir,
        max_factor_rank=arguments.max_factor_rank,
        spectrum_factor_rank=arguments.spectrum_factor_rank,
        path_count=arguments.path_count,
        path_days=arguments.path_days,
        seed=arguments.seed,
    )
    print(
        json.dumps(
            {
                "selected_factor_rank": output["selected_factor_rank"],
                "reaches_90_percent_by_max_rank": output[
                    "reaches_90_percent_by_max_rank"
                ],
                "first_rank_above_90_percent": output[
                    "first_rank_above_90_percent"
                ],
                "selected_explained_variance_fraction": output[
                    "selected_explained_variance_fraction"
                ],
                "output_dir": str(arguments.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
