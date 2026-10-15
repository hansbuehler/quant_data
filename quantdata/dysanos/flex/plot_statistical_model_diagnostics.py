"""Regenerate paper figures from completed two-eta Direct-200 checkpoints."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import torch
from cdxcore.bs import bs
from cdxcore.verbose import Context
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
from scipy.linalg import solve_discrete_lyapunov

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.dysanos.flex.config import (
    FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    NativeStateConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.model import NativeStateSanosModel
from quantdata.dysanos.flex.presentation_hedging import (
    _decode_surface,
    load_presentation_model,
)
from quantdata.torch import BaseEnvironment
from quantdata.widb.day import enrich_day

plt.switch_backend("Agg")

_COMPACT_FIGURE_SCALE = 0.5
_COMPACT_SUPTITLE_FONTSIZE = 13
_PCA_SPECTRUM_TITLE_FONTSIZE = 0.75 * _COMPACT_SUPTITLE_FONTSIZE
_PCA_SPECTRUM_TEXT_FONTSIZE = 7.5
_SURFACE_SUPTITLE_FONTSIZE = 11
_SURFACE_AXES_TITLE_FONTSIZE = 10
_SURFACE_LABEL_FONTSIZE = 9
_SURFACE_TICK_FONTSIZE = 8
_PCA_FACTOR_SUPTITLE_FONTSIZE = 13.75
_PCA_FACTOR_AXES_TITLE_FONTSIZE = 11.25
_PCA_FACTOR_LABEL_FONTSIZE = 10
_PCA_FACTOR_TICK_FONTSIZE = 8.75


@dataclass(frozen=True)
class DecodedIvPca:
    """Local decoded-IV covariance eigendecomposition at the invariant mean."""

    expiry_business_days: np.ndarray
    nms: np.ndarray
    mean_iv: np.ndarray
    variances: np.ndarray
    one_standard_deviation_surfaces: np.ndarray
    invariant_covariance: np.ndarray

    @property
    def explained_variance(self) -> np.ndarray:
        """Individual explained-variance fractions."""
        return self.variances / np.sum(self.variances)


def select_pca_expiries(model_expiry_business_days: np.ndarray) -> np.ndarray:
    """Select seven increasing model nodes spanning the available term structure."""
    values = np.asarray(model_expiry_business_days, dtype=np.float64)
    if values.ndim != 1 or len(values) < 7 or np.any(np.diff(values) <= 0.0):
        raise ValueError("at least seven increasing model expiries are required")
    indices = np.rint(np.linspace(0, len(values) - 1, 7)).astype(np.int64)
    selected = values[indices]
    if len(np.unique(selected)) != 7:
        raise ValueError("could not select seven distinct model expiries")
    return selected


def stationary_state_covariance(
    transition: np.ndarray,
    innovation_covariance: np.ndarray,
) -> np.ndarray:
    """Solve the row-state invariant covariance ``P=A.T P A+Q``."""
    matrix = np.asarray(transition, dtype=np.float64)
    covariance = np.asarray(innovation_covariance, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or covariance.shape != matrix.shape
        or np.any(~np.isfinite(matrix))
        or np.any(~np.isfinite(covariance))
    ):
        raise ValueError("transition and innovation covariance must be aligned matrices")
    result = solve_discrete_lyapunov(matrix.T, covariance)
    result = 0.5 * (result + result.T)
    eigenvalues, eigenvectors = np.linalg.eigh(result)
    tolerance = 256.0 * np.finfo(np.float64).eps * max(
        float(np.max(np.abs(eigenvalues))), 1.0
    )
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError("invariant state covariance is not positive semidefinite")
    return eigenvectors @ np.diag(np.maximum(eigenvalues, tolerance)) @ eigenvectors.T


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-only", action="store_true")
    for geometry in ("weekly", "daily"):
        required = geometry == "daily"
        parser.add_argument(
            f"--{geometry}-checkpoint", type=Path, required=required
        )
        parser.add_argument(f"--{geometry}-target-history", type=Path)
        parser.add_argument(
            f"--{geometry}-student-report", type=Path, required=required
        )
        parser.add_argument(
            f"--{geometry}-diffusion-report", type=Path, required=required
        )
    parser.add_argument("--weekly-native-checkpoint", type=Path)
    parser.add_argument("--weekly-native-completion", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=54_391)
    parser.add_argument("--representative-date", type=date.fromisoformat, default=date(2024, 6, 14))
    return parser.parse_args()


def extract_tail_probabilities(
    student_report: Mapping[str, Any],
    diffusion_report: Mapping[str, Any],
) -> dict[str, list[float]]:
    """Extract comparable market and simulated pre-2025-threshold tail rates."""
    if student_report.get("schema") != "flex-score-driven-student-t-v1":
        raise ValueError("unexpected Student-t tail-report schema")
    if diffusion_report.get("schema") != "flex-log-variance-diffusion-v1":
        raise ValueError("unexpected stochastic-volatility report schema")
    market = student_report.get("market_tail_diagnostics")
    simulation = student_report.get("simulation")
    diffusion_simulation = diffusion_report.get("simulation")
    if not isinstance(market, Mapping) or not isinstance(simulation, Mapping):
        raise ValueError("tail report has no market or simulation diagnostics")
    if not isinstance(diffusion_simulation, Mapping) or not isinstance(
        diffusion_simulation.get("results"), Mapping
    ):
        raise ValueError("stochastic-volatility report has no simulation results")
    candidate_results: dict[str, Mapping[str, Any]] = {}
    for output_name, report_name in (
        ("iid_student_t", "iid_student_t"),
        ("dynamic_student_t", "dynamic_student_t"),
        ("historical_replay", "historical_replay"),
    ):
        candidate = simulation.get(report_name)
        if not isinstance(candidate, Mapping) or not isinstance(
            candidate.get("results"), Mapping
        ):
            raise ValueError(f"tail report has no {report_name} results")
        candidate_results[output_name] = cast(Mapping[str, Any], candidate["results"])
    candidate_results["stochastic_volatility"] = cast(
        Mapping[str, Any], diffusion_simulation["results"]
    )
    output = {
        "market_2025": [],
        "gaussian": [],
        **{name: [] for name in candidate_results},
    }
    for week in (1, 2, 3):
        market_row = market.get(f"{week}w")
        if not isinstance(market_row, Mapping):
            raise ValueError(f"tail report is missing the {week}w horizon")
        output["market_2025"].append(
            float(market_row["2025_exceedance_of_pre_2025_q95"])
        )
        output["gaussian"].append(
            float(market_row["gaussian_exceedance_of_pre_2025_q95"])
        )
        for name, results in candidate_results.items():
            row = results.get(f"{week}w")
            if not isinstance(row, Mapping):
                raise ValueError(f"{name} results are missing the {week}w horizon")
            output[name].append(float(row["exceedance_probability"]))
    if any(
        not 0.0 <= value <= 1.0
        for values in output.values()
        for value in values
    ):
        raise ValueError("tail probabilities must lie in [0,1]")
    return output


def _stationary_model_quantities(model: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        transition = model.weekly_transition().cpu().double().numpy()
        innovation_factor = model.innovation_cholesky().cpu().double().numpy()
        mean = model.equilibrium_mean.cpu().double().numpy()
    covariance = stationary_state_covariance(
        transition,
        innovation_factor @ innovation_factor.T,
    )
    return mean, covariance, transition


def _iv_from_surface(
    decoder: Any,
    surface: Any,
    *,
    expiry_business_days: np.ndarray,
    nms: np.ndarray,
) -> np.ndarray:
    """Decode annualized IV on a state-specific normalized-moneyness grid."""
    expiry_days = np.asarray(expiry_business_days, dtype=np.float64)
    nms_values = np.asarray(nms, dtype=np.float64)
    if expiry_days.ndim != 1 or nms_values.ndim != 1:
        raise ValueError("expiry days and normalized moneyness must be vectors")
    batch = int(surface.densities.shape[0])
    expiries = expiry_days / DAYS_PER_YEAR
    device = surface.densities.device
    dtype = surface.densities.dtype
    expiry_tensor = torch.as_tensor(expiries, dtype=dtype, device=device)
    ones = torch.ones((batch, len(expiries)), dtype=dtype, device=device)
    with torch.inference_mode():
        atm_calls = decoder.price_calls_batch(
            surface,
            expiries_y=expiry_tensor.unsqueeze(0).expand(batch, -1),
            strikes=ones,
        ).cpu().double().numpy()
    atm_strikes = np.ones_like(atm_calls)
    atm_sqrt_t = np.broadcast_to(np.sqrt(expiries)[None, :], atm_calls.shape)
    atm_vols = bs.implied(
        k=atm_strikes,
        prices=atm_calls,
        sqrtT=atm_sqrt_t,
        price_tol=1.0e-8,
        on_exceed_bounds="quiet",
    )
    strikes = NMS.from_nms(
        nms_values[None, None, :],
        expiries[None, :, None],
        vol=atm_vols[:, :, None],
        eps_vol=atm_vols[:, :1, None],
    )
    flattened_expiries = np.broadcast_to(
        expiries[None, :, None], strikes.shape
    ).reshape(batch, -1).copy()
    flattened_strikes = np.asarray(strikes).reshape(batch, -1).copy()
    with torch.inference_mode():
        calls = decoder.price_calls_batch(
            surface,
            expiries_y=torch.as_tensor(
                flattened_expiries, dtype=dtype, device=device
            ),
            strikes=torch.as_tensor(flattened_strikes, dtype=dtype, device=device),
        ).cpu().double().numpy()
    default_vol = np.broadcast_to(atm_vols[:, :, None], strikes.shape).reshape(
        batch, -1
    )
    vols = bs.implied(
        k=flattened_strikes,
        prices=calls,
        sqrtT=np.sqrt(flattened_expiries),
        default_vol=default_vol,
        price_tol=1.0e-8,
        on_exceed_bounds="quiet",
    )
    result = np.asarray(vols, dtype=np.float64).reshape(
        batch, len(expiries), len(nms_values)
    )
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise RuntimeError("decoded IV grid is not finite and positive")
    return result


def _decoded_iv_grid(
    model: Any,
    states: np.ndarray,
    *,
    expiry_business_days: np.ndarray,
    nms: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != model.config.state_size:
        raise ValueError("state matrix has the wrong Direct-200 geometry")
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for first in range(0, len(values), batch_size):
            state = torch.as_tensor(
                values[first : first + batch_size],
                dtype=model.fdtype,
                device=model.device,
            )
            surface = _decode_surface(model, state)
            chunks.append(
                _iv_from_surface(
                    model.sanos_decoder,
                    surface,
                    expiry_business_days=expiry_business_days,
                    nms=nms,
                )
            )
    return np.concatenate(chunks)


def decoded_iv_pca(
    model: Any,
    *,
    finite_difference_step: float = 1.0e-3,
) -> DecodedIvPca:
    """Linearize the decoded 7x100 IV surface and propagate invariant covariance."""
    if finite_difference_step <= 0.0:
        raise ValueError("finite_difference_step must be positive")
    mean, invariant, _ = _stationary_model_quantities(model)
    model_expiries = np.asarray(model.config.geometry.model_expiries_y) * DAYS_PER_YEAR
    expiry_days = select_pca_expiries(model_expiries)
    nms = np.linspace(-3.0, 2.0, 100)
    dimension = model.config.state_size
    perturbations = np.eye(dimension) * finite_difference_step
    states = np.vstack((mean[None, :], mean + perturbations, mean - perturbations))
    surfaces = _decoded_iv_grid(
        model,
        states,
        expiry_business_days=expiry_days,
        nms=nms,
    )
    mean_iv = surfaces[0]
    jacobian = (
        surfaces[1 : dimension + 1] - surfaces[dimension + 1 :]
    ).reshape(dimension, -1).T / (2.0 * finite_difference_step)
    invariant_factor = np.linalg.cholesky(invariant)
    propagated = jacobian @ invariant_factor
    left, singular_values, _ = np.linalg.svd(propagated, full_matrices=False)
    components = (left * singular_values[None, :]).T.reshape(
        dimension, len(expiry_days), len(nms)
    )
    for index in range(len(components)):
        if float(np.mean(components[index])) < 0.0:
            components[index] *= -1.0
    return DecodedIvPca(
        expiry_business_days=expiry_days,
        nms=nms,
        mean_iv=mean_iv,
        variances=np.square(singular_values),
        one_standard_deviation_surfaces=components,
        invariant_covariance=invariant,
    )


def _save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary, format=path.suffix.lstrip("."), dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(path)


def _plot_pca_spectrum(
    pcas: Mapping[str, DecodedIvPca],
    *,
    output: Path,
) -> None:
    names = tuple(pcas)
    figure, axes = plt.subplots(
        1,
        len(names),
        figsize=(
            _COMPACT_FIGURE_SCALE * 7.9 * len(names),
            _COMPACT_FIGURE_SCALE * 6.9,
        ),
        squeeze=False,
    )
    colors = {"weekly": "#5b83c4", "daily": "#f28e45"}
    for axis, name in zip(axes[0], names, strict=True):
        explained = 100.0 * pcas[name].explained_variance
        shown = min(20, len(explained))
        locations = np.arange(1, shown + 1)
        axis.bar(locations, explained[:shown], color=colors[name], alpha=0.9)
        axis.set_xlabel("PCA Components", fontsize=_PCA_SPECTRUM_TEXT_FONTSIZE)
        axis.set_ylabel(
            "Individual IV variance (%)",
            fontsize=_PCA_SPECTRUM_TEXT_FONTSIZE,
        )
        axis.tick_params(labelsize=_PCA_SPECTRUM_TEXT_FONTSIZE)
        axis.grid(axis="y", alpha=0.25)
        twin = axis.twinx()
        cumulative = 100.0 * np.cumsum(pcas[name].explained_variance[:shown])
        twin.plot(locations, cumulative, color="#d62728", linestyle="-", linewidth=2)
        twin.set_ylim(90.0, 100.15)
        twin.set_ylabel(
            "Cumulative IV variance (%)",
            fontsize=_PCA_SPECTRUM_TEXT_FONTSIZE,
        )
        twin.tick_params(labelsize=_PCA_SPECTRUM_TEXT_FONTSIZE)
        text = (
            f"PC1 {explained[0]:.1f}%\n"
            f"PC1–4 {np.sum(explained[:4]):.1f}%\n"
            f"PC1–8 {np.sum(explained[:8]):.1f}%"
        )
        axis.text(
            0.98,
            0.05,
            text,
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=_PCA_SPECTRUM_TEXT_FONTSIZE,
            bbox={"facecolor": "white", "edgecolor": "#aaaaaa", "alpha": 0.9},
        )
    figure.suptitle(
        "Explained Implied Volatility Variance",
        fontsize=_PCA_SPECTRUM_TITLE_FONTSIZE,
    )
    figure.tight_layout()
    _save_figure(figure, output)


def _plot_tail_exceedance(
    probabilities: Mapping[str, Mapping[str, list[float]]],
    *,
    output: Path,
) -> None:
    names = tuple(probabilities)
    figure, axes = plt.subplots(
        1,
        len(names),
        figsize=(8.05 * len(names), 7.0),
        sharey=True,
        squeeze=False,
    )
    locations = np.arange(3)
    titles = {"weekly": "Weekly", "daily": "Daily + weekly"}
    if names == ("daily",):
        titles["daily"] = "DYSANOS model"
    colors = {
        "market_2025": "#c81e1e",
        "gaussian": "#a8a8a8",
        "iid_student_t": "#62a0ea",
        "dynamic_student_t": "#1f5aa6",
        "stochastic_volatility": "#7b3fb4",
        "historical_replay": "#6cab3f",
    }
    labels = {
        "market_2025": "2025 market",
        "gaussian": "Gaussian",
        "iid_student_t": "IID Student-t",
        "dynamic_student_t": "Score-driven Student-t",
        "stochastic_volatility": "Log-variance diffusion",
        "historical_replay": "Independent historical replay",
    }
    styles = {
        "market_2025": ("o", "-", 2.8),
        "gaussian": ("x", "--", 2.0),
        "iid_student_t": ("^", ":", 2.0),
        "dynamic_student_t": ("D", "-", 2.0),
        "stochastic_volatility": ("s", "-", 2.0),
        "historical_replay": ("v", "-.", 2.0),
    }
    for axis, name in zip(axes[0], names, strict=True):
        values = probabilities[name]
        for method in labels:
            marker, linestyle, linewidth = styles[method]
            axis.semilogy(
                locations,
                100.0 * np.asarray(values[method]),
                marker=marker,
                linestyle=linestyle,
                linewidth=linewidth,
                markersize=7,
                color=colors[method],
                label=labels[method],
            )
        axis.set_xticks(locations, ("1 week", "2 weeks", "3 weeks"))
        axis.set_xlabel("Return horizon")
        axis.set_title(titles[name])
        axis.grid(axis="y", which="both", alpha=0.25)
    axes[0, 0].set_ylabel("Probability above pre-2025 market 95% radius (%)")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower center", ncol=3)
    figure.suptitle(
        "No-replay heavy-tail laws recover far more market-tail mass than Gaussian dynamics",
        fontsize=19,
    )
    figure.tight_layout(rect=(0.0, 0.13, 1.0, 0.95))
    _save_figure(figure, output)


def _base_surface_grid(model: Any, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expiry_days = np.asarray(model.config.geometry.model_expiries_y) * DAYS_PER_YEAR
    nms = np.linspace(-3.0, 2.0, 31)
    iv = _decoded_iv_grid(
        model,
        state[None, :],
        expiry_business_days=expiry_days,
        nms=nms,
    )[0]
    return expiry_days, nms, iv


def _plot_two_surfaces(
    grids: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    title: str,
    output: Path,
    font_scale: float = 1.0,
    show_panel_titles: bool = True,
) -> None:
    names = tuple(grids)
    figure = plt.figure(
        figsize=(
            _COMPACT_FIGURE_SCALE * 7.7 * len(names),
            _COMPACT_FIGURE_SCALE * 7.1,
        )
    )
    axes = [
        figure.add_subplot(1, len(names), index + 1, projection="3d")
        for index in range(len(names))
    ]
    minimum = min(float(np.min(item[2])) for item in grids.values()) * 100.0
    maximum = max(float(np.max(item[2])) for item in grids.values()) * 100.0
    titles = {"weekly": "Weekly nodes", "daily": "Daily + weekly nodes"}
    if names == ("daily",):
        titles["daily"] = "DYSANOS model"
    last_surface = None
    for axis, name in zip(axes, names, strict=True):
        expiry_days, nms, iv = grids[name]
        xx, yy = np.meshgrid(np.sqrt(expiry_days / DAYS_PER_YEAR), nms, indexing="ij")
        x_axis_total = float(np.min(xx) + np.max(xx))
        last_surface = axis.plot_surface(
            x_axis_total - xx,
            yy,
            100.0 * iv,
            cmap="viridis",
            vmin=minimum,
            vmax=maximum,
            linewidth=0.15,
            antialiased=True,
        )
        axis.view_init(elev=25.0, azim=60.0)
        axis.xaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _position, total=x_axis_total: f"{total - value:.1f}"
            )
        )
        if show_panel_titles:
            axis.set_title(
                titles[name],
                fontsize=font_scale * _SURFACE_AXES_TITLE_FONTSIZE,
            )
        # A 3D x-label is placed outside compact canvases by Matplotlib's
        # projected-axis layout. Anchor it in axes coordinates so it remains
        # fully visible at presentation size.
        axis.set_xlabel("")
        axis.text2D(
            0.72,
            -0.10,
            r"$\sqrt{\mathrm{TTE}}$",
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=font_scale * _SURFACE_LABEL_FONTSIZE,
        )
        axis.set_ylabel("NMS", fontsize=font_scale * _SURFACE_LABEL_FONTSIZE)
        axis.tick_params(labelsize=font_scale * _SURFACE_TICK_FONTSIZE)
    figure.subplots_adjust(bottom=0.18, top=0.88)
    if last_surface is not None:
        # The colorbar carries the vertical-axis label in the compact layout.
        colorbar = figure.colorbar(
            last_surface,
            ax=axes,
            shrink=0.65,
            pad=0.10,
        )
        colorbar.set_label(
            "Implied vol (%)",
            fontsize=font_scale * _SURFACE_LABEL_FONTSIZE,
        )
        colorbar.ax.tick_params(labelsize=font_scale * _SURFACE_TICK_FONTSIZE)
    figure.suptitle(title, fontsize=font_scale * _SURFACE_SUPTITLE_FONTSIZE)
    _save_figure(figure, output)


def _plot_pca_factors(pca: DecodedIvPca, *, title: str, output: Path) -> None:
    figure = plt.figure(figsize=(8.4, 12.0))
    xx, yy = np.meshgrid(
        np.sqrt(pca.expiry_business_days / DAYS_PER_YEAR),
        pca.nms,
        indexing="ij",
    )
    x_axis_total = float(np.min(xx) + np.max(xx))
    explained = 100.0 * pca.explained_variance
    values_by_component = 100.0 * pca.one_standard_deviation_surfaces[:8]
    for index in range(8):
        axis = figure.add_subplot(4, 2, index + 1, projection="3d")
        values = values_by_component[index]
        component_maximum = max(float(np.max(np.abs(values))), 1.0e-8)
        axis.plot_surface(
            x_axis_total - xx,
            yy,
            values,
            cmap="viridis",
            norm=TwoSlopeNorm(
                vmin=-component_maximum,
                vcenter=0.0,
                vmax=component_maximum,
            ),
            linewidth=0.0,
            antialiased=True,
        )
        axis.view_init(elev=25.0, azim=60.0)
        axis.xaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _position, total=x_axis_total: f"{total - value:.1f}"
            )
        )
        axis.set_zlim(-component_maximum, component_maximum)
        axis.set_title(
            f"PC {index + 1}: {explained[index]:.2f}%\n"
            f"scale ±{component_maximum:.2f} vol pts",
            fontsize=_PCA_FACTOR_AXES_TITLE_FONTSIZE,
        )
        axis.xaxis.set_rotate_label(False)
        axis.set_xlabel(
            r"$\sqrt{\mathrm{TTE}}$",
            fontsize=_PCA_FACTOR_LABEL_FONTSIZE,
            labelpad=6,
            rotation=0,
        )
        axis.set_ylabel("NMS", fontsize=_PCA_FACTOR_LABEL_FONTSIZE)
        axis.set_zlabel(r"$\Delta$ IV", fontsize=_PCA_FACTOR_LABEL_FONTSIZE)
        axis.tick_params(labelsize=_PCA_FACTOR_TICK_FONTSIZE)
    figure.suptitle(title, fontsize=_PCA_FACTOR_SUPTITLE_FONTSIZE)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98), h_pad=3.0, w_pad=1.0)
    _save_figure(figure, output)


def _plot_invariant_atm_draws(
    models: Mapping[str, Any],
    *,
    seed: int,
    output: Path,
) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed)
    standard = generator.normal(size=(6, 200))
    expiry_days = np.asarray((1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0))
    nms = np.asarray((0.0,))
    grids: dict[str, np.ndarray] = {}
    names = tuple(models)
    figure, axes = plt.subplots(
        1,
        len(names),
        figsize=(
            _COMPACT_FIGURE_SCALE * 8.05 * len(names),
            _COMPACT_FIGURE_SCALE * 7.2,
        ),
        sharey=True,
        squeeze=False,
    )
    titles = {
        "weekly": "Weekly nodes (5D, 10D, …)",
        "daily": "Daily front nodes (1D–5D) + weekly",
    }
    if names == ("daily",):
        titles["daily"] = "DYSANOS model"
    for axis, name in zip(axes[0], names, strict=True):
        model = models[name]
        mean, invariant, _ = _stationary_model_quantities(model)
        factor = np.linalg.cholesky(invariant)
        states = mean + standard @ factor.T
        draw_iv = _decoded_iv_grid(
            model,
            states,
            expiry_business_days=expiry_days,
            nms=nms,
        )[:, :, 0]
        mean_iv = _decoded_iv_grid(
            model,
            mean[None, :],
            expiry_business_days=expiry_days,
            nms=nms,
        )[0, :, 0]
        grids[name] = draw_iv
        for row in draw_iv:
            axis.plot(expiry_days, 100.0 * row, linewidth=1.7, alpha=0.9)
        axis.plot(expiry_days, 100.0 * mean_iv, color="black", linewidth=3.0)
        axis.set_title(titles[name])
        axis.set_xlabel("Expiry (business days)")
        axis.set_ylabel("ATM implied volatility (%)")
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Six draws from the fitted invariant distribution",
        fontsize=_COMPACT_SUPTITLE_FONTSIZE,
    )
    figure.tight_layout()
    _save_figure(figure, output)
    return grids


def _load_native_model(
    checkpoint: Path,
    *,
    device: str,
) -> tuple[BaseEnvironment, NativeStateSanosModel]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    identity = payload.get("model_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("native checkpoint has no model identity")
    values = identity.get("model_config")
    if not isinstance(values, Mapping):
        raise ValueError("native checkpoint has no model configuration")
    geometry_values = values.get("geometry")
    if not isinstance(geometry_values, Mapping):
        raise ValueError("native checkpoint has no geometry")
    geometry_payload = dict(geometry_values)
    for key in ("nms_range", "explicit_model_business_dtes"):
        if geometry_payload.get(key) is not None:
            geometry_payload[key] = tuple(geometry_payload[key])
    geometry_type = (
        FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig
        if geometry_payload.get("front_term_eta") is True
        else SharedStrikeCappedUnboundedDlvFlexGeometryConfig
    )
    config = NativeStateConfig(
        label=str(values["label"]),
        group=str(values["group"]),
        user_version=str(values["user_version"]),
        geometry=geometry_type(**geometry_payload),
    )
    if identity.get("config_version") != config.version.unique_id48:
        raise ValueError("native checkpoint code version is stale")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("native checkpoint has no state dictionary")
    raw = state_dict.get("daily_raw_embedding.weight")
    if not isinstance(raw, torch.Tensor) or raw.ndim != 2:
        raise ValueError("native checkpoint has no daily raw states")
    env = BaseEnvironment(precision=32, device=device, debug=False)
    model = NativeStateSanosModel(env, num_days=len(raw), config=config)
    model.load_state_dict(cast(Mapping[str, torch.Tensor], state_dict))
    model.eval()
    return env, model


def _iv_for_coordinates(
    decoder: Any,
    surface: Any,
    *,
    expiry_y: np.ndarray,
    strikes: np.ndarray,
) -> np.ndarray:
    expiries = np.asarray(expiry_y, dtype=np.float64)
    strike_values = np.asarray(strikes, dtype=np.float64)
    if expiries.shape != strike_values.shape:
        raise ValueError("expiry and strike coordinates must be aligned")
    with torch.inference_mode():
        calls = decoder.price_calls_batch(
            surface,
            expiries_y=torch.as_tensor(
                expiries[None, :], dtype=surface.densities.dtype, device=surface.densities.device
            ),
            strikes=torch.as_tensor(
                strike_values[None, :],
                dtype=surface.densities.dtype,
                device=surface.densities.device,
            ),
        )[0].cpu().double().numpy()
    return np.asarray(
        bs.implied(
            k=strike_values,
            prices=calls,
            sqrtT=np.sqrt(expiries),
            price_tol=1.0e-8,
            on_exceed_bounds="quiet",
        ),
        dtype=np.float64,
    )


def _representative_expiries(frame: Any) -> list[Any]:
    eligible = frame.loc[(~frame["is_am"]) & (~frame["is_american"])].copy()
    valid = (
        np.isfinite(eligible["bid_vol"])
        & np.isfinite(eligible["ask_vol"])
        & (eligible["bid_vol"] > 0.0)
        & (eligible["ask_vol"] >= eligible["bid_vol"])
        & np.isfinite(eligible["pure_strike"])
        & (eligible["pure_strike"] > 0.0)
    )
    eligible = eligible.loc[valid]
    groups = [group for _, group in eligible.groupby("expiry") if len(group) >= 30]
    if len(groups) < 4:
        raise ValueError("representative date has fewer than four liquid expiries")
    selected: list[Any] = []
    used: set[object] = set()
    for target in (10.0, 23.0, 42.0, 66.0):
        candidates = [group for group in groups if group.iloc[0].expiry not in used]
        group = min(candidates, key=lambda item: abs(float(item.iloc[0].bdte) - target))
        used.add(group.iloc[0].expiry)
        selected.append(group.sort_values("pure_strike"))
    return selected


def _plot_representative_fit(
    *,
    joint_model: Any,
    joint_target_history: Path,
    native_model: NativeStateSanosModel,
    native_dates: list[str],
    representative_date: date,
    output: Path,
) -> dict[str, object]:
    history = torch.load(joint_target_history, map_location="cpu", weights_only=False)
    dates = history.get("dates")
    states = history.get("states")
    if not isinstance(dates, list) or not isinstance(states, torch.Tensor):
        raise ValueError("joint target history is invalid")
    date_label = representative_date.isoformat()
    joint_index = dates.index(date_label)
    native_index = native_dates.index(date_label)
    with torch.inference_mode():
        joint_state = states[joint_index].to(
            dtype=joint_model.fdtype, device=joint_model.device
        ).unsqueeze(0)
        joint_surface = _decode_surface(joint_model, joint_state)
        _, native_surface = native_model.decode_days(
            torch.as_tensor((native_index,), dtype=torch.long, device=native_model.device)
        )
    market = enrich_day("SPX", representative_date, verbose=Context.quiet)
    groups = _representative_expiries(market)
    figure, axes = plt.subplots(2, 2, figsize=(15.8, 10.8))
    output_rows: list[dict[str, object]] = []
    for axis, group in zip(axes.flat, groups, strict=True):
        strike = group["pure_strike"].to_numpy(dtype=np.float64)
        expiry = group["tte_y"].to_numpy(dtype=np.float64)
        native_iv = _iv_for_coordinates(
            native_model.sanos_decoder,
            native_surface,
            expiry_y=expiry,
            strikes=strike,
        )
        joint_iv = _iv_for_coordinates(
            joint_model.sanos_decoder,
            joint_surface,
            expiry_y=expiry,
            strikes=strike,
        )
        axis.fill_between(
            strike,
            100.0 * group["bid_vol"].to_numpy(dtype=np.float64),
            100.0 * group["ask_vol"].to_numpy(dtype=np.float64),
            color="#b8c4d6",
            alpha=0.55,
            label="Observed OTM bid–ask IV interval",
        )
        axis.plot(
            strike,
            100.0 * native_iv,
            "--",
            color="#d97706",
            linewidth=2.0,
            label="Per-date direct SANOS",
        )
        axis.plot(
            strike,
            100.0 * joint_iv,
            color="#2563eb",
            linewidth=2.2,
            label="Two-eta affine-200",
        )
        first = group.iloc[0]
        axis.set_title(
            f"{first.expiry:%Y-%m-%d}  •  {float(first.bdte):.1f} business DTE"
            f"  •  {len(group)} books"
        )
        axis.set_xlabel("Normalized strike K/F")
        axis.set_ylabel("Annualized Black IV (%)")
        axis.grid(alpha=0.2)
        output_rows.append(
            {
                "expiry": first.expiry.isoformat(),
                "business_dte": float(first.bdte),
                "books": len(group),
                "native_affine_iv_rmse_vol_points": float(
                    100.0 * np.sqrt(np.mean(np.square(native_iv - joint_iv)))
                ),
            }
        )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=True)
    figure.suptitle(
        f"SPX {date_label} — representative two-eta weekly fits beyond one week",
        fontsize=18,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    _save_figure(figure, output)
    return {"date": date_label, "expiries": output_rows}


def _pca_report(pca: DecodedIvPca) -> dict[str, object]:
    explained = pca.explained_variance
    eigenvalues = np.linalg.eigvalsh(pca.invariant_covariance)
    return {
        "surface_grid": {
            "expiry_business_days": pca.expiry_business_days.tolist(),
            "normalized_moneyness_points": len(pca.nms),
            "normalized_moneyness_range": [float(pca.nms[0]), float(pca.nms[-1])],
        },
        "explained_variance": {
            "pc1": float(explained[0]),
            "pc1_to_2": float(np.sum(explained[:2])),
            "pc1_to_4": float(np.sum(explained[:4])),
            "pc1_to_8": float(np.sum(explained[:8])),
        },
        "invariant_covariance": {
            "condition_number": float(np.max(eigenvalues) / np.min(eigenvalues)),
            "effective_rank": float(
                np.exp(
                    -np.sum(
                        (weights := eigenvalues / np.sum(eigenvalues))
                        * np.log(np.maximum(weights, np.finfo(np.float64).tiny))
                    )
                )
            ),
        },
    }


def main() -> None:
    arguments = _arguments()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    models: dict[str, Any] = {}
    checkpoint_paths = {"daily": arguments.daily_checkpoint}
    student_paths = {"daily": arguments.daily_student_report}
    diffusion_paths = {"daily": arguments.daily_diffusion_report}
    if not arguments.daily_only:
        required_weekly = {
            "weekly checkpoint": arguments.weekly_checkpoint,
            "weekly target history": arguments.weekly_target_history,
            "weekly Student-t report": arguments.weekly_student_report,
            "weekly diffusion report": arguments.weekly_diffusion_report,
            "weekly native checkpoint": arguments.weekly_native_checkpoint,
            "weekly native completion": arguments.weekly_native_completion,
        }
        missing = [name for name, path in required_weekly.items() if path is None]
        if missing:
            raise ValueError(
                "the two-geometry report requires " + ", ".join(missing)
            )
        checkpoint_paths = {
            "weekly": cast(Path, arguments.weekly_checkpoint),
            **checkpoint_paths,
        }
        student_paths = {
            "weekly": cast(Path, arguments.weekly_student_report),
            **student_paths,
        }
        diffusion_paths = {
            "weekly": cast(Path, arguments.weekly_diffusion_report),
            **diffusion_paths,
        }
    for name, checkpoint in checkpoint_paths.items():
        # The IV Jacobian is a central difference of nearby nonlinear decodes.
        # Float64 avoids cancellation that materially distorted its PCA spectrum.
        _, models[name] = load_presentation_model(
            checkpoint,
            device=arguments.device,
            precision=64,
        )
        if not isinstance(
            models[name].config.geometry,
            FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig,
        ):
            raise ValueError(f"{name} checkpoint is not the two-eta geometry")

    pcas = {name: decoded_iv_pca(model) for name, model in models.items()}
    tail_probabilities = {
        name: extract_tail_probabilities(
            json.loads(path.read_text()),
            json.loads(diffusion_paths[name].read_text()),
        )
        for name, path in student_paths.items()
    }
    _plot_tail_exceedance(
        tail_probabilities,
        output=arguments.output_directory / "presentation_tail_exceedance.png",
    )
    _plot_pca_spectrum(
        pcas,
        output=arguments.output_directory / "presentation_decoded_iv_pca_spectrum.png",
    )
    if "weekly" in pcas:
        _plot_pca_factors(
            pcas["weekly"],
            title="Weekly geometry: decoded IV principal components",
            output=arguments.output_directory / "presentation_weekly_iv_pca_factors.png",
        )
    _plot_pca_factors(
        pcas["daily"],
        title=(
            "DYSANOS decoded IV principal components"
            if arguments.daily_only
            else "Daily + weekly geometry: decoded IV principal components"
        ),
        output=arguments.output_directory / "presentation_iv_pca_factors.png",
    )
    mean_grids: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    draw_grids: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    standard_draw = np.random.default_rng(arguments.seed).normal(size=200)
    for name, model in models.items():
        mean, invariant, _ = _stationary_model_quantities(model)
        mean_grids[name] = _base_surface_grid(model, mean)
        draw = mean + np.linalg.cholesky(invariant) @ standard_draw
        draw_grids[name] = _base_surface_grid(model, draw)
    _plot_two_surfaces(
        mean_grids,
        title="Mean Implied Volatility",
        output=arguments.output_directory / "presentation_invariant_mean_iv_surfaces.png",
        font_scale=0.75,
        show_panel_titles=False,
    )
    _plot_two_surfaces(
        draw_grids,
        title="Illustrative invariant IV draw",
        output=arguments.output_directory / "presentation_invariant_surface_draws.png",
    )
    _plot_invariant_atm_draws(
        models,
        seed=arguments.seed + 1,
        output=arguments.output_directory / "presentation_invariant_atm_draws.png",
    )

    representative: dict[str, object] | None = None
    if not arguments.daily_only:
        _, native_model = _load_native_model(
            cast(Path, arguments.weekly_native_checkpoint),
            device=arguments.device,
        )
        native_completion = json.loads(
            cast(Path, arguments.weekly_native_completion).read_text()
        )
        native_dates = native_completion.get("market_dates")
        if not isinstance(native_dates, list):
            raise ValueError("weekly native completion has no market dates")
        representative = _plot_representative_fit(
            joint_model=models["weekly"],
            joint_target_history=cast(Path, arguments.weekly_target_history),
            native_model=native_model,
            native_dates=native_dates,
            representative_date=arguments.representative_date,
            output=(
                arguments.output_directory
                / "presentation_representative_surface_fit.png"
            ),
        )
    report = {
        "schema": "dysanos-two-eta-statistical-figures-v1",
        "status": "complete",
        "models": {
            name: {
                "checkpoint": str(path),
                "checkpoint_sha256": checkpoint_digest(path),
                "geometry": asdict(models[name].config.geometry),
                "pca": _pca_report(pcas[name]),
            }
            for name, path in checkpoint_paths.items()
        },
        "tail_probabilities": tail_probabilities,
        "tail_reports": {
            name: {
                "student_t": {
                    "path": str(path),
                    "sha256": checkpoint_digest(path),
                },
                "stochastic_volatility": {
                    "path": str(diffusion_paths[name]),
                    "sha256": checkpoint_digest(diffusion_paths[name]),
                },
            }
            for name, path in student_paths.items()
        },
        "figures": sorted(
            str(path)
            for path in arguments.output_directory.glob("presentation_*.png")
        ),
    }
    if representative is not None:
        report["representative_fit"] = representative
    write_json_atomic(report, arguments.report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
