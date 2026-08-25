"""PCA and continuous-time AR(1) diagnostics for local-volatility surfaces."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from cdxcore.err import verify, verify_inp
from scipy.linalg import expm, logm
from scipy.optimize import least_squares

from quantdata.biz_calendar import DAYS_PER_YEAR

STATE_COLUMNS = {
    "log_dlvsq": "Log DLV squared",
    "log_forward_volatility": "Log forward volatility",
}


@dataclass(frozen=True, kw_only=True)
class SurfaceDynamicsConfig:
    """Dimension and rendering controls for the first surface model."""

    num_factors: int = 5
    num_visualized_factors: int = 3
    mean_reversion_rank_check: int = 10
    video_fps: int = 12
    video_dpi: int = 100
    optimization_max_nfev: int = 4_000

    def __post_init__(self) -> None:
        verify_inp(
            self.num_factors >= 1,
            lambda: f"num_factors must be positive, found {self.num_factors}",
        )
        verify_inp(
            1 <= self.num_visualized_factors <= self.num_factors,
            lambda: (
                "num_visualized_factors must lie between one and num_factors, "
                f"found {self.num_visualized_factors} and {self.num_factors}"
            ),
        )
        verify_inp(
            self.mean_reversion_rank_check >= self.num_factors,
            lambda: (
                "mean_reversion_rank_check must be at least num_factors, found "
                f"{self.mean_reversion_rank_check} and {self.num_factors}"
            ),
        )
        verify_inp(
            self.video_fps >= 1,
            lambda: f"video_fps must be positive, found {self.video_fps}",
        )
        verify_inp(
            self.video_dpi >= 50,
            lambda: f"video_dpi must be at least 50, found {self.video_dpi}",
        )
        verify_inp(
            self.optimization_max_nfev >= 100,
            lambda: (
                "optimization_max_nfev must be at least 100, found "
                f"{self.optimization_max_nfev}"
            ),
        )


@dataclass(frozen=True)
class SurfacePanel:
    """Balanced date by expiry by NMS panel."""

    dates: list[date]
    expiry_days: np.ndarray
    sqrt_expiry_years: np.ndarray
    effective_nms: np.ndarray
    states: dict[str, np.ndarray]


@dataclass(frozen=True)
class PCAResult:
    """Centered spatial PCA with deterministic component signs."""

    mean_surface: np.ndarray
    loadings: np.ndarray
    scores: np.ndarray
    explained_variance_ratio: np.ndarray


@dataclass(frozen=True)
class MatrixExponentialAR1:
    """Continuous-time factor AR(1) estimate in standardized PCA coordinates."""

    generator: np.ndarray
    factor_scales: np.ndarray
    one_calendar_day_transition: np.ndarray
    eigenvalues: np.ndarray
    half_life_calendar_days: np.ndarray
    spectral_radius_one_day: float
    r_squared: float
    optimizer_success: bool
    optimizer_message: str
    num_function_evaluations: int

    @property
    def is_mean_reverting(self) -> bool:
        return bool(np.all(np.real(self.eigenvalues) < 0.0))


def load_surface_panel(panel_file: Path) -> SurfacePanel:
    """Load and validate the complete balanced statistical panel."""

    frame = pl.read_parquet(panel_file).sort(
        "date_index", "expiry_index", "nms_index"
    )
    required = {
        "date",
        "date_index",
        "expiry_index",
        "expiry_days",
        "nms_index",
        "effective_nms",
        *STATE_COLUMNS,
    }
    missing = required - set(frame.columns)
    verify(not missing, lambda: f"panel is missing columns {sorted(missing)}")

    num_dates = int(frame["date_index"].n_unique())
    num_expiries = int(frame["expiry_index"].n_unique())
    num_strikes = int(frame["nms_index"].n_unique())
    expected_rows = num_dates * num_expiries * num_strikes
    verify(
        frame.height == expected_rows,
        lambda: f"panel has {frame.height} rows; expected {expected_rows}",
    )
    key_count = int(
        frame.select(
            pl.struct("date_index", "expiry_index", "nms_index").n_unique()
        ).item()
    )
    verify(
        key_count == expected_rows,
        lambda: "panel keys are not a complete unique Cartesian grid",
    )

    dates = (
        frame.group_by("date_index")
        .agg(pl.col("date").first())
        .sort("date_index")["date"]
        .to_list()
    )
    expiry_days = (
        frame.group_by("expiry_index")
        .agg(pl.col("expiry_days").first())
        .sort("expiry_index")["expiry_days"]
        .to_numpy()
    )
    effective_nms = (
        frame.group_by("nms_index")
        .agg(pl.col("effective_nms").first())
        .sort("nms_index")["effective_nms"]
        .to_numpy()
    )
    nms_cube = frame["effective_nms"].to_numpy().reshape(
        num_dates, num_expiries, num_strikes
    )
    expiry_cube = frame["expiry_days"].to_numpy().reshape(
        num_dates, num_expiries, num_strikes
    )
    verify(
        np.allclose(nms_cube, effective_nms[None, None, :], rtol=0.0, atol=1e-13),
        lambda: "effective NMS coordinates vary across dates or expiries",
    )
    verify(
        np.all(expiry_cube == expiry_days[None, :, None]),
        lambda: "expiry coordinates vary across dates or strikes",
    )

    states: dict[str, np.ndarray] = {}
    for column in STATE_COLUMNS:
        values = frame[column].to_numpy().reshape(
            num_dates, num_expiries, num_strikes
        )
        verify(
            np.all(np.isfinite(values)),
            lambda column=column: f"{column} contains non-finite observations",
        )
        states[column] = values
    return SurfacePanel(
        dates=dates,
        expiry_days=np.asarray(expiry_days, dtype=np.int16),
        sqrt_expiry_years=np.sqrt(
            np.asarray(expiry_days, dtype=np.float64) / DAYS_PER_YEAR
        ),
        effective_nms=np.asarray(effective_nms, dtype=np.float64),
        states=states,
    )


def compute_surface_pca(values: np.ndarray, *, num_factors: int) -> PCAResult:
    """Compute spatial PCA after centering each expiry/NMS grid point over time."""

    verify_inp(
        values.ndim == 3,
        lambda: f"values must have shape [date, expiry, strike], found {values.shape}",
    )
    num_dates, num_expiries, num_strikes = values.shape
    verify_inp(
        1 <= num_factors < num_dates,
        lambda: f"num_factors must lie in [1, {num_dates - 1}], found {num_factors}",
    )
    mean_surface = np.mean(values, axis=0)
    centered = (values - mean_surface[None, :, :]).reshape(num_dates, -1)
    left, singular_values, right_t = np.linalg.svd(centered, full_matrices=False)
    total_variance = float(np.sum(np.square(singular_values)))
    verify(total_variance > 0.0, lambda: "surface panel has zero temporal variance")
    explained = np.square(singular_values) / total_variance
    loadings = right_t[:num_factors].reshape(
        num_factors, num_expiries, num_strikes
    )
    scores = left[:, :num_factors] * singular_values[None, :num_factors]

    # Fix the arbitrary SVD sign so repeated runs produce identical factor plots.
    for factor_index in range(num_factors):
        flat = loadings[factor_index].reshape(-1)
        if flat[int(np.argmax(np.abs(flat)))] < 0.0:
            loadings[factor_index] *= -1.0
            scores[:, factor_index] *= -1.0
    verify(
        np.allclose(
            np.mean(scores, axis=0),
            0.0,
            rtol=0.0,
            atol=1e-10,
        ),
        lambda: "centered PCA scores do not have zero sample mean",
    )
    return PCAResult(
        mean_surface=mean_surface,
        loadings=loadings,
        scores=scores,
        explained_variance_ratio=explained,
    )


def fit_matrix_exponential_ar1(
    scores: np.ndarray,
    dates: list[date],
    *,
    max_nfev: int,
) -> MatrixExponentialAR1:
    """Fit ``z(t+dt) = exp(G dt) z(t) + error`` using actual calendar gaps.

    PCA defines the long-run mean surface, so the factor-space equilibrium is
    fixed at zero. Factors are scaled to unit sample standard deviation during
    estimation to keep the nonlinear least-squares objective balanced.
    """

    verify_inp(
        scores.ndim == 2 and scores.shape[0] == len(dates),
        lambda: (
            "scores must have shape [date, factor] and align with dates; found "
            f"{scores.shape} and {len(dates)} dates"
        ),
    )
    verify_inp(
        scores.shape[0] >= scores.shape[1] * 4,
        lambda: "too few time observations for the requested matrix AR(1)",
    )
    factor_scales = np.std(scores, axis=0, ddof=1)
    verify(
        np.all(np.isfinite(factor_scales)) and np.all(factor_scales > 0.0),
        lambda: f"invalid PCA factor scales {factor_scales}",
    )
    standardized = scores / factor_scales[None, :]
    previous = standardized[:-1]
    current = standardized[1:]
    delta_years = np.asarray(
        [
            (right - left).days / 365.25
            for left, right in zip(dates, dates[1:], strict=False)
        ],
        dtype=np.float64,
    )
    verify(
        np.all(delta_years > 0.0),
        lambda: "surface dates must be strictly increasing",
    )
    unique_gaps = np.unique(delta_years)
    num_factors = scores.shape[1]

    one_step = np.linalg.lstsq(previous, current, rcond=None)[0].T
    initial = logm(one_step) / float(np.mean(delta_years))
    if np.max(np.abs(np.imag(initial))) > 1e-8:
        lag_correlations = np.asarray(
            [
                np.corrcoef(previous[:, index], current[:, index])[0, 1]
                for index in range(num_factors)
            ]
        )
        lag_correlations = np.clip(lag_correlations, 1e-3, 0.999)
        initial = np.diag(np.log(lag_correlations) / float(np.mean(delta_years)))
    else:
        initial = np.real(initial)

    def predict(generator: np.ndarray) -> np.ndarray:
        predictions = np.empty_like(current)
        for gap in unique_gaps:
            mask = delta_years == gap
            transition = expm(generator * gap)
            predictions[mask] = previous[mask] @ transition.T
        return predictions

    def residuals(parameters: np.ndarray) -> np.ndarray:
        generator = parameters.reshape(num_factors, num_factors)
        return (current - predict(generator)).reshape(-1)

    optimized = least_squares(
        residuals,
        np.asarray(initial, dtype=np.float64).reshape(-1),
        max_nfev=max_nfev,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )
    generator = optimized.x.reshape(num_factors, num_factors)
    predictions = predict(generator)
    residual_sum_squares = float(np.sum(np.square(current - predictions)))
    total_sum_squares = float(np.sum(np.square(current)))
    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    eigenvalues = np.linalg.eigvals(generator)
    real_parts = np.real(eigenvalues)
    half_lives = np.full(num_factors, np.inf, dtype=np.float64)
    reverting = real_parts < 0.0
    half_lives[reverting] = math.log(2.0) / (-real_parts[reverting]) * 365.25
    one_day_transition = expm(generator / 365.25)
    return MatrixExponentialAR1(
        generator=generator,
        factor_scales=factor_scales,
        one_calendar_day_transition=one_day_transition,
        eigenvalues=eigenvalues,
        half_life_calendar_days=half_lives,
        spectral_radius_one_day=float(
            np.max(np.abs(np.linalg.eigvals(one_day_transition)))
        ),
        r_squared=float(r_squared),
        optimizer_success=bool(optimized.success),
        optimizer_message=str(optimized.message),
        num_function_evaluations=int(optimized.nfev),
    )


def _pca_frames(
    panel: SurfacePanel,
    pca: dict[str, PCAResult],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    loading_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for state, result in pca.items():
        cumulative = np.cumsum(result.explained_variance_ratio)
        for factor_index, variance_ratio in enumerate(
            result.explained_variance_ratio, start=1
        ):
            summary_rows.append(
                {
                    "state": state,
                    "factor": factor_index,
                    "explained_variance_ratio": float(variance_ratio),
                    "cumulative_explained_variance_ratio": float(
                        cumulative[factor_index - 1]
                    ),
                }
            )
        for factor_index, loading in enumerate(result.loadings, start=1):
            for expiry_index, (expiry_days, sqrt_expiry, values) in enumerate(
                zip(
                    panel.expiry_days,
                    panel.sqrt_expiry_years,
                    loading,
                    strict=True,
                )
            ):
                for nms_index, (nms, value) in enumerate(
                    zip(panel.effective_nms, values, strict=True)
                ):
                    loading_rows.append(
                        {
                            "state": state,
                            "factor": factor_index,
                            "expiry_index": expiry_index,
                            "expiry_days": int(expiry_days),
                            "sqrt_expiry_years": float(sqrt_expiry),
                            "nms_index": nms_index,
                            "effective_nms": float(nms),
                            "loading": float(value),
                        }
                    )
        for date_index, (market_date, values) in enumerate(
            zip(panel.dates, result.scores, strict=True)
        ):
            for factor_index, value in enumerate(values, start=1):
                score_rows.append(
                    {
                        "state": state,
                        "date": market_date,
                        "date_index": date_index,
                        "factor": factor_index,
                        "score": float(value),
                    }
                )
    return (
        pl.DataFrame(summary_rows),
        pl.DataFrame(loading_rows),
        pl.DataFrame(score_rows),
    )


def _plot_pca_surfaces(
    panel: SurfacePanel,
    pca: dict[str, PCAResult],
    *,
    num_factors: int,
    output_file: Path,
) -> None:
    import matplotlib.pyplot as plt

    nms_mesh, expiry_mesh = np.meshgrid(
        panel.effective_nms, panel.sqrt_expiry_years
    )
    figure = plt.figure(figsize=(6.3 * num_factors, 10.5))
    for row, (state, result) in enumerate(pca.items()):
        for factor_index in range(num_factors):
            axis = figure.add_subplot(
                len(pca), num_factors, row * num_factors + factor_index + 1,
                projection="3d",
            )
            values = result.loadings[factor_index]
            bound = float(np.max(np.abs(values)))
            surface = axis.plot_surface(
                nms_mesh,
                expiry_mesh,
                values,
                cmap="coolwarm",
                vmin=-bound,
                vmax=bound,
                linewidth=0.0,
                antialiased=True,
            )
            axis.set_xlabel("effective NMS")
            axis.set_ylabel(r"$\sqrt{T}$ (years$^{1/2}$)")
            axis.set_zlabel("PCA loading")
            axis.set_title(
                f"{STATE_COLUMNS[state]} PC{factor_index + 1}\n"
                f"{100.0 * result.explained_variance_ratio[factor_index]:.2f}% variance"
            )
            axis.view_init(elev=28, azim=-125)
            figure.colorbar(surface, ax=axis, fraction=0.025, pad=0.08)
    figure.suptitle("SPX surface PCA loadings", fontsize=16)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(output_file, dpi=150)
    plt.close(figure)


def _find_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is not None:
        return executable
    try:
        import imageio_ffmpeg
    except ImportError as exception:
        raise RuntimeError(
            "ffmpeg is required for MP4 output; install ffmpeg or imageio-ffmpeg"
        ) from exception
    return imageio_ffmpeg.get_ffmpeg_exe()


def _render_surface_video(
    panel: SurfacePanel,
    *,
    output_file: Path,
    fps: int,
    dpi: int,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    mpl.rcParams["animation.ffmpeg_path"] = _find_ffmpeg()
    nms_mesh, expiry_mesh = np.meshgrid(
        panel.effective_nms, panel.sqrt_expiry_years
    )
    state_items = list(STATE_COLUMNS.items())
    limits = {
        state: (
            float(np.min(panel.states[state])),
            float(np.max(panel.states[state])),
        )
        for state, _ in state_items
    }
    figure = plt.figure(figsize=(13.5, 6.4))
    axes = [
        figure.add_subplot(1, len(state_items), index + 1, projection="3d")
        for index in range(len(state_items))
    ]

    def render(frame_index: int) -> list[object]:
        artists: list[object] = []
        for axis, (state, label) in zip(axes, state_items, strict=True):
            axis.clear()
            values = panel.states[state][frame_index]
            surface = axis.plot_surface(
                nms_mesh,
                expiry_mesh,
                values,
                cmap="viridis",
                linewidth=0.0,
                antialiased=True,
            )
            axis.set_xlim(
                float(panel.effective_nms[0]), float(panel.effective_nms[-1])
            )
            axis.set_ylim(
                float(panel.sqrt_expiry_years[0]),
                float(panel.sqrt_expiry_years[-1]),
            )
            axis.set_zlim(*limits[state])
            axis.set_xlabel("effective NMS")
            axis.set_ylabel(r"$\sqrt{T}$ (years$^{1/2}$)")
            axis.set_zlabel(label)
            axis.set_title(f"{label}\n{panel.dates[frame_index].isoformat()}")
            axis.view_init(elev=28, azim=-125)
            artists.append(surface)
        figure.suptitle("SPX forward-relative SANOS state surface")
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        return artists

    animation = FuncAnimation(
        figure,
        render,
        frames=len(panel.dates),
        interval=1_000 / fps,
        blit=False,
    )
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=3_000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    animation.save(output_file, writer=writer, dpi=dpi)
    plt.close(figure)


def _matrix_rows(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def _ar_report(model: MatrixExponentialAR1) -> dict[str, object]:
    return {
        "factor_coordinate": "PCA scores divided by sample standard deviation",
        "time_coordinate": "calendar years using actual gaps between observations",
        "equilibrium": "zero PCA score, corresponding to the sample mean surface",
        "equation": "z(t+dt) = exp(G*dt) z(t) + error",
        "generator_G_per_calendar_year": _matrix_rows(model.generator),
        "one_calendar_day_transition": _matrix_rows(
            model.one_calendar_day_transition
        ),
        "generator_eigenvalues": [
            {"real": float(np.real(value)), "imaginary": float(np.imag(value))}
            for value in model.eigenvalues
        ],
        "mode_half_life_calendar_days": model.half_life_calendar_days.tolist(),
        "spectral_radius_one_calendar_day": model.spectral_radius_one_day,
        "mean_reverting": model.is_mean_reverting,
        "in_sample_r_squared_standardized_factors": model.r_squared,
        "factor_sample_standard_deviations": model.factor_scales.tolist(),
        "optimizer": {
            "success": model.optimizer_success,
            "message": model.optimizer_message,
            "num_function_evaluations": model.num_function_evaluations,
        },
    }


def _rank_check_frame(
    panel: SurfacePanel,
    pca: dict[str, PCAResult],
    *,
    maximum_rank: int,
    max_nfev: int,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for state, result in pca.items():
        for rank in range(1, maximum_rank + 1):
            model = fit_matrix_exponential_ar1(
                result.scores[:, :rank],
                panel.dates,
                max_nfev=max_nfev,
            )
            finite_half_lives = model.half_life_calendar_days[
                np.isfinite(model.half_life_calendar_days)
            ]
            rows.append(
                {
                    "state": state,
                    "num_factors": rank,
                    "mean_reverting": model.is_mean_reverting,
                    "spectral_radius_one_calendar_day": (
                        model.spectral_radius_one_day
                    ),
                    "minimum_mode_half_life_calendar_days": (
                        float(np.min(finite_half_lives))
                        if len(finite_half_lives)
                        else None
                    ),
                    "maximum_mode_half_life_calendar_days": (
                        float(np.max(finite_half_lives))
                        if len(finite_half_lives)
                        else None
                    ),
                    "in_sample_r_squared_standardized_factors": model.r_squared,
                }
            )
    return pl.DataFrame(rows)


def run_surface_dynamics(
    *,
    panel_file: Path = Path(
        "aae/ideas/localvolbysanos/results/2025_full_reference_grid/"
        "balanced_panel.parquet"
    ),
    output_directory: Path = Path(
        "aae/ideas/localvolbysanos/results/2025_full_reference_grid"
    ),
    config: SurfaceDynamicsConfig = SurfaceDynamicsConfig(),
    render_video: bool = True,
) -> dict[str, object]:
    """Fit PCA/AR models and write diagnostics for the balanced surface panel."""

    panel = load_surface_panel(panel_file)
    rank_pca = {
        state: compute_surface_pca(
            values, num_factors=config.mean_reversion_rank_check
        )
        for state, values in panel.states.items()
    }
    pca = {
        state: PCAResult(
            mean_surface=result.mean_surface,
            loadings=result.loadings[: config.num_factors],
            scores=result.scores[:, : config.num_factors],
            explained_variance_ratio=result.explained_variance_ratio,
        )
        for state, result in rank_pca.items()
    }
    rank_check = _rank_check_frame(
        panel,
        rank_pca,
        maximum_rank=config.mean_reversion_rank_check,
        max_nfev=config.optimization_max_nfev,
    )
    ar_models = {
        state: fit_matrix_exponential_ar1(
            result.scores,
            panel.dates,
            max_nfev=config.optimization_max_nfev,
        )
        for state, result in pca.items()
    }
    pca_summary, pca_loadings, pca_scores = _pca_frames(panel, pca)
    output_directory.mkdir(parents=True, exist_ok=True)
    pca_summary.write_parquet(
        output_directory / "pca_summary.parquet", compression="zstd"
    )
    pca_loadings.write_parquet(
        output_directory / "pca_loadings.parquet", compression="zstd"
    )
    pca_scores.write_parquet(
        output_directory / "pca_scores.parquet", compression="zstd"
    )
    rank_check.write_parquet(
        output_directory / "mean_reversion_rank_check.parquet", compression="zstd"
    )
    _plot_pca_surfaces(
        panel,
        pca,
        num_factors=config.num_visualized_factors,
        output_file=output_directory / "pca_factors_3d.png",
    )
    if render_video:
        _render_surface_video(
            panel,
            output_file=output_directory / "surface_history_3d.mp4",
            fps=config.video_fps,
            dpi=config.video_dpi,
        )

    report: dict[str, object] = {
        "panel_file": str(panel_file),
        "num_dates": len(panel.dates),
        "date_range": [panel.dates[0].isoformat(), panel.dates[-1].isoformat()],
        "surface_shape": [len(panel.expiry_days), len(panel.effective_nms)],
        "expiry_coordinate": "sqrt(expiry_years)",
        "strike_coordinate": "front_week_vol_scaled_effective_NMS",
        "num_factors": config.num_factors,
        "pca": {
            state: {
                "explained_variance_ratio": result.explained_variance_ratio[
                    : config.num_factors
                ].tolist(),
                "cumulative_explained_variance_ratio": np.cumsum(
                    result.explained_variance_ratio[: config.num_factors]
                ).tolist(),
            }
            for state, result in pca.items()
        },
        "matrix_exponential_ar1": {
            state: _ar_report(model) for state, model in ar_models.items()
        },
        "mean_reversion_rank_check": {
            "maximum_num_factors": config.mean_reversion_rank_check,
            "all_models_mean_reverting": bool(
                rank_check["mean_reverting"].all()
            ),
            "models": rank_check.to_dicts(),
        },
        "artifacts": {
            "pca_summary": "pca_summary.parquet",
            "pca_loadings": "pca_loadings.parquet",
            "pca_scores": "pca_scores.parquet",
            "mean_reversion_rank_check": "mean_reversion_rank_check.parquet",
            "pca_3d_plot": "pca_factors_3d.png",
            "surface_video": "surface_history_3d.mp4" if render_video else None,
        },
    }
    with (output_directory / "surface_dynamics_report.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


if __name__ == "__main__":
    completed_report = run_surface_dynamics()
    print(json.dumps(completed_report, indent=2))
