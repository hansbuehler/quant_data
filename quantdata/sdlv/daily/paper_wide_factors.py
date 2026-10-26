"""Reproduce the main21 mean-IV and PCA figures from the completed wider fit.

Run on CPU. Load the saved Q-audit diffusion rather than fitting new factors;
average implied volatilities across dates on a common market-ATM NMS grid.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from cdxcore.err import verify_inp
from numpy.typing import NDArray

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.sdlv.daily.front_spot.estimation import decode_dlvsq
from quantdata.sdlv.daily.front_spot.jax_model import decode_density_tower
from quantdata.sdlv.daily.front_spot.paper_experiments import implied_volatility_from_calls
from quantdata.sdlv.daily.paper_wide_grid import sha256
from quantdata.sdlv.daily.vix_comparison import mixture_prices

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


def saved_factor_spectrum(
    annualized_loadings: NDArray[np.float64],
    residual_eigenvalues: NDArray[np.float64],
    *,
    expected_fraction: float,
) -> NDArray[np.float64]:
    """Spot-loading power followed by residual PCA power, not PCA of IV levels."""
    loadings = np.asarray(annualized_loadings, dtype=np.float64)
    eigenvalues = np.asarray(residual_eigenvalues, dtype=np.float64)
    verify_inp(loadings.ndim == 3 and eigenvalues.ndim == 1, "Invalid factor geometry")
    verify_inp(np.all(np.isfinite(loadings)) and np.all(np.isfinite(eigenvalues))
               and np.all(eigenvalues >= 0), "Invalid saved diffusion")
    rank = loadings.shape[-1]
    daily_power = np.sum(loadings**2, axis=(0, 1)) / DAYS_PER_YEAR
    # The first loading is spot-oriented; only the remaining columns are
    # covariance eigenvectors. Do not count their eigenvalues a second time.
    np.testing.assert_allclose(daily_power[1:], eigenvalues[:rank - 1], rtol=1e-10, atol=1e-12)
    power = np.concatenate(([daily_power[0]], eigenvalues))
    if power.sum() <= 0:
        raise ValueError("Zero total diffusion variance")
    fractions = power / power.sum()
    np.testing.assert_allclose(fractions[:rank].sum(), expected_fraction, rtol=1e-10, atol=1e-12)
    return fractions


def model_iv_surface(
    support: NDArray[np.float64],
    tower: NDArray[np.float64],
    strikes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Annual Black IV at expiry-specific strikes, with stable OTM inversion.

    Prices are undiscounted and forward-normalized. Using puts below forward
    avoids subtracting intrinsic value from near-unit calls in the left wing.
    """
    support, tower, strikes = (np.asarray(value, dtype=np.float64) for value in (support, tower, strikes))
    verify_inp(support.ndim == 1 and tower.ndim == 2 and strikes.ndim == 2
               and tower.shape == (strikes.shape[0], len(support)), "Invalid surface geometry")
    verify_inp(np.all(np.isfinite(support)) and np.all(support > 0)
               and np.all(np.isfinite(strikes)) and np.all(strikes > 0)
               and np.all(np.isfinite(tower)) and np.all(tower >= 0), "Invalid surface inputs")
    np.testing.assert_allclose(tower.sum(axis=-1), 1.0, atol=1e-10, rtol=0)
    np.testing.assert_allclose(tower @ support, 1.0, atol=1e-10, rtol=0)
    days = np.arange(1, len(tower) + 1, dtype=np.float64)
    otm_prices = []
    for day, density, strike in zip(days, tower, strikes, strict=True):
        calls, puts = mixture_prices(support, density, strike, business_days=float(day))
        otm_prices.append(np.where(strike < 1, puts, calls))
    target = np.asarray(otm_prices)
    if np.any(~np.isfinite(target) | (target <= 0) | (target >= np.minimum(strikes, 1.0))):
        raise ValueError("Non-invertible OTM prices; do not silently clip wing IV")
    # Black put(F=1,K) equals Black call(F=K,K=1) at identical volatility.
    iv = implied_volatility_from_calls(np.minimum(strikes, 1.0), np.maximum(strikes, 1.0),
                                       days[:, None] / DAYS_PER_YEAR, target)
    if np.any(~np.isfinite(iv) | (iv <= 0)):
        raise FloatingPointError("Invalid implied volatility")
    return iv


def nms_strikes(atm_vol: NDArray[np.float64], nms: NDArray[np.float64]) -> NDArray[np.float64]:
    """The same regularized ATM scale as the calibration, at each expiry."""
    atm_vol, nms = np.asarray(atm_vol), np.asarray(nms)
    verify_inp(atm_vol.ndim == 1 and nms.ndim == 1 and np.all(np.isfinite(atm_vol))
               and np.all(atm_vol > 0) and np.all(np.isfinite(nms)), "Invalid NMS coordinates")
    return NMS.from_nms(nms[None, :], np.arange(1, len(atm_vol) + 1)[:, None] / DAYS_PER_YEAR,
                        vol=atm_vol[:, None], eps_vol=float(atm_vol[0]))


def _save(figure: plt.Figure, output: Path, name: str) -> None:
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight", pad_inches=0.1)
    plt.close(figure)


def _surface_axis(axis: plt.Axes, nms: np.ndarray, values: np.ndarray, *, norm: Normalize) -> object:
    days, scores = np.meshgrid(np.arange(1, len(values) + 1), nms, indexing="ij")
    surface = axis.plot_surface(days, scores, values, cmap="coolwarm", norm=norm,
                                linewidth=0, rcount=len(days), ccount=len(nms), rasterized=True)
    axis.view_init(elev=25, azim=-128)
    axis.set_box_aspect((1.25, 1, 0.75))
    axis.xaxis.set_major_locator(MaxNLocator(3))
    axis.yaxis.set_major_locator(MaxNLocator(3))
    axis.zaxis.set_major_locator(MaxNLocator(3))
    axis.tick_params(labelsize=9, pad=1)
    return surface


def plot_surfaces(nms: np.ndarray, mean_iv: np.ndarray, reference_iv: np.ndarray,
                  responses: np.ndarray, fractions: np.ndarray, output: Path) -> None:
    """Plot historical mean, reference state, and all eight saved factor responses."""
    figure = plt.figure(figsize=(10.5, 4.5), layout="constrained")
    axes = []
    norm = Normalize(100 * min(mean_iv.min(), reference_iv.min()), 100 * max(mean_iv.max(), reference_iv.max()))
    for index, (values, title) in enumerate(((mean_iv, "Historical mean: 891 dates"),
                                            (reference_iv, "Reference state: 29 Aug 2025"))):
        axis = figure.add_subplot(1, 2, index + 1, projection="3d")
        surface = _surface_axis(axis, nms, 100 * values, norm=norm)
        axis.set_title(title, fontsize=12)
        axis.set_xlabel("Business days", fontsize=10, labelpad=2)
        axis.set_ylabel("NMS", fontsize=10, labelpad=2)
        axis.set_zlim(norm.vmin, norm.vmax)
        axes.append(axis)
    figure.colorbar(surface, ax=axes, shrink=0.62, pad=0.04, label="Annualized implied volatility (%)")
    figure.suptitle("Gromm-v1-4,3: implied-volatility surfaces\nFinal fit 20261023-kVyePV", fontsize=13)
    _save(figure, output, "mean_and_reference_iv")

    # Each signed panel has its own symmetric scale, stated in the caption.
    # Heatmaps keep all eight directions readable in the compact addendum.
    figure, axes = plt.subplots(2, 4, figsize=(10.5, 5.7), layout="constrained", sharex=True, sharey=True)
    for factor, (axis, response) in enumerate(zip(axes.flat, responses, strict=True)):
        limit = max(float(np.max(np.abs(response))), 1e-12)
        mesh = axis.pcolormesh(nms, np.arange(1, len(response) + 1), response,
                               shading="nearest", cmap="coolwarm", norm=TwoSlopeNorm(0, -limit, limit),
                               rasterized=True)
        label = "F1: spot-aligned" if factor == 0 else f"F{factor + 1}: residual PC{factor}"
        axis.set_title(f"{label}\n{100 * fractions[factor]:.2f}% variance", fontsize=10)
        axis.set_xticks([-4, -2, 0, 3])
        axis.set_yticks([1, 5, 15, 25])
        axis.tick_params(labelsize=9)
        figure.colorbar(mesh, ax=axis, shrink=0.85).ax.tick_params(labelsize=8)
    figure.supxlabel("NMS (reference-date ATM scale held fixed)", fontsize=11)
    figure.supylabel("Business days to expiry", fontsize=11)
    figure.suptitle("Gromm-v1-4,3: central one-day factor IV responses (bp)\nSaved Q-audit diffusion; 29 Aug 2025 reference state", fontsize=13)
    _save(figure, output, "pca_iv_responses")


def plot_spectrum(fractions: np.ndarray, output: Path) -> None:
    """Show the saved spectrum through the first 90% crossing and the rank-eight cap."""
    cumulative = np.cumsum(fractions)
    rank90 = int(np.flatnonzero(cumulative >= 0.90)[0]) + 1
    shown = min(max(32, rank90 + 2), len(fractions))
    ranks = np.arange(1, shown + 1)
    figure, axis = plt.subplots(figsize=(9, 3.5), layout="constrained")
    axis.bar(ranks, 100 * fractions[:shown], color="#4C78A8", label="Individual")
    axis.set_xlabel("Factor rank: spot direction, then residual PCs")
    axis.set_ylabel("Individual variance (%)")
    other = axis.twinx()
    other.plot(ranks, 100 * cumulative[:shown], color="#E45756", marker=".", label="Cumulative")
    other.axvline(8, color="black", linestyle=":", label="Retained rank 8")
    other.axhline(90, color="grey", linestyle="--", linewidth=1)
    other.set_ylim(0, 100)
    other.set_ylabel("Cumulative variance (%)")
    axis.set_title(f"Gromm-v1-4,3: saved innovation spectrum (710 training pairs)\n"
                   f"Rank 8: {100 * cumulative[7]:.2f}%; first 90% crossing: rank {rank90}")
    handles, labels = axis.get_legend_handles_labels()
    more_handles, more_labels = other.get_legend_handles_labels()
    axis.legend(handles + more_handles, labels + more_labels, loc="center right", fontsize=9)
    _save(figure, output, "pca_explained_variance")


def run(*, training_run: Path, audit_directory: Path, output_directory: Path,
        publish_directory: Path | None = None) -> dict:
    """Generate new illustrations without recalibrating or simulating paths."""
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise RuntimeError("Set JAX_PLATFORMS=cpu and JAX_ENABLE_X64=1")
    report_path, audit_path = training_run / "fit/report.json", audit_directory / "q-audit.json"
    fit, audit = (json.loads(path.read_text()) for path in (report_path, audit_path))
    checkpoint, dataset, diffusion = Path(fit["checkpoint"]), training_run / "market.npz", audit_directory / "diffusion.npz"
    if fit["status"] != "complete" or fit["completed_epochs"] != 1000 or audit["status"] != "complete":
        raise ValueError("Figures require the completed fit and Q audit")
    digests = {str(path.resolve()): sha256(path) for path in (report_path, audit_path, checkpoint, dataset, diffusion)}
    for key, path in (("checkpoint_sha256", checkpoint), ("dataset_sha256", dataset)):
        if audit["source"][key] != digests[str(path.resolve())]:
            raise ValueError(f"Q audit and fitted source disagree: {key}")
    if fit["dataset_sha256"] != digests[str(dataset.resolve())]:
        raise ValueError("Training report refers to a different dataset")
    with np.load(checkpoint, allow_pickle=False) as saved:
        raw = saved["raw"]
        if str(saved["dataset_sha256"].item()) != fit["dataset_sha256"]:
            raise ValueError("Checkpoint dataset identity mismatch")
    with np.load(dataset, allow_pickle=False) as saved:
        dates = saved["dates"].astype(str)
        supports = saved["supports"].astype(np.float64)
        # The initializer records the market ATM volatility scale, not the
        # trained DLV or the model's post-fit ATM implied volatility.
        atm_vol = np.sqrt(decode_dlvsq(saved["raw_initial"][..., 0]))
    with np.load(diffusion, allow_pickle=False) as saved:
        loadings, eigenvalues = saved["annualized_loadings"], saved["eigenvalues"]
        ref_support, ref_dlvsq, ref_tower = saved["support"], saved["dlvsq"], saved["density_tower"]
        pair_count = len(saved["origin_indices"])
    if len(dates) != 891 or dates[-1] != "2025-08-29" or pair_count != 710 or loadings.shape != (24, 200, 8):
        raise ValueError("Unexpected completed-run geometry")
    np.testing.assert_array_equal(ref_support, supports[-1])
    np.testing.assert_array_equal(ref_dlvsq, decode_dlvsq(raw[-1]))
    fractions = saved_factor_spectrum(loadings, eigenvalues,
                                      expected_fraction=audit["calibration"]["smoothed_explained_variance_fraction"])
    output_directory.mkdir(parents=True, exist_ok=False)
    nms = np.linspace(-4, 3, 57)
    decode = jax.jit(lambda support, dlvsq: decode_density_tower(support, dlvsq, dt=1 / DAYS_PER_YEAR))
    mean_iv = np.zeros((25, len(nms)))
    started = time.monotonic()
    reference_iv = None
    for index, support in enumerate(supports):
        tower = np.asarray(decode(jnp.asarray(support), jnp.asarray(decode_dlvsq(raw[index]))))
        iv = model_iv_surface(support, tower, nms_strikes(atm_vol[index], nms))
        mean_iv += (iv - mean_iv) / (index + 1)
        if index == len(dates) - 1:
            np.testing.assert_allclose(tower, ref_tower, rtol=1e-12, atol=1e-14)
            reference_iv = iv
        if (index + 1) % 50 == 0 or index == len(dates) - 1:
            elapsed = time.monotonic() - started
            print(json.dumps(dict(stage="historical-IV", completed=index + 1, total=len(dates),
                                  elapsed_seconds=elapsed, remaining_seconds=elapsed * (len(dates) / (index + 1) - 1))), flush=True)
    assert reference_iv is not None
    responses = []
    strikes = nms_strikes(atm_vol[-1], nms)
    for factor in range(loadings.shape[-1]):
        shock = loadings[..., factor] / np.sqrt(DAYS_PER_YEAR)
        shocked_ivs = []
        for sign in (1, -1):
            dlvsq = ref_dlvsq.copy()
            dlvsq[1:] *= np.exp(sign * shock)
            tower = np.asarray(decode(jnp.asarray(ref_support), jnp.asarray(dlvsq)))
            shocked_ivs.append(model_iv_surface(ref_support, tower, strikes))
        responses.append(5000 * (shocked_ivs[0] - shocked_ivs[1]))
    responses = np.asarray(responses)
    np.testing.assert_allclose(responses[:, 0], 0, atol=1e-10)
    arrays_path = output_directory / "factor_surfaces.npz"
    np.savez_compressed(arrays_path, nms=nms, expiry_days=np.arange(1, 26), mean_iv=mean_iv,
                        reference_iv=reference_iv, factor_responses_bp=responses, variance_fractions=fractions,
                        reference_strikes=strikes)
    rank90 = int(np.flatnonzero(np.cumsum(fractions) >= 0.90)[0]) + 1
    report = dict(schema="gromm-main21-wide-factors-v1", model_label="Gromm-v1-4,3",
                  training_run=training_run.name, reference_date=str(dates[-1]),
                  mean_dates=[str(dates[0]), str(dates[-1])], mean_date_count=len(dates),
                  diffusion_pair_count=pair_count, diffusion_end=audit["calibration"]["last_endpoint"],
                  retained_rank=8, retained_variance_fraction=float(fractions[:8].sum()),
                  first_rank_above_90_percent=rank90, individual_variance_fractions=fractions.tolist(),
                  mean_iv_range_percent=[float(100 * mean_iv.min()), float(100 * mean_iv.max())],
                  reference_iv_range_percent=[float(100 * reference_iv.min()), float(100 * reference_iv.max())],
                  factor_max_absolute_iv_response_bp=np.max(np.abs(responses), axis=(1, 2)).tolist(),
                  conventions=dict(mean="Equal-date arithmetic mean of Black IV, not IV of mean prices",
                                   coordinates="NMS [-4,3] using each date's market ATM scale from raw_initial; 1..25 business days",
                                   responses="Half difference at +/- annualized loading / sqrt(252), in IV bp; future DLV only; front row and spot fixed",
                                   spectrum="Spot loading power plus residual PCA eigenvalues of smoothed AR innovations; not IV-level PCA"),
                  input_sha256=digests, source_code_sha256=sha256(Path(__file__)),
                  plot_arrays_sha256=sha256(arrays_path))
    (output_directory / "factor_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output_directory / "factor_numbers.tex").write_text(
        rf"\newcommand{{\WidePcaNinetyRank}}{{{rank90}}}" + "\n")
    plot_surfaces(nms, mean_iv, reference_iv, responses, fractions, output_directory)
    plot_spectrum(fractions, output_directory)
    if publish_directory is not None:
        publish_directory.mkdir(parents=True, exist_ok=True)
        for artifact in output_directory.iterdir():
            shutil.copy2(artifact, publish_directory / artifact.name)
    print(json.dumps(dict(stage="complete", output=str(output_directory), rank90=rank90,
                          retained_variance_fraction=report["retained_variance_fraction"])), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-run", "audit-directory", "output-directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--publish-directory", type=Path)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
