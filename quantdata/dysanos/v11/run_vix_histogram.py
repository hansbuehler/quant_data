"""Compare official historic VIX with stationary daily DYSANOS samples.

The simulated index uses the as-of-date listed strike strips, repriced by
SANOS, with actual-calendar interpolation to 30 days. This holds liquidity
and the expiry calendar fixed; it is not a simulation of Cboe's future book.
The fitted transition is daily: no weekly-to-daily conversion is applied.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Literal

import matplotlib
import numpy as np
import polars as pl
import torch
from scipy.linalg import solve_discrete_lyapunov, solve_triangular

from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.presentation_hedging import _decode_surface, load_presentation_model
from quantdata.dysanos.flex.stochastic_volatility_diffusion import (
    LogVarianceDiffusionConfig,
    fit_log_variance_diffusion,
)
from quantdata.dysanos.v11.daily_joint_dynamics import _historical_spot_scores
from quantdata.dysanos.v11.run_rolling_direct_ar import (
    _historical_sanos_directory,
    historical_sanos_cache,
)
from quantdata.dysanos.v11.tail_copula import normal_score_from_scale_mixture
from quantdata.vixdata import load_cboe_vix
from quantdata.vixdata.formula import select_vix_strip

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def stationary_state_law(
    *, transition: np.ndarray, equilibrium: np.ndarray,
    residual_factor: np.ndarray, spot_loading: np.ndarray, spot_score_mean: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Stationary law of row states m+(h-m)A+beta(z-zbar)+L epsilon."""
    dimension = len(equilibrium)
    if (
        transition.shape != (dimension, dimension)
        or residual_factor.shape != (dimension, dimension)
        or spot_loading.shape != (dimension,)
        or not all(np.all(np.isfinite(v)) for v in (
            transition, equilibrium, residual_factor, spot_loading, spot_score_mean
        ))
        or np.max(np.abs(np.linalg.eigvals(transition))) >= 1.0
    ):
        raise ValueError("A finite, stable, shape-consistent daily law is required")
    innovation_covariance = residual_factor @ residual_factor.T + np.outer(spot_loading, spot_loading)
    covariance = solve_discrete_lyapunov(transition.T, innovation_covariance)
    covariance = (covariance + covariance.T) / 2
    np.linalg.cholesky(covariance)  # Reject non-positive laws rather than clipping them.
    mean = equilibrium + np.linalg.solve(
        np.eye(dimension) - transition.T, -spot_score_mean * spot_loading
    )
    return mean, covariance


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1)),
        **{f"p{100 * q:g}": float(np.quantile(values, q)) for q in (.01, .05, .5, .95, .99)},
        "minimum": float(np.min(values)), "maximum": float(np.max(values)),
        "fraction_above_30": float(np.mean(values > 30)),
    }


def sample_tail_states(
    *, transition: np.ndarray, equilibrium: np.ndarray, residual_factor: np.ndarray,
    spot_loading: np.ndarray, spot_score_mean: float, config: LogVarianceDiffusionConfig,
    samples: int, burn_in: int, seed: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Sample the persistent tail law, retaining conditional front-spot PIT.

    Each pair shares its log-variance path and has opposite Gaussian shocks.
    The OU starts stationarily, while surface states start at their stationary
    mean and are burned in. A single scale applied to a stationary Gaussian
    state would not reproduce this persistent AR/OU law.
    """
    if samples < 2 or samples % 2 or burn_in < 1:
        raise ValueError("Need an even sample count and positive burn-in")
    mean, _ = stationary_state_law(
        transition=transition, equilibrium=equilibrium, residual_factor=residual_factor,
        spot_loading=spot_loading, spot_score_mean=spot_score_mean,
    )
    contraction = float(np.linalg.norm(np.linalg.matrix_power(transition, burn_in), ord=2))
    if contraction > 1e-8:
        raise ValueError(f"Burn-in leaves excessive initial-state dependence: {contraction:g}")
    rng = np.random.default_rng(seed)
    half = samples // 2
    dimension = len(mean)
    deviations = np.zeros((half, dimension))
    latent = rng.normal(scale=np.sqrt(config.stationary_log_variance), size=half)
    innovation_std = config.log_variance_innovation_standard_deviation
    conditional_variance = innovation_std**2
    started = perf_counter()
    for step in range(burn_in):
        conditional_mean = config.persistence * latent
        latent = conditional_mean + innovation_std * rng.standard_normal(half)
        multiplier = np.sqrt(config.unconditional_scale * np.exp(
            latent - .5 * config.stationary_log_variance
        ))
        spot_score = normal_score_from_scale_mixture(
            multiplier * rng.standard_normal(half),
            stationary_log_variance=config.stationary_log_variance,
            unconditional_scale=config.unconditional_scale,
            conditional_log_variance_mean=conditional_mean,
            conditional_log_variance_variance=conditional_variance,
        )
        shock = (rng.standard_normal((half, dimension)) @ residual_factor.T) * multiplier[:, None]
        deviations = deviations @ transition + spot_score[:, None] * spot_loading + shock
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == burn_in:
            elapsed = perf_counter() - started
            print(f"tail burn-in={step + 1}/{burn_in} paths={samples} elapsed={elapsed:.1f}s "
                  f"remaining={elapsed * (burn_in - step - 1) / (step + 1):.1f}s", flush=True)
    states = mean + np.concatenate((deviations, -deviations))
    if not np.all(np.isfinite(states)):
        raise RuntimeError("Nonfinite Tail Diffusion states")
    return states, {"burn_in_sessions": burn_in, "initial_state_contraction_norm": contraction}


def run(
    *, run_directory: Path, output: Path, samples: int, batch_size: int, seed: int,
    innovation_law: Literal["normal", "tail"] = "normal", burn_in: int = 256,
    gaussian_comparison: Path | None = None,
) -> dict:
    if samples < 2 or samples % 2 or batch_size < 1:
        raise ValueError("Need an even sample count >=2 and a positive batch size")
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((run_directory / "reports/run-manifest.json").read_text())
    dates = tuple(date.fromisoformat(v) for v in manifest["training_dates"])
    as_of = date.fromisoformat(manifest["as_of"])
    checkpoint = run_directory / "checkpoints/joint-direct-200.pt"
    affine_checkpoint = run_directory / "checkpoints/direct-200.pt"
    _, model = load_presentation_model(checkpoint, device="cpu", precision=64)
    directory = _historical_sanos_directory(ticker="SPX", dates=dates)
    cache_root = Path(historical_sanos_cache.path)
    spots = []
    market = None
    for index, trading_date in enumerate(dates):
        payload = historical_sanos_cache.read(
            str((directory / str(trading_date)).relative_to(cache_root)),
            delete_wrong_version=False, raise_on_error=True,
        )
        if payload["d"] != trading_date:
            raise RuntimeError("Cached market date does not match the requested date")
        spots.append(float(payload["spot_ref"]))
        if trading_date == as_of:
            market = payload["market"]
        if (index + 1) % 50 == 0:
            print(f"historic inputs={index + 1}/{len(dates)}", flush=True)
    if market is None:
        raise ValueError("The as-of date must belong to the training panel")

    # Training fixed these spot scores at affine initialization. Recreate that
    # decoder, not the subsequently modified joint decoder, to recover beta.
    _, initial_decoder = load_presentation_model(checkpoint, device="cpu", precision=32)
    source = torch.load(affine_checkpoint, map_location="cpu", weights_only=False)
    expected_digest = torch.load(checkpoint, map_location="cpu", weights_only=False)["source_decoder_sha256"]
    if checkpoint_digest(affine_checkpoint) != expected_digest:
        raise ValueError("The affine initialization checkpoint has changed")
    initial_decoder.load_state_dict(source["model_state_dict"], strict=False)
    scores = _historical_spot_scores(
        initial_decoder, initial_decoder.daily_states.detach().numpy()[:-1],
        np.asarray(spots[1:]) / spots[:-1], batch_size=32,
    )
    with torch.inference_mode():
        states = model.daily_states.numpy()
        transition = model.weekly_transition().numpy()  # Legacy name; this fit uses daily pairs.
        equilibrium = model.equilibrium_mean.numpy()
        factor = model.innovation_cholesky().numpy()
    residuals = states[1:] - (equilibrium + (states[:-1] - equilibrium) @ transition)
    centered = scores - scores.mean()
    beta = (residuals * centered[:, None]).sum(axis=0) / (centered @ centered + 1e-6)
    mean, covariance = stationary_state_law(
        transition=transition, equilibrium=equilibrium, residual_factor=factor,
        spot_loading=beta, spot_score_mean=float(scores.mean()),
    )
    tail_report = None
    sampling_diagnostics = {}
    if innovation_law == "tail":
        white = solve_triangular(factor, (residuals - centered[:, None] * beta).T, lower=True).T
        radius = np.mean(white**2, axis=1)
        starts = np.zeros(len(radius), dtype=bool)
        starts[0] = True
        fit = fit_log_variance_diffusion(radius, starts, dimension=white.shape[1])
        tail_report = {
            "schema": "dysanos-v11-rolling-daily-tail-diffusion-v1",
            "status": "complete" if fit.success else "optimizer_warning",
            "checkpoint_sha256": checkpoint_digest(checkpoint),
            "training_range": [str(dates[0]), str(dates[-1])],
            "daily_residual_count": len(radius), "innovation_dimension": white.shape[1],
            "estimation": "Post-fit tail law; original decoder, AR and spot loading remain frozen",
            "fit": fit.to_dict(step_sessions=1),
        }
        write_json_atomic(tail_report, output / "tail_fit.json")
        np.savez_compressed(output / "tail_residuals.npz", whitened=white, radius=radius)
        print(f"tail fit={tail_report['status']} config={fit.config}", flush=True)
        sampled_states, sampling_diagnostics = sample_tail_states(
            transition=transition, equilibrium=equilibrium, residual_factor=factor,
            spot_loading=beta, spot_score_mean=float(scores.mean()), config=fit.config,
            samples=samples, burn_in=burn_in, seed=seed,
        )
        # The fitted scale is allowed to differ from one, as in the existing
        # Tail Diffusion. Record this covariance rescaling, not just kurtosis.
        mean, covariance = stationary_state_law(
            transition=transition, equilibrium=equilibrium,
            residual_factor=factor * np.sqrt(fit.config.unconditional_scale),
            spot_loading=beta, spot_score_mean=float(scores.mean()),
        )
    elif innovation_law == "normal":
        rng = np.random.default_rng(seed)
        normals = rng.standard_normal((samples // 2, len(mean)))
        sampled_states = mean + np.concatenate((normals, -normals)) @ np.linalg.cholesky(covariance).T
    else:
        raise ValueError(f"Unknown innovation law {innovation_law!r}")
    np.savez_compressed(output / "sampled_states.npz", states=sampled_states, mean=mean,
                        covariance=covariance, spot_loading=beta, spot_score_mean=scores.mean())

    # Use eligible Friday expiries at 23--37 actual calendar days. Model
    # valuation still uses each expiry's cached business-year coordinate.
    eligible = []
    for j, (expiry, is_am) in enumerate(zip(market["expiry_dates"], market["is_am"], strict=True)):
        expiry = date.fromisoformat(str(expiry))
        days = (expiry - as_of).days
        if expiry.weekday() == 4 and 23 < days < 37 and not is_am:
            eligible.append((days, j, expiry))
    near = max((x for x in eligible if x[0] <= 30), default=None)
    far = min((x for x in eligible if x[0] > 30), default=None)
    if near is None or far is None:
        raise ValueError("The cached book lacks two eligible PM expiries bracketing 30 days")
    terms = []
    for days, j, expiry in (near, far):
        strip = select_vix_strip(
            strikes=market["strikes"][j], call_bids=market["call_bids"][j],
            call_asks=market["call_asks"][j], put_bids=market["put_bids"][j],
            put_asks=market["put_asks"][j], discount_factor=1.0,
        )
        # Market prices are already normalized by discounted forward; the
        # model forward is exactly one. Check empirical parity selects its K0.
        if strip.k0 != np.max(strip.strikes[strip.strikes <= 1]):
            raise ValueError("Market and model forward select different K0 contracts")
        terms.append((days, float(market["expiries"][j]), strip))
    print(f"pricing stationary states={samples}, calendar expiries={near[0]}/{far[0]} days", flush=True)
    all_strikes = np.concatenate([term[2].strikes for term in terms])
    all_expiries = np.concatenate([np.full(len(term[2].strikes), term[1]) for term in terms])
    strike_tensor = torch.as_tensor(all_strikes, dtype=model.fdtype)
    expiry_tensor = torch.as_tensor(all_expiries, dtype=model.fdtype)
    vix = np.empty(samples)
    fitted_vix = np.empty(len(states))
    all_states = np.concatenate((sampled_states, states))
    results = np.empty(len(all_states))
    cap_count = 0
    started = perf_counter()
    with torch.inference_mode():
        for first in range(0, len(all_states), batch_size):
            batch = torch.as_tensor(all_states[first:first + batch_size], dtype=model.fdtype)
            calls = model.sanos_decoder.price_calls_batch(
                _decode_surface(model, batch),
                expiries_y=expiry_tensor[None].expand(len(batch), -1),
                strikes=strike_tensor[None].expand(len(batch), -1),
            ).numpy()
            total_variances = []
            offset = 0
            for _, _, strip in terms:
                k = strip.strikes
                c = calls[:, offset:offset + len(k)]
                puts = c + k - 1
                if np.min(puts) < -1e-9 or np.min(c) < -1e-9:
                    raise RuntimeError("Negative SANOS option price")
                q = np.where(k < strip.k0, np.maximum(puts, 0), np.maximum(c, 0))
                q[:, k == strip.k0] = (c[:, k == strip.k0] + puts[:, k == strip.k0]) / 2
                tv = 2 * (q @ (strip.delta_strikes / k**2)) - (1 / strip.k0 - 1)**2
                if not np.all(np.isfinite(tv)) or np.min(tv) < 0:
                    raise RuntimeError("Invalid sampled VIX total variance")
                total_variances.append(tv)
                offset += len(k)
            alpha = (30 - near[0]) / (far[0] - near[0])
            values = 100 * np.sqrt(((1 - alpha) * total_variances[0] + alpha * total_variances[1]) * 365 / 30)
            last = first + len(batch)
            results[first:last] = values
            if first < samples:
                raw = model.raw_states_from_states(batch[:min(len(batch), samples - first)])
                dlvs = raw[:, :model.config.geometry.num_model_expiries * model.config.geometry.num_inner_strikes]
                cap_count += int((dlvs >= model.config.geometry.maximum_log_dlvsq).sum())
            if first == 0 or last % (batch_size * 20) == 0 or last == len(all_states):
                elapsed = perf_counter() - started
                print(f"VIX priced={last}/{len(all_states)} elapsed={elapsed:.1f}s remaining={elapsed * (len(all_states) - last) / last:.1f}s", flush=True)
    vix[:] = results[:samples]
    fitted_vix[:] = results[samples:]
    historic = load_cboe_vix(start=dates[0], end=dates[-1]).filter(pl.col("date").is_in(dates)).sort("date")
    if historic["date"].to_list() != list(dates):
        raise ValueError("Official historic VIX does not cover every fitted market date")
    historic.write_parquet(output / "historic_vix.parquet")
    pl.DataFrame({"sample": np.arange(samples), "vix": vix}).write_parquet(output / "sampled_vix.parquet")
    pl.DataFrame({"date": dates, "vix_on_reference_strips": fitted_vix}).write_parquet(output / "fitted_vix.parquet")
    historic_values = historic["close"].to_numpy()
    law_label = "Gaussian" if innovation_law == "normal" else "Tail Diffusion"
    nms = list(model.config.geometry.nms_range)
    summary = {
        "status": "complete", "model": f"daily {law_label} joint AR(1), affine-200, NMS {nms}",
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_digest(checkpoint),
        "training_start": str(dates[0]), "training_end": str(dates[-1]),
        "historic_observations": len(historic), "sampled_states": samples,
        "sampling": ("stationary Gaussian law" if innovation_law == "normal" else "persistent OU log-variance simulation with burn-in and conditional front-spot PIT") + ", antithetic draws, including profiled spot loading",
        "sampling_diagnostics": sampling_diagnostics, "tail_fit": tail_report,
        "independent_antithetic_pairs": samples // 2, "seed": seed,
        "vix_definition": "SANOS-repriced reference-date market strips, PM Friday expiries; calendar-day 30-day total-variance interpolation, ACT/365; normalized forward=1",
        "limitation": "Calendar and available market strikes are held fixed at the as-of date; historical official VIX uses each actual day's book. No simulated future liquidity or minute-by-minute clock.",
        "expiry_strips": [{"expiry": str(expiry), "calendar_days": days,
                           "business_years": term[1], "strikes": len(term[2].strikes),
                           "minimum_strike": float(term[2].strikes.min()), "maximum_strike": float(term[2].strikes.max())}
                          for (days, _, expiry), term in zip((near, far), terms, strict=True)],
        "historic_vix": _summary(historic_values), "sampled_vix": _summary(vix),
        "fitted_state_vix_on_reference_strips": _summary(fitted_vix),
        "sampled_dlv_fraction_at_cap": cap_count / (samples * model.config.geometry.num_model_expiries * model.config.geometry.num_inner_strikes),
    }
    gaussian_vix = None
    if gaussian_comparison is not None:
        reference = json.loads((gaussian_comparison / "summary.json").read_text())
        if any(reference[k] != summary[k] for k in ("checkpoint_sha256", "training_start", "training_end", "expiry_strips")):
            raise ValueError("Gaussian comparison uses a different model, period or VIX book")
        gaussian_vix = pl.read_parquet(gaussian_comparison / "sampled_vix.parquet")["vix"].to_numpy()
        summary["gaussian_comparison"] = {"source": str(gaussian_comparison), "sampled_vix": _summary(gaussian_vix)}
    write_json_atomic(summary, output / "summary.json")
    plotted = [historic_values, vix] + ([] if gaussian_vix is None else [gaussian_vix])
    bins = np.linspace(min(x.min() for x in plotted), max(x.max() for x in plotted), 41)
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    ax.hist(historic_values, bins=bins, density=True, alpha=.4, color="#348abd",
            label=f"Historical Cboe VIX ({len(historic_values)} closes)", edgecolor="white", linewidth=.4)
    ax.hist(vix, bins=bins, density=True, histtype="step", linewidth=1.8, color="#d95f02",
            label=f"DYSANOS {law_label} ({samples:,} samples)")
    if gaussian_vix is not None:
        ax.hist(gaussian_vix, bins=bins, density=True, histtype="step", linewidth=1.5,
                linestyle="--", color="#7a5195", label="DYSANOS Gaussian (previous run)")
    ax.set(xlabel="VIX level (annualized volatility %)", ylabel="Probability density",
           title="Historical and sampled DYSANOS VIX")
    ax.legend(frameon=False, fontsize=9)
    ax.text(.99, .97, f"Daily AR(1), affine-200; NMS [{nms[0]:g}, {nms[1]:g}]\nFit: {dates[0]}–{dates[-1]}\nReference-date VIX strips held fixed",
            ha="right", va="top", transform=ax.transAxes, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "vix_histogram.png", dpi=180)
    fig.savefig(output / "vix_histogram.pdf")
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--innovation-law", choices=("normal", "tail"), default="normal")
    parser.add_argument("--burn-in", type=int, default=256)
    parser.add_argument("--gaussian-comparison", type=Path)
    args = vars(parser.parse_args())
    torch.set_num_threads(args.pop("threads"))
    print(json.dumps(run(**args), indent=2), flush=True)


if __name__ == "__main__":
    main()
