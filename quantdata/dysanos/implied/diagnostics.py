"""Historical, invariant, and aggregate PCA benchmark diagnostics."""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any

import numpy as np
from cdxcore.err import verify_inp

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.grid.gendysanos.run_surface_first_diagnostics import (
    _distribution_metrics,
    _weekly_path_statistics,
)

from .decoder import PCATotalVarianceDecoder, WingHitDiagnostics
from .simulator import ImpliedPCASampler
from .static_arbitrage import StaticArbitrageAccumulator

DATA_QUALIFICATION = (
    "GridMarketDefinition.load() first fits a smooth statically arbitrage-free "
    "SANOS surface to each date and evaluates it on the regular grid. This "
    "comparison tests the SANOS generative decoder against direct PCA when both "
    "use the same cleaned, SANOS-interpolated mid_vols observations; it does not "
    "test SANOS preprocessing against raw OptionMetrics interpolation."
)

DYNAMIC_ARBITRAGE_QUALIFICATION = (
    "Static surface arbitrage and sampled fixed-contract return diagnostics are "
    "distinct. Failure to detect a one-sided opportunity in the sampled "
    "contracts does not prove absence of arbitrary dynamic arbitrage."
)


def invariant_surface_diagnostics(
    sampler: ImpliedPCASampler,
    *,
    num_samples: int = 100_000,
    seed_reset: bool = True,
) -> dict[str, Any]:
    """Compare invariant Gaussian draws with historical PCA surface states."""

    verify_inp(
        isinstance(num_samples, int) and num_samples > 0,
        lambda: f"'num_samples' must be positive, got {num_samples!r}",
    )
    rng_state = copy.deepcopy(sampler.rng.bit_generator.state)
    try:
        if seed_reset:
            sampler.reset_rng()
        mean, covariance = sampler.invariant_surface_distribution()
        samples = sampler.sample_invariant_surface_states(num_samples)
    finally:
        # Diagnostics must not change the logically seeded path universe.
        sampler.rng.bit_generator.state = rng_state
    metrics = _distribution_metrics(
        sampler.xstate[:, 1:].astype(np.float64, copy=False),
        samples.astype(np.float64, copy=False),
        rng=np.random.default_rng(sampler.config.seed + 400_003),
    )
    return {
        "sample_count": num_samples,
        "fitted_invariant_mean": mean.tolist(),
        "fitted_invariant_covariance": covariance.tolist(),
        "historical_comparison": metrics,
        "surface_generator_eigenvalues": [
            {"real": float(value.real), "imaginary": float(value.imag)}
            for value in np.linalg.eigvals(sampler.pcaxar1.k[1:, 1:])
        ],
    }


def weekly_path_diagnostics(
    sampler: ImpliedPCASampler,
    *,
    num_samples: int = 100_000,
    seed_reset: bool = True,
) -> dict[str, Any]:
    """Compare exact simulated five-day increments with historical ones."""

    verify_inp(
        isinstance(num_samples, int) and num_samples > 0,
        lambda: f"'num_samples' must be positive, got {num_samples!r}",
    )
    verify_inp(
        sampler.xstate.shape[0] > 5,
        "weekly diagnostics require at least six historical observations",
    )
    rng_state = copy.deepcopy(sampler.rng.bit_generator.state)
    try:
        if seed_reset:
            sampler.reset_rng()
        paths = sampler.free_sample_xstates(
            num_steps=6,
            num_samples=num_samples,
            dt=1.0 / DAYS_PER_YEAR,
            bandwidth=1.0,
        )
    finally:
        sampler.rng.bit_generator.state = rng_state
    historical_change = sampler.xstate[5:] - sampler.xstate[:-5]
    simulated_change = paths[:, -1] - paths[:, 0]
    return {
        "sample_count": num_samples,
        "weekly_increment_vs_historical_weekly_increment": (
            _distribution_metrics(
                historical_change,
                simulated_change,
                rng=np.random.default_rng(sampler.config.seed + 400_009),
            )
        ),
        "historical_path_statistics": _weekly_path_statistics(
            np.stack((sampler.xstate[:-5], sampler.xstate[5:]), axis=1)
        ),
        "simulated_path_statistics": _weekly_path_statistics(paths),
        "independent_path_interpretation": (
            "each simulated row is one independently initialized and driven "
            "six-state daily path"
        ),
    }


def historical_static_arbitrage_diagnostics(
    decoder: PCATotalVarianceDecoder,
    scores: np.ndarray,
    *,
    strikes: np.ndarray,
    roundoff_multipliers: tuple[int, ...] = (0, 1, 8, 32, 128),
    block_size: int = 512,
) -> tuple[dict[str, Any], WingHitDiagnostics]:
    """Check every historical PCA reconstruction without any repair."""

    scores = np.asarray(scores)
    verify_inp(
        scores.ndim == 2
        and scores.shape[1] == decoder.pca.num_components
        and scores.shape[0] > 0,
        lambda: f"invalid historical score matrix {scores.shape}",
    )
    strikes = np.asarray(strikes, dtype=np.float64)
    accumulator = StaticArbitrageAccumulator(
        strikes=strikes,
        expiries=decoder.pca.expiries,
        roundoff_multipliers=roundoff_multipliers,
    )
    wing_hits = WingHitDiagnostics(queries=0, left_hits=0, right_hits=0)
    for start in range(0, scores.shape[0], block_size):
        block = scores[start : start + block_size]
        calls, hits = decoder.normalized_call_prices(
            block,
            decoder.pca.expiries[:, None],
            strikes[None, :],
        )
        nms = decoder.nms_from_relative_strikes(
            block,
            decoder.pca.expiries[:, None],
            strikes[None, :],
        )
        accumulator.update(calls, normalized_moneyness=nms)
        wing_hits = wing_hits + hits
    return accumulator.report(), wing_hits


def sampler_fit_report(sampler: ImpliedPCASampler) -> dict[str, Any]:
    """Serialize the shared surface-first fit without changing its semantics."""

    fit = sampler.pcaxar1
    quality = fit.fit_quality
    assert quality is not None
    return {
        "surface_first": fit.surface_first,
        "num_surface_factors": fit.num_surface_alpha,
        "surface_innovation_variance_explained": fit.surface_var_explained,
        "spot_innovation_variance_explained_by_surface_factors": (
            fit.spot_surface_r2
        ),
        "orthogonal_spot_residual_std": fit.spot_residual_std,
        "mean_historical_dt": fit.mean_dt,
        "generator": fit.k.tolist(),
        "affine_drift": fit.U0.tolist(),
        "innovation_loadings": fit.dU.tolist(),
        "fit_quality": {
            "r2": quality.r2,
            "r2_by_state": quality.r2_by_state.tolist(),
            "rmse": quality.rmse,
            "rmse_by_state": quality.rmse_by_state.tolist(),
            "sample_mean": quality.sample_mean.tolist(),
            "sample_std": quality.sample_std.tolist(),
            "fitted_mean": quality.fitted_mean.tolist(),
            "mean_error": quality.mean_error.tolist(),
            "summary": quality.summary_statistics(),
        },
        "configuration": asdict(sampler.config),
    }


__all__ = [
    "DATA_QUALIFICATION",
    "DYNAMIC_ARBITRAGE_QUALIFICATION",
    "historical_static_arbitrage_diagnostics",
    "invariant_surface_diagnostics",
    "sampler_fit_report",
    "weekly_path_diagnostics",
]
