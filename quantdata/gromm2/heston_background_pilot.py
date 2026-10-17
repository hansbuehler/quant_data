"""One-date GROMM2 calibration with an independent Heston background."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from cdxcore.bs import bs
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.special import ndtr

from quantdata.dysanos.bates_crc.heston_riccati import (
    GaussianJumpParams,
    HestonParams,
    riccati_A_closed_form,
    riccati_D_closed_form,
)
from quantdata.dysanos.bates_crc.real_data_study.fast_bates_pricer import (
    price_calls_ordinary_bates,
)
from quantdata.gromm2.calibration import _black_call_kernels
from quantdata.gromm2.dlv import decode_density_tower
from quantdata.gromm2.geometry import Gromm2GeometryConfig, project_total_variance
from quantdata.gromm2.market import (
    PublicSanosDay,
    load_public_sanos_days,
    public_sanos_atm_total_variances,
    public_sanos_prices,
)

FloatArray = NDArray[np.float64]

_FOURIER_ALPHA = 1.5
_FFT_SIZE = 32_768
_FFT_FREQUENCY_SPACING = 0.50
_MINIMUM_DLVSQ = 0.01**2
_MAXIMUM_DLVSQ = 3.0**2


@dataclass(frozen=True, kw_only=True)
class HestonBackground:
    """Risk-neutral Heston kernel parameters in forward-one units."""

    mean_reversion: float
    long_run_variance: float
    volatility_of_variance: float
    correlation: float
    initial_variance: float

    def heston(self) -> HestonParams:
        return HestonParams(
            kappa=self.mean_reversion,
            theta=self.long_run_variance,
            xi=self.volatility_of_variance,
            rho=self.correlation,
        )


@dataclass(frozen=True, kw_only=True)
class OneDateFit:
    """One directly optimized DLV tower and its fit diagnostics."""

    dlvsq: FloatArray
    densities: FloatArray
    prices: FloatArray
    completed_epochs: int
    stop_reason: str
    objective: float
    quote_loss: float
    cap_penalty: float
    percent_inside_spread: float
    midpoint_mae_half_spreads: float
    mean_excess_half_spreads: float
    maximum_excess_half_spreads: float
    strike_roughness: float
    expiry_roughness: float
    floored_fraction: float
    capped_fraction: float


def _black_calls(strikes: FloatArray, expiries: FloatArray, volatility: FloatArray) -> FloatArray:
    standard_deviation = volatility * np.sqrt(expiries)
    d1 = -np.log(strikes) / standard_deviation + 0.5 * standard_deviation
    d2 = d1 - standard_deviation
    return ndtr(d1) - strikes * ndtr(d2)


def _heston_calls_one_expiry(
    background: HestonBackground,
    *,
    strikes: FloatArray,
    expiry_y: float,
    quadrature_order: int,
    frequency_limit: float,
    chunk_size: int = 2_048,
) -> FloatArray:
    """Price forward-one Heston calls with one cached Fourier transform."""
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    half = 0.5 * frequency_limit
    frequencies = half * (nodes + 1.0)
    quadrature_weights = half * weights
    shifted = frequencies - 1j * (_FOURIER_ALPHA + 1.0)
    heston = background.heston()
    transform = np.exp(
        background.initial_variance
        * riccati_D_closed_form(shifted, expiry_y, heston)
        + riccati_A_closed_form(shifted, expiry_y, heston)
    )
    denominator = (
        _FOURIER_ALPHA**2
        + _FOURIER_ALPHA
        - frequencies**2
        + 1j * (2.0 * _FOURIER_ALPHA + 1.0) * frequencies
    )
    weighted_transform = quadrature_weights * transform / denominator
    result = np.empty_like(strikes)
    log_strikes = np.log(strikes)
    for start in range(0, len(strikes), chunk_size):
        stop = min(start + chunk_size, len(strikes))
        current = log_strikes[start:stop]
        integral = np.sum(
            np.exp(-1j * np.outer(frequencies, current))
            * weighted_transform[:, None],
            axis=0,
        )
        result[start:stop] = (
            np.exp(-_FOURIER_ALPHA * current) * integral.real / math.pi
        )
    intrinsic = np.maximum(1.0 - strikes, 0.0)
    result = np.clip(result, intrinsic, 1.0)
    order = np.argsort(strikes)
    sorted_result = result[order]
    sorted_result = np.maximum.accumulate(sorted_result[::-1])[::-1]
    result[order] = sorted_result
    return result


def _heston_call_grid_one_expiry_fft(
    background: HestonBackground,
    *,
    expiry_y: float,
    fft_size: int = _FFT_SIZE,
    frequency_spacing: float = _FFT_FREQUENCY_SPACING,
) -> tuple[FloatArray, FloatArray]:
    """Return the Carr--Madan log-strike grid and forward-one calls."""
    if fft_size <= 0 or fft_size & (fft_size - 1):
        raise ValueError("fft_size must be a positive power of two")
    if frequency_spacing <= 0.0:
        raise ValueError("frequency_spacing must be positive")
    frequencies = frequency_spacing * np.arange(fft_size, dtype=np.float64)
    log_strike_spacing = 2.0 * math.pi / (fft_size * frequency_spacing)
    log_strike_half_width = 0.5 * fft_size * log_strike_spacing
    shifted = frequencies - 1j * (_FOURIER_ALPHA + 1.0)
    heston = background.heston()
    transform = np.exp(
        background.initial_variance
        * riccati_D_closed_form(shifted, expiry_y, heston)
        + riccati_A_closed_form(shifted, expiry_y, heston)
    )
    denominator = (
        _FOURIER_ALPHA**2
        + _FOURIER_ALPHA
        - frequencies**2
        + 1j * (2.0 * _FOURIER_ALPHA + 1.0) * frequencies
    )
    trapezoid_weights = np.full(fft_size, frequency_spacing, dtype=np.float64)
    trapezoid_weights[0] *= 0.5
    fft_input = (
        np.exp(1j * log_strike_half_width * frequencies)
        * transform
        / denominator
        * trapezoid_weights
    )
    fft_values = np.fft.fft(fft_input).real
    log_strike_grid = (
        -log_strike_half_width
        + log_strike_spacing * np.arange(fft_size, dtype=np.float64)
    )
    call_grid = (
        np.exp(-_FOURIER_ALPHA * log_strike_grid) * fft_values / math.pi
    )
    return log_strike_grid, call_grid


def _heston_calls_one_expiry_fft(
    background: HestonBackground,
    *,
    strikes: FloatArray,
    expiry_y: float,
    fft_size: int = _FFT_SIZE,
    frequency_spacing: float = _FFT_FREQUENCY_SPACING,
) -> FloatArray:
    """Price forward-one Heston calls by Carr--Madan FFT interpolation."""
    log_strike_grid, call_grid = _heston_call_grid_one_expiry_fft(
        background,
        expiry_y=expiry_y,
        fft_size=fft_size,
        frequency_spacing=frequency_spacing,
    )
    log_strikes = np.log(strikes)
    if np.min(log_strikes) < log_strike_grid[0] or np.max(log_strikes) > log_strike_grid[-1]:
        raise ValueError("requested strikes exceed the Carr--Madan FFT grid")
    result = np.interp(log_strikes, log_strike_grid, call_grid)
    intrinsic = np.maximum(1.0 - strikes, 0.0)
    result = np.clip(result, intrinsic, 1.0)
    order = np.argsort(strikes)
    result[order] = np.maximum.accumulate(result[order][::-1])[::-1]
    return result


def heston_return_cdf_grid_fft(
    background: HestonBackground,
    *,
    expiry_y: float,
) -> tuple[FloatArray, FloatArray]:
    """Return a monotone FFT grid for the Heston gross-return CDF."""
    log_strikes, calls = _heston_call_grid_one_expiry_fft(
        background,
        expiry_y=expiry_y,
    )
    spacing = log_strikes[1] - log_strikes[0]
    log_derivative = np.empty_like(calls)
    log_derivative[2:-2] = (
        calls[:-4]
        - 8.0 * calls[1:-3]
        + 8.0 * calls[3:-1]
        - calls[4:]
    ) / (12.0 * spacing)
    boundary_derivative = np.gradient(calls, log_strikes, edge_order=2)
    log_derivative[:2] = boundary_derivative[:2]
    log_derivative[-2:] = boundary_derivative[-2:]
    retained = (log_strikes >= -2.0) & (log_strikes <= 2.0)
    log_strikes = log_strikes[retained]
    log_derivative = log_derivative[retained]
    strikes = np.exp(log_strikes)
    cumulative = np.clip(1.0 + log_derivative / strikes, 0.0, 1.0)
    cumulative = np.maximum.accumulate(cumulative)
    return log_strikes, cumulative


def heston_call_kernels(
    supports: FloatArray,
    strikes: FloatArray,
    expiries_y: FloatArray,
    *,
    background: HestonBackground,
) -> FloatArray:
    """Return Heston component-price matrices for every model expiry."""
    kernels = np.empty(
        (len(expiries_y), strikes.shape[1], supports.shape[1]),
        dtype=np.float64,
    )
    started = perf_counter()
    for expiry_index, expiry_y in enumerate(expiries_y):
        ratios = strikes[expiry_index, :, None] / supports[expiry_index, None, :]
        rounded_logs = np.round(np.log(ratios).reshape(-1), 13)
        unique_logs, inverse = np.unique(rounded_logs, return_inverse=True)
        unit_prices = _heston_calls_one_expiry_fft(
            background,
            strikes=np.exp(unique_logs),
            expiry_y=float(expiry_y),
        )[inverse].reshape(ratios.shape)
        kernels[expiry_index] = supports[expiry_index, None, :] * unit_prices
        elapsed = perf_counter() - started
        rate = (expiry_index + 1) / max(elapsed, 1.0e-9)
        eta = (len(expiries_y) - expiry_index - 1) / max(rate, 1.0e-9)
        print(
            f"heston-kernel expiry={expiry_index + 1}/{len(expiries_y)} "
            f"dte={252.0 * expiry_y:.0f} fft={_FFT_SIZE} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )
    return kernels


def fit_market_heston(
    day: PublicSanosDay,
    *,
    expiries_y: FloatArray,
    projected_total_variances: FloatArray,
    nms_points: int = 13,
) -> tuple[HestonBackground, dict[str, float]]:
    """Fit one Heston model to the date-t SANOS surface."""
    nms = np.linspace(-2.0, 1.0, nms_points)
    strikes = np.exp(np.sqrt(projected_total_variances)[:, None] * nms[None, :])
    expiry_grid = np.broadcast_to(expiries_y[:, None], strikes.shape)
    market_calls = public_sanos_prices(
        day,
        expiries_y=expiry_grid,
        strikes=strikes,
    )
    market_volatilities = np.asarray(
        bs.implied(
            k=strikes,
            prices=market_calls,
            sqrtT=np.sqrt(expiry_grid),
            eps=1.0e-10,
        ),
        dtype=np.float64,
    )
    total_standard_deviation = market_volatilities * np.sqrt(expiry_grid)
    d1 = -np.log(strikes) / total_standard_deviation + 0.5 * total_standard_deviation
    vegas = (
        np.exp(-0.5 * d1**2)
        / math.sqrt(2.0 * math.pi)
        * np.sqrt(expiry_grid)
    )
    def unpack(raw: FloatArray) -> HestonBackground:
        return HestonBackground(
            mean_reversion=float(np.exp(raw[0])),
            long_run_variance=float(np.exp(2.0 * raw[1])),
            volatility_of_variance=float(np.exp(raw[2])),
            correlation=float(np.tanh(raw[3])),
            initial_variance=float(np.exp(2.0 * raw[4])),
        )

    evaluations = 0
    no_jumps = GaussianJumpParams(lambda_J=0.0, m_J=0.0, delta_J=0.1)

    def residual(raw: FloatArray) -> FloatArray:
        nonlocal evaluations
        background = unpack(raw)
        prices = price_calls_ordinary_bates(
            background.heston(),
            no_jumps,
            background.initial_variance,
            strikes.reshape(-1),
            expiry_grid.reshape(-1),
            n_quad=300,
            u_max=1_200.0,
        ).reshape(market_calls.shape)
        evaluations += 1
        errors = (prices - market_calls) / np.maximum(vegas, 1.0e-5)
        if evaluations == 1 or evaluations % 10 == 0:
            print(
                f"market-heston evaluation={evaluations} "
                f"vol-rmse={np.sqrt(np.mean(errors**2)):.6f}",
                flush=True,
            )
        return errors.reshape(-1)

    short_volatility = math.sqrt(projected_total_variances[0] / expiries_y[0])
    long_volatility = math.sqrt(projected_total_variances[-1] / expiries_y[-1])
    initial = np.asarray([
        math.log(5.0),
        math.log(long_volatility),
        math.log(0.50),
        np.arctanh(-0.60),
        math.log(short_volatility),
    ])
    lower = np.asarray(
        [
            math.log(0.10),
            math.log(0.02),
            math.log(0.001),
            np.arctanh(-0.99),
            math.log(0.02),
        ],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            math.log(30.0),
            math.log(0.50),
            math.log(3.0),
            np.arctanh(0.50),
            math.log(0.50),
        ],
        dtype=np.float64,
    )
    fit = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        max_nfev=100,
        ftol=1.0e-9,
        xtol=1.0e-9,
        gtol=1.0e-9,
    )
    background = unpack(fit.x)
    errors = residual(fit.x)
    return background, {
        "converged": bool(fit.success),
        "function_evaluations": int(fit.nfev),
        "market_implied_volatility_rmse": float(np.sqrt(np.mean(errors**2))),
        "market_implied_volatility_max_abs": float(np.max(np.abs(errors))),
    }


def scale_heston_background(
    background: HestonBackground,
    *,
    volatility_scale: float,
) -> HestonBackground:
    """Scale the Heston volatility level without changing its time scale."""
    return HestonBackground(
        mean_reversion=background.mean_reversion,
        long_run_variance=background.long_run_variance * volatility_scale**2,
        volatility_of_variance=background.volatility_of_variance * volatility_scale,
        correlation=background.correlation,
        initial_variance=background.initial_variance * volatility_scale**2,
    )


def _roughness(log_dlvsq: FloatArray) -> tuple[float, float]:
    strike = float(np.sqrt(np.mean(np.diff(log_dlvsq, n=2, axis=1) ** 2)))
    expiry = float(np.sqrt(np.mean(np.diff(log_dlvsq, n=2, axis=0) ** 2)))
    return strike, expiry


def fit_one_date_dlv(
    *,
    supports: FloatArray,
    transports: FloatArray,
    expiries_y: FloatArray,
    kernels: FloatArray,
    strikes: FloatArray,
    mids: FloatArray,
    half_spreads: FloatArray,
    mask: NDArray[np.bool_],
    initial_dlvsq: FloatArray,
    maximum_epochs: int = 4_000,
    minimum_epochs: int = 1_000,
    report_every: int = 100,
) -> OneDateFit:
    """Directly fit one DLV tower, stopping only after objective convergence."""
    arrays = {
        name: jnp.asarray(value, dtype=jnp.float32)
        for name, value in (
            ("supports", supports),
            ("transports", transports),
            ("expiries", expiries_y),
            ("kernels", kernels),
            ("strikes", strikes),
            ("mids", mids),
            ("half_spreads", half_spreads),
        )
    }
    valid = jnp.asarray(mask, dtype=jnp.float32)
    intervals = jnp.diff(arrays["expiries"], prepend=0.0)
    maximum_raw = math.log(_MAXIMUM_DLVSQ - _MINIMUM_DLVSQ)
    initial = np.log(np.maximum(initial_dlvsq - _MINIMUM_DLVSQ, 1.0e-10))
    raw = jnp.asarray(initial, dtype=jnp.float32)
    first = jnp.zeros_like(raw)
    second = jnp.zeros_like(raw)
    step = jnp.asarray(0, dtype=jnp.int32)
    is_call = arrays["strikes"] >= 1.0

    def decode(current: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        dlvsq = _MINIMUM_DLVSQ + jnp.exp(jnp.minimum(current, maximum_raw))
        densities = decode_density_tower(
            arrays["supports"], arrays["transports"], dlvsq, intervals
        )
        calls = jnp.einsum("jkn,jn->jk", arrays["kernels"], densities)
        prices = jnp.where(is_call, calls, calls - 1.0 + arrays["strikes"])
        return dlvsq, densities, prices

    def loss(current: jax.Array) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        _, _, prices = decode(current)
        scaled_error = jnp.abs(prices - arrays["mids"]) / arrays["half_spreads"]
        quote_loss = jnp.sum(
            (jax.nn.relu(scaled_error - 1.0) + 0.05 * scaled_error) * valid
        ) / jnp.sum(valid)
        scaled_cap = (current - maximum_raw) / 0.10
        barrier = 0.10 * jax.nn.softplus(scaled_cap) / 0.35
        cap_penalty = 0.5 * jnp.sum(barrier**2)
        return quote_loss + cap_penalty, (quote_loss, cap_penalty)

    value_and_gradient = jax.jit(jax.value_and_grad(loss, has_aux=True))
    decode_only = jax.jit(decode)

    @jax.jit
    def update(
        current: jax.Array,
        first_moment: jax.Array,
        second_moment: jax.Array,
        current_step: jax.Array,
        gradient: jax.Array,
        learning_rate: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        next_step = current_step + 1
        next_first = 0.9 * first_moment + 0.1 * gradient
        next_second = 0.999 * second_moment + 0.001 * gradient**2
        corrected_first = next_first / (1.0 - 0.9**next_step)
        corrected_second = next_second / (1.0 - 0.999**next_step)
        next_value = current - learning_rate * corrected_first / (
            jnp.sqrt(corrected_second) + 1.0e-8
        )
        return next_value, next_first, next_second, next_step

    checkpoints: list[float] = []
    started = perf_counter()
    stop_reason = "maximum_epochs"
    final_objective = final_quote = final_cap = math.nan
    for epoch in range(1, maximum_epochs + 1):
        fraction = (epoch - 1) / max(maximum_epochs - 1, 1)
        learning_rate = 0.01 * (
            0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * fraction))
        )
        (objective, auxiliary), gradient = value_and_gradient(raw)
        quote_loss, cap_penalty = auxiliary
        raw, first, second, step = update(
            raw,
            first,
            second,
            step,
            gradient,
            jnp.asarray(learning_rate, dtype=jnp.float32),
        )
        if epoch == 1 or epoch % report_every == 0 or epoch == maximum_epochs:
            final_objective = float(objective)
            final_quote = float(quote_loss)
            final_cap = float(cap_penalty)
            checkpoints.append(final_objective)
            elapsed = perf_counter() - started
            rate = epoch / max(elapsed, 1.0e-9)
            eta = (maximum_epochs - epoch) / max(rate, 1.0e-9)
            print(
                f"dlv epoch={epoch}/{maximum_epochs} objective={final_objective:.7g} "
                f"quote={final_quote:.7g} cap={final_cap:.3g} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
            if epoch >= minimum_epochs and len(checkpoints) >= 6:
                old = checkpoints[-6]
                relative_improvement = (old - checkpoints[-1]) / max(abs(old), 1.0e-8)
                if 0.0 <= relative_improvement < 1.0e-5:
                    stop_reason = "five_checkpoint_relative_objective_improvement_below_1e-5"
                    break

    dlvsq, densities, prices = decode_only(raw)
    dlvsq_np = np.asarray(dlvsq, dtype=np.float64)
    densities_np = np.asarray(densities, dtype=np.float64)
    prices_np = np.asarray(prices, dtype=np.float64)
    scaled_error = np.abs(prices_np - mids) / half_spreads
    valid_error = scaled_error[mask]
    excess = np.maximum(valid_error - 1.0, 0.0)
    log_dlvsq = np.log(dlvsq_np)
    strike_roughness, expiry_roughness = _roughness(log_dlvsq)
    return OneDateFit(
        dlvsq=dlvsq_np,
        densities=densities_np,
        prices=prices_np,
        completed_epochs=epoch,
        stop_reason=stop_reason,
        objective=final_objective,
        quote_loss=final_quote,
        cap_penalty=final_cap,
        percent_inside_spread=float(100.0 * np.mean(excess == 0.0)),
        midpoint_mae_half_spreads=float(np.mean(valid_error)),
        mean_excess_half_spreads=float(np.mean(excess)),
        maximum_excess_half_spreads=float(np.max(excess)),
        strike_roughness=strike_roughness,
        expiry_roughness=expiry_roughness,
        floored_fraction=float(np.mean(dlvsq_np <= 1.01 * _MINIMUM_DLVSQ)),
        capped_fraction=float(np.mean(dlvsq_np >= 0.99 * _MAXIMUM_DLVSQ)),
    )


def _fit_report(fit: OneDateFit) -> dict[str, object]:
    result = asdict(fit)
    for key in ("dlvsq", "densities", "prices"):
        result.pop(key)
    return result


def _atomic_npz(path: Path, **payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_call_kernel_bundle(
    path: Path,
    *,
    kernels: FloatArray,
    supports: FloatArray,
    strikes: FloatArray,
    expiries_y: FloatArray,
    background: HestonBackground,
    reference_date: str,
) -> str:
    """Persist one geometry-specific Heston call-kernel bundle."""
    metadata = json.dumps(
        {
            "model": "independent-Heston",
            "reference_date": reference_date,
            "background": asdict(background),
            "fft_size": _FFT_SIZE,
            "fft_frequency_spacing": _FFT_FREQUENCY_SPACING,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(metadata.encode("utf-8"))
    for values in (supports, strikes, expiries_y):
        contiguous = np.ascontiguousarray(values, dtype=np.float64)
        digest.update(contiguous.shape.__repr__().encode("ascii"))
        digest.update(contiguous.tobytes())
    identity = f"independent-Heston:{digest.hexdigest()}"
    _atomic_npz(
        path,
        schema=np.asarray("gromm2-call-kernels-v1"),
        identity=np.asarray(identity),
        metadata=np.asarray(metadata),
        call_kernels=np.asarray(kernels, dtype=np.float32),
    )
    return identity


def write_call_kernel_grid_bundle(
    path: Path,
    *,
    kernel_grid: FloatArray,
    variance_grid: FloatArray,
    date_variances: FloatArray,
    dates: NDArray[np.str_],
    supports: FloatArray,
    strikes: FloatArray,
    expiries_y: FloatArray,
    background: HestonBackground,
    reference_date: str,
) -> str:
    """Persist date-varying Heston kernels on an initial-variance grid."""
    metadata = json.dumps(
        {
            "model": "independent-Heston-date-varying-initial-variance",
            "reference_date": reference_date,
            "background": asdict(background),
            "interpolation": "linear-in-initial-variance",
            "fft_size": _FFT_SIZE,
            "fft_frequency_spacing": _FFT_FREQUENCY_SPACING,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(metadata.encode("utf-8"))
    for values in (
        supports,
        strikes,
        expiries_y,
        variance_grid,
        date_variances,
        dates,
    ):
        contiguous = np.ascontiguousarray(values)
        digest.update(contiguous.shape.__repr__().encode("ascii"))
        digest.update(contiguous.tobytes())
    identity = f"independent-Heston-date-varying:{digest.hexdigest()}"
    _atomic_npz(
        path,
        schema=np.asarray("gromm2-call-kernels-v2"),
        identity=np.asarray(identity),
        metadata=np.asarray(metadata),
        call_kernels=np.asarray(kernel_grid, dtype=np.float32),
        variance_grid=np.asarray(variance_grid, dtype=np.float64),
        date_variances=np.asarray(date_variances, dtype=np.float64),
        dates=np.asarray(dates, dtype="S10"),
    )
    return identity


def _plot_comparison(
    path: Path,
    *,
    business_dtes: FloatArray,
    nms_nodes: FloatArray,
    black: OneDateFit,
    heston: OneDateFit,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), constrained_layout=True)
    for axis, fit, title in (
        (axes[0, 0], black, "5% Black background"),
        (axes[0, 1], heston, "Independent Heston background"),
    ):
        image = axis.pcolormesh(
            nms_nodes,
            business_dtes,
            100.0 * np.sqrt(fit.dlvsq),
            shading="auto",
            cmap="viridis",
        )
        axis.set_title(title)
        axis.set_xlabel("Normalized moneyness")
        axis.set_ylabel("Business days to expiry")
        figure.colorbar(image, ax=axis, label="DLV (%)")
    difference = np.log(heston.dlvsq) - np.log(black.dlvsq)
    image = axes[1, 0].pcolormesh(
        nms_nodes,
        business_dtes,
        difference,
        shading="auto",
        cmap="coolwarm",
    )
    axes[1, 0].set_title("Log-DLVsq difference: Heston minus Black")
    axes[1, 0].set_xlabel("Normalized moneyness")
    axes[1, 0].set_ylabel("Business days to expiry")
    figure.colorbar(image, ax=axes[1, 0])
    names = ["Inside spread (%)", "Midpoint MAE", "Mean excess"]
    black_values = [
        black.percent_inside_spread,
        black.midpoint_mae_half_spreads,
        black.mean_excess_half_spreads,
    ]
    heston_values = [
        heston.percent_inside_spread,
        heston.midpoint_mae_half_spreads,
        heston.mean_excess_half_spreads,
    ]
    positions = np.arange(len(names))
    axes[1, 1].bar(positions - 0.18, black_values, 0.36, label="Black")
    axes[1, 1].bar(positions + 0.18, heston_values, 0.36, label="Heston")
    axes[1, 1].set_xticks(positions, names, rotation=15)
    axes[1, 1].set_title("One-date fit diagnostics")
    axes[1, 1].legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def run_pilot(
    *,
    dataset: Path,
    prepared_books: Path,
    calibrated_history: Path,
    background_estimates: Path,
    output_directory: Path,
    trading_date: str,
    background_annual_volatility: float = 0.05,
    call_kernel_output: Path | None = None,
) -> dict[str, object]:
    """Run and persist the final-date independent-Heston pilot."""
    days = {day.trading_date.isoformat(): day for day in load_public_sanos_days(dataset)}
    if trading_date not in days:
        raise ValueError(f"public SANOS dataset does not contain {trading_date}")
    day = days[trading_date]
    with h5py.File(prepared_books, "r") as stream:
        dates = np.asarray(stream["dates"]).astype("U10")
        matches = np.flatnonzero(dates == trading_date)
        if len(matches) != 1:
            raise ValueError(f"prepared books do not contain one row for {trading_date}")
        date_index = int(matches[0])
        expiries_y = np.asarray(stream["model_expiries_y"], dtype=np.float64)
        business_dtes = np.asarray(stream["model_business_dtes"], dtype=np.float64)
        supports = np.asarray(stream["supports"], dtype=np.float64)
        transports = np.asarray(stream["transports"], dtype=np.float64)
        nms_nodes = np.asarray(stream["nms_nodes"], dtype=np.float64)
        strikes = np.asarray(stream["target_strikes"], dtype=np.float64)
        mids = np.asarray(stream["mids"][date_index], dtype=np.float64)
        half_spreads = np.asarray(
            stream["half_spreads"][date_index], dtype=np.float64
        )
        mask = np.asarray(stream["mask"][date_index], dtype=bool)
    with h5py.File(calibrated_history, "r") as stream:
        history_dates = np.asarray(stream["dates"]).astype("U10")
        matches = np.flatnonzero(history_dates == trading_date)
        if len(matches) != 1:
            raise ValueError(f"calibrated history does not contain one row for {trading_date}")
        initial_dlvsq = np.asarray(stream["dlvsq"][int(matches[0])], dtype=np.float64)
    raw_total_variances = public_sanos_atm_total_variances(day, expiries_y)
    projected_total_variances = project_total_variance(
        expiries_y,
        raw_total_variances,
        forward_volatility_floor=0.01,
    )
    market_background, market_fit = fit_market_heston(
        day,
        expiries_y=expiries_y,
        projected_total_variances=projected_total_variances,
    )
    with background_estimates.open() as stream:
        historical = json.load(stream)
    observed_correlation = float(
        historical["heston_proxy"]
        ["standardized_variance_innovation_product_spot_score_correlation"]
    )
    total_one_day_volatility = math.sqrt(projected_total_variances[0] / expiries_y[0])
    if background_annual_volatility <= 0.0:
        raise ValueError("background_annual_volatility must be positive")
    volatility_scale = (
        background_annual_volatility
        / math.sqrt(market_background.initial_variance)
    )
    scaled_background = scale_heston_background(
        market_background,
        volatility_scale=volatility_scale,
    )
    background_spot_volatility = math.sqrt(scaled_background.initial_variance)
    adjusted_correlation = float(
        np.clip(
            observed_correlation
            * total_one_day_volatility
            / background_spot_volatility,
            -0.99,
            0.99,
        )
    )
    background = HestonBackground(
        mean_reversion=scaled_background.mean_reversion,
        long_run_variance=scaled_background.long_run_variance,
        volatility_of_variance=scaled_background.volatility_of_variance,
        correlation=adjusted_correlation,
        initial_variance=scaled_background.initial_variance,
    )
    implied_reduced_form_correlation = (
        background_spot_volatility
        / total_one_day_volatility
        * background.correlation
    )
    print(
        f"heston fitted_rho={market_background.correlation:.6f} "
        f"volatility_scale={volatility_scale:.6f} "
        f"adjusted_rho={background.correlation:.6f} "
        f"implied_reduced_form={implied_reduced_form_correlation:.6f}",
        flush=True,
    )
    black_kernels = _black_call_kernels(
        supports,
        strikes,
        expiries_y,
        annual_volatility=background_annual_volatility,
    )
    heston_kernels = heston_call_kernels(
        supports,
        strikes,
        expiries_y,
        background=background,
    )
    call_kernel_identity = None
    if call_kernel_output is not None:
        call_kernel_identity = write_call_kernel_bundle(
            call_kernel_output,
            kernels=heston_kernels,
            supports=supports,
            strikes=strikes,
            expiries_y=expiries_y,
            background=background,
            reference_date=trading_date,
        )
    print("fitting Black-background control", flush=True)
    black_fit = fit_one_date_dlv(
        supports=supports,
        transports=transports,
        expiries_y=expiries_y,
        kernels=black_kernels,
        strikes=strikes,
        mids=mids,
        half_spreads=half_spreads,
        mask=mask,
        initial_dlvsq=initial_dlvsq,
    )
    print("fitting independent-Heston-background pilot", flush=True)
    heston_fit = fit_one_date_dlv(
        supports=supports,
        transports=transports,
        expiries_y=expiries_y,
        kernels=heston_kernels,
        strikes=strikes,
        mids=mids,
        half_spreads=half_spreads,
        mask=mask,
        initial_dlvsq=black_fit.dlvsq,
    )
    report: dict[str, object] = {
        "schema": "gromm2-independent-heston-one-date-pilot-v1",
        "trading_date": trading_date,
        "background_annual_volatility": background_annual_volatility,
        "call_kernel_identity": call_kernel_identity,
        "construction": "Heston fitted to the full date-t SANOS surface, its variance process scaled to a 5% instantaneous spot-volatility level, and rho subsequently set to match the historical product-spot/kernel-variance innovation correlation; the Heston subsystem remains independent of DLV factors",
        "market_fitted_heston": asdict(market_background),
        "market_surface_fit": market_fit,
        "scaled_surface_background_before_correlation_adjustment": asdict(
            scaled_background
        ),
        "dynamics_adjusted_background": asdict(background),
        "correlation_adjustment": {
            "observed_product_spot_kernel_variance_innovation_correlation": observed_correlation,
            "date_t_total_one_day_atm_volatility": total_one_day_volatility,
            "target_background_instantaneous_spot_volatility": (
                background_annual_volatility
            ),
            "variance_process_volatility_scale": volatility_scale,
            "background_instantaneous_spot_volatility": background_spot_volatility,
            "adjusted_heston_return_variance_correlation": adjusted_correlation,
            "implied_reduced_form_correlation": implied_reduced_form_correlation,
        },
        "black_background_fit": _fit_report(black_fit),
        "heston_background_fit": _fit_report(heston_fit),
        "caveat": "This pilot tests one date's static residual-DLV extraction. Historical DLV re-estimation and a one-day joint leverage simulation have not yet been run.",
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / "pilot.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _atomic_npz(
        output_directory / "pilot.npz",
        business_dtes=business_dtes,
        nms_nodes=nms_nodes,
        supports=supports,
        strikes=strikes,
        black_dlvsq=black_fit.dlvsq,
        heston_dlvsq=heston_fit.dlvsq,
        black_densities=black_fit.densities,
        heston_densities=heston_fit.densities,
        black_prices=black_fit.prices,
        heston_prices=heston_fit.prices,
    )
    _plot_comparison(
        output_directory / "pilot.pdf",
        business_dtes=business_dtes,
        nms_nodes=nms_nodes,
        black=black_fit,
        heston=heston_fit,
    )
    print(f"wrote {output_directory}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prepared-books", type=Path, required=True)
    parser.add_argument("--calibrated-history", type=Path, required=True)
    parser.add_argument("--background-estimates", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--trading-date", default="2025-08-29")
    parser.add_argument("--background-annual-volatility", type=float, default=0.05)
    parser.add_argument("--call-kernel-output", type=Path)
    arguments = parser.parse_args()
    run_pilot(
        dataset=arguments.dataset,
        prepared_books=arguments.prepared_books,
        calibrated_history=arguments.calibrated_history,
        background_estimates=arguments.background_estimates,
        output_directory=arguments.output_directory,
        trading_date=arguments.trading_date,
        background_annual_volatility=arguments.background_annual_volatility,
        call_kernel_output=arguments.call_kernel_output,
    )


if __name__ == "__main__":
    main()
